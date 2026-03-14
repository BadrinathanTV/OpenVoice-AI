import os
import re
import warnings
from typing import Any, cast

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

from src.core.interfaces import ASRStreamHandle, IASR

# Suppress verbose generation warnings from Qwen
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

DEFAULT_VLLM_MAX_MODEL_LEN = 8192


def _is_truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_estimated_max_model_len(message: str) -> int | None:
    match = re.search(r"estimated maximum model length is (\d+)", message)
    if match is None:
        return None
    return int(match.group(1))


def _is_kv_cache_capacity_error(message: str) -> bool:
    return "KV cache" in message and "max seq len" in message


def _build_vllm_memory_error_message(
    requested_max_model_len: int,
    estimated_max_model_len: int | None,
) -> str:
    suggestion = (
        f"Set ASR_VLLM_MAX_MODEL_LEN to {estimated_max_model_len} or lower"
        if estimated_max_model_len is not None
        else "Set ASR_VLLM_MAX_MODEL_LEN to a smaller value such as 8192"
    )
    return (
        "vLLM could not reserve enough GPU KV-cache memory for the configured "
        f"ASR context window (ASR_VLLM_MAX_MODEL_LEN={requested_max_model_len}). "
        f"{suggestion}, or enable ASR_ALLOW_BACKEND_FALLBACK=true to use the "
        "transformers backend instead."
    )


class ASRModel(IASR):
    def __init__(self) -> None:
        self.model_path = os.getenv("ASR_MODEL_PATH", "Qwen3-ASR-0.6B")
        self.backend = os.getenv("ASR_BACKEND", "transformers").strip().lower()
        self.language = os.getenv("ASR_LANGUAGE", "English").strip() or "English"
        self.streaming_chunk_size_sec = float(
            os.getenv("ASR_STREAMING_CHUNK_SIZE_SEC", "0.64")
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_name = "GPU" if device.startswith("cuda") else "CPU"
        print(
            f"[ASR] Loading Qwen3-ASR-0.6B on {device_name} "
            f"(backend={self.backend})..."
        )

        if self.backend == "vllm":
            self.model = self._load_vllm_backend()
        else:
            self.model = self._load_transformers_backend(device)

        print(
            f"[ASR] Model loaded on {device_name} "
            f"(backend={self.backend}, streaming={self.supports_streaming})"
        )

    def _load_transformers_backend(self, device: str) -> Qwen3ASRModel:
        return Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map=device,
            max_new_tokens=256,
        )

    def _load_vllm_backend(self) -> Qwen3ASRModel:
        llm_kwargs: dict[str, Any] = {}
        tensor_parallel_size = os.getenv("ASR_TENSOR_PARALLEL_SIZE")
        gpu_memory_utilization = os.getenv("ASR_GPU_MEMORY_UTILIZATION")
        max_model_len = os.getenv("ASR_VLLM_MAX_MODEL_LEN")

        if tensor_parallel_size:
            llm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
        if gpu_memory_utilization:
            llm_kwargs["gpu_memory_utilization"] = float(gpu_memory_utilization)

        requested_max_model_len = (
            int(max_model_len) if max_model_len else DEFAULT_VLLM_MAX_MODEL_LEN
        )
        llm_kwargs["max_model_len"] = requested_max_model_len

        try:
            return Qwen3ASRModel.LLM(
                model=self.model_path,
                max_new_tokens=256,
                **llm_kwargs,
            )
        except Exception as exc:
            message = str(exc)
            estimated_max_model_len = _parse_estimated_max_model_len(message)
            retry_max_model_len = (
                estimated_max_model_len
                if estimated_max_model_len is not None
                and estimated_max_model_len < requested_max_model_len
                else None
            )

            if retry_max_model_len is not None and _is_kv_cache_capacity_error(message):
                print(
                    "[ASR] vLLM max_model_len="
                    f"{requested_max_model_len} exceeds this GPU's KV-cache budget. "
                    f"Retrying with max_model_len={retry_max_model_len}."
                )
                llm_kwargs["max_model_len"] = retry_max_model_len
                requested_max_model_len = retry_max_model_len
                try:
                    return Qwen3ASRModel.LLM(
                        model=self.model_path,
                        max_new_tokens=256,
                        **llm_kwargs,
                    )
                except Exception as retry_exc:
                    exc = retry_exc
                    message = str(retry_exc)
                    estimated_max_model_len = _parse_estimated_max_model_len(message)

            fallback_allowed = _is_truthy(
                os.getenv("ASR_ALLOW_BACKEND_FALLBACK", "true")
            )
            if not fallback_allowed:
                if _is_kv_cache_capacity_error(message):
                    raise RuntimeError(
                        _build_vllm_memory_error_message(
                            requested_max_model_len,
                            estimated_max_model_len,
                        )
                    ) from exc
                raise

            print(
                f"[ASR] Failed to initialize vLLM backend ({exc}). "
                "Falling back to transformers backend."
            )
            self.backend = "transformers"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            return self._load_transformers_backend(device)

    @property
    def supports_streaming(self) -> bool:
        return self.backend == "vllm"


    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        audio_data : numpy float32 waveform
        sample_rate : expected 16000
        """

        results = cast(
            list[Any],
            self.model.transcribe(
            audio=(audio_data, sample_rate),
            language=self.language
            ),
        )
        if not results:
            return ""

        return str(results[0].text).strip()

    def create_stream(
        self,
        context: str = "",
        language: str | None = None,
        chunk_size_sec: float | None = None,
    ) -> ASRStreamHandle:
        if not self.supports_streaming:
            raise NotImplementedError("Streaming ASR requires the vLLM backend.")

        stream_state = self.model.init_streaming_state(
            context=context,
            language=language or self.language,
            chunk_size_sec=chunk_size_sec or self.streaming_chunk_size_sec,
        )
        return ASRStreamHandle(backend_state=stream_state)

    def stream_transcribe(
        self,
        audio_chunk: np.ndarray,
        stream: ASRStreamHandle,
    ) -> str:
        if not self.supports_streaming:
            raise NotImplementedError("Streaming ASR requires the vLLM backend.")

        backend_state = self.model.streaming_transcribe(audio_chunk, stream.backend_state)
        stream.backend_state = backend_state
        stream.language = str(getattr(backend_state, "language", "") or "")
        stream.text = str(getattr(backend_state, "text", "") or "").strip()
        return stream.text

    def finish_stream(self, stream: ASRStreamHandle) -> str:
        if not self.supports_streaming:
            raise NotImplementedError("Streaming ASR requires the vLLM backend.")

        backend_state = self.model.finish_streaming_transcribe(stream.backend_state)
        stream.backend_state = backend_state
        stream.language = str(getattr(backend_state, "language", "") or "")
        stream.text = str(getattr(backend_state, "text", "") or "").strip()
        return stream.text
