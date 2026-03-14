import os
import re
import warnings
from typing import Any, cast

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

from pathlib import Path
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
        configured_model_path = os.getenv("ASR_MODEL_PATH", "Qwen/Qwen3-ASR-0.6B")
        model_path = self._normalize_model_path(configured_model_path)
        print(f"[ASR] Loading {model_path} on {device_name}...")

        try:
            self.model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                device_map=device,
                max_new_tokens=256,
            )
        except OSError as exc:
            raise RuntimeError(
                "Failed to load the ASR model. "
                f"Resolved ASR model path: '{model_path}'. "
                "If you are downloading from Hugging Face, use a valid repo id such as "
                "'Qwen/Qwen3-ASR-0.6B' or set ASR_MODEL_PATH to a local model directory. "
                "If the repo is gated/private in your environment, authenticate with "
                "'hf auth login' first."
            ) from exc

        print(f"[ASR] Model loaded on {device_name}")

    @staticmethod
    def _normalize_model_path(model_path: str) -> str:
        if not model_path:
            return "Qwen/Qwen3-ASR-0.6B"
        if os.path.isdir(model_path):
            return model_path
        if "/" not in model_path and model_path.startswith("Qwen3-ASR-"):
            model_path = f"Qwen/{model_path}"

        cached_snapshot = ASRModel._resolve_cached_snapshot(model_path)
        if cached_snapshot:
            print(f"[ASR] Using cached local model at {cached_snapshot}")
            return cached_snapshot

        return model_path

    @staticmethod
    def _resolve_cached_snapshot(model_path: str) -> str | None:
        if "/" not in model_path:
            return None

        owner, repo = model_path.split("/", 1)
        cache_roots = []

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            cache_roots.append(Path(hf_home) / "hub")

        xdg_cache = os.getenv("XDG_CACHE_HOME")
        if xdg_cache:
            cache_roots.append(Path(xdg_cache) / "huggingface" / "hub")

        cache_roots.extend([
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / ".huggingface" / "hub",
        ])

        model_cache_dirname = f"models--{owner}--{repo}"
        for cache_root in cache_roots:
            snapshot_root = cache_root / model_cache_dirname / "snapshots"
            if not snapshot_root.is_dir():
                continue

            for snapshot_dir in sorted(snapshot_root.iterdir(), reverse=True):
                if not snapshot_dir.is_dir():
                    continue
                if (snapshot_dir / "config.json").is_file():
                    return str(snapshot_dir)

        return None


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

