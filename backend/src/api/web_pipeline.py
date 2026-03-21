"""
WebSocket-adapted voice pipeline for browser-based audio streaming.
Receives audio chunks from WebSocket, processes through VAD → ASR → LLM → TTS,
and sends results back over WebSocket.
"""

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.core.interfaces import ASRStreamHandle, IASR, ITTS, IVAD
from src.agents.session import VoiceSession
from src.utils.chunker import SentenceChunker
from src.utils.language_guard import EnglishLanguageGuard
from src.audio.denoiser import Denoiser

SendCallback = Callable[[dict[str, Any]], Awaitable[None]]
ENGLISH_RETRY_MESSAGE = "I can help in English. Please say that again in English."


@dataclass
class TurnTrace:
    turn_id: int
    speech_started_at: float
    transcript_preview: str = ""
    first_partial_at: float | None = None
    final_transcript_at: float | None = None
    response_started_at: float | None = None
    llm_first_token_at: float | None = None
    first_tts_chunk_at: float | None = None
    first_audio_sent_at: float | None = None
    user_speech_finished_at: float | None = None
    response_finished_at: float | None = None
    cancelled_at: float | None = None


class WebVoicePipeline:
    """Pipeline that processes voice from a WebSocket instead of local mic/speakers."""

    def __init__(
        self,
        vad: IVAD,
        asr: IASR,
        tts_models: Dict[str, ITTS],
        sample_rate: int = 16000,
    ):
        self.vad = vad
        self.asr = asr
        self.tts_models = tts_models
        self.sample_rate = sample_rate
        self.min_speech_samples = int(sample_rate * 0.6)
        self.manual_stop_min_samples = int(sample_rate * 0.2)
        self.language_guard = EnglishLanguageGuard()
        self.session: Optional[VoiceSession] = None
        self.use_denoising = False
        try:
            self.denoiser = Denoiser()
        except Exception as e:
            print(f"[WebPipeline] Failed to load denoiser: {e}")
            self.denoiser = None
        self._response_lock = asyncio.Lock()
        self._cancel_response = False
        self._turn_sequence = 0
        self.current_turn_trace: TurnTrace | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.speech_buffer: list[np.ndarray] = []
        self.is_speaking = False
        self.silence_chunks = 0
        self.chunks_since_last_partial = 0
        self.active_speech_chunks = 0
        self.asr_stream: Optional[ASRStreamHandle] = None
        self.last_partial_text = ""
        if not hasattr(self, "short_query_history"):
            self.short_query_history: list[float] = []
        # Reset denoiser buffer between speech turns to prevent audio bleed
        if hasattr(self, "denoiser") and self.denoiser is not None:
            self.denoiser.buffer = np.array([], dtype=np.float32)

    def _start_turn_trace(self) -> TurnTrace:
        self._turn_sequence += 1
        trace = TurnTrace(
            turn_id=self._turn_sequence,
            speech_started_at=time.perf_counter(),
        )
        self.current_turn_trace = trace
        return trace

    def _ensure_turn_trace(self, transcript_preview: str = "") -> TurnTrace:
        trace = self.current_turn_trace or self._start_turn_trace()
        if transcript_preview and not trace.transcript_preview:
            trace.transcript_preview = transcript_preview[:120]
        return trace

    def _mark_first_partial(self, partial_text: str) -> None:
        trace = self.current_turn_trace
        if trace is None:
            trace = self._start_turn_trace()
        if trace.first_partial_at is None:
            trace.first_partial_at = time.perf_counter()
        if partial_text and not trace.transcript_preview:
            trace.transcript_preview = partial_text[:120]

    def _mark_final_transcript(self, transcript_text: str) -> TurnTrace:
        trace = self._ensure_turn_trace(transcript_text)
        if trace.final_transcript_at is None:
            trace.final_transcript_at = time.perf_counter()
        trace.transcript_preview = transcript_text[:120]
        return trace

    def _discard_turn_trace(self, reason: str) -> None:
        trace = self.current_turn_trace
        if trace is not None:
            print(
                f"[Latency][turn={trace.turn_id}] dropped reason={reason} "
                f"preview='{trace.transcript_preview}'"
            )
        self.current_turn_trace = None

    def _log_turn_trace(self, trace: TurnTrace, outcome: str) -> None:
        def ms(start: float | None, end: float | None) -> str:
            if start is None or end is None:
                return "n/a"
            return f"{(end - start) * 1000:.0f}ms"

        def raw_ms(start: float | None, end: float | None) -> float:
            if start is None or end is None:
                return 0.0
            return (end - start) * 1000.0

        thread_id = "unknown"
        if self.session is not None:
            thread_id = str(self.session.config["configurable"]["thread_id"])

        ttft = raw_ms(trace.response_started_at, trace.llm_first_token_at)
        ttfa = raw_ms(trace.llm_first_token_at, trace.first_audio_sent_at)
        combined = ttft + ttfa

        print(
            f"[Latency][turn={trace.turn_id}] "
            f"TTFT={ttft:.0f}ms | TTFA={ttfa:.0f}ms | Combined={combined:.0f}ms | "
            f"preview='{trace.transcript_preview}'"
        )

        if self.current_turn_trace is trace:
            self.current_turn_trace = None

    @staticmethod
    async def _drain_pending_tts_items(
        tts_queue: asyncio.Queue[tuple[str, str] | None],
    ) -> None:
        while True:
            try:
                pending = tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            tts_queue.task_done()
            if pending is None:
                break

    def init_session(self):
        """Create a fresh LangGraph voice session."""
        self.session = VoiceSession()
        self._reset_state()
        self.short_query_history = []
        # Full denoiser reset (including neural network state) for new sessions
        if self.denoiser is not None:
            self.denoiser.reset()
        return self.session.config["configurable"]["thread_id"]

    @property
    def active_agent_name(self) -> str:
        if self.session:
            return self.session.active_agent_name
        return "CustomerCare"

    def process_audio_chunk(self, raw_bytes: bytes) -> dict[str, Any]:
        """
        Process a single audio chunk (16-bit PCM, 16kHz mono).
        Returns a dict with status info:
          - {"status": "listening"} — waiting for speech
          - {"status": "speech_detected"} — speech started
          - {"status": "speech_partial", "audio": np.ndarray} — ongoing speech buffer (sent every ~480ms)
          - {"status": "speech_end", "audio": np.ndarray} — speech segment complete
          - {"status": "accumulating_pre_speech"} — buffering initial clicks to see if they are real speech
          - {"status": "accumulating"} — collecting speech
        """
        # Convert raw 16-bit PCM bytes to float32 numpy array
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if self.use_denoising and self.denoiser:
            try:
                samples = self.denoiser.enhance(samples)
            except Exception as e:
                print(f"[WebPipeline] Denoising error: {e}")

        rms_volume = np.sqrt(np.mean(samples**2)) if len(samples) > 0 else 0.0
        is_vad_speech = self.vad.is_speech(samples) if len(samples) >= 160 else False
        speech_detected = is_vad_speech and rms_volume > 0.02

        if speech_detected:
            if not self.is_speaking:
                # We haven't officially started a speech block yet.
                self.speech_buffer.append(samples)
                self.active_speech_chunks += 1
                self._stream_audio_chunk(samples)

                # Require 6 consecutive chunks (192ms) of constant "speech" to barge-in
                if self.active_speech_chunks >= 6:
                    self.is_speaking = True
                    self.silence_chunks = 0
                    self.chunks_since_last_partial = 0
                    self._start_turn_trace()
                    return {"status": "speech_detected"}

                return {"status": "accumulating_pre_speech"}
            else:
                # We are officially speaking. Keep buffering.
                self.speech_buffer.append(samples)
                self.silence_chunks = 0
                self.chunks_since_last_partial += 1

                partial_text = self._stream_audio_chunk(samples)
                if partial_text:
                    return {"status": "speech_partial", "text": partial_text}

                # Emit a partial transcript every 15 chunks (~480ms)
                if (
                    not self.asr.supports_streaming
                    and self.chunks_since_last_partial >= 15
                ):
                    self.chunks_since_last_partial = 0
                    audio_array = np.concatenate(self.speech_buffer)
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array[:, 0]
                    return {"status": "speech_partial", "audio": audio_array}

                return {"status": "accumulating"}
        elif self.is_speaking:
            self.speech_buffer.append(samples)
            self.silence_chunks += 1
            # ~800ms of silence (32ms * 25 chunks)
            if self.silence_chunks > 25:
                if self.asr.supports_streaming:
                    final_text = self._finish_streaming_text(
                        min_samples=self.min_speech_samples
                    )
                    if final_text is None:
                        return {"status": "listening"}
                    return {"status": "speech_end", "text": final_text}

                audio_array = self._consume_speech_buffer(
                    min_samples=self.min_speech_samples
                )
                if audio_array is None:
                    return {"status": "listening"}

                trace = self._ensure_turn_trace()
                if trace.user_speech_finished_at is None:
                    trace.user_speech_finished_at = time.perf_counter()

                return {"status": "speech_end", "audio": audio_array}
            return {"status": "accumulating"}
        else:
            # We are not speaking, and it's silence.
            # If we had a few active chunks but they didn't reach the threshold, discard them.
            if len(self.speech_buffer) > 0 and not self.is_speaking:
                self._reset_state()

            return {"status": "listening"}

    def _build_audio_array(self) -> Optional[np.ndarray]:
        """Build the current audio buffer into a mono waveform without resetting state."""
        if not self.speech_buffer:
            return None

        audio_array = np.concatenate(self.speech_buffer)
        if len(audio_array.shape) > 1:
            audio_array = audio_array[:, 0]
        return audio_array

    def _consume_speech_buffer(self, min_samples: int) -> Optional[np.ndarray]:
        """Extract and clear the current speech buffer if it is large enough."""
        audio_array = self._build_audio_array()
        if audio_array is None:
            self._reset_state()
            self._discard_turn_trace("empty_buffer")
            return None

        if len(audio_array) < min_samples:
            self._reset_state()
            self._discard_turn_trace("too_short")
            return None

        self._reset_state()
        return audio_array

    def _stream_audio_chunk(self, samples: np.ndarray) -> Optional[str]:
        """
        Feed the next PCM chunk into the streaming ASR backend and return a changed partial
        transcript when one becomes available.
        """
        if not self.asr.supports_streaming:
            return None

        if self.asr_stream is None:
            self.asr_stream = self.asr.create_stream()

        partial_text = self.asr.stream_transcribe(samples, self.asr_stream).strip()
        if partial_text and partial_text != self.last_partial_text:
            self.last_partial_text = partial_text
            self._mark_first_partial(partial_text)
            return partial_text
        return None

    def _finish_streaming_text(self, min_samples: int) -> Optional[str]:
        """
        Finalize the streaming ASR session if enough speech has been collected and return the
        committed transcript text.
        """
        audio_array = self._build_audio_array()
        stream = self.asr_stream
        cached_partial = self.last_partial_text

        if audio_array is None or len(audio_array) < min_samples:
            self._reset_state()
            self._discard_turn_trace("too_short")
            return None

        final_text = cached_partial
        if stream is not None:
            final_text = self.asr.finish_stream(stream).strip() or cached_partial

        self._reset_state()
        if final_text:
            trace = self._mark_final_transcript(final_text)
            if trace.user_speech_finished_at is None:
                trace.user_speech_finished_at = time.perf_counter()
        return final_text or None

    def finalize_current_utterance(self) -> dict[str, Any]:
        """
        Force-complete the current utterance, typically when the user taps stop.
        Uses a shorter minimum duration than silence-based finalization because the
        explicit user action indicates they are done speaking.
        """
        if self.asr.supports_streaming:
            final_text = self._finish_streaming_text(
                min_samples=self.manual_stop_min_samples
            )
            if final_text is None:
                return {"status": "listening"}
            return {"status": "speech_end", "text": final_text}

        audio_array = self._consume_speech_buffer(
            min_samples=self.manual_stop_min_samples
        )
        if audio_array is None:
            return {"status": "listening"}

        trace = self._ensure_turn_trace()
        if trace.user_speech_finished_at is None:
            trace.user_speech_finished_at = time.perf_counter()

        return {"status": "speech_end", "audio": audio_array}

    def transcribe(self, audio_array: np.ndarray) -> str:
        """Run ASR on a complete speech segment."""
        start = time.time()
        text = self.asr.transcribe(audio_array)
        latency = time.time() - start
        print(f"[WebPipeline ASR] '{text}'")
        return text

    def cancel_active_response(self) -> None:
        """Signal any in-flight response generation to stop as soon as possible."""
        self._cancel_response = True
        trace = self.current_turn_trace
        if trace is not None and trace.cancelled_at is None:
            trace.cancelled_at = time.perf_counter()

    async def _send_english_retry(self, send_callback: SendCallback) -> None:
        active_agent = self.active_agent_name
        await send_callback(
            {
                "type": "transcript",
                "role": "ai",
                "text": ENGLISH_RETRY_MESSAGE,
                "agent": active_agent,
                "partial": False,
            }
        )

        tts = self.tts_models.get(active_agent, list(self.tts_models.values())[0])
        audio_data, fs = await asyncio.get_event_loop().run_in_executor(
            None, tts.synthesize, ENGLISH_RETRY_MESSAGE
        )
        if len(audio_data) > 0:
            pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
            await send_callback(
                {
                    "type": "audio",
                    "data": pcm_data,
                    "sample_rate": fs,
                }
            )

        await send_callback({"type": "status", "value": "idle"})

    async def generate_response(self, text: str, send_callback: SendCallback) -> None:
        """
        Run LLM → TTS pipeline and call send_callback for each result.
        send_callback receives dicts like:
          {"type": "transcript", "role": "user", "text": ...}
          {"type": "transcript", "role": "ai", "text": ..., "agent": ...}
          {"type": "audio", "data": bytes, "sample_rate": int}
          {"type": "agent", "name": str}
          {"type": "status", "value": str}
        """
        if not self.session:
            self.init_session()

        stripped_text = text.strip()
        if len(stripped_text) < 2:
            return

        trace = self._mark_final_transcript(stripped_text)
        if trace.user_speech_finished_at is None:
            trace.user_speech_finished_at = time.perf_counter()

        language_decision = self.language_guard.evaluate(stripped_text)
        if not language_decision.allow:
            print(
                "[WebPipeline] Rejected non-English transcript "
                f"(language={language_decision.language}, "
                f"confidence={language_decision.confidence:.3f}): '{stripped_text}'"
            )
            await self._send_english_retry(send_callback)
            trace.response_finished_at = time.perf_counter()
            self._log_turn_trace(trace, "rejected")
            return

        # LAYER 2 FILTER: Short Text Rolling Window
        words = stripped_text.split()
        if len(words) < 4:
            current_time = time.time()
            self.short_query_history.append(current_time)

            # Remove timestamps older than 5 seconds
            self.short_query_history = [
                t for t in self.short_query_history if current_time - t <= 5.0
            ]

            # If we've had 3 or more short queries in the last 5 seconds, it's chatter. Block it.
            if len(self.short_query_history) >= 3:
                print(f"[WebPipeline] Noise filter blocked text: '{text}'")
                await send_callback({"type": "status", "value": "idle"})
                trace.response_finished_at = time.perf_counter()
                self._log_turn_trace(trace, "blocked_noise")
                return

        async with self._response_lock:
            self._cancel_response = False
            await send_callback({"type": "status", "value": "thinking"})
            await self._do_generate(stripped_text, send_callback, trace)

    async def _do_generate(
        self,
        text: str,
        send_callback: SendCallback,
        trace: TurnTrace,
    ) -> None:
        """Internal: actual LLM→TTS pipeline, runs under _response_lock."""
        assert self.session is not None
        session = self.session
        chunker = SentenceChunker()
        ai_full_text = ""
        trace.response_started_at = time.perf_counter()

        # Queue to decouple LLM generation from TTS generation
        tts_queue: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue(maxsize=4)

        async def tts_worker() -> None:
            """Background worker that streams TTS sub-chunks to the browser as they're generated."""
            while True:
                item = await tts_queue.get()
                if item is None:  # Sentinel value to exit worker
                    tts_queue.task_done()
                    break

                sentence, active_agent = item

                if self._cancel_response:
                    tts_queue.task_done()
                    continue

                tts = self.tts_models.get(
                    active_agent, list(self.tts_models.values())[0]
                )

                tts_start = time.time()

                # Use streaming TTS when available for lower time-to-first-audio
                if hasattr(tts, "synthesize_streaming"):
                    loop = asyncio.get_event_loop()
                    audio_queue = asyncio.Queue()

                    def run_synthesis():
                        try:
                            for chunk in tts.synthesize_streaming(sentence):
                                if self._cancel_response:
                                    break
                                loop.call_soon_threadsafe(audio_queue.put_nowait, chunk)
                        except Exception as e:
                            print(f"[WebPipeline] TTS Synthesis error: {e}")
                        finally:
                            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

                    loop.run_in_executor(None, run_synthesis)

                    while True:
                        chunk = await audio_queue.get()
                        if chunk is None:
                            break

                        audio_data, fs = chunk
                    stream = tts.synthesize_streaming(sentence)
                    print(f"  [TTS] '{sentence[:40]}...'")

                    while True:
                        chunk = await loop.run_in_executor(
                            None, lambda: next(stream, None)
                        )
                        if chunk is None:
                            break

                        audio_data, fs = chunk
                        if self._cancel_response or len(audio_data) == 0:
                            continue

                        if trace.first_audio_sent_at is None:
                            trace.first_audio_sent_at = time.perf_counter()
                            # Log TTFA for the first chunk of the first sentence
                            print(f"  [TTS] First audio chunk sent after {time.time() - tts_start:.3f}s")

                        pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                        await send_callback(
                            {
                                "type": "audio",
                                "data": pcm_data,
                                "sample_rate": fs,
                            }
                        )
                else:
                    # Fallback for TTS models without streaming
                    audio_data, fs = await asyncio.get_event_loop().run_in_executor(
                        None, tts.synthesize, sentence
                    )
                    tts_latency = time.time() - tts_start
                    print(f"  [TTS] '{sentence[:40]}...'")

                    if not self._cancel_response and len(audio_data) > 0:
                        if trace.first_audio_sent_at is None:
                            trace.first_audio_sent_at = time.perf_counter()
                        pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                        await send_callback(
                            {
                                "type": "audio",
                                "data": pcm_data,
                                "sample_rate": fs,
                            }
                        )

                tts_queue.task_done()

        # Start the background TTS worker
        worker_task = asyncio.create_task(tts_worker())

        try:
            for token in session.stream_response(text):
                # Check if this response was cancelled (new speech started)
                if self._cancel_response:
                    print("[WebPipeline] Response cancelled by new input.")
                    break

                if trace.llm_first_token_at is None:
                    trace.llm_first_token_at = time.perf_counter()

                # Instantly send the partial text token to the frontend
                ai_full_text += token
                active_agent = session.active_agent_name

                await send_callback({"type": "agent", "name": active_agent})

                await send_callback(
                    {
                        "type": "transcript",
                        "role": "ai",
                        "text": token,  # Send raw token immediately
                        "agent": active_agent,
                        "partial": True,
                    }
                )

                # Process the token through the chunker to get words/sentences for TTS
                for sentence in chunker.process_token(token):
                    if trace.first_tts_chunk_at is None:
                        trace.first_tts_chunk_at = time.perf_counter()
                    # Put it in the queue for the worker to synthesize
                    await tts_queue.put((sentence, active_agent))

            if not self._cancel_response:
                # Flush remaining text from chunker into the TTS queue
                for sentence in chunker.flush():
                    active_agent = session.active_agent_name
                    if trace.first_tts_chunk_at is None:
                        trace.first_tts_chunk_at = time.perf_counter()
                    await tts_queue.put((sentence, active_agent))
            else:
                await self._drain_pending_tts_items(tts_queue)

            # Send sentinel to stop the TTS worker.
            await tts_queue.put(None)
            await tts_queue.join()
            await worker_task

            if self._cancel_response:
                if trace.cancelled_at is None:
                    trace.cancelled_at = time.perf_counter()
                self._log_turn_trace(trace, "cancelled")
                return

            # Final complete transcript
            await send_callback(
                {
                    "type": "transcript",
                    "role": "ai",
                    "text": ai_full_text.strip(),
                    "agent": session.active_agent_name,
                    "partial": False,
                }
            )
            await send_callback({"type": "status", "value": "idle"})
            trace.response_finished_at = time.perf_counter()
            self._log_turn_trace(trace, "completed")
        except asyncio.CancelledError:
            self._cancel_response = True
            if trace.cancelled_at is None:
                trace.cancelled_at = time.perf_counter()
            raise
        except Exception as e:
            print(f"[WebPipeline] Error during response generation: {e}")
            await send_callback({"type": "status", "value": "idle"})
            trace.cancelled_at = time.perf_counter()
            self._log_turn_trace(trace, "failed")
        finally:
            self._cancel_response = True
            if not worker_task.done():
                worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_task
