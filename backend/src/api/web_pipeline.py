"""
WebSocket-adapted voice pipeline for browser-based audio streaming.
Receives audio chunks from WebSocket, processes through VAD → ASR → LLM → TTS,
and sends results back over WebSocket.
"""

import time
import asyncio
import numpy as np
import struct
from typing import Dict, Optional

from src.utils.chunker import SentenceChunker
from src.core.interfaces import IVAD, IASR, ITTS
from src.agents.session import VoiceSession


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
        self.session: Optional[VoiceSession] = None
        self._reset_state()

    def _reset_state(self):
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_chunks = 0

    def init_session(self):
        """Create a fresh LangGraph voice session."""
        self.session = VoiceSession()
        self._reset_state()
        return self.session.config["configurable"]["thread_id"]

    @property
    def active_agent_name(self) -> str:
        if self.session:
            return self.session.active_agent_name
        return "CustomerCare"

    def process_audio_chunk(self, raw_bytes: bytes) -> dict:
        """
        Process a single audio chunk (16-bit PCM, 16kHz mono).
        Returns a dict with status info:
          - {"status": "listening"} — waiting for speech
          - {"status": "speech_detected"} — speech started
          - {"status": "speech_end", "audio": np.ndarray} — speech segment complete
          - {"status": "accumulating"} — collecting speech
        """
        # Convert raw 16-bit PCM bytes to float32 numpy array
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        rms_volume = np.sqrt(np.mean(samples ** 2))
        speech_detected = self.vad.is_speech(samples) and rms_volume > 0.005

        if speech_detected:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_buffer = []
                self.silence_chunks = 0
                self.speech_buffer.append(samples)
                return {"status": "speech_detected"}
            self.speech_buffer.append(samples)
            self.silence_chunks = 0
            return {"status": "accumulating"}
        elif self.is_speaking:
            self.speech_buffer.append(samples)
            self.silence_chunks += 1
            # ~960ms of silence (32ms * 30 chunks)
            if self.silence_chunks > 30:
                audio_array = np.concatenate(self.speech_buffer)
                if len(audio_array.shape) > 1:
                    audio_array = audio_array[:, 0]
                self._reset_state()
                return {"status": "speech_end", "audio": audio_array}
            return {"status": "accumulating"}
        else:
            return {"status": "listening"}

    def transcribe(self, audio_array: np.ndarray) -> str:
        """Run ASR on a complete speech segment."""
        start = time.time()
        text = self.asr.transcribe(audio_array)
        latency = time.time() - start
        print(f"[WebPipeline ASR] '{text}' ({latency:.2f}s)")
        return text

    async def generate_response(self, text: str, send_callback):
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

        # Send user transcript
        await send_callback({"type": "transcript", "role": "user", "text": text})

        if len(text.strip()) < 2:
            return

        await send_callback({"type": "status", "value": "thinking"})

        chunker = SentenceChunker()
        ai_full_text = ""

        for token in self.session.stream_response(text):
            for sentence in chunker.process_token(token):
                ai_full_text += sentence + " "
                active_agent = self.session.active_agent_name

                await send_callback({"type": "agent", "name": active_agent})

                tts = self.tts_models.get(
                    active_agent, list(self.tts_models.values())[0]
                )
                tts_start = time.time()
                audio_data, fs = tts.synthesize(sentence)
                tts_latency = time.time() - tts_start
                print(f"  [TTS] '{sentence[:40]}...' ({tts_latency:.2f}s)")

                # Send partial AI transcript text
                await send_callback({
                    "type": "transcript",
                    "role": "ai",
                    "text": sentence,
                    "agent": active_agent,
                    "partial": True,
                })

                # Convert float32 audio to 16-bit PCM bytes for the browser
                pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                await send_callback({
                    "type": "audio",
                    "data": pcm_data,
                    "sample_rate": fs,
                })

        # Flush remaining text
        for sentence in chunker.flush():
            ai_full_text += sentence + " "
            active_agent = self.session.active_agent_name
            tts = self.tts_models.get(
                active_agent, list(self.tts_models.values())[0]
            )
            audio_data, fs = tts.synthesize(sentence)
            pcm_data = (audio_data * 32767).astype(np.int16).tobytes()

            await send_callback({
                "type": "transcript",
                "role": "ai",
                "text": sentence,
                "agent": active_agent,
                "partial": True,
            })
            await send_callback({
                "type": "audio",
                "data": pcm_data,
                "sample_rate": fs,
            })

        # Final complete transcript
        await send_callback({
            "type": "transcript",
            "role": "ai",
            "text": ai_full_text.strip(),
            "agent": self.session.active_agent_name,
            "partial": False,
        })

        await send_callback({"type": "status", "value": "idle"})
