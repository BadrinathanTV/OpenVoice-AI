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
        self._response_lock = asyncio.Lock()
        self._cancel_response = False
        self._reset_state()

    def _reset_state(self):
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_chunks = 0
        self.chunks_since_last_partial = 0
        self.active_speech_chunks = 0
        if not hasattr(self, 'short_query_history'):
            self.short_query_history = []

    def init_session(self):
        """Create a fresh LangGraph voice session."""
        self.session = VoiceSession()
        self._reset_state()
        self.short_query_history = []
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
          - {"status": "speech_partial", "audio": np.ndarray} — ongoing speech buffer (sent every ~480ms)
          - {"status": "speech_end", "audio": np.ndarray} — speech segment complete
          - {"status": "accumulating_pre_speech"} — buffering initial clicks to see if they are real speech
          - {"status": "accumulating"} — collecting speech
        """
        # Convert raw 16-bit PCM bytes to float32 numpy array
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        rms_volume = np.sqrt(np.mean(samples ** 2))
        speech_detected = self.vad.is_speech(samples) and rms_volume > 0.04

        if speech_detected:
            if not self.is_speaking:
                # We haven't officially started a speech block yet.
                self.speech_buffer.append(samples)
                self.active_speech_chunks += 1
                
                # Require 6 consecutive chunks (192ms) of constant "speech" to barge-in
                if self.active_speech_chunks >= 6:
                    self.is_speaking = True
                    self.silence_chunks = 0
                    self.chunks_since_last_partial = 0
                    return {"status": "speech_detected"}
                
                return {"status": "accumulating_pre_speech"}
            else:
                # We are officially speaking. Keep buffering.
                self.speech_buffer.append(samples)
                self.silence_chunks = 0
                self.chunks_since_last_partial += 1
                
                # Emit a partial transcript every 15 chunks (~480ms)
                if self.chunks_since_last_partial >= 15:
                    self.chunks_since_last_partial = 0
                    audio_array = np.concatenate(self.speech_buffer)
                    if len(audio_array.shape) > 1:
                        audio_array = audio_array[:, 0]
                    return {"status": "speech_partial", "audio": audio_array}
                    
                return {"status": "accumulating"}
        elif self.is_speaking:
            self.speech_buffer.append(samples)
            self.silence_chunks += 1
            # ~1600ms of silence (32ms * 50 chunks)
            if self.silence_chunks > 50:
                audio_array = np.concatenate(self.speech_buffer)
                if len(audio_array.shape) > 1:
                    audio_array = audio_array[:, 0]
                
                # LAYER 1 FILTER: Audio Duration Check
                # If total audio is less than ~0.6s (16000 * 0.6 = 9600 samples)
                if len(audio_array) < 9600:
                    self._reset_state()
                    return {"status": "listening"}

                self._reset_state()
                return {"status": "speech_end", "audio": audio_array}
            return {"status": "accumulating"}
        else:
            # We are not speaking, and it's silence.
            # If we had a few active chunks but they didn't reach the threshold, discard them.
            if len(self.speech_buffer) > 0 and not self.is_speaking:
                self._reset_state()
                
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

        # LAYER 2 FILTER: Short Text Rolling Window
        words = text.strip().split()
        if len(words) < 4:
            current_time = time.time()
            self.short_query_history.append(current_time)
            
            # Remove timestamps older than 5 seconds
            self.short_query_history = [t for t in self.short_query_history if current_time - t <= 5.0]
            
            # If we've had 3 or more short queries in the last 5 seconds, it's chatter. Block it.
            if len(self.short_query_history) >= 3:
                print(f"[WebPipeline] Noise filter blocked text: '{text}'")
                await send_callback({"type": "status", "value": "idle"})
                return

        await send_callback({"type": "status", "value": "thinking"})

        # Cancel any in-flight response before starting a new one
        self._cancel_response = True
        async with self._response_lock:
            self._cancel_response = False
            await self._do_generate(text, send_callback)

    async def _do_generate(self, text: str, send_callback):
        """Internal: actual LLM→TTS pipeline, runs under _response_lock."""
        chunker = SentenceChunker()
        ai_full_text = ""
        
        # Queue to decouple LLM generation from TTS generation
        tts_queue = asyncio.Queue()
        
        async def tts_worker():
            """Background worker to consume chunks from the queue and send TTS audio sequentially."""
            while True:
                item = await tts_queue.get()
                if item is None:  # Sentinel value to exit worker
                    tts_queue.task_done()
                    break
                    
                sentence, active_agent = item
                
                tts = self.tts_models.get(
                    active_agent, list(self.tts_models.values())[0]
                )
                
                tts_start = time.time()
                # Run synchronous TTS via executor to avoid blocking the event loop
                audio_data, fs = await asyncio.get_event_loop().run_in_executor(
                    None, tts.synthesize, sentence
                )
                tts_latency = time.time() - tts_start
                print(f"  [TTS] '{sentence[:40]}...' ({tts_latency:.2f}s)")

                # Convert float32 audio to 16-bit PCM bytes for the browser
                if len(audio_data) > 0:
                    pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                    await send_callback({
                        "type": "audio",
                        "data": pcm_data,
                        "sample_rate": fs,
                    })
                
                tts_queue.task_done()

        # Start the background TTS worker
        worker_task = asyncio.create_task(tts_worker())

        try:
            for token in self.session.stream_response(text):
                # Check if this response was cancelled (new speech started)
                if self._cancel_response:
                    print("[WebPipeline] Response cancelled by new input.")
                    break

                # Instantly send the partial text token to the frontend
                ai_full_text += token
                active_agent = self.session.active_agent_name
                
                await send_callback({"type": "agent", "name": active_agent})
                
                await send_callback({
                    "type": "transcript",
                    "role": "ai",
                    "text": token, # Send raw token immediately
                    "agent": active_agent,
                    "partial": True,
                })
                
                # Process the token through the chunker to get words/sentences for TTS
                for sentence in chunker.process_token(token):
                    # Put it in the queue for the worker to synthesize
                    await tts_queue.put((sentence, active_agent))

            # Flush remaining text from chunker into the TTS queue
            for sentence in chunker.flush():
                active_agent = self.session.active_agent_name
                await tts_queue.put((sentence, active_agent))

            # Send Sentinel to stop the TTS worker
            await tts_queue.put(None)
            
            # Wait for all TTS generation and callbacks to complete
            await tts_queue.join()
            await worker_task
            
            # Final complete transcript
            await send_callback({
                "type": "transcript",
                "role": "ai",
                "text": ai_full_text.strip(),
                "agent": self.session.active_agent_name,
                "partial": False,
            })

            await send_callback({"type": "status", "value": "idle"})
            
        except Exception as e:
            print(f"[WebPipeline] Error during response generation: {e}")
            await tts_queue.put(None)
            await worker_task
            await send_callback({"type": "status", "value": "idle"})
