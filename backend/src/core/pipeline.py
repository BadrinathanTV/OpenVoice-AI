import time
import numpy as np
import threading
import queue
from typing import Dict

from src.utils.chunker import SentenceChunker
from src.audio.io import AudioIO
from src.core.interfaces import IVAD, IASR, ILLM, ITTS

class VoicePipeline:
    def __init__(
        self,
        audio_io: AudioIO,
        vad: IVAD,
        asr: IASR,
        llm: ILLM,
        tts_models: Dict[str, ITTS],
        default_agent_name: str = "CustomerCare"
    ):
        self.audio_io = audio_io
        self.vad = vad
        self.asr = asr
        self.llm = llm
        self.tts_models = tts_models
        self.default_agent_name = default_agent_name

    def start(self):
        """Starts the voice pipeline loop."""
        self.audio_io.start_recording()
        try:
            interrupt_buffer = []
            while True:
                speech_buffer = interrupt_buffer.copy()
                interrupt_buffer.clear()
                is_speaking = len(speech_buffer) > 0 # If we carried over interrupt audio, we are already speaking
                silence_chunks = 0
                was_interrupted = False
                
                while True:
                    chunk = self.audio_io.get_audio_chunk()
                    rms_volume = np.sqrt(np.mean(chunk**2))
                    
                    # Base threshold (0.005) + VAD ensures we only trigger on actual speech, not static
                    speech_detected = self.vad.is_speech(chunk) and rms_volume > 0.005
                    
                    if speech_detected:
                        if not is_speaking:
                            print("Speech detected!")
                            is_speaking = True
                        speech_buffer.append(chunk)
                        silence_chunks = 0
                    elif is_speaking:
                        speech_buffer.append(chunk)
                        silence_chunks += 1
                        # Stop after ~1 second of silence (32ms * 30 chunks = ~960ms)
                        if silence_chunks > 30:
                            break
                
                audio_array = np.concatenate(speech_buffer)
                # Flatten to 1D if stereo
                if len(audio_array.shape) > 1:
                     audio_array = audio_array[:, 0]
                     
                # Transcribe
                start_time = time.time()
                text = self.asr.transcribe(audio_array)
                asr_latency = time.time() - start_time
                print(f"\n[ASR Result]: '{text}' (Latency: {asr_latency:.2f}s)")

                
                if len(text.strip()) >= 2: # Ignore empty murmurs, but allow short answers like 'No' or 'Hi'
                    llm_start = time.time()
                    
                    chunker = SentenceChunker()
                    interrupt_flag = [False]
                    is_synthesizing = [False] # New flag for echo cancellation
                    tts_start_time = [0.0]
                    ai_spoken_text = ""
                    
                    def listen_for_interrupt():
                        # Clear trailing queue before listening for interrupt
                        while not self.audio_io.q_in.empty():
                            self.audio_io.q_in.get()
                            
                        consecutive_speech = 0
                        recent_chunks = []
                        while not interrupt_flag[0]:
                            try:
                                # Non-blocking get to allow thread to exit if not interrupted
                                chunk = self.audio_io.q_in.get(timeout=0.1) 
                                
                                recent_chunks.append(chunk)
                                if len(recent_chunks) > 40: # Buffer ~1.28 seconds of audio so we don't cut off their first word
                                    recent_chunks.pop(0)
                                    
                                rms_volume = np.sqrt(np.mean(chunk**2))
                                is_speech = self.vad.is_speech(chunk)
                                
                                # DYNAMIC DUCKING:
                                # Since we are actively generating and playing TTS audio during this loop,
                                # the microphone will definitely hear the AI's "echo" from the speakers.
                                # We set a high threshold (0.05) so the VAD strictly ignores the quiet echoes,
                                # but will instantly trigger if the human speaks loudly or close to the mic.
                                volume_threshold = 0.05
                                
                                if is_speech and rms_volume > volume_threshold:
                                    consecutive_speech += 1
                                    if consecutive_speech > 3: # Require ~96ms of loud speech to interrupt
                                        print("\n[INTERRUPTED BY USER]")
                                        interrupt_flag[0] = True
                                        interrupt_buffer.extend(recent_chunks)
                                        # Do NOT abort the ALSA stream aggressively here, it crashes the Linux PA host
                                        break
                                else:
                                    consecutive_speech = 0
                            except queue.Empty:
                                continue

                    interrupt_thread = threading.Thread(target=listen_for_interrupt)
                    interrupt_thread.daemon = True
                    interrupt_thread.start()
                    
                    # Only print the 'Speaking' label once per response before the chunks start streaming
                    print("\n[Speaking]: ", end="", flush=True)
                    
                    first_token_received = False
                    for token in self.llm.generate_response_stream(text):
                        if not first_token_received:
                            ttft = time.time() - llm_start
                            print(f"[LLM TTFT: {ttft:.2f}s] ", end="", flush=True)
                            first_token_received = True

                        if interrupt_flag[0]:
                            break
                        
                        print(token, end="", flush=True)
                        for sentence in chunker.process_token(token):
                            if interrupt_flag[0]: break
                            ai_spoken_text += sentence + " "
                            active_agent = self.llm.active_agent_name
                            tts = self.tts_models.get(active_agent, self.tts_models.get(self.default_agent_name))
                            if not tts:
                                # Fallback if standard dictionary lookup fails
                                tts = list(self.tts_models.values())[0]
                            
                            tts_gen_start = time.time()
                            audio_data, fs = tts.synthesize(sentence)
                            tts_latency = time.time() - tts_gen_start
                            print(f" [TTS: {tts_latency:.2f}s] ", end="", flush=True)
                            
                            if interrupt_flag[0]: break
                            if tts_start_time[0] == 0.0:
                                tts_start_time[0] = time.time()
                                
                            is_synthesizing[0] = True
                            try:
                                self.audio_io.play_audio(audio_data, fs, interrupt_flag)
                            finally:
                                is_synthesizing[0] = False
                                
                    if not interrupt_flag[0]:
                        for sentence in chunker.flush():
                            if interrupt_flag[0]: break
                            ai_spoken_text += sentence + " "
                            active_agent = self.llm.active_agent_name
                            tts = self.tts_models.get(active_agent, self.tts_models.get(self.default_agent_name))
                            if not tts:
                                tts = list(self.tts_models.values())[0]
                            
                            tts_gen_start = time.time()
                            audio_data, fs = tts.synthesize(sentence)
                            tts_latency = time.time() - tts_gen_start
                            print(f" [TTS: {tts_latency:.2f}s] ", end="", flush=True)
                            
                            if interrupt_flag[0]: break
                            if tts_start_time[0] == 0.0:
                                tts_start_time[0] = time.time()
                                
                            is_synthesizing[0] = True
                            try:
                                self.audio_io.play_audio(audio_data, fs, interrupt_flag)
                            finally:
                                is_synthesizing[0] = False
                    
                    # Capture the actual interruption state before we force the flag to True to kill the thread
                    was_interrupted = interrupt_flag[0]
                    
                    # Save whatever the AI managed to speak into its conversational memory
                    self.llm.add_ai_message(ai_spoken_text.strip())
                    
                    # If we were interrupted, add a system note so the LLM knows why its text was cut off
                    if was_interrupted:
                        self.llm.add_human_message("[System Note: The user interrupted your previous response mid-sentence here. The next message is what they said over you.]")
                    
                    # Wait for the thread to exit before touching the OS audio device
                    interrupt_flag[0] = True 
                    interrupt_thread.join()
                    
                    print() # Newline after response
                    
                    # We need to stop the output audio instantly if interrupted, safely on the main thread
                    if was_interrupted:
                        self.audio_io.interrupt()
                    
                # Clear trailing mic buffer picked up during processing or speaking
                # If interrupted, DO NOT clear the queue so the new speech isn't lost!
                if not was_interrupted:
                    while not self.audio_io.q_in.empty():
                        self.audio_io.q_in.get()
                        
                    # Cooldown flush: discard ~500ms of mic audio to prevent
                    # the ASR from transcribing the AI's own residual echo
                    cooldown_end = time.time() + 0.5
                    while time.time() < cooldown_end:
                        try:
                            self.audio_io.q_in.get(timeout=0.05)
                        except queue.Empty:
                            pass

        except KeyboardInterrupt:
            self.audio_io.stop_recording()
