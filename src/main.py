import os
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.io import AudioIO
from src.vad.silero import SileroVAD
from src.asr.whisper import ASRModel
from src.llm.client import LLMModel
from src.tts.piper import TTSModel

def main():
    print("Loading models (this takes a few seconds)...")
    audio_io = AudioIO(sample_rate=16000, chunk_duration_ms=32) # 32ms = 512 samples
    vad = SileroVAD()
    asr = ASRModel(model_size="base.en")
    llm = LLMModel()
    tts = TTSModel()
    
    print("Models loaded. Starting audio stream...")
    audio_io.start_recording()
    
    try:
        interrupt_buffer = []
        while True:
            print("\n--- Listening ---")
            speech_buffer = interrupt_buffer.copy()
            interrupt_buffer.clear()
            is_speaking = len(speech_buffer) > 0 # If we carried over interrupt audio, we are already speaking
            silence_chunks = 0
            was_interrupted = False
            
            while True:
                chunk = audio_io.get_audio_chunk()
                rms_volume = np.sqrt(np.mean(chunk**2))
                
                # Base threshold (0.005) + VAD ensures we only trigger on actual speech, not static
                speech_detected = vad.is_speech(chunk) and rms_volume > 0.005
                
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
            
            print("Processing audio...")
            audio_array = np.concatenate(speech_buffer)
            # Flatten to 1D if stereo
            if len(audio_array.shape) > 1:
                 audio_array = audio_array[:, 0]
                 
            # Transcribe
            start_time = time.time()
            text = asr.transcribe(audio_array)
            asr_latency = time.time() - start_time
            print(f"USER ({asr_latency:.2f}s): {text}")
            
            if len(text.strip()) >= 2: # Ignore empty murmurs, but allow short answers like 'No' or 'Hi'
                llm_start = time.time()
                
                from src.utils.chunker import SentenceChunker
                import threading
                
                chunker = SentenceChunker()
                interrupt_flag = [False]
                tts_start_time = [0.0]
                ai_spoken_text = ""
                
                def listen_for_interrupt():
                    # Clear trailing queue before listening for interrupt
                    while not audio_io.q_in.empty():
                        audio_io.q_in.get()
                        
                    consecutive_speech = 0
                    recent_chunks = []
                    while not interrupt_flag[0]:
                        try:
                            # Non-blocking get to allow thread to exit if not interrupted
                            chunk = audio_io.q_in.get(timeout=0.1) 
                            
                            recent_chunks.append(chunk)
                            if len(recent_chunks) > 15:
                                recent_chunks.pop(0)
                                
                            rms_volume = np.sqrt(np.mean(chunk**2))
                            
                            # DYNAMIC DUCKING:
                            # Since we are actively generating and playing TTS audio during this loop,
                            # the microphone will definitely hear the AI's "echo" from the speakers.
                            # We set a high threshold (0.05) so the VAD strictly ignores the quiet echoes,
                            # but will instantly trigger if the human speaks loudly or close to the mic.
                            volume_threshold = 0.05
                            
                            if vad.is_speech(chunk) and rms_volume > volume_threshold:
                                consecutive_speech += 1
                                if consecutive_speech > 7: # Require ~224ms of loud speech to interrupt
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
                
                print("AI: ", end="", flush=True)
                
                for token in llm.generate_response_stream(text):
                    if interrupt_flag[0]:
                        break
                    
                    print(token, end="", flush=True)
                    for sentence in chunker.process_token(token):
                        if interrupt_flag[0]: break
                        ai_spoken_text += sentence + " "
                        audio_data, fs = tts.synthesize(sentence)
                        if interrupt_flag[0]: break
                        if tts_start_time[0] == 0.0:
                            tts_start_time[0] = time.time()
                        audio_io.play_audio(audio_data, fs)
                
                if not interrupt_flag[0]:
                    for sentence in chunker.flush():
                        if interrupt_flag[0]: break
                        ai_spoken_text += sentence + " "
                        audio_data, fs = tts.synthesize(sentence)
                        if interrupt_flag[0]: break
                        if tts_start_time[0] == 0.0:
                            tts_start_time[0] = time.time()
                        audio_io.play_audio(audio_data, fs)
                
                # Capture the actual interruption state before we force the flag to True to kill the thread
                was_interrupted = interrupt_flag[0]
                
                # Save whatever the AI managed to speak into its conversational memory
                llm.add_ai_message(ai_spoken_text.strip())
                
                # If we were interrupted, add a system note so the LLM knows why its text was cut off
                if was_interrupted:
                    llm.add_human_message("[System Note: The user interrupted your previous response mid-sentence here. The next message is what they said over you.]")
                
                # Wait for the thread to exit before touching the OS audio device
                interrupt_flag[0] = True 
                interrupt_thread.join()
                
                print() # Newline after response
                
                # We need to stop the output audio instantly if interrupted, safely on the main thread
                if was_interrupted:
                    audio_io.interrupt()
                
            # Clear trailing mic buffer picked up during processing or speaking
            # If interrupted, DO NOT clear the queue so the new speech isn't lost!
            if not was_interrupted:
                while not audio_io.q_in.empty():
                    audio_io.q_in.get()

    except KeyboardInterrupt:
        print("\nStopping application...")
        audio_io.stop_recording()

if __name__ == "__main__":
    main()
