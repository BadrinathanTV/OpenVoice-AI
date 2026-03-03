import os
import sys
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.io import AudioIO
from src.models.vad import SileroVAD
from src.models.asr import ASRModel
from src.models.llm import LLMModel
from src.models.tts import TTSModel

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
        while True:
            print("\n--- Listening ---")
            speech_buffer = []
            is_speaking = False
            silence_chunks = 0
            
            while True:
                chunk = audio_io.get_audio_chunk()
                speech_detected = vad.is_speech(chunk)
                
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
            
            if len(text.strip()) > 3: # Ignore tiny murmurs
                llm_start = time.time()
                response = llm.generate_response(text)
                llm_latency = time.time() - llm_start
                print(f"AI ({llm_latency:.2f}s): {response}")
                
                tts_start = time.time()
                audio_data, fs = tts.synthesize(response)
                tts_latency = time.time() - tts_start
                print(f"TTS Synthesized ({tts_latency:.2f}s)")
                
                print("Playing...")
                audio_io.play_audio(audio_data, fs)
                
            # Clear trailing mic buffer picked up during processing
            while not audio_io.q_in.empty():
                audio_io.q_in.get()

    except KeyboardInterrupt:
        print("\nStopping application...")
        audio_io.stop_recording()

if __name__ == "__main__":
    main()
