import os
import sys

# Add backend to python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tts.sarvam_tts import SarvamTTS

def test_tts():
    print("Initializing SarvamTTS...")
    tts = SarvamTTS(speaker="priya")
    
    test_text = "Hello, can you hear me? This is a test of the Sarvam TTS API."
    print(f"Synthesizing text: '{test_text}'")
    
    audio_array, sample_rate = tts.synthesize(test_text)
    
    if len(audio_array) > 0:
        print(f"SUCCESS! Received audio array of shape {audio_array.shape}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Max amplitude: {audio_array.max():.4f}, Min amplitude: {audio_array.min():.4f}")
    else:
        print("FAILED! Audio array is empty.")

if __name__ == "__main__":
    test_tts()
