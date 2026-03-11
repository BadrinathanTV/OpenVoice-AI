import os
import sys
import numpy as np

# Add backend to python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vad.silero import SileroVAD

def test_vad_noise_gate():
    print("Initializing SileroVAD with volume_threshold=0.005...")
    vad = SileroVAD(threshold=0.7, volume_threshold=0.005)
    
    # Generate 1 second (16000 samples) of dead silence
    dead_silence = np.zeros(16000, dtype=np.float32)
    
    # Generate 1 second of very quiet noise (e.g. distant chatter)
    # Amplitude of 0.002 (well below 0.005 threshold)
    quiet_noise = np.random.normal(0, 0.002, 16000).astype(np.float32)
    
    # Generate 1 second of loud noise (e.g. person speaking closely)
    # Amplitude of 0.05 (well above 0.005 threshold)
    loud_speech = np.random.normal(0, 0.05, 16000).astype(np.float32)
    
    print("\n--- Testing Dead Silence ---")
    rms1 = np.sqrt(np.mean(np.square(dead_silence)))
    is_speech_1 = vad.is_speech(dead_silence)
    print(f"RMS: {rms1:.6f} | Detected Speech: {is_speech_1}")
    
    print("\n--- Testing Distant Quiet Chatter ---")
    rms2 = np.sqrt(np.mean(np.square(quiet_noise)))
    is_speech_2 = vad.is_speech(quiet_noise)
    print(f"RMS: {rms2:.6f} | Detected Speech: {is_speech_2}")
    
    print("\n--- Testing Loud Speech ---")
    rms3 = np.sqrt(np.mean(np.square(loud_speech)))
    is_speech_3 = vad.is_speech(loud_speech)
    print(f"RMS: {rms3:.6f} | Detected Speech: {is_speech_3} (Note: Silero might STILL say False if it's just white noise, but it should pass the Noise Gate)")

if __name__ == "__main__":
    test_vad_noise_gate()
