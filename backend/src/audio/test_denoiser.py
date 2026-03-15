import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.audio.denoiser import Denoiser

def test_denoiser():
    print("Initializing Denoiser...")
    try:
        denoiser = Denoiser()
    except Exception as e:
        print(f"Failed to initialize Denoiser: {e}")
        return

    # Create a noisy signal: 1 sec of 16kHz audio
    # 440Hz sine wave + white noise
    fs = 16000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.2 * np.random.normal(size=fs)
    noisy_signal = (sine + noise).astype(np.float32)

    print(f"Noisy signal RMS: {np.sqrt(np.mean(noisy_signal**2)):.4f}")

    # Process in 10ms (160 samples) chunks
    chunk_size = 160
    enhanced_signal = []
    
    import time
    print(f"Processing {len(noisy_signal)//chunk_size} chunks...")
    latencies = []
    for i in range(0, len(noisy_signal), chunk_size):
        chunk = noisy_signal[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
        
        start = time.perf_counter()
        enhanced_chunk = denoiser.enhance(chunk)
        latencies.append((time.perf_counter() - start) * 1000)
        enhanced_signal.extend(enhanced_chunk)

    print(f"Average latency per 10ms chunk: {sum(latencies)/len(latencies):.2f}ms")
    print(f"Max latency: {max(latencies):.2f}ms")

    enhanced_signal = np.array(enhanced_signal)
    print(f"Enhanced signal RMS: {np.sqrt(np.mean(enhanced_signal**2)):.4f}")
    
    if np.sqrt(np.mean(enhanced_signal**2)) < np.sqrt(np.mean(noisy_signal**2)):
        print("Success: RMS reduced (noise likely suppressed).")
    else:
        print("Warning: RMS not reduced. Check if denoiser is working as expected.")

if __name__ == "__main__":
    test_denoiser()
