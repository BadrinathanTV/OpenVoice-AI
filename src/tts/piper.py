import os
import subprocess
import numpy as np
import tempfile
import wave

class TTSModel:
    def __init__(self, model_path="en_US-lessac-medium.voice"):
        # We will use Piper TTS python bindings.
        # For simplicity in initialization, we download a default onnx model if missing.
        from piper.voice import PiperVoice
        
        self.model_name = "en_US-lessac-medium"
        self.onnx_path = f"{self.model_name}.onnx"
        self.json_path = f"{self.onnx_path}.json"
        
        if not os.path.exists(self.onnx_path):
            print(f"Downloading Piper TTS model ({self.model_name})...")
            subprocess.run(["wget", "-q", f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/{self.model_name}.onnx", "-O", self.onnx_path])
            subprocess.run(["wget", "-q", f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/{self.model_name}.onnx.json", "-O", self.json_path])
        
        self.voice = PiperVoice.load(self.onnx_path)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        # Synthesize text to numpy array directly from Piper's yields
        audio_arrays = []
        sample_rate = self.voice.config.sample_rate
        for audio_chunk in self.voice.synthesize(text):
            audio_arrays.append(audio_chunk.audio_float_array)
        
        if len(audio_arrays) > 0:
            return np.concatenate(audio_arrays), sample_rate
        else:
            return np.array([], dtype=np.float32), sample_rate
