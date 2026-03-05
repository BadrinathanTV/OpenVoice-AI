import os
import subprocess
import numpy as np
import tempfile
import wave

class TTSModel:
    def __init__(self, model_name="en_US-amy-medium"):
        # We will use Piper TTS python bindings.
        # For simplicity in initialization, we download a default onnx model if missing.
        from piper.voice import PiperVoice
        import onnxruntime
        
        self.model_name = model_name
        self.onnx_path = f"{self.model_name}.onnx"
        self.json_path = f"{self.onnx_path}.json"
        
        # Enable GPU acceleration for ONNX Runtime
        available_providers = onnxruntime.get_available_providers()
        device_name = "GPU" if 'CUDAExecutionProvider' in available_providers else "CPU"

        
        if not os.path.exists(self.onnx_path):
            parts = self.model_name.split('-')
            lang_speaker = parts[0]
            speaker = parts[1]
            quality = parts[2]
            
            # Extract just the "en" or "fr" from "en_US" or "fr_FR"
            lang = lang_speaker[:2]
            
            base_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/{lang}/{lang_speaker}/{speaker}/{quality}"
            
            print(f"Downloading {self.model_name}...")
            subprocess.run(["curl", "-L", "-s", "-o", self.onnx_path, f"{base_url}/{self.model_name}.onnx"], check=True)
            subprocess.run(["curl", "-L", "-s", "-o", self.json_path, f"{base_url}/{self.model_name}.onnx.json"], check=True)
        
        # Set providers for GPU acceleration
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        os.environ['ONNXRUNTIME_PROVIDERS'] = ','.join(providers)

        import torch
        use_cuda = torch.cuda.is_available()
        self.voice = PiperVoice.load(self.onnx_path, use_cuda=use_cuda)


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