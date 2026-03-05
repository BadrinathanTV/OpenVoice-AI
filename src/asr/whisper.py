import torch
import numpy as np
from qwen_asr import Qwen3ASRModel
import warnings
import os

# Suppress verbose generation warnings from Qwen
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


class ASRModel:
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_name = "GPU" if device.startswith("cuda") else "CPU"
        print(f"[ASR] Loading Qwen3-ASR-0.6B on {device_name}...")

        self.model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            device_map=device,
            max_new_tokens=256,
        )
        print(f"[ASR] Model loaded on {device_name}")


    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        audio_data : numpy float32 waveform
        sample_rate : expected 16000
        """

        results = self.model.transcribe(
            audio=(audio_data, sample_rate),
            language="English"
        )

        return results[0].text.strip()