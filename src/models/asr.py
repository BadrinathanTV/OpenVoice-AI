from faster_whisper import WhisperModel
import numpy as np

class ASRModel:
    def __init__(self, model_size="base.en", compute_type="int8"):
        # Runs on GPU if available, else CPU.
        # Int8 on RTX 3070 Ti is extremely fast.
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_data: np.ndarray) -> str:
        # faster-whisper expects 16kHz float32 numpy array
        segments, info = self.model.transcribe(audio_data, beam_size=1, language="en", vad_filter=False)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
