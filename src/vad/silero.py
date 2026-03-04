import torch
import numpy as np
from silero_vad import load_silero_vad

class SileroVAD:
    def __init__(self, threshold=0.75, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        
        # Load the Silero VAD model from the python package (downloads ONNX to cache natively)
        self.model = load_silero_vad()

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if a single audio chunk contains speech.
        Audio chunk should be a numpy array of float32, length equivalent to e.g., 30ms.
        """
        # Silero VAD requires chunks of 512 (32ms), 1024, or 1536 samples at 16kHz
        # Let's pad or truncate to 512 for a quick check.
        tensor_chunk = torch.from_numpy(audio_chunk).float()
        
        # Handle stereo by flattening or taking first channel
        if len(tensor_chunk.shape) > 1:
            tensor_chunk = tensor_chunk[:, 0]
            
        # Ensure it's correctly sized. For continuous we might want to use the VADIterator, 
        # but for phase 1 simple chunk check:
        # We can just process windows of 512.
        if len(tensor_chunk) > 512:
            tensor_chunk = tensor_chunk[:512]
        elif len(tensor_chunk) < 512:
            # Pad
            padding = 512 - len(tensor_chunk)
            tensor_chunk = torch.nn.functional.pad(tensor_chunk, (0, padding))

        with torch.no_grad():
            speech_prob = self.model(tensor_chunk, self.sampling_rate).item()
        
        return speech_prob > self.threshold
