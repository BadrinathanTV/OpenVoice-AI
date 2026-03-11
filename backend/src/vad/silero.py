import torch
import numpy as np
from silero_vad import load_silero_vad
from src.core.interfaces import IVAD

class SileroVAD(IVAD):
    def __init__(self, threshold=0.7, volume_threshold=0.02, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.volume_threshold = volume_threshold
        
        # Load the Silero VAD model with GPU acceleration if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_silero_vad().to(self.device)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if a single audio chunk contains speech.
        Audio chunk should be a numpy array of float32, length equivalent to e.g., 30ms.
        """
        # 1. Volume Noise Gate (RMS Threshold)
        # Calculate Root Mean Square energy of the audio chunk to ignore distant/quiet background chatter
        rms = np.sqrt(np.mean(np.square(audio_chunk)))
        if rms < self.volume_threshold:
            return False

        # Silero VAD requires chunks of 512 (32ms), 1024, or 1536 samples at 16kHz
        # Let's pad or truncate to 512 for a quick check.
        tensor_chunk = torch.from_numpy(audio_chunk).float().to(self.device)
        
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
        # Reset model state periodically to avoid false positives from accumulated internal state
        if speech_prob < 0.3:
            self.model.reset_states()
        
        return speech_prob > self.threshold
