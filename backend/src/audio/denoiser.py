import os
import time
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
from typing import Optional

class Denoiser:
    """
    Real-time audio denoiser using DeepFilterNet3 ONNX model.
    Optimized for low latency with proper polyphase resampling.
    """
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), 
                "../../models/deepfilternet/DeepFilterNet3.onnx"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DeepFilterNet3 model not found at {model_path}")

        # Optimization: Use fixed thread count for real-time consistency
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts)
        
        # Default attenuation limit. 12 dB is standard for "gentle" denoising.
        self.atten_lim_db = np.array(12.0, dtype=np.float32)
        
        # Sampling rates — 16kHz ↔ 48kHz is an exact 1:3 ratio
        self.project_sr = 16000
        self.model_sr = 48000
        self.model_frame_size = 480 
        self.project_frame_size = 160 # 10ms at 16kHz
        
        # Initialize mutable state
        self.reset()

    def reset(self) -> None:
        """Reset all internal state to factory defaults.
        
        Must be called between sessions/connections to prevent stale neural
        network state from bleeding into a new user's audio stream.
        """
        self.states = np.zeros(45304, dtype=np.float32)
        self.buffer = np.array([], dtype=np.float32)

    def _resample(self, x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Anti-aliased polyphase resampling between 16kHz and 48kHz."""
        if src_sr == dst_sr or len(x) == 0:
            return x
        
        # 16kHz ↔ 48kHz is always a 1:3 or 3:1 ratio
        if dst_sr > src_sr:
            up, down = dst_sr // src_sr, 1   # e.g. 3, 1
        else:
            up, down = 1, src_sr // dst_sr    # e.g. 1, 3
        
        return resample_poly(x, up, down).astype(np.float32)

    def enhance(self, chunk: np.ndarray) -> np.ndarray:
        """
        Enhances audio by processing only full 10ms increments.
        Maintains an internal buffer for remainders.
        """
        if len(chunk) == 0:
            return chunk
            
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, chunk])
        
        # Determine how many 10ms (160 sample) blocks we can process
        num_blocks = len(self.buffer) // self.project_frame_size
        if num_blocks == 0:
            return np.array([], dtype=np.float32)
            
        # Extract the part to process
        to_process = self.buffer[:num_blocks * self.project_frame_size]
        self.buffer = self.buffer[num_blocks * self.project_frame_size:]
        
        # 1. Resample 16kHz → 48kHz with anti-aliased polyphase filter
        resampled_full = self._resample(to_process, self.project_sr, self.model_sr)
        
        enhanced_resampled_list = []
        
        # 2. Process in 10ms (480 samples) increments
        for i in range(0, len(resampled_full), self.model_frame_size):
            sub_chunk = resampled_full[i:i + self.model_frame_size]
            
            # This should always be exactly 480 samples because of our block calculation
            if len(sub_chunk) < self.model_frame_size:
                break 
            
            inputs = {
                "input_frame": sub_chunk,
                "states": self.states,
                "atten_lim_db": self.atten_lim_db
            }
            
            outputs = self.session.run(None, inputs)
            self.states = outputs[1]
            enhanced_resampled_list.append(outputs[0])
        
        if not enhanced_resampled_list:
             return np.array([], dtype=np.float32)
             
        enhanced_resampled_full = np.concatenate(enhanced_resampled_list)
        
        # 3. Resample 48kHz → 16kHz with anti-aliased polyphase filter
        enhanced_back = self._resample(enhanced_resampled_full, self.model_sr, self.project_sr)
        
        # 4. Clamp to [-1.0, 1.0] to prevent clipping from neural network output
        np.clip(enhanced_back, -1.0, 1.0, out=enhanced_back)
        
        return enhanced_back
