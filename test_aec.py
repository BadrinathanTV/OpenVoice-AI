import webrtc_audio_processing as webrtc
ap = webrtc.AudioProcessingModule(aec_type=1)
ap.set_stream_format(16000, 1)
ap.set_reverse_stream_format(16000, 1)
import numpy as np

try:
    ap.process_reverse_stream(np.zeros(512, dtype=np.int16).tobytes())
    print("512 samples worked")
except Exception as e:
    print(f"512 samples failed: {e}")

try:
    ap.process_reverse_stream(np.zeros(160, dtype=np.int16).tobytes())
    print("160 samples worked")
except Exception as e:
    print(f"160 samples failed: {e}")
