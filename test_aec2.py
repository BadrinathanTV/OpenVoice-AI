import webrtc_audio_processing as webrtc
import numpy as np

ap = webrtc.AudioProcessingModule(aec_type=1)
ap.set_stream_format(16000, 1)
ap.set_reverse_stream_format(16000, 1)

out = ap.process_reverse_stream(np.zeros(512, dtype=np.int16).tobytes())
print("process_reverse_stream returned type:", type(out))

out_stream = ap.process_stream(np.zeros(512, dtype=np.int16).tobytes())
print("process_stream returned type:", type(out_stream))
print("process_stream returned length in bytes:", len(out_stream) if out_stream else "None")
