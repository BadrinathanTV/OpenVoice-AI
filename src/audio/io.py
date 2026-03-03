import numpy as np
import sounddevice as sd
import queue

class AudioIO:
    def __init__(self, sample_rate=16000, channels=1, chunk_duration_ms=30):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.q_in = queue.Queue()
        self.stream_in = None
        self.stream_out = None

    def _audio_callback(self, indata, frames, time, status):
        # This is called for each audio block
        if status:
            print(status, flush=True)
        self.q_in.put(indata.copy())

    def start_recording(self):
        self.stream_in = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size
        )
        self.stream_in.start()

    def stop_recording(self):
        if self.stream_in:
            self.stream_in.stop()
            self.stream_in.close()

    def get_audio_chunk(self):
        # Blocking call until a chunk is available
        return self.q_in.get()

    def play_audio(self, audio_data, sample_rate=None):
        if sample_rate is None:
            sample_rate = self.sample_rate
        # This is blocking. We will upgrade this to async streaming in phase 2/3.
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()
