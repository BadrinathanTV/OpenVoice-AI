import time
import numpy as np
import sounddevice as sd
import queue
import threading

class AudioIO:
    def __init__(self, sample_rate=16000, channels=1, chunk_duration_ms=30):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        self.q_in = queue.Queue()
        self.stream_in = None
        self.stream_out = None

    def _audio_callback(self, indata, frames, time, status):
        # This is called for each audio block from the microphone
        if status:
            print(status, flush=True)

        mic_float = indata.copy().flatten()
        self.q_in.put(mic_float)

    def start_recording(self):
        self.stream_in = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.chunk_size,
            dtype='float32'
        )
        self.stream_in.start()

    def stop_recording(self):
        if self.stream_in:
            self.stream_in.stop()
            self.stream_in.close()
            self.stream_in = None
        if self.stream_out:
            self.stream_out.stop()
            self.stream_out.close()
            self.stream_out = None

    def get_audio_chunk(self):
        # Blocking call until a chunk is available
        return self.q_in.get()

    def play_audio(self, audio_data, sample_rate=None, interrupt_flag=None):
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Use sd.play() for discrete audio chunks with blocking via sd.wait()
        # This ensures each audio chunk plays completely before the function returns
        if len(audio_data) > 0:
            try:
                sd.play(audio_data, samplerate=sample_rate, blocking=False)
                if interrupt_flag is not None:
                    duration = len(audio_data) / sample_rate
                    start_time = time.time()
                    while time.time() - start_time < duration:
                        if interrupt_flag[0]:
                            sd.stop()
                            break
                        time.sleep(0.01)
                else:
                    sd.wait()  # Block until playback completes
            except Exception as e:
                print(f"Error playing audio: {e}")

    def interrupt(self):
        """Immediately aborts playback."""
        try:
            sd.stop()
        except Exception:
            pass
