import os
import base64
import io
import wave

import numpy as np
import requests
from src.core.interfaces import ITTS

class SarvamTTS(ITTS):
    def __init__(self, speaker: str = "meera") -> None:
        super().__init__()
        self.speaker = speaker
        self.api_key = os.getenv("SARVAM_API_KEY", "")
        self.url = "https://api.sarvam.ai/text-to-speech"
        
        if not self.api_key:
            print("WARNING: SARVAM_API_KEY not found in environment variables. TTS will fail.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesizes text into audio using Sarvam AI.
        Returns a tuple of (audio_array, sample_rate).
        """
        if not self.api_key:
            return np.array([], dtype=np.float32), 16000

        payload = {
            "inputs": [text],
            "target_language_code": "en-IN",
            "speaker": self.speaker,
            "pace": 1.0,
            "speech_sample_rate": 16000,
            "enable_preprocessing": True,
            "model": "bulbul:v3"
        }
        
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            audios = data.get("audios", [])
            if isinstance(audios, list) and audios:
                base64_audio = audios[0]
                
                # Decode base64 to WAV bytes
                audio_bytes = base64.b64decode(base64_audio)
                
                # Parse the WAV file to extract pure PCM audio frames and ignore metadata headers
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                    raw_pcm = wav_file.readframes(wav_file.getnframes())
                    # Ensure sample rate matches what we expected
                    ret_sample_rate = wav_file.getframerate()
                
                # Convert 16-bit PCM bytes to float32 numpy array
                int16_array = np.frombuffer(raw_pcm, dtype=np.int16)
                float32_array = int16_array.astype(np.float32) / 32768.0
                
                return float32_array, ret_sample_rate
            else:
                print(f"[SarvamTTS] Error: No audio returned in response. {data}")
                return np.array([], dtype=np.float32), 16000
                
        except Exception as e:
            print(f"[SarvamTTS] API Error: {e}")
            response = getattr(e, "response", None)
            if response is not None:
                print(f"Details: {response.text}")
            return np.array([], dtype=np.float32), 16000
