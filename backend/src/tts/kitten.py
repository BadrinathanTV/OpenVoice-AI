import os
import subprocess

import numpy as np
from src.core.interfaces import ITTS

KITTEN_SAMPLE_RATE = 24_000
DEFAULT_REPO = "KittenML/kitten-tts-nano-0.1"
DEFAULT_MODEL_FILE = "kitten_tts_nano_v0_1.onnx"
DEFAULT_VOICES_FILE = "voices.npz"

DEFAULT_VOICE = "Kiki"
VOICE_ALIASES = {
    "Bella": "expr-voice-2-f",
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-3-f",
    "Bruno": "expr-voice-3-m",
    "Rosie": "expr-voice-4-f",
    "Hugo": "expr-voice-4-m",
    "Kiki": "expr-voice-5-f",
    "Leo": "expr-voice-5-m",
}


class TTSModel(ITTS):

    def __init__(
        self,
        model_name: str = DEFAULT_REPO,
        model_path: str | None = None,
        voices_path: str | None = None,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
    ):
        from kittentts import KittenTTS

        self.model_name = model_name
        self.voice = VOICE_ALIASES.get(voice, voice)
        self.speed = speed
        self.sample_rate = KITTEN_SAMPLE_RATE

        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
            "kitten",
        )
        os.makedirs(models_dir, exist_ok=True)

        model_path = model_path or os.path.join(models_dir, DEFAULT_MODEL_FILE)
        voices_path = voices_path or os.path.join(models_dir, DEFAULT_VOICES_FILE)

        if not os.path.exists(model_path) or not os.path.exists(voices_path):
            base_url = f"https://huggingface.co/{self.model_name}/resolve/main"
            if not os.path.exists(model_path):
                print(f"[KittenTTS] Downloading model from {self.model_name}...")
                subprocess.run(
                    ["curl", "-L", "-s", "-o", model_path, f"{base_url}/{DEFAULT_MODEL_FILE}"],
                    check=True,
                )
            if not os.path.exists(voices_path):
                print(f"[KittenTTS] Downloading voices from {self.model_name}...")
                subprocess.run(
                    ["curl", "-L", "-s", "-o", voices_path, f"{base_url}/{DEFAULT_VOICES_FILE}"],
                    check=True,
                )

        print(f"[KittenTTS] Loading model '{self.model_name}' ...")
        self._model = KittenTTS(model_path=model_path, voices_path=voices_path)
        if self.voice not in getattr(self._model, "available_voices", []):
            available = getattr(self._model, "available_voices", [])
            fallback_voice = available[0] if available else "expr-voice-2-f"
            print(
                f"[KittenTTS] Voice '{voice}' unavailable. Falling back to '{fallback_voice}'."
            )
            self.voice = fallback_voice
        print(f"[KittenTTS] Model ready. Voice='{self.voice}', speed={speed}")


    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        audio = self._model.generate(
            text,
            voice=self.voice,
            speed=self.speed,
        )
        audio = np.asarray(audio, dtype=np.float32).flatten()
        return audio, self.sample_rate

    def synthesize_streaming(self, text: str):
        audio, sample_rate = self.synthesize(text)
        if audio.size > 0:
            yield audio, sample_rate
