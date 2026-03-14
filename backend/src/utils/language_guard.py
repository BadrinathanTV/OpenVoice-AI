import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageDecision:
    language: str
    confidence: float
    allow: bool


class EnglishLanguageGuard:
    """
    Production-grade transcript gate backed by a real language classifier.

    This keeps policy separate from the WebSocket pipeline and makes the
    behavior tunable via environment variables.
    """

    def __init__(self) -> None:
        self.enabled = self._is_truthy(os.getenv("ASR_ENFORCE_ENGLISH", "true"))
        self.min_confidence = float(
            os.getenv("ASR_ENGLISH_CONFIDENCE_THRESHOLD", "0.80")
        )
        self.min_chars = int(os.getenv("ASR_ENGLISH_MIN_CHARS", "24"))
        self.min_words = int(os.getenv("ASR_ENGLISH_MIN_WORDS", "5"))

        if not self.enabled:
            self._identifier = None
            return

        try:
            from langid.langid import LanguageIdentifier, model
        except ImportError as exc:
            raise RuntimeError(
                "English transcript enforcement requires the `langid` package. "
                "Run `uv sync` from the repo root to install updated dependencies."
            ) from exc

        identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        identifier.set_languages(None)
        self._identifier = identifier

    def evaluate(self, text: str) -> LanguageDecision:
        stripped = text.strip()
        if not self.enabled or len(stripped) < 2:
            return LanguageDecision(language="unknown", confidence=0.0, allow=True)

        alpha_chars = sum(char.isalpha() for char in stripped)
        if alpha_chars < 3:
            return LanguageDecision(language="unknown", confidence=0.0, allow=True)

        word_count = sum(1 for token in stripped.split() if any(ch.isalpha() for ch in token))
        if alpha_chars < self.min_chars or word_count < self.min_words:
            return LanguageDecision(language="unknown", confidence=0.0, allow=True)

        assert self._identifier is not None
        language, confidence = self._identifier.classify(stripped)

        if language == "en":
            return LanguageDecision(
                language=language,
                confidence=float(confidence),
                allow=True,
            )

        allow = float(confidence) < self.min_confidence
        return LanguageDecision(
            language=language,
            confidence=float(confidence),
            allow=allow,
        )

    @staticmethod
    def _is_truthy(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
