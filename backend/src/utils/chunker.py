import re


class SentenceChunker:
    """
    Buffers incoming text tokens and yields small clause-first chunks
    suitable for low-latency TTS.

    The first spoken chunk is intentionally short so playback can start early,
    then later chunks grow a bit to reduce over-fragmented audio.
    Strips markdown formatting so TTS reads clean, natural text.
    """

    def __init__(
        self,
        first_chunk_words: int = 4,
        continuation_words: int = 7,
        max_words: int = 14,
    ) -> None:
        self.buffer = ""
        self.first_chunk_words = first_chunk_words
        self.continuation_words = continuation_words
        self.max_words = max_words
        self.has_emitted_chunk = False
        self.sentence_pattern = re.compile(r'([.?!]+|\n)')
        self.clause_pattern = re.compile(r'([,;:]+| - | -- | — )')

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """
        Strip markdown formatting so TTS reads clean, natural text.
        """
        text = text.replace('`', '')
        # Remove bold/italic markers
        text = text.replace('**', '').replace('*', '')
        # Remove heading markers
        text = re.sub(r'^#{1,6}\s*', '', text)
        # Remove numbered list prefixes like "1. " or "2. "
        text = re.sub(r'^\d+\.\s+', '', text.strip())
        # Remove bullet list prefixes like "- " or "* "
        text = re.sub(r'^[-*]\s+', '', text.strip())
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _target_words(self) -> int:
        return self.continuation_words if self.has_emitted_chunk else self.first_chunk_words

    def _emit_chunk(self, chunk: str) -> str | None:
        cleaned = self._clean_for_tts(chunk)
        if not cleaned:
            return None
        self.has_emitted_chunk = True
        return cleaned

    def _split_at_match_end(self, match_end: int) -> str | None:
        end_idx = match_end
        while end_idx < len(self.buffer) and self.buffer[end_idx].isspace():
            end_idx += 1

        chunk = self.buffer[:end_idx].strip()
        self.buffer = self.buffer[end_idx:]
        return self._emit_chunk(chunk)

    def _split_at_word_count(self, word_count: int) -> str | None:
        raw_words = self.buffer.split()
        if len(raw_words) < word_count:
            return None

        take_raw = ' '.join(raw_words[:word_count])
        start_idx = self.buffer.find(take_raw)
        if start_idx < 0:
            return None

        end_idx = start_idx + len(take_raw)
        while end_idx < len(self.buffer) and self.buffer[end_idx].isspace():
            end_idx += 1

        chunk = self.buffer[:end_idx].strip()
        self.buffer = self.buffer[end_idx:]
        return self._emit_chunk(chunk)

    def process_token(self, token: str):
        """
        Takes a token string. Yields completed chunks suitable for TTS.
        Splits eagerly at punctuation, then clause boundaries, then adaptive
        word counts to keep first-audio latency low.
        """
        self.buffer += token

        while True:
            cleaned_buf = self._clean_for_tts(self.buffer)
            word_count = len(cleaned_buf.split()) if cleaned_buf else 0
            target_words = self._target_words()

            sentence_match = self.sentence_pattern.search(self.buffer)
            if sentence_match and word_count >= 1:
                emitted = self._split_at_match_end(sentence_match.end())
                if emitted:
                    yield emitted
                continue

            clause_match = self.clause_pattern.search(self.buffer)
            if clause_match and word_count >= max(3, target_words - 1):
                emitted = self._split_at_match_end(clause_match.end())
                if emitted:
                    yield emitted
                continue

            if word_count >= self.max_words:
                emitted = self._split_at_word_count(self.max_words)
                if emitted:
                    yield emitted
                continue

            if word_count >= target_words and self.buffer.endswith((" ", "\n")):
                emitted = self._split_at_word_count(target_words)
                if emitted:
                    yield emitted
                continue

            break

    def flush(self):
        """
        Yields any remaining text in the buffer (used at the end of generation).
        """
        cleaned = self._emit_chunk(self.buffer.strip())
        self.buffer = ""
        if cleaned:
            yield cleaned
