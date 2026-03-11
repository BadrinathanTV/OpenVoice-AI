import re

class SentenceChunker:
    """
    Buffers incoming text tokens and yields small word-group chunks
    suitable for low-latency TTS. Yields at punctuation boundaries or
    every `min_words` words, whichever comes first.
    Strips markdown formatting so TTS reads clean, natural text.
    """
    def __init__(self, min_words=10):
        self.buffer = ""
        self.min_words = min_words
        # Split on sentence-ending punctuation, commas, or colons,
        # followed by a space or end of string.
        self.punct_pattern = re.compile(r'([.?!,:]+|\n)(\s|$)')

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """
        Strip markdown formatting so TTS reads clean, natural text.
        """
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

    def process_token(self, token: str):
        """
        Takes a token string. Yields completed chunks suitable for TTS.
        Splits eagerly at punctuation, or every min_words words.
        """
        self.buffer += token

        while True:
            cleaned_buf = self._clean_for_tts(self.buffer)
            word_count = len(cleaned_buf.split()) if cleaned_buf else 0

            # Strategy 1: Always split at punctuation if we have at least 1 word
            punct_match = self.punct_pattern.search(self.buffer)
            if punct_match and word_count >= 1:
                end_idx = punct_match.end()
                chunk = self.buffer[:end_idx].strip()
                self.buffer = self.buffer[end_idx:]
                cleaned = self._clean_for_tts(chunk)
                if cleaned:
                    yield cleaned
                continue

            # Strategy 2: Split at word boundaries if we have enough words
            if word_count >= self.min_words:
                raw_words = self.buffer.split()
                # Take min_words words from the front
                take_raw = ' '.join(raw_words[:self.min_words])
                idx = self.buffer.find(take_raw)
                if idx >= 0:
                    end_pos = idx + len(take_raw)
                    chunk = self.buffer[:end_pos].strip()
                    self.buffer = self.buffer[end_pos:]
                    cleaned = self._clean_for_tts(chunk)
                    if cleaned:
                        yield cleaned
                    continue

            break

    def flush(self):
        """
        Yields any remaining text in the buffer (used at the end of generation).
        """
        cleaned = self._clean_for_tts(self.buffer.strip())
        self.buffer = ""
        if cleaned:
            yield cleaned
