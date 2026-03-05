import re

class SentenceChunker:
    """
    Buffers incoming text tokens and yields full sentences or logical clauses.
    This prevents TTS engines from trying to synthesize half-words or disjointed grammar.
    Strips markdown formatting so TTS reads clean, natural text.
    """
    def __init__(self, min_words=3):
        self.buffer = ""
        self.min_words = min_words
        # Split on sentence-ending punctuation, commas, or colons, followed by a space or end of string.
        # This makes chunks smaller, reducing TTS latency significantly.
        self.split_pattern = re.compile(r'([.?!,:]+|\n)(\s|$)')

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
        Takes a token string. Yields a completed sentence if a boundary is crossed.
        Skips tiny fragments (< min_words) by keeping them in the buffer.
        """
        self.buffer += token
        
        while True:
            match = self.split_pattern.search(self.buffer)
            if not match:
                break
                
            end_idx = match.end()
            raw_sentence = self.buffer[:end_idx].strip()
            remaining = self.buffer[end_idx:]
            
            cleaned = self._clean_for_tts(raw_sentence)
            
            # If the fragment is too short, keep it in the buffer and wait for more
            word_count = len(cleaned.split()) if cleaned else 0
            if word_count < self.min_words:
                # Only break out if there's no more matches to try
                break
            
            self.buffer = remaining
            if cleaned:
                yield cleaned

    def flush(self):
        """
        Yields any remaining text in the buffer (used at the end of generation).
        """
        cleaned = self._clean_for_tts(self.buffer.strip())
        self.buffer = ""
        if cleaned:
            yield cleaned
