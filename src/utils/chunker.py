import re

class SentenceChunker:
    """
    Buffers incoming text tokens and yields full sentences or logical clauses.
    This prevents TTS engines from trying to synthesize half-words or disjointed grammar.
    """
    def __init__(self):
        self.buffer = ""
        # Split on sentence-ending punctuation, commas, or colons, followed by a space or end of string.
        # This makes chunks smaller, reducing TTS latency significantly.
        self.split_pattern = re.compile(r'([.?!,:]+|\n)(\s|$)')

    def process_token(self, token: str):
        """
        Takes a token string. Yields a completed sentence if a boundary is crossed.
        """
        self.buffer += token
        
        # Check for sentence boundaries
        match = self.split_pattern.search(self.buffer)
        if match:
            # We found a boundary!
            end_idx = match.end()
            sentence = self.buffer[:end_idx].strip()
            self.buffer = self.buffer[end_idx:]
            
            if sentence:
                yield sentence

    def flush(self):
        """
        Yields any remaining text in the buffer (used at the end of generation).
        """
        sentence = self.buffer.strip()
        self.buffer = ""
        if sentence:
            yield sentence
