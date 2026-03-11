from src.agents.session import VoiceSession
import os
import time

session = VoiceSession()

start = time.time()
print("Sending message...")
for token in session.stream_response("Tell me a story about a brave knight in 3 sentences"):
    t = time.time() - start
    print(f"[{t:.2f}s] Token: {repr(token)}")
