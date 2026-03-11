from src.utils.chunker import SentenceChunker

chunker = SentenceChunker(min_words=1)
tokens = ["\n\n", "Hello", " World.", " I", " am", " here.", "\nAnd", " now", " for", " something", " else!"]

result = []
for t in tokens:
    for sentence in chunker.process_token(t):
         result.append(sentence)
         print(f"Yielded: '{sentence}'")

print(f"Remaining in buffer before flush: '{chunker.buffer}'")
for sentence in chunker.flush():
    result.append(sentence)
    print(f"Yielded on flush: '{sentence}'")
         
print("Total Expected Length: 3")
print(f"Total Actual Length: {len(result)}")
