import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMModel:
    def __init__(self, model_name="gpt-4o-mini"):
        # Initialize OpenAI client. Assumes OPENAI_API_KEY is in .env or system environment
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.system_prompt = "You are a helpful, concise AI voice assistant. Speak naturally in a conversational tone. Do not use markdown, emojis, or special characters."

    def generate_response(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text}
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False # Synchronous for Phase 1
        )
        return response.choices[0].message.content
