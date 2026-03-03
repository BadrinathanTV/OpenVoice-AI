import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

class LLMModel:
    def __init__(self, 
                 model_name="gpt-4o-mini", 
                 model_provider="openai", 
                 api_base=None):
        """
        Initialize the LLM using LangChain. 
        To use Ollama locally, pass model_provider="ollama" and the local model_name.
        """
        self.system_prompt = "You are a helpful, concise AI voice assistant. Speak naturally in a conversational tone. Do not use markdown, emojis, or special characters."
        self.messages = [SystemMessage(content=self.system_prompt)]
        
        # init_chat_model dynamically loads the right integration based on model_provider
        self.llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            base_url=api_base,
            temperature=0.7
        )

    def add_human_message(self, text: str):
        self.messages.append(HumanMessage(content=text))
        
    def add_ai_message(self, text: str):
        if text.strip():
            self.messages.append(AIMessage(content=text))

    def generate_response_sync(self, user_text: str) -> str:
        self.add_human_message(user_text)
        response = self.llm.invoke(self.messages)
        self.add_ai_message(response.content)
        return response.content

    def generate_response_stream(self, user_text: str):
        """
        Yields text chunks (tokens) as they arrive from the LLM.
        Note: Caller must manually call `add_ai_message()` with the final spoken text afterward to support interrupts.
        """
        self.add_human_message(user_text)
        
        for chunk in self.llm.stream(self.messages):
            if chunk.content:
                yield chunk.content

