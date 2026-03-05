import os
from dotenv import load_dotenv, find_dotenv
from src.agents.session import VoiceSession
import warnings

# Suppress verbose generation warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv(find_dotenv())

class LLMModel:
    def __init__(self, 
                 model_name=None, 
                 model_provider=None, 
                 api_base=None):
        """
        Initialize the Independent Modular Agents wrapper.
        """
        self.session = VoiceSession()

    def add_human_message(self, text: str):
        self.session.add_human_message(text)
        
    def add_ai_message(self, text: str):
        # The VoiceSession already appends the full generated AIMessage to history.
        # So we replace the last text message with the actual spoken text, so the LLM
        # knows if it was cut off mid-sentence by an interrupt.
        from langchain_core.messages import AIMessage
        if self.session.messages and isinstance(self.session.messages[-1], AIMessage):
            # Don't overwrite it if it was a tool call (though it shouldn't be here)
            if not getattr(self.session.messages[-1], "tool_calls", None):
                self.session.messages[-1] = AIMessage(content=text)
                return
        self.session.add_ai_message(text)

    def generate_response_sync(self, user_text: str) -> str:
        """
        We only implement the streaming function for TTS. Sync isn't easily supported 
        with the custom generator loop unless we accumulate it.
        """
        response = ""
        for chunk in self.session.stream_response(user_text):
            response += chunk + " "
        return response

    def generate_response_stream(self, user_text: str):
        """
        Yields text chunks (tokens) as they arrive from the active LLM Agent.
        """
        yield from self.session.stream_response(user_text)

