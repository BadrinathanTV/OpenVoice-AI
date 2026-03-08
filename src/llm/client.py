import os
from dotenv import load_dotenv, find_dotenv
from src.agents.session import VoiceSession
import warnings
from src.core.interfaces import ILLM
from typing import Iterable

# Suppress verbose generation warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv(find_dotenv())

class LLMModel(ILLM):
    def __init__(self):
        """
        Initialize the LangGraph agent session wrapper.
        """
        self.session = VoiceSession()

    def add_human_message(self, text: str):
        self.session.add_human_message(text)
        
    def add_ai_message(self, text: str):
        self.session.update_last_ai_message(text)

    @property
    def active_agent_name(self) -> str:
        return self.session.active_agent_name

    def generate_response_stream(self, user_text: str) -> Iterable[str]:
        """
        Yields text chunks (tokens) as they arrive from the active LLM Agent.
        """
        yield from self.session.stream_response(user_text)
