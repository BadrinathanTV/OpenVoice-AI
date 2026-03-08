import operator
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

class VoiceState(TypedDict):
    """
    Global Shared State for the Voice Swarm.
    Uses TypedDict for proper LangGraph state initialization.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    active_agent: str
