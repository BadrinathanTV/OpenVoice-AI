from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from src.agents.state import switch_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ORDER_OPS_PROMPT = """You are the Order Operations Voice Agent for an e-commerce platform.
Your personality is precise, efficient, and professional.

CRITICAL RULES FOR VOICE OUTPUT:
1. NEVER use markdown, hash symbols (###), asterisks (*), bolding, bullet points, or lists.
2. Speak in short, clear, conversational sentences as if talking on the phone. Do not read long blocks of text or raw JSON variables.
3. Keep your responses highly actionable and concise.

HANDOFF PROTOCOL:
If the user asks about general store policies, returns rules, or complaints -> "CustomerCare".
If the user wants product recommendations or wants to browse the catalog -> "Shopper".
1. ALWAYS tell the user politely that you are transferring them to the correct department. (e.g., "I can definitely get someone to help you find a new pair of shoes. Let me connect you with our Personal Shopper now.")
2. Immediately use the `switch_agent` tool.

RECEIVING A HANDOFF:
If you just received a transferred user, acknowledge what they need naturally. (e.g., "Hello! I understand you need an update on your recent order. Let me pull that up for you.")
"""

# Dummy tool for the Order Ops agent
from langchain_core.tools import tool
@tool
def check_order_status(order_id: str = "latest") -> str:
    """Check the shipping status of an order."""
    return "Your latest order is currently out for delivery and should arrive by 8 PM tonight."

def get_order_ops_agent(model_name="openai/gpt-oss-120b", model_provider="groq"):
    llm = init_chat_model(model=model_name, model_provider=model_provider, temperature=0.1) # low temp for precise operations
    
    tools = [switch_agent, check_order_status]
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools
