from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from src.agents.state import switch_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SHOPPER_PROMPT = """You are the Personal Shopper Voice Agent for an e-commerce platform.
Your personality is enthusiastic, knowledgeable, and helpful.

CRITICAL RULES FOR VOICE OUTPUT:
1. NEVER use markdown, hash symbols (###), asterisks (*), bolding, bullet points, or lists. 
2. If comparing items, describe them naturally in sentences, not in bulleted lists. Limit comparisons to 2 or 3 key differences.
3. Speak in short, conversational phrasing as if talking to a friend. 
4. Keep your responses under 3 sentences whenever possible.

HANDOFF PROTOCOL:
If the user asks about returns, refunds, general complaints -> "CustomerCare".
If the user wants to checkout, track an active order, or modify a cart -> "OrderOps".
1. ALWAYS tell the user politely that you are transferring them to the appropriate department. (e.g., "It sounds like you need help tracking that package. Let me get you over to our Order Operations specialist right now.")
2. Immediately use the `switch_agent` tool. 

RECEIVING A HANDOFF:
If you just received a transferred user, acknowledge the context naturally first. (e.g., "Hi! I heard you're looking for a new smartphone. I'd love to help you find the perfect one!")
"""

# Dummy tool for the Shopper agent
from langchain_core.tools import tool
@tool
def search_catalog(query: str) -> str:
    """Search the product catalog for items matching the query."""
    return f"I found 3 items matching '{query}': A red shirt for $20, blue jeans for $40, and black shoes for $60."

def get_shopper_agent(model_name="gpt-4o-mini", model_provider="openai"):
    # We use a more capable model for the shopper to handle complex recommendations
    llm = init_chat_model(model=model_name, model_provider=model_provider, temperature=0.7)
    
    tools = [switch_agent, search_catalog]
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools
