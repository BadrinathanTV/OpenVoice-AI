from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from typing import Annotated
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

CUSTOMER_CARE_PROMPT = """You are the Customer Care Voice Agent for an e-commerce platform.
You are warm, helpful, and concise. You are the first point of contact.

VOICE RULES:
- Never use markdown, asterisks, bullet points, or lists.
- Speak in short conversational sentences, as if on the phone.

TRANSFERS:
- If the user wants to buy or browse products, call the transfer_to_shopper function immediately.
- If the user wants to track or check on an order, call the transfer_to_order_ops function immediately.
- You MUST use the function call to transfer. Do NOT write tool names as text.
- When transferring, do NOT say goodbye or announce the transfer. Just call the function silently.

RECEIVING A HANDOFF:
When you receive a user from another agent, DO NOT say you were transferred or mention any handoff.
Just naturally start helping with their request based on the conversation history.
For example, if they asked about returns, jump straight into helping them with returns.
"""

@tool
def lookup_policy(topic: str) -> str:
    """Lookup the store policy for a given topic (e.g., 'returns', 'shipping', 'refunds')."""
    if "return" in topic.lower():
        return "We offer a 30-day return policy for all unused items."
    elif "shipping" in topic.lower():
        return "Standard shipping takes 3-5 business days. Expedited takes 1-2 days."
    return "I couldn't find a specific policy for that, but I'm happy to help you figure it out."

@tool
def transfer_to_shopper(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Shopper agent. Use this when the user needs to find a product, get recommendations, or search the catalog."""
    return Command(
        goto="Shopper",
        update={
            "active_agent": "Shopper",
            "messages": [ToolMessage(content="Successfully transferred to Shopper.", tool_call_id=tool_call_id)]
        }
    )

@tool
def transfer_to_order_ops(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Order Operations agent. Use this when the user needs help tracking their order."""
    return Command(
        goto="OrderOps",
        update={
            "active_agent": "OrderOps",
            "messages": [ToolMessage(content="Successfully transferred to OrderOps.", tool_call_id=tool_call_id)]
        }
    )

def get_customer_care_agent(model_name="gpt-4o-mini", model_provider="openai"):
    llm = init_chat_model(model=model_name, model_provider=model_provider, temperature=0)
    
    # Bind the handoff tools AND the specialist tools
    tools = [lookup_policy, transfer_to_shopper, transfer_to_order_ops]
    llm_with_tools = llm.bind_tools(tools)
    
    # Prepend the system prompt
    def call_model(state):
        messages = [SystemMessage(content=CUSTOMER_CARE_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    return call_model 
