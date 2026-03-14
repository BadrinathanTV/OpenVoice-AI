from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from typing import Annotated
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

ORDER_OPS_PROMPT = """You are the Order Operations Voice Agent for an e-commerce platform.
You are precise, efficient, and professional.

YOUR RESPONSIBILITIES (handle these YOURSELF, never transfer these):
- Order tracking, delivery status, shipment updates. Use check_order_status to look up orders.
- Cart and checkout issues.

VOICE RULES:
- Never use markdown, asterisks, bullet points, or numbered lists.
- Speak in short, clear conversational sentences.
- Always reply in English only.
- If the user's message is not English, ask them in English to repeat it in English.
- Keep responses under 3 sentences.

AGENT NAMES (the user may refer to agents informally):
- "Customer Care" or "care agent" or "support" = the CustomerCare agent
- "Shopper" or "shop agent" or "shopping agent" = the Shopper agent

TRANSFERS (only for things outside your responsibilities):
- If the user asks about returns, refunds, policies, or complaints, ask if they'd like to be transferred to Customer Care. If they agree, call transfer_to_customer_care immediately.
- If the user wants to browse or search for products, ask if they'd like to be transferred to the Shopper agent. If they agree, call transfer_to_shopper immediately.
- If the user explicitly asks to be switched to another agent by name, just do it.
- Transfer silently. Do NOT announce or narrate the transfer. Just call the function.

RECEIVING A HANDOFF:
When you receive a user from another agent:
- Do NOT say you were transferred or mention any handoff.
- Do NOT say "I can't help with that" or "Please hold on while I connect you."
- Read the conversation history and focus on the user's LATEST message. Help with that.
"""

@tool
def check_order_status(order_id: str = "latest") -> str:
    """Check the shipping status of an order."""
    return "Your latest order is currently out for delivery and should arrive by 8 PM tonight."

@tool
def transfer_to_customer_care(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Customer Care agent. Use this for general policies, returns, and complaints."""
    return Command(
        goto="CustomerCare",
        update={
            "active_agent": "CustomerCare",
            "messages": [
                ToolMessage(content="Successfully transferred to Customer Care.", tool_call_id=tool_call_id),
                SystemMessage(content="[SYSTEM]: You just received a handoff from another agent. The user is now talking to YOU, the Customer Care agent. Do NOT say 'I can't help with that' or acknowledge the transfer. Look at their return/complaint/policy request and start helping them immediately.")
            ]
        }
    )

@tool
def transfer_to_shopper(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Shopper agent. Use this when the user needs product recommendations or catalog search."""
    return Command(
        goto="Shopper",
        update={
            "active_agent": "Shopper",
            "messages": [
                ToolMessage(content="Successfully transferred to Shopper.", tool_call_id=tool_call_id),
                SystemMessage(content="[SYSTEM]: You just received a handoff from another agent. The user is now talking to YOU, the Shopper agent. Do NOT say 'I can't help with that' or acknowledge the transfer. Look at what they want to buy and start helping them immediately.")
            ]
        }
    )

def get_order_ops_agent(model_name="gpt-4o-mini", model_provider="openai"):
    llm = init_chat_model(model=model_name, model_provider=model_provider, temperature=0)
    
    tools = [check_order_status, transfer_to_customer_care, transfer_to_shopper]
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state):
        messages = [SystemMessage(content=ORDER_OPS_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    return call_model
