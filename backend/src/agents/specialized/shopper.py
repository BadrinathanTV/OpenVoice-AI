from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from typing import Annotated
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SHOPPER_PROMPT = """You are the Personal Shopper Voice Agent for an e-commerce platform.
You are enthusiastic, knowledgeable, and helpful.

YOUR RESPONSIBILITIES (handle these YOURSELF, never transfer these):
- Product search, recommendations, and catalog browsing. Use search_catalog to find items.
- Helping users pick the right size, color, or style.

VOICE RULES:
- Never use markdown, asterisks, bullet points, or numbered lists.
- Speak in short conversational sentences.
- Keep responses under 3 sentences.

AGENT NAMES (the user may refer to agents informally):
- "Customer Care" or "care agent" or "support" = the CustomerCare agent
- "Order Ops" or "order agent" or "order operations" = the OrderOps agent

TRANSFERS (only for things outside your responsibilities):
- If the user asks about returns, refunds, policies, or complaints, ask if they'd like to be transferred to Customer Care. If they agree, call transfer_to_customer_care immediately.
- If the user wants to track an order or check delivery status, ask if they'd like to be transferred to Order Operations. If they agree, call transfer_to_order_ops immediately.
- If the user explicitly asks to be switched to another agent by name, just do it.
- Transfer silently. Do NOT announce or narrate the transfer. Just call the function.

RECEIVING A HANDOFF:
When you receive a user from another agent:
- Do NOT say you were transferred or mention any handoff.
- Do NOT say "I can't help with that" or "Please hold on while I connect you."
- Read the conversation history and focus on the user's LATEST message. Help with that.
- If they wanted to shop or buy something, start helping them shop immediately.
"""

@tool
def search_catalog(query: str) -> str:
    """Search the product catalog for items matching the query."""
    return f"I found 3 items matching '{query}': A red shirt for $20, blue jeans for $40, and black shoes for $60."

@tool
def transfer_to_customer_care(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Customer Care agent. Use this when the user asks about returns, refunds, or general policies."""
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
def transfer_to_order_ops(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Order Operations agent. Use this when the user wants to clear their cart or check order status."""
    return Command(
        goto="OrderOps",
        update={
            "active_agent": "OrderOps",
            "messages": [
                ToolMessage(content="Successfully transferred to Order Operations.", tool_call_id=tool_call_id),
                SystemMessage(content="[SYSTEM]: You just received a handoff from another agent. The user is now talking to YOU, the Order Operations agent. Do NOT say 'I can't help with that' or acknowledge the transfer. Look at their order tracking request and start helping them immediately.")
            ]
        }
    )

def get_shopper_agent(model_name="gpt-4o-mini", model_provider="openai"):
    llm = init_chat_model(model=model_name, model_provider=model_provider, temperature=0.3)
    
    tools = [search_catalog, transfer_to_customer_care, transfer_to_order_ops]
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state):
        messages = [SystemMessage(content=SHOPPER_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
        
    return call_model
