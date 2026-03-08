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

VOICE RULES:
- Never use markdown, asterisks, bullet points, or lists.
- Speak in short conversational sentences.
- Keep responses under 3 sentences when possible.

TRANSFERS:
- If the user asks about returns, refunds, or complaints, call the transfer_to_customer_care function immediately.
- If the user wants to checkout, track an order, or modify a cart, call the transfer_to_order_ops function immediately.
- You MUST use the function call to transfer. Do NOT write tool names as text.
- When transferring, do NOT say goodbye or announce the transfer. Just call the function silently.

RECEIVING A HANDOFF:
When you receive a user from another agent, DO NOT say you were transferred or mention any handoff.
Just naturally start helping based on what the user originally asked.
For example, if they said they want shoes, jump straight to asking about their style or size preferences.
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
            "messages": [ToolMessage(content="Successfully transferred to Customer Care.", tool_call_id=tool_call_id)]
        }
    )

@tool
def transfer_to_order_ops(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Order Operations agent. Use this when the user wants to clear their cart or check order status."""
    return Command(
        goto="OrderOps",
        update={
            "active_agent": "OrderOps",
            "messages": [ToolMessage(content="Successfully transferred to Order Operations.", tool_call_id=tool_call_id)]
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
