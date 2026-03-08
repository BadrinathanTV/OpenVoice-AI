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

VOICE RULES:
- Never use markdown, asterisks, bullet points, or lists.
- Speak in short, clear conversational sentences.
- Keep responses concise and actionable.

TRANSFERS:
- If the user asks about store policies, returns, or complaints, call the transfer_to_customer_care function immediately.
- If the user wants product recommendations or to browse the catalog, call the transfer_to_shopper function immediately.
- You MUST use the function call to transfer. Do NOT write tool names as text.
- When transferring, do NOT say goodbye or announce the transfer. Just call the function silently.

RECEIVING A HANDOFF:
When you receive a user from another agent, DO NOT say you were transferred or mention any handoff.
Just naturally start helping with their request based on the conversation history.
For example, if they asked about an order, jump straight into checking it.
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
            "messages": [ToolMessage(content="Successfully transferred to Customer Care.", tool_call_id=tool_call_id)]
        }
    )

@tool
def transfer_to_shopper(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Shopper agent. Use this when the user needs product recommendations or catalog search."""
    return Command(
        goto="Shopper",
        update={
            "active_agent": "Shopper",
            "messages": [ToolMessage(content="Successfully transferred to Shopper.", tool_call_id=tool_call_id)]
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
