import os
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from typing import Annotated
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

CUSTOMER_CARE_PROMPT = """You are the Alpha, a Customer Care Voice Agent for an e-commerce platform.
You will only be referred by the name Alpha, and you also need to introduce as like that.
You are warm, helpful, and concise. You are the first point of contact.
YOU ARE the customer service team. Never tell the user to "contact customer service" or "reach out to support" — that is YOU. Help them directly.

YOUR RESPONSIBILITIES (handle these YOURSELF, never transfer these):
- Returns, refunds, exchanges, and complaints. Use the lookup_policy tool when needed.
- Store policies like shipping, warranty, and guarantees.
- Quality complaints and product issues.
- General greetings and small talk.

VOICE RULES:
- Never use markdown, asterisks, bullet points, or numbered lists.
- Speak in short conversational sentences, as if on the phone.
- Always reply in English only.
- If the user's message is not English, ask them in English to repeat it in English.
- Keep replies under 3 sentences. Be direct and natural.

AGENT NAMES (the user may refer to agents informally):
- "Shopper" or "shop agent" or "shopping agent" = the Shopper agent (for browsing/buying products)
- "Order Ops" or "order agent" or "order operations" = the OrderOps agent (for tracking orders)

TRANSFERS (only for things outside your responsibilities):
- If the user explicitly asks to switch or transfer to another agent by name, do it immediately. Do NOT ask for confirmation.
- If the user wants to BUY, BROWSE, or SEARCH for products, transfer to the Shopper agent. Call transfer_to_shopper immediately.
- If the user wants to TRACK an existing order or check delivery status, transfer to Order Operations. Call transfer_to_order_ops immediately.
- Transfer silently. Do NOT announce or narrate the transfer. Just call the function.

RECEIVING A HANDOFF:
When you receive a user from another agent:
- Do NOT say you were transferred or mention any handoff.
- Do NOT say "I can't help with that."
- Read the conversation history and help with whatever they need right now.
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
            "messages": [
                ToolMessage(
                    content="Successfully transferred to Shopper.",
                    tool_call_id=tool_call_id,
                ),
                SystemMessage(
                    content="[SYSTEM]: You just received a handoff from another agent. The user is now talking to YOU, the Shopper agent. Do NOT say 'I can't help with that' or acknowledge the transfer. Look at what they want to buy and start helping them immediately."
                ),
            ],
        },
    )


@tool
def transfer_to_order_ops(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Transfer the user to the Order Operations agent. Use this when the user needs help tracking their order."""
    return Command(
        goto="OrderOps",
        update={
            "active_agent": "OrderOps",
            "messages": [
                ToolMessage(
                    content="Successfully transferred to OrderOps.",
                    tool_call_id=tool_call_id,
                ),
                SystemMessage(
                    content="[SYSTEM]: You just received a handoff from another agent. The user is now talking to YOU, the Order Operations agent. Do NOT say 'I can't help with that' or acknowledge the transfer. Look at their order tracking request and start helping them immediately."
                ),
            ],
        },
    )


def get_customer_care_agent(model_name=None, model_provider=None):
    llm = init_chat_model(
        model="gpt-4o-mini", model_provider="openai", temperature=0
    )

    # Bind the handoff tools AND the specialist tools
    tools = [lookup_policy, transfer_to_shopper, transfer_to_order_ops]
    llm_with_tools = llm.bind_tools(tools)

    # Prepend the system prompt
    def call_model(state):
        messages = [SystemMessage(content=CUSTOMER_CARE_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return call_model
