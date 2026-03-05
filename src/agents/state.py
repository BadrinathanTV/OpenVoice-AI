from langchain_core.tools import tool

@tool
def switch_agent(target_agent: str, reason: str):
    """
    Use this tool to transfer the user to a different specialized agent when their request is outside your domain.
    Valid target_agent options:
    - "CustomerCare": For general inquiries, policies, returns, refunds, or general chat.
    - "Shopper": For product recommendations, searching the catalog, or adding items to a cart.
    - "OrderOps": For managing existing orders, tracking packages, modifying a cart, or checking out.
    """
    # This function body is a stub. The VoiceSession wrapper will intercept this tool call and handle the actual agent switching.
    return f"Transferring to {target_agent}..."
