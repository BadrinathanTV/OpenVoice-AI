from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.agents.state import SessionState
from src.agents.specialized.customer_care import get_customer_care_agent, lookup_policy
from src.agents.specialized.shopper import get_shopper_agent, search_catalog
from src.agents.specialized.order_ops import get_order_ops_agent, check_order_status

class VoiceSession:
    def __init__(self):
        """
        Manages the independent agent sessions.
        """
        print("[VoiceSession] Initializing Independent Agents...")
        self.agents = {
            "CustomerCare": get_customer_care_agent(),
            "Shopper": get_shopper_agent(),
            "OrderOps": get_order_ops_agent()
        }
        
        self.tools = {
            "lookup_policy": lookup_policy,
            "search_catalog": search_catalog,
            "check_order_status": check_order_status
        }
        
        # State
        self.active_agent_name = "CustomerCare"
        self.messages = []
        
    def add_human_message(self, text: str):
        self.messages.append(HumanMessage(content=text))
        
    def add_ai_message(self, text: str):
        self.messages.append(AIMessage(content=text))

    def stream_response(self, user_text: str):
        """
        True streaming: yields token chunks as they arrive from the LLM.
        If the LLM calls a tool, we collect the full response, execute the tool,
        then re-prompt the agent (which may produce another streamed text response).
        """
        self.add_human_message(user_text)
        
        while True:
            active_llm = self.agents.get(self.active_agent_name, self.agents["CustomerCare"])
            
            print(f"\n--- [VoiceSession: Routing directly to {self.active_agent_name}] ---")
            
            # Stream tokens from the LLM
            collected_content = ""
            collected_tool_calls = []
            full_response = None
            
            for chunk in active_llm.stream(self.messages):
                # Accumulate tool calls if present
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    for tc_chunk in chunk.tool_call_chunks:
                        # Build up tool calls from streamed chunks
                        idx = tc_chunk.get("index", 0)
                        while len(collected_tool_calls) <= idx:
                            collected_tool_calls.append({"name": "", "args": "", "id": ""})
                        if tc_chunk.get("name"):
                            collected_tool_calls[idx]["name"] += tc_chunk["name"]
                        if tc_chunk.get("args"):
                            collected_tool_calls[idx]["args"] += tc_chunk["args"]
                        if tc_chunk.get("id"):
                            collected_tool_calls[idx]["id"] += tc_chunk["id"]
                
                # Yield text content immediately as it arrives
                if chunk.content:
                    collected_content += chunk.content
                    yield chunk.content
            
            # Build the full AIMessage for history
            if collected_tool_calls and collected_tool_calls[0]["name"]:
                # Parse tool call args from JSON string
                import json
                parsed_tool_calls = []
                for tc in collected_tool_calls:
                    try:
                        args = json.loads(tc["args"]) if tc["args"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    parsed_tool_calls.append({
                        "name": tc["name"],
                        "args": args,
                        "id": tc["id"]
                    })
                
                response_msg = AIMessage(content=collected_content, tool_calls=parsed_tool_calls)
            else:
                response_msg = AIMessage(content=collected_content)
            
            self.messages.append(response_msg)
            
            # Handle tool calls
            if hasattr(response_msg, "tool_calls") and response_msg.tool_calls:
                tool_call = response_msg.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"[Tool Execution] {tool_name}({tool_args})")
                
                if tool_name == "switch_agent":
                    target = tool_args.get("target_agent", "CustomerCare")
                    reason = tool_args.get("reason", "No reason provided")
                    
                    if target in self.agents:
                        print(f"[Intercept Handoff] Switching to {target}. Reason: {reason}")
                        self.messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                            content=f"Successfully transferred to {target}."
                        ))
                        self.active_agent_name = target
                        continue
                    else:
                        self.messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                            content=f"Error: Agent '{target}' not found."
                        ))
                        continue
                
                elif tool_name in self.tools:
                    tool_func = self.tools[tool_name]
                    result = tool_func.invoke(tool_args)
                    self.messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=str(result)
                    ))
                    continue
                else:
                    self.messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=f"Error: Unknown tool '{tool_name}'"
                    ))
                    continue
            
            # No tool calls — we streamed text, print the full response and break
            print(f"\n[{self.active_agent_name} Response]: {collected_content}")
            break
