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
        Streams words from the active agent.
        If the active agent calls a tool, we execute it.
        If the active agent calls 'switch_agent', we intercept and instantly jump to the target agent.
        """
        self.add_human_message(user_text)
        
        while True:
            active_llm = self.agents.get(self.active_agent_name, self.agents["CustomerCare"])
            
            print(f"\n--- [VoiceSession: Routing directly to {self.active_agent_name}] ---")
            
            # 1. Invoke the LLM with the current conversation history
            response_msg = active_llm.invoke(self.messages)
            
            # The LLM generated a message or a tool call, add it to history
            self.messages.append(response_msg)
            
            # 2. Check for Tool Calls
            if hasattr(response_msg, "tool_calls") and len(response_msg.tool_calls) > 0:
                tool_call = response_msg.tool_calls[0]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                print(f"[Tool Execution] {tool_name}({tool_args})")
                
                # Intercept the exact switch_agent tool
                if tool_name == "switch_agent":
                    target = tool_args.get("target_agent", "CustomerCare")
                    reason = tool_args.get("reason", "No reason provided")
                    
                    if target in self.agents:
                        print(f"[Intercept Handoff] Switching to {target}. Reason: {reason}")
                        
                        # Add a fake tool response so the LLM history is satisfied
                        self.messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                            content=f"Successfully transferred to {target}."
                        ))
                        
                        # Apply the switch
                        self.active_agent_name = target
                        
                        # Loop back around to instantly prompt the NEW agent with the user's original request + handoff context
                        continue 
                    else:
                        self.messages.append(ToolMessage(
                            tool_call_id=tool_call["id"],
                            name=tool_name,
                            content=f"Error: Agent '{target}' not found."
                        ))
                        continue # loop back to current agent
                
                # Execute normal data tools
                elif tool_name in self.tools:
                    tool_func = self.tools[tool_name]
                    result = tool_func.invoke(tool_args)
                    self.messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=str(result)
                    ))
                    # Loop back around to the SAME agent to summarize the tool output
                    continue
                else:
                    self.messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                        content=f"Error: Unknown tool '{tool_name}'"
                    ))
                    continue
            
            # 3. No Tool Calls (or we finished processing tools) -> It's a standard text response
            final_text = response_msg.content
            
            # Yield words to simulate streaming (Since we ran invoke synchronously for stability)
            if final_text:
                words = final_text.split(" ")
                for i, word in enumerate(words):
                    yield word + (" " if i < len(words) -1 else "")
                    
            # Break out of the orchestration loop, we're done speaking to the user
            break
