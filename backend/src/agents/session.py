from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os
import uuid
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

from src.agents.state import VoiceState
from src.agents.specialized.customer_care import get_customer_care_agent, lookup_policy, transfer_to_order_ops, transfer_to_shopper
from src.agents.specialized.shopper import get_shopper_agent, search_catalog, transfer_to_customer_care as shopper_to_cc, transfer_to_order_ops as shopper_to_oo
from src.agents.specialized.order_ops import get_order_ops_agent, check_order_status, transfer_to_customer_care as oo_to_cc, transfer_to_shopper as oo_to_shopper

class VoiceSession:
    def __init__(self, db_url=None):
        """
        Manages the independent agent sessions using LangGraph.
        """
        print("[VoiceSession] Initializing LangGraph Swarm...")
        
        # Tools grouped for the ToolNode
        self.tools = [
            lookup_policy, search_catalog, check_order_status,
            transfer_to_order_ops, transfer_to_shopper,
            shopper_to_cc, shopper_to_oo,
            oo_to_cc, oo_to_shopper
        ]
        
        # Initialize MongoDB Connection
        # Fallback to a default localhost DB if not specified in ENV
        self.db_url = db_url or os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
        self.mongo_client = MongoClient(self.db_url)
        self.checkpointer = MongoDBSaver(self.mongo_client)
        
        # Build the graph with a FRESH thread_id every time the session starts,
        # so we don't replay stale conversation history from previous runs.
        self.config = {"configurable": {"thread_id": f"session_{uuid.uuid4().hex[:8]}"}}
        self.graph = self._build_graph()
        self._cached_agent = "CustomerCare"  # Cache to avoid DB calls per token
        print(f"[VoiceSession] Thread: {self.config['configurable']['thread_id']}")

    def _build_graph(self):
        builder = StateGraph(VoiceState)
        
        # Nodes
        builder.add_node("CustomerCare", get_customer_care_agent())
        builder.add_node("Shopper", get_shopper_agent())
        builder.add_node("OrderOps", get_order_ops_agent())
        builder.add_node("tools", ToolNode(self.tools))
        
        # Route from START to the currently active agent
        def route_start(state):
            return state.get("active_agent", "CustomerCare")
            
        builder.add_conditional_edges(START, route_start)
        
        # After an agent responds, check if it wants to call tools or we're done
        def route_node(state):
            messages = state.get("messages", [])
            if not messages:
                return END
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_names = [tc['name'] for tc in last_message.tool_calls]
                print(f"  → Tool calls: {tool_names}")
                return "tools"
            return END
            
        builder.add_conditional_edges("CustomerCare", route_node)
        builder.add_conditional_edges("Shopper", route_node)
        builder.add_conditional_edges("OrderOps", route_node)
        
        # After tools finish, route to the (possibly new) active agent
        def route_tools(state):
            active = state.get("active_agent", "CustomerCare")
            print(f"  → Routed to: {active}")
            return active
            
        builder.add_conditional_edges("tools", route_tools)
        
        return builder.compile(checkpointer=self.checkpointer)

    @property
    def active_agent_name(self) -> str:
        """Returns the cached agent name (fast, no DB call)."""
        return self._cached_agent

    def _refresh_agent_name(self) -> str:
        """Reads the actual agent name from the DB and updates the cache."""
        state = self.graph.get_state(self.config)
        if state and state.values:
            self._cached_agent = state.values.get("active_agent", "CustomerCare")
        return self._cached_agent

    def add_human_message(self, text: str):
        self.graph.update_state(self.config, {"messages": [HumanMessage(content=text)]})
        
    def add_ai_message(self, text: str):
        self.graph.update_state(self.config, {"messages": [AIMessage(content=text)]})

    def update_last_ai_message(self, text: str):
        """
        Replaces the most recent AIMessage in the graph state with the actual spoken text.
        This ensures the LLM knows exactly where it was interrupted.
        """
        state = self.graph.get_state(self.config)
        if not state or not state.values.get("messages"):
            self.add_ai_message(text)
            return
            
        messages = state.values["messages"]
        last_msg = messages[-1]
        
        if isinstance(last_msg, AIMessage) and not getattr(last_msg, "tool_calls", None):
            updated_msg = AIMessage(content=text, id=last_msg.id)
            self.graph.update_state(self.config, {"messages": [updated_msg]})
        else:
            self.add_ai_message(text)

    def stream_response(self, user_text: str):
        """
        Streams token chunks as they arrive from the LangGraph execution.
        """
        
        agent_before = self._cached_agent
        print(f"\n[Agent: {agent_before}] Processing: \"{user_text}\"")
        
        # We only pass the new user message. The checkpointer retains the rest.
        inputs = {"messages": [HumanMessage(content=user_text)]}
        
        yield_buffer = ""
        agent_nodes = {"CustomerCare", "Shopper", "OrderOps"}
        
        for msg, metadata in self.graph.stream(
            inputs, 
            config=self.config,
            stream_mode="messages"
        ):
            node_name = metadata.get("langgraph_node")
            
            # Update cached agent name when we see tokens from a different agent node
            if node_name in agent_nodes and node_name != self._cached_agent:
                print(f"  ✓ Switched: {self._cached_agent} → {node_name}")
                self._cached_agent = node_name
                
            # Yield text content from AI messages for TTS
            if isinstance(msg, AIMessage) and msg.content:
                yield msg.content
                yield_buffer += msg.content
                
            # Log tool call chunks
            if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
                tc_chunk = msg.tool_call_chunks[0]
                if tc_chunk.get("name"):
                    print(f"  [Calling: {tc_chunk['name']}]")

        # Sync cache with DB at the end to stay consistent
        self._refresh_agent_name()
        
        if yield_buffer:
            print(f"[{self._cached_agent}]: {yield_buffer.strip()}")
