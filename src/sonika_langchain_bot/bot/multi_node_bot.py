"""Bot Multi-Nodo con LangGraph - True ReAct Pattern with Loop."""

from typing import List, Dict, Any, Optional, Callable
import logging
from langchain.schema import BaseMessage
from langchain_community.tools import BaseTool
from langgraph.graph import StateGraph, END
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

from sonika_langchain_bot.bot.state import ChatState
from sonika_langchain_bot.bot.nodes.react_agent_node import ReActAgentNode
from sonika_langchain_bot.bot.nodes.executor_node import ExecutorNode
from sonika_langchain_bot.bot.nodes.output_node import OutputNode
from sonika_langchain_bot.bot.nodes.logger_node import LoggerNode
from langchain_community.callbacks.manager import get_openai_callback


class MultiNodeBot:
    """Bot with true ReAct pattern - Agent → Tool → Observation → Agent (loop)."""
    
    def __init__(
        self,
        language_model,
        embeddings,
        function_purpose: str,
        personality_tone: str,
        limitations: str,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        max_messages: int = 100,
        max_logs: int = 20,
        max_iterations: int = 10,
        executor_max_retries: int = 2,
        on_planner_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_tool_start: Optional[Callable[[str, str], None]] = None,
        on_tool_end: Optional[Callable[[str, str], None]] = None,
        on_tool_error: Optional[Callable[[str, str], None]] = None,
        on_logs_generated: Optional[Callable[[List[str]], None]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())
        
        self.language_model = language_model
        self.embeddings = embeddings
        self.function_purpose = function_purpose
        self.personality_tone = personality_tone
        self.limitations = limitations
        self.tools = tools or []

        if mcp_servers:
            self._initialize_mcp(mcp_servers)

        self.max_messages = max_messages
        self.max_logs = max_logs
        self.max_iterations = max_iterations
        self.executor_max_retries = executor_max_retries
        
        self.on_planner_update = on_planner_update
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error
        self.on_logs_generated = on_logs_generated
        
        self.model = language_model.model
        self.graph = self._build_workflow()
        
    def _initialize_mcp(self, mcp_servers: Dict[str, Any]):
        """Initialize MCP (Model Context Protocol) connections and load available tools."""
        try:
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = asyncio.run(self.mcp_client.get_tools())
            self.tools.extend(mcp_tools)
        except Exception as e:
            self.logger.error(f"Error initializing MCP: {e}")
            self.mcp_client = None

    def _build_workflow(self) -> StateGraph:
        """Build workflow: Thinking → ReAct → Output."""
        
        
        react_agent = ReActAgentNode(
            model=self.model,
            tools=self.tools,
            max_iterations=self.max_iterations,
            on_planner_update=self.on_planner_update,
            logger=self.logger
        )
        
        executor = ExecutorNode(
            tools=self.tools,
            max_retries=self.executor_max_retries,
            on_tool_start=self.on_tool_start,
            on_tool_end=self.on_tool_end,
            on_tool_error=self.on_tool_error,
            logger=self.logger
        )
        
        output = OutputNode(
            model=self.model,
            logger=self.logger
        )
        
        logger = LoggerNode(
            on_logs_generated=self.on_logs_generated,
            logger=self.logger
        )
        
        workflow = StateGraph(ChatState)
        
        # SOLO estos nodos
        workflow.add_node("agent", react_agent)
        workflow.add_node("executor", executor)
        workflow.add_node("output", output)
        workflow.add_node("logger", logger)

        # Set entry point - start directly with agent
        workflow.set_entry_point("agent")
    
        
        # ReAct decide si ejecutar tool o terminar
        def route_after_agent(state: ChatState) -> str:
            planner_output = state.get("planner_output", {})
            decision = planner_output.get("decision", "finish")
            
            if decision == "execute_tool":
                return "executor"
            return "output"
        
        workflow.add_conditional_edges(
            "agent",
            route_after_agent,
            {
                "executor": "executor",
                "output": "output"
            }
        )
        
        # Loop: Executor → ReAct
        workflow.add_edge("executor", "agent")
        
        # Output → Logger → END
        workflow.add_edge("output", "logger")
        workflow.add_edge("logger", END)
        
        return workflow.compile()
    
    def get_response(
        self,
        user_input: str,
        messages: List[BaseMessage],
        logs: List[str],
        dynamic_info: str
    ) -> Dict[str, Any]:
        
        limited_messages = self._limit_messages(messages)
        limited_logs = self._limit_logs(logs)
        
        initial_state: ChatState = {
            "user_input": user_input,
            "messages": limited_messages,
            "logs": limited_logs,
            "dynamic_info": dynamic_info,
            "function_purpose": self.function_purpose,
            "personality_tone": self.personality_tone,
            "limitations": self.limitations,
            "planner_output": None,
            "executor_output": None,
            "validator_output": None,
            "output_node_response": None,
            "logger_output": None,
            "planning_attempts": 0,
            "execution_attempts": 0,
            "tools_executed": [],
            "react_iteration": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        with get_openai_callback() as cb:
            result = asyncio.run(self.graph.ainvoke(initial_state))
            result["token_usage"] = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            }
        
        content = result.get("output_node_response", "")
        new_logs = result.get("logger_output", [])
        tools_executed = result.get("tools_executed", [])
        token_usage = result.get("token_usage", {})
        
        return {
            "content": content,
            "logs": new_logs,
            "tools_executed": tools_executed,
            "token_usage": token_usage
        }
    
    def _limit_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Limit historical messages."""
        if len(messages) <= self.max_messages:
            return messages
        return messages[-self.max_messages:]
    
    def _limit_logs(self, logs: List[str]) -> List[str]:
        """Limit historical logs."""
        if len(logs) <= self.max_logs:
            return logs
        return logs[-self.max_logs:]