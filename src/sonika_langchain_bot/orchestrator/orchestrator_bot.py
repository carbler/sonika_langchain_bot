"""Orchestrator Bot - The Future Architecture."""

from typing import List, Dict, Any, Optional, Callable
import logging
import asyncio
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.callbacks.manager import get_openai_callback

from sonika_langchain_bot.orchestrator.state import OrchestratorState
from sonika_langchain_bot.orchestrator.nodes.orchestrator_node import OrchestratorNode
from sonika_langchain_bot.orchestrator.nodes.policy_agent_node import PolicyAgentNode
from sonika_langchain_bot.orchestrator.nodes.chitchat_agent_node import ChitchatAgentNode
from sonika_langchain_bot.orchestrator.nodes.research_agent_node import ResearchAgentNode
from sonika_langchain_bot.orchestrator.nodes.task_agent_node import TaskAgentNode

class OrchestratorBot:
    """
    Bot based on Orchestrator-Workers pattern.
    """

    def __init__(
        self,
        language_model,
        embeddings, # kept for signature compatibility
        function_purpose: str,
        personality_tone: str,
        limitations: str,
        dynamic_info: str,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs # Catch other args
    ):
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

        self.model = language_model.model
        self.function_purpose = function_purpose
        self.personality_tone = personality_tone
        self.limitations = limitations
        self.dynamic_info = dynamic_info
        self.tools = tools or []

        if mcp_servers:
            self._initialize_mcp(mcp_servers)

        self.graph = self._build_workflow()

    def _initialize_mcp(self, mcp_servers: Dict[str, Any]):
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = asyncio.run(self.mcp_client.get_tools())
            self.tools.extend(mcp_tools)
        except Exception as e:
            self.logger.error(f"Error initializing MCP: {e}")

    def _build_workflow(self) -> StateGraph:
        orchestrator = OrchestratorNode(self.model, self.logger)

        # Specialists
        policy_agent = PolicyAgentNode(self.model, self.logger)
        chitchat_agent = ChitchatAgentNode(self.model, self.logger)
        research_agent = ResearchAgentNode(self.model, self.tools, self.logger)
        task_agent = TaskAgentNode(self.model, self.tools, self.logger)

        workflow = StateGraph(OrchestratorState)

        workflow.add_node("orchestrator", orchestrator)
        workflow.add_node("policy_agent", policy_agent)
        workflow.add_node("chitchat_agent", chitchat_agent)
        workflow.add_node("research_agent", research_agent)
        workflow.add_node("task_agent", task_agent)

        workflow.set_entry_point("orchestrator")

        def route(state):
            # Map simplified names to node names
            agent = state.get("next_agent", "chitchat")
            if agent == "policy": return "policy_agent"
            if agent == "research": return "research_agent"
            if agent == "task": return "task_agent"
            return "chitchat_agent"

        workflow.add_conditional_edges(
            "orchestrator",
            route,
            {
                "policy_agent": "policy_agent",
                "research_agent": "research_agent",
                "task_agent": "task_agent",
                "chitchat_agent": "chitchat_agent"
            }
        )

        # All agents go to END (returning final response)
        workflow.add_edge("policy_agent", END)
        workflow.add_edge("research_agent", END)
        workflow.add_edge("task_agent", END)
        workflow.add_edge("chitchat_agent", END)

        return workflow.compile()

    def get_response(
        self,
        user_input: str,
        messages: List[Any],
        logs: List[str],
    ) -> Dict[str, Any]:

        initial_state: OrchestratorState = {
            "user_input": user_input,
            "messages": messages, # Should be history
            "logs": logs,
            "dynamic_info": self.dynamic_info,
            "function_purpose": self.function_purpose,
            "personality_tone": self.personality_tone,
            "limitations": self.limitations,
            "next_agent": None,
            "orchestrator_reasoning": None,
            "agent_response": None,
            "tools_executed": [],
            "token_usage": {}
        }

        with get_openai_callback() as cb:
            result = asyncio.run(self.graph.ainvoke(initial_state))
            token_usage = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens
            }

        content = result.get("agent_response", "")
        new_logs = result.get("logs", [])
        tools_executed = result.get("tools_executed", [])

        return {
            "content": content,
            "logs": new_logs,
            "tools_executed": tools_executed,
            "token_usage": token_usage
        }
