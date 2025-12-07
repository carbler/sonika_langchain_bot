"""
Dynamic Planner Bot - Generates a LangGraph execution graph on-the-fly per request.
"""

from typing import List, Dict, Any, Optional, Callable
import logging
import asyncio
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_community.callbacks.manager import get_openai_callback

from sonika_langchain_bot.planner_dynamic.state import PlannerState
from sonika_langchain_bot.planner_dynamic.nodes.architect import Architect
from sonika_langchain_bot.planner_dynamic.nodes.research_node import ResearchNode
from sonika_langchain_bot.planner_dynamic.nodes.task_node import TaskNode
from sonika_langchain_bot.planner_dynamic.nodes.policy_node import PolicyNode
from sonika_langchain_bot.planner_dynamic.nodes.response_synthesizer import ResponseSynthesizer

class NullLogger:
    def info(self, msg): pass
    def error(self, msg): pass
    def warning(self, msg): pass

class PlannerBot:
    """
    Bot that creates a dynamic execution graph for each user message.
    """

    def __init__(
        self,
        language_model,
        embeddings, # Kept for signature compatibility
        function_purpose: str,
        personality_tone: str,
        limitations: str,
        dynamic_info: str,
        tools: Optional[List[BaseTool]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        self.logger = logger or NullLogger()
        self.model = language_model.model

        # Context
        self.function_purpose = function_purpose
        self.personality_tone = personality_tone
        self.limitations = limitations
        self.dynamic_info = dynamic_info

        self.tools = tools or []

        # Categorize Tools
        self.tools_map = self._categorize_tools(self.tools)

        # Initialize Node Handlers (Shared logic)
        self.architect = Architect(self.model, self.tools_map, self.logger)

        # We pre-initialize the node logic classes, but they will be added to the graph dynamically
        self.node_handlers = {
            "research_node": ResearchNode(self.model, self.tools_map.get("research_node", []), self.logger),
            "task_node": TaskNode(self.model, self.tools_map.get("task_node", []), self.logger),
            "policy_node": PolicyNode(self.model, self.tools_map.get("policy_node", []), self.logger),
            "response_node": ResponseSynthesizer(self.model, self.logger)
        }

    def _categorize_tools(self, tools: List[BaseTool]) -> Dict[str, List[BaseTool]]:
        """Groups tools into categories for the nodes."""
        categories = {
            "policy_node": [],
            "research_node": [],
            "task_node": []
        }

        for tool in tools:
            name = tool.name.lower()
            desc = tool.description.lower()

            # Simple heuristic categorization
            if "policy" in name or "policies" in name or "politica" in name:
                categories["policy_node"].append(tool)
            elif "search" in name or "buscar" in name or "knowledge" in name or "consultar" in name:
                categories["research_node"].append(tool)
            else:
                # Default to task for everything else (booking, calculator, etc)
                categories["task_node"].append(tool)

        return categories

    def get_response(
        self,
        user_input: str,
        messages: List[Any],
        logs: List[str],
    ) -> Dict[str, Any]:
        """
        Main entry point.
        1. Ask Architect for a plan.
        2. Build graph.
        3. Execute.
        """

        # 0. Initial State
        state: PlannerState = {
            "user_input": user_input,
            "messages": messages,
            "dynamic_info": self.dynamic_info,
            "function_purpose": self.function_purpose,
            "personality_tone": self.personality_tone,
            "limitations": self.limitations,
            "execution_plan": [],
            "current_step_index": 0,
            "research_results": [],
            "task_results": [],
            "final_response": None,
            "logs": logs,
            "tools_executed": [],
            "token_usage": {}
        }

        # 1. Architect Step (Synchronous call to get the blueprint)
        # We run this outside the graph to define the graph structure
        # Or we could have a graph that runs Architect first, then conditional edge?
        # The requirement was "dynamically factorize nodes".
        # A fully dynamic structure is easier to build if we know the plan first.

        try:
            architect_result = asyncio.run(self.architect(state))
            plan_steps = architect_result.get("execution_plan", ["response_node"])
            state["logs"].extend(architect_result.get("logs", []))
        except Exception as e:
            self.logger.error(f"Architect failed: {e}")
            plan_steps = ["response_node"]

        # 2. Build Dynamic Graph
        workflow = StateGraph(PlannerState)

        # Add nodes present in the plan
        # We use a set to avoid duplicates if the architect outputs ["research", "research"] (though we asked for linear)
        # Actually, let's allow duplicates if we want, but standard StateGraph requires unique node names.
        # If the plan needs same node twice, we'd need to alias them.
        # For simplicity, we assume unique node types for now or single pass.
        # If Architect says ["research", "task"], we add "research_node" and "task_node".

        unique_nodes = set(plan_steps)
        for node_name in unique_nodes:
            if node_name in self.node_handlers:
                workflow.add_node(node_name, self._wrap_async_node(self.node_handlers[node_name]))

        # Connect Edges Sequentially
        # plan_steps = ["policy_node", "research_node", "response_node"]
        # policy -> research -> response -> END

        if not plan_steps:
             # Fallback
             workflow.add_node("response_node", self._wrap_async_node(self.node_handlers["response_node"]))
             workflow.set_entry_point("response_node")
             workflow.add_edge("response_node", END)
        else:
            workflow.set_entry_point(plan_steps[0])
            for i in range(len(plan_steps) - 1):
                workflow.add_edge(plan_steps[i], plan_steps[i+1])
            workflow.add_edge(plan_steps[-1], END)

        app = workflow.compile()

        # 3. Execute
        with get_openai_callback() as cb:
            final_state = asyncio.run(app.ainvoke(state))

            token_usage = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens
            }

        return {
            "content": final_state.get("final_response", ""),
            "logs": final_state.get("logs", []),
            "tools_executed": final_state.get("tools_executed", []),
            "token_usage": token_usage
        }

    def _wrap_async_node(self, node_handler):
        async def wrapper(state: PlannerState) -> Dict[str, Any]:
            return await node_handler(state)
        return wrapper

    def update_dynamic_info(self, dynamic_info: str):
        self.dynamic_info = dynamic_info
