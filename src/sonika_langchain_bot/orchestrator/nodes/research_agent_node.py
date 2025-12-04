"""Research Agent - Complex Specialist."""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode
from sonika_langchain_bot.orchestrator.nodes.inner_planner import InnerPlanner
from sonika_langchain_bot.orchestrator.nodes.inner_executor import InnerExecutor
from sonika_langchain_bot.orchestrator.state import OrchestratorState

class ResearchAgentNode(BaseNode):
    """
    Sub-graph that runs a ReAct loop for Research.
    """

    def __init__(self, model, tools: List[Any], logger=None):
        super().__init__(logger)
        self.model = model
        # Filter tools
        self.tools = [t for t in tools if "search" in t.name.lower() or "knowledge" in t.name.lower()]

        self.planner = InnerPlanner(
            model,
            self.tools,
            system_prompt="You are a Researcher. Use tools to find info. If found, answer. If not, say 'Unknown'.",
            logger=logger
        )
        self.executor = InnerExecutor(self.tools, logger)
        self.subgraph = self._build_subgraph()

    def _build_subgraph(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("plan", self.planner)
        workflow.add_node("act", self.executor)
        workflow.set_entry_point("plan")

        def should_continue(state):
            # Logic to check if planner is done (no tool calls)
            resp = state.get("planner_response")
            if resp and resp.tool_calls:
                return "act"
            return END

        workflow.add_conditional_edges("plan", should_continue)
        workflow.add_edge("act", "plan")
        return workflow.compile()

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the sub-graph."""
        result = await self.subgraph.ainvoke(state)
        # Extract final answer from last message
        final_msg = result.get("planner_response")
        content = final_msg.content if final_msg else "Error in Research"

        return {
            "agent_response": content,
            **self._add_log(state, "ResearchAgent finished.")
        }
