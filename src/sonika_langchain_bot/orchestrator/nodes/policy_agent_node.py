"""Policy Agent Node."""

from typing import Dict, Any, List, Optional, Callable
from langgraph.graph import StateGraph, END
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode
from sonika_langchain_bot.orchestrator.nodes.inner_planner import InnerPlanner
from sonika_langchain_bot.orchestrator.nodes.inner_executor import InnerExecutor
from sonika_langchain_bot.orchestrator.state import OrchestratorState

class PolicyAgentNode(BaseNode):
    """
    Specialist: Handles Policy Acceptance using ReAct loop.
    """

    def __init__(
        self,
        model,
        tools: List[Any],
        logger=None,
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
        on_tool_error: Optional[Callable] = None
    ):
        super().__init__(logger)
        self.model = model
        # Give access to all tools so it can find the acceptance tool
        self.tools = tools

        # STRICT FILTERING: Only allow Policy Acceptance tools
        self.tools = [t for t in tools if "policy" in t.name.lower() or "policies" in t.name.lower()]

        if not self.tools:
            self.logger.error("WARNING: No AcceptPoliciesTool found for PolicyAgent.")

        self.planner = InnerPlanner(
            model,
            self.tools,
            system_prompt=(
                "You are the Policy Enforcement Agent.\n"
                "Your ONLY goal is to register the user's acceptance of the Terms and Privacy Policy.\n\n"
                "RULES:\n"
                "1. CHECK DYNAMIC INFO FIRST: If it says 'Policies accepted: Yes', say 'Policies are already accepted. How can I help?' and STOP.\n"
                "2. If the user says 'yes', 'ok', 'agree', 'claro', or confirms acceptance -> IMMEDIATELY USE THE TOOL `AcceptPoliciesTool`.\n"
                "3. If the user asks a question, DO NOT ANSWER IT. Instead, say: 'I cannot answer until you accept the policies.'\n"
                "4. Use the GLOBAL INSTRUCTIONS below to find the correct policy links."
            ),
            logger=logger
        )
        self.executor = InnerExecutor(
            self.tools,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_tool_error=on_tool_error,
            logger=logger
        )
        self.subgraph = self._build_subgraph()

    def _build_subgraph(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("plan", self.planner)
        workflow.add_node("act", self.executor)
        workflow.set_entry_point("plan")

        def should_continue(state):
            resp = state.get("planner_response")
            if resp and resp.tool_calls:
                return "act"
            return END

        workflow.add_conditional_edges("plan", should_continue)
        workflow.add_edge("act", "plan")
        return workflow.compile()

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the sub-graph."""
        initial_tools_count = len(state.get("tools_executed", []))

        result = await self.subgraph.ainvoke(state)

        final_msg = result.get("planner_response")
        content = final_msg.content if final_msg else "Error in Policy Agent"

        # Calculate new tools executed
        final_tools = result.get("tools_executed", [])
        new_tools = final_tools[initial_tools_count:]

        return {
            "agent_response": content,
            "tools_executed": new_tools,
            **self._add_log(state, "PolicyAgent finished.")
        }
