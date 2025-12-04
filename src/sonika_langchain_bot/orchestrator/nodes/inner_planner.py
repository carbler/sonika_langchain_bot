"""Inner Planner - ReAct Brain for Specialists."""

from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode

class InnerPlanner(BaseNode):
    """
    Generic ReAct Planner used by specialist agents.
    """

    def __init__(self, model, tools: List[Any], system_prompt: str, logger=None):
        super().__init__(logger)
        self.model = model.bind_tools(tools) if tools else model
        self.system_prompt = system_prompt

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan next step."""

        messages = [
            SystemMessage(content=self.system_prompt),
            *state.get("messages", []) # Full history
        ]

        # Add scratchpad if available (tools history for current turn)
        scratchpad = state.get("scratchpad", [])
        if scratchpad:
            messages.extend(scratchpad)

        try:
            response = self.model.invoke(messages)
            return {"planner_response": response}
        except Exception as e:
            self.logger.error(f"InnerPlanner failed: {e}")
            return {"error": str(e)}
