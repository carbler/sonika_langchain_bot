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
        self.logger.info(f"INNER PLANNER STARTING. Tools: {[t.name for t in self.model.kwargs.get('tools', [])] if hasattr(self.model, 'kwargs') else 'Bound'}")

        # Build history from state
        history = state.get("messages", [])
        scratchpad = state.get("scratchpad", [])

        user_input = state.get("user_input", "")

        # Combine messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"User Request: {user_input}"),
            *scratchpad # Append previous tool interactions
        ]

        try:
            self.logger.info(f"Invoking InnerPlanner Model with {len(messages)} messages...")
            response = self.model.invoke(messages)
            self.logger.info(f"InnerPlanner Response: {response.content} | Tool Calls: {response.tool_calls}")

            # CRITICAL: If no tool calls and no content, create dummy content to avoid empty response error downstream
            if not response.content and not response.tool_calls:
                response.content = "I have completed the task."

            return {"planner_response": response}

        except Exception as e:
            self.logger.error(f"InnerPlanner CRITICAL FAILURE: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"planner_response": AIMessage(content=f"Error planning: {e}")}
