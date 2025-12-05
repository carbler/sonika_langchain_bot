"""Orchestrator Node - The Brain."""

from typing import Dict, Any, List
import os
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode

class OrchestratorNode(BaseNode):
    """
    Decides which specialist agent to call based on user input and rules.
    """

    def __init__(self, model, logger=None):
        super().__init__(logger)
        self.model = model
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.system_prompt_template = self._load_prompt("orchestrator_system.txt")

    def _load_prompt(self, filename: str) -> str:
        try:
            path = os.path.join(self.base_path, "prompts", filename)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _format_tools_executed(self, tools: List[Dict[str, Any]]) -> str:
        """Summarize tools executed in this turn."""
        if not tools:
            return "None"

        summary = []
        for tool in tools:
            name = tool.get("tool_name", "unknown")
            status = tool.get("status", "unknown")
            summary.append(f"- {name} ({status})")
        return "\n".join(summary)

    def _convert_messages(self, messages: List[Any]) -> List[BaseMessage]:
        """Convert custom Message objects to LangChain BaseMessage objects."""
        converted = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                converted.append(msg)
                continue

            # Duck typing for custom Message class
            is_bot = getattr(msg, "is_bot", False)
            content = getattr(msg, "content", str(msg))

            if is_bot:
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route to the correct agent."""

        user_input = state.get("user_input", "")
        dynamic_info = state.get("dynamic_info", "")
        function_purpose = state.get("function_purpose", "")

        # Get recent tools to detect policy acceptance within the loop
        tools_executed = state.get("tools_executed", [])
        recent_activity = self._format_tools_executed(tools_executed)

        # Get Chat History and convert to objects
        raw_messages = state.get("messages", [])
        history_messages = self._convert_messages(raw_messages)

        system_prompt = self.system_prompt_template.format(
            user_input=user_input,
            dynamic_info=dynamic_info,
            function_purpose=function_purpose,
            recent_activity=recent_activity
        )

        # Construct final message list: System -> History -> Trigger
        messages_input = [
            SystemMessage(content=system_prompt),
            *history_messages,
            HumanMessage(content="Analyze history and input. Route now.")
        ]

        try:
            # Force JSON output for reliable routing
            response = self.model.invoke(
                messages_input,
                config={"temperature": 0.0},
                response_format={"type": "json_object"}
            )
            content = response.content
            decision_data = json.loads(content)

            next_agent = decision_data.get("next_agent", "chitchat")
            reasoning = decision_data.get("reasoning", "")

            log_update = self._add_log(state, f"Routing to: {next_agent.upper()} | Reason: {reasoning}")

            return {
                "next_agent": next_agent,
                "orchestrator_reasoning": reasoning,
                **log_update
            }

        except Exception as e:
            self.logger.error(f"Orchestrator failed: {e}")
            # Fallback safe
            return {"next_agent": "chitchat", "orchestrator_reasoning": "Error in routing"}
