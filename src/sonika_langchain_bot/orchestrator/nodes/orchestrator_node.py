"""Orchestrator Node - The Brain."""

from typing import Dict, Any
import os
import json
from langchain_core.messages import SystemMessage, HumanMessage
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

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route to the correct agent."""

        user_input = state.get("user_input", "")
        dynamic_info = state.get("dynamic_info", "")
        function_purpose = state.get("function_purpose", "")

        system_prompt = self.system_prompt_template.format(
            user_input=user_input,
            dynamic_info=dynamic_info,
            function_purpose=function_purpose
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Analyze and route now.")
        ]

        try:
            # Force JSON output for reliable routing
            response = self.model.invoke(
                messages,
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
