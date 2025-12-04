"""Policy Agent Node."""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode

class PolicyAgentNode(BaseNode):
    """
    Specialist: Only cares about getting policy acceptance.
    """

    def __init__(self, model, logger=None):
        super().__init__(logger)
        self.model = model

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response demanding policy acceptance."""

        system_prompt = """You are the Policy Enforcement Agent.
        Your ONLY job is to ask the user to accept the Terms and Privacy Policy.

        RULES:
        1. Ignore any other request (booking, questions, etc.).
        2. State clearly that you cannot proceed without acceptance.
        3. Provide the links if available in instructions.
        4. Be polite but firm.
        """

        # Inject instructions to get links if present
        instructions = state.get("function_purpose", "")

        messages = [
            SystemMessage(content=f"{system_prompt}\n\nGlobal Instructions:\n{instructions}"),
            HumanMessage(content=state.get("user_input", ""))
        ]

        response = self.model.invoke(messages)

        return {
            "agent_response": response.content,
            **self._add_log(state, "PolicyAgent executed.")
        }
