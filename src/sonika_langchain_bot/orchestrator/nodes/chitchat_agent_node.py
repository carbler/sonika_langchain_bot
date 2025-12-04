"""Chitchat Agent Node."""

from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode

class ChitchatAgentNode(BaseNode):
    """
    Specialist: Handles greetings and small talk.
    """

    def __init__(self, model, logger=None):
        super().__init__(logger)
        self.model = model

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate friendly response."""

        system_prompt = """You are a Friendly Assistant.
        Your job is to handle greetings, farewells, and identity questions ("Who are you?").

        RULES:
        1. Be warm and professional.
        2. If the user asks for help, offer it generally.
        3. Do NOT invent business data.
        """

        # Inject personality if present
        tone = state.get("personality_tone", "")

        messages = [
            SystemMessage(content=f"{system_prompt}\n\nTone:\n{tone}"),
            HumanMessage(content=state.get("user_input", ""))
        ]

        response = self.model.invoke(messages)

        return {
            "agent_response": response.content,
            **self._add_log(state, "ChitchatAgent executed.")
        }
