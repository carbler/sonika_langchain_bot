
import os
import sys
import unittest
from unittest.mock import MagicMock
import logging

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sonika_langchain_bot.tasker.tasker_bot import TaskerBot
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

# --- MOCKS ---

class MockTool(BaseTool):
    name: str
    description: str
    def _run(self, *args, **kwargs): return "Mock success"
    async def _arun(self, *args, **kwargs): return "Mock success"

class CreateOrUpdateContactTool(MockTool):
    name: str = "create_or_update_contact"
    description: str = "Saves contact info"

class AcceptPoliciesTool(MockTool):
    name: str = "accept_policies"
    description: str = "Saves policy acceptance"

class KnowledgeDocumentsTool(MockTool):
    name: str = "search_knowledge_documents"
    description: str = "Search docs"

class CloseConversationTool(MockTool):
    name: str = "close_conversation"
    description: str = "Close chat"

class MockModel:
    def __init__(self):
        self.call_count = 0
        self.model = self # Fix: TaskerBot expects .model attribute

    def bind_tools(self, tools): return self

    def invoke(self, messages, config=None):
        self.call_count += 1
        last_msg = messages[-1].content

        # Simular comportamiento ante "hola" y Regla de Políticas

        if "Verify if the Planner has appropriately handled" in last_msg:
            # Soy el Validator
            # Debo ver si el planner quiere preguntar por politicas
            if "policy" in last_msg.lower():
                return AIMessage(content="Status: approved\nFeedback: Asking for policies is correct.")
            return AIMessage(content="Status: rejected\nFeedback: You did nothing.")

        if "Generate your response now" in last_msg:
            # Soy el Output
            return AIMessage(content="Please accept our policies.")

        # Soy el Planner
        # Iteración 1: No tengo historial. Debo pedir políticas.
        return AIMessage(content="I need to ask for policy acceptance first as per Rule 1.")

    async def ainvoke(self, messages, config=None):
        return self.invoke(messages, config)

# --- TEST ---

class TestReproInfiniteLoop(unittest.TestCase):

    def test_infinite_loop_policy(self):
        print("\n--- TEST REPRODUCCIÓN BUCLE INFINITO (POLITICAS) ---")

        tools = [
            CreateOrUpdateContactTool(),
            AcceptPoliciesTool(),
            KnowledgeDocumentsTool(),
            CloseConversationTool()
        ]

        # Usar el prompt real del usuario (resumido para el test)
        function_purpose = """
## 1. POLICY ACCEPTANCE (MANDATORY)
Before performing any action, the assistant must request explicit confirmation that the user accepts the Privacy Policy.
Rules:
- If the system has no confirmed acceptance, the assistant must stop every other action and ask for acceptance first.
"""

        bot = TaskerBot(
            language_model=MockModel(),
            embeddings=MagicMock(),
            function_purpose=function_purpose,
            personality_tone="Friendly",
            limitations="None",
            dynamic_info="User: Javier",
            tools=tools,
            logger=logging.getLogger("test_logger")
        )

        # Ejecutar
        try:
            response = bot.get_response("hola", [], [])
            print(f"\nResultado: {response['content']}")
            print("LOGS:")
            for log in response['logs']:
                print(log)
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            self.fail(f"El bot falló con: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
