
import os
import sys
import unittest
from unittest.mock import MagicMock
import logging
import json

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sonika_langchain_bot.orchestrator.orchestrator_bot import OrchestratorBot
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

# --- MOCKS ---

class MockTool(BaseTool):
    name: str
    description: str
    def _run(self, *args, **kwargs): return "Mock success"
    async def _arun(self, *args, **kwargs): return "Mock success"

class CreateContact(MockTool):
    name: str = "create_contact"
    description: str = "Saves contact"

class SearchCars(MockTool):
    name: str = "search_cars"
    description: str = "Search cars"

class MockModel:
    def __init__(self):
        self.model = self

    def bind_tools(self, tools): return self

    def invoke(self, messages, config=None, response_format=None):
        # Inspect input to decide what to return
        prompt_content = str(messages[0].content).lower()

        # 1. ORCHESTRATOR LOGIC (Routing)
        if "master orchestrator" in prompt_content:
            # User Input está inyectado en el prompt, busquémoslo ahí
            if "hola" in prompt_content:
                return AIMessage(content=json.dumps({"next_agent": "chitchat", "reasoning": "User is greeting"}))
            if "cotizar" in prompt_content:
                return AIMessage(content=json.dumps({"next_agent": "task", "reasoning": "Booking action"}))
            # Default fallback
            return AIMessage(content=json.dumps({"next_agent": "chitchat", "reasoning": "Default"}))

        # 2. AGENTS LOGIC
        # Policy Agent
        if "policy enforcement agent" in prompt_content:
            return AIMessage(content="Please accept our policies first.")

        # Chitchat Agent
        if "friendly assistant" in prompt_content:
            return AIMessage(content="Hello! I am Alkilautos Assistant.")

        # Task Agent (InnerPlanner)
        if "you are a task executor" in prompt_content:
            # Simulate tool call logic
            if "cotizar" in prompt_content: # El prompt del InnerPlanner no tiene el user input inyectado igual
                # InnerPlanner tiene el user input en messages[-1] o en el historial
                # Pero para simplificar el mock, asumimos que si llega aquí es para cotizar
                return AIMessage(content="", tool_calls=[{"name": "search_cars", "args": {}, "id": "123"}])
            return AIMessage(content="Task completed.")

        return AIMessage(content="Generic response")

    async def ainvoke(self, messages, config=None, **kwargs):
        return self.invoke(messages, config, **kwargs)

# --- TEST ---

class TestOrchestratorBot(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("test_logger")
        self.tools = [CreateContact(), SearchCars()]
        self.alkilautos_prompt = "Alkilautos Assistant Prompt..."

    def test_simple_greeting(self):
        print("\n--- TEST 1: SALUDO SIMPLE ---")
        bot = OrchestratorBot(
            language_model=MockModel(),
            embeddings=MagicMock(),
            function_purpose=self.alkilautos_prompt,
            personality_tone="Friendly",
            limitations="None",
            dynamic_info="User: Guest",
            tools=self.tools,
            logger=self.logger
        )

        response = bot.get_response("Hola", [], [])
        print(f"Input: Hola")
        print(f"Output: {response['content']}")
        print(f"Logs: {response['logs']}")

        self.assertIn("Hello", response['content'])
        self.assertTrue(any("Routing to: CHITCHAT" in log for log in response['logs']))

    def test_complex_flow_alkilautos(self):
        print("\n--- TEST 2: FLUJO COMPLEJO (ALKILAUTOS) ---")
        bot = OrchestratorBot(
            language_model=MockModel(),
            embeddings=MagicMock(),
            function_purpose=self.alkilautos_prompt,
            personality_tone="Professional",
            limitations="None",
            dynamic_info="User: Guest",
            tools=self.tools,
            logger=self.logger
        )

        # User asks to quote (Trigger TaskAgent)
        response = bot.get_response("Quiero cotizar un carro", [], [])
        print(f"Input: Quiero cotizar un carro")
        print(f"Output (Agent Response): {response['content']}")
        print(f"Logs: {response['logs']}")

        # Verify routing
        self.assertTrue(any("Routing to: TASK" in log for log in response['logs']))
        # Verify execution logic (TaskAgent should have run)
        self.assertTrue(any("TaskAgent finished" in log for log in response['logs']))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
