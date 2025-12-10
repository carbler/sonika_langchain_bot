from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from sonika_langchain_bot.planner_dynamic.state import PlannerState

class ArchitectOutput(BaseModel):
    steps: List[str] = Field(description="List of node names to execute. Valid options: 'policy_node', 'research_node', 'task_node', 'response_node'. Order matters.")
    reasoning: str = Field(description="Brief explanation of why these steps were chosen.")

class Architect:
    """
    Decides the dynamic execution path for the current conversation turn.
    """

    def __init__(self, model, available_tools_map: Dict[str, List[Any]], logger=None):
        self.model = model
        self.logger = logger
        self.available_node_types = list(available_tools_map.keys()) + ["response_node"]

    async def __call__(self, state: PlannerState) -> Dict[str, Any]:
        parser = JsonOutputParser(pydantic_object=ArchitectOutput)

        system_prompt = """You are the Architect of a conversational AI.
Your goal is to design a linear execution plan (a list of steps) to handle the user's latest message.

Available Steps (Nodes):
{node_descriptions}

Rules:
1. Always end with 'response_node'.
2. If user asks for info -> 'research_node'.
3. If user wants action -> 'task_node'.
4. If checking policies is needed (see Instructions/Dynamic Info) and not yet done -> 'policy_node' FIRST.
5. If just chatting -> ['response_node'].

Context:
Dynamic Info: {dynamic_info}
Instructions: {function_purpose}

Current User Input: {user_input}

Respond ONLY with valid JSON matching this schema:
{{
  "steps": ["node_name_1", "node_name_2"],
  "reasoning": "explanation"
}}
"""

        node_desc = []
        for node in self.available_node_types:
            if node == "response_node":
                node_desc.append("- response_node: Generates the text response.")
            elif node == "policy_node":
                node_desc.append("- policy_node: Checks policies.")
            elif node == "research_node":
                node_desc.append("- research_node: Searches knowledge base.")
            elif node == "task_node":
                node_desc.append("- task_node: Executes business actions.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Design the plan."),
        ])

        # Attempt to bind json_object response format if model supports it (OpenAI)
        # We try-except this binding in case the model object doesn't support it standardly
        try:
            runnable_model = self.model.bind(response_format={"type": "json_object"})
        except Exception:
            runnable_model = self.model

        chain = prompt | runnable_model | parser

        try:
            response = await chain.ainvoke({
                "node_descriptions": "\n".join(node_desc),
                "dynamic_info": state.get("dynamic_info", ""),
                "function_purpose": state.get("function_purpose", ""),
                "user_input": state.get("user_input", "")
            })

            # Robust check for response type
            if not isinstance(response, dict):
                 # Try to force parse if it came back as string
                 if isinstance(response, str):
                     import json
                     response = json.loads(response)
                 else:
                     raise ValueError(f"Architect Output Parser returned non-dict: {type(response)} - {response}")

            steps = response.get("steps", [])

            if "response_node" not in steps:
                steps.append("response_node")

            if self.logger:
                self.logger.info(f"Architect Plan: {steps} (Reasoning: {response.get('reasoning')})")

            return {
                "execution_plan": steps,
                "logs": [f"Architect Plan: {steps}"]
            }

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            if self.logger:
                self.logger.error(f"Architect failed: {e}\n{error_details}")
            return {
                "execution_plan": ["response_node"],
                "logs": [f"Architect Error: {e}. Fallback to response only."]
            }
