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
        # available_tools_map e.g. {'policy_node': [tool1], 'research_node': [tool2]}
        self.available_node_types = list(available_tools_map.keys()) + ["response_node"]

    async def __call__(self, state: PlannerState) -> Dict[str, Any]:
        if self.logger:
            self.logger.info("Architect analyzing request...")

        parser = JsonOutputParser(pydantic_object=ArchitectOutput)

        system_prompt = """You are the Architect of a conversational AI.
Your goal is to design a linear execution plan (a list of steps) to handle the user's latest message, based on the provided Context and Instructions.

Available Steps (Nodes):
{node_descriptions}

Rules:
1. Always end with 'response_node' to generate the final answer.
2. If the user asks for information that might be in documents/knowledge base, include 'research_node'.
3. If the user wants to perform an action (book, update, calculate) and you have tools for it, include 'task_node'.
4. If the instructions or 'dynamic_info' suggest checking policies/permissions and they haven't been accepted/verified, include 'policy_node' at the start.
5. Be efficient. Do not add nodes if not needed.
6. If the user is just saying hello or chatting, just use ['response_node'].

Context:
Dynamic Info: {dynamic_info}
Instructions: {function_purpose}

Current User Input: {user_input}
"""

        # Construct node descriptions dynamically
        node_desc = []
        for node in self.available_node_types:
            if node == "response_node":
                node_desc.append("- response_node: Generates the text response to the user.")
            elif node == "policy_node":
                node_desc.append("- policy_node: Checks privacy policies or permissions.")
            elif node == "research_node":
                node_desc.append("- research_node: Searches internal knowledge base/documents.")
            elif node == "task_node":
                node_desc.append("- task_node: Executes specific business actions/tools.")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Design the execution plan."),
        ])

        chain = prompt | self.model | parser

        try:
            response = await chain.ainvoke({
                "node_descriptions": "\n".join(node_desc),
                "dynamic_info": state.get("dynamic_info", ""),
                "function_purpose": state.get("function_purpose", ""),
                "user_input": state.get("user_input", "")
            })

            steps = response.get("steps", [])

            # Validation: Ensure response_node is present
            if "response_node" not in steps:
                steps.append("response_node")

            if self.logger:
                self.logger.info(f"Architect Plan: {steps} (Reasoning: {response.get('reasoning')})")

            return {
                "execution_plan": steps,
                "logs": [f"Architect Plan: {steps}"]
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"Architect failed: {e}")
            # Fallback plan
            return {
                "execution_plan": ["response_node"],
                "logs": [f"Architect Error: {e}. Fallback to response only."]
            }
