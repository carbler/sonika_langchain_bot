from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sonika_langchain_bot.planner_dynamic.state import PlannerState

class ResponseSynthesizer:
    """
    Generates the final response to the user based on all gathered info.
    """

    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger

    async def __call__(self, state: PlannerState) -> Dict[str, Any]:
        system_prompt = """You are a helpful assistant.
Generate a response to the user based on the Execution Log and Findings below.

Inputs:
- User Input: {user_input}
- Research Findings: {research_results}
- Task Results: {task_results}
- Instructions: {function_purpose}
- Tone: {personality_tone}

Guidelines:
1. Answer the user's question directly using the Research Findings.
2. If an Action was performed, confirm the result using Task Results.
3. If Policies were checked/needed, mention that.
4. Adhere to the Tone and Instructions.
5. Do not invent information not present in the Findings.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Generate response."),
        ])

        chain = prompt | self.model | StrOutputParser()

        research = state.get("research_results", [])
        tasks = state.get("task_results", [])

        try:
            response_text = await chain.ainvoke({
                "user_input": state.get("user_input", ""),
                "research_results": "\n".join(research) if research else "None",
                "task_results": "\n".join(tasks) if tasks else "None",
                "function_purpose": state.get("function_purpose", ""),
                "personality_tone": state.get("personality_tone", "Professional"),
            })

            return {
                "final_response": response_text,
                "logs": ["Response generated successfully."]
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"Synthesizer failed: {e}")
            return {"final_response": "I apologize, I encountered an error generating the response."}
