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

        # 1. Get Context from State
        function_purpose = state.get("function_purpose", "")
        personality_tone = state.get("personality_tone", "")
        limitations = state.get("limitations", "")
        dynamic_info = state.get("dynamic_info", "")

        # 2. Get History
        # We need the global conversation history to have context
        global_history = state.get("messages", [])

        # 3. Get ReAct Scratchpad (intermediate steps for this turn)
        scratchpad = state.get("scratchpad", [])

        user_input = state.get("user_input", "")

        # 4. Construct Full System Prompt
        full_system_prompt = (
            f"{self.system_prompt}\n\n"
            f"--- GLOBAL INSTRUCTIONS ---\n"
            f"{function_purpose}\n\n"
            f"--- PERSONALITY ---\n"
            f"{personality_tone}\n\n"
            f"--- LIMITATIONS ---\n"
            f"{limitations}\n\n"
            f"--- DYNAMIC INFO (Date/Time/User) ---\n"
            f"{dynamic_info}\n\n"
            f"--- INSTRUCTION ON DATES ---\n"
            f"If the user uses relative dates (e.g. 'tomorrow', 'in 3 days'), you MUST calculate the exact date based on the 'Date/Time' provided in 'DYNAMIC INFO'. Do NOT ask the user for the date if you can calculate it.\n"
            f"ALWAYS respond in the same language as the user (likely Spanish)."
        )

        # 5. Combine Messages
        # System Prompt + Global History + Current Input + Scratchpad
        messages = [
            SystemMessage(content=full_system_prompt),
            *global_history, # Add history so the agent remembers previous turns
            HumanMessage(content=f"User Request: {user_input}"),
            *scratchpad # Append previous tool interactions for this specific turn
        ]

        try:
            # Removed INFO logs as requested
            response = self.model.invoke(messages)

            # CRITICAL: If no tool calls and no content, create dummy content to avoid empty response error downstream
            if not response.content and not response.tool_calls:
                response.content = "I have completed the task."

            return {"planner_response": response}

        except Exception as e:
            self.logger.error(f"InnerPlanner CRITICAL FAILURE: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"planner_response": AIMessage(content=f"Error planning: {e}")}
