from typing import List, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from sonika_langchain_bot.planner_dynamic.state import PlannerState

class TaskNode:
    """
    Executes business action tools (e.g., booking, updates).
    """

    def __init__(self, model, tools: List[BaseTool], logger=None):
        self.model = model
        self.tools = tools
        self.logger = logger
        self.tool_map = {t.name: t for t in tools}

    async def __call__(self, state: PlannerState) -> Dict[str, Any]:
        if not self.tools:
            return {"logs": ["TaskNode: No tools available."]}

        user_input = state.get("user_input", "")
        tools_executed = state.get("tools_executed", [])

        # Use previous research/context AND executed tools to inform the task
        context_str = f"""User Input: {user_input}
Research Results: {state.get('research_results', [])}
Tools Already Executed in this Turn: {tools_executed}
"""

        # 1. Decide action
        action_plan = await self._plan_action(context_str, state)

        if not action_plan or not action_plan.get("tool_name"):
            return {"logs": ["TaskNode: No action determined."]}

        tool_name = action_plan["tool_name"]
        tool_args = action_plan.get("arguments", {})

        target_tool = self.tool_map.get(tool_name)
        if not target_tool:
             return {"logs": [f"TaskNode: Tool {tool_name} not found."]}

        # 2. Execute
        logs_to_add = []
        executed_tools_info = []
        task_output = []

        try:
            if self.logger:
                self.logger.info(f"Task executing: {tool_name} with {tool_args}")

            result = await target_tool.ainvoke(tool_args)

            logs_to_add.append(f"Task executed: {tool_name}")
            task_output.append(f"Result of {tool_name}: {result}")

            executed_tools_info.append({
                "name": tool_name,
                "input": tool_args,
                "output": str(result),
                "status": "success"
            })

        except Exception as e:
            error_msg = f"Error executing {tool_name}: {e}"
            logs_to_add.append(error_msg)
            if self.logger:
                self.logger.error(error_msg)

            executed_tools_info.append({
                "name": tool_name,
                "input": tool_args,
                "output": str(e),
                "status": "failed"
            })

        return {
            "task_results": task_output,
            "logs": logs_to_add,
            "tools_executed": executed_tools_info
        }

    async def _plan_action(self, context: str, state: PlannerState) -> Dict[str, Any]:
        system_prompt = """You are an Action Specialist.
Analyze the request and available information to trigger a specific business tool.

Available Tools:
{tool_descriptions}

Output JSON format:
{{
    "tool_name": "name_of_tool",
    "arguments": {{ "arg1": "val1", ... }}
}}

If no action is possible or info is missing, return empty JSON {{}}.
"""
        tool_descs = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{context}"),
        ])

        chain = prompt | self.model | JsonOutputParser()

        try:
            return await chain.ainvoke({
                "tool_descriptions": tool_descs,
                "context": context
            })
        except Exception as e:
            if self.logger:
                self.logger.error(f"Task planning failed: {e}")
            return {}
