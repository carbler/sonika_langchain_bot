"""Inner Executor - Tool Runner for Specialists."""

from typing import Dict, Any, List
import json
from langchain_core.messages import ToolMessage
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode

class InnerExecutor(BaseNode):
    """
    Generic Tool Executor.
    """

    def __init__(self, tools: List[Any], logger=None):
        super().__init__(logger)
        self.tools = {t.name: t for t in tools}

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools from planner response."""

        response = state.get("planner_response")
        if not response or not response.tool_calls:
            return {"executor_done": True}

        tool_outputs = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]

            if tool_name in self.tools:
                try:
                    # Validate empty strings if needed (Smart Executor logic copied)
                    # For brevity in this file, we assume standard execution
                    result = await self.tools[tool_name].ainvoke(args)
                    output = str(result)
                except Exception as e:
                    output = f"Error: {e}"
            else:
                output = f"Tool {tool_name} not found"

            tool_outputs.append(ToolMessage(content=output, tool_call_id=call_id))

        return {
            "scratchpad": [response] + tool_outputs, # Append (AIMessage + ToolMessages)
            "executor_done": False # Loop continues
        }
