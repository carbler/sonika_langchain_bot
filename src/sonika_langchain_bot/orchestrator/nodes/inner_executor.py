"""Inner Executor - Tool Runner for Specialists."""

from typing import Dict, Any, List, Optional, Callable
import json
from langchain_core.messages import ToolMessage
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode

class InnerExecutor(BaseNode):
    """
    Generic Tool Executor.
    """

    def __init__(
        self,
        tools: List[Any],
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
        on_tool_error: Optional[Callable] = None,
        logger=None
    ):
        super().__init__(logger)
        self.tools = {t.name: t for t in tools}
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tools from planner response."""

        response = state.get("planner_response")
        if not response or not response.tool_calls:
            return {"executor_done": True}

        tool_outputs = []
        tools_executed_list = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]

            # Serialize args for logging and reporting
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)
            status = "failed"
            output = ""

            if tool_name in self.tools:
                try:
                    # Notify Start
                    if self.on_tool_start:
                        try:
                            self.on_tool_start(tool_name, args_str)
                        except Exception:
                            pass

                    # Execute
                    result = await self.tools[tool_name].ainvoke(args)
                    output = str(result)
                    status = "success"

                    # Notify End
                    if self.on_tool_end:
                        try:
                            self.on_tool_end(tool_name, output)
                        except Exception:
                            pass

                except Exception as e:
                    output = f"Error: {e}"
                    status = "failed"
                    if self.on_tool_error:
                        try:
                            self.on_tool_error(tool_name, str(e))
                        except Exception:
                            pass
            else:
                output = f"Tool {tool_name} not found"
                status = "failed"
                if self.on_tool_error:
                    try:
                        self.on_tool_error(tool_name, "Tool not found")
                    except Exception:
                        pass

            tool_outputs.append(ToolMessage(content=output, tool_call_id=call_id))

            tools_executed_list.append({
                "tool_name": tool_name,
                "args": args_str,
                "output": output,
                "status": status
            })

        return {
            "scratchpad": [response] + tool_outputs, # Append (AIMessage + ToolMessages)
            "tools_executed": tools_executed_list,   # Append to global state
            "executor_done": False # Loop continues
        }
