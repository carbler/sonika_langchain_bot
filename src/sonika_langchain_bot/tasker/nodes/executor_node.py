"""Executor Node - Executes ONE tool at a time for ReAct loop."""

from typing import Dict, Any, Optional, Callable, List
import json
from langchain_core.messages import ToolMessage
from sonika_langchain_bot.tasker.nodes.base_node import BaseNode


class ExecutorNode(BaseNode):
    """Executes a single tool and returns observation to agent."""

    def __init__(
        self,
        tools: List[Any],
        max_retries: int = 2,
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
        on_tool_error: Optional[Callable] = None,
        logger=None
    ):
        super().__init__(logger)
        self.tools = {tool.name: tool for tool in tools}
        self.max_retries = max_retries
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the single tool specified by planner."""

        planner_output = state.get("planner_output", {})
        tool_name = planner_output.get("tool")
        params = planner_output.get("params", {})
        tool_call_id = planner_output.get("tool_call_id", "unknown_id") # Obtener ID

        # Log inicio
        self._add_log(state, f"Executing tool '{tool_name}'")

        if not tool_name:
            self.logger.error("No tool specified by planner")
            return self._create_error_output("No tool specified", tool_name, tool_call_id)

        if tool_name not in self.tools:
            self.logger.error(f"Tool {tool_name} not found")
            return self._create_error_output(f"Tool {tool_name} not found", tool_name, tool_call_id)

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._execute_tool(tool_name, params)
                output_str = result.get('output', '')

                # Log éxito
                log_update = self._add_log(state, f"Tool '{tool_name}' completed.")

                # CREAR TOOL MESSAGE NATIVO
                tool_msg = ToolMessage(
                    content=str(output_str),
                    tool_call_id=tool_call_id,
                    name=tool_name
                )

                return {
                    "executor_output": {
                        "status": "success",
                        "tools_executed": [result]
                    },
                    "tools_executed": [result],
                    "messages": [tool_msg], # <--- AGREGAR AL HISTORIAL
                    **log_update
                }

            except Exception as e:
                self.logger.error(f"Tool execution failed (attempt {attempt + 1}): {e}")

                if self.on_tool_error:
                    try:
                        self.on_tool_error(tool_name, str(e))
                    except:
                        pass

                if attempt >= self.max_retries:
                    error_update = self._create_error_output(str(e), tool_name, tool_call_id)
                    log_update = self._add_log(state, f"Tool '{tool_name}' failed after retries: {e}")
                    return {**error_update, **log_update}

        return self._create_error_output("Max retries exceeded", tool_name, tool_call_id)

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool."""
        tool = self.tools[tool_name]

        if self.on_tool_start:
            try:
                self.on_tool_start(tool_name, json.dumps(params))
            except:
                pass

        # Some tools are async, some are sync. LangChain tools usually support ainvoke.
        output = await tool.ainvoke(params)

        if self.on_tool_end:
            try:
                self.on_tool_end(tool_name, str(output))
            except:
                pass

        return {
            "tool_name": tool_name,
            "args": json.dumps(params),
            "output": str(output),
            "status": "success"
        }

    def _create_error_output(self, error: str, tool_name: str = None, tool_call_id: str = "unknown") -> Dict[str, Any]:
        """Create error output and corresponding tool message."""

        error_result = {
            "tool_name": tool_name or "unknown",
            "output": f"ERROR: {error}",
            "status": "failed"
        }

        tool_msg = ToolMessage(
            content=f"ERROR: {error}",
            tool_call_id=tool_call_id,
            name=tool_name or "unknown",
            status="error"
        )

        return {
            "executor_output": {
                "status": "failed",
                "tools_executed": [error_result] if tool_name else [],
                "error": error
            },
            "tools_executed": [error_result] if tool_name else [],
            "messages": [tool_msg] # Retornar el error como mensaje también
        }
