"""Executor Node - Executes ONE tool at a time for ReAct loop."""

from typing import Dict, Any, Optional, Callable, List
import json
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

        # Log inicio
        self._add_log(state, f"Executing tool '{tool_name}'")

        if not tool_name:
            self.logger.error("No tool specified by planner")
            return self._create_error_output("No tool specified")

        if tool_name not in self.tools:
            self.logger.error(f"Tool {tool_name} not found")
            return self._create_error_output(f"Tool {tool_name} not found", tool_name)

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._execute_tool(tool_name, params)

                # Log éxito
                # Note: _add_log returns a dict now, but we can't easily merge it here inside async function
                # without modifying state explicitly if we were returning it.
                # Since we return a dict to update state at the end, we can include the log there if we wanted.
                # But BaseNode._add_log is designed to return the update dict.

                # Let's handle logging via the return value to be consistent with Annotated
                log_update = self._add_log(state, f"Tool '{tool_name}' completed.")

                return {
                    "executor_output": {
                        "status": "success",
                        "tools_executed": [result]
                    },
                    "tools_executed": [result],
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
                    error_update = self._create_error_output(str(e), tool_name)
                    log_update = self._add_log(state, f"Tool '{tool_name}' failed after retries: {e}")
                    return {**error_update, **log_update}

        return self._create_error_output("Max retries exceeded", tool_name)

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

    def _create_error_output(self, error: str, tool_name: str = None) -> Dict[str, Any]:
        """Create error output."""

        error_result = {
            "tool_name": tool_name or "unknown",
            "output": f"ERROR: {error}",
            "status": "failed"
        }

        return {
            "executor_output": {
                "status": "failed",
                "tools_executed": [error_result] if tool_name else [],
                "error": error
            },
            "tools_executed": [error_result] if tool_name else []
        }
