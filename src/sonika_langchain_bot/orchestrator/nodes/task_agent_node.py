"""Task Agent - Complex Specialist."""

from typing import Dict, Any, List, Optional, Callable
from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage, AIMessage
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode
from sonika_langchain_bot.orchestrator.nodes.inner_planner import InnerPlanner
from sonika_langchain_bot.orchestrator.nodes.inner_executor import InnerExecutor
from sonika_langchain_bot.orchestrator.state import OrchestratorState

class TaskAgentNode(BaseNode):
    """
    Sub-graph that runs a ReAct loop for Business Tasks.
    """

    def __init__(
        self,
        model,
        tools: List[Any],
        logger=None,
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
        on_tool_error: Optional[Callable] = None
    ):
        super().__init__(logger)
        self.model = model
        # Filter tools (everything NOT search)
        self.tools = [t for t in tools if "search" not in t.name.lower()]

        self.planner = InnerPlanner(
            model,
            self.tools,
            system_prompt="You are a Task Executor. Validate params strictly. If missing, ASK user. If done, provide final answer.",
            logger=logger
        )
        self.executor = InnerExecutor(
            self.tools,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_tool_error=on_tool_error,
            logger=logger
        )
        self.subgraph = self._build_subgraph()

    def _build_subgraph(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("plan", self.planner)
        workflow.add_node("act", self.executor)
        workflow.set_entry_point("plan")

        def should_continue(state):
            resp = state.get("planner_response")
            if resp and resp.tool_calls:
                return "act"
            return END

        workflow.add_conditional_edges("plan", should_continue)
        workflow.add_edge("act", "plan")
        return workflow.compile()

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the sub-graph."""
        try:
            # Ejecutar subgrafo
            result = await self.subgraph.ainvoke(state)

            final_msg = result.get("planner_response")
            content = final_msg.content if final_msg else "Error: No planner response"

            # --- RECONSTRUCCIÓN DE HISTORIAL DE HERRAMIENTAS ---
            scratchpad = result.get("scratchpad", [])
            tools_executed_list = []
            pending_calls = {}

            for msg in scratchpad:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        call_id = call.get("id")
                        if call_id:
                            pending_calls[call_id] = {
                                "name": call.get("name"),
                                "args": call.get("args")
                            }
                elif isinstance(msg, ToolMessage):
                    call_id = msg.tool_call_id
                    call_info = pending_calls.get(call_id, {})
                    tool_name = call_info.get("name", msg.name or "UnknownTool")
                    args_val = call_info.get("args", "{}")

                    if isinstance(args_val, dict):
                        import json
                        try:
                            args_str = json.dumps(args_val)
                        except:
                            args_str = str(args_val)
                    else:
                        args_str = str(args_val)

                    tools_executed_list.append({
                        "tool_name": tool_name,
                        "args": args_str,
                        "output": msg.content,
                        "status": "success" if "Error" not in str(msg.content) else "failed"
                    })

            return {
                "agent_response": content,
                "tools_executed": tools_executed_list,
                **self._add_log(state, "TaskAgent finished.")
            }
        except Exception as e:
            self.logger.error(f"TaskAgent Subgraph ERROR: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"agent_response": f"System Error in TaskAgent: {e}"}
