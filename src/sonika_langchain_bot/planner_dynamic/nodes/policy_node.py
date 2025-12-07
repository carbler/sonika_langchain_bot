from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool

from sonika_langchain_bot.planner_dynamic.state import PlannerState

class PolicyNode:
    """
    Handles policy verification.
    If policies are already accepted (in dynamic_info), it skips.
    Otherwise, it triggers the policy acceptance tool logic.
    """

    def __init__(self, model, tools: List[BaseTool], logger=None):
        self.model = model
        self.tools = tools
        self.logger = logger
        # Assumption: There is a tool like 'accept_policies' or similar
        self.tool_map = {t.name: t for t in tools}

    async def __call__(self, state: PlannerState) -> Dict[str, Any]:
        dynamic_info = state.get("dynamic_info", "").lower()

        # Simple heuristic check in dynamic info
        # The Orchestrator usually injects "Policies accepted: Yes"
        if "policies accepted: yes" in dynamic_info or "politicas aceptadas: si" in dynamic_info:
            return {"logs": ["PolicyNode: Policies already accepted. Skipping."]}

        # If not accepted, find policy tool
        # We look for a tool with 'policy' or 'politica' in the name
        policy_tool = None
        for name, tool in self.tool_map.items():
            if "policy" in name.lower() or "politica" in name.lower():
                policy_tool = tool
                break

        if not policy_tool:
             return {"logs": ["PolicyNode: No policy tool found."]}

        # Execute policy tool (or just return info that it's needed)
        # In this architecture, we execute it.
        # Often policy tools just check status or send a link.

        logs = []
        executed = []

        try:
            # We assume the policy tool might take user input or empty args
            res = await policy_tool.ainvoke(state.get("user_input", ""))

            logs.append(f"PolicyNode executed: {policy_tool.name}")
            executed.append({
                "name": policy_tool.name,
                "input": {},
                "output": str(res),
                "status": "success"
            })

            # Important: If policy check fails/asks for acceptance, we might stop execution here?
            # For now, we return the result so the synthesizer can tell the user.
            return {
                "task_results": [f"Policy Check Result: {res}"], # Treat as task result
                "logs": logs,
                "tools_executed": executed
            }

        except Exception as e:
            return {"logs": [f"PolicyNode Error: {e}"]}
