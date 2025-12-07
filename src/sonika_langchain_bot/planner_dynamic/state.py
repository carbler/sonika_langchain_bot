"""State definition for the Dynamic Planner."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add
from langchain_core.messages import BaseMessage

class ExecutionResult(TypedDict):
    """Result of a tool execution."""
    tool_name: str
    params: Dict[str, Any]
    status: str  # "success", "failed"
    output: str
    timestamp: float

class PlannerState(TypedDict):
    """
    Shared state for the Dynamic Planner.
    Designed to be generic and self-contained.
    """

    # --- Inputs (from User/System) ---
    user_input: str
    messages: List[Any]  # Conversation history

    # --- Context (Static per turn) ---
    dynamic_info: str      # Context about user, policies, etc.
    function_purpose: str  # Business instructions
    personality_tone: str  # Bot persona
    limitations: str       # Restrictions

    # --- Internal Flow Control ---
    # The architect populates this list of steps (e.g., ["policy_node", "research_node", "response_node"])
    execution_plan: List[str]
    current_step_index: int

    # --- Execution Data ---
    # Stores results from research/search tools
    research_results: Annotated[List[str], add]
    # Stores results from action/task tools
    task_results: Annotated[List[str], add]

    # --- Outputs ---
    final_response: Optional[str]

    # --- Telemetry & Logs (Accumulative) ---
    logs: Annotated[List[str], add]
    # Standard format for external compatibility: {"tool": name, "input": args, "output": res}
    tools_executed: List[Dict[str, Any]]
    token_usage: Dict[str, int]
