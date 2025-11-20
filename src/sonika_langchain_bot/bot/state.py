"""Estado compartido para el workflow multi-nodo de LangGraph."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain.schema import BaseMessage
from operator import add


# Modificar ChatState - agregar campos nuevos

class ChatState(TypedDict):
    """Estado compartido entre todos los nodos del workflow."""
    
    # ENTRADA
    user_input: str
    messages: List[BaseMessage]
    logs: Annotated[List[str], add]  # ✅ También debería acumularse
    dynamic_info: str
    
    # INSTRUCCIONES
    function_purpose: str
    personality_tone: str
    limitations: str
    
    # OUTPUTS DE CADA NODO
    router_decision: Optional[str]  # NUEVO
    thinking_output: Optional[str]  # NUEVO
    planner_output: Optional[Dict[str, Any]]
    executor_output: Optional[Dict[str, Any]]
    validator_output: Optional[Dict[str, Any]]
    output_node_response: Optional[str]
    logger_output: Optional[List[str]]
    
    # CONTROL DE FLUJO
    planning_attempts: int
    execution_attempts: int
    react_iteration: int
    
    # TRACKING
    tools_executed: Annotated[List[Dict[str, Any]], add]  # ✅ CAMBIO CRÍTICO
    token_usage: Dict[str, int]