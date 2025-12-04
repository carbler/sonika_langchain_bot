"""Estado compartido para el workflow Orchestrator."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage, AIMessage
from operator import add


class OrchestratorState(TypedDict):
    """Estado compartido entre todos los nodos del workflow."""

    # ENTRADA
    user_input: str

    # HISTORIAL DE CONVERSACIÓN (Acumulativo)
    messages: Annotated[List[BaseMessage], add]

    # LOGS DEL SISTEMA (Acumulativo)
    logs: Annotated[List[str], add]

    # CONTEXTO
    dynamic_info: str
    function_purpose: str
    personality_tone: str
    limitations: str

    # DECISIONES DEL ORQUESTADOR
    next_agent: Optional[str]  # "policy", "research", "task", "chitchat"
    orchestrator_reasoning: Optional[str]

    # OUTPUTS DE AGENTES
    agent_response: Optional[str]

    # ESTADO INTERNO PARA SUB-GRAFOS (ReAct Loops)
    planner_response: Optional[AIMessage] # Última decisión del planner
    scratchpad: Annotated[List[BaseMessage], add] # Historial intermedio de tools

    # TRACKING
    tools_executed: Annotated[List[Dict[str, Any]], add]
    token_usage: Dict[str, int]
