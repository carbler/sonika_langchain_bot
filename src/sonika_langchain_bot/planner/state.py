"""Estado compartido para el Planner Bot."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


class ActionItem(TypedDict):
    """Representa una acción a ejecutar."""
    type: str  # "save_contact", "search_knowledge", "accept_policies", "execute_tool"
    priority: int  # 0 = máxima prioridad
    tool_name: Optional[str]  # Nombre de la tool a ejecutar
    params: Dict[str, Any]  # Parámetros para la tool


class ActionPlan(TypedDict):
    """Plan generado por el Smart Orchestrator."""
    analysis: str  # Análisis del mensaje del usuario
    detected_contact_data: Optional[Dict[str, Any]]  # Datos de contacto detectados
    actions: List[ActionItem]  # Lista de acciones a ejecutar
    requires_policy_acceptance: bool  # Si necesita aceptar políticas primero
    conversation_intent: str  # "greeting", "information_request", "task_request", "policy_confirmation"


class ExecutionResult(TypedDict):
    """Resultado de una ejecución de tool."""
    action_type: str
    tool_name: str
    params: Dict[str, Any]
    status: str  # "success", "failed", "skipped"
    output: str
    attempts: int  # Número de intentos realizados


class PlannerState(TypedDict):
    """Estado compartido entre todos los nodos del Planner Bot."""
    
    # ===== INPUT =====
    user_input: str
    messages: List[Any]  # Historial de conversación (NO acumulativo)
    
    # ===== CONTEXT (inmutable durante el turno) =====
    dynamic_info: str
    function_purpose: str
    personality_tone: str
    limitations: str
    
    # ===== PLAN DEL ORQUESTADOR =====
    action_plan: Optional[ActionPlan]
    
    # ===== RESULTADOS DE EJECUCIÓN =====
    execution_results: List[ExecutionResult]
    
    # ===== OUTPUT FINAL =====
    final_response: Optional[str]
    
    # ===== TRACKING (acumulativo) =====
    logs: Annotated[List[str], add]
    tools_executed: List[Dict[str, Any]]  # Para compatibilidad con API existente
    token_usage: Dict[str, int]
