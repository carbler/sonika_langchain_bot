"""
Modelos Pydantic para el workflow multi-nodo.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
import json


class Action(BaseModel):
    """Una acción a ejecutar - args como JSON string para máxima flexibilidad."""
    model_config = ConfigDict(extra="forbid")
    
    action: str = Field(..., description="Nombre exacto de la tool a ejecutar")
    args: str = Field(
        ..., 
        description='Argumentos como JSON string. Ejemplo: \'{"name": "Juan", "email": "test@test.com"}\''
    )
    
    def get_parsed_args(self) -> dict:
        """Parsea los args de JSON string a dict."""
        try:
            return json.loads(self.args)
        except json.JSONDecodeError:
            return {}


class PlannerDecision(BaseModel):
    """Output estructurado del nodo Planificador - Compatible con OpenAI."""
    model_config = ConfigDict(extra="forbid")
    
    decision: Literal["execute_actions", "request_data", "generic"] = Field(
        ...,
        description="Decisión final: ejecutar acciones o solicitar datos"
    )
    
    reasoning: List[str] = Field(
        ...,
        description="Pasos de razonamiento que llevaron a esta decisión"
    )
    
    confidence: Literal["low", "medium", "high"] = Field(
        ...,
        description="Nivel de confianza en la decisión"
    )
    
    # Campos para execute_actions
    actions: Optional[List[Action]] = Field(
        None,
        description="Acciones a ejecutar (solo si decision='execute_actions')"
    )
    
    # Campos para request_data
    field_needed: Optional[str] = Field(
        None,
        description="Campo que se necesita solicitar (solo si decision='request_data')"
    )
    
    context_for_user: Optional[str] = Field(
        None,
        description="Contexto para explicar al usuario por qué se necesita ese dato"
    )
    
    def validate_consistency(self) -> tuple[bool, Optional[str]]:
        """Valida consistencia interna del plan."""
        if self.decision == "execute_actions":
            if not self.actions or len(self.actions) == 0:
                return False, "Decision is execute_actions but no actions provided"
            if self.field_needed or self.context_for_user:
                return False, "execute_actions should not have field_needed or context_for_user"
        
        elif self.decision == "request_data":
            if not self.field_needed or not self.context_for_user:
                return False, "Decision is request_data but field_needed or context_for_user missing"
            if self.actions:
                return False, "request_data should not have actions"
        
        return True, None


class ExecutionResult(BaseModel):
    """Output del nodo Ejecutor."""
    
    status: Literal["success", "failed"] = Field(
        description="Estado general"
    )
    tools_executed: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tools ejecutadas"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Errores encontrados"
    )


class ValidationResult(BaseModel):
    """Output del nodo Verificador."""
    
    approved: bool = Field(
        description="Si es aprobado"
    )
    violations: List[str] = Field(
        default_factory=list,
        description="Violaciones detectadas"
    )
    feedback_for_planner: Optional[str] = Field(
        default=None,
        description="Feedback para corrección"
    )


class TokenUsage(BaseModel):
    """Tracking de tokens del LLM."""
    
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    
    def add(self, other: 'TokenUsage') -> 'TokenUsage':
        """Suma dos instancias."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )