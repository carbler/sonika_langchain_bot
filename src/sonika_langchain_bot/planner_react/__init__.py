"""
Planner Bot - Arquitectura inteligente de bot conversacional.

Esta arquitectura implementa un flujo lineal sin ciclos:

1. SmartOrchestrator: Analiza el mensaje y genera un plan de acciones
2. ActionExecutor: Ejecuta las acciones con retry automático (3 intentos)
3. ConversationSynthesizer: Genera una respuesta coherente

Características:
- Multi-empresa: Cada empresa configura sus tools e instrucciones
- Detección automática de datos de contacto
- Manejo de políticas de privacidad
- Búsqueda en base de conocimiento
- Retry automático de tools fallidas
- Una sola respuesta coherente por turno
- Sin ciclos infinitos

Uso básico:
    from sonika_langchain_bot.planner import PlannerBot
    
    bot = PlannerBot(
        language_model=model,
        embeddings=embeddings,
        function_purpose="Instrucciones de negocio...",
        personality_tone="Profesional y amigable",
        limitations="No puede hacer X, Y, Z",
        dynamic_info="Contexto actual del usuario...",
        tools=[tool1, tool2, tool3]
    )
    
    response = bot.get_response(
        user_input="Hola, me llamo Juan",
        messages=[],
        logs=[]
    )
    
    print(response["content"])  # Respuesta del bot
"""

from sonika_langchain_bot.planner_react.planner_bot import PlannerBot
from sonika_langchain_bot.planner_react.state import (
    PlannerState,
    ActionPlan,
    ActionItem,
    ExecutionResult
)

__all__ = [
    "PlannerBot",
    "PlannerState",
    "ActionPlan",
    "ActionItem",
    "ExecutionResult"
]
