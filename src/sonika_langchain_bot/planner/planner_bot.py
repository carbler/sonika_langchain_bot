"""Planner Bot - Punto de entrada principal."""

from typing import List, Dict, Any, Optional, Callable
import logging
import asyncio
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_community.callbacks.manager import get_openai_callback

from sonika_langchain_bot.planner.state import PlannerState
from sonika_langchain_bot.planner.nodes.smart_orchestrator import SmartOrchestrator
from sonika_langchain_bot.planner.nodes.action_executor import ActionExecutor
from sonika_langchain_bot.planner.nodes.conversation_synthesizer import ConversationSynthesizer


class PlannerBot:
    """
    Bot conversacional basado en arquitectura Planner.
    
    Flujo:
    1. SmartOrchestrator → Analiza mensaje y genera plan de acciones
    2. ActionExecutor → Ejecuta acciones con retry (3 intentos)
    3. ConversationSynthesizer → Genera respuesta coherente
    
    Características:
    - Multi-empresa: Cada empresa configura sus tools e instrucciones
    - Detección automática de datos de contacto
    - Manejo de políticas de privacidad
    - Búsqueda en base de conocimiento
    - Retry automático de tools fallidas
    - Una sola respuesta coherente por turno
    - Sin ciclos infinitos (flujo lineal)
    """

    def __init__(
        self,
        language_model,
        embeddings,
        function_purpose: str,
        personality_tone: str,
        limitations: str,
        dynamic_info: str,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        on_tool_start: Optional[Callable[[str, str], None]] = None,
        on_tool_end: Optional[Callable[[str, str], None]] = None,
        on_tool_error: Optional[Callable[[str, str], None]] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Inicializa el Planner Bot.
        
        Args:
            language_model: Modelo de lenguaje (debe tener atributo .model)
            embeddings: Modelo de embeddings (para compatibilidad)
            function_purpose: Instrucciones de negocio para el bot
            personality_tone: Tono y personalidad del bot
            limitations: Limitaciones del bot
            dynamic_info: Información dinámica del contexto (usuario, políticas, etc.)
            tools: Lista de tools disponibles
            mcp_servers: Configuración de servidores MCP (opcional)
            on_tool_start: Callback cuando inicia una tool
            on_tool_end: Callback cuando termina una tool exitosamente
            on_tool_error: Callback cuando falla una tool
            logger: Logger personalizado
        """
        # Logger
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())
        
        # Modelo
        self.model = language_model.model
        self.embeddings = embeddings
        
        # Configuración de negocio
        self.function_purpose = function_purpose
        self.personality_tone = personality_tone
        self.limitations = limitations
        self.dynamic_info = dynamic_info
        
        # Tools
        self.tools = tools or []
        
        # Callbacks
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error
        
        # Inicializar MCP si está configurado
        if mcp_servers:
            self._initialize_mcp(mcp_servers)
        
        # Construir el workflow
        self.graph = self._build_workflow()

    def _initialize_mcp(self, mcp_servers: Dict[str, Any]):
        """Inicializa conexiones MCP y carga tools."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = asyncio.run(self.mcp_client.get_tools())
            self.tools.extend(mcp_tools)
            self.logger.info(f"MCP initialized with {len(mcp_tools)} tools")
        except Exception as e:
            self.logger.error(f"Error initializing MCP: {e}")

    def _build_workflow(self) -> StateGraph:
        """
        Construye el workflow de LangGraph.
        
        Flujo lineal sin ciclos:
        orchestrator → executor → synthesizer → END
        """
        # Crear nodos
        orchestrator = SmartOrchestrator(
            model=self.model,
            tools=self.tools,
            logger=self.logger
        )
        
        executor = ActionExecutor(
            tools=self.tools,
            on_tool_start=self.on_tool_start,
            on_tool_end=self.on_tool_end,
            on_tool_error=self.on_tool_error,
            logger=self.logger
        )
        
        synthesizer = ConversationSynthesizer(
            model=self.model,
            logger=self.logger
        )
        
        # Crear grafo
        workflow = StateGraph(PlannerState)
        
        # Agregar nodos
        workflow.add_node("orchestrator", orchestrator)
        workflow.add_node("executor", self._wrap_async_node(executor))
        workflow.add_node("synthesizer", synthesizer)
        
        # Definir flujo lineal
        workflow.set_entry_point("orchestrator")
        
        # Orchestrator → Executor (condicional: solo si hay acciones)
        def should_execute(state: PlannerState) -> str:
            action_plan = state.get("action_plan")
            if action_plan and action_plan.get("actions"):
                return "executor"
            return "synthesizer"
        
        workflow.add_conditional_edges(
            "orchestrator",
            should_execute,
            {
                "executor": "executor",
                "synthesizer": "synthesizer"
            }
        )
        
        # Executor → Synthesizer (siempre)
        workflow.add_edge("executor", "synthesizer")
        
        # Synthesizer → END (siempre)
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()

    def _wrap_async_node(self, node):
        """Wrapper para nodos async que los hace compatibles con el grafo."""
        async def async_wrapper(state: PlannerState) -> Dict[str, Any]:
            return await node(state)
        return async_wrapper

    def get_response(
        self,
        user_input: str,
        messages: List[Any],
        logs: List[str],
    ) -> Dict[str, Any]:
        """
        Genera una respuesta para el mensaje del usuario.
        
        Args:
            user_input: Mensaje del usuario
            messages: Historial de conversación
            logs: Logs previos
            
        Returns:
            Dict con:
            - content: Respuesta del bot
            - logs: Logs actualizados
            - tools_executed: Lista de tools ejecutadas
            - token_usage: Uso de tokens
        """
        # Estado inicial
        initial_state: PlannerState = {
            "user_input": user_input,
            "messages": messages,
            "dynamic_info": self.dynamic_info,
            "function_purpose": self.function_purpose,
            "personality_tone": self.personality_tone,
            "limitations": self.limitations,
            "action_plan": None,
            "execution_results": [],
            "final_response": None,
            "logs": logs,
            "tools_executed": [],
            "token_usage": {}
        }
        
        # Ejecutar workflow con tracking de tokens
        with get_openai_callback() as cb:
            result = asyncio.run(self.graph.ainvoke(initial_state))
            token_usage = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens
            }
        
        # Extraer resultados
        content = result.get("final_response", "")
        new_logs = result.get("logs", [])
        tools_executed = result.get("tools_executed", [])
        
        return {
            "content": content,
            "logs": new_logs,
            "tools_executed": tools_executed,
            "token_usage": token_usage
        }

    def update_dynamic_info(self, dynamic_info: str):
        """
        Actualiza la información dinámica del contexto.
        
        Útil cuando el contexto cambia entre llamadas (ej: después de
        aceptar políticas, el dynamic_info debería actualizarse).
        """
        self.dynamic_info = dynamic_info

    def get_tools_info(self) -> List[Dict[str, str]]:
        """Retorna información sobre las tools disponibles."""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]
