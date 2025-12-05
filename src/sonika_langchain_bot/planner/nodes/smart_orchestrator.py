"""Smart Orchestrator - El cerebro que genera planes de acción."""

from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from sonika_langchain_bot.planner.nodes.base_node import BaseNode


class SmartOrchestrator(BaseNode):
    """
    Orquestador inteligente que usa bind_tools para que el modelo
    conozca los schemas exactos y genere tool_calls con parámetros correctos.
    
    El modelo es lo suficientemente inteligente para:
    - Detectar datos de contacto naturalmente
    - Decidir cuándo llamar cada tool
    - Generar los parámetros correctos
    """

    def __init__(self, model, tools: List[Any] = None, logger=None):
        super().__init__(logger)
        self.tools = tools or []
        self.tool_names = {t.name for t in self.tools}
        
        # Bind tools al modelo - esto es todo lo que necesitamos
        # El modelo conocerá los schemas exactos de cada tool
        self.model = model.bind_tools(self.tools) if self.tools else model

    def _build_system_prompt(self, dynamic_info: str, function_purpose: str) -> str:
        """Construye el prompt del sistema basado en el contexto."""
        
        # Extraer estado de políticas del dynamic_info
        policies_accepted = "Policies accepted: Yes" in dynamic_info
        
        # Construir reglas específicas basadas en las tools disponibles
        tool_rules = []
        
        for tool in self.tools:
            name = tool.name.lower()
            
            # Regla para tools de contacto
            if any(k in name for k in ['contact', 'contacto', 'save']):
                tool_rules.append(f"""
**{tool.name}**: Call this tool when the user provides contact information (name, email, phone).
- Only include data explicitly provided by the user
- Call this BEFORE other business tools""")
            
            # Regla para tools de políticas
            elif any(k in name for k in ['polic', 'consent', 'accept']):
                if not policies_accepted:
                    tool_rules.append(f"""
**{tool.name}**: Call this when the user confirms policy acceptance (says yes/sí/ok/acepto/agree).
- Policies are NOT yet accepted
- Pass the user's confirmation message as parameter""")
            
            # Regla para tools de búsqueda
            elif any(k in name for k in ['search', 'knowledge', 'document', 'buscar']):
                tool_rules.append(f"""
**{tool.name}**: Call this for questions about requirements, policies, prices, procedures, or any factual information.""")
        
        rules_text = "\n".join(tool_rules) if tool_rules else "Use tools as appropriate based on their descriptions."
        
        return f"""You are a Smart Orchestrator. Analyze the user's message and call the appropriate tools.

## CONTEXT
{function_purpose}

## TOOL USAGE RULES
{rules_text}

## CRITICAL INSTRUCTIONS
1. Use EXACT parameter names as defined in each tool's schema
2. You can call MULTIPLE tools in one response (they execute in order)
3. If you lack required parameters, DO NOT call the tool - ask the user instead
4. NEVER invent data the user didn't provide
5. For optional parameters, only include them if explicitly provided

## CURRENT STATE
{dynamic_info}"""

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el mensaje y genera el plan de acciones usando tool_calls nativas."""
        
        user_input = state.get("user_input", "")
        messages = state.get("messages", [])
        dynamic_info = state.get("dynamic_info", "")
        function_purpose = state.get("function_purpose", "")
        
        # Convertir historial a formato LangChain
        history = self._convert_messages_to_langchain(messages)
        
        # Construir mensajes para el LLM
        llm_messages = [
            SystemMessage(content=self._build_system_prompt(dynamic_info, function_purpose)),
            *history[-6:],  # Últimos 6 mensajes de contexto
            HumanMessage(content=user_input)
        ]
        
        try:
            # Invocar modelo - bind_tools hace la magia
            response = self.model.invoke(llm_messages)
            
            # Extraer tool_calls nativas
            tool_calls = getattr(response, 'tool_calls', []) or []
            
            # Convertir a nuestro formato de acciones
            actions = [
                {
                    "type": "execute_tool",
                    "priority": i,
                    "tool_name": tc["name"],
                    "params": tc["args"]  # Parámetros correctos gracias a bind_tools
                }
                for i, tc in enumerate(tool_calls)
            ]
            
            # Determinar intent basado en las acciones
            intent = self._determine_intent(actions, messages, dynamic_info)
            
            # Construir plan
            plan_data = {
                "analysis": response.content or "Tool calls generated",
                "actions": actions,
                "requires_policy_acceptance": self._should_request_policies(messages, dynamic_info),
                "conversation_intent": intent
            }
            
            log_msg = f"Plan generated: {len(actions)} actions, intent: {intent}"
            
            return {
                "action_plan": plan_data,
                **self._add_log(log_msg)
            }
            
        except Exception as e:
            self.logger.error(f"Orchestrator error: {e}")
            
            return {
                "action_plan": {
                    "analysis": f"Error: {str(e)}",
                    "actions": [],
                    "requires_policy_acceptance": self._should_request_policies(messages, dynamic_info),
                    "conversation_intent": "error"
                },
                **self._add_log(f"Orchestrator error: {e}")
            }

    def _determine_intent(self, actions: List[Dict], messages: List, dynamic_info: str) -> str:
        """Determina el intent basado en las acciones generadas."""
        if not actions:
            if not messages and "Policies accepted: Yes" not in dynamic_info:
                return "policy_request"
            return "chitchat"
        
        first_tool = actions[0]["tool_name"].lower()
        
        if any(k in first_tool for k in ['polic', 'consent', 'accept']):
            return "policy_confirmation"
        if any(k in first_tool for k in ['search', 'knowledge', 'document']):
            return "information_request"
        
        return "task_request"

    def _should_request_policies(self, messages: List, dynamic_info: str) -> bool:
        """Determina si se debe pedir aceptación de políticas."""
        # Solo en primer mensaje y si hay tool de políticas
        if messages:
            return False
        if "Policies accepted: Yes" in dynamic_info:
            return False
        
        # Verificar si existe alguna tool de políticas
        return any(
            any(k in t.name.lower() for k in ['polic', 'consent', 'accept'])
            for t in self.tools
        )
