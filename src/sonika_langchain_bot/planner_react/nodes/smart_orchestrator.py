"""Smart Orchestrator - El cerebro que genera planes de acción."""

from typing import Dict, Any, List, Optional
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from sonika_langchain_bot.planner_react.nodes.base_node import BaseNode


class SmartOrchestrator(BaseNode):
    """
    Orquestador inteligente con capacidad de ReAct (Reasoning + Acting).
    
    Usa bind_tools para que el modelo conozca los schemas exactos.
    Puede reflexionar sobre resultados de tools y decidir si reintentar.
    """

    MAX_REACT_ITERATIONS = 3  # Máximo de ciclos de reflexión

    def __init__(self, model, tools: List[Any] = None, logger=None):
        super().__init__(logger)
        self.tools = tools or []
        self.tool_names = {t.name for t in self.tools}
        self.base_model = model
        
        # Bind tools al modelo
        # tool_choice="auto" permite al modelo decidir, pero con nuestro prompt agresivo debería llamar tools
        self.model = model.bind_tools(self.tools) if self.tools else model
        
    def _build_system_prompt(self, dynamic_info: str, function_purpose: str, is_retry: bool = False) -> str:
        """Construye el prompt del sistema - genérico para cualquier negocio."""
        
        policies_accepted = "Policies accepted: Yes" in dynamic_info
        
        # Construir lista de tools con sus descripciones
        tool_descriptions = []
        for tool in self.tools:
            name = tool.name.lower()
            desc = tool.description[:150] if tool.description else "No description"
            
            # Agregar contexto adicional para tools comunes
            if any(k in name for k in ['contact', 'contacto', 'save']):
                tool_descriptions.append(f"- **{tool.name}**: {desc} (call BEFORE other business tools)")
            elif any(k in name for k in ['polic', 'consent', 'accept']):
                if not policies_accepted:
                    tool_descriptions.append(f"- **{tool.name}**: {desc}")
            else:
                tool_descriptions.append(f"- **{tool.name}**: {desc}")
        
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available."
        
        retry_instructions = ""
        if is_retry:
            retry_instructions = """
## RETRY INSTRUCTIONS
The previous tool call didn't return useful results. Consider:
1. Try with different/broader search terms
2. Try alternative parameters
3. If still no results, explain what you tried and suggest alternatives"""
        
        return f"""You are a Smart Orchestrator. Your PRIMARY job is to CALL TOOLS when appropriate.

## CRITICAL RULES
1. **ACTION OVER WORDS**: When the user wants something done, CALL THE TOOL. Don't just describe what you would do.
2. **USE CONTEXT**: All dates, times, and contextual information are in CURRENT STATE below.
3. **EXTRACT DATA**: If the user provides contact info (name, email, phone), save it.
4. **CONFIRM POLICIES**: If user confirms acceptance (yes/sí/ok/acepto), call the policy tool.

## BUSINESS CONTEXT
{function_purpose}

## AVAILABLE TOOLS
{tools_text}
{retry_instructions}

## CURRENT STATE
{dynamic_info}

Analyze the user's request and call the appropriate tool(s). If no tool is needed, respond naturally."""

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el mensaje y genera el plan de acciones."""
        
        user_input = state.get("user_input", "")
        messages = state.get("messages", [])
        dynamic_info = state.get("dynamic_info", "")  # Debe incluir fecha actual si es necesario
        function_purpose = state.get("function_purpose", "")
        previous_results = state.get("execution_results", [])
        react_iteration = state.get("react_iteration", 0)
        
        # Verificar si es un retry (hay resultados previos sin éxito útil)
        is_retry = self._should_retry(previous_results)
        
        # Convertir historial
        history = self._convert_messages_to_langchain(messages)
        
        # Construir mensajes - conversación completa para contexto
        llm_messages = [
            SystemMessage(content=self._build_system_prompt(dynamic_info, function_purpose, is_retry)),
            *history,  # Conversación completa
            HumanMessage(content=user_input)
        ]
        
        # Si es retry, agregar contexto de resultados previos
        if is_retry and previous_results:
            retry_context = self._build_retry_context(previous_results)
            llm_messages.append(HumanMessage(content=retry_context))
        
        try:
            response = self.model.invoke(llm_messages)
            
            tool_calls = getattr(response, 'tool_calls', []) or []
            reasoning = response.content or ""
            
            
            # Convertir a acciones
            actions = [
                {
                    "type": "execute_tool",
                    "priority": i,
                    "tool_name": tc["name"],
                    "params": tc["args"]
                }
                for i, tc in enumerate(tool_calls)
            ]
            
            intent = self._determine_intent(actions, messages, dynamic_info)
            
            # Construir plan con metadata detallada
            plan_data = {
                "reasoning": reasoning,
                "actions": actions,
                "requires_policy_acceptance": self._should_request_policies(messages, dynamic_info),
                "conversation_intent": intent,
                "is_retry": is_retry,
                "iteration": react_iteration
            }
            
            # Log descriptivo con valores de parámetros
            if actions:
                tools_desc = ", ".join(
                    f"{a['tool_name']}({', '.join(f'{k}={repr(v)[:50]}' for k, v in a['params'].items())})"
                    for a in actions
                )
            else:
                tools_desc = "none"
            log_parts = [
                f"Intent: {intent}",
                f"Tools: [{tools_desc}]",
            ]
            if reasoning:
                # Truncar razonamiento para el log
                short_reasoning = reasoning[:150] + "..." if len(reasoning) > 150 else reasoning
                log_parts.append(f"Reasoning: {short_reasoning}")
            if is_retry:
                log_parts.append(f"(Retry #{react_iteration})")
            
            log_msg = " | ".join(log_parts)
            
            return {
                "action_plan": plan_data,
                "react_iteration": react_iteration,
                **self._add_log(log_msg)
            }
            
        except Exception as e:
            self.logger.error(f"Orchestrator error: {e}")
            return {
                "action_plan": {
                    "reasoning": f"Error: {str(e)}",
                    "actions": [],
                    "requires_policy_acceptance": self._should_request_policies(messages, dynamic_info),
                    "conversation_intent": "error"
                },
                **self._add_log(f"Error: {e}")
            }

    def _should_retry(self, results: List[Dict]) -> bool:
        """Determina si debería reintentar basado en resultados previos."""
        if not results:
            return False
        
        for result in results:
            output = result.get("output", "").lower()
            status = result.get("status", "")
            
            # Casos que ameritan retry
            if status == "failed":
                return True
            if any(phrase in output for phrase in [
                "no se encontr", "not found", "no results", 
                "no hay", "no disponible", "not available",
                "sin resultados", "empty"
            ]):
                return True
        
        return False

    def _build_retry_context(self, results: List[Dict]) -> str:
        """Construye contexto de retry para el modelo."""
        context_parts = ["## PREVIOUS ATTEMPT RESULTS"]
        
        for result in results:
            tool = result.get("tool_name", "unknown")
            params = result.get("params", {})
            output = result.get("output", "")[:300]
            status = result.get("status", "")
            
            context_parts.append(f"""
Tool: {tool}
Parameters: {json.dumps(params, ensure_ascii=False)}
Status: {status}
Result: {output}""")
        
        context_parts.append("\nPlease try a different approach or parameters.")
        return "\n".join(context_parts)

    def _determine_intent(self, actions: List[Dict], messages: List, dynamic_info: str) -> str:
        """Determina el intent basado en las acciones."""
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
        # Si ya aceptó, no pedir
        if "Policies accepted: Yes" in dynamic_info:
            return False
        
        # Verificar si existe alguna tool de políticas
        has_policy_tool = any(
            any(k in t.name.lower() for k in ['polic', 'consent', 'accept'])
            for t in self.tools
        )
        
        # Si no hay tool de políticas, no pedir
        if not has_policy_tool:
            return False
        
        # Si es el primer mensaje, pedir políticas
        if not messages:
            return True
        
        # Si hay mensajes pero no ha aceptado, seguir pidiendo
        # (el synthesizer decidirá cómo comunicarlo)
        return True
