"""Conversation Synthesizer - Genera la respuesta final coherente."""

from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from sonika_langchain_bot.planner_react.nodes.base_node import BaseNode


class ConversationSynthesizer(BaseNode):
    """
    Sintetizador de conversación que genera UNA respuesta coherente
    basada en:
    - El plan del orquestador
    - Los resultados de las ejecuciones
    - El historial de conversación
    - El tono/personalidad configurado
    
    Este nodo es el ÚNICO que genera la respuesta visible para el usuario.
    """

    def __init__(self, model, logger=None):
        super().__init__(logger)
        self.model = model

    def _build_execution_summary(self, execution_results: List[Dict[str, Any]]) -> str:
        """Construye un resumen de lo que se ejecutó."""
        if not execution_results:
            return "No actions were executed."
        
        summary_parts = []
        
        for result in execution_results:
            tool_name = result.get("tool_name", "unknown")
            status = result.get("status", "unknown")
            output = result.get("output", "")
            action_type = result.get("action_type", "")
            
            if status == "success":
                summary_parts.append(f"✓ **{tool_name}** executed successfully:\n{output}")
            elif status == "missing_params":
                missing = result.get("missing_params", [])
                summary_parts.append(f"⚠️ **{tool_name}** NOT executed - missing required data: {', '.join(missing)}\nYou MUST ask the user for: {', '.join(missing)}")
            elif status == "failed":
                summary_parts.append(f"✗ **{tool_name}** failed:\n{output}")
            else:
                summary_parts.append(f"⊘ **{tool_name}** skipped:\n{output}")
        
        return "\n\n".join(summary_parts)

    def _build_system_prompt(
        self,
        personality_tone: str,
        function_purpose: str,
        limitations: str
    ) -> str:
        """Construye el prompt del sistema para el sintetizador."""
        
        prompt = f"""You are a Conversation Synthesizer. Generate a SINGLE, COHERENT response to the user.

## PERSONALITY & TONE
{personality_tone}

Adopt this tone naturally. Be warm and human-like.

## BUSINESS CONTEXT
{function_purpose}

## LIMITATIONS
{limitations}

## YOUR TASK
Generate ONE response that:
- Uses tool results naturally (don't say "I searched...", just provide the info)
- Maintains conversational flow with the history
- Matches the user's language
- Follows the personality/tone

## RULES
1. **Policy Request**: If `requires_policy_acceptance` is TRUE, ask user to accept policies politely but clearly
2. **Tool Results**: Use successful tool outputs to answer. If no results, say you don't have that info
3. **Contact Saved**: Briefly acknowledge if contact was saved, don't over-focus on it
4. **Failed Tools**: Don't expose errors. Apologize briefly and offer alternatives
5. **Missing Parameters**: If a tool has status "missing_params", ask the user for the specific missing information naturally
6. **Chitchat**: Respond naturally, offer to help with something specific

## OUTPUT
Generate ONLY the response text. No JSON, no headers, no explanations."""

        return prompt

    def _format_conversation_history(self, messages: List[Any]) -> str:
        """Formatea el historial de conversación para el contexto."""
        if not messages:
            return "No previous conversation."
        
        converted = self._convert_messages_to_langchain(messages)
        
        history_parts = []
        for msg in converted:  # Conversación completa
            if isinstance(msg, HumanMessage):
                history_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"Assistant: {msg.content}")
        
        return "\n".join(history_parts)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Genera la respuesta final coherente."""
        
        # Extraer datos del estado
        user_input = state.get("user_input", "")
        messages = state.get("messages", [])
        action_plan = state.get("action_plan", {})
        execution_results = state.get("execution_results", [])
        
        # Contexto
        personality_tone = state.get("personality_tone", "Professional and friendly")
        function_purpose = state.get("function_purpose", "")
        limitations = state.get("limitations", "")
        dynamic_info = state.get("dynamic_info", "")
        
        # Análisis del plan
        requires_policy = action_plan.get("requires_policy_acceptance", False)
        conversation_intent = action_plan.get("conversation_intent", "chitchat")
        plan_reasoning = action_plan.get("reasoning", "")
        is_retry = action_plan.get("is_retry", False)
        iteration = action_plan.get("iteration", 0)
        
        # Verificar si las políticas fueron aceptadas en esta ejecución
        policies_just_accepted = any(
            r.get("tool_name") == "accept_policies" and r.get("status") == "success"
            for r in execution_results
        )
        
        # Si las políticas fueron aceptadas, no pedir de nuevo
        if policies_just_accepted:
            requires_policy = False
        
        # Construir resumen de ejecución
        execution_summary = self._build_execution_summary(execution_results)
        
        # Construir historial
        conversation_history = self._format_conversation_history(messages)
        
        # Construir el prompt del sistema
        system_prompt = self._build_system_prompt(
            personality_tone,
            function_purpose,
            limitations
        )
        
        # Construir el contexto para el LLM
        context_parts = [
            f"## CURRENT CONTEXT\n{dynamic_info}",
            f"## CONVERSATION HISTORY\n{conversation_history}",
            f"## USER'S CURRENT MESSAGE\n{user_input}",
            f"## ORCHESTRATOR REASONING\n{plan_reasoning}",
            f"## CONVERSATION INTENT\n{conversation_intent}",
            f"## REQUIRES POLICY ACCEPTANCE\n{'YES - You MUST ask for policy acceptance' if requires_policy else 'NO'}",
        ]
        
        if is_retry:
            context_parts.append(f"## NOTE\nThis is retry attempt #{iteration + 1}. Previous attempts didn't find useful results.")
        
        if execution_results:
            context_parts.append(f"## ACTIONS EXECUTED\n{execution_summary}")
        else:
            context_parts.append("## ACTIONS EXECUTED\nNo actions were executed this turn.")
        
        context = "\n\n".join(context_parts)
        
        # Convertir historial a mensajes LangChain
        conversation_messages = self._convert_messages_to_langchain(messages)
        
        # Construir input con contexto de ejecución
        analysis_input = f"""{context}

Now generate the response to the user:"""
        
        # Mensajes para el LLM - incluir conversación real para mejor contexto
        llm_messages = [
            SystemMessage(content=system_prompt),
            *conversation_messages,  # Conversación completa como mensajes reales
            HumanMessage(content=analysis_input)
        ]
        
        try:
            response = self.model.invoke(llm_messages)
            final_response = response.content.strip()
            
            # Limpiar respuesta si tiene formato no deseado
            if final_response.startswith('"') and final_response.endswith('"'):
                final_response = final_response[1:-1]
            
            # Log descriptivo
            log_parts = [f"Generated response ({len(final_response)} chars)"]
            if requires_policy:
                log_parts.append("📋 Policy request included")
            if policies_just_accepted:
                log_parts.append("✅ Policies accepted")
            if execution_results:
                success_tools = [r["tool_name"] for r in execution_results if r.get("status") == "success"]
                if success_tools:
                    log_parts.append(f"Using results from: {', '.join(success_tools)}")
            
            return {
                "final_response": final_response,
                **self._add_log(" | ".join(log_parts))
            }
            
        except Exception as e:
            self.logger.error(f"Synthesizer error: {e}")
            
            fallback = "Lo siento, tuve un problema procesando tu mensaje. ¿Podrías intentarlo de nuevo?"
            
            return {
                "final_response": fallback,
                **self._add_log(f"❌ Error: {e} | Using fallback response")
            }
