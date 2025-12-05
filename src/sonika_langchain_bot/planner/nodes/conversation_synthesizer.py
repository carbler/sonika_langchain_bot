"""Conversation Synthesizer - Genera la respuesta final coherente."""

from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from sonika_langchain_bot.planner.nodes.base_node import BaseNode


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
            
            # Truncar output si es muy largo
            if len(output) > 500:
                output = output[:500] + "..."
            
            if status == "success":
                summary_parts.append(f"✓ **{tool_name}** executed successfully:\n{output}")
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
        
        prompt = f"""You are a Conversation Synthesizer. Your job is to generate a SINGLE, COHERENT response to the user.

## YOUR PERSONALITY & TONE
{personality_tone}

CRITICAL: You MUST adopt this tone in every response. Be natural, warm, and human-like.

## BUSINESS CONTEXT
{function_purpose}

## LIMITATIONS
{limitations}

## YOUR TASK
Based on:
1. The user's original message
2. What actions were executed (and their results)
3. The conversation history
4. Whether policies need to be requested

Generate ONE response that:
- Acknowledges what was done (if applicable)
- Answers the user's question (using tool results if available)
- Maintains conversational flow
- Uses the configured personality/tone
- Is in the SAME LANGUAGE as the user (likely Spanish)

## CRITICAL RULES

### Rule 1: Policy Request
If `requires_policy_acceptance` is TRUE and policies were NOT just accepted:
- Your response MUST ask the user to accept the privacy policies and terms of use
- Be polite but clear that this is required to continue
- Include the policy links if provided in the business context

### Rule 2: Use Tool Results
If tools were executed successfully:
- Use their output to answer the user
- Don't say "I searched and found..." - just provide the information naturally
- If search returned no results, say you don't have that information

### Rule 3: Contact Data Confirmation
If contact data was saved:
- Briefly acknowledge it (e.g., "Perfecto, Erley!" or "Guardé tu información")
- Don't make it the main focus unless that was the user's only request

### Rule 4: Failed Tools
If a tool failed:
- Don't expose technical errors to the user
- Apologize briefly and offer to try again or help differently

### Rule 5: Chitchat
If no actions were needed (pure greeting/chitchat):
- Respond naturally according to your personality
- Offer to help with something specific

## OUTPUT
Generate ONLY the response text. No JSON, no markdown headers, no explanations.
Just the natural response to send to the user."""

        return prompt

    def _format_conversation_history(self, messages: List[Any]) -> str:
        """Formatea el historial de conversación para el contexto."""
        if not messages:
            return "No previous conversation."
        
        converted = self._convert_messages_to_langchain(messages)
        
        history_parts = []
        for msg in converted[-10:]:  # Últimos 10 mensajes
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
        plan_analysis = action_plan.get("analysis", "")
        detected_contact = action_plan.get("detected_contact_data")
        
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
            f"## ORCHESTRATOR ANALYSIS\n{plan_analysis}",
            f"## CONVERSATION INTENT\n{conversation_intent}",
            f"## REQUIRES POLICY ACCEPTANCE\n{'YES - You MUST ask for policy acceptance' if requires_policy else 'NO'}",
        ]
        
        if detected_contact:
            context_parts.append(f"## CONTACT DATA DETECTED\n{detected_contact}")
        
        if execution_results:
            context_parts.append(f"## ACTIONS EXECUTED\n{execution_summary}")
        else:
            context_parts.append("## ACTIONS EXECUTED\nNo actions were executed this turn.")
        
        context = "\n\n".join(context_parts)
        
        # Mensajes para el LLM
        llm_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"{context}\n\nNow generate the response to the user:")
        ]
        
        try:
            response = self.model.invoke(llm_messages)
            final_response = response.content.strip()
            
            # Limpiar respuesta si tiene formato no deseado
            if final_response.startswith('"') and final_response.endswith('"'):
                final_response = final_response[1:-1]
            
            return {
                "final_response": final_response,
                **self._add_log(f"Response generated ({len(final_response)} chars)")
            }
            
        except Exception as e:
            self.logger.error(f"Synthesizer error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Respuesta de fallback
            fallback = "Lo siento, tuve un problema procesando tu mensaje. ¿Podrías intentarlo de nuevo?"
            
            return {
                "final_response": fallback,
                **self._add_log(f"Synthesizer error, using fallback: {e}")
            }
