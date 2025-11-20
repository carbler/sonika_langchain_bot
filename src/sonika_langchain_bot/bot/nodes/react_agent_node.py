"""ReAct Agent Node - Con bind_tools y conversión de mensajes custom."""

from typing import Dict, Any, Optional, Callable, List
from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from sonika_langchain_bot.bot.nodes.base_node import BaseNode


class ReActAgentNode(BaseNode):
    """Pure ReAct agent usando tool calling nativo del modelo."""
    
    def __init__(
        self,
        model,
        tools: List[Any],
        max_iterations: int = 10,
        on_planner_update: Optional[Callable] = None,
        logger=None
    ):
        super().__init__(logger)
        # Bind tools al modelo
        self.model = model.bind_tools(tools) if tools else model
        self.tools = tools
        self.max_iterations = max_iterations
        self.on_planner_update = on_planner_update
        self.conditional_rules = self._build_conditional_rules()
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one ReAct cycle."""
        
        iteration = state.get("react_iteration", 0)
        
        if iteration >= self.max_iterations:
            return self._finish(state, "Maximum iterations reached")
        
        observation = self._get_last_observation(state)
        response = self._generate_reasoning(state, observation, iteration)
        decision = self._extract_decision(response)
        reasoning = decision.get("reasoning", "Sin razonamiento")
        
        if decision.get("decision") == "execute_tool":
            tool_name = decision.get("tool", "unknown")
            params = decision.get("params", {})
            reasoning = decision.get("reasoning", "")
            self._add_log(
                state, 
                f"Iteración {iteration + 1}: DECISIÓN → execute_tool | Tool: {tool_name} | Params: {params} | Razonamiento: {reasoning}"
            )
        else:
            reasoning = decision.get("reasoning", "")
            self._add_log(state,f"Iteración {iteration + 1}: DECISIÓN → finish | Razonamiento: {reasoning}")

        
        # Crear entrada de historial
        react_entry = {
            'iteration': iteration,
            'reasoning': decision.get('reasoning', ''),
            'decision': decision.get('decision', ''),
            'tool': decision.get('tool'),
            'params': decision.get('params', {})
        }
        
        # Callback
        if self.on_planner_update:
            try:
                self.on_planner_update({
                    "decision": decision.get("decision", "finish"),
                    "reasoning": decision.get("reasoning", ""),
                    "iteration": iteration
                })
            except Exception as e:
                self.logger.warning(f"Callback error: {e}")
        
        # Retornar actualizaciones
        return {
            "react_iteration": iteration + 1,
            "react_history": [react_entry],
            "planner_output": decision
        }
    
    def _convert_messages_to_langchain(self, messages: List[Any]) -> List[BaseMessage]:
        """
        Convierte mensajes custom a mensajes nativos de LangChain.
        
        Detecta si son:
        - Ya BaseMessage nativos → los retorna tal cual
        - Objetos custom con .is_bot y .content → los convierte
        - Dicts con 'role' y 'content' → los convierte
        """
        converted = []
        
        for msg in messages:
            # Caso 1: Ya es un BaseMessage nativo
            if isinstance(msg, BaseMessage):
                converted.append(msg)
                continue
            
            # Caso 2: Objeto custom con atributos is_bot y content
            if hasattr(msg, 'is_bot') and hasattr(msg, 'content'):
                if msg.is_bot:
                    converted.append(AIMessage(content=msg.content))
                else:
                    converted.append(HumanMessage(content=msg.content))
                continue
            
            # Caso 3: Dict con role y content
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'assistant' or role == 'bot':
                    converted.append(AIMessage(content=content))
                elif role == 'system':
                    converted.append(SystemMessage(content=content))
                else:
                    converted.append(HumanMessage(content=content))
                continue
            
            # Si no es ninguno de los casos anteriores, intentar convertir a string
            converted.append(HumanMessage(content=str(msg)))
        
        return converted
    
    def _get_last_observation(self, state: Dict[str, Any]) -> Optional[str]:
        """Get observation from last tool execution."""
        executor_output = state.get("executor_output")
        if not executor_output:
            return None
        
        tools_executed = executor_output.get("tools_executed", [])
        if not tools_executed:
            return None
        
        last_tool = tools_executed[-1]
        tool_name = last_tool.get('tool_name', 'unknown')
        output = last_tool.get('output', 'No output')
        status = last_tool.get('status', 'unknown')

        observation = f"Tool: {tool_name}\nStatus: {status}\nResult: {output}"
        
        return observation
    
    def _build_system_prompt(self, state: Dict[str, Any], iteration: int) -> str:
        """Build system prompt with strict rules."""
        
        dynamic_info = state.get('dynamic_info', '')
        is_first_message = self._is_first_message(state)
        
        system_prompt = f"""You are a ReAct planning agent (iteration {iteration + 1}/{self.max_iterations}).

    ## Dynamic Context
    {dynamic_info}

    {self.conditional_rules}

    ## YOUR ROLE (CRITICAL):
    You are NOT a conversational assistant. You are a PLANNER that decides which tools to use.

    Your ONLY job is to:
    1. Analyze if you need to call a tool → call it
    2. If you already have all info from tools → explain what info you have (NOT the final answer to user)

    ## CRITICAL RULES:
    1. **Use the current date/time from context for any date-related operations**
    2. **Follow ALL conditional rules above** - they are MANDATORY
    3. You may execute multiple different tools, but NEVER the same tool twice.
    4. If you need info → call the appropriate tool
    5. NEVER execute the same tool twice
    6. **NEVER EVER generate conversational responses to the user**
    7. **NEVER say things like "How can I help you?" or "I appreciate your interest"**
    8. **Your reasoning should be technical, not conversational**

    {"## IMPORTANT: This is the FIRST message in the conversation" if is_first_message else ""}

    ## EXAMPLES OF CORRECT REASONING:

    ❌ BAD (conversational):
    "Parece que estás preguntando cómo estoy. Aprecio tu interés..."

    ✅ GOOD (technical):
    "Need to check policies first befeore to continue" OR "Already have search results, ready to pass to output node"

    ## Your Task:
    Analyze and decide ONLY:
    - Need a tool? → Call it
    - Have info? → State what you have (DON'T answer the user)"""
        
        return system_prompt
    
    def _build_analysis_input(self, state: Dict[str, Any], observation: Optional[str]) -> str:
        """Build the analysis input for the current iteration."""
        
        user_input = state.get('user_input', '')
        tools_history = self._get_tools_history(state)
        
        analysis_parts = []
        
        # User request
        analysis_parts.append(f"## User Request\n{user_input}")
        
        # Tools history
        if tools_history:
            analysis_parts.append(f"## Tools Already Executed\n{tools_history}")
        
        # Last observation
        if observation:
            analysis_parts.append(f"## Last Observation\n{observation}")
        
        # Final instruction
        analysis_parts.append("""
## Your Task
Analyze the situation and decide:
- Do you need to call a tool? If yes, call it.
- Do you have enough information? If yes, explain your reasoning.

Now analyze and decide.""")
                
        return "\n\n".join(analysis_parts)
    
    def _generate_reasoning(self, state: Dict[str, Any], observation: Optional[str], iteration: int) -> AIMessage:
        """Generate reasoning with tool calling support."""
        
        # Build system prompt
        system_prompt = self._build_system_prompt(state, iteration)
        
        # ✅ CONVERSIÓN: Mensajes custom → BaseMessage nativos
        custom_messages = state.get('messages', [])
        conversation_messages = self._convert_messages_to_langchain(custom_messages)
        
        # Build analysis input
        analysis_input = self._build_analysis_input(state, observation)
        
        # Construir lista de mensajes
        message_list = [
            SystemMessage(content=system_prompt),
            *conversation_messages,
            HumanMessage(content=analysis_input)
        ]
        
        # Invoke model
        response = self.model.invoke(message_list, config={"temperature": 0.1})
        
        return response
    
    def _extract_decision(self, response: AIMessage) -> Dict[str, Any]:
        """Extrae la decisión del AIMessage con tool_calls nativos."""
        
        decision = {
            "decision": "finish",
            "reasoning": "",
            "tool": None,
            "params": {}
        }
        
        # Extraer razonamiento del contenido
        if hasattr(response, 'content') and response.content:
            decision["reasoning"] = response.content.strip()
        
        # Extraer tool_calls directamente
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # El modelo quiere ejecutar una herramienta
            tool_call = response.tool_calls[0]  # Tomar la primera tool call
            
            decision["decision"] = "execute_tool"
            decision["tool"] = tool_call.get("name", "")
            decision["params"] = tool_call.get("args", {})
        else:
            # El modelo no llamó ninguna tool, está listo para finalizar
            decision["decision"] = "finish"
        
        return decision
        
    def _get_tools_history(self, state: Dict[str, Any]) -> str:
        """Get summary of ALL tools executed in the session."""
        # ✅ Leer de tools_executed directamente (que se acumula automáticamente)
        tools_executed = state.get('tools_executed', [])
        
        if not tools_executed:
            return ""
        
        history = []
        seen_tools = set()
        
        for tool in tools_executed:
            tool_name = tool.get('tool_name', 'unknown')
            status = tool.get('status', 'unknown')
            
            # Registrar para evitar duplicados en el mensaje
            if tool_name in seen_tools:
                continue
            
            history.append(f"- {tool_name}: {status}")
            seen_tools.add(tool_name)
        
        return "\n".join(history)
    
    def _finish(self, state: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Finish the ReAct loop."""
        decision = {
            "decision": "finish",
            "reasoning": reason,
            "tool": None,
            "params": {}
        }
        return {"planner_output": decision}
    
    def _build_conditional_rules(self) -> str:
        """Build conditional rules based on available tools."""
        if not self.tools:
            return ""
        
        tool_names = {tool.name for tool in self.tools}
        rules = []
        
        # Rule 1: CORPORATE RULE - search_knowledge_documents
        if 'search_knowledge_documents' in tool_names:
            rules.append("""
## CORPORATE RULE — MANDATORY USE OF `search_knowledge_documents`
If the user's query might be answered by internal documents:
- ALWAYS call `search_knowledge_documents` FIRST before responding
- Use the user's message as the query
- Never invent information if it might exist in documents
""")
        
        # Rule 2: ASK BEFORE EXECUTING POLICY TOOL
        if 'accept_policies' in tool_names:
            rules.append("""
## POLICY ACCEPTANCE HANDLING
- On the FIRST user message of the conversation, you MUST ask if they accept the privacy policies and terms of use.
- Do NOT call the `accept_policies` tool automatically.
- Wait for the user's explicit confirmation (e.g. "yes", "sí", "acepto", "ok").
- As soon as the user confirms, you MUST immediately call the `accept_policies` tool.
- Pass the user's confirmation inside the `user_message` parameter.
- If the user does NOT confirm, do not call the tool and continue waiting for acceptance.
- This rule is applied only once: after successfully executing `accept_policies`, NEVER ask for acceptance again.
""")
        
        # Rule 3: AUTOMATIC CONTACT UPDATE
        if 'create_or_update_contact' in tool_names:
            rules.append("""
## AUTOMATIC CONTACT UPDATE
If the user provides contact information (name, email, phone):
- ALWAYS call `create_or_update_contact` immediately
- Include any information provided (don't wait for all fields)
- Execute this BEFORE any other action
""")
        
        return "\n".join(rules) if rules else ""
    
    def _is_first_message(self, state: Dict[str, Any]) -> bool:
        """Check if this is the first message in the conversation."""
        messages = state.get('messages', [])
        return len(messages) == 0