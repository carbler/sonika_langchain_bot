"""Planner Node - The Brain."""

from typing import Dict, Any, Optional, Callable, List
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from sonika_langchain_bot.bot.tasker.nodes.base_node import BaseNode

class PlannerNode(BaseNode):
    """
    ReAct Planner that decides the next step.
    It reads instructions from text files for maximum flexibility.
    """

    def __init__(
        self,
        model,
        tools: List[Any],
        max_iterations: int = 10,
        on_planner_update: Optional[Callable] = None,
        logger=None
    ):
        super().__init__(logger)
        self.model = model.bind_tools(tools) if tools else model
        self.tools = tools
        self.max_iterations = max_iterations
        self.on_planner_update = on_planner_update

        # Load prompts
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.system_prompt_template = self._load_prompt("planner_system.txt")
        self.domain_rules = self._load_prompt("domain_rules.txt")

    def _load_prompt(self, filename: str) -> str:
        """Loads a prompt from the prompts directory."""
        try:
            path = os.path.join(self.base_path, "prompts", filename)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error loading prompt {filename}: {e}")
            return ""

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one ReAct planning cycle."""

        iteration = state.get("react_iteration", 0)

        # Check iteration limit
        if iteration >= self.max_iterations:
            return self._finish("Maximum iterations reached")

        # Build prompt inputs
        function_purpose = state.get("function_purpose", "")
        limitations = state.get("limitations", "")

        # Combine static domain rules with whatever dynamic rules might exist (future proofing)
        # For now, we just use the loaded domain_rules
        conditional_rules = self.domain_rules

        # Construct System Prompt
        system_prompt = self.system_prompt_template.format(
            function_purpose=function_purpose,
            limitations=limitations,
            conditional_rules=conditional_rules
        )

        # Build Analysis Input (User + History + Tools)
        observation = self._get_last_observation(state)
        analysis_input = self._build_analysis_input(state, observation)

        # Convert History
        custom_messages = state.get('messages', [])
        conversation_messages = self._convert_messages_to_langchain(custom_messages)

        # Assemble Messages
        messages = [
            SystemMessage(content=system_prompt),
            *conversation_messages,
            HumanMessage(content=analysis_input)
        ]

        # Invoke Model
        try:
            response = self.model.invoke(messages, config={"temperature": 0.1})
        except Exception as e:
            self.logger.error(f"Model invocation failed: {e}")
            return self._finish(f"Model error: {e}")

        # Extract Decision
        decision = self._extract_decision(response)

        # Logging
        log_msg = f"Iteración {iteration + 1}: DECISIÓN → {decision['decision']} | Tool: {decision.get('tool')} | Razonamiento: {decision.get('reasoning')}"
        log_update = self._add_log(state, log_msg)

        # Update State
        planner_update = {
            "react_iteration": iteration + 1,
            "planner_output": decision,
            **log_update
        }

        # Callback
        if self.on_planner_update:
            try:
                self.on_planner_update({
                    "decision": decision.get("decision"),
                    "reasoning": decision.get("reasoning"),
                    "iteration": iteration
                })
            except Exception as e:
                pass

        return planner_update

    def _finish(self, reason: str) -> Dict[str, Any]:
        """Helper to create a finish decision."""
        return {
            "planner_output": {
                "decision": "finish",
                "reasoning": reason,
                "tool": None,
                "params": {}
            }
        }

    def _get_last_observation(self, state: Dict[str, Any]) -> Optional[str]:
        """Get observation from last tool execution."""
        # In new architecture, we look at the last element of tools_executed list
        tools_executed = state.get("tools_executed", [])
        if not tools_executed:
            return None

        last_tool = tools_executed[-1]
        tool_name = last_tool.get('tool_name', 'unknown')
        output = last_tool.get('output', 'No output')
        status = last_tool.get('status', 'unknown')

        return f"Tool: {tool_name}\nStatus: {status}\nResult: {output}"

    def _build_analysis_input(self, state: Dict[str, Any], observation: Optional[str]) -> str:
        """Build the analysis input for the current iteration."""
        user_input = state.get('user_input', '')
        tools_history = self._get_tools_history(state)

        parts = [f"## User Request\n{user_input}"]

        if tools_history:
            parts.append(f"## Tools Already Executed\n{tools_history}")

        if observation:
            parts.append(f"## Last Observation (Immediate Context)\n{observation}")

        parts.append("""
## Your Task
Analyze the situation and decide:
- Do you need to call a tool? If yes, call it.
- Do you have enough information? If yes, explain your reasoning and FINISH.
""")
        return "\n\n".join(parts)

    def _get_tools_history(self, state: Dict[str, Any]) -> str:
        """Get summary of ALL tools executed in the session."""
        tools_executed = state.get('tools_executed', [])
        if not tools_executed:
            return ""

        history = []
        for tool in tools_executed:
            name = tool.get('tool_name', 'unknown')
            status = tool.get('status', 'unknown')
            # Maybe include snippet of output if needed? For now keep it brief to save tokens
            history.append(f"- {name}: {status}")
        return "\n".join(history)

    def _convert_messages_to_langchain(self, messages: List[Any]) -> List[BaseMessage]:
        """Convierte mensajes custom a mensajes nativos de LangChain."""
        converted = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                converted.append(msg)
                continue

            if hasattr(msg, 'is_bot') and hasattr(msg, 'content'):
                if msg.is_bot:
                    converted.append(AIMessage(content=msg.content))
                else:
                    converted.append(HumanMessage(content=msg.content))
                continue

            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role in ('assistant', 'bot'):
                    converted.append(AIMessage(content=content))
                elif role == 'system':
                    converted.append(SystemMessage(content=content))
                else:
                    converted.append(HumanMessage(content=content))
                continue

            converted.append(HumanMessage(content=str(msg)))
        return converted

    def _extract_decision(self, response: AIMessage) -> Dict[str, Any]:
        """Extrae la decisión del AIMessage."""
        decision = {
            "decision": "finish",
            "reasoning": response.content.strip() if response.content else "",
            "tool": None,
            "params": {}
        }

        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            decision["decision"] = "execute_tool"
            decision["tool"] = tool_call.get("name")
            decision["params"] = tool_call.get("args", {})

        return decision
