"""Orchestrator Node - The Brain."""

from typing import Dict, Any, List, Optional
import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from sonika_langchain_bot.orchestrator.nodes.base_node import BaseNode

class OrchestratorNode(BaseNode):
    """
    Decides which specialist agent to call based on user input and rules.
    """

    def __init__(self, model, tools: List[Any] = None, logger=None):
        super().__init__(logger)
        self.model = model
        self.tools = tools or []
        self.system_prompt_base = self._build_dynamic_prompt()

    def _build_dynamic_prompt(self) -> str:
        """Construct the base system prompt dynamically based on tools."""

        tool_names = {t.name for t in self.tools}

        # Base Definition
        prompt = [
            "You are the Master Orchestrator.",
            "## OBJECTIVE",
            "Analyze the User Input, the Current Context, AND the Conversation History to route the conversation to the ONE specialist agent best suited to handle the CURRENT turn.",
            "",
            "## AVAILABLE SPECIALISTS"
        ]

        # 1. Policy Agent (Conditional)
        if "accept_policies" in tool_names:
            prompt.append("""
1. **PolicyAgent**:
   - **Role**: Handles mandatory policy acceptance.
   - **Trigger**:
     - IF the system instructions say policies are mandatory AND context shows they are NOT accepted yet.
     - IF the user says "yes", "ok", "agree", "claro" AND the policies are NOT accepted yet.
   - **Priority**: HIGHEST. Overrides everything else.""")
        else:
            prompt.append("""
1. **PolicyAgent**:
   - **Role**: Handles mandatory policy acceptance.
   - **Trigger**: IF policies are mandatory and NOT accepted yet.
   - **Note**: Only route here if absolutely necessary, as no automatic acceptance tool is available.""")

        # 2. Research Agent (Conditional)
        if any("search" in t or "knowledge" in t for t in tool_names):
            prompt.append("""
2. **ResearchAgent**:
   - **Role**: Searches for information in the Knowledge Base (documents).
   - **Trigger**: IF the user asks a specific QUESTION about requirements, policies, locations, general info, or "how to".
   - **Exclusion**: Do NOT use for checking real-time car availability, prices, quotes, or reservations. Use TaskAgent for that.""")
        else:
             prompt.append("""
2. **ResearchAgent**:
   - **Role**: Answers general questions based on internal knowledge.
   - **Trigger**: Informational queries.""")

        # 3. Task Agent (Conditional)
        # Assuming TaskAgent handles business logic if tools exist OR if standard task tools are present
        prompt.append("""
3. **TaskAgent**:
   - **Role**: Executes business actions (Quote, Reserve, Save Contact, Send Email, Check Availability).
   - **Trigger**:
     - IF the user provides ANY contact data (Name, Phone, Email), even if mixed with a greeting.
     - IF the user asks to check availability, prices, or see cars (e.g., "cheapest car", "prices for tomorrow").
     - IF the user explicitly asks to perform an action (reserve, quote, send email).
     - **CONTEXTUAL TRIGGER**: IF the user says "Yes", "Help me", "Proceed" AND the previous bot message offered to help with a task.""")

        # 4. Chitchat Agent (Always available)
        prompt.append("""
4. **ChitchatAgent**:
   - **Role**: Handles greetings, identity questions ("Who are you?"), and small talk.
   - **Trigger**: IF the user input is purely conversational and requires no business action or information AND contains NO contact data.""")

        # Inputs Section
        prompt.append("""
## INPUTS
- **User Request**: {user_input}
- **Context**: {dynamic_info}
- **Instructions**: {function_purpose}
- **Recent Activity (Tools executed in this turn)**:
{recent_activity}

## CONVERSATION HISTORY ANALYSIS (CRITICAL)
Before deciding, review the MESSAGE HISTORY provided in the context to detect **PENDING INTENTS**:
1. **Interrupted Tasks:** Did the user ask for something (e.g., "price of car") but was interrupted (e.g., by policy request)?
   - If YES, and policies are now accepted -> Route to **TaskAgent** to resume the request.
2. **Bot Questions:** Did the ASSISTANT just ask a specific question (e.g., "Do you want to reserve?", "Confirm city?")?
   - If YES, and user says "Si", "Ayudame", "Dale" -> Route to **TaskAgent** to answer the question and proceed.
3. **Specific Questions Override:** If the user asks a NEW specific question (e.g., "What are requirements?"), route to **ResearchAgent** even if there is a pending task.""")

        # Decision Logic (Dynamic)
        prompt.append("""
## DECISION LOGIC""")

        # Policy Logic
        if "accept_policies" in tool_names:
            prompt.append("""1. **CHECK POLICIES (CRITICAL)**:
   - **STEP 1:** Check `Recent Activity`. Does it show `accept_policies (success)`?
     - **IF YES** -> Policies are JUST ACCEPTED. **STOP** checking policies. Proceed to STEP 2 (Check Intent).
   - **STEP 2:** Check `Context` (`dynamic_info`). Does it say **"Policies accepted: Yes"**?
     - **IF YES** -> Policies are DONE. **STOP** checking policies. Proceed to STEP 2.
   - **STEP 3:** If neither of above, and `Instructions` say mandatory -> Route to **PolicyAgent**.""")
        else:
             prompt.append("""1. **CHECK POLICIES**: Check `Context`. If policies are missing and mandatory -> Route to **PolicyAgent**.""")

        prompt.append("""
2. **CHECK INTENT (Only if Policies are Accepted)**:
   - **SPECIFIC QUESTION PRIORITY:** Does the user ask "What are requirements?", "How much is deposit?" -> **ResearchAgent**.
   - **PENDING INTENT PRIORITY:** Is there an interrupted task or a bot offer that matches the user's reply? -> **TaskAgent**.
   - **DATA SAVING:** Does the input contain a name, phone, or email? -> **TaskAgent**.
   - **GREETING:** Greeting/Identity (WITHOUT data/pending intent)? -> **ChitchatAgent**.
   - **TASK:** Real-time Info (Prices, Cars, Availability) OR Actions? -> **TaskAgent**.
""")

        return "\n".join(prompt)

    def _format_tools_executed(self, tools: List[Dict[str, Any]]) -> str:
        """Summarize tools executed in this turn."""
        if not tools:
            return "None"

        summary = []
        for tool in tools:
            name = tool.get("tool_name", "unknown")
            status = tool.get("status", "unknown")
            summary.append(f"- {name} ({status})")
        return "\n".join(summary)

    def _convert_messages(self, messages: List[Any]) -> List[BaseMessage]:
        """Convert custom Message objects to LangChain BaseMessage objects."""
        converted = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                converted.append(msg)
                continue

            # Duck typing for custom Message class
            is_bot = getattr(msg, "is_bot", False)
            content = getattr(msg, "content", str(msg))

            if is_bot:
                converted.append(AIMessage(content=content))
            else:
                converted.append(HumanMessage(content=content))
        return converted

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Route to the correct agent."""

        user_input = state.get("user_input", "")
        dynamic_info = state.get("dynamic_info", "")
        function_purpose = state.get("function_purpose", "")

        # Get recent tools to detect policy acceptance within the loop
        tools_executed = state.get("tools_executed", [])
        recent_activity = self._format_tools_executed(tools_executed)

        # Get Chat History and convert to objects
        raw_messages = state.get("messages", [])
        history_messages = self._convert_messages(raw_messages)

        # Fill the dynamic prompt with runtime values
        system_prompt = self.system_prompt_base.format(
            user_input=user_input,
            dynamic_info=dynamic_info,
            function_purpose=function_purpose,
            recent_activity=recent_activity
        )

        # Construct final message list: System -> History -> Trigger
        messages_input = [
            SystemMessage(content=system_prompt),
            *history_messages,
            HumanMessage(content=f"""
Analyze history and input. Route now.

## RECENT ACTIVITY
{recent_activity}

## OUTPUT FORMAT
Return a JSON object:
{{
    "reasoning": "Explanation of why you chose this agent based on rules AND history analysis.",
    "next_agent": "policy" | "research" | "task" | "chitchat"
}}
""")
        ]

        try:
            # Force JSON output for reliable routing
            response = self.model.invoke(
                messages_input,
                config={"temperature": 0.0},
                response_format={"type": "json_object"}
            )
            content = response.content
            decision_data = json.loads(content)

            next_agent = decision_data.get("next_agent", "chitchat")
            reasoning = decision_data.get("reasoning", "")

            log_update = self._add_log(state, f"Routing to: {next_agent.upper()} | Reason: {reasoning}")

            return {
                "next_agent": next_agent,
                "orchestrator_reasoning": reasoning,
                **log_update
            }

        except Exception as e:
            self.logger.error(f"Orchestrator failed: {e}")
            # Fallback safe
            return {"next_agent": "chitchat", "orchestrator_reasoning": "Error in routing"}
