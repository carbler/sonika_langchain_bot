# sonika_langchain_bot/langchain_bot.py
from typing import Generator, List, Optional, Dict, Any, TypedDict, Annotated, Callable, Literal
import asyncio
import logging
import json
from pydantic import BaseModel, Field
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import BaseTool
from langchain.callbacks.base import BaseCallbackHandler
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

# Import your existing interfaces
from sonika_langchain_bot.langchain_class import (
    FileProcessorInterface, 
    IEmbeddings, 
    ILanguageModel, 
    Message, 
    ResponseModel
)


# ============= REASONING SYSTEM MODELS =============

class ReasoningPlan(BaseModel):
    """
    Generic reasoning plan structure for decision making.
    
    This model enforces structured reasoning output from the language model,
    ensuring consistent decision-making patterns regardless of the specific use case.
    
    Attributes:
        user_intent: Clear description of what the user is requesting
        context_summary: Summary of relevant conversation context
        available_actions: List of possible actions the bot can take
        constraints: Any constraints or rules that apply to this request
        decision: The final decision (execute_tool, respond_directly, request_clarification)
        reasoning_chain: Step-by-step reasoning process
        selected_tool: Name of the tool to use (if decision is execute_tool)
        tool_arguments: Arguments for the selected tool
        response_to_user: Direct response text (if decision is respond_directly or request_clarification)
        confidence_level: Confidence in the decision (low, medium, high)
    """
    user_intent: str = Field(description="Clear understanding of user's request")
    context_summary: str = Field(description="Relevant context from conversation")
    available_actions: List[str] = Field(description="Possible actions based on available tools and capabilities")
    constraints: List[str] = Field(description="Rules, limitations, or requirements that apply")
    decision: Literal["execute_tool", "respond_directly", "request_clarification"] = Field(
        description="Final decision on how to proceed"
    )
    reasoning_chain: List[str] = Field(description="Step-by-step reasoning process")
    selected_tool: Optional[str] = Field(default=None, description="Tool name if executing a tool")
    tool_arguments: Optional[Dict[str, Any]] = Field(default=None, description="Arguments for the tool")
    response_to_user: Optional[str] = Field(default=None, description="Direct response text if not executing tool")
    confidence_level: Literal["low", "medium", "high"] = Field(description="Confidence in this decision")


class TokenUsage(BaseModel):
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def add(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add another TokenUsage to this one."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


class ChatState(TypedDict):
    """
    Enhanced chat state for LangGraph workflow with reasoning support.
    
    Attributes:
        messages: List of conversation messages with automatic message handling
        context: Contextual information from processed files
        reasoning_plan: Optional reasoning plan from the reasoning node
        reasoning_attempts: Number of reasoning attempts made
        token_usage: Accumulated token usage across all model invocations
    """
    messages: Annotated[List[BaseMessage], add_messages]
    context: str
    reasoning_plan: Optional[Dict[str, Any]]
    reasoning_attempts: int
    token_usage: Dict[str, int]


# ============= CALLBACK HANDLER =============

class _InternalToolLogger(BaseCallbackHandler):
    """
    Internal callback handler that bridges LangChain callbacks to user-provided functions.
    
    This class is used internally to forward tool execution events to the optional
    callback functions provided by the user during bot initialization.
    """
    
    def __init__(self, 
                 on_start: Optional[Callable[[str, str], None]] = None,
                 on_end: Optional[Callable[[str, str], None]] = None,
                 on_error: Optional[Callable[[str, str], None]] = None):
        """
        Initialize the internal tool logger.
        
        Args:
            on_start: Optional callback function called when a tool starts execution
            on_end: Optional callback function called when a tool completes successfully
            on_error: Optional callback function called when a tool encounters an error
        """
        super().__init__()
        self.on_start_callback = on_start
        self.on_end_callback = on_end
        self.on_error_callback = on_error
        self.current_tool_name = None
        self.tool_executions = []
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing."""
        tool_name = serialized.get("name", "unknown")
        self.current_tool_name = tool_name
        
        self.tool_executions.append({
            "tool": tool_name,
            "input": input_str,
            "status": "started"
        })
        
        if self.on_start_callback:
            try:
                self.on_start_callback(tool_name, input_str)
            except Exception as e:
                logging.error(f"Error in on_tool_start callback: {e}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool completes successfully."""
        tool_name = self.current_tool_name or "unknown"
        
        if hasattr(output, 'content'):
            output_str = output.content
        elif isinstance(output, str):
            output_str = output
        else:
            output_str = str(output)
        
        if self.tool_executions:
            self.tool_executions[-1]["status"] = "success"
            self.tool_executions[-1]["output"] = output_str
        
        if self.on_end_callback:
            try:
                self.on_end_callback(tool_name, output_str)
            except Exception as e:
                logging.error(f"Error in on_tool_end callback: {e}")
        
        self.current_tool_name = None

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error."""
        tool_name = self.current_tool_name or "unknown"
        error_message = str(error)
        
        if self.tool_executions:
            self.tool_executions[-1]["status"] = "error"
            self.tool_executions[-1]["error"] = error_message
        
        if self.on_error_callback:
            try:
                self.on_error_callback(tool_name, error_message)
            except Exception as e:
                logging.error(f"Error in on_tool_error callback: {e}")
        
        self.current_tool_name = None


# ============= REASONING NODE =============

class ReasoningNode:
    """
    Generic reasoning node that analyzes user requests and plans actions.
    
    This node enforces structured decision-making based on the bot's instructions
    and available tools, without requiring domain-specific business rules.
    
    Features:
        - Structured JSON output for consistent decision-making
        - Validation of plan coherence
        - Retry mechanism for malformed plans
        - Generic reasoning based on provided instructions
        - Token usage tracking
        - Reasoning callbacks for monitoring
    """
    
    def __init__(self, 
                 model, 
                 instructions: str,
                 tools: List[BaseTool],
                 max_retries: int = 2,
                 logger: Optional[logging.Logger] = None,
                 on_reasoning_update: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the reasoning node.
        
        Args:
            model: The language model to use for reasoning
            instructions: System instructions that define the bot's behavior and rules
            tools: Available tools that can be used
            max_retries: Maximum number of retries for generating valid plans
            logger: Optional logger instance
            on_reasoning_update: Callback called when reasoning completes with the full plan
        """
        self.model = model
        self.instructions = instructions
        self.tools = tools
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger(__name__)
        self.on_reasoning_update = on_reasoning_update
        
        # Build tool descriptions for reasoning
        self.tool_descriptions = self._build_tool_descriptions()
    
    def _build_tool_descriptions(self) -> str:
        """Build a concise description of available tools for reasoning."""
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for tool in self.tools:
            desc = f"- {tool.name}: {tool.description}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def _extract_token_usage(self, response) -> TokenUsage:
        """Extract token usage from model response."""
        if hasattr(response, 'response_metadata'):
            token_data = response.response_metadata.get('token_usage', {})
            return TokenUsage(
                prompt_tokens=token_data.get('prompt_tokens', 0),
                completion_tokens=token_data.get('completion_tokens', 0),
                total_tokens=token_data.get('total_tokens', 0)
            )
        return TokenUsage()
    
    def __call__(self, state: ChatState) -> ChatState:
        """
        Execute reasoning on the current conversation state.
        
        Args:
            state: Current chat state
            
        Returns:
            Updated state with reasoning plan and token usage
        """
        last_user_message = self._get_last_user_message(state["messages"])
        
        if not last_user_message:
            return state
        
        # Initialize token tracking for this reasoning session
        accumulated_tokens = TokenUsage()
        
        # Attempt to generate a valid reasoning plan
        for attempt in range(self.max_retries + 1):
            try:
                plan, tokens = self._generate_reasoning_plan(
                    last_user_message, 
                    state, 
                    attempt
                )
                
                # Accumulate tokens from this attempt
                accumulated_tokens = accumulated_tokens.add(tokens)
                
                # Validate plan structure
                is_valid, error = self._validate_plan_structure(plan)
                
                if not is_valid:
                    self.logger.warning(f"Plan validation failed (attempt {attempt + 1}): {error}")
                    if attempt < self.max_retries:
                        continue
                    else:
                        # Force fallback on final attempt
                        plan.decision = "respond_directly"
                        plan.response_to_user = "I need to think about this more carefully. Could you rephrase your request?"
                
                # Plan is valid - trigger callback
                plan_dict = plan.dict()
                if self.on_reasoning_update:
                    try:
                        self.on_reasoning_update(plan_dict)
                    except Exception as e:
                        self.logger.error(f"Error in on_reasoning_update callback: {e}")
                
                # Update state token usage
                current_tokens = state.get("token_usage", {})
                new_tokens = {
                    "prompt_tokens": current_tokens.get("prompt_tokens", 0) + accumulated_tokens.prompt_tokens,
                    "completion_tokens": current_tokens.get("completion_tokens", 0) + accumulated_tokens.completion_tokens,
                    "total_tokens": current_tokens.get("total_tokens", 0) + accumulated_tokens.total_tokens
                }
                
                return {
                    **state,
                    "reasoning_plan": plan_dict,
                    "reasoning_attempts": attempt + 1,
                    "token_usage": new_tokens
                }
                
            except Exception as e:
                self.logger.error(f"Error generating reasoning plan (attempt {attempt + 1}): {e}")
                if attempt >= self.max_retries:
                    # Final fallback
                    fallback_plan = {
                        "decision": "respond_directly",
                        "response_to_user": "I apologize, but I'm having trouble processing your request. Could you try rephrasing it?"
                    }
                    
                    if self.on_reasoning_update:
                        try:
                            self.on_reasoning_update(fallback_plan)
                        except Exception as e:
                            self.logger.error(f"Error in on_reasoning_update callback: {e}")
                    
                    # Update tokens even on failure
                    current_tokens = state.get("token_usage", {})
                    new_tokens = {
                        "prompt_tokens": current_tokens.get("prompt_tokens", 0) + accumulated_tokens.prompt_tokens,
                        "completion_tokens": current_tokens.get("completion_tokens", 0) + accumulated_tokens.completion_tokens,
                        "total_tokens": current_tokens.get("total_tokens", 0) + accumulated_tokens.total_tokens
                    }
                    
                    return {
                        **state,
                        "reasoning_plan": fallback_plan,
                        "reasoning_attempts": attempt + 1,
                        "token_usage": new_tokens
                    }
        
        return state
    
    def _generate_reasoning_plan(self, 
                                  user_message: str, 
                                  state: ChatState, 
                                  attempt: int) -> tuple[ReasoningPlan, TokenUsage]:
        """
        Generate a structured reasoning plan using the language model.
        
        Args:
            user_message: Latest user message
            state: Current conversation state
            attempt: Current attempt number
            
        Returns:
            Tuple of (ReasoningPlan, TokenUsage)
        """
        
        # Build reasoning prompt
        reasoning_prompt = f"""You are an analytical reasoning engine. Your task is to analyze the user's request and create a structured action plan.

# Your Instructions and Capabilities
{self.instructions}

# Available Tools
{self.tool_descriptions}

# Current User Request
{user_message}

# Conversation Context
{self._get_context_summary(state)}

# Your Task
Analyze the request and generate a JSON plan following this EXACT structure:

{{
    "user_intent": "What is the user trying to accomplish?",
    "context_summary": "Relevant context from the conversation",
    "available_actions": ["action1", "action2", "..."],
    "constraints": ["rule1", "rule2", "..."],
    "decision": "execute_tool|respond_directly|request_clarification",
    "reasoning_chain": [
        "Step 1: Analysis point",
        "Step 2: Consideration",
        "Step 3: Decision rationale"
    ],
    "selected_tool": "tool_name or null",
    "tool_arguments": {{"arg": "value"}} or null,
    "response_to_user": "response text or null",
    "confidence_level": "low|medium|high"
}}

# Decision Guidelines
- Use "execute_tool" when you have a tool that can fulfill the request and all required information
- Use "respond_directly" when you can answer without tools or to provide information
- Use "request_clarification" when critical information is missing

# Important Rules
1. If executing a tool, you MUST specify selected_tool and tool_arguments
2. If responding directly or requesting clarification, you MUST provide response_to_user
3. Constraints should list any rules from instructions that apply to this request
4. Reasoning chain should explain your decision-making process step by step

Respond ONLY with valid JSON. No additional text."""

        if attempt > 0:
            reasoning_prompt += f"\n\nATTENPT {attempt + 1}: Previous plan was invalid. Ensure strict adherence to the schema."
        
        # Invoke model and track tokens
        response = self.model.invoke([
            {"role": "system", "content": reasoning_prompt}
        ])
        
        tokens = self._extract_token_usage(response)
        
        # Extract and parse JSON
        content = response.content
        
        # Clean markdown formatting
        if "```json" in content:
            content = content.split("```json")[1].split("```").strip()
        elif "```" in content:
            content = content.split("```").split("```")[0].strip()
        
        # Parse JSON and validate with Pydantic
        plan_dict = json.loads(content)
        plan = ReasoningPlan(**plan_dict)
        
        return plan, tokens
    
    def _validate_plan_structure(self, plan: ReasoningPlan) -> tuple[bool, Optional[str]]:
        """
        Validate the structural coherence of a reasoning plan.
        
        Args:
            plan: The reasoning plan to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check decision-specific requirements
        if plan.decision == "execute_tool":
            if not plan.selected_tool:
                return False, "Decision is execute_tool but no tool selected"
            if not plan.tool_arguments:
                return False, "Decision is execute_tool but no tool arguments provided"
            
            # Normalize tool name: remove prefix if present (e.g., "functions.tool_name" -> "tool_name")
            tool_name = plan.selected_tool.split('.')[-1] if '.' in plan.selected_tool else plan.selected_tool
            
            # Verify tool exists
            tool_names = [t.name for t in self.tools]
            if tool_name not in tool_names:
                return False, f"Selected tool '{tool_name}' not in available tools: {tool_names}"
            
            # Update plan with normalized name
            plan.selected_tool = tool_name
        
        elif plan.decision in ["respond_directly", "request_clarification"]:
            if not plan.response_to_user:
                return False, f"Decision is {plan.decision} but no response_to_user provided"
        
        # Ensure reasoning chain is not empty
        if not plan.reasoning_chain or len(plan.reasoning_chain) == 0:
            return False, "Reasoning chain is empty"
        
        return True, None
    
    def _get_last_user_message(self, messages: List[BaseMessage]) -> str:
        """Extract the last user message from conversation history."""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""
    
    def _get_context_summary(self, state: ChatState) -> str:
        """Generate a summary of recent conversation context."""
        messages = state.get("messages", [])
        if len(messages) <= 2:
            return "New conversation"
        
        recent = messages[-6:] if len(messages) >= 6 else messages
        summary = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                summary.append(f"User: {msg.content[:150]}")
            elif isinstance(msg, AIMessage) and msg.content:
                summary.append(f"Assistant: {msg.content[:150]}")
        
        return "\n".join(summary)


# ============= EXECUTION NODE =============

class ExecutionNode:
    """
    Execution node that acts on reasoning decisions.
    
    This node interprets the reasoning plan and either:
    - Prepares tool execution messages
    - Generates direct responses to the user
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def __call__(self, state: ChatState) -> ChatState:
        """
        Execute the decision from the reasoning plan.
        
        Args:
            state: Current chat state with reasoning_plan
            
        Returns:
            Updated state with appropriate message
        """
        plan = state.get("reasoning_plan", {})
        decision = plan.get("decision")
        
        if decision == "execute_tool":
            # Create AI message with tool call for ToolNode to execute
            tool_call_message = AIMessage(
                content="",
                tool_calls=[{
                    "name": plan["selected_tool"],
                    "args": plan["tool_arguments"],
                    "id": f"call_{plan['selected_tool']}"
                }]
            )
            return {
                **state,
                "messages": [tool_call_message]
            }
        
        elif decision in ["respond_directly", "request_clarification"]:
            # Create direct response message
            response_message = AIMessage(
                content=plan.get("response_to_user", "I apologize, but I couldn't process your request.")
            )
            return {
                **state,
                "messages": [response_message]
            }
        
        else:
            # Fallback for unexpected decision types
            self.logger.warning(f"Unexpected decision type: {decision}")
            fallback_message = AIMessage(
                content="I'm not sure how to proceed with your request. Could you rephrase it?"
            )
            return {
                **state,
                "messages": [fallback_message]
            }


# ============= MAIN BOT CLASS =============

class LangChainBot:
    """
    Modern LangGraph-based conversational bot with MCP support and reasoning capabilities.
    
    This implementation provides 100% API compatibility with existing ChatService
    while using modern LangGraph workflows and native tool calling internally.
    
    Features:
        - Native tool calling (no manual parsing)
        - MCP (Model Context Protocol) support
        - Optional reasoning layer for complex decision-making
        - Complete token usage tracking across all model invocations
        - File processing with vector search
        - Thread-based conversation persistence
        - Streaming responses
        - Tool execution callbacks for real-time monitoring
        - Reasoning callbacks for decision transparency
        - Backward compatibility with legacy APIs
    """

    def __init__(self, 
                 language_model: ILanguageModel, 
                 embeddings: IEmbeddings, 
                 instructions: str, 
                 tools: Optional[List[BaseTool]] = None,
                 mcp_servers: Optional[Dict[str, Any]] = None,
                 use_checkpointer: bool = False,
                 use_reasoning: bool = False,
                 reasoning_max_retries: int = 2,
                 logger: Optional[logging.Logger] = None,
                 on_tool_start: Optional[Callable[[str, str], None]] = None,
                 on_tool_end: Optional[Callable[[str, str], None]] = None,
                 on_tool_error: Optional[Callable[[str, str], None]] = None,
                 on_reasoning_update: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Initialize the modern LangGraph bot with optional MCP support, reasoning, and callbacks.

        Args:
            language_model (ILanguageModel): The language model to use for generation
            embeddings (IEmbeddings): Embedding model for file processing and context retrieval
            instructions (str): System instructions that will be modernized automatically
            tools (List[BaseTool], optional): Traditional LangChain tools to bind to the model
            mcp_servers (Dict[str, Any], optional): MCP server configurations for dynamic tool loading
            use_checkpointer (bool): Enable automatic conversation persistence using LangGraph checkpoints
            use_reasoning (bool): Enable reasoning layer for structured decision-making
            reasoning_max_retries (int): Maximum retries for reasoning plan generation (default: 2)
            logger (Optional[logging.Logger]): Logger instance for error tracking (silent by default if not provided)
            on_tool_start (Callable[[str, str], None], optional): Callback when a tool starts.
                Receives (tool_name: str, input_data: str)
            on_tool_end (Callable[[str, str], None], optional): Callback when a tool completes successfully.
                Receives (tool_name: str, output: str)
            on_tool_error (Callable[[str, str], None], optional): Callback when a tool fails.
                Receives (tool_name: str, error_message: str)
            on_reasoning_update (Callable[[Dict[str, Any]], None], optional): Callback that receives the complete
                reasoning plan as soon as it's generated. Receives a dictionary with the full reasoning details.
        
        Note:
            When use_reasoning=True, the bot will analyze requests before acting, providing:
            - Better adherence to complex instructions
            - Structured decision-making process
            - Validation of tool usage
            - Clear reasoning chain for debugging
            - Complete token tracking across all reasoning steps
            
        Example:
            def on_reasoning(plan: Dict):
                print(f"Decision: {plan['decision']}")
                print(f"Reasoning: {plan['reasoning_chain']}")
                
            bot = LangChainBot(
                language_model=model,
                embeddings=embeddings,
                instructions="You are a helpful assistant...",
                use_reasoning=True,
                on_reasoning_update=on_reasoning
            )
        """
        # Configure logger (silent by default if not provided)
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())
        
        # Core components
        self.language_model = language_model
        self.embeddings = embeddings
        self.base_instructions = instructions
        self.use_reasoning = use_reasoning
        self.reasoning_max_retries = reasoning_max_retries
        
        # Backward compatibility attributes
        self.chat_history: List[BaseMessage] = []
        self.vector_store = None
        
        # Tool configuration
        self.tools = tools or []
        self.mcp_client = None
        
        # Tool execution callbacks
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error
        
        # Reasoning callbacks
        self.on_reasoning_update = on_reasoning_update
        
        # Initialize MCP servers if provided
        if mcp_servers:
            self._initialize_mcp(mcp_servers)
        
        # Configure persistence layer
        self.checkpointer = MemorySaver() if use_checkpointer else None
        
        # Prepare model with bound tools for native function calling
        self.model_with_tools = self._prepare_model_with_tools()
        
        # Build modern instruction set with tool descriptions
        self.instructions = self._build_modern_instructions()
        
        # Create the LangGraph workflow
        self.graph = self._create_workflow()
        
        # Legacy compatibility attributes (maintained for API compatibility)
        self.conversation = None
        self.agent_executor = None

    def _initialize_mcp(self, mcp_servers: Dict[str, Any]):
        """Initialize MCP (Model Context Protocol) connections and load available tools."""
        try:
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = asyncio.run(self.mcp_client.get_tools())
            self.tools.extend(mcp_tools)
            self.logger.info(f"MCP initialized with {len(mcp_tools)} tools")
        except Exception as e:
            self.logger.error(f"Error inicializando MCP: {e}")
            self.logger.exception("Traceback completo:")
            self.mcp_client = None

    def _prepare_model_with_tools(self):
        """Prepare the language model with bound tools for native function calling."""
        if self.tools:
            return self.language_model.model.bind_tools(self.tools)
        return self.language_model.model

    def _build_modern_instructions(self) -> str:
        """Build enhanced instructions with tool descriptions."""
        instructions = self.base_instructions
        
        if self.tools:
            tools_description = "\n\n# Available Tools\n\n"
            
            for tool in self.tools:
                tools_description += f"## {tool.name}\n"
                tools_description += f"**Description:** {tool.description}\n\n"
                
                # Handle different tool schema formats
                if hasattr(tool, 'args_schema') and tool.args_schema and hasattr(tool.args_schema, '__fields__'):
                    tools_description += f"**Parameters:**\n"
                    for field_name, field_info in tool.args_schema.__fields__.items():
                        required = "**REQUIRED**" if field_info.is_required() else "*optional*"
                        tools_description += f"- `{field_name}` ({field_info.annotation.__name__}, {required}): {field_info.description}\n"
                
                elif hasattr(tool, 'args_schema') and isinstance(tool.args_schema, dict):
                    if 'properties' in tool.args_schema:
                        tools_description += f"**Parameters:**\n"
                        for param_name, param_info in tool.args_schema['properties'].items():
                            required = "**REQUIRED**" if param_name in tool.args_schema.get('required', []) else "*optional*"
                            param_desc = param_info.get('description', 'No description')
                            param_type = param_info.get('type', 'any')
                            tools_description += f"- `{param_name}` ({param_type}, {required}): {param_desc}\n"
                
                elif hasattr(tool, '_run'):
                    tools_description += f"**Parameters:**\n"
                    import inspect
                    sig = inspect.signature(tool._run)
                    for param_name, param in sig.parameters.items():
                        if param_name != 'self':
                            param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'any'
                            required = "*optional*" if param.default != inspect.Parameter.empty else "**REQUIRED**"
                            default_info = f" (default: {param.default})" if param.default != inspect.Parameter.empty else ""
                            tools_description += f"- `{param_name}` ({param_type}, {required}){default_info}\n"
                            
                tools_description += "\n"
            
            tools_description += ("## Usage Instructions\n"
                                "- Use the standard function calling format\n"
                                "- **MUST** provide all REQUIRED parameters\n"
                                "- Do NOT call tools with empty arguments\n")
            
            instructions += tools_description
        
        return instructions

    def _extract_token_usage_from_message(self, message: BaseMessage) -> TokenUsage:
        """Extract token usage from a message's metadata."""
        if hasattr(message, 'response_metadata'):
            token_data = message.response_metadata.get('token_usage', {})
            return TokenUsage(
                prompt_tokens=token_data.get('prompt_tokens', 0),
                completion_tokens=token_data.get('completion_tokens', 0),
                total_tokens=token_data.get('total_tokens', 0)
            )
        return TokenUsage()

    def _create_workflow(self) -> StateGraph:
        """
        Create LangGraph workflow with optional reasoning layer.
        
        Returns:
            Compiled StateGraph workflow
        """
        if self.use_reasoning:
            return self._create_reasoning_workflow()
        else:
            return self._create_standard_workflow()

    def _create_standard_workflow(self) -> StateGraph:
        """Create standard workflow without reasoning layer."""
        
        def agent_node(state: ChatState) -> ChatState:
            """Main agent node for standard workflow."""
            last_user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            
            if not last_user_message:
                return state
            
            context = self._get_context(last_user_message)
            
            system_content = self.instructions
            if context:
                system_content += f"\n\nContext from uploaded files:\n{context}"
            
            messages = [{"role": "system", "content": system_content}]
            
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content or ""})
                elif isinstance(msg, ToolMessage):
                    messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
            
            try:
                response = self.model_with_tools.invoke(messages)
                
                # Extract token usage
                tokens = self._extract_token_usage_from_message(response)
                current_tokens = state.get("token_usage", {})
                new_tokens = {
                    "prompt_tokens": current_tokens.get("prompt_tokens", 0) + tokens.prompt_tokens,
                    "completion_tokens": current_tokens.get("completion_tokens", 0) + tokens.completion_tokens,
                    "total_tokens": current_tokens.get("total_tokens", 0) + tokens.total_tokens
                }
                
                return {
                    **state,
                    "context": context,
                    "messages": [response],
                    "token_usage": new_tokens
                }
            except Exception as e:
                self.logger.error(f"Error en agent_node: {e}")
                self.logger.exception("Traceback completo:")
                fallback_response = AIMessage(content="I apologize, but I encountered an error processing your request.")
                return {
                    **state,
                    "context": context,
                    "messages": [fallback_response]
                }

        def should_continue(state: ChatState) -> str:
            """Determine if tools should be executed."""
            last_message = state["messages"][-1]
            if (isinstance(last_message, AIMessage) and 
                hasattr(last_message, 'tool_calls') and 
                last_message.tool_calls):
                return "tools"
            return "end"

        workflow = StateGraph(ChatState)
        workflow.add_node("agent", agent_node)
        
        if self.tools:
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)
        
        workflow.set_entry_point("agent")
        
        if self.tools:
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "tools": "tools",
                    "end": END
                }
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)
        
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()

    def _create_reasoning_workflow(self) -> StateGraph:
        """
        Create workflow with reasoning layer.
        
        Workflow: reasoning → execution → [tools] → [agent_formatter] → end
        """
        
        # Initialize reasoning and execution nodes
        reasoning_node = ReasoningNode(
            model=self.model_with_tools,
            instructions=self.instructions,
            tools=self.tools,
            max_retries=self.reasoning_max_retries,
            logger=self.logger,
            on_reasoning_update=self.on_reasoning_update
        )
        
        execution_node = ExecutionNode(logger=self.logger)
        
        def agent_formatter_node(state: ChatState) -> ChatState:
            """
            Final formatting node that runs after tool execution.
            Formats tool results into a natural response.
            """
            # Get the last tool message
            last_tool_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    last_tool_message = msg
                    break
            
            if not last_tool_message:
                return state
            
            # Build context for formatting
            system_content = self.instructions + "\n\nFormat the tool result into a natural, helpful response for the user."
            
            messages = [{"role": "system", "content": system_content}]
            
            # Add conversation context
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, ToolMessage):
                    messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
            
            try:
                response = self.model_with_tools.invoke(messages)
                
                # Extract and accumulate tokens
                tokens = self._extract_token_usage_from_message(response)
                current_tokens = state.get("token_usage", {})
                new_tokens = {
                    "prompt_tokens": current_tokens.get("prompt_tokens", 0) + tokens.prompt_tokens,
                    "completion_tokens": current_tokens.get("completion_tokens", 0) + tokens.completion_tokens,
                    "total_tokens": current_tokens.get("total_tokens", 0) + tokens.total_tokens
                }
                
                return {
                    **state,
                    "messages": [response],
                    "token_usage": new_tokens
                }
            except Exception as e:
                self.logger.error(f"Error formatting tool result: {e}")
                fallback = AIMessage(content="The operation completed successfully.")
                return {
                    **state,
                    "messages": [fallback]
                }
        
        def should_execute_tool(state: ChatState) -> str:
            """Route based on reasoning decision."""
            plan = state.get("reasoning_plan", {})
            decision = plan.get("decision")
            
            if decision == "execute_tool":
                return "tools"
            return "end"
        
        # Build workflow
        workflow = StateGraph(ChatState)
        
        workflow.add_node("reasoning", reasoning_node)
        workflow.add_node("execution", execution_node)
        
        if self.tools:
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)
            workflow.add_node("agent_formatter", agent_formatter_node)
        
        # Define edges
        workflow.set_entry_point("reasoning")
        workflow.add_edge("reasoning", "execution")
        
        if self.tools:
            workflow.add_conditional_edges(
                "execution",
                should_execute_tool,
                {
                    "tools": "tools",
                    "end": END
                }
            )
            workflow.add_edge("tools", "agent_formatter")
            workflow.add_edge("agent_formatter", END)
        else:
            workflow.add_edge("execution", END)
        
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()

    # ===== PUBLIC API METHODS =====

    def get_response(self, user_input: str) -> ResponseModel:
        """
        Generate a response while maintaining 100% API compatibility.
        
        This method tracks ALL token usage across the entire workflow including:
        - Reasoning invocations (if enabled)
        - Tool executions
        - Final response formatting
        
        Args:
            user_input (str): The user's message or query
            
        Returns:
            ResponseModel: Structured response with complete token counts and response text
        """
        initial_state = {
            "messages": self.chat_history + [HumanMessage(content=user_input)],
            "context": "",
            "reasoning_plan": None,
            "reasoning_attempts": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        config = {}
        if self.on_tool_start or self.on_tool_end or self.on_tool_error:
            tool_logger = _InternalToolLogger(
                on_start=self.on_tool_start,
                on_end=self.on_tool_end,
                on_error=self.on_tool_error
            )
            config["callbacks"] = [tool_logger]
        
        result = asyncio.run(self.graph.ainvoke(initial_state, config=config))
        
        self.chat_history = result["messages"]
        
        # Extract final response
        final_response = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break
        
        # Get accumulated token usage from state
        token_usage = result.get("token_usage", {})
        
        return ResponseModel(
            user_tokens=token_usage.get("prompt_tokens", 0),
            bot_tokens=token_usage.get("completion_tokens", 0),
            response=final_response
        )

    def get_response_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Generate a streaming response for real-time user interaction.
        
        Args:
            user_input (str): The user's message or query
            
        Yields:
            str: Response chunks as they are generated
        """
        initial_state = {
            "messages": self.chat_history + [HumanMessage(content=user_input)],
            "context": "",
            "reasoning_plan": None,
            "reasoning_attempts": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        config = {}
        if self.on_tool_start or self.on_tool_end or self.on_tool_error:
            tool_logger = _InternalToolLogger(
                on_start=self.on_tool_start,
                on_end=self.on_tool_end,
                on_error=self.on_tool_error
            )
            config["callbacks"] = [tool_logger]
        
        accumulated_response = ""
        
        for chunk in self.graph.stream(initial_state, config=config):
            if "agent" in chunk or "agent_formatter" in chunk:
                node_key = "agent" if "agent" in chunk else "agent_formatter"
                for message in chunk[node_key]["messages"]:
                    if isinstance(message, AIMessage) and message.content:
                        accumulated_response = message.content
                        yield message.content
        
        if accumulated_response:
            self.chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=accumulated_response)
            ])

    def load_conversation_history(self, messages: List[Message]):
        """Load conversation history from Django model instances."""
        self.chat_history.clear()
        for message in messages:
            if message.is_bot:
                self.chat_history.append(AIMessage(content=message.content))
            else:
                self.chat_history.append(HumanMessage(content=message.content))

    def save_messages(self, user_message: str, bot_response: str):
        """Save messages to internal conversation history."""
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=bot_response))

    def process_file(self, file: FileProcessorInterface):
        """Process and index a file for contextual retrieval."""
        document = file.getText()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)

        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                [doc.page_content for doc in texts], 
                self.embeddings
            )
        else:
            self.vector_store.add_texts([doc.page_content for doc in texts])

    def clear_memory(self):
        """Clear conversation history and processed file context."""
        self.chat_history.clear()
        self.vector_store = None

    def get_chat_history(self) -> List[BaseMessage]:
        """Retrieve a copy of the current conversation history."""
        return self.chat_history.copy()

    def set_chat_history(self, history: List[BaseMessage]):
        """Set the conversation history from a list of BaseMessage instances."""
        self.chat_history = history.copy()

    def _get_context(self, query: str) -> str:
        """Retrieve relevant context from processed files using similarity search."""
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=4)
            return "\n".join([doc.page_content for doc in docs])
        return ""

    def get_last_reasoning_plan(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the last reasoning plan for debugging or monitoring.
        
        Returns:
            Dict with reasoning plan details or None if reasoning not used
        """
        if not self.use_reasoning:
            return None
        
        # Search for last AI message with reasoning metadata
        for msg in reversed(self.chat_history):
            if hasattr(msg, 'additional_kwargs') and 'reasoning_plan' in msg.additional_kwargs:
                return msg.additional_kwargs['reasoning_plan']
        
        return None