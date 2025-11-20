from typing import Generator, List, Optional, Dict, Any, TypedDict, Annotated, Callable
import asyncio
import logging
from pydantic import BaseModel
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
from langchain_community.callbacks.manager import get_openai_callback

# Import your existing interfaces
from sonika_langchain_bot.langchain_class import (
    FileProcessorInterface, 
    IEmbeddings, 
    ILanguageModel, 
    Message, 
)
# ============= TOKEN USAGE MODEL =============

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
    Chat state for LangGraph workflow.
    
    Attributes:
        messages: List of conversation messages with automatic message handling
        context: Contextual information from processed files
        token_usage: Accumulated token usage across all model invocations
    """
    messages: Annotated[List[BaseMessage], add_messages]
    context: str
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
            "tool_name": tool_name,
            "args": input_str,
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


# ============= MAIN BOT CLASS =============

class LangChainBot:
    """
    Modern LangGraph-based conversational bot with MCP support.
    
    This implementation provides 100% API compatibility with existing ChatService
    while using modern LangGraph workflows and native tool calling internally.
    
    Features:
        - Native tool calling (no manual parsing)
        - MCP (Model Context Protocol) support
        - Complete token usage tracking across all model invocations
        - File processing with vector search
        - Thread-based conversation persistence
        - Streaming responses
        - Tool execution callbacks for real-time monitoring
        - Backward compatibility with legacy APIs
    """

    def __init__(self, 
                 language_model: ILanguageModel, 
                 embeddings: IEmbeddings, 
                 instructions: str, 
                 tools: Optional[List[BaseTool]] = None,
                 mcp_servers: Optional[Dict[str, Any]] = None,
                 use_checkpointer: bool = False,
                 logger: Optional[logging.Logger] = None,
                 on_tool_start: Optional[Callable[[str, str], None]] = None,
                 on_tool_end: Optional[Callable[[str, str], None]] = None,
                 on_tool_error: Optional[Callable[[str, str], None]] = None):
        """
        Initialize the modern LangGraph bot with optional MCP support and callbacks.

        Args:
            language_model (ILanguageModel): The language model to use for generation
            embeddings (IEmbeddings): Embedding model for file processing and context retrieval
            instructions (str): System instructions that will be modernized automatically
            tools (List[BaseTool], optional): Traditional LangChain tools to bind to the model
            mcp_servers (Dict[str, Any], optional): MCP server configurations for dynamic tool loading
            use_checkpointer (bool): Enable automatic conversation persistence using LangGraph checkpoints
            logger (Optional[logging.Logger]): Logger instance for error tracking (silent by default if not provided)
            on_tool_start (Callable[[str, str], None], optional): Callback when a tool starts.
                Receives (tool_name: str, input_data: str)
            on_tool_end (Callable[[str, str], None], optional): Callback when a tool completes successfully.
                Receives (tool_name: str, output: str)
            on_tool_error (Callable[[str, str], None], optional): Callback when a tool fails.
                Receives (tool_name: str, error_message: str)
        """
        # Configure logger (silent by default if not provided)
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())
        
        # Core components
        self.language_model = language_model
        self.embeddings = embeddings
        self.instructions = instructions
        
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
        
        # Initialize MCP servers if provided
        if mcp_servers:
            self._initialize_mcp(mcp_servers)
        
        # Configure persistence layer
        self.checkpointer = MemorySaver() if use_checkpointer else None
        
        # Prepare model with bound tools for native function calling
        self.model_with_tools = self.language_model.model.bind_tools(self.tools) if self.tools else self.language_model.model
        
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

    def _create_workflow(self) -> StateGraph:
        """
        Create standard LangGraph workflow.
        
        Returns:
            Compiled StateGraph workflow
        """
        
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
            
            # Add conditional rules based on available tools
            conditional_rules = self._build_conditional_rules()
            if conditional_rules:
                system_content += f"\n\n{conditional_rules}"
            
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

    # ===== PUBLIC API METHODS =====

    def get_response(self, user_input: str) -> dict:
        """
        Generate a response with logs and tool execution tracking.
        
        This method tracks ALL token usage across the entire workflow including:
        - Tool executions
        - Final response formatting
        
        Args:
            user_input (str): The user's message or query
            
        Returns:
            dict: Structured response with content, logs, tools_executed, and token_usage
        """
        initial_state = {
            "messages": self.chat_history + [HumanMessage(content=user_input)],
            "context": "",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        config = {}
        # Siempre crear el tool_logger para rastrear ejecuciones
        tool_logger = _InternalToolLogger(
            on_start=self.on_tool_start,
            on_end=self.on_tool_end,
            on_error=self.on_tool_error
        )
        config["callbacks"] = [tool_logger]
        
        # Usar get_openai_callback para trackear tokens
        with get_openai_callback() as cb:
            result = asyncio.run(self.graph.ainvoke(initial_state, config=config))
            
            # Actualizar token_usage con los datos del callback
            result["token_usage"] = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            }
        
        self.chat_history = result["messages"]
        
        # Extract final response
        final_response = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break
        
        # Get accumulated token usage from callback
        token_usage = result["token_usage"]
        
        # Get tools executed from callback handler
        tools_executed = tool_logger.tool_executions
        
        # Generate logs
        logs = []
        for i, msg in enumerate(result["messages"]):
            if isinstance(msg, HumanMessage):
                logs.append(f"[USER] {msg.content}")
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    logs.append(f"[AGENT] Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
                elif msg.content:
                    logs.append(f"[AGENT] Response generated")
            elif isinstance(msg, ToolMessage):
                logs.append(f"[TOOL] Result received")
        
        return {
            "content": final_response,
            "logs": logs,
            "tools_executed": tools_executed,
            "token_usage": token_usage
        }

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
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        config = {}
        tool_logger = _InternalToolLogger(
            on_start=self.on_tool_start,
            on_end=self.on_tool_end,
            on_error=self.on_tool_error
        )
        config["callbacks"] = [tool_logger]
        
        accumulated_response = ""
        
        for chunk in self.graph.stream(initial_state, config=config):
            if "agent" in chunk:
                for message in chunk["agent"]["messages"]:
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