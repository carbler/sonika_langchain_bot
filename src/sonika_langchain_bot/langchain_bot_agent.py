from typing import Generator, List, Optional, Dict, Any, TypedDict, Annotated, Callable
import asyncio
import logging
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
from sonika_langchain_bot.langchain_class import FileProcessorInterface, IEmbeddings, ILanguageModel, Message, ResponseModel


class ChatState(TypedDict):
    """
    Modern chat state for LangGraph workflow.
    
    Attributes:
        messages: List of conversation messages with automatic message handling
        context: Contextual information from processed files
    """
    messages: Annotated[List[BaseMessage], add_messages]
    context: str

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
        self.tool_executions = []  # Para tracking interno si se necesita
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:

        print(f"DEBUG: on_tool_start se ejecutó!")  # ← AGREGAR ESTO


        """Called when a tool starts executing."""
        tool_name = serialized.get("name", "unknown")
        self.current_tool_name = tool_name
        
        # Track execution internally
        self.tool_executions.append({
            "tool": tool_name,
            "input": input_str,
            "status": "started"
        })
        
        # Call user's callback if provided
        if self.on_start_callback:
            try:
                self.on_start_callback(tool_name, input_str)
            except Exception as e:
                # Don't let user callback errors break the workflow
                logging.error(f"Error in on_tool_start callback: {e}")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        print(f"DEBUG: on_tool_end se ejecutó!")
        tool_name = self.current_tool_name or "unknown"
        
        # Convert output to string if it's a ToolMessage or other object
        if hasattr(output, 'content'):
            output_str = output.content
        elif isinstance(output, str):
            output_str = output
        else:
            output_str = str(output)
        
        # Update internal tracking
        if self.tool_executions:
            self.tool_executions[-1]["status"] = "success"
            self.tool_executions[-1]["output"] = output_str
        
        # Call user's callback if provided
        if self.on_end_callback:
            try:
                self.on_end_callback(tool_name, output_str)
            except Exception as e:
                logging.error(f"Error in on_tool_end callback: {e}")
        
        self.current_tool_name = None

    def on_tool_error(self, error: Exception, **kwargs) -> None:  # ← CORRECTO
        print(f"DEBUG: on_tool_error se ejecutó!")
        tool_name = self.current_tool_name or "unknown"
        error_message = str(error)
        
        # Update internal tracking
        if self.tool_executions:
            self.tool_executions[-1]["status"] = "error"
            self.tool_executions[-1]["error"] = error_message
        
        # Call user's callback if provided
        if self.on_error_callback:
            try:
                self.on_error_callback(tool_name, error_message)
            except Exception as e:
                logging.error(f"Error in on_tool_error callback: {e}")
        
        self.current_tool_name = None

class LangChainBot:
    """
    Modern LangGraph-based conversational bot with MCP support.
    
    This implementation provides 100% API compatibility with existing ChatService
    while using modern LangGraph workflows and native tool calling internally.
    
    Features:
        - Native tool calling (no manual parsing)
        - MCP (Model Context Protocol) support
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
        Initialize the modern LangGraph bot with optional MCP support and tool execution callbacks.

        Args:
            language_model (ILanguageModel): The language model to use for generation
            embeddings (IEmbeddings): Embedding model for file processing and context retrieval
            instructions (str): System instructions that will be modernized automatically
            tools (List[BaseTool], optional): Traditional LangChain tools to bind to the model
            mcp_servers (Dict[str, Any], optional): MCP server configurations for dynamic tool loading
            use_checkpointer (bool): Enable automatic conversation persistence using LangGraph checkpoints
            logger (Optional[logging.Logger]): Logger instance for error tracking (silent by default if not provided)
            on_tool_start (Callable[[str, str], None], optional): Callback function executed when a tool starts.
                Receives (tool_name: str, input_data: str)
            on_tool_end (Callable[[str, str], None], optional): Callback function executed when a tool completes successfully.
                Receives (tool_name: str, output: str)
            on_tool_error (Callable[[str, str], None], optional): Callback function executed when a tool fails.
                Receives (tool_name: str, error_message: str)
        
        Note:
            The instructions will be automatically enhanced with tool descriptions
            when tools are provided, eliminating the need for manual tool instruction formatting.
            
        Example:
```python
            def on_tool_execution(tool_name: str, input_data: str):
                print(f"Tool {tool_name} started with input: {input_data}")
                
            bot = LangChainBot(
                language_model=model,
                embeddings=embeddings,
                instructions="You are a helpful assistant",
                on_tool_start=on_tool_execution
            )
```
        """
        # Configure logger (silent by default if not provided)
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())
        
        # Core components
        self.language_model = language_model
        self.embeddings = embeddings
        self.base_instructions = instructions
        
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
        self.model_with_tools = self._prepare_model_with_tools()
        
        # Build modern instruction set with tool descriptions
        self.instructions = self._build_modern_instructions()
        
        # Create the LangGraph workflow
        self.graph = self._create_modern_workflow()
        
        # Legacy compatibility attributes (maintained for API compatibility)
        self.conversation = None
        self.agent_executor = None

    def _initialize_mcp(self, mcp_servers: Dict[str, Any]):
        """
        Initialize MCP (Model Context Protocol) connections and load available tools.
        
        This method establishes connections to configured MCP servers and automatically
        imports their tools into the bot's tool collection.
        
        Args:
            mcp_servers (Dict[str, Any]): Dictionary of MCP server configurations
                Example: {
                    "server_name": {
                        "command": "python",
                        "args": ["/path/to/server.py"],
                        "transport": "stdio"
                    }
                }
        
        Note:
            MCP tools are automatically appended to the existing tools list and
            will be included in the model's tool binding process.
        """
        try:
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = asyncio.run(self.mcp_client.get_tools())
            self.tools.extend(mcp_tools)
        except Exception as e:
            self.logger.error(f"Error inicializando MCP: {e}")
            self.logger.exception("Traceback completo:")
            self.mcp_client = None

    def _prepare_model_with_tools(self):
        """
        Prepare the language model with bound tools for native function calling.
        
        This method binds all available tools (both traditional and MCP) to the language model,
        enabling native function calling without manual parsing or instruction formatting.
        
        Returns:
            The language model with tools bound, or the original model if no tools are available
        """
        if self.tools:
            return self.language_model.model.bind_tools(self.tools)
        return self.language_model.model

    def _build_modern_instructions(self) -> str:
        instructions = self.base_instructions
        
        if self.tools:
            tools_description = "\n\n# Available Tools\n\n"
            
            for tool in self.tools:
                tools_description += f"## {tool.name}\n"
                tools_description += f"**Description:** {tool.description}\n\n"
                
                # Opción 1: args_schema es una clase Pydantic (HTTPTool)
                if hasattr(tool, 'args_schema') and tool.args_schema and hasattr(tool.args_schema, '__fields__'):
                    tools_description += f"**Parameters:**\n"
                    for field_name, field_info in tool.args_schema.__fields__.items():
                        required = "**REQUIRED**" if field_info.is_required() else "*optional*"
                        tools_description += f"- `{field_name}` ({field_info.annotation.__name__}, {required}): {field_info.description}\n"
                
                # Opción 2: args_schema es un dict (MCP Tools) ← NUEVO
                elif hasattr(tool, 'args_schema') and isinstance(tool.args_schema, dict):
                    if 'properties' in tool.args_schema:
                        tools_description += f"**Parameters:**\n"
                        for param_name, param_info in tool.args_schema['properties'].items():
                            required = "**REQUIRED**" if param_name in tool.args_schema.get('required', []) else "*optional*"
                            param_desc = param_info.get('description', 'No description')
                            param_type = param_info.get('type', 'any')
                            tools_description += f"- `{param_name}` ({param_type}, {required}): {param_desc}\n"
                
                # Opción 3: Tool básico con _run (fallback)
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

    def _create_modern_workflow(self) -> StateGraph:
        """
        Create a modern LangGraph workflow using idiomatic patterns.
        
        This method constructs a state-based workflow that handles:
        - Agent reasoning and response generation
        - Automatic tool execution via ToolNode
        - Context integration from processed files
        - Error handling and fallback responses
        
        Returns:
            StateGraph: Compiled LangGraph workflow ready for execution
        """
        
        def agent_node(state: ChatState) -> ChatState:
            """
            Main agent node responsible for generating responses and initiating tool calls.
            
            This node:
            1. Extracts the latest user message from the conversation state
            2. Retrieves relevant context from processed files
            3. Constructs a complete message history for the model
            4. Invokes the model with tool binding for native function calling
            5. Returns updated state with the model's response
            
            Args:
                state (ChatState): Current conversation state
                
            Returns:
                ChatState: Updated state with agent response
            """
            # Extract the most recent user message
            last_user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            
            if not last_user_message:
                return state
            
            # Retrieve contextual information from processed files
            context = self._get_context(last_user_message)
            
            # Build system prompt with optional context
            system_content = self.instructions
            if context:
                system_content += f"\n\nContext from uploaded files:\n{context}"
            
            # Construct message history in OpenAI format
            messages = [{"role": "system", "content": system_content}]
            
            # Add conversation history with simplified message handling
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content or ""})
                elif isinstance(msg, ToolMessage):
                    # Convert tool results to user messages for context
                    messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
            
            try:
                # Invoke model with native tool binding
                response = self.model_with_tools.invoke(messages)
                
                # Return updated state
                return {
                    **state,
                    "context": context,
                    "messages": [response]  # add_messages annotation handles proper appending
                }
                
            except Exception as e:
                self.logger.error(f"Error en agent_node: {e}")
                self.logger.exception("Traceback completo:")
                # Graceful fallback for error scenarios
                fallback_response = AIMessage(content="I apologize, but I encountered an error processing your request.")
                return {
                    **state,
                    "context": context,
                    "messages": [fallback_response]
                }

        def should_continue(state: ChatState) -> str:
            """
            Conditional edge function to determine workflow continuation.
            
            Analyzes the last message to decide whether to execute tools or end the workflow.
            This leverages LangGraph's native tool calling detection.
            
            Args:
                state (ChatState): Current conversation state
                
            Returns:
                str: Next node to execute ("tools" or "end")
            """
            last_message = state["messages"][-1]
            
            # Check for pending tool calls using native tool calling detection
            if (isinstance(last_message, AIMessage) and 
                hasattr(last_message, 'tool_calls') and 
                last_message.tool_calls):
                return "tools"
            
            return "end"

        # Construct the workflow graph
        workflow = StateGraph(ChatState)
        
        # Add primary agent node
        workflow.add_node("agent", agent_node)
        
        # Add tool execution node if tools are available
        if self.tools:
            # ToolNode automatically handles tool execution and result formatting
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)
        
        # Define workflow edges and entry point
        workflow.set_entry_point("agent")
        
        if self.tools:
            # Conditional routing based on tool call presence
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "tools": "tools",
                    "end": END
                }
            )
            # Return to agent after tool execution for final response formatting
            workflow.add_edge("tools", "agent")
        else:
            # Direct termination if no tools are available
            workflow.add_edge("agent", END)
        
        # Compile workflow with optional checkpointing
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()

    # ===== LEGACY API COMPATIBILITY =====
    
    def get_response(self, user_input: str) -> ResponseModel:
        """
        Generate a response while maintaining 100% API compatibility.
        
        This method provides the primary interface for single-turn conversations,
        maintaining backward compatibility with existing ChatService implementations.
        Tool execution callbacks (if provided) will be triggered during execution.
        
        Args:
            user_input (str): The user's message or query
            
        Returns:
            ResponseModel: Structured response containing:
                - user_tokens: Input token count
                - bot_tokens: Output token count  
                - response: Generated response text
        
        Note:
            This method automatically handles tool execution and context integration
            from processed files while maintaining the original API signature.
        """
        # Prepare initial workflow state
        initial_state = {
            "messages": self.chat_history + [HumanMessage(content=user_input)],
            "context": ""
        }
        
        # Create callback handler if any callbacks are provided
        config = {}
        if self.on_tool_start or self.on_tool_end or self.on_tool_error:
            tool_logger = _InternalToolLogger(
                on_start=self.on_tool_start,
                on_end=self.on_tool_end,
                on_error=self.on_tool_error
            )
            config["callbacks"] = [tool_logger]
        
        # Execute the LangGraph workflow with callbacks
        result = asyncio.run(self.graph.ainvoke(initial_state, config=config))
        
        # Update internal conversation history
        self.chat_history = result["messages"]
        
        # Extract final response from the last assistant message
        final_response = ""
        total_input_tokens = 0
        total_output_tokens = 0
        
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break
        
        # Extract token usage from response metadata
        last_message = result["messages"][-1]
        if hasattr(last_message, 'response_metadata'):
            token_usage = last_message.response_metadata.get('token_usage', {})
            total_input_tokens = token_usage.get('prompt_tokens', 0)
            total_output_tokens = token_usage.get('completion_tokens', 0)
        
        return ResponseModel(
            user_tokens=total_input_tokens,
            bot_tokens=total_output_tokens,
            response=final_response
        )
    
    def get_response_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Generate a streaming response for real-time user interaction.
        
        This method provides streaming capabilities while maintaining backward
        compatibility with the original API. Tool execution callbacks (if provided)
        will be triggered during execution.
        
        Args:
            user_input (str): The user's message or query
            
        Yields:
            str: Response chunks as they are generated
            
        Note:
            Current implementation streams complete responses. For token-level
            streaming, consider using the model's native streaming capabilities.
        """
        initial_state = {
            "messages": self.chat_history + [HumanMessage(content=user_input)],
            "context": ""
        }
        
        # Create callback handler if any callbacks are provided
        config = {}
        if self.on_tool_start or self.on_tool_end or self.on_tool_error:
            tool_logger = _InternalToolLogger(
                on_start=self.on_tool_start,
                on_end=self.on_tool_end,
                on_error=self.on_tool_error
            )
            config["callbacks"] = [tool_logger]
        
        accumulated_response = ""
        
        # Stream workflow execution with callbacks
        for chunk in self.graph.stream(initial_state, config=config):
            # Extract content from workflow chunks
            if "agent" in chunk:
                for message in chunk["agent"]["messages"]:
                    if isinstance(message, AIMessage) and message.content:
                        # Stream complete responses (can be enhanced for token-level streaming)
                        accumulated_response = message.content
                        yield message.content
        
        # Update conversation history after streaming completion
        if accumulated_response:
            self.chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=accumulated_response)
            ])

    def load_conversation_history(self, messages: List[Message]):
        """
        Load conversation history from Django model instances.
        
        This method maintains compatibility with existing Django-based conversation
        storage while preparing the history for modern LangGraph processing.
        
        Args:
            messages (List[Message]): List of Django Message model instances
                Expected to have 'content' and 'is_bot' attributes
        """
        self.chat_history.clear()
        for message in messages:
            if message.is_bot:
                self.chat_history.append(AIMessage(content=message.content))
            else:
                self.chat_history.append(HumanMessage(content=message.content))

    def save_messages(self, user_message: str, bot_response: str):
        """
        Save messages to internal conversation history.
        
        This method provides backward compatibility for manual history management.
        
        Args:
            user_message (str): The user's input message
            bot_response (str): The bot's generated response
        """
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=bot_response))

    def process_file(self, file: FileProcessorInterface):
        """
        Process and index a file for contextual retrieval.
        
        This method maintains compatibility with existing file processing workflows
        while leveraging FAISS for efficient similarity search.
        
        Args:
            file (FileProcessorInterface): File processor instance that implements getText()
            
        Note:
            Processed files are automatically available for context retrieval
            in subsequent conversations without additional configuration.
        """
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
        """
        Clear conversation history and processed file context.
        
        This method resets the bot to a clean state, removing all conversation
        history and processed file context.
        """
        self.chat_history.clear()
        self.vector_store = None

    def get_chat_history(self) -> List[BaseMessage]:
        """
        Retrieve a copy of the current conversation history.
        
        Returns:
            List[BaseMessage]: Copy of the conversation history
        """
        return self.chat_history.copy()

    def set_chat_history(self, history: List[BaseMessage]):
        """
        Set the conversation history from a list of BaseMessage instances.
        
        Args:
            history (List[BaseMessage]): New conversation history to set
        """
        self.chat_history = history.copy()

    def _get_context(self, query: str) -> str:
        """
        Retrieve relevant context from processed files using similarity search.
        
        This method performs semantic search over processed file content to find
        the most relevant information for the current query.
        
        Args:
            query (str): The query to search for relevant context
            
        Returns:
            str: Concatenated relevant context from processed files
        """
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=4)
            return "\n".join([doc.page_content for doc in docs])
        return ""