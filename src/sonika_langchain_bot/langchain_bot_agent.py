from typing import Generator, List, Optional, Dict, Any, TypedDict, Annotated
import asyncio
import logging
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import BaseTool
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
        - Backward compatibility with legacy APIs
        - Debug logging injection for production troubleshooting
    """

    def __init__(self, 
                 language_model: ILanguageModel, 
                 embeddings: IEmbeddings, 
                 instructions: str, 
                 tools: Optional[List[BaseTool]] = None,
                 mcp_servers: Optional[Dict[str, Any]] = None,
                 use_checkpointer: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the modern LangGraph bot with optional MCP support.

        Args:
            language_model (ILanguageModel): The language model to use for generation
            embeddings (IEmbeddings): Embedding model for file processing and context retrieval
            instructions (str): System instructions that will be modernized automatically
            tools (List[BaseTool], optional): Traditional LangChain tools to bind to the model
            mcp_servers (Dict[str, Any], optional): MCP server configurations for dynamic tool loading
            use_checkpointer (bool): Enable automatic conversation persistence using LangGraph checkpoints
            logger (logging.Logger, optional): Logger instance for debugging. If None, uses silent NullHandler
        
        Note:
            The instructions will be automatically enhanced with tool descriptions
            when tools are provided, eliminating the need for manual tool instruction formatting.
        """
        # Configure logger (silent by default if not provided)
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())
        
        self.logger.info("="*80)
        self.logger.info("🚀 Inicializando LangChainBot")
        self.logger.info("="*80)
        
        # Core components
        self.language_model = language_model
        self.embeddings = embeddings
        self.base_instructions = instructions
        
        self.logger.debug(f"📋 Instrucciones base: {len(instructions)} caracteres")
        
        # Backward compatibility attributes
        self.chat_history: List[BaseMessage] = []
        self.vector_store = None
        
        # Tool configuration
        self.tools = tools or []
        self.mcp_client = None
        
        self.logger.info(f"🔧 Herramientas iniciales: {len(self.tools)}")
        
        # Initialize MCP servers if provided
        if mcp_servers:
            self.logger.info(f"🌐 Servidores MCP detectados: {len(mcp_servers)}")
            self._initialize_mcp(mcp_servers)
        else:
            self.logger.debug("⚪ Sin servidores MCP configurados")
        
        # Configure persistence layer
        self.checkpointer = MemorySaver() if use_checkpointer else None
        self.logger.debug(f"💾 Checkpointer: {'Habilitado' if use_checkpointer else 'Deshabilitado'}")
        
        # Prepare model with bound tools for native function calling
        self.logger.info("🤖 Preparando modelo con herramientas...")
        self.model_with_tools = self._prepare_model_with_tools()
        
        # Build modern instruction set with tool descriptions
        self.logger.info("📝 Construyendo instrucciones modernas...")
        self.instructions = self._build_modern_instructions()
        self.logger.debug(f"📋 Instrucciones finales: {len(self.instructions)} caracteres")
        
        # Create the LangGraph workflow
        self.logger.info("🔄 Creando workflow de LangGraph...")
        self.graph = self._create_modern_workflow()
        
        # Legacy compatibility attributes (maintained for API compatibility)
        self.conversation = None
        self.agent_executor = None
        
        self.logger.info("✅ LangChainBot inicializado correctamente")
        self.logger.info(f"📊 Resumen: {len(self.tools)} herramientas, {len(self.chat_history)} mensajes en historial")
        self.logger.info("="*80 + "\n")

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
        self.logger.info("="*80)
        self.logger.info("🌐 INICIALIZANDO MCP (Model Context Protocol)")
        self.logger.info("="*80)
        
        try:
            self.logger.info(f"📋 Servidores a inicializar: {len(mcp_servers)}")
            
            for server_name, server_config in mcp_servers.items():
                self.logger.info(f"\n🔌 Servidor: {server_name}")
                self.logger.debug(f"   Command: {server_config.get('command')}")
                self.logger.debug(f"   Args: {server_config.get('args')}")
                self.logger.debug(f"   Transport: {server_config.get('transport')}")
            
            self.logger.info("\n🔄 Creando MultiServerMCPClient...")
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            self.logger.info("✅ MultiServerMCPClient creado")
            
            # ===== FIX PARA APACHE/MOD_WSGI =====
            self.logger.info("🔧 Aplicando fix para compatibilidad Apache/mod_wsgi...")
            
            import subprocess
            original_create = asyncio.create_subprocess_exec
            
            async def fixed_create(*args, stdin=None, stdout=None, stderr=None, **kwargs):
                """Forzar PIPE para evitar heredar sys.stderr de Apache"""
                return await original_create(
                    *args,
                    stdin=stdin or subprocess.PIPE,
                    stdout=stdout or subprocess.PIPE,
                    stderr=stderr or subprocess.PIPE,
                    **kwargs
                )
            
            # Aplicar parche temporalmente
            asyncio.create_subprocess_exec = fixed_create
            self.logger.debug("✅ Parche temporal aplicado a asyncio.create_subprocess_exec")
            
            try:
                self.logger.info("🔄 Obteniendo herramientas desde servidores MCP...")
                mcp_tools = asyncio.run(self.mcp_client.get_tools())
                self.logger.info(f"📥 Herramientas MCP recibidas: {len(mcp_tools)}")
            finally:
                # Restaurar original
                asyncio.create_subprocess_exec = original_create
                self.logger.debug("✅ Parche temporal removido, asyncio restaurado")
            # =====================================
            
            if mcp_tools:
                for i, tool in enumerate(mcp_tools, 1):
                    tool_name = getattr(tool, 'name', 'Unknown')
                    tool_desc = getattr(tool, 'description', 'Sin descripción')
                    self.logger.debug(f"   {i}. {tool_name}: {tool_desc[:100]}...")
            
            self.tools.extend(mcp_tools)
            
            self.logger.info(f"✅ MCP inicializado exitosamente")
            self.logger.info(f"📊 Total herramientas disponibles: {len(self.tools)}")
            self.logger.info(f"   - Herramientas MCP: {len(mcp_tools)}")
            self.logger.info(f"   - Herramientas previas: {len(self.tools) - len(mcp_tools)}")
            self.logger.info("="*80 + "\n")
            
        except Exception as e:
            self.logger.error("="*80)
            self.logger.error("❌ ERROR EN INICIALIZACIÓN MCP")
            self.logger.error("="*80)
            self.logger.error(f"Tipo de error: {type(e).__name__}")
            self.logger.error(f"Mensaje: {str(e)}")
            self.logger.exception("Traceback completo:")
            self.logger.error("="*80 + "\n")
            
            self.mcp_client = None
            
            # Mensaje de diagnóstico
            self.logger.warning("⚠️ Continuando sin MCP - solo herramientas locales disponibles")
            self.logger.warning(f"   Herramientas disponibles: {len(self.tools)}")

    def _prepare_model_with_tools(self):
        """
        Prepare the language model with bound tools for native function calling.
        
        This method binds all available tools (both traditional and MCP) to the language model,
        enabling native function calling without manual parsing or instruction formatting.
        
        Returns:
            The language model with tools bound, or the original model if no tools are available
        """
        if self.tools:
            self.logger.info(f"🔗 Vinculando {len(self.tools)} herramientas al modelo")
            try:
                bound_model = self.language_model.model.bind_tools(self.tools)
                self.logger.info("✅ Herramientas vinculadas correctamente")
                return bound_model
            except Exception as e:
                self.logger.error(f"❌ Error vinculando herramientas: {e}")
                self.logger.exception("Traceback:")
                return self.language_model.model
        else:
            self.logger.debug("⚪ Sin herramientas para vincular, usando modelo base")
            return self.language_model.model

    def _build_modern_instructions(self) -> str:
        """
        Build modern instructions with automatic tool documentation.
        
        Returns:
            str: Enhanced instructions with tool descriptions
        """
        instructions = self.base_instructions
        
        if self.tools:
            self.logger.info(f"📝 Generando documentación para {len(self.tools)} herramientas")
            
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
                
                # Opción 2: args_schema es un dict (MCP Tools)
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
            self.logger.info(f"✅ Documentación de herramientas agregada ({len(tools_description)} caracteres)")
        
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
        self.logger.info("🔄 Construyendo workflow de LangGraph")
        
        def agent_node(state: ChatState) -> ChatState:
            """
            Main agent node responsible for generating responses and initiating tool calls.
            """
            self.logger.debug("🤖 Ejecutando agent_node")
            
            # Extract the most recent user message
            last_user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            
            if not last_user_message:
                self.logger.warning("⚠️ No se encontró mensaje de usuario")
                return state
            
            self.logger.debug(f"💬 Mensaje usuario: {last_user_message[:100]}...")
            
            # Retrieve contextual information from processed files
            context = self._get_context(last_user_message)
            if context:
                self.logger.debug(f"📚 Contexto recuperado: {len(context)} caracteres")
            
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
                    messages.append({"role": "user", "content": f"Tool result: {msg.content}"})
            
            self.logger.debug(f"📨 Enviando {len(messages)} mensajes al modelo")
            
            try:
                # Invoke model with native tool binding
                response = self.model_with_tools.invoke(messages)
                
                self.logger.debug(f"✅ Respuesta recibida del modelo")
                
                # Check for tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    self.logger.info(f"🔧 Llamadas a herramientas detectadas: {len(response.tool_calls)}")
                    for i, tc in enumerate(response.tool_calls, 1):
                        tool_name = tc.get('name', 'Unknown')
                        self.logger.debug(f"   {i}. {tool_name}")
                
                # Return updated state
                return {
                    **state,
                    "context": context,
                    "messages": [response]
                }
                
            except Exception as e:
                self.logger.error(f"❌ Error en agent_node: {e}")
                self.logger.exception("Traceback:")
                fallback_response = AIMessage(content="I apologize, but I encountered an error processing your request.")
                return {
                    **state,
                    "context": context,
                    "messages": [fallback_response]
                }

        def should_continue(state: ChatState) -> str:
            """
            Conditional edge function to determine workflow continuation.
            """
            last_message = state["messages"][-1]
            
            if (isinstance(last_message, AIMessage) and 
                hasattr(last_message, 'tool_calls') and 
                last_message.tool_calls):
                self.logger.debug("➡️ Continuando a ejecución de herramientas")
                return "tools"
            
            self.logger.debug("🏁 Finalizando workflow")
            return "end"

        # Construct the workflow graph
        workflow = StateGraph(ChatState)
        
        # Add primary agent node
        workflow.add_node("agent", agent_node)
        self.logger.debug("✅ Nodo 'agent' agregado")
        
        # Add tool execution node if tools are available
        if self.tools:
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)
            self.logger.debug("✅ Nodo 'tools' agregado")
        
        # Define workflow edges and entry point
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
            self.logger.debug("✅ Edges condicionales configurados")
        else:
            workflow.add_edge("agent", END)
            self.logger.debug("✅ Edge directo a END configurado")
        
        # Compile workflow with optional checkpointing
        if self.checkpointer:
            compiled = workflow.compile(checkpointer=self.checkpointer)
            self.logger.info("✅ Workflow compilado con checkpointer")
        else:
            compiled = workflow.compile()
            self.logger.info("✅ Workflow compilado sin checkpointer")
        
        return compiled

    # ===== LEGACY API COMPATIBILITY =====
    
    def get_response(self, user_input: str) -> ResponseModel:
        """
        Generate a response while maintaining 100% API compatibility.
        
        This method provides the primary interface for single-turn conversations,
        maintaining backward compatibility with existing ChatService implementations.
        
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
        self.logger.info("="*80)
        self.logger.info("📨 GET_RESPONSE llamado")
        self.logger.debug(f"💬 Input: {user_input[:200]}...")
        
        # Prepare initial workflow state
        initial_state = {
            "messages": self.chat_history + [HumanMessage(content=user_input)],
            "context": ""
        }
        
        self.logger.debug(f"📊 Estado inicial: {len(initial_state['messages'])} mensajes")
        
        try:
            # Execute the LangGraph workflow
            self.logger.info("🔄 Ejecutando workflow...")
            result = asyncio.run(self.graph.ainvoke(initial_state))
            self.logger.info("✅ Workflow completado")
            
            # Update internal conversation history
            self.chat_history = result["messages"]
            self.logger.debug(f"💾 Historial actualizado: {len(self.chat_history)} mensajes")
            
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
            
            self.logger.info(f"📊 Tokens: input={total_input_tokens}, output={total_output_tokens}")
            self.logger.info(f"📝 Respuesta: {len(final_response)} caracteres")
            self.logger.info("="*80 + "\n")
            
            return ResponseModel(
                user_tokens=total_input_tokens,
                bot_tokens=total_output_tokens,
                response=final_response
            )
            
        except Exception as e:
            self.logger.error("="*80)
            self.logger.error("❌ ERROR EN GET_RESPONSE")
            self.logger.error(f"Mensaje: {str(e)}")
            self.logger.exception("Traceback:")
            self.logger.error("="*80 + "\n")
            raise
    
    def get_response_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Generate a streaming response for real-time user interaction.
        """
        self.logger.info("📨 GET_RESPONSE_STREAM llamado")
        self.logger.debug(f"💬 Input: {user_input[:200]}...")
        
        initial_state = {
            "messages": self.chat_history + [HumanMessage(content=user_input)],
            "context": ""
        }
        
        accumulated_response = ""
        
        try:
            for chunk in self.graph.stream(initial_state):
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
                
            self.logger.info(f"✅ Stream completado: {len(accumulated_response)} caracteres")
            
        except Exception as e:
            self.logger.error(f"❌ Error en stream: {e}")
            self.logger.exception("Traceback:")
            raise

    def load_conversation_history(self, messages: List[Message]):
        """
        Load conversation history from Django model instances.
        """
        self.logger.info(f"📥 Cargando historial: {len(messages)} mensajes")
        self.chat_history.clear()
        for message in messages:
            if message.is_bot:
                self.chat_history.append(AIMessage(content=message.content))
            else:
                self.chat_history.append(HumanMessage(content=message.content))
        self.logger.debug("✅ Historial cargado")

    def save_messages(self, user_message: str, bot_response: str):
        """
        Save messages to internal conversation history.
        """
        self.logger.debug("💾 Guardando mensajes en historial interno")
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=bot_response))

    def process_file(self, file: FileProcessorInterface):
        """
        Process and index a file for contextual retrieval.
        """
        self.logger.info("📄 Procesando archivo para indexación")
        try:
            document = file.getText()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(document)
            
            self.logger.debug(f"✂️ Documento dividido en {len(texts)} chunks")

            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    [doc.page_content for doc in texts], 
                    self.embeddings
                )
                self.logger.info("✅ Vector store creado")
            else:
                self.vector_store.add_texts([doc.page_content for doc in texts])
                self.logger.info("✅ Textos agregados a vector store existente")
                
        except Exception as e:
            self.logger.error(f"❌ Error procesando archivo: {e}")
            self.logger.exception("Traceback:")
            raise

    def clear_memory(self):
        """
        Clear conversation history and processed file context.
        """
        self.logger.info("🗑️ Limpiando memoria")
        self.chat_history.clear()
        self.vector_store = None
        self.logger.debug("✅ Memoria limpiada")

    def get_chat_history(self) -> List[BaseMessage]:
        """
        Retrieve a copy of the current conversation history.
        """
        return self.chat_history.copy()

    def set_chat_history(self, history: List[BaseMessage]):
        """
        Set the conversation history from a list of BaseMessage instances.
        """
        self.logger.info(f"📝 Estableciendo historial: {len(history)} mensajes")
        self.chat_history = history.copy()

    def _get_context(self, query: str) -> str:
        """
        Retrieve relevant context from processed files using similarity search.
        """
        if self.vector_store:
            self.logger.debug(f"🔍 Buscando contexto para query: {query[:100]}...")
            docs = self.vector_store.similarity_search(query, k=4)
            context = "\n".join([doc.page_content for doc in docs])
            self.logger.debug(f"✅ Contexto encontrado: {len(context)} caracteres")
            return context
        return ""