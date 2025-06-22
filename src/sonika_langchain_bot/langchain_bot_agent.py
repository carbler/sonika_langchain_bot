from typing import Generator, List
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sonika_langchain_bot.langchain_class import FileProcessorInterface, IEmbeddings, ILanguageModel, Message, ResponseModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import BaseTool
import re

class LangChainBot:
    """
    Implementación principal del bot conversacional con capacidades de procesamiento de archivos,
    memoria de conversación y uso de herramientas personalizadas.
    """

    def __init__(self, language_model: ILanguageModel, embeddings: IEmbeddings, instructions: str, tools: List[BaseTool]):
        """
        Inicializa el bot con el modelo de lenguaje, embeddings y herramientas necesarias.

        Args:
            language_model (ILanguageModel): Modelo de lenguaje a utilizar
            embeddings (IEmbeddings): Modelo de embeddings para procesamiento de texto
            instructions (str): Instrucciones del sistema
            tools (List[BaseTool]): Lista de herramientas disponibles
        """
        self.language_model = language_model
        self.embeddings = embeddings
        # Reemplazamos ConversationBufferMemory con una lista simple de mensajes
        self.chat_history: List[BaseMessage] = []
        self.memory_agent = MemorySaver()
        self.vector_store = None
        self.tools = tools
        self.instructions = instructions
        self.add_tools_to_instructions(tools)
        self.conversation = self._create_conversation_chain()
        self.agent_executor = self._create_agent_executor()

    def add_tools_to_instructions(self, tools: List[BaseTool]):
        """Agrega información de las herramientas a las instrucciones base del sistema."""
        if len(tools) == 0:
            return
            
        # Instrucciones sobre el uso de herramientas
        tools_instructions = '''\n\nWhen you want to execute a tool, enclose the command with three asterisks and provide all parameters needed.
Ensure you gather all relevant information from the conversation to use the parameters.
If information is missing, search online.

This is a list of the tools you can execute:
'''
        
        # Procesar cada herramienta y agregarla a las instrucciones
        for tool in tools:
            tool_name = tool.name
            tool_description = tool.description
            
            tools_instructions += f"\nTool Name: {tool_name}\n"
            tools_instructions += f"Description: {tool_description}\n"
            
            # Intentar obtener información de parámetros
            run_method = getattr(tool, '_run', None)
            if run_method:
                try:
                    import inspect
                    params = inspect.signature(run_method)
                    tools_instructions += f"Parameters: {params}\n"
                except:
                    tools_instructions += "Parameters: Not available\n"
            else:
                tools_instructions += "Parameters: Not available\n"
            
            tools_instructions += "---\n"
        
        # Agregar las instrucciones de herramientas a las instrucciones base
        self.instructions += tools_instructions
        

    def _create_conversation_chain(self):
        """
        Crea la cadena de conversación con el prompt template y la memoria.
        """
        full_system_prompt = f"{self.instructions}\n\n"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(full_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # Usando RunnableSequence para reemplazar LLMChain
        return prompt | self.language_model.model 

    def _create_agent_executor(self):
        """
        Crea el ejecutor del agente con las herramientas configuradas.

        Returns:
            Agent: Agente configurado con las herramientas
        """
        return create_react_agent(self.language_model.model, self.tools, checkpointer=self.memory_agent)

    def _getInstruccionTool(self, bot_response):
        """
        Extrae las instrucciones para herramientas del texto de respuesta del bot.

        Args:
            bot_response (str): Respuesta del bot a analizar

        Returns:
            str: Instrucción extraída o cadena vacía si no se encuentra
        """
        patron = r'\*\*\*(.*?)\*\*\*'
        coincidencia = re.search(patron, bot_response)
        return coincidencia.group(1).strip() if coincidencia else ''

    def get_response(self, user_input: str) -> ResponseModel:
        """
        Genera una respuesta para la entrada del usuario, procesando el contexto y ejecutando herramientas si es necesario.

        Args:
            user_input (str): Entrada del usuario

        Returns:
            ResponseModel: Modelo de respuesta con tokens y texto
        """
        context = self._get_context(user_input)
        augmented_input = f"User question: {user_input}"
        if context:
            augmented_input = f"Context from attached files:\n{context}\n\nUser question: {user_input}"

        # Usamos el historial de chat directamente
        bot_response = self.conversation.invoke({
            "input": augmented_input, 
            "history": self.chat_history
        })

        token_usage = bot_response.response_metadata.get('token_usage', {})
        bot_response_content = bot_response.content
        
        instruction_tool = self._getInstruccionTool(bot_response_content)

        if instruction_tool:
            messages = [HumanMessage(content=instruction_tool)]
            thread_id = "abc123"
            config = {"configurable": {"thread_id": thread_id}}

            result_stream = self.agent_executor.stream(
                {"messages": messages}, config
            )

            tool_response = ""
            agent_response = ""

            for response in result_stream:
                if 'tools' in response:
                    for message in response['tools']['messages']:
                        tool_response = message.content
                if 'agent' in response:
                    for message in response['agent']['messages']:
                        agent_response = message.content

            bot_response_content = agent_response if agent_response else tool_response

        user_tokens = token_usage.get('prompt_tokens', 0)
        bot_tokens = token_usage.get('completion_tokens', 0)

        self.save_messages(user_input, bot_response_content)

        return ResponseModel(user_tokens=user_tokens, bot_tokens=bot_tokens, response=bot_response_content)
    
    def get_response_stream(self, user_input: str) -> Generator[str, None, None]:
        """
        Genera una respuesta en streaming para la entrada del usuario, procesando el contexto.

        Args:
            user_input (str): Entrada del usuario

        Yields:
            str: Fragmentos de la respuesta generada por el modelo en tiempo real
        """
        context = self._get_context(user_input)
        augmented_input = f"User question: {user_input}"
        if context:
            augmented_input = f"Context from attached files:\n{context}\n\nUser question: {user_input}"

        # Usamos el historial de chat directamente
        result_stream = self.conversation.stream({
            "input": augmented_input, 
            "history": self.chat_history
        })
        
        full_response = ""
        for response in result_stream:
            content = response.content
            full_response += content
            yield content
        
        # Guardamos los mensajes después del streaming
        self.save_messages(user_input, full_response)

    def _get_context(self, query: str) -> str:
        """
        Obtiene el contexto relevante para una consulta del almacén de vectores.

        Args:
            query (str): Consulta para buscar contexto

        Returns:
            str: Contexto encontrado o cadena vacía
        """
        if self.vector_store:
            docs = self.vector_store.similarity_search(query)
            return "\n".join([doc.page_content for doc in docs])
        return ""

    def clear_memory(self):
        """
        Limpia la memoria de conversación y el almacén de vectores.
        """
        self.chat_history.clear()
        self.vector_store = None

    def load_conversation_history(self, messages: List[Message]):
        """
        Carga el historial de conversación previo usando la estructura de mensajes simplificada.

        Args:
            messages: Lista de objetos Message que representan cada mensaje.
        """
        self.chat_history.clear()
        for message in messages:
            if message.is_bot:
                self.chat_history.append(AIMessage(content=message.content))
            else:
                self.chat_history.append(HumanMessage(content=message.content))

    def save_messages(self, user_message: str, bot_response: str):
        """
        Guarda los mensajes en el historial de conversación.

        Args:
            user_message (str): Mensaje del usuario
            bot_response (str): Respuesta del bot
        """
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=bot_response))

    def process_file(self, file: FileProcessorInterface):
        """
        Procesa un archivo y lo añade al almacén de vectores.

        Args:
            file (FileProcessorInterface): Archivo a procesar
        """
        document = file.getText()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)

        if self.vector_store is None:
            self.vector_store = FAISS.from_texts([doc.page_content for doc in texts], self.embeddings)
        else:
            self.vector_store.add_texts([doc.page_content for doc in texts])

    def get_chat_history(self) -> List[BaseMessage]:
        """
        Obtiene el historial completo de la conversación.

        Returns:
            List[BaseMessage]: Lista de mensajes de la conversación
        """
        return self.chat_history.copy()

    def set_chat_history(self, history: List[BaseMessage]):
        """
        Establece el historial de conversación.

        Args:
            history (List[BaseMessage]): Lista de mensajes a establecer
        """
        self.chat_history = history.copy()