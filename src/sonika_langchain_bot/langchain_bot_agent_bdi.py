from typing import Generator
from typing import List
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sonika_langchain_bot.langchain_bdi import Belief, BotBeliefSystem
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

    def __init__(self, language_model: ILanguageModel, embeddings: IEmbeddings, beliefs: List[Belief], tools: List[BaseTool]):
        """
        Inicializa el bot con el modelo de lenguaje, embeddings y herramientas necesarias.

        Args:
            language_model (ILanguageModel): Modelo de lenguaje a utilizar
            embeddings (IEmbeddings): Modelo de embeddings para procesamiento de texto
            beliefs (List[Belief]): Lista de creencias iniciales
            tools (List[BaseTool]): Lista de herramientas disponibles
        """
        self.language_model = language_model
        self.embeddings = embeddings
        self.memory = ConversationBufferMemory(return_messages=True)
        self.memory_agent = MemorySaver()
        self.vector_store = None
        self.tools = tools
        self.beliefs = beliefs
        self.belief_system = BotBeliefSystem('Hal9000', '1.0.0', beliefs_init=beliefs, tools=tools)
        self.conversation = self._create_conversation_chain()
        self.agent_executor = self._create_agent_executor()

    def _create_conversation_chain(self):
        """
        Crea la cadena de conversación con el prompt template y la memoria.
        Ahora incluye el contexto del sistema de creencias.
        """
        beliefs_context = self.belief_system.generate_prompt_context()
        full_system_prompt = f"{beliefs_context}\n\n"
        print(full_system_prompt)

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

        print(augmented_input)

        bot_response = self.conversation.invoke({"input": augmented_input, "history": self.memory.chat_memory.messages})

        token_usage= bot_response.response_metadata['token_usage']
        bot_response = bot_response.content
        
        instruction_tool = self._getInstruccionTool(bot_response)

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

            bot_response = agent_response if agent_response else tool_response

        user_tokens = token_usage['prompt_tokens']
        bot_tokens = token_usage['completion_tokens']

        self.save_messages(user_input, bot_response)

        return ResponseModel(user_tokens=user_tokens, bot_tokens=bot_tokens, response=bot_response)
    
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

        result_stream = self.conversation.stream({"input": augmented_input, "history": self.memory.chat_memory.messages})
        for response in result_stream:
            yield response.content


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
        self.memory.clear()
        self.vector_store = None

    def load_conversation_history(self, messages: List[Message]):
        """
        Carga el historial de conversación previo usando la estructura de mensajes simplificada.

        Args:
            messages: Lista de objetos Message que representan cada mensaje.
        """
        for message in messages:
            if message.is_bot:
                self.memory.chat_memory.add_message(AIMessage(content=message.content))
            else:
                self.memory.chat_memory.add_message(HumanMessage(content=message.content))

    def save_messages(self, user_message: str, bot_response: str):
        """
        Guarda los mensajes en la memoria de conversación.

        Args:
            user_message (str): Mensaje del usuario
            bot_response (str): Respuesta del bot
        """
        self.memory.chat_memory.add_message(HumanMessage(content=user_message))
        self.memory.chat_memory.add_message(AIMessage(content=bot_response))

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
