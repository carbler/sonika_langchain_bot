from typing import List
import tiktoken
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from apps.aichat.langchain_class import FileProcessorInterface, IEmbeddings, ILanguageModel, ResponseModel
from langchain.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import BaseTool
import inspect
import re


class OpenAILanguageModel(ILanguageModel):
    """
    Clase que implementa la interfaz ILanguageModel para interactuar con los modelos de lenguaje de OpenAI.
    Proporciona funcionalidades para generar respuestas y contar tokens.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Inicializa el modelo de lenguaje de OpenAI.
        
        Args:
            api_key (str): Clave API de OpenAI
            model_name (str): Nombre del modelo a utilizar
            temperature (float): Temperatura para la generación de respuestas
        """
        self.model = ChatOpenAI(temperature=temperature, model_name=model_name, api_key=api_key)
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def get_response(self, prompt: str) -> str:
        """
        Genera una respuesta basada en el prompt proporcionado.
        
        Args:
            prompt (str): Texto de entrada para generar la respuesta
            
        Returns:
            str: Respuesta generada por el modelo
        """
        return self.model.predict(prompt)

    def count_tokens(self, text: str) -> int:
        """
        Cuenta el número de tokens en un texto dado.
        
        Args:
            text (str): Texto para contar tokens
            
        Returns:
            int: Número de tokens en el texto
        """
        return len(self.tokenizer.encode(text))


class LangChainBot:
    """
    Implementación principal del bot conversacional con capacidades de procesamiento de archivos,
    memoria de conversación y uso de herramientas personalizadas.
    """

    def __init__(self, language_model: ILanguageModel, embeddings: IEmbeddings, instructions: List[str], tools: List[BaseTool]):
        """
        Inicializa el bot con el modelo de lenguaje, embeddings y herramientas necesarias.
        
        Args:
            language_model (ILanguageModel): Modelo de lenguaje a utilizar
            embeddings (IEmbeddings): Modelo de embeddings para procesamiento de texto
            instructions (List[str]): Lista de instrucciones para el bot
            tools (List[BaseTool]): Lista de herramientas disponibles
        """
        self.language_model = language_model
        self.embeddings = embeddings
        self.memory = ConversationBufferMemory(return_messages=True)
        self.memory_agent = MemorySaver()
        self.vector_store = None
        self.instructions = instructions
        self.tools = tools
        self.loadTools(self.tools)
        self.conversation = self._create_conversation_chain()
        self.agent_executor = self._create_agent_executor()

    def loadTools(self, tools: List[BaseTool]):
        """
        Carga y procesa las herramientas disponibles para el bot.
        
        Args:
            tools (List[BaseTool]): Lista de herramientas a cargar
        """
        text_tools = ""
        for tool in tools:
            tool_name = tool.name
            tool_description = tool.description
            text_tools += f"Tool Name: {tool_name}\nDescription: {tool_description}\n"
            run_method = getattr(tool, '_run', None)
            
            if run_method:
                params = inspect.signature(run_method)
                text_tools += f"Parameters: {params}\n"
            else:
                text_tools += "No _run method found.\n"
            text_tools += "\n---\n"

        encabezado = "\nEste es un listado de los tools que puedes ejecutar : "
        instrucciones_de_uso = '''\n En el momento que desees ejecutar un tool como estos, abre 3 asteriscos y genera un orden escrita con todo los los parametros
                                    \n Debes ser capaz de obtener todos los datos de la conversacion para usar los parametros
                                   \n  En el caso de que no poseas una información siempre debes buscar en internet '''
        self.instructions = self.instructions + encabezado + text_tools + instrucciones_de_uso

    def _create_conversation_chain(self):
        """
        Crea la cadena de conversación con el prompt template y la memoria.
        
        Returns:
            LLMChain: Cadena de conversación configurada
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.instructions),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return LLMChain(llm=self.language_model.model, prompt=prompt, memory=self.memory)

    def _create_agent_executor(self):
        """
        Crea el ejecutor del agente con las herramientas configuradas.
        
        Returns:
            Agent: Agente configurado con las herramientas
        """
        return create_react_agent(self.language_model.model, self.tools, checkpointer=self.memory_agent)
    
    def getInstruccionTool(self, bot_response):
        """
        Extrae las instrucciones para herramientas del texto de respuesta del bot.
        
        Args:
            bot_response (str): Respuesta del bot a analizar
            
        Returns:
            str: Instrucción extraída o cadena vacía si no se encuentra
        """
        patron = r'\*\*\*(.*?)\*\*\*'
        coincidencia = re.search(patron, bot_response)
        if coincidencia:
            return coincidencia.group(1).strip()
        else:
            return ''

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
        if(context):
            augmented_input = f"Context from attached files:\n{context}\n\nUser question: {user_input}"
        
        bot_response = self.conversation.predict(input=augmented_input)
        instruction_tool = self.getInstruccionTool(bot_response)

        print(instruction_tool)

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
                     
        user_tokens = self.language_model.count_tokens(augmented_input)
        bot_tokens = self.language_model.count_tokens(bot_response)

        self.save_messages(user_input,bot_response)

        return ResponseModel(user_tokens=user_tokens, bot_tokens=bot_tokens, response=bot_response)

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

    def load_conversation_history(self, messages):
        """
        Carga el historial de conversación previo.
        
        Args:
            messages: Lista de mensajes a cargar
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
    
    def get_total_tokens(self):
        """
        Obtiene el total de tokens utilizados en la conversación.
        
        Returns:
            int: Total de tokens
        """
        return self.language_model.count_tokens(self.memory.chat_memory.get_messages())