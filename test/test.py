import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import sys
# Añadir la carpeta 'src' al PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from sonika_langchain_bot.langchain_tools import EmailTool
from sonika_langchain_bot.langchain_bot_agent import LangChainBot
from sonika_langchain_bot.langchain_clasificator import  TextClassifier
from sonika_langchain_bot.langchain_class import Message, ResponseModel
from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from pydantic import BaseModel, Field


env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

 # Ejemplo 1: Con callbacks
def on_tool_start(tool_name: str, input_data: str):
    print(f"🔧 Ejecutando tool: {tool_name}")
    print(f"   Input: {input_data}")
def on_tool_end(tool_name: str, output: str):
    print(f"✅ Tool completada: {tool_name}")
    print(f"   Output: {output[:100]}...")  # Primeros 100 chars

def on_tool_error(tool_name: str, error: str):
    print(f"❌ Tool falló: {tool_name}")
    print(f"   Error: {error}")


def bot_bdi():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("OPENAI_API_KEY")

    language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    tools =[EmailTool()]
    bot = LangChainBot(
        language_model, 
        embeddings,  
        instructions="Eres un agente" ,
        tools=tools, 
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_tool_error=on_tool_error)

    user_message = 'Envia un email con la tool a erley@gmail.com con el asunto Hola y el mensaje Hola Erley'

    bot.load_conversation_history([Message(content="Mi nombre es Erley", is_bot=False)])
    # Obtener la respuesta del bot
    response_model: ResponseModel = bot.get_response(user_message)
    bot_response = response_model

    print("Bot response",bot_response)



def bot_bdi_streaming():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_tavily = os.getenv("TAVILY_API_KEY")

    language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Configuración de herramientas y bots    
    tools =[]
    bot = LangChainBot(language_model, embeddings, instructions="Only answers in english", tools=tools)

    user_message = 'Hola como me llamo?'

    bot.load_conversation_history([Message(content="Mi nombre es Erley", is_bot=False)])
    for chunk in bot.get_response_stream(user_message):
        print(chunk)

    

# Definir la clase 'Classification' con Pydantic para validar la estructura
class Classification(BaseModel):
    intention: str = Field()
    sentiment: str = Field(..., enum=["feliz", "neutral", "triste", "excitado"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["español", "ingles", "frances", "aleman", "italiano"]
    )

def clasification():
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAILanguageModel(api_key=api_key)
    classifier = TextClassifier(llm=model, validation_class=Classification)
    result = classifier.classify("how are you?")
    print(result)

bot_bdi()
#bot_bdi_streaming()
#clasification()