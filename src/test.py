import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from langchain_bot.langchain_bdi import Belief, BeliefType
from langchain_bot.langchain_bot_agent_bdi import LangChainBot
from langchain_bot.langchain_clasificator import OpenAIModel, TextClassifier
from langchain_bot.langchain_class import ResponseModel
from langchain_bot.langchain_models import OpenAILanguageModel
from langchain_bot.langchain_tools import EmailTool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

def bot_bdi():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_tavily = os.getenv("TAVILY_API_KEY")

    language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Configuración de herramientas y bots
    search = TavilySearchResults(max_results=2, api_key=api_key_tavily)
    email_tool = EmailTool()
    
    tools =[search, email_tool]
    beliefs = [Belief(content="Eres un asistente de chat", type=BeliefType.PERSONALITY, confidence=1, source='personality')]
    bot = LangChainBot(language_model, embeddings, beliefs=beliefs, tools=tools)

    user_message = 'Hola como te llamas?'
    # Obtener la respuesta del bot
    response_model: ResponseModel = bot.get_response(user_message)
    bot_response = response_model.response

    print(bot_response)

def bot():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_tavily = os.getenv("TAVILY_API_KEY")

    language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    # Configuración de herramientas y bots
    search = TavilySearchResults(max_results=2, api_key=api_key_tavily)
    email_tool = EmailTool()
    
    tools =[search, email_tool]
    bot = LangChainBot(language_model, embeddings, instructions="Eres un bot de telegram", tools=tools)

    user_message = 'Hola como te llamas?'
    # Obtener la respuesta del bot
    response_model: ResponseModel = bot.get_response(user_message)
    bot_response = response_model.response

    print(bot_response)

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
    model = OpenAIModel(api_key=api_key,validation_class=Classification)
    classifier = TextClassifier(api_key=api_key,llm=model, validation_class=Classification)
    result = classifier.classify("venga, quiero que vengas a mi casa y nos tomamos un vino tu y yo solos, en mi cuarto sin ropa, que dices")
    print(result)

bot_bdi()
#bot()
#clasification()