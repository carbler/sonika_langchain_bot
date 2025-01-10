# Sonika LangChain Bot

Una librería Python que implementa un bot conversacional utilizando LangChain con capacidades BDI (Belief-Desire-Intention) y clasificación de texto.

## Instalación

```bash
pip install sonika-langchain-bot
```

## Requisitos previos

Necesitarás las siguientes API keys:

- OpenAI API Key

Crea un archivo `.env` en la raíz de tu proyecto con las siguientes variables:

```env
OPENAI_API_KEY=tu_api_key_aqui
```

## Características principales

- Bot conversacional con arquitectura BDI
- Clasificación de texto
- Ejecución de código personalizado por medio de tools

## Uso básico

### Ejemplo de Bot BDI

```python
from sonika_langchain_bot.langchain_bdi import Belief, BeliefType
from sonika_langchain_bot.langchain_bot_agent_bdi import LangChainBot
from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from langchain_openai import OpenAIEmbeddings

# Inicializar el modelo de lenguaje
language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini', temperature=1)
embeddings = OpenAIEmbeddings(api_key=api_key)

# Configurar herramientas propias o de terceros
search = TavilySearchResults(max_results=2, api_key=api_key_tavily)
tools = [search]

# Configurar creencias
beliefs = [
    Belief(
        content="Eres un asistente de chat",
        type=BeliefType.PERSONALITY,
        confidence=1,
        source='personality'
    )
]

# Crear instancia del bot
bot = LangChainBot(language_model, embeddings, beliefs=beliefs, tools=tools)

# Obtener respuesta
response = bot.get_response("Hola como te llamas?")

bot = LangChainBot(language_model, embeddings, beliefs=beliefs, tools=tools)

user_message = 'Hola como me llamo?'

#Cargas la conversacion previa con el bot
bot.load_conversation_history([Message(content="Mi nombre es Erley", is_bot=False)])
# Obtener la respuesta del bot
response_model: ResponseModel = bot.get_response(user_message)
bot_response = response_model

print(bot_response)

```

### Ejemplo de Clasificación de Texto

```python
from sonika_langchain_bot.langchain_clasificator import TextClassifier
from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from pydantic import BaseModel, Field

# Definir estructura de clasificación
class Classification(BaseModel):
    intention: str = Field()
    sentiment: str = Field(..., enum=["feliz", "neutral", "triste", "excitado"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["español", "ingles", "frances", "aleman", "italiano"]
    )

# Inicializar clasificador
model = OpenAILanguageModel(api_key=api_key)
classifier = TextClassifier(llm=model, validation_class=Classification)

# Clasificar texto
result = classifier.classify("Tu texto aquí")
```

## Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios importantes que te gustaría hacer.

<a href="https://pepy.tech/projects/sonika-langchain-bot"><img src="https://static.pepy.tech/badge/sonika-langchain-bot" alt="PyPI Downloads"></a>
