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
language_model = OpenAILanguageModel(api_key, model_name='gpt-4-mini-2024-07-18', temperature=1)
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
```

### Ejemplo de Clasificación de Texto

```python
from sonika_langchain_bot.langchain_clasificator import OpenAIModel, TextClassifier
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
model = OpenAIModel(api_key=api_key, validation_class=Classification)
classifier = TextClassifier(api_key=api_key, llm=model, validation_class=Classification)

# Clasificar texto
result = classifier.classify("Tu texto aquí")
```

## Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir los cambios importantes que te gustaría hacer.
