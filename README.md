# Sonika LangChain Bot <a href="https://pepy.tech/projects/sonika-langchain-bot"><img src="https://static.pepy.tech/badge/sonika-langchain-bot" alt="PyPI Downloads"></a>

A Python library that implements a conversational agent using LangChain with tool execution capabilities and text classification.

## Installation

```bash
pip install sonika-langchain-bot
```

## Prerequisites

You'll need the following API keys depending on the model you wish to use:

- OpenAI API Key
- DeepSeek API Key (Optional)
- Google Gemini API Key (Optional)

Create a `.env` file in the root of your project with the following variables:

```env
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
GOOGLE_API_KEY=your_gemini_key_here
```

## Key Features

- **Multi-Model Support**: Seamlessly switch between OpenAI, DeepSeek, and Google Gemini models.
- **Conversational Agent**: Robust agent (`LangChainBot`) with native tool execution capabilities.
- **Structured Classification**: Text classification with strongly typed outputs.
- **Custom Tools**: Easy integration of custom tools via Pydantic and LangChain.
- **Streaming**: Full support for streaming responses.
- **History Management**: Built-in conversation history tracking.

## Basic Usage

### Agent with Tools Example

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sonika_langchain_bot.langchain_tools import EmailTool
from sonika_langchain_bot.langchain_bot_agent import LangChainBot
from sonika_langchain_bot.langchain_class import Message, ResponseModel
from sonika_langchain_bot.langchain_models import OpenAILanguageModel, DeepSeekLanguageModel, GeminiLanguageModel

# Load environment variables
load_dotenv()

# Example 1: Using OpenAI
# api_key = os.getenv("OPENAI_API_KEY")
# language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini', temperature=1)

# Example 2: Using Gemini
api_key = os.getenv("GOOGLE_API_KEY")
language_model = GeminiLanguageModel(api_key, model_name='gemini-3-flash-preview', temperature=1)

# Embeddings (usually OpenAI is used for embeddings regardless of the LLM)
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Configure tools
tools = [EmailTool()]

# Create agent instance
bot = LangChainBot(language_model, embeddings, instructions="You are an agent", tools=tools)

# Load conversation history
bot.load_conversation_history([Message(content="My name is Erley", is_bot=False)])

# Get response
user_message = 'Send an email with the tool to erley@gmail.com with subject Hello and message Hello Erley'
response_model: ResponseModel = bot.get_response(user_message)

print(response_model)
```

### Streaming Response Example

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sonika_langchain_bot.langchain_bot_agent import LangChainBot
from sonika_langchain_bot.langchain_class import Message
from sonika_langchain_bot.langchain_models import OpenAILanguageModel

# Load environment variables
load_dotenv()

# Get API key from .env file
api_key = os.getenv("OPENAI_API_KEY")

# Initialize language model and embeddings
language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
embeddings = OpenAIEmbeddings(api_key=api_key)

# Create agent instance
bot = LangChainBot(language_model, embeddings, instructions="Only answers in english", tools=[])

# Load conversation history
bot.load_conversation_history([Message(content="My name is Erley", is_bot=False)])

# Get streaming response
user_message = 'Hello, what is my name?'
for chunk in bot.get_response_stream(user_message):
    print(chunk)
```

### Text Classification Example

```python
import os
from dotenv import load_dotenv
from sonika_langchain_bot.langchain_clasificator import TextClassifier
from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Define classification structure with Pydantic
class Classification(BaseModel):
    intention: str = Field()
    sentiment: str = Field(..., enum=["happy", "neutral", "sad", "excited"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )

# Initialize classifier
api_key = os.getenv("OPENAI_API_KEY")
model = OpenAILanguageModel(api_key=api_key)
classifier = TextClassifier(llm=model, validation_class=Classification)

# Classify text
result = classifier.classify("how are you?")
print(result)
```

## Available Classes and Components

### Core Classes

- **LangChainBot**: Main conversational agent for task execution with tools
- **OpenAILanguageModel**: Wrapper for OpenAI language models
- **DeepSeekLanguageModel**: Wrapper for DeepSeek language models
- **GeminiLanguageModel**: Wrapper for Google Gemini models
- **TextClassifier**: Text classification using structured output
- **Message**: Message structure for conversation history
- **ResponseModel**: Response structure from agent interactions

### Tools

- **EmailTool**: Tool for sending emails through the agent

## Project Structure

```
your_project/
├── .env                    # Environment variables
├── src/
│   └── sonika_langchain_bot/
│       ├── langchain_bot_agent.py
│       ├── langchain_clasificator.py
│       ├── langchain_class.py
│       ├── langchain_models.py
│       └── langchain_tools.py
└── tests/
    └── test_bot.py
```

## Contributing

Contributions are welcome. Please open an issue to discuss major changes you'd like to make.

## License

This project is licensed under the MIT License.
