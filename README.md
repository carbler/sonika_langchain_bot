# Sonika LangChain Bot <a href="https://pepy.tech/projects/sonika-langchain-bot"><img src="https://static.pepy.tech/badge/sonika-langchain-bot" alt="PyPI Downloads"></a>

A Python library that implements a conversational agent using LangChain with tool execution capabilities and text classification.

## Installation

```bash
pip install sonika-langchain-bot
```

## Prerequisites

You'll need the following API keys:

- OpenAI API Key

Create a `.env` file in the root of your project with the following variables:

```env
OPENAI_API_KEY=your_api_key_here
```

## Key Features

- Conversational agent with tool execution capabilities
- Text classification with structured output
- Custom tool integration
- Streaming responses
- Conversation history management
- Flexible instruction-based behavior

## Basic Usage

### Agent with Tools Example

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from sonika_langchain_bot.langchain_tools import EmailTool
from sonika_langchain_bot.langchain_bot_agent import LangChainBot
from sonika_langchain_bot.langchain_class import Message, ResponseModel
from sonika_langchain_bot.langchain_models import OpenAILanguageModel

# Load environment variables
load_dotenv()

# Get API key from .env file
api_key = os.getenv("OPENAI_API_KEY")

# Initialize language model and embeddings
language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
embeddings = OpenAIEmbeddings(api_key=api_key)

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
