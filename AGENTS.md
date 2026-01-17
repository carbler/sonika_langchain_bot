# AGENTS.md

Este archivo describe el proyecto para AI assistants (opencode, Claude Code, etc.).

## Project Overview

**sonika-langchain-bot** es una librería Python que implementa agentes conversacionales usando LangChain con capacidades de ejecución de herramientas y clasificación de texto.

## Key Components

### Core Bots
1. **LangChainBot** (59% score) - Agente conversacional principal con ejecución de herramientas
2. **TaskerBot** (52% score) - Bot con patrón ReAct mejorado y arquitectura robusta

### Supporting Modules
- `langchain_models.py`: Wrapper para modelos de lenguaje OpenAI
- `langchain_class.py`: Estructuras de datos (Message, ResponseModel, etc.)
- `langchain_tools.py`: Herramientas disponibles (EmailTool, etc.)
- `langchain_clasificator.py`: Clasificador de texto con salida estructurada
- `document_processor.py`: Procesador de documentos (PDF, DOCX, etc.)

## Project Structure

```
sonika-langchain-bot/
├── src/sonika_langchain_bot/
│   ├── langchain_bot_agent.py     # LangChainBot
│   ├── langchain_models.py        # Model wrapper
│   ├── langchain_class.py         # Data classes
│   ├── langchain_tools.py         # Tools
│   ├── langchain_clasificator.py  # Text classifier
│   ├── document_processor.py      # Document processor
│   ├── tasker/                    # TaskerBot implementation
│   │   ├── tasker_bot.py          # Main TaskerBot class
│   │   ├── state.py              # State definitions
│   │   ├── nodes/                # Graph nodes
│   │   └── prompts/              # System prompts
│   └── __init__.py
├── test/                          # Unit tests
├── test_ultimate/                 # Comprehensive stress tests
│   ├── banking_operations/        # Banking domain tests
│   └── reports/                   # Test reports
├── setup.py                       # Package configuration
├── requirements.txt               # Dependencies
├── README.md                      # User documentation
└── AGENTS.md                      # This file
```

## Development Commands

### Installation
```bash
pip install -e .
```

### Running Tests
```bash
# Unit tests
python -m pytest test/

# Stress tests (LangChainBot + TaskerBot)
cd test_ultimate/banking_operations
python batch_runner.py
```

### Code Quality
```bash
# Type checking (if using mypy)
mypy src/

# Linting (if using flake8/black)
black src/
flake8 src/
```

## Key Dependencies

- `langchain-core`, `langchain-openai`: Core LangChain functionality
- `langgraph`: Graph-based agent workflows
- `pydantic`: Data validation
- `faiss-cpu`: Vector similarity search
- `python-dotenv`: Environment variables

## How the Bots Work

### LangChainBot
- Usa `langgraph` para flujos de trabajo con herramientas
- Soporta streaming de respuestas
- Maneja historial de conversación
- Incluye callbacks para monitoreo de herramientas

### TaskerBot
- Arquitectura Planner → Executor → Output → Validator
- Patrón ReAct mejorado con límites de recursión configurables
- Nodos separados para planificación, ejecución y validación
- Integración con MCP (Model Context Protocol)

## Common Tasks for AI Assistants

### Adding New Tools
1. Crear clase en `langchain_tools.py` heredando de `BaseTool`
2. Implementar `_run` method con lógica
3. Actualizar documentación en README.md

### Modifying Bot Behavior
- LangChainBot: ajustar `instructions` parámetro
- TaskerBot: modificar prompts en `tasker/prompts/`

### Running Performance Tests
1. Asegurar `OPENAI_API_KEY` en `.env`
2. Ejecutar `test_ultimate/banking_operations/batch_runner.py`
3. Revisar reportes en `test_ultimate/reports/`

## Troubleshooting

### TaskerBot Recursion Limit Error
Si TaskerBot falla con "recursion limit reached":
1. Incrementar `recursion_limit` en instanciación
2. Ajustar `max_iterations` en `tasker_bot.py`
3. Verificar que los tools retornen correctamente

### Import Errors
Asegurar que todas las dependencias estén instaladas:
```bash
pip install -r requirements.txt
```

## Contributing

1. Sigue patrones existentes en código
2. Ejecuta tests antes de commits
3. Actualiza AGENTS.md si cambia estructura del proyecto
4. Mantén compatibilidad con los dos bots principales

## Contact

- Autor: Erley Blanco Carvajal
- Licencia: MIT