# AI Agents Documentation (`AGENTS.md`)

This document serves as the primary knowledge base for AI Agents (such as OpenDevin, Claude Code, GitHub Copilot Workspace, etc.) working on this repository. It defines the project context, architecture, available skills, and development standards.

## 🧠 Project Context & Mission

**sonika-langchain-bot** is a robust Python library designed to build state-of-the-art conversational agents. It leverages `LangChain` and `LangGraph` to create autonomous bots capable of:
1.  **Complex Reasoning**: Using ReAct patterns and graph-based workflows.
2.  **Tool Execution**: Interacting with external systems (Email, CRM, etc.) via structured tool definitions.
3.  **Multi-Model Support**: Agnostic integration with **OpenAI**, **DeepSeek**, and **Google Gemini**.

The goal is to provide a standardized, scalable framework for banking and customer service bots that is easy to extend and stress-test.

---

## 🤖 Bot Architectures

The project features two primary bot implementations:

### 1. `LangChainBot` (Standard Agent)
*   **Path**: `src/sonika_langchain_bot/langchain_bot_agent.py`
*   **Architecture**: Uses `LangGraph` state graph (`agent` -> `tools` -> `agent`).
*   **Features**:
    *   Streaming response support.
    *   Native tool calling handling.
    *   Robust error handling with meta-prompt injection for model compatibility (e.g., Gemini).
    *   Comprehensive token usage tracking.

### 2. `TaskerBot` (Advanced Planner)
*   **Path**: `src/sonika_langchain_bot/tasker/`
*   **Architecture**: Enhanced ReAct pattern with explicit `Planner`, `Executor`, and `Validator` nodes.
*   **Features**:
    *   Iterative problem solving with recursion limits.
    *   Separation of concerns between planning and acting.
    *   Ideal for complex, multi-step tasks.

---

## 🛠 Skills & Tools

Agents working on this repo should be aware of the "Skills" (Tools) available to the bots. These are defined in `src/sonika_langchain_bot/langchain_tools.py` and other modules.

### Core Skills
| Skill / Tool Name | Class Name | Description | Inputs |
| :--- | :--- | :--- | :--- |
| **Email Sender** | `EmailTool` | Sends emails to users. | `to_email` (str), `subject` (str), `message` (str) |
| **Contact Saver** | `SaveContact` | Saves/Updates contact info in CRM. | `nombre` (str), `correo` (str), `telefono` (str) |

### Banking Domain Skills (in Stress Tests)
*Located in `test_ultimate/banking_operations/tools.py`*
*   `GetUserProfile`
*   `TransactionTool`
*   `CreateTicket`
*   `BlockAccountTool`
*   `RefundTool`
*   `CheckFraudScore`
*   etc.

---

## 🌐 Supported Models

This project implements a unified `ILanguageModel` interface to support multiple providers.

| Provider | Class Name | Config File | Env Variable |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `OpenAILanguageModel` | `langchain_models.py` | `OPENAI_API_KEY` |
| **DeepSeek** | `DeepSeekLanguageModel` | `langchain_models.py` | `DEEPSEEK_API_KEY` |
| **Google Gemini** | `GeminiLanguageModel` | `langchain_models.py` | `GOOGLE_API_KEY` |
| **Amazon Bedrock** | `BedrockLanguageModel` | `langchain_models.py` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |

> **Note for Agents**: When implementing new features, ensure compatibility with ALL supported providers. Gemini, in particular, has strict requirements regarding system message placement (see `LangChainBot` implementation details).

---

## 💻 Development Standards

### 1. Code Style
*   **Python**: Follow PEP 8.
*   **Typing**: Use type hints (`typing` module) for all function signatures.
*   **Docstrings**: All classes and public methods must have docstrings describing args and returns.

### 2. Testing Strategy
*   **Unit Tests**: Located in `test/`. Run with `python test/test.py` or `pytest`.
*   **Stress Tests**: Located in `test_ultimate/`.
    *   Use `UltimateStressTestRunner` in `test_ultimate/banking_operations/stress_test_runner.py`.
    *   Configure batches in `test_ultimate/banking_operations/batch_runner.py`.
    *   **Mandatory**: Run stress tests before submitting core changes to bot logic.

### 3. Environment Setup
Create a `.env` file in the root:
```env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

### 4. Workflow for AI Agents
1.  **Read Context**: Always read `AGENTS.md` and `README.md` first.
2.  **Plan**: create a step-by-step plan using `set_plan`.
3.  **Implement**: Write code, ensuring multi-model support.
4.  **Verify**: Run `test/test.py` for quick checks and `batch_runner.py` for regression.
5.  **Reflect**: Update memory or documentation if new patterns are discovered.

---

## 📂 Project Structure Map

*   `src/sonika_langchain_bot/`: Library source code.
    *   `langchain_bot_agent.py`: **Main Bot Logic**.
    *   `langchain_models.py`: **LLM Wrappers** (OpenAI, DeepSeek, Gemini).
    *   `langchain_tools.py`: **Tool Definitions**.
*   `test/`: Quick functional tests.
*   `test_ultimate/`: Advanced stress testing framework.
