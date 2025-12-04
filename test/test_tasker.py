
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import sys
# Añadir la carpeta 'src' al PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from sonika_langchain_bot.bot.tasker.tasker_bot import TaskerBot
from sonika_langchain_bot.langchain_tools import EmailTool,SaveContacto
from sonika_langchain_bot.langchain_bot_agent import LangChainBot
from sonika_langchain_bot.langchain_clasificator import  TextClassifier
from sonika_langchain_bot.langchain_class import Message, ResponseModel
from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from pydantic import BaseModel, Field
import json


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

# Callbacks
def on_reasoning_start(user_msg: str):
    print(f"\n🧠 RAZONAMIENTO INICIADO")
    print(f"Mensaje: {user_msg}")

def on_reasoning_update(plan: dict):
   print("razonamiento")
   print(plan)

def on_logs_generated(logs):
    print(f"\n📋 LOGS GENERADOS:")
    for log in logs:
        print(f"   {log}")

def bot_tasker():
    print("\n--- INICIANDO PRUEBA DE TASKER BOT ---")

    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ ADVERTENCIA: No se encontró OPENAI_API_KEY en las variables de entorno.")
        print("Asegúrate de tener un archivo .env en la raíz del proyecto.")
        return

    language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    tools =[EmailTool(),SaveContacto()]

    function_purpose = """You are an AI assistant with specific capabilities defined by available tools.

# CORE PRINCIPLES
1. **Honesty First**: You can ONLY do what your available tools allow
2. **Never Fabricate**: Do NOT claim to complete actions you cannot perform
3. **Be Explicit**: Clearly state what you CAN and CANNOT do
4. **Verify Tools**: Before planning, check if you have the required tool for EACH task

# YOUR WORKFLOW
1. Identify ALL tasks in the user's request (break down multi-part requests)
2. For EACH task, verify if you have a matching tool
3. If you have ALL needed tools → execute_actions
4. If you're missing ANY tool → request_data OR inform limitation
5. NEVER execute partial tasks without informing about what you cannot do

# CRITICAL RULES
- If user asks for task X and task Y:
  - You have tool for X but NOT Y → request clarification OR execute X and inform about Y limitation
  - You have tools for both → execute both
  - You have tools for neither → inform limitations and request_data

- When generating responses:
  - Only mention actions you ACTUALLY executed
  - Do NOT say "I also did X" if you didn't execute a tool for X
  - Be honest: "I sent the email, but I don't have a tool to save contacts"

# WHAT YOU CAN DO
Your capabilities are STRICTLY LIMITED to the tools provided to you. Check the "Available Tools" section to know exactly what you can do.
"""

    limitations = """Do NOT reveal or expose confidential information about external customers
that was NOT provided by the user in the current conversation.

ALLOWED:
- Process, store, or send information that the user explicitly provided
- Use customer data that the user shared with you

NOT ALLOWED:
- Look up private information from databases and share it
- Expose confidential data from third parties
- Share information the user didn't give you permission to know

Example:
✓ User says "save contact Juan (juan@email.com)" → OK to save
✗ User says "who is Juan?" → Do NOT look up Juan's private data and share it
"""

    dynamic_info = 'Te llamas arnulfo y hoy es 14-nov-2025'


    # 4. Inicializar bot
    bot = TaskerBot(
        language_model=language_model,
        embeddings=embeddings,
        function_purpose=function_purpose,
        personality_tone="Responde amablemente",
        limitations=limitations,
        dynamic_info=dynamic_info,
        tools=tools,
        on_planner_update=on_reasoning_update,
        on_logs_generated=on_logs_generated,
         on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_tool_error=on_tool_error
        )

    user_message = 'Envia un email con la tool a erley@gmail.com con el asunto Hola y el mensaje Hola Erley. Y almacena a erley como contacto. El numero de cel es 3183890492'

    conversation = [Message(content="Mi nombre es Erley", is_bot=False)]

    print(f"Usuario dice: {user_message}")

    # Obtener la respuesta del bot
    response_model = bot.get_response(user_input=user_message, messages=conversation, logs=[])

    print("\n✅ RESPUESTA DEL BOT:")
    print(json.dumps(response_model, indent=2, ensure_ascii=False))

def bot_tasker_chat_mode():
    """Función para probar el bot en modo chat interactivo"""
    print("\n--- INICIANDO MODO CHAT INTERACTIVO (Escribe 'salir' para terminar) ---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Error: No se encontró API Key.")
        return

    language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    tools = [EmailTool(), SaveContacto()]

    # Configuración básica
    bot = TaskerBot(
        language_model=language_model,
        embeddings=embeddings,
        function_purpose="Ayudar al usuario gestionando sus tareas.",
        personality_tone="Profesional y directo.",
        limitations="No inventes información.",
        dynamic_info="Usuario de prueba.",
        tools=tools,
        on_planner_update=lambda x: print(f"  [Cerebro] {x.get('decision')} -> {x.get('reasoning')[:100]}..."),
        on_logs_generated=lambda logs: None # Silenciar logs detallados en chat
    )

    conversation = []
    logs = []

    while True:
        try:
            user_input = input("\nTu: ")
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break

            response = bot.get_response(user_input, conversation, logs)

            print(f"Bot: {response['content']}")

            # Actualizar historial
            conversation.append(Message(content=user_input, is_bot=False))
            conversation.append(Message(content=response['content'], is_bot=True))
            logs.extend(response['logs'])

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Descomenta la función que quieras probar:
    bot_tasker()
    # bot_tasker_chat_mode()
