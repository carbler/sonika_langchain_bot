"""Test de integración para el Planner Bot."""

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sonika_langchain_bot.planner import PlannerBot
from sonika_langchain_bot.langchain_tools import EmailTool, SaveContacto
from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from langchain_openai import OpenAIEmbeddings

# Configuración de logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_planner")
logger.setLevel(logging.INFO)

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)


def print_separator(title: str):
    """Imprime un separador visual."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_planner_basic():
    """Test básico: saludo con nombre."""
    print_separator("TEST 1: Saludo con nombre")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ SKIPPING: No API Key found.")
        return
    
    model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    tools = [SaveContacto()]
    
    bot = PlannerBot(
        language_model=model,
        embeddings=embeddings,
        function_purpose="""
        # Asistente de Alkilautos
        Ayuda a los clientes con información sobre alquiler de vehículos.
        Guarda los datos de contacto cuando el usuario los proporcione.
        """,
        personality_tone="Amigable, profesional, usa emojis ocasionalmente",
        limitations="No puede hacer reservas directamente",
        dynamic_info="""
        ## CURRENT CONTEXT
        ### Your identity
        - Name: Asistente Virtual
        - Date/Time: 2024-12-05 15:00
        - Channel: WhatsApp
        
        ### Current contact
        - Status: Anonymous
        - Policies accepted: No
        """,
        tools=tools,
        logger=logger
    )
    
    user_input = "Hola, me llamo Erley"
    print(f"👤 Usuario: {user_input}")
    
    response = bot.get_response(user_input, [], [])
    
    print(f"\n🤖 Bot: {response['content']}")
    print(f"\n📋 Tools ejecutadas: {len(response['tools_executed'])}")
    for tool in response['tools_executed']:
        status_icon = "✅" if tool['status'] == 'success' else "❌"
        print(f"   {status_icon} {tool['tool_name']}")
    
    print(f"\n📊 Tokens: {response['token_usage']}")


def test_planner_multiple_actions():
    """Test: mensaje con múltiples acciones (nombre + pregunta)."""
    print_separator("TEST 2: Múltiples acciones")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ SKIPPING: No API Key found.")
        return
    
    model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    tools = [SaveContacto(), EmailTool()]
    
    bot = PlannerBot(
        language_model=model,
        embeddings=embeddings,
        function_purpose="""
        # Asistente de Alkilautos
        - Guarda datos de contacto cuando el usuario los proporcione
        - Envía emails cuando se solicite
        """,
        personality_tone="Profesional pero cercano",
        limitations="",
        dynamic_info="""
        ## CURRENT CONTEXT
        - Policies accepted: Yes ✓
        """,
        tools=tools,
        logger=logger
    )
    
    user_input = "Soy Juan García, mi email es juan@test.com. Envía un correo a soporte@empresa.com diciendo que necesito ayuda"
    print(f"👤 Usuario: {user_input}")
    
    response = bot.get_response(user_input, [], [])
    
    print(f"\n🤖 Bot: {response['content']}")
    print(f"\n📋 Tools ejecutadas: {len(response['tools_executed'])}")
    for tool in response['tools_executed']:
        status_icon = "✅" if tool['status'] == 'success' else "❌"
        print(f"   {status_icon} {tool['tool_name']}: {tool['output'][:80]}...")
    
    print(f"\n📊 Tokens: {response['token_usage']}")


def test_planner_policy_request():
    """Test: primer mensaje sin políticas aceptadas."""
    print_separator("TEST 3: Solicitud de políticas")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ SKIPPING: No API Key found.")
        return
    
    # Crear una tool mock de accept_policies
    from langchain_core.tools import tool
    
    @tool
    def accept_policies(user_message: str) -> str:
        """Registra la aceptación de políticas del usuario."""
        return "Políticas aceptadas correctamente"
    
    model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    tools = [accept_policies, SaveContacto()]
    
    bot = PlannerBot(
        language_model=model,
        embeddings=embeddings,
        function_purpose="""
        # Asistente de Alkilautos
        IMPORTANTE: Antes de ayudar, el usuario debe aceptar las políticas de privacidad.
        Link de políticas: https://ejemplo.com/politicas
        """,
        personality_tone="Amigable y profesional",
        limitations="",
        dynamic_info="""
        ## CURRENT CONTEXT
        - Policies accepted: No
        """,
        tools=tools,
        logger=logger
    )
    
    # Primer mensaje - debería pedir políticas
    user_input = "Hola, quiero información"
    print(f"👤 Usuario: {user_input}")
    
    response = bot.get_response(user_input, [], [])
    
    print(f"\n🤖 Bot: {response['content']}")
    print(f"\n📋 Tools ejecutadas: {len(response['tools_executed'])}")
    
    # Segundo mensaje - usuario acepta
    print("\n" + "-"*40 + "\n")
    user_input2 = "Sí, acepto"
    print(f"👤 Usuario: {user_input2}")
    
    # Simular historial
    from sonika_langchain_bot.langchain_class import Message
    messages = [
        Message(content="Hola, quiero información", is_bot=False),
        Message(content=response['content'], is_bot=True)
    ]
    
    response2 = bot.get_response(user_input2, messages, response['logs'])
    
    print(f"\n🤖 Bot: {response2['content']}")
    print(f"\n📋 Tools ejecutadas: {len(response2['tools_executed'])}")
    for tool in response2['tools_executed']:
        status_icon = "✅" if tool['status'] == 'success' else "❌"
        print(f"   {status_icon} {tool['tool_name']}")


def test_planner_chitchat():
    """Test: conversación casual sin acciones."""
    print_separator("TEST 4: Chitchat puro")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ SKIPPING: No API Key found.")
        return
    
    model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    bot = PlannerBot(
        language_model=model,
        embeddings=embeddings,
        function_purpose="Asistente de atención al cliente",
        personality_tone="Muy amigable, usa emojis, habla en español",
        limitations="",
        dynamic_info="Policies accepted: Yes",
        tools=[],  # Sin tools
        logger=logger
    )
    
    user_input = "¿Cómo estás?"
    print(f"👤 Usuario: {user_input}")
    
    response = bot.get_response(user_input, [], [])
    
    print(f"\n🤖 Bot: {response['content']}")
    print(f"\n📋 Tools ejecutadas: {len(response['tools_executed'])}")
    print(f"📊 Tokens: {response['token_usage']}")


if __name__ == "__main__":
    print("\n" + "🚀 PLANNER BOT - TESTS DE INTEGRACIÓN 🚀".center(60))
    
    test_planner_basic()
    test_planner_multiple_actions()
    test_planner_policy_request()
    test_planner_chitchat()
    
    print_separator("TESTS COMPLETADOS")
