
import os
import sys
import asyncio
import logging
import json
from dotenv import load_dotenv

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sonika_langchain_bot.orchestrator.orchestrator_bot import OrchestratorBot
from sonika_langchain_bot.langchain_tools import EmailTool, SaveContacto
from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from langchain_openai import OpenAIEmbeddings

# Configuración de logging (Silenciar librerías ruidosas)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_integration")
logger.setLevel(logging.INFO)

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

def test_orchestrator_integration():
    print("\n--- TEST INTEGRACIÓN ORCHESTRATOR BOT ---")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ SKIPPING: No API Key found.")
        return

    model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=0)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    tools = [EmailTool(), SaveContacto()]

    purpose = """
    # Alkilautos Assistant
    ## Function
    Manage contact data and bookings.

    ## 3. SAVE CONTACT DATA
    Whenever the user provides first name, email, phone, call `create_or_update_contact` (SaveContact).

    ## 6. EMAIL QUOTATION
    When user asks to send email, call email tool.
    """

    bot = OrchestratorBot(
        language_model=model,
        embeddings=embeddings,
        function_purpose=purpose,
        personality_tone="Professional",
        limitations="None",
        dynamic_info="User: Erley",
        tools=tools,
        logger=logger
    )

    user_input = "Envia un email a erley@test.com diciendo Hola y guarda mi contacto Erley con cel 123456"
    print(f"Usuario: {user_input}")

    try:
        # Simulamos que no hay historial previo
        response = bot.get_response(user_input, [], [])

        # --- FORMATO BONITO ---
        print("\n\n✅ RESPUESTA DEL BOT:")

        # Limpiar y mostrar Logs con Iconos
        print("\n📋 LOGS GENERADOS:")
        for log in response['logs']:
            # Simular formato bonito si el log lo permite
            print(f"   {log}")

        # Mostrar Tools
        print("\n🔧 TOOLS EXECUTED:")
        tools = response.get('tools_executed', [])
        if tools:
            for t in tools:
                status_icon = "✅" if t.get("status") == "success" else "❌"
                print(f"   {status_icon} Tool: {t.get('tool_name')}")
                print(f"      Output: {t.get('output')[:100]}...")
        else:
            print("   (No tools executed)")

        # Mostrar JSON Final
        print("\n📦 JSON COMPLETO:")
        print(json.dumps(response, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_orchestrator_integration()
