
import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sonika_langchain_bot.orchestrator.orchestrator_bot import OrchestratorBot
from sonika_langchain_bot.langchain_tools import EmailTool, SaveContacto
from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from langchain_openai import OpenAIEmbeddings

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_integration")

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

    # Prompt complejo de Alkilautos (resumido)
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

    # Input que requiere acción (TaskAgent)
    user_input = "Envia un email a erley@test.com diciendo Hola y guarda mi contacto Erley con cel 123456"

    print(f"User: {user_input}")

    try:
        response = bot.get_response(user_input, [], [])
        print("\n✅ RESPONSE:")
        print(f"Content: {response['content']}")
        print("Logs:")
        for log in response['logs']:
            print(log)

        print("\nTools Executed:")
        for tool in response['tools_executed']:
            print(tool)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_orchestrator_integration()
