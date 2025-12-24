import os
import sys
import time
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE RUTAS ---
# Ajustar para importar desde src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sonika_langchain_bot.langchain_models import OpenAILanguageModel
from sonika_langchain_bot.langchain_bot_agent import Message
from sonika_langchain_bot.planner_react import PlannerBot
from langchain_openai import OpenAIEmbeddings

# Importar componentes locales
from tools import (
    GetUserProfile, TransactionTool, CreateTicket, BlockAccountTool,
    RefundTool, GetTransactionHistory, VerifyIdentityDocument,
    ApplyPromoCode, CheckFraudScore, UpdateAccountTier,
    ScheduleCallback, AdjustCreditLimit
)
from instructions import PERSONALITY_TONE, LIMITATIONS, FUNCTION_PURPOSE
from test_cases import tests_data

load_dotenv()

# ==========================================
# TEST RUNNER CON SCORING
# ==========================================

class UltimateStressTestRunner:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY no encontrada")
        self.llm = OpenAILanguageModel(api_key, model_name='gpt-4o-mini', temperature=0)
        self.total_score = 0
        self.max_score = 0
        self.test_results = []
        self.embeddings = OpenAIEmbeddings(api_key=api_key)

    def build_conversation_history(self, messages_data):
        """Convierte lista de tuplas (content, is_bot) a objetos Message"""
        return [Message(content=msg[0], is_bot=msg[1]) for msg in messages_data]

    def run_test(self, test_num, test_name, conversation_history, user_input, validation_fn, max_points=100):
        print(f"\n{'='*90}")
        print(f"🧪 TEST #{test_num}: {test_name}")
        print(f"   💯 Max Score: {max_points} points")
        print(f"   📜 Conversación previa: {len(conversation_history)} mensajes")
        print(f"   📥 Input Usuario: '{user_input}'")

        try:
            tools = [
                GetUserProfile(), TransactionTool(), CreateTicket(), BlockAccountTool(),
                RefundTool(), GetTransactionHistory(), VerifyIdentityDocument(),
                ApplyPromoCode(), CheckFraudScore(), UpdateAccountTier(),
                ScheduleCallback(), AdjustCreditLimit()
            ]

            bot = PlannerBot(
                embeddings=self.embeddings,
                language_model=self.llm,
                function_purpose=FUNCTION_PURPOSE,
                personality_tone=PERSONALITY_TONE,
                limitations=LIMITATIONS,
                dynamic_info='',
                tools=tools,
                on_tool_start=lambda x, y: None,
                on_tool_end=lambda x, y: None,
                on_tool_error=lambda x, y: None
            )

            # Cargar conversación previa
            history_messages = self.build_conversation_history(conversation_history)

            # NOTA: PlannerBot.get_response acepta 'messages' como argumento,
            # no necesitamos llamar a load_conversation_history explícitamente si se pasa en get_response.
            # Sin embargo, si la implementación interna lo requiere, podemos hacerlo.
            # En la implementación de referencia:
            # bot.get_response(user_input, messages=history_messages, logs=[])

            start_time = time.time()
            response = bot.get_response(user_input=user_input, logs=[], messages=history_messages)
            execution_time = time.time() - start_time

        except Exception as e:
            print(f"   ❌ CRASH: {e}")
            import traceback
            traceback.print_exc()
            self.record_result(test_num, test_name, 0, max_points, f"CRASH: {str(e)}", execution_time=0)
            return

        bot_content = response.get('content', '')
        tools_executed = response.get('tools_executed', [])

        adapted_logs = [{"name": t.get('tool_name'), "input": str(t.get('args'))} for t in tools_executed]
        tool_names = [t['name'] for t in adapted_logs]

        # Validación
        try:
            score, feedback = validation_fn(adapted_logs, bot_content, conversation_history)
        except Exception as e:
            print(f"   ❌ Error en validación: {e}")
            score = 0
            feedback = f"Error en validación: {str(e)}"

        passed = score >= (max_points * 0.7)  # 70% es passing
        status = "✅ PASSED" if passed else "❌ FAILED"

        print(f"   {status} - Score: {score}/{max_points} ({int(score/max_points*100)}%)")
        print(f"   ⏱️  Execution Time: {execution_time:.2f}s")
        print(f"   📊 Feedback: {feedback}")
        print(f"\n   🔍 DEBUG INFO:")
        print(f"   🤖 Bot Response: \"{bot_content[:200]}...\"" if len(bot_content) > 200 else f"   🤖 Bot Response: \"{bot_content}\"")
        print(f"   🔧 Tools Executed: {tool_names if tool_names else '[NINGUNA]'}")
        print(f"{'='*90}")

        self.record_result(test_num, test_name, score, max_points, feedback, execution_time)

    def record_result(self, test_num, test_name, score, max_score, feedback, execution_time):
        self.total_score += score
        self.max_score += max_score
        self.test_results.append({
            "test_num": test_num,
            "name": test_name,
            "score": score,
            "max_score": max_score,
            "percentage": int(score/max_score*100) if max_score > 0 else 0,
            "feedback": feedback,
            "execution_time": execution_time
        })

    def print_final_report(self):
        print(f"\n\n{'='*90}")
        print(f"📈 REPORTE FINAL")
        print(f"{'='*90}")
        print(f"Total Score: {self.total_score}/{self.max_score} ({int(self.total_score/self.max_score*100)}%)")
        print(f"\nDetalle por Test:")
        for result in self.test_results:
            emoji = "✅" if result['percentage'] >= 70 else "❌"
            print(f"{emoji} Test #{result['test_num']}: {result['score']}/{result['max_score']} ({result['percentage']}%) - {result['name']}")
        print(f"{'='*90}\n")

if __name__ == "__main__":
    print("🚀 INICIANDO ULTIMATE STRESS TEST DE NEOFIN AI...")

    try:
        runner = UltimateStressTestRunner()
    except ValueError as e:
        print(f"❌ Error de configuración: {e}")
        sys.exit(1)

    # Ejecutar Loop de Pruebas importado de test_cases
    for t_num, t_name, t_hist, t_input, t_val in tests_data:
        runner.run_test(
            test_num=t_num,
            test_name=t_name,
            conversation_history=t_hist,
            user_input=t_input,
            validation_fn=t_val
        )

    # Imprimir Reporte
    runner.print_final_report()
