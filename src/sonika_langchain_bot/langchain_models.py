from langchain_openai import ChatOpenAI
import tiktoken
from sonika_langchain_bot.langchain_class import ILanguageModel


class OpenAILanguageModel(ILanguageModel):
    """
    Clase que implementa la interfaz ILanguageModel para interactuar con los modelos de lenguaje de OpenAI.
    Proporciona funcionalidades para generar respuestas y contar tokens.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Inicializa el modelo de lenguaje de OpenAI.
        
        Args:
            api_key (str): Clave API de OpenAI
            model_name (str): Nombre del modelo a utilizar
            temperature (float): Temperatura para la generación de respuestas
        """
        self.model = ChatOpenAI(temperature=temperature, model_name=model_name, api_key=api_key)
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def get_response(self, prompt: str) -> str:
        """
        Genera una respuesta basada en el prompt proporcionado.
        
        Args:
            prompt (str): Texto de entrada para generar la respuesta
            
        Returns:
            str: Respuesta generada por el modelo
        """
        return self.model.predict(prompt)

    def count_tokens(self, text: str) -> int:
        """
        Cuenta el número de tokens en un texto dado.
        
        Args:
            text (str): Texto para contar tokens
            
        Returns:
            int: Número de tokens en el texto
        """
        return len(self.tokenizer.encode(text))