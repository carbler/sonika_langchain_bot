from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any, Type

from apps.aichat.langchain_class import ILanguageModel

class OpenAIModel(ILanguageModel):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.7, validation_class: Type[BaseModel] = None,):
        self.model =  ChatOpenAI(api_key=api_key, temperature=temperature, model=model_name).with_structured_output(validation_class)

    def get_response(self, prompt: str) -> str:
        message = HumanMessage(content=prompt)
        # Invocar el modelo con el mensaje creado
        response = self.model.invoke([message])
        return response

# Clase para realizar la clasificación de texto
class TextClassifier:
    def __init__(self, api_key: str, validation_class: Type[BaseModel], llm: ILanguageModel, temperature: float = 0):
        self.llm =llm
        self.validation_class = validation_class

    def classify(self, text: str) -> Dict[str, Any]:
        # Crear el template del prompt
        prompt = f"""
        Classify the following text based on the properties defined in the validation class.
        
        Text: {text}
        
        Only extract the properties mentioned in the validation class.
        """
        response = self.llm.get_response(prompt=prompt)
        
        # Asegurarse de que el `response` es de la clase de validación proporcionada
        if isinstance(response, self.validation_class):
            # Crear el resultado dinámicamente basado en los atributos de la clase de validación
            result = {field: getattr(response, field) for field in self.validation_class.__fields__.keys()}
            return result
        else:
            raise ValueError(f"The response is not of type '{self.validation_class.__name__}'")
