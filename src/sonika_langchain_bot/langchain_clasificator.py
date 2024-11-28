from pydantic import BaseModel
from typing import Dict, Any, Type
from sonika_langchain_bot.langchain_class import ILanguageModel

# Clase para realizar la clasificación de texto
class TextClassifier:
    def __init__(self, validation_class: Type[BaseModel], llm: ILanguageModel):
        self.llm =llm
        self.validation_class = validation_class
        self.llm.set_clasificator(validation_class=validation_class)

    def classify(self, text: str) -> Dict[str, Any]:
        # Crear el template del prompt
        prompt = f"""
        Classify the following text based on the properties defined in the validation class.
        
        Text: {text}
        
        Only extract the properties mentioned in the validation class.
        """
        response = self.llm.invoke(prompt=prompt)
        
        # Asegurarse de que el `response` es de la clase de validación proporcionada
        if isinstance(response, self.validation_class):
            # Crear el resultado dinámicamente basado en los atributos de la clase de validación
            result = {field: getattr(response, field) for field in self.validation_class.__fields__.keys()}
            return result
        else:
            raise ValueError(f"The response is not of type '{self.validation_class.__name__}'")
