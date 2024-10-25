from abc import ABC, abstractmethod
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from django.core.mail import send_mail


class ResponseModel():
    def __init__(self, user_tokens=None, bot_tokens=None,  response = None):
        self.user_tokens = user_tokens
        self.bot_tokens = bot_tokens
        self.response = response
# Definir la interfaz para procesar archivos
class FileProcessorInterface(ABC):

    @abstractmethod
    def getText(self):
        pass

class PDFProcessor(FileProcessorInterface):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def getText(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        return documents

class ILanguageModel(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

class IEmbeddings(ABC):
    @abstractmethod
    def embed_documents(self, documents: List[str]):
        pass

    @abstractmethod
    def embed_query(self, query: str):
        pass

from langchain_community.tools import BaseTool

# Crear una clase que herede de BaseTool
from pydantic import BaseModel

class EmailTool(BaseTool, BaseModel):
    name: str = "EmailTool"
    description: str = "Esta herramienta envía correos electrónicos."

    def _run(self, to_email: str, subject: str, message: str) -> str:
        success = send_mail(
            subject,
            message,
            to_email,
            ["erley.bc@gmail.com"],
            fail_silently=False,
        )
        if success:
            return "Correo enviado con éxito."
        else:
            return "No se pudo enviar el correo."