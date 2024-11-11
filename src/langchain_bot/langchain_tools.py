from langchain_community.tools import BaseTool
from django.core.mail import send_mail
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