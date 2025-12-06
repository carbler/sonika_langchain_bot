"""Base Node para la arquitectura Planner."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging


class NullLogger:
    """Logger que no hace nada - evita checks de None en todo el código."""
    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def critical(self, *args, **kwargs): pass


class BaseNode(ABC):
    """Clase base para todos los nodos del Planner Bot."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or NullLogger()

    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa el estado y retorna actualizaciones."""
        pass

    def format_timestamp(self) -> str:
        """Genera timestamp formateado."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _add_log(self, message: str, node_name: str = None) -> Dict[str, Any]:
        """Retorna actualización de logs."""
        timestamp = self.format_timestamp()
        node = node_name or self.__class__.__name__
        log_entry = f"[{timestamp}] [{node}] {message}"
        return {"logs": [log_entry]}

    def _convert_messages_to_langchain(self, messages: List[Any]) -> List[Any]:
        """
        Convierte mensajes custom a formato LangChain.
        
        Soporta:
        - BaseMessage nativos de LangChain
        - Objetos custom con .is_bot y .content
        - Dicts con 'role' y 'content'
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
        
        converted = []
        
        for msg in messages:
            # Caso 1: Ya es BaseMessage
            if isinstance(msg, BaseMessage):
                converted.append(msg)
                continue
            
            # Caso 2: Objeto custom con is_bot y content
            if hasattr(msg, 'is_bot') and hasattr(msg, 'content'):
                if msg.is_bot:
                    converted.append(AIMessage(content=msg.content))
                else:
                    converted.append(HumanMessage(content=msg.content))
                continue
            
            # Caso 3: Dict con role y content
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role in ('assistant', 'bot', 'ai'):
                    converted.append(AIMessage(content=content))
                elif role == 'system':
                    converted.append(SystemMessage(content=content))
                else:
                    converted.append(HumanMessage(content=content))
                continue
            
            # Fallback: convertir a string
            converted.append(HumanMessage(content=str(msg)))
        
        return converted
