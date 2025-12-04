"""Base Node for Orchestrator Architecture."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime


class BaseNode(ABC):
    """Clase base para nodos."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa el estado."""
        pass

    def format_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _add_log(self, state: Dict[str, Any], message: str, node_name: str = None) -> Dict[str, Any]:
        """Retorna actualización de logs."""
        timestamp = self.format_timestamp()
        node = node_name or self.__class__.__name__.replace('Node', '').upper()
        log_entry = f"[{timestamp}] [{node}] {message}"
        return {"logs": [log_entry]}
