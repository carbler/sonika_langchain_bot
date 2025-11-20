"""Logger Node - Tracks and logs events."""

from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from sonika_langchain_bot.bot.nodes.base_node import BaseNode

class LoggerNode(BaseNode):
    """Logs events and generates audit trail."""
    
    def __init__(
        self,
        on_logs_generated: Optional[Callable] = None,
        logger=None
    ):
        super().__init__(logger)
        self.on_logs_generated = on_logs_generated
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Consolida logs y notifica callbacks."""
        
        # Los logs ya están en el estado, solo notificamos
        all_logs = state.get("logs", [])
        
        # Extraer solo los logs nuevos de esta interacción
        existing_count = len(state.get("initial_logs", []))
        new_logs = all_logs[existing_count:]
        
        if self.on_logs_generated and new_logs:
            try:
                self.on_logs_generated(new_logs)
            except Exception as e:
                self.logger.warning(f"Callback failed: {e}")
        
        # ✅ SOLO retornar lo que actualizas, NO todo el state
        return {"logger_output": new_logs}