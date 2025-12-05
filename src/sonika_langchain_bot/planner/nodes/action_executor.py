"""Action Executor - Ejecuta las acciones del plan con retry."""

from typing import Dict, Any, List, Optional, Callable
import json
import asyncio
from sonika_langchain_bot.planner.nodes.base_node import BaseNode


class ActionExecutor(BaseNode):
    """
    Ejecutor de acciones que procesa el plan generado por el Smart Orchestrator.
    
    Con bind_tools en el SmartOrchestrator, los parámetros ya vienen correctos.
    Este ejecutor simplemente:
    - Ejecuta acciones en orden de prioridad
    - Retry automático (hasta 3 intentos por tool)
    - Callbacks para monitoreo de ejecución
    """

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 0.5

    def __init__(
        self,
        tools: List[Any],
        on_tool_start: Optional[Callable[[str, str], None]] = None,
        on_tool_end: Optional[Callable[[str, str], None]] = None,
        on_tool_error: Optional[Callable[[str, str], None]] = None,
        logger=None
    ):
        super().__init__(logger)
        self.tools = {t.name: t for t in tools}
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error

    async def _execute_with_retry(
        self, 
        tool_name: str, 
        params: Dict[str, Any],
        action_type: str
    ) -> Dict[str, Any]:
        """
        Ejecuta una tool con retry automático.
        
        Returns:
            Dict con status, output, y attempts
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return {
                "status": "failed",
                "output": f"Tool '{tool_name}' not found",
                "attempts": 0
            }
        
        last_error = None
        params_str = json.dumps(params) if isinstance(params, dict) else str(params)
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Callback: inicio
                if self.on_tool_start:
                    try:
                        self.on_tool_start(tool_name, params_str)
                    except Exception:
                        pass
                
                # Ejecutar tool
                result = await tool.ainvoke(params)
                output = str(result)
                
                # Callback: éxito
                if self.on_tool_end:
                    try:
                        self.on_tool_end(tool_name, output)
                    except Exception:
                        pass
                
                return {
                    "status": "success",
                    "output": output,
                    "attempts": attempt
                }
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Tool {tool_name} attempt {attempt} failed: {e}")
                
                # Callback: error
                if self.on_tool_error:
                    try:
                        self.on_tool_error(tool_name, f"Attempt {attempt}: {last_error}")
                    except Exception:
                        pass
                
                # Esperar antes de reintentar (excepto en el último intento)
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_DELAY_SECONDS)
        
        # Todos los intentos fallaron
        return {
            "status": "failed",
            "output": f"Failed after {self.MAX_RETRIES} attempts. Last error: {last_error}",
            "attempts": self.MAX_RETRIES
        }

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta todas las acciones del plan en orden de prioridad."""
        action_plan = state.get("action_plan")
        
        if not action_plan or not action_plan.get("actions"):
            return {
                "execution_results": [],
                "tools_executed": [],
                **self._add_log("No actions to execute")
            }
        
        # Ordenar por prioridad
        actions = sorted(action_plan["actions"], key=lambda x: x.get("priority", 99))
        
        execution_results = []
        tools_executed = []
        
        for action in actions:
            tool_name = action.get("tool_name")
            params = action.get("params", {})
            
            # Verificar que la tool existe
            if tool_name not in self.tools:
                self.logger.warning(f"Tool '{tool_name}' not found, skipping")
                execution_results.append({
                    "tool_name": tool_name,
                    "status": "skipped",
                    "output": f"Tool '{tool_name}' not found"
                })
                continue
            
            # Ejecutar con retry - los parámetros ya vienen correctos de bind_tools
            self.logger.info(f"Executing {tool_name} with params: {params}")
            exec_result = await self._execute_with_retry(tool_name, params, action.get("type", ""))
            
            execution_results.append({
                "tool_name": tool_name,
                "params": params,
                "status": exec_result["status"],
                "output": exec_result["output"],
                "attempts": exec_result["attempts"]
            })
            
            tools_executed.append({
                "tool_name": tool_name,
                "args": json.dumps(params) if isinstance(params, dict) else str(params),
                "output": exec_result["output"],
                "status": exec_result["status"]
            })
        
        # Resumen
        success = sum(1 for r in execution_results if r["status"] == "success")
        failed = sum(1 for r in execution_results if r["status"] == "failed")
        skipped = sum(1 for r in execution_results if r["status"] == "skipped")
        
        return {
            "execution_results": execution_results,
            "tools_executed": tools_executed,
            **self._add_log(f"Executed {len(execution_results)} actions: {success} success, {failed} failed, {skipped} skipped")
        }
