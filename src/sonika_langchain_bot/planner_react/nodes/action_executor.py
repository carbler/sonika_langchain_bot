"""Action Executor - Ejecuta las acciones del plan con retry."""

from typing import Dict, Any, List, Optional, Callable
import json
import asyncio
from sonika_langchain_bot.planner_react.nodes.base_node import BaseNode


class ActionExecutor(BaseNode):
    """
    Ejecutor de acciones que procesa el plan generado por el Smart Orchestrator.
    
    Responsabilidades:
    - Validar parámetros requeridos antes de ejecutar (usando schema de la tool)
    - Ejecutar acciones en orden de prioridad
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
        
        # Extraer schemas de las tools para validación
        self._tool_schemas = {}
        for t in tools:
            self._tool_schemas[t.name] = self._extract_required_params(t)

    def _extract_required_params(self, tool) -> Dict[str, List[str]]:
        """
        Extrae los parámetros requeridos del schema de una tool.
        
        Soporta:
        - LangChain BaseTool con args_schema (Pydantic v1/v2)
        - LangChain BaseTool con _run y type hints (Optional detection)
        - MCP Tools con inputSchema (JSON Schema)
        - HTTPTool con args_schema dinámico
        
        Returns:
            Dict con 'required' (lista de campos requeridos) y 'all' (todos los campos)
        """
        required = []
        all_params = []
        
        try:
            # Método 1: MCP Tools - inputSchema (JSON Schema format)
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema = tool.inputSchema
                if isinstance(schema, dict):
                    # JSON Schema: properties contiene todos, required es lista
                    all_params = list(schema.get('properties', {}).keys())
                    required = schema.get('required', [])
                    return {'required': required, 'all': all_params}
            
            # Método 2: args_schema (Pydantic model - usado por HTTPTool y otros)
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema
                
                # Pydantic v2
                if hasattr(schema, 'model_fields'):
                    for name, field in schema.model_fields.items():
                        all_params.append(name)
                        if field.is_required():
                            required.append(name)
                    return {'required': required, 'all': all_params}
                
                # Pydantic v1
                elif hasattr(schema, '__fields__'):
                    for name, field in schema.__fields__.items():
                        all_params.append(name)
                        if field.required:
                            required.append(name)
                    return {'required': required, 'all': all_params}
                
                # JSON Schema dict (algunos tools lo definen así)
                elif isinstance(schema, dict):
                    all_params = list(schema.get('properties', {}).keys())
                    required = schema.get('required', [])
                    return {'required': required, 'all': all_params}
            
            # Método 3: Inspección del método _run con type hints
            import inspect
            from typing import get_origin, get_args, Union
            
            if hasattr(tool, '_run'):
                sig = inspect.signature(tool._run)
                type_hints = {}
                try:
                    type_hints = tool._run.__annotations__
                except Exception:
                    pass
                
                for name, param in sig.parameters.items():
                    if name in ('self', 'kwargs', 'args'):
                        continue
                    
                    all_params.append(name)
                    
                    # Verificar si es Optional en type hints
                    is_optional = False
                    if name in type_hints:
                        hint = type_hints[name]
                        origin = get_origin(hint)
                        # Optional[X] es Union[X, None]
                        if origin is Union:
                            args = get_args(hint)
                            if type(None) in args:
                                is_optional = True
                    
                    # Si tiene default value, es opcional
                    has_default = param.default != inspect.Parameter.empty
                    
                    # Es requerido si NO es Optional y NO tiene default
                    if not is_optional and not has_default:
                        required.append(name)
                        
        except Exception as e:
            self.logger.warning(f"Could not extract schema for {tool.name}: {e}")
        
        return {'required': required, 'all': all_params}

    def _validate_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida que los parámetros requeridos estén presentes y no vacíos.
        
        Returns:
            Dict con 'valid' (bool) y 'missing' (lista de campos faltantes)
        """
        schema_info = self._tool_schemas.get(tool_name, {})
        required = schema_info.get('required', [])
        
        if not required:
            return {'valid': True, 'missing': []}
        
        missing = []
        for field in required:
            value = params.get(field)
            # Considerar vacío: None, string vacío, o solo espacios
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field)
        
        return {
            'valid': len(missing) == 0,
            'missing': missing
        }

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
        
        # Validar parámetros requeridos ANTES de ejecutar
        validation = self._validate_params(tool_name, params)
        if not validation['valid']:
            missing_fields = validation['missing']
            return {
                "status": "missing_params",
                "output": f"Missing required parameters: {', '.join(missing_fields)}",
                "missing_params": missing_fields,
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
        
        log_details = []
        
        for action in actions:
            tool_name = action.get("tool_name")
            params = action.get("params", {})
            
            # Verificar que la tool existe
            if tool_name not in self.tools:
                execution_results.append({
                    "tool_name": tool_name,
                    "status": "skipped",
                    "output": f"Tool '{tool_name}' not found"
                })
                log_details.append(f"⏭️ {tool_name}: skipped (not found)")
                continue
            
            # Ejecutar con retry
            exec_result = await self._execute_with_retry(tool_name, params, action.get("type", ""))
            
            result_entry = {
                "tool_name": tool_name,
                "params": params,
                "status": exec_result["status"],
                "output": exec_result["output"],
                "attempts": exec_result["attempts"]
            }
            # Incluir missing_params si existe
            if "missing_params" in exec_result:
                result_entry["missing_params"] = exec_result["missing_params"]
            execution_results.append(result_entry)
            
            tools_executed.append({
                "tool_name": tool_name,
                "args": json.dumps(params, ensure_ascii=False) if isinstance(params, dict) else str(params),
                "output": exec_result["output"],
                "status": exec_result["status"]
            })
            
            # Log descriptivo por tool
            status = exec_result["status"]
            if status == "success":
                status_icon = "✅"
            elif status == "missing_params":
                status_icon = "⚠️"
            else:
                status_icon = "❌"
            
            params_short = ", ".join(f"{k}={str(v)[:30]}" for k, v in params.items())
            output_preview = exec_result["output"][:100] + "..." if len(exec_result["output"]) > 100 else exec_result["output"]
            log_details.append(f"{status_icon} {tool_name}({params_short}) → {output_preview}")
        
        # Resumen
        success = sum(1 for r in execution_results if r["status"] == "success")
        failed = sum(1 for r in execution_results if r["status"] == "failed")
        skipped = sum(1 for r in execution_results if r["status"] == "skipped")
        missing = sum(1 for r in execution_results if r["status"] == "missing_params")
        
        # Log con detalles de cada ejecución
        summary_parts = []
        if success: summary_parts.append(f"{success}✅")
        if failed: summary_parts.append(f"{failed}❌")
        if skipped: summary_parts.append(f"{skipped}⏭️")
        if missing: summary_parts.append(f"{missing}⚠️ missing params")
        summary = f"Results: {' '.join(summary_parts)}" if summary_parts else "No actions executed"
        
        result = {
            "execution_results": execution_results,
            "tools_executed": tools_executed,
            **self._add_log(summary)
        }
        
        # Agregar logs detallados por cada tool
        for detail in log_details:
            result = {**result, **self._add_log(detail)}
        
        return result
