"""Output Node - Generates natural language response based on planner output and tools executed."""

from typing import Dict, Any, List
from langchain.schema import SystemMessage
from sonika_langchain_bot.bot.nodes.base_node import BaseNode


class OutputNode(BaseNode):
    """Generates final response to user based on planner_output and tools executed."""
    
    def __init__(self, model, logger=None):
        super().__init__(logger)
        self.model = model
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final response."""
        try:
            response_text = self._generate_response(state)

            # Log completado
            preview = response_text[:80].replace('\n', ' ')
            self._add_log(state, f"Respuesta generada: {preview}...")
            
            return {"output_node_response": response_text}
            
        except Exception as e:
            self.logger.error(f"Output generation failed: {e}")
            return {"output_node_response": "Disculpa, encontré un error al procesar tu solicitud."}
    
    def _generate_response(self, state: Dict[str, Any]) -> str:
        """Generate response based on planner reasoning and tools."""

        user_input = state.get("user_input", "")
        personality_tone = state.get("personality_tone", "")
        limitations = state.get("limitations", "")
        planner_output = state.get("planner_output", {})
        tools_executed = state.get("tools_executed", [])
        dynamic_info = state.get("dynamic_info", "")

        results_summary = self._build_results_summary(tools_executed)

        context_summary = f"""
Dynamic Context:
{dynamic_info}

Planner Reasoning:
{planner_output.get('reasoning', 'No reasoning provided')}

Information from Tools:
{results_summary}
"""

        prompt = f"""# RESPONSE GENERATOR

## PERSONALITY
{personality_tone}

## LIMITATIONS (MANDATORY)
{limitations}

## USER MESSAGE
{user_input}

## CONTEXT
{context_summary}

## INSTRUCTIONS
1. Follow all limitations strictly
2. Use information from Planner Reasoning and Tools Executed
3. Be conversational, helpful, and natural
4. Match the user's language
5. Never invent information not provided

Generate the response below:
"""
        
        response = self.model.invoke([SystemMessage(content=prompt)], config={"temperature": 0.3})
        
        if hasattr(response, 'content'):
            return response.content.strip()
        return str(response).strip()
    
    def _build_results_summary(self, tools_executed: List[Dict[str, Any]]) -> str:
        """Build summary of tool results."""
        if not tools_executed:
            return "No tools were executed. Agent may need more information from user."
        
        summary = []
        for tool in tools_executed:
            tool_name = tool.get("tool_name", "unknown")
            output = tool.get("output", "No output")
            status = tool.get("status", "unknown")
            
            if status == "success":
                summary.append(f"From {tool_name}: {output}")
            else:
                summary.append(f"{tool_name} failed: {output}")
        
        return "\n\n".join(summary)