"""Output Node - Generates natural language response based on planner output and tools executed."""

from typing import Dict, Any, List
from langchain.schema import SystemMessage, HumanMessage
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

        planner_reasoning = planner_output.get('reasoning', 'No reasoning provided')
        results_summary = self._build_results_summary(tools_executed)

        system_prompt = f"""# RESPONSE GENERATOR

You are a response generator in a multi-agent system. Your job is to create the final response to the user.

## YOUR ROLE
You receive instructions from a strategic planner that has already analyzed the conversation, checked business rules, and determined what action should be taken. Your job is NOT to make decisions - your job is to EXECUTE the planner's instructions and communicate them naturally to the user.

## HOW YOU WORK
1. **PLANNER REASONING** tells you WHAT to do - this is your primary directive
2. **PERSONALITY** tells you HOW to communicate - tone, style, structure
3. **LIMITATIONS** are absolute rules you must never break
4. **DYNAMIC CONTEXT** and **TOOLS RESULTS** provide factual information to use
5. **USER MESSAGE** is what the user said - respond appropriately in their language

## CRITICAL RULES
- FOLLOW the planner's reasoning exactly - if it says to do something, you MUST do it
- APPLY the personality guidelines for tone and communication style
- RESPECT all limitations without exception
- USE only information provided in the context - never invent facts
- RESPOND in the same language as the user
- BE natural and conversational, not robotic

Generate the appropriate response based on the information provided below.

## PERSONALITY (HOW TO COMMUNICATE)
{personality_tone}

## LIMITATIONS (MANDATORY RULES)
{limitations}
"""

        analysis_input = f"""## PLANNER REASONING (WHAT TO DO)
{planner_reasoning}

## DYNAMIC CONTEXT
{dynamic_info}

## TOOLS RESULTS
{results_summary}

## USER MESSAGE
{user_input}

---
Generate your response now:"""
        
        message_list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_input)
        ]
        
        response = self.model.invoke(message_list, config={"temperature": 0.3})
        
        if hasattr(response, 'content'):
            return response.content.strip()
        return str(response).strip()
    
    def _build_results_summary(self, tools_executed: List[Dict[str, Any]]) -> str:
        """Build summary of tool results."""
        if not tools_executed:
            return "No tools were executed."
        
        summary = []
        for tool in tools_executed:
            tool_name = tool.get("tool_name", "unknown")
            output = tool.get("output", "No output")
            status = tool.get("status", "unknown")
            
            if status == "success":
                summary.append(f"✓ {tool_name}: {output}")
            else:
                summary.append(f"✗ {tool_name} failed: {output}")
        
        return "\n".join(summary)