import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from sonika_langchain_bot.planner_dynamic.state import PlannerState

class SearchQuery(BaseModel):
    query: str = Field(description="The search query to execute.")

class ResearchNode:
    """
    Executes search/knowledge retrieval tools.
    Implements a self-correction loop: if no results, reformulates the query.
    """

    MAX_RETRIES = 2

    def __init__(self, model, tools: List[BaseTool], logger=None):
        self.model = model
        self.tools = tools
        self.logger = logger
        self.tool_map = {t.name: t for t in tools}

    async def __call__(self, state: PlannerState) -> Dict[str, Any]:
        if not self.tools:
            if self.logger:
                self.logger.warning("ResearchNode visited but no tools available.")
            return {"logs": ["ResearchNode: No tools available, skipping."]}

        user_input = state.get("user_input", "")
        context_logs = state.get("logs", [])

        # Simple loop for retry
        attempt = 0
        results = []
        logs_to_add = []
        executed_tools_info = []

        current_query_intent = user_input

        while attempt <= self.MAX_RETRIES:
            attempt += 1

            # 1. Decide if/what to search
            search_decision = await self._generate_search_query(current_query_intent, attempt, state)

            if not search_decision:
                logs_to_add.append(f"Research attempt {attempt}: Model decided not to search.")
                break

            tool_name = search_decision.get("tool_name") # Assumes model picks a tool name
            query = search_decision.get("query")

            # Default to first tool if name not clear, or pick specific logic
            target_tool = self.tool_map.get(tool_name)
            if not target_tool:
                # Fallback: use the first available search tool
                target_tool = self.tools[0]

            if self.logger:
                self.logger.info(f"Researching: {query} using {target_tool.name} (Attempt {attempt})")

            # 2. Execute
            try:
                # Some tools expect string, some JSON. Adapting generically.
                # Assuming standard LangChain tools usually take a string or dict.
                # Here we pass string for search tools typically.
                tool_output = await target_tool.ainvoke(query)
            except Exception as e:
                tool_output = f"Error executing {target_tool.name}: {str(e)}"

            # Record execution
            logs_to_add.append(f"Research executed: {target_tool.name}('{query}') -> {str(tool_output)[:100]}...")
            executed_tools_info.append({
                "name": target_tool.name,
                "input": {"query": query},
                "output": str(tool_output),
                "status": "success"
            })

            # 3. Validate results
            if self._is_valid_result(str(tool_output)):
                results.append(f"Source ({target_tool.name}): {tool_output}")
                break # Success!
            else:
                logs_to_add.append(f"Research attempt {attempt}: No relevant results found.")
                # Update intent for next loop (reformulate)
                current_query_intent = f"Previous search for '{query}' yielded no results. Try a broader term or synonym for: {user_input}"

        return {
            "research_results": results,
            "logs": logs_to_add,
            "tools_executed": executed_tools_info
        }

    async def _generate_search_query(self, context_input: str, attempt: int, state: PlannerState) -> Optional[Dict[str, str]]:
        """
        Uses LLM to decide what to search.
        """
        system_prompt = """You are a Research Specialist.
Your task is to generate a search query to find information relevant to the user's request.
Select the most appropriate tool from the list below.

Available Tools:
{tool_descriptions}

Output JSON format:
{{
    "tool_name": "name_of_tool",
    "query": "search terms"
}}

If no search is needed, return null.
"""
        tool_descs = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context: {context_input}\nAttempt: {attempt}"),
        ])

        chain = prompt | self.model | JsonOutputParser()

        try:
            return await chain.ainvoke({
                "tool_descriptions": tool_descs,
                "context_input": context_input,
                "attempt": attempt
            })
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating search query: {e}")
            return None

    def _is_valid_result(self, output: str) -> bool:
        """Heuristic to check if search result is empty/useless."""
        invalid_markers = [
            "no results", "not found", "no information", "empty", "null", "none",
            "no se encontraron", "sin resultados"
        ]
        lower_out = output.lower().strip()
        if not lower_out:
            return False
        for marker in invalid_markers:
            if marker in lower_out:
                return False
        return True
