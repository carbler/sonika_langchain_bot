from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
import json
from enum import Enum
from langchain.tools import Tool
import inspect

class BeliefType(Enum):
    CORE = "core"              # Creencias fundamentales sobre sí mismo
    PERSONALITY = "personality" # Rasgos de personalidad
    USER = "user"              # Creencias sobre el usuario
    CONTEXT = "context"        # Creencias sobre el contexto actual
    KNOWLEDGE = "knowledge"    # Conocimiento general
    TEMPORAL = "temporal"      # Creencias temporales
    TOOLS = "tools"      # Creencias temporales


@dataclass
class Belief:
    """Representa una creencia individual del bot."""
    content: str
    type: BeliefType
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "initial"

@dataclass
class Desire:
    """Representa un deseo o objetivo del bot."""
    description: str
    priority: int
    conditions: List[str]
    achieved: bool = False

@dataclass
class Intention:
    """Representa una intención o plan de acción del bot."""
    description: str
    steps: List[str]
    related_desire: str
    completed: bool = False

class BotBeliefSystem:
    """Sistema principal de creencias del bot."""
    
    def __init__(self, bot_name: str, bot_version: str, tools: List[Tool], beliefs_init: List[Belief]):
        self.bot_name = bot_name
        self.bot_version = bot_version
        self.beliefs_init = beliefs_init
        self.beliefs: Dict[BeliefType, Dict[str, Belief]] = {belief_type: {} for belief_type in BeliefType}
        self.desires: List[Desire] = []
        self.intentions: List[Intention] = []
        self.conversation_history: List[Dict] = []
        self._initialize_core_beliefs()
        self.add_tools_beliefs(tools)
        self.set_beliefs_init()
    
    def set_beliefs_init(self):
        for belief in self.beliefs_init:
            self.add_belief(belief)
        
    def _initialize_core_beliefs(self):
        """Initializes the core beliefs of the bot."""
        core_beliefs = [
            # Add more core beliefs as needed
        ]
        
        for belief in core_beliefs:
            self.add_belief(belief)

    def add_tools_beliefs(self, tools: List[Tool]):

        if len(tools) == 0:
            return
        """Carga y procesa las herramientas disponibles para el bot."""
        # Instrucciones sobre el uso de herramientas en inglés
        instrucciones_de_uso = '''\nWhen you want to execute a tool, enclose the command with three asterisks and provide all parameters needed.
                                \nEnsure you gather all relevant information from the conversation to use the parameters.
                                \nIf information is missing, search online.'''
        
        # Agregar un encabezado para la lista de herramientas
        instrucciones_de_uso = instrucciones_de_uso + "\nThis is a list of the tools you can execute:"
        # Almacenar las instrucciones de uso como una creencia
        self.add_belief(Belief(content=instrucciones_de_uso, type=BeliefType.TOOLS))

        # Procesar cada herramienta y almacenarla como creencia
        for tool in tools:
            tool_name = tool.name
            tool_description = tool.description
            
            # Crear un texto para la herramienta
            tool_text = f"Tool Name: {tool_name}\nDescription: {tool_description}\n"
            run_method = getattr(tool, '_run', None)
            
            if run_method:
                params = inspect.signature(run_method)
                tool_text += f"Parameters: {params}\n"
            else:
                tool_text += "No _run method found.\n"
            
            tool_text += "\n---\n"
            
            # Almacenar la herramienta como una creencia
            self.add_belief(Belief(content=tool_text.strip(), type=BeliefType.TOOLS))

        
    def add_belief(self, belief: Belief):
        """Agrega o actualiza una creencia."""
        self.beliefs[belief.type][str(datetime.now())] = belief
    
    def get_beliefs_by_type(self, belief_type: BeliefType) -> List[Belief]:
        """Obtiene todas las creencias de un tipo específico."""
        return list(self.beliefs[belief_type].values())
    
    def add_desire(self, desire: Desire):
        """Agrega un nuevo deseo."""
        self.desires.append(desire)
        self.desires.sort(key=lambda x: x.priority, reverse=True)
    
    def add_intention(self, intention: Intention):
        """Agrega una nueva intención."""
        self.intentions.append(intention)
    
    def update_user_belief(self, key: str, value: str):
        """Actualiza una creencia sobre el usuario."""
        belief = Belief(
            content=f"{key}: {value}",
            type=BeliefType.USER,
            confidence=0.8,
            source="user_interaction"
        )
        self.add_belief(belief)
    
    def get_current_context(self) -> str:
        """Obtiene el contexto actual basado en las creencias."""
        context_beliefs = self.get_beliefs_by_type(BeliefType.CONTEXT)
        return "\n".join([belief.content for belief in context_beliefs])
    
    def generate_prompt_context(self) -> str:
        """Genera el contexto para el prompt, incluyendo creencias y guías sobre cómo el agente debe utilizar esta información."""
        context_parts = []

        # Instrucciones base para el agente BDI
        instructions = [
            '''''',
        ]
        context_parts.extend(instructions)

        # Agregar información de creencias núcleo (Core Beliefs)
        core_beliefs = self.get_beliefs_by_type(BeliefType.CORE)
        context_parts.append("\nCore Beliefs:")
        context_parts.extend([f"- {belief.content}" for belief in core_beliefs])

        # Agregar información de personalidad (Personality Traits)
        personality_beliefs = self.get_beliefs_by_type(BeliefType.PERSONALITY)
        if personality_beliefs:
            # Agregar una sección de escala de intensidad y los rasgos de personalidad al contexto
            context_parts.append("\nPersonality Traits:\n")
            # Agregar los rasgos de personalidad de la lista
            context_parts.extend([f"- {belief.content}" for belief in personality_beliefs])

        # Agregar información del usuario (User Information)
        user_beliefs = self.get_beliefs_by_type(BeliefType.USER)
        if user_beliefs:
            context_parts.append("\nUser Information:")
            context_parts.extend([f"- {belief.content}" for belief in user_beliefs])

        user_beliefs = self.get_beliefs_by_type(BeliefType.TOOLS)
        if user_beliefs:
            context_parts.append("\nTools Information:")
            context_parts.extend([f"- {belief.content}" for belief in user_beliefs])

        # Generación final del prompt
        return "\n".join(context_parts)