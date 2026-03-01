"""Multi-agent collaboration system for distributed test generation.

This module provides:
- AgentCoordinator: Central coordinator for multi-agent collaboration
- SpecializedAgent: Base class for specialized agents
- MessageBus: Communication infrastructure between agents
- SharedKnowledgeBase: Knowledge sharing mechanism
"""

from .agent_coordinator import AgentCoordinator, TaskAllocation, AgentRole
from .specialized_agent import SpecializedAgent, AgentCapability
from .message_bus import MessageBus, AgentMessage, MessageType
from .shared_knowledge import SharedKnowledgeBase, ExperienceReplay

__all__ = [
    'AgentCoordinator',
    'TaskAllocation',
    'AgentRole',
    'SpecializedAgent',
    'AgentCapability',
    'MessageBus',
    'AgentMessage',
    'MessageType',
    'SharedKnowledgeBase',
    'ExperienceReplay',
]
