"""Multi-agent collaboration system for distributed test generation.

This module provides:
- AgentCoordinator: Central coordinator for multi-agent collaboration
- SpecializedAgent: Base class for specialized agents
- MessageBus: Communication infrastructure between agents
- SharedKnowledgeBase: Knowledge sharing mechanism
- CodeAnalysisAgent: Specialized agent for code analysis
- TestGenerationAgent: Specialized agent for test generation
- TestFixAgent: Specialized agent for fixing test errors
"""

from .agent_coordinator import AgentCoordinator, TaskAllocation, AgentRole
from .specialized_agent import SpecializedAgent, AgentCapability, AgentTask, TaskResult
from .message_bus import MessageBus, AgentMessage, MessageType
from .shared_knowledge import SharedKnowledgeBase, ExperienceReplay
from .code_analysis_agent import CodeAnalysisAgent
from .test_generation_agent import TestGenerationAgent
from .test_fix_agent import TestFixAgent

__all__ = [
    'AgentCoordinator',
    'TaskAllocation',
    'AgentRole',
    'SpecializedAgent',
    'AgentCapability',
    'AgentTask',
    'TaskResult',
    'MessageBus',
    'AgentMessage',
    'MessageType',
    'SharedKnowledgeBase',
    'ExperienceReplay',
    'CodeAnalysisAgent',
    'TestGenerationAgent',
    'TestFixAgent',
]
