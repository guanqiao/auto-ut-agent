"""PyUT Agent - AI-powered Java Unit Test Generator

This package provides an intelligent agent for generating Java unit tests
using Large Language Models (LLMs).

Quick Start:
    >>> from pyutagent import ReActAgent, TestGeneratorAgent
    >>> agent = TestGeneratorAgent(project_path="/path/to/project")
    >>> result = await agent.generate_tests("com.example.MyClass")

Main Components:
    - ReActAgent: Core agent implementing ReAct pattern
    - EnhancedAgent: Advanced agent with P0/P1/P2/P3 capabilities
    - TestGeneratorAgent: High-level interface for test generation
    - IntegrationManager: Manages component lifecycle

Example:
    Basic usage with EnhancedAgent:
    
    >>> from pyutagent import EnhancedAgent, EnhancedAgentConfig
    >>> from pyutagent.llm import LLMClient
    >>> from pyutagent.memory import WorkingMemory
    >>> 
    >>> config = EnhancedAgentConfig(
    ...     model_name="gpt-4",
    ...     enable_multi_agent=True,
    ...     enable_error_prediction=True
    ... )
    >>> agent = EnhancedAgent(
    ...     llm_client=LLMClient(),
    ...     working_memory=WorkingMemory(),
    ...     project_path="/path/to/project",
    ...     config=config
    ... )
    >>> result = await agent.generate_tests("TargetClass.java")
"""

__version__ = "0.1.0"

# Core agent classes
from .agent import (
    ReActAgent,
    EnhancedAgent,
    EnhancedAgentConfig,
    TestGeneratorAgent,
    IntegrationManager,
)

# Core protocols and types
from .core.protocols import (
    AgentState,
    AgentResult,
    ComponentStatus,
    TestResult,
    CoverageResult,
    ClassInfo,
    MethodInfo,
)

# LLM client
from .llm.client import LLMClient

# Memory
from .memory.working_memory import WorkingMemory

__all__ = [
    # Version
    "__version__",
    # Core agents
    "ReActAgent",
    "EnhancedAgent",
    "EnhancedAgentConfig",
    "TestGeneratorAgent",
    "IntegrationManager",
    # Protocols
    "AgentState",
    "AgentResult",
    "ComponentStatus",
    "TestResult",
    "CoverageResult",
    "ClassInfo",
    "MethodInfo",
    # Infrastructure
    "LLMClient",
    "WorkingMemory",
]
