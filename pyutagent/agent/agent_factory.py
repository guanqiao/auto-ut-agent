"""Agent Meta-Programming Framework.

This module provides:
- Dynamic agent creation
- Agent configuration management
- Agent template system
- Agent factory pattern
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents."""
    REACT = "react"
    ENHANCED = "enhanced"
    UNIVERSAL = "universal"
    TEST_GENERATOR = "test_generator"
    DEBUGGER = "debugger"
    REVIEWER = "reviewer"
    CUSTOM = "custom"


class AgentScope(Enum):
    """Agent scope."""
    TASK = "task"
    PROJECT = "project"
    SESSION = "session"
    GLOBAL = "global"


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    agent_type: AgentType
    scope: AgentScope = AgentScope.TASK
    max_iterations: int = 50
    timeout_seconds: int = 300
    enable_memory: bool = True
    enable_tools: bool = True
    enable_learning: bool = False
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetadata:
    """Agent metadata."""
    agent_id: str
    name: str
    agent_type: AgentType
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class AgentTemplate:
    """Agent template for creating agents."""
    name: str
    description: str
    agent_type: AgentType
    capabilities: List[str]
    default_config: Dict[str, Any]
    tool_requirements: List[str]
    memory_requirements: List[str]


class AgentRegistry:
    """Registry for agent types and templates.

    Features:
    - Register agent types
    - Register agent templates
    - Create agents from templates
    - Track agent instances
    """

    def __init__(self):
        """Initialize agent registry."""
        self._agent_types: Dict[AgentType, Type] = {}
        self._templates: Dict[str, AgentTemplate] = {}
        self._instances: Dict[str, AgentMetadata] = {}

    def register_agent_type(
        self,
        agent_type: AgentType,
        agent_class: Type
    ):
        """Register an agent type.

        Args:
            agent_type: Agent type
            agent_class: Agent class
        """
        self._agent_types[agent_type] = agent_class
        logger.info(f"[AgentRegistry] Registered agent type: {agent_type.value}")

    def register_template(self, template: AgentTemplate):
        """Register an agent template.

        Args:
            template: Agent template
        """
        self._templates[template.name] = template
        logger.info(f"[AgentRegistry] Registered template: {template.name}")

    def get_template(self, name: str) -> Optional[AgentTemplate]:
        """Get template by name.

        Args:
            name: Template name

        Returns:
            Template or None
        """
        return self._templates.get(name)

    def list_templates(self) -> List[AgentTemplate]:
        """List all templates.

        Returns:
            List of templates
        """
        return list(self._templates.values())

    def create_agent(
        self,
        config: AgentConfig,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        memory: Optional[Any] = None
    ) -> Any:
        """Create an agent from config.

        Args:
            config: Agent configuration
            llm_client: LLM client
            tool_registry: Tool registry
            memory: Memory system

        Returns:
            Agent instance
        """
        if config.agent_type not in self._agent_types:
            raise ValueError(f"Unknown agent type: {config.agent_type}")

        agent_class = self._agent_types[config.agent_type]

        agent = agent_class(
            name=config.name,
            max_iterations=config.max_iterations,
            timeout=config.timeout_seconds
        )

        if llm_client:
            agent.set_llm_client(llm_client)
        if tool_registry:
            agent.set_tool_registry(tool_registry)
        if memory:
            agent.set_memory(memory)

        metadata = AgentMetadata(
            agent_id=config.name,
            name=config.name,
            agent_type=config.agent_type,
            created_at=datetime.now()
        )
        self._instances[config.name] = metadata

        logger.info(f"[AgentRegistry] Created agent: {config.name}")
        return agent

    def create_from_template(
        self,
        template_name: str,
        agent_name: str,
        **kwargs
    ) -> Any:
        """Create agent from template.

        Args:
            template_name: Template name
            agent_name: Agent name
            **kwargs: Additional config

        Returns:
            Agent instance
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        config = AgentConfig(
            name=agent_name,
            agent_type=template.agent_type,
            custom_config={**template.default_config, **kwargs}
        )

        return self.create_agent(config)

    def get_agent_metadata(self, agent_name: str) -> Optional[AgentMetadata]:
        """Get agent metadata.

        Args:
            agent_name: Agent name

        Returns:
            Metadata or None
        """
        return self._instances.get(agent_name)

    def list_agents(self) -> List[AgentMetadata]:
        """List all agent instances.

        Returns:
            List of metadata
        """
        return list(self._instances.values())


class AgentPool:
    """Agent pool for managing multiple agents.

    Features:
    - Pool agents for reuse
    - Agent lifecycle management
    - Resource allocation
    """

    def __init__(self, max_size: int = 10):
        """Initialize agent pool.

        Args:
            max_size: Maximum pool size
        """
        self.max_size = max_size
        self._available: List[Any] = []
        self._in_use: Dict[str, Any] = {}
        self._registry = AgentRegistry()

    def acquire(
        self,
        config: AgentConfig,
        llm_client: Optional[Any] = None,
        tool_registry: Optional[Any] = None,
        memory: Optional[Any] = None
    ) -> Any:
        """Acquire an agent from pool.

        Args:
            config: Agent config
            llm_client: LLM client
            tool_registry: Tool registry
            memory: Memory

        Returns:
            Agent instance
        """
        if self._available:
            agent = self._available.pop()
            self._in_use[config.name] = agent
            return agent

        if len(self._in_use) + len(self._available) >= self.max_size:
            raise RuntimeError("Agent pool exhausted")

        agent = self._registry.create_agent(
            config, llm_client, tool_registry, memory
        )
        self._in_use[config.name] = agent
        return agent

    def release(self, agent_name: str):
        """Release agent back to pool.

        Args:
            agent_name: Agent name
        """
        if agent_name in self._in_use:
            agent = self._in_use.pop(agent_name)
            if len(self._available) < self.max_size:
                self._available.append(agent)

    def shutdown(self):
        """Shutdown the pool."""
        self._available.clear()
        self._in_use.clear()


class AgentFactory:
    """Factory for creating configured agents."""

    def __init__(self):
        """Initialize agent factory."""
        self._registry = AgentRegistry()
        self._pool = AgentPool()

    @property
    def registry(self) -> AgentRegistry:
        """Get registry."""
        return self._registry

    def register_default_templates(self):
        """Register default templates."""
        templates = [
            AgentTemplate(
                name="test_generator",
                description="Generate unit tests",
                agent_type=AgentType.TEST_GENERATOR,
                capabilities=["test_generation", "code_understanding"],
                default_config={"max_iterations": 30},
                tool_requirements=["read_file", "write_file", "run_command"],
                memory_requirements=["working_memory"]
            ),
            AgentTemplate(
                name="code_reviewer",
                description="Review code quality",
                agent_type=AgentType.REVIEWER,
                capabilities=["code_analysis", "quality_check"],
                default_config={"max_iterations": 20},
                tool_requirements=["read_file", "grep"],
                memory_requirements=["working_memory", "short_term_memory"]
            ),
            AgentTemplate(
                name="debugger",
                description="Debug and fix bugs",
                agent_type=AgentType.DEBUGGER,
                capabilities=["error_analysis", "code_fixing"],
                default_config={"max_iterations": 40},
                tool_requirements=["read_file", "run_command", "grep"],
                memory_requirements=["working_memory"]
            ),
        ]

        for template in templates:
            self._registry.register_template(template)

    def create_agent(
        self,
        agent_type: AgentType,
        name: str,
        **config
    ) -> Any:
        """Create an agent.

        Args:
            agent_type: Agent type
            name: Agent name
            **config: Configuration

        Returns:
            Agent instance
        """
        agent_config = AgentConfig(
            name=name,
            agent_type=agent_type,
            custom_config=config
        )

        return self._registry.create_agent(agent_config)

    def create_from_template(
        self,
        template_name: str,
        agent_name: str,
        **kwargs
    ) -> Any:
        """Create agent from template.

        Args:
            template_name: Template name
            agent_name: Agent name
            **kwargs: Additional config

        Returns:
            Agent instance
        """
        return self._registry.create_from_template(
            template_name, agent_name, **kwargs
        )

    def get_pool(self) -> AgentPool:
        """Get agent pool."""
        return self._pool


def create_agent_factory() -> AgentFactory:
    """Create agent factory with defaults.

    Returns:
        AgentFactory instance
    """
    factory = AgentFactory()
    factory.register_default_templates()
    return factory
