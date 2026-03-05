"""SubAgent Factory for dynamic agent creation and lifecycle management.

This module provides:
- SubAgentFactory: Factory for creating and managing SubAgents
- Agent templates for common use cases
- Pool management for agent reuse
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import uuid4

from .subagent_base import (
    SubAgent,
    SubAgentConfig,
    AgentStatus,
    Task,
    AgentCapability,
)
from .delegating_subagent import DelegatingSubAgent, DelegationMode
from .skills import Skill, SkillRegistry, get_skill_registry

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Predefined agent types."""
    GENERIC = "generic"
    TEST_GENERATOR = "test_generator"
    CODE_REVIEWER = "code_reviewer"
    BUG_FIXER = "bug_fixer"
    REFACTORER = "refactorer"
    DOC_GENERATOR = "doc_generator"
    ANALYZER = "analyzer"
    DEBUGGER = "debugger"
    CUSTOM = "custom"


@dataclass
class AgentTemplate:
    """Template for creating agents."""
    name: str
    agent_type: AgentType
    description: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    default_skills: List[str] = field(default_factory=list)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    llm_required: bool = True
    tool_service_required: bool = True


AGENT_TEMPLATES: Dict[AgentType, AgentTemplate] = {
    AgentType.GENERIC: AgentTemplate(
        name="Generic Agent",
        agent_type=AgentType.GENERIC,
        description="General-purpose agent for various tasks",
        capabilities=[],
        default_skills=[],
        llm_required=True,
        tool_service_required=True
    ),
    AgentType.TEST_GENERATOR: AgentTemplate(
        name="Test Generator",
        agent_type=AgentType.TEST_GENERATOR,
        description="Agent specialized in generating unit tests",
        capabilities=[
            AgentCapability(name="test_generation", description="Generate unit tests"),
            AgentCapability(name="mock_creation", description="Create mocks and stubs"),
        ],
        default_skills=["generate_unit_test", "debug_test"],
        llm_required=True,
        tool_service_required=True
    ),
    AgentType.CODE_REVIEWER: AgentTemplate(
        name="Code Reviewer",
        agent_type=AgentType.CODE_REVIEWER,
        description="Agent specialized in code review and analysis",
        capabilities=[
            AgentCapability(name="code_review", description="Review code quality"),
            AgentCapability(name="static_analysis", description="Run static analysis"),
        ],
        default_skills=["analyze_code"],
        llm_required=True,
        tool_service_required=True
    ),
    AgentType.BUG_FIXER: AgentTemplate(
        name="Bug Fixer",
        agent_type=AgentType.BUG_FIXER,
        description="Agent specialized in fixing bugs",
        capabilities=[
            AgentCapability(name="bug_fixing", description="Fix bugs in code"),
            AgentCapability(name="error_analysis", description="Analyze error messages"),
        ],
        default_skills=["fix_compilation_error", "debug_test"],
        llm_required=True,
        tool_service_required=True
    ),
    AgentType.REFACTORER: AgentTemplate(
        name="Refactorer",
        agent_type=AgentType.REFACTORER,
        description="Agent specialized in code refactoring",
        capabilities=[
            AgentCapability(name="refactoring", description="Refactor code"),
            AgentCapability(name="code_improvement", description="Improve code quality"),
        ],
        default_skills=["refactor_code"],
        llm_required=True,
        tool_service_required=True
    ),
    AgentType.DOC_GENERATOR: AgentTemplate(
        name="Documentation Generator",
        agent_type=AgentType.DOC_GENERATOR,
        description="Agent specialized in generating documentation",
        capabilities=[
            AgentCapability(name="documentation", description="Generate documentation"),
            AgentCapability(name="javadoc", description="Create Javadoc comments"),
        ],
        default_skills=["generate_doc", "explain_code"],
        llm_required=True,
        tool_service_required=False
    ),
    AgentType.ANALYZER: AgentTemplate(
        name="Code Analyzer",
        agent_type=AgentType.ANALYZER,
        description="Agent specialized in code analysis",
        capabilities=[
            AgentCapability(name="analysis", description="Analyze code structure"),
            AgentCapability(name="metrics", description="Calculate code metrics"),
        ],
        default_skills=["analyze_code"],
        llm_required=True,
        tool_service_required=True
    ),
    AgentType.DEBUGGER: AgentTemplate(
        name="Debugger",
        agent_type=AgentType.DEBUGGER,
        description="Agent specialized in debugging",
        capabilities=[
            AgentCapability(name="debugging", description="Debug code issues"),
            AgentCapability(name="trace_analysis", description="Analyze execution traces"),
        ],
        default_skills=["debug_test", "fix_compilation_error"],
        llm_required=True,
        tool_service_required=True
    ),
}


@dataclass
class AgentPoolConfig:
    """Configuration for agent pool."""
    max_size: int = 10
    min_idle: int = 2
    idle_timeout: int = 300
    max_age: int = 3600


@dataclass
class AgentInfo:
    """Information about a managed agent."""
    agent: DelegatingSubAgent
    agent_type: AgentType
    created_at: datetime
    last_used: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    is_pooled: bool = False


class SubAgentFactory:
    """Factory for creating and managing SubAgents.

    Features:
    - Create agents from templates or custom configurations
    - Manage agent lifecycle (creation, pooling, destruction)
    - Bind skills automatically based on templates
    - Track agent statistics
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tool_service: Optional[Any] = None,
        skill_registry: Optional[SkillRegistry] = None,
        pool_config: Optional[AgentPoolConfig] = None
    ):
        """Initialize SubAgentFactory.

        Args:
            llm_client: Optional LLM client for agents
            tool_service: Optional tool service for agents
            skill_registry: Optional skill registry
            pool_config: Optional pool configuration
        """
        self.llm_client = llm_client
        self.tool_service = tool_service
        self.skill_registry = skill_registry or get_skill_registry()
        self.pool_config = pool_config or AgentPoolConfig()

        self._agents: Dict[str, AgentInfo] = {}
        self._pools: Dict[AgentType, List[DelegatingSubAgent]] = {
            t: [] for t in AgentType
        }
        self._custom_templates: Dict[str, AgentTemplate] = {}

        self._stats = {
            "agents_created": 0,
            "agents_destroyed": 0,
            "pool_hits": 0,
            "pool_misses": 0
        }

        logger.info("[SubAgentFactory] Initialized")

    def register_template(self, template: AgentTemplate) -> None:
        """Register a custom agent template.

        Args:
            template: Template to register
        """
        self._custom_templates[template.name] = template
        logger.info(f"[SubAgentFactory] Registered custom template: {template.name}")

    def create_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
        tool_service: Optional[Any] = None
    ) -> DelegatingSubAgent:
        """Create a SubAgent of the specified type.

        Args:
            agent_type: Type of agent to create
            config: Optional configuration overrides
            llm_client: Optional LLM client (uses factory default if not provided)
            tool_service: Optional tool service (uses factory default if not provided)

        Returns:
            Created DelegatingSubAgent
        """
        try:
            agent_type_enum = AgentType(agent_type)
        except ValueError:
            agent_type_enum = AgentType.CUSTOM

        template = AGENT_TEMPLATES.get(agent_type_enum)

        if not template:
            template = AgentTemplate(
                name=agent_type,
                agent_type=agent_type_enum,
                description=f"Custom agent of type {agent_type}",
                capabilities=[],
                default_skills=[]
            )

        return self._create_from_template(
            template=template,
            config=config,
            llm_client=llm_client or self.llm_client,
            tool_service=tool_service or self.tool_service
        )

    def _create_from_template(
        self,
        template: AgentTemplate,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
        tool_service: Optional[Any] = None
    ) -> DelegatingSubAgent:
        """Create an agent from a template.

        Args:
            template: Agent template
            config: Optional configuration overrides
            llm_client: LLM client
            tool_service: Tool service

        Returns:
            Created DelegatingSubAgent
        """
        config = config or {}
        merged_config = {**template.config_overrides, **config}

        agent_config = SubAgentConfig(
            name=merged_config.get("name", f"{template.name}_{uuid4().hex[:8]}"),
            agent_type=template.agent_type.value,
            description=merged_config.get("description", template.description),
            capabilities=template.capabilities,
            max_concurrent_tasks=merged_config.get("max_concurrent_tasks", 1),
            timeout=merged_config.get("timeout", 300),
            auto_restart=merged_config.get("auto_restart", True),
            max_retries=merged_config.get("max_retries", 3)
        )

        agent = DelegatingSubAgent(
            config=agent_config,
            llm_client=llm_client,
            skill_registry=self.skill_registry,
            tool_service=tool_service
        )

        for skill_name in template.default_skills:
            skill = self.skill_registry.get(skill_name)
            if skill:
                agent.bind_skill(skill)

        agent_info = AgentInfo(
            agent=agent,
            agent_type=template.agent_type,
            created_at=datetime.now()
        )
        self._agents[agent.id] = agent_info

        self._stats["agents_created"] += 1

        logger.info(f"[SubAgentFactory] Created agent: {agent.id} (type={template.agent_type.value})")

        return agent

    def create_from_skill(
        self,
        skill: Skill,
        llm_client: Optional[Any] = None,
        tool_service: Optional[Any] = None
    ) -> DelegatingSubAgent:
        """Create a SubAgent specialized for a specific skill.

        Args:
            skill: Skill to create agent for
            llm_client: Optional LLM client
            tool_service: Optional tool service

        Returns:
            DelegatingSubAgent bound to the skill
        """
        capability = AgentCapability(
            name=skill.name,
            description=skill.metadata.description
        )

        template = AgentTemplate(
            name=f"{skill.name}_agent",
            agent_type=AgentType.CUSTOM,
            description=f"Agent specialized for {skill.name}",
            capabilities=[capability],
            default_skills=[skill.name],
            llm_required=True,
            tool_service_required=bool(skill.metadata.requires_tools)
        )

        agent = self._create_from_template(
            template=template,
            llm_client=llm_client or self.llm_client,
            tool_service=tool_service or self.tool_service
        )

        agent.bind_skill(skill)

        return agent

    def create_specialized(
        self,
        capability: AgentCapability,
        llm_client: Optional[Any] = None,
        tool_service: Optional[Any] = None
    ) -> DelegatingSubAgent:
        """Create a SubAgent with a specific capability.

        Args:
            capability: Capability for the agent
            llm_client: Optional LLM client
            tool_service: Optional tool service

        Returns:
            DelegatingSubAgent with the specified capability
        """
        template = AgentTemplate(
            name=f"{capability.name}_specialist",
            agent_type=AgentType.CUSTOM,
            description=f"Agent specialized in {capability.description}",
            capabilities=[capability],
            default_skills=[]
        )

        matching_skills = self._find_skills_for_capability(capability)
        template.default_skills = matching_skills

        return self._create_from_template(
            template=template,
            llm_client=llm_client or self.llm_client,
            tool_service=tool_service or self.tool_service
        )

    def _find_skills_for_capability(self, capability: AgentCapability) -> List[str]:
        """Find skills that match a capability.

        Args:
            capability: Capability to match

        Returns:
            List of matching skill names
        """
        matching = []
        cap_name_lower = capability.name.lower()
        cap_desc_lower = capability.description.lower()

        for skill_name in self.skill_registry.list_skills():
            skill = self.skill_registry.get(skill_name)
            if not skill:
                continue

            metadata = skill.metadata

            if cap_name_lower in skill_name.lower():
                matching.append(skill_name)
                continue

            for tag in metadata.tags:
                if cap_name_lower in tag.lower() or cap_desc_lower in tag.lower():
                    matching.append(skill_name)
                    break

        return matching

    def get_or_create_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> DelegatingSubAgent:
        """Get an agent from pool or create a new one.

        Args:
            agent_type: Type of agent needed
            config: Optional configuration

        Returns:
            DelegatingSubAgent (may be from pool)
        """
        try:
            agent_type_enum = AgentType(agent_type)
        except ValueError:
            agent_type_enum = AgentType.CUSTOM

        pooled_agent = self._get_from_pool(agent_type_enum)
        if pooled_agent:
            self._stats["pool_hits"] += 1
            return pooled_agent

        self._stats["pool_misses"] += 1
        return self.create_agent(agent_type, config)

    def _get_from_pool(self, agent_type: AgentType) -> Optional[DelegatingSubAgent]:
        """Get an available agent from the pool.

        Args:
            agent_type: Type of agent to get

        Returns:
            Available agent or None
        """
        pool = self._pools.get(agent_type, [])

        for agent in pool:
            if agent.status == AgentStatus.IDLE and agent.is_available:
                if agent.id in self._agents:
                    self._agents[agent.id].last_used = datetime.now()
                logger.debug(f"[SubAgentFactory] Reusing pooled agent: {agent.id}")
                return agent

        return None

    def return_to_pool(self, agent: DelegatingSubAgent) -> bool:
        """Return an agent to the pool for reuse.

        Args:
            agent: Agent to return

        Returns:
            True if agent was returned to pool
        """
        if agent.id not in self._agents:
            return False

        agent_info = self._agents[agent.id]

        pool = self._pools.get(agent_info.agent_type, [])
        if len(pool) >= self.pool_config.max_size:
            logger.debug(f"[SubAgentFactory] Pool full for type {agent_info.agent_type}")
            return False

        if agent.status not in [AgentStatus.IDLE, AgentStatus.WAITING]:
            logger.debug(f"[SubAgentFactory] Agent {agent.id} not idle, cannot pool")
            return False

        if agent not in pool:
            pool.append(agent)
            agent_info.is_pooled = True
            agent_info.last_used = datetime.now()
            logger.info(f"[SubAgentFactory] Returned agent {agent.id} to pool")

        return True

    def destroy_agent(self, agent_id: str) -> bool:
        """Destroy an agent and clean up resources.

        Args:
            agent_id: ID of agent to destroy

        Returns:
            True if agent was destroyed
        """
        if agent_id not in self._agents:
            logger.warning(f"[SubAgentFactory] Agent not found: {agent_id}")
            return False

        agent_info = self._agents[agent_id]
        agent = agent_info.agent

        pool = self._pools.get(agent_info.agent_type, [])
        if agent in pool:
            pool.remove(agent)

        asyncio.create_task(agent.cleanup())

        del self._agents[agent_id]

        self._stats["agents_destroyed"] += 1

        logger.info(f"[SubAgentFactory] Destroyed agent: {agent_id}")

        return True

    def get_agent_pool(self, agent_type: str) -> List[DelegatingSubAgent]:
        """Get all agents in the pool for a specific type.

        Args:
            agent_type: Type of agents to get

        Returns:
            List of agents in the pool
        """
        try:
            agent_type_enum = AgentType(agent_type)
        except ValueError:
            return []

        return self._pools.get(agent_type_enum, []).copy()

    def get_all_agents(self) -> List[DelegatingSubAgent]:
        """Get all managed agents.

        Returns:
            List of all agents
        """
        return [info.agent for info in self._agents.values()]

    def get_agent(self, agent_id: str) -> Optional[DelegatingSubAgent]:
        """Get a specific agent by ID.

        Args:
            agent_id: Agent ID

        Returns:
            Agent or None
        """
        agent_info = self._agents.get(agent_id)
        return agent_info.agent if agent_info else None

    def cleanup_idle_agents(self, max_idle_time: Optional[int] = None) -> int:
        """Clean up idle agents that have been idle too long.

        Args:
            max_idle_time: Maximum idle time in seconds

        Returns:
            Number of agents cleaned up
        """
        max_idle_time = max_idle_time or self.pool_config.idle_timeout
        now = datetime.now()
        cleaned = 0

        to_destroy = []
        for agent_id, agent_info in self._agents.items():
            if agent_info.is_pooled and agent_info.last_used:
                idle_seconds = (now - agent_info.last_used).total_seconds()
                if idle_seconds > max_idle_time:
                    to_destroy.append(agent_id)

        for agent_id in to_destroy:
            if self.destroy_agent(agent_id):
                cleaned += 1

        if cleaned > 0:
            logger.info(f"[SubAgentFactory] Cleaned up {cleaned} idle agents")

        return cleaned

    def record_task_result(self, agent_id: str, success: bool) -> None:
        """Record a task result for an agent.

        Args:
            agent_id: Agent ID
            success: Whether the task succeeded
        """
        if agent_id not in self._agents:
            return

        agent_info = self._agents[agent_id]
        if success:
            agent_info.tasks_completed += 1
        else:
            agent_info.tasks_failed += 1
        agent_info.last_used = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get factory statistics.

        Returns:
            Statistics dictionary
        """
        pool_stats = {}
        for agent_type, pool in self._pools.items():
            pool_stats[agent_type.value] = {
                "pool_size": len(pool),
                "idle": sum(1 for a in pool if a.status == AgentStatus.IDLE)
            }

        return {
            "total_agents": len(self._agents),
            "agents_created": self._stats["agents_created"],
            "agents_destroyed": self._stats["agents_destroyed"],
            "pool_hits": self._stats["pool_hits"],
            "pool_misses": self._stats["pool_misses"],
            "pool_hit_rate": (
                self._stats["pool_hits"] /
                (self._stats["pool_hits"] + self._stats["pool_misses"])
                if (self._stats["pool_hits"] + self._stats["pool_misses"]) > 0
                else 0
            ),
            "pools": pool_stats
        }

    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available agent templates.

        Returns:
            List of template information
        """
        templates = []

        for agent_type, template in AGENT_TEMPLATES.items():
            templates.append({
                "type": agent_type.value,
                "name": template.name,
                "description": template.description,
                "capabilities": [c.name for c in template.capabilities],
                "default_skills": template.default_skills,
                "llm_required": template.llm_required,
                "tool_service_required": template.tool_service_required
            })

        for name, template in self._custom_templates.items():
            templates.append({
                "type": template.agent_type.value,
                "name": template.name,
                "description": template.description,
                "capabilities": [c.name for c in template.capabilities],
                "default_skills": template.default_skills,
                "custom": True
            })

        return templates


def create_subagent_factory(
    llm_client: Optional[Any] = None,
    tool_service: Optional[Any] = None,
    skill_registry: Optional[SkillRegistry] = None,
    max_pool_size: int = 10
) -> SubAgentFactory:
    """Create a SubAgentFactory.

    Args:
        llm_client: Optional LLM client
        tool_service: Optional tool service
        skill_registry: Optional skill registry
        max_pool_size: Maximum pool size per agent type

    Returns:
        SubAgentFactory instance
    """
    pool_config = AgentPoolConfig(max_size=max_pool_size)

    return SubAgentFactory(
        llm_client=llm_client,
        tool_service=tool_service,
        skill_registry=skill_registry,
        pool_config=pool_config
    )
