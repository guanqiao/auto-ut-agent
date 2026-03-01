"""Subagents mechanism for specialized task handling.

This module provides:
- SubAgent: Independent agent for specialized tasks
- SubAgentManager: Management of subagents
- AgentPool: Pool of reusable agents
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of a subagent."""
    IDLE = auto()
    BUSY = auto()
    WAITING = auto()
    ERROR = auto()
    TERMINATED = auto()


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Task:
    """A task to be executed by a subagent."""
    id: str
    name: str
    description: str
    input_data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.completed_at is not None


@dataclass
class AgentCapability:
    """Capability of a subagent."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentConfig:
    """Configuration for a subagent."""
    name: str
    agent_type: str
    description: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_concurrent_tasks: int = 1
    timeout: int = 300
    auto_restart: bool = True
    max_retries: int = 3


class SubAgent(ABC):
    """Base class for subagents.

    Subagents are independent agents that can handle specific
    types of tasks with their own context.
    """

    def __init__(self, config: SubAgentConfig):
        """Initialize subagent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.id = str(uuid4())
        self.status = AgentStatus.IDLE
        self.current_task: Optional[Task] = None
        self.task_history: List[Task] = []
        self._running = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the subagent.

        Returns:
            True if initialization succeeded
        """
        pass

    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a task.

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        pass

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources."""
        pass

    async def process_task(self, task: Task):
        """Process a task.

        Args:
            task: Task to process
        """
        self.status = AgentStatus.BUSY
        self.current_task = task
        task.started_at = datetime.now()

        try:
            logger.info(f"[SubAgent {self.config.name}] Processing task: {task.name}")

            for attempt in range(self.config.max_retries):
                try:
                    result = await asyncio.wait_for(
                        self.execute_task(task),
                        timeout=self.config.timeout
                    )
                    task.result = result
                    task.completed_at = datetime.now()
                    logger.info(f"[SubAgent {self.config.name}] Task completed: {task.name}")
                    break

                except asyncio.TimeoutError:
                    if attempt == self.config.max_retries - 1:
                        raise
                    logger.warning(f"[SubAgent {self.config.name}] Task timed out, retrying...")

        except Exception as e:
            logger.exception(f"[SubAgent {self.config.name}] Task failed: {e}")
            task.error = str(e)
            task.completed_at = datetime.now()
            self.status = AgentStatus.ERROR

        finally:
            self.task_history.append(task)
            self.current_task = None
            if self.status != AgentStatus.ERROR:
                self.status = AgentStatus.IDLE

    @property
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (
            self.status == AgentStatus.IDLE
            and len([t for t in self.task_history if not t.is_complete]) < self.config.max_concurrent_tasks
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.

        Returns:
            Statistics dictionary
        """
        completed = [t for t in self.task_history if t.is_complete and not t.error]
        failed = [t for t in self.task_history if t.error]

        return {
            "id": self.id,
            "name": self.config.name,
            "type": self.config.agent_type,
            "status": self.status.name,
            "total_tasks": len(self.task_history),
            "completed_tasks": len(completed),
            "failed_tasks": len(failed),
            "success_rate": len(completed) / len(self.task_history) if self.task_history else 0
        }


class SubAgentManager:
    """Manager for subagents."""

    def __init__(self):
        """Initialize manager."""
        self._agents: Dict[str, SubAgent] = {}
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        logger.debug("[SubAgentManager] Initialized")

    def register_agent(self, agent: SubAgent):
        """Register a subagent.

        Args:
            agent: SubAgent to register
        """
        self._agents[agent.id] = agent
        logger.info(f"[SubAgentManager] Registered agent: {agent.config.name}")

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister a subagent.

        Args:
            agent_id: Agent ID

        Returns:
            True if agent was unregistered
        """
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            logger.info(f"[SubAgentManager] Unregistered agent: {agent.config.name}")
            return True
        return False

    def get_agent(self, name: str) -> Optional[SubAgent]:
        """Get agent by name.

        Args:
            name: Agent name

        Returns:
            SubAgent or None
        """
        for agent in self._agents.values():
            if agent.config.name == name:
                return agent
        return None

    def get_available_agent(self, agent_type: Optional[str] = None) -> Optional[SubAgent]:
        """Get an available agent.

        Args:
            agent_type: Optional agent type filter

        Returns:
            Available SubAgent or None
        """
        available = [a for a in self._agents.values() if a.is_available]
        
        if agent_type:
            available = [a for a in available if a.config.agent_type == agent_type]
        
        return available[0] if available else None

    async def submit_task(
        self,
        name: str,
        description: str,
        input_data: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Submit a task to be processed.

        Args:
            name: Task name
            description: Task description
            input_data: Task input data
            priority: Task priority

        Returns:
            Task ID
        """
        task = Task(
            id=str(uuid4()),
            name=name,
            description=description,
            input_data=input_data,
            priority=priority
        )

        await self._task_queue.put((priority.value, task))
        logger.info(f"[SubAgentManager] Submitted task: {task.id}")
        
        return task.id

    async def start(self):
        """Start the manager."""
        self._running = True
        logger.info("[SubAgentManager] Started")

    async def stop(self):
        """Stop the manager."""
        self._running = False
        logger.info("[SubAgentManager] Stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_agents": len(self._agents),
            "available_agents": len([a for a in self._agents.values() if a.is_available]),
            "queued_tasks": self._task_queue.qsize(),
            "agents": [a.get_stats() for a in self._agents.values()]
        }


class AgentPool:
    """Pool of reusable subagents."""

    def __init__(self, max_size: int = 5):
        """Initialize agent pool.

        Args:
            max_size: Maximum pool size
        """
        self.max_size = max_size
        self._available: List[SubAgent] = []
        self._busy: List[SubAgent] = []
        logger.debug(f"[AgentPool] Initialized with max_size={max_size}")

    def add_agent(self, agent: SubAgent):
        """Add an agent to the pool.

        Args:
            agent: Agent to add
        """
        if len(self._available) + len(self._busy) < self.max_size:
            self._available.append(agent)
            logger.info(f"[AgentPool] Added agent: {agent.config.name}")
        else:
            logger.warning(f"[AgentPool] Pool is full, cannot add agent")

    async def acquire(self, timeout: float = 30) -> Optional[SubAgent]:
        """Acquire an agent from the pool.

        Args:
            timeout: Acquisition timeout in seconds

        Returns:
            Acquired agent or None
        """
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            if self._available:
                agent = self._available.pop()
                self._busy.append(agent)
                logger.debug(f"[AgentPool] Acquired agent: {agent.config.name}")
                return agent
            
            await asyncio.sleep(0.1)
        
        logger.warning(f"[AgentPool] Failed to acquire agent within {timeout}s")
        return None

    async def release(self, agent: SubAgent):
        """Release an agent back to the pool.

        Args:
            agent: Agent to release
        """
        if agent in self._busy:
            self._busy.remove(agent)
            self._available.append(agent)
            logger.debug(f"[AgentPool] Released agent: {agent.config.name}")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "max_size": self.max_size,
            "available": len(self._available),
            "busy": len(self._busy),
            "utilization": len(self._busy) / self.max_size if self.max_size > 0 else 0
        }


class SpecializedSubAgent(SubAgent):
    """Generic specialized subagent implementation."""

    def __init__(
        self,
        config: SubAgentConfig,
        handler: Callable[[Task], Any]
    ):
        """Initialize specialized subagent.

        Args:
            config: Agent configuration
            handler: Task handler function
        """
        super().__init__(config)
        self._handler = handler

    async def initialize(self) -> bool:
        """Initialize the subagent."""
        logger.info(f"[SpecializedSubAgent] Initializing: {self.config.name}")
        return True

    async def execute_task(self, task: Task) -> Any:
        """Execute a task using the handler."""
        return await self._handler(task)

    async def cleanup(self):
        """Cleanup resources."""
        logger.info(f"[SpecializedSubAgent] Cleaning up: {self.config.name}")


def create_subagent_manager() -> SubAgentManager:
    """Create a subagent manager.

    Returns:
        SubAgentManager
    """
    return SubAgentManager()


def create_specialized_agent(
    name: str,
    agent_type: str,
    description: str,
    handler: Callable[[Task], Any],
    capabilities: Optional[List[AgentCapability]] = None
) -> SpecializedSubAgent:
    """Create a specialized subagent.

    Args:
        name: Agent name
        agent_type: Agent type
        description: Agent description
        handler: Task handler
        capabilities: Agent capabilities

    Returns:
        SpecializedSubAgent
    """
    config = SubAgentConfig(
        name=name,
        agent_type=agent_type,
        description=description,
        capabilities=capabilities or []
    )

    return SpecializedSubAgent(config, handler)
