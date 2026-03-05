"""Subagents mechanism for specialized task handling.

.. deprecated::
    Use pyutagent.agent.unified_agent_base.UnifiedAgentBase instead.
    This module is kept for backward compatibility.

This module provides:
- SubAgent: Independent agent for specialized tasks (deprecated)
- SubAgentManager: Management of subagents (deprecated)
- AgentPool: Pool of reusable agents (deprecated)
"""

import asyncio
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Emit deprecation warning when module is imported
warnings.warn(
    "pyutagent.agent.subagent_base is deprecated. "
    "Use pyutagent.agent.unified_agent_base.UnifiedAgentBase instead.",
    DeprecationWarning,
    stacklevel=2
)


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
    """Manager for subagents with delegation and result aggregation."""

    def __init__(
        self,
        agent_factory: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        tool_service: Optional[Any] = None
    ):
        """Initialize manager.
        
        Args:
            agent_factory: Optional SubAgentFactory for creating agents
            llm_client: Optional LLM client for agents
            tool_service: Optional tool service for agents
        """
        self._agents: Dict[str, SubAgent] = {}
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._running = False
        self._agent_factory = agent_factory
        self._llm_client = llm_client
        self._tool_service = tool_service
        self._task_results: Dict[str, Task] = {}
        self._delegation_callbacks: Dict[str, Callable] = {}
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

    async def delegate_task(
        self,
        task: Task,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Task:
        """Delegate a task to an agent.

        Args:
            task: Task to delegate
            agent_type: Optional agent type to use
            agent_id: Optional specific agent ID

        Returns:
            Task with result populated
        """
        agent = None

        if agent_id:
            agent = self._agents.get(agent_id)
        elif agent_type:
            agent = self.get_available_agent(agent_type)
        else:
            agent = self.get_available_agent()

        if not agent:
            if self._agent_factory:
                agent = self._agent_factory.get_or_create_agent(
                    agent_type or "generic"
                )
                self.register_agent(agent)
            else:
                task.error = f"No available agent for task: {task.name}"
                task.completed_at = datetime.now()
                return task

        logger.info(f"[SubAgentManager] Delegating task {task.id} to agent {agent.id}")

        await agent.process_task(task)

        self._task_results[task.id] = task

        if task.id in self._delegation_callbacks:
            callback = self._delegation_callbacks.pop(task.id)
            callback(task)

        return task

    async def delegate_to_skill(
        self,
        task: Task,
        skill_name: str,
        agent_type: Optional[str] = None
    ) -> Task:
        """Delegate a task to an agent with a specific skill.

        Args:
            task: Task to delegate
            skill_name: Name of skill to use
            agent_type: Optional agent type

        Returns:
            Task with result populated
        """
        task.input_data["skill_name"] = skill_name

        if self._agent_factory:
            from .skills import get_skill_registry
            skill_registry = get_skill_registry()
            skill = skill_registry.get(skill_name)

            if skill:
                agent = self._agent_factory.create_from_skill(
                    skill,
                    llm_client=self._llm_client,
                    tool_service=self._tool_service
                )
                self.register_agent(agent)
                return await self.delegate_task(task, agent_id=agent.id)

        return await self.delegate_task(task, agent_type=agent_type)

    async def broadcast_task(
        self,
        task: Task,
        agent_types: Optional[List[str]] = None
    ) -> List[Task]:
        """Broadcast a task to multiple agents.

        Args:
            task: Task to broadcast
            agent_types: Optional list of agent types to broadcast to

        Returns:
            List of task results from each agent
        """
        agents = []

        if agent_types:
            for agent_type in agent_types:
                agent = self.get_available_agent(agent_type)
                if agent:
                    agents.append(agent)
        else:
            agents = [a for a in self._agents.values() if a.is_available]

        if not agents:
            task.error = "No available agents for broadcast"
            task.completed_at = datetime.now()
            return [task]

        logger.info(f"[SubAgentManager] Broadcasting task {task.id} to {len(agents)} agents")

        tasks = []
        for agent in agents:
            task_copy = Task(
                id=str(uuid4()),
                name=task.name,
                description=task.description,
                input_data=task.input_data.copy(),
                priority=task.priority
            )
            tasks.append((agent, task_copy))

        results = await asyncio.gather(
            *[agent.process_task(t) for agent, t in tasks],
            return_exceptions=True
        )

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tasks[i][1].error = str(result)
                tasks[i][1].completed_at = datetime.now()
            else:
                tasks[i][1] = result
            final_results.append(tasks[i][1])
            self._task_results[tasks[i][1].id] = tasks[i][1]

        return final_results

    def get_or_create_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None
    ) -> SubAgent:
        """Get an available agent or create a new one.

        Args:
            agent_type: Type of agent needed
            config: Optional configuration for new agent

        Returns:
            SubAgent instance
        """
        agent = self.get_available_agent(agent_type)
        if agent:
            return agent

        if self._agent_factory:
            agent = self._agent_factory.create_agent(agent_type, config)
            self.register_agent(agent)
            return agent

        from .delegating_subagent import create_delegating_subagent
        agent = create_delegating_subagent(
            name=f"{agent_type}_{uuid4().hex[:8]}",
            agent_type=agent_type,
            description=f"Auto-created {agent_type} agent",
            llm_client=self._llm_client,
            tool_service=self._tool_service
        )
        self.register_agent(agent)
        return agent

    async def aggregate_results(
        self,
        task_ids: List[str]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple tasks.

        Args:
            task_ids: List of task IDs to aggregate

        Returns:
            Aggregated results dictionary
        """
        results = []
        for task_id in task_ids:
            if task_id in self._task_results:
                results.append(self._task_results[task_id])

        if not results:
            return {"success": False, "error": "No results found"}

        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]

        aggregated = {
            "success": len(successful) > 0 and len(failed) == 0,
            "total_tasks": len(results),
            "successful_tasks": len(successful),
            "failed_tasks": len(failed),
            "results": [
                {
                    "task_id": r.id,
                    "task_name": r.name,
                    "success": r.error is None,
                    "result": r.result,
                    "error": r.error,
                    "duration": r.duration
                }
                for r in results
            ]
        }

        if successful:
            result_values = [r.result for r in successful if r.result is not None]
            if result_values:
                aggregated["combined_result"] = result_values

        return aggregated

    def set_delegation_callback(
        self,
        task_id: str,
        callback: Callable[[Task], None]
    ) -> None:
        """Set a callback for when a task delegation completes.

        Args:
            task_id: Task ID to watch
            callback: Function to call when task completes
        """
        self._delegation_callbacks[task_id] = callback

    def get_task_result(self, task_id: str) -> Optional[Task]:
        """Get the result of a delegated task.

        Args:
            task_id: Task ID

        Returns:
            Task with result or None
        """
        return self._task_results.get(task_id)

    async def execute_parallel(
        self,
        tasks: List[Task],
        max_concurrent: int = 5
    ) -> List[Task]:
        """Execute multiple tasks in parallel.

        Args:
            tasks: List of tasks to execute
            max_concurrent: Maximum concurrent executions

        Returns:
            List of completed tasks
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(task: Task) -> Task:
            async with semaphore:
                return await self.delegate_task(task)

        results = await asyncio.gather(
            *[execute_with_semaphore(t) for t in tasks],
            return_exceptions=True
        )

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tasks[i].error = str(result)
                tasks[i].completed_at = datetime.now()
                final_results.append(tasks[i])
            else:
                final_results.append(result)

        return final_results

    async def execute_sequential(
        self,
        tasks: List[Task]
    ) -> List[Task]:
        """Execute multiple tasks sequentially.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of completed tasks in order
        """
        results = []
        for task in tasks:
            result = await self.delegate_task(task)
            results.append(result)
        return results

    async def cleanup_idle_agents(self, max_idle_time: int = 300) -> int:
        """Clean up idle agents.

        Args:
            max_idle_time: Maximum idle time in seconds

        Returns:
            Number of agents cleaned up
        """
        if self._agent_factory:
            return self._agent_factory.cleanup_idle_agents(max_idle_time)

        cleaned = 0
        to_remove = []

        for agent_id, agent in self._agents.items():
            if agent.status == AgentStatus.IDLE:
                if agent.task_history:
                    last_task = agent.task_history[-1]
                    if last_task.completed_at:
                        idle_seconds = (datetime.now() - last_task.completed_at).total_seconds()
                        if idle_seconds > max_idle_time:
                            to_remove.append(agent_id)

        for agent_id in to_remove:
            self.unregister_agent(agent_id)
            cleaned += 1

        if cleaned > 0:
            logger.info(f"[SubAgentManager] Cleaned up {cleaned} idle agents")

        return cleaned


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
