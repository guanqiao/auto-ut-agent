"""Agent Delegation Mixin for adding delegation capabilities to agents.

This module provides:
- AgentDelegationMixin: Mixin class for task delegation
- Delegation helpers and utilities
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .subagent_base import SubAgent, Task
    from .delegating_subagent import DelegatingSubAgent, DelegationResult
    from .subagent_factory import SubAgentFactory
    from .hierarchical_planner import Subtask

logger = logging.getLogger(__name__)


class DelegationMode(Enum):
    """Modes for task delegation."""
    SYNC = "sync"
    ASYNC = "async"
    FIRE_AND_FORGET = "fire_and_forget"


@dataclass
class DelegationOptions:
    """Options for task delegation."""
    mode: DelegationMode = DelegationMode.SYNC
    timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    progress_callback: Optional[Callable] = None
    result_callback: Optional[Callable] = None
    share_context: bool = True
    agent_type: Optional[str] = None
    skill_name: Optional[str] = None


@dataclass
class DelegationRecord:
    """Record of a delegation operation."""
    delegation_id: str
    subtask_id: str
    agent_id: str
    status: str = "pending"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    retries: int = 0


class AgentDelegationMixin:
    """Mixin class that adds delegation capabilities to agents.

    This mixin provides:
    - Task delegation to SubAgents
    - Parallel delegation support
    - Retry mechanisms
    - Progress tracking
    - Result handling
    """

    _subagent_manager: Optional[Any] = None
    _subagent_factory: Optional["SubAgentFactory"] = None
    _delegation_history: List[DelegationRecord] = []
    _active_delegations: Dict[str, DelegationRecord] = {}

    def init_delegation(
        self,
        subagent_manager: Optional[Any] = None,
        subagent_factory: Optional["SubAgentFactory"] = None,
        llm_client: Optional[Any] = None,
        tool_service: Optional[Any] = None
    ) -> None:
        """Initialize delegation capabilities.

        Args:
            subagent_manager: Optional SubAgentManager
            subagent_factory: Optional SubAgentFactory
            llm_client: Optional LLM client
            tool_service: Optional tool service
        """
        if subagent_manager:
            self._subagent_manager = subagent_manager
        else:
            from .subagents import SubAgentManager
            self._subagent_manager = SubAgentManager(
                agent_factory=subagent_factory,
                llm_client=llm_client,
                tool_service=tool_service
            )

        self._subagent_factory = subagent_factory
        self._delegation_history = []
        self._active_delegations = {}

        logger.info(f"[AgentDelegationMixin] Initialized for {getattr(self, 'agent_id', 'unknown')}")

    async def delegate_subtask(
        self,
        subtask: "Subtask",
        options: Optional[DelegationOptions] = None
    ) -> "DelegationResult":
        """Delegate a subtask to a SubAgent.

        Args:
            subtask: Subtask to delegate
            options: Optional delegation options

        Returns:
            DelegationResult
        """
        options = options or DelegationOptions()

        self._ensure_initialized()

        from .subagents import Task, TaskPriority
        from .delegating_subagent import DelegationContext, DelegationMode as DMode

        task = Task(
            id=subtask.id,
            name=subtask.name,
            description=subtask.description,
            input_data=subtask.input_data,
            priority=TaskPriority(subtask.priority) if subtask.priority <= 3 else TaskPriority.NORMAL
        )

        record = DelegationRecord(
            delegation_id=str(uuid4()),
            subtask_id=subtask.id,
            agent_id=""
        )
        self._active_delegations[record.delegation_id] = record

        try:
            if options.skill_name:
                result_task = await self._subagent_manager.delegate_to_skill(
                    task, options.skill_name, options.agent_type
                )
            else:
                result_task = await self._subagent_manager.delegate_task(
                    task, options.agent_type
                )

            record.agent_id = result_task.result.get("agent_id", "") if isinstance(result_task.result, dict) else ""

            if result_task.error:
                if options.max_retries > 0 and record.retries < options.max_retries:
                    await asyncio.sleep(options.retry_delay)
                    record.retries += 1
                    return await self.delegate_subtask(subtask, options)

                record.status = "failed"
                record.error = result_task.error
                record.completed_at = datetime.now().isoformat()

                from .delegating_subagent import DelegationResult
                return DelegationResult(
                    success=False,
                    task_id=subtask.id,
                    agent_id=record.agent_id,
                    error=result_task.error
                )

            record.status = "completed"
            record.result = result_task.result
            record.completed_at = datetime.now().isoformat()

            from .delegating_subagent import DelegationResult
            return DelegationResult(
                success=True,
                task_id=subtask.id,
                agent_id=record.agent_id,
                output=result_task.result
            )

        except Exception as e:
            logger.exception(f"[AgentDelegationMixin] Delegation failed: {e}")
            record.status = "error"
            record.error = str(e)
            record.completed_at = datetime.now().isoformat()

            from .delegating_subagent import DelegationResult
            return DelegationResult(
                success=False,
                task_id=subtask.id,
                agent_id=record.agent_id,
                error=str(e)
            )

        finally:
            self._delegation_history.append(record)
            if record.delegation_id in self._active_delegations:
                del self._active_delegations[record.delegation_id]

    async def delegate_parallel(
        self,
        subtasks: List["Subtask"],
        max_concurrent: int = 5,
        options: Optional[DelegationOptions] = None
    ) -> List["DelegationResult"]:
        """Delegate multiple subtasks in parallel.

        Args:
            subtasks: List of subtasks to delegate
            max_concurrent: Maximum concurrent delegations
            options: Optional delegation options

        Returns:
            List of DelegationResults
        """
        self._ensure_initialized()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def delegate_with_semaphore(subtask: "Subtask") -> "DelegationResult":
            async with semaphore:
                return await self.delegate_subtask(subtask, options)

        results = await asyncio.gather(
            *[delegate_with_semaphore(s) for s in subtasks],
            return_exceptions=True
        )

        from .delegating_subagent import DelegationResult

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(DelegationResult(
                    success=False,
                    task_id=subtasks[i].id,
                    agent_id="",
                    error=str(result)
                ))
            else:
                final_results.append(result)

        return final_results

    async def delegate_with_retry(
        self,
        subtask: "Subtask",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff: float = 2.0
    ) -> "DelegationResult":
        """Delegate with automatic retry on failure.

        Args:
            subtask: Subtask to delegate
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay
            backoff: Backoff multiplier

        Returns:
            DelegationResult
        """
        options = DelegationOptions(
            max_retries=0,
            retry_delay=retry_delay
        )

        last_result = None
        current_delay = retry_delay

        for attempt in range(max_retries + 1):
            result = await self.delegate_subtask(subtask, options)

            if result.success:
                return result

            last_result = result

            if attempt < max_retries:
                logger.warning(
                    f"[AgentDelegationMixin] Delegation attempt {attempt + 1} failed, "
                    f"retrying in {current_delay}s"
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff

        return last_result

    def create_subagent_for_task(
        self,
        task: "Task",
        agent_type: Optional[str] = None
    ) -> "DelegatingSubAgent":
        """Create a SubAgent for a specific task.

        Args:
            task: Task to create agent for
            agent_type: Optional agent type

        Returns:
            DelegatingSubAgent
        """
        self._ensure_initialized()

        if self._subagent_factory:
            return self._subagent_factory.get_or_create_agent(
                agent_type or "generic",
                {"task": task}
            )

        return self._subagent_manager.get_or_create_agent(
            agent_type or "generic"
        )

    async def delegate_to_skill(
        self,
        subtask: "Subtask",
        skill_name: str,
        options: Optional[DelegationOptions] = None
    ) -> "DelegationResult":
        """Delegate a subtask to a specific skill.

        Args:
            subtask: Subtask to delegate
            skill_name: Name of skill to use
            options: Optional delegation options

        Returns:
            DelegationResult
        """
        options = options or DelegationOptions()
        options.skill_name = skill_name

        return await self.delegate_subtask(subtask, options)

    async def delegate_async(
        self,
        subtask: "Subtask",
        callback: Optional[Callable[["DelegationResult"], None]] = None
    ) -> str:
        """Delegate a subtask asynchronously.

        Args:
            subtask: Subtask to delegate
            callback: Optional callback when complete

        Returns:
            Delegation ID for tracking
        """
        delegation_id = str(uuid4())

        async def execute():
            result = await self.delegate_subtask(subtask)
            if callback:
                callback(result)

        asyncio.create_task(execute())

        return delegation_id

    def get_delegation_status(self, delegation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an async delegation.

        Args:
            delegation_id: Delegation ID

        Returns:
            Status dictionary or None
        """
        if delegation_id in self._active_delegations:
            record = self._active_delegations[delegation_id]
            return {
                "delegation_id": delegation_id,
                "status": record.status,
                "subtask_id": record.subtask_id,
                "agent_id": record.agent_id,
                "started_at": record.started_at,
                "retries": record.retries
            }

        for record in self._delegation_history:
            if record.delegation_id == delegation_id:
                return {
                    "delegation_id": delegation_id,
                    "status": record.status,
                    "subtask_id": record.subtask_id,
                    "agent_id": record.agent_id,
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                    "error": record.error
                }

        return None

    def get_active_delegations(self) -> List[Dict[str, Any]]:
        """Get all active delegations.

        Returns:
            List of active delegation info
        """
        return [
            {
                "delegation_id": d_id,
                "subtask_id": record.subtask_id,
                "agent_id": record.agent_id,
                "status": record.status,
                "started_at": record.started_at
            }
            for d_id, record in self._active_delegations.items()
        ]

    def get_delegation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get delegation history.

        Args:
            limit: Maximum records

        Returns:
            List of delegation records
        """
        return [
            {
                "delegation_id": r.delegation_id,
                "subtask_id": r.subtask_id,
                "agent_id": r.agent_id,
                "status": r.status,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
                "error": r.error,
                "retries": r.retries
            }
            for r in self._delegation_history[-limit:]
        ]

    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get delegation statistics.

        Returns:
            Statistics dictionary
        """
        total = len(self._delegation_history)
        successful = sum(1 for r in self._delegation_history if r.status == "completed")
        failed = sum(1 for r in self._delegation_history if r.status in ["failed", "error"])

        return {
            "total_delegations": total,
            "successful_delegations": successful,
            "failed_delegations": failed,
            "active_delegations": len(self._active_delegations),
            "success_rate": successful / total if total > 0 else 0
        }

    def _ensure_initialized(self) -> None:
        """Ensure delegation is initialized."""
        if not hasattr(self, '_subagent_manager') or self._subagent_manager is None:
            self.init_delegation()


def create_delegation_mixin(
    subagent_manager: Optional[Any] = None,
    subagent_factory: Optional["SubAgentFactory"] = None
) -> AgentDelegationMixin:
    """Create an AgentDelegationMixin instance.

    Args:
        subagent_manager: Optional SubAgentManager
        subagent_factory: Optional SubAgentFactory

    Returns:
        AgentDelegationMixin instance
    """
    mixin = AgentDelegationMixin()
    mixin.init_delegation(
        subagent_manager=subagent_manager,
        subagent_factory=subagent_factory
    )
    return mixin
