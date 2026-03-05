"""SubAgent Orchestrator for coordinating multiple SubAgents.

This module provides:
- SubAgentOrchestrator: Unified orchestration of SubAgents
- Parallel and sequential execution
- Status monitoring
- Exception handling and recovery
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from .subagent_base import SubAgent, Task, AgentStatus
from .delegating_subagent import DelegatingSubAgent, DelegationResult
from .subagent_factory import SubAgentFactory, AgentType
from .hierarchical_planner import (
    HierarchicalTaskPlanner,
    TaskTree,
    ExecutionPlan,
    Subtask,
    SubtaskStatus
)
from .task_router import IntelligentTaskRouter, RoutingStrategy
from .conflict_resolver import ConflictResolver, Conflict
from .shared_context import SharedContextManager
from .result_aggregator import ResultAggregator, AggregationStrategy

logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Modes for orchestration."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    PIPELINE = "pipeline"


class OrchestrationStatus(Enum):
    """Status of orchestration."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class OrchestrationTask:
    """Task in orchestration."""
    task_id: str
    task: Task
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class OrchestrationResult:
    """Result of orchestration."""
    orchestration_id: str
    success: bool
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: int = 0
    status: OrchestrationStatus = OrchestrationStatus.COMPLETED


@dataclass
class OrchestrationStatus:
    """Current status of orchestration."""
    status: str = "idle"
    current_phase: str = ""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_agents: int = 0
    progress: float = 0.0
    started_at: Optional[str] = None
    estimated_completion: Optional[str] = None


class SubAgentOrchestrator:
    """Orchestrator for coordinating multiple SubAgents.

    Features:
    - Unified orchestration interface
    - Parallel and sequential execution
    - Status monitoring
    - Exception handling and recovery
    - Context sharing
    """

    def __init__(
        self,
        agent_factory: Optional[SubAgentFactory] = None,
        task_router: Optional[IntelligentTaskRouter] = None,
        conflict_resolver: Optional[ConflictResolver] = None,
        context_manager: Optional[SharedContextManager] = None,
        result_aggregator: Optional[ResultAggregator] = None,
        llm_client: Optional[Any] = None,
        tool_service: Optional[Any] = None,
        max_parallel_tasks: int = 10
    ):
        """Initialize SubAgentOrchestrator.

        Args:
            agent_factory: Optional SubAgentFactory
            task_router: Optional IntelligentTaskRouter
            conflict_resolver: Optional ConflictResolver
            context_manager: Optional SharedContextManager
            result_aggregator: Optional ResultAggregator
            llm_client: Optional LLM client
            tool_service: Optional tool service
            max_parallel_tasks: Maximum parallel tasks
        """
        self.agent_factory = agent_factory or SubAgentFactory(
            llm_client=llm_client,
            tool_service=tool_service
        )
        self.task_router = task_router or IntelligentTaskRouter()
        self.conflict_resolver = conflict_resolver or ConflictResolver()
        self.context_manager = context_manager or SharedContextManager()
        self.result_aggregator = result_aggregator or ResultAggregator()
        self.llm_client = llm_client
        self.tool_service = tool_service
        self.max_parallel_tasks = max_parallel_tasks

        self._task_planner = HierarchicalTaskPlanner(llm_client=llm_client)

        self._active_agents: Dict[str, DelegatingSubAgent] = {}
        self._orchestration_tasks: Dict[str, OrchestrationTask] = {}
        self._status = OrchestrationStatus()
        self._stop_requested = False
        self._pause_requested = False

        self._orchestration_history: List[OrchestrationResult] = []

        self._stats = {
            "orchestrations": 0,
            "successful_orchestrations": 0,
            "failed_orchestrations": 0,
            "total_tasks_executed": 0
        }

        logger.info(f"[SubAgentOrchestrator] Initialized (max_parallel={max_parallel_tasks})")

    async def orchestrate(
        self,
        plan: ExecutionPlan,
        mode: OrchestrationMode = OrchestrationMode.ADAPTIVE
    ) -> OrchestrationResult:
        """Execute an orchestration plan.

        Args:
            plan: Execution plan to orchestrate
            mode: Orchestration mode

        Returns:
            OrchestrationResult
        """
        orchestration_id = str(uuid4())
        start_time = asyncio.get_event_loop().time()

        self._stats["orchestrations"] += 1
        self._status.status = "executing"
        self._status.started_at = datetime.now().isoformat()

        logger.info(f"[SubAgentOrchestrator] Starting orchestration {orchestration_id}")

        try:
            self._status.total_tasks = len(plan.execution_order)
            self._status.current_phase = "executing"

            if mode == OrchestrationMode.SEQUENTIAL:
                results = await self._execute_sequential(plan)
            elif mode == OrchestrationMode.PARALLEL:
                results = await self._execute_parallel(plan)
            elif mode == OrchestrationMode.PIPELINE:
                results = await self._execute_pipeline(plan)
            else:
                results = await self._execute_adaptive(plan)

            execution_time_ms = int(
                (asyncio.get_event_loop().time() - start_time) * 1000
            )

            completed = sum(1 for r in results.values() if r.get("success", False))
            failed = len(results) - completed

            result = OrchestrationResult(
                orchestration_id=orchestration_id,
                success=failed == 0,
                total_tasks=len(results),
                completed_tasks=completed,
                failed_tasks=failed,
                results=results,
                execution_time_ms=execution_time_ms,
                status=OrchestrationStatus.COMPLETED if failed == 0 else OrchestrationStatus.FAILED
            )

            if result.success:
                self._stats["successful_orchestrations"] += 1
            else:
                self._stats["failed_orchestrations"] += 1

            self._stats["total_tasks_executed"] += len(results)

            self._orchestration_history.append(result)

            logger.info(
                f"[SubAgentOrchestrator] Orchestration {orchestration_id} completed: "
                f"{completed}/{len(results)} tasks successful"
            )

            return result

        except Exception as e:
            logger.exception(f"[SubAgentOrchestrator] Orchestration failed: {e}")

            execution_time_ms = int(
                (asyncio.get_event_loop().time() - start_time) * 1000
            )

            self._stats["failed_orchestrations"] += 1

            return OrchestrationResult(
                orchestration_id=orchestration_id,
                success=False,
                total_tasks=0,
                completed_tasks=0,
                failed_tasks=1,
                errors=[str(e)],
                execution_time_ms=execution_time_ms,
                status=OrchestrationStatus.FAILED
            )

        finally:
            self._status.status = "idle"

    async def _execute_sequential(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute tasks sequentially.

        Args:
            plan: Execution plan

        Returns:
            Results dictionary
        """
        results = {}

        for task_id in plan.execution_order:
            if self._stop_requested:
                break

            while self._pause_requested:
                await asyncio.sleep(0.1)

            subtask = plan.task_tree.get_subtask(task_id)
            if not subtask:
                continue

            result = await self._execute_single_task(subtask)
            results[task_id] = result

            self._status.completed_tasks += 1
            self._status.progress = (
                self._status.completed_tasks / self._status.total_tasks
                if self._status.total_tasks > 0 else 0
            )

        return results

    async def _execute_parallel(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute tasks in parallel.

        Args:
            plan: Execution plan

        Returns:
            Results dictionary
        """
        results = {}
        completed: Set[str] = set()

        for parallel_group in plan.parallel_groups:
            if self._stop_requested:
                break

            while self._pause_requested:
                await asyncio.sleep(0.1)

            tasks_to_run = [
                plan.task_tree.get_subtask(task_id)
                for task_id in parallel_group
                if task_id not in completed
            ]

            tasks_to_run = [t for t in tasks_to_run if t is not None]

            if not tasks_to_run:
                continue

            group_results = await asyncio.gather(
                *[self._execute_single_task(t) for t in tasks_to_run],
                return_exceptions=True
            )

            for subtask, result in zip(tasks_to_run, group_results):
                if isinstance(result, Exception):
                    results[subtask.id] = {
                        "success": False,
                        "error": str(result)
                    }
                else:
                    results[subtask.id] = result

                completed.add(subtask.id)
                self._status.completed_tasks += 1

            self._status.progress = (
                self._status.completed_tasks / self._status.total_tasks
                if self._status.total_tasks > 0 else 0
            )

        return results

    async def _execute_pipeline(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute tasks in pipeline mode.

        Args:
            plan: Execution plan

        Returns:
            Results dictionary
        """
        results = {}
        pipeline_data: Dict[str, Any] = {}

        for task_id in plan.execution_order:
            if self._stop_requested:
                break

            while self._pause_requested:
                await asyncio.sleep(0.1)

            subtask = plan.task_tree.get_subtask(task_id)
            if not subtask:
                continue

            subtask.input_data.update(pipeline_data)

            result = await self._execute_single_task(subtask)
            results[task_id] = result

            if result.get("success"):
                pipeline_data.update(result.get("output", {}))

            self._status.completed_tasks += 1
            self._status.progress = (
                self._status.completed_tasks / self._status.total_tasks
                if self._status.total_tasks > 0 else 0
            )

        return results

    async def _execute_adaptive(
        self,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Execute tasks adaptively based on dependencies.

        Args:
            plan: Execution plan

        Returns:
            Results dictionary
        """
        results = {}
        completed: Set[str] = set()
        failed: Set[str] = set()

        while len(completed) + len(failed) < len(plan.execution_order):
            if self._stop_requested:
                break

            while self._pause_requested:
                await asyncio.sleep(0.1)

            ready_tasks = plan.get_next_tasks(completed)

            ready_tasks = [
                t for t in ready_tasks
                if t not in failed and not any(
                    dep in failed for dep in plan.dependency_graph.get_dependencies(t)
                )
            ]

            if not ready_tasks:
                remaining = set(plan.execution_order) - completed - failed
                if remaining:
                    logger.warning(f"[SubAgentOrchestrator] Deadlock detected, forcing execution")
                    ready_tasks = [list(remaining)[0]]
                else:
                    break

            batch_size = min(len(ready_tasks), self.max_parallel_tasks)
            batch = ready_tasks[:batch_size]

            tasks_to_run = [
                plan.task_tree.get_subtask(task_id)
                for task_id in batch
            ]
            tasks_to_run = [t for t in tasks_to_run if t is not None]

            batch_results = await asyncio.gather(
                *[self._execute_single_task(t) for t in tasks_to_run],
                return_exceptions=True
            )

            for subtask, result in zip(tasks_to_run, batch_results):
                if isinstance(result, Exception):
                    results[subtask.id] = {"success": False, "error": str(result)}
                    failed.add(subtask.id)
                else:
                    results[subtask.id] = result
                    if result.get("success"):
                        completed.add(subtask.id)
                    else:
                        failed.add(subtask.id)

                self._status.completed_tasks += 1
                self._status.progress = (
                    (len(completed) + len(failed)) / self._status.total_tasks
                    if self._status.total_tasks > 0 else 0
                )

        return results

    async def _execute_single_task(
        self,
        subtask: Subtask
    ) -> Dict[str, Any]:
        """Execute a single task.

        Args:
            subtask: Subtask to execute

        Returns:
            Task result
        """
        task = Task(
            id=subtask.id,
            name=subtask.name,
            description=subtask.description,
            input_data=subtask.input_data
        )

        agent = self.agent_factory.get_or_create_agent(
            subtask.task_type.value if hasattr(subtask.task_type, 'value') else "generic"
        )

        self._active_agents[agent.id] = agent
        self.task_router.register_agent(agent)

        try:
            result = await agent.delegate(task)

            self.task_router.record_routing_result(
                task.id,
                agent.id,
                result.success,
                result.execution_time_ms
            )

            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "agent_id": agent.id
            }

        except Exception as e:
            logger.error(f"[SubAgentOrchestrator] Task {subtask.id} failed: {e}")
            return {"success": False, "error": str(e)}

        finally:
            if agent.id in self._active_agents:
                del self._active_agents[agent.id]

    async def execute_parallel(
        self,
        tasks: List[Task]
    ) -> List[Dict[str, Any]]:
        """Execute multiple tasks in parallel.

        Args:
            tasks: Tasks to execute

        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)

        async def execute_with_semaphore(task: Task) -> Dict[str, Any]:
            async with semaphore:
                agent = self.agent_factory.get_or_create_agent("generic")
                self._active_agents[agent.id] = agent

                try:
                    result = await agent.delegate(task)
                    return {
                        "task_id": task.id,
                        "success": result.success,
                        "output": result.output,
                        "error": result.error
                    }
                finally:
                    if agent.id in self._active_agents:
                        del self._active_agents[agent.id]

        results = await asyncio.gather(
            *[execute_with_semaphore(t) for t in tasks],
            return_exceptions=True
        )

        return [
            r if isinstance(r, dict) else {"success": False, "error": str(r)}
            for r in results
        ]

    async def execute_sequential(
        self,
        tasks: List[Task]
    ) -> List[Dict[str, Any]]:
        """Execute tasks sequentially.

        Args:
            tasks: Tasks to execute

        Returns:
            List of results in order
        """
        results = []

        for task in tasks:
            if self._stop_requested:
                break

            agent = self.agent_factory.get_or_create_agent("generic")
            self._active_agents[agent.id] = agent

            try:
                result = await agent.delegate(task)
                results.append({
                    "task_id": task.id,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error
                })
            except Exception as e:
                results.append({
                    "task_id": task.id,
                    "success": False,
                    "error": str(e)
                })
            finally:
                if agent.id in self._active_agents:
                    del self._active_agents[agent.id]

        return results

    def get_execution_status(self) -> OrchestrationStatus:
        """Get current execution status.

        Returns:
            OrchestrationStatus
        """
        self._status.active_agents = len(self._active_agents)
        return self._status

    def stop(self) -> None:
        """Request stop of current orchestration."""
        self._stop_requested = True
        logger.info("[SubAgentOrchestrator] Stop requested")

    def pause(self) -> None:
        """Request pause of current orchestration."""
        self._pause_requested = True
        logger.info("[SubAgentOrchestrator] Pause requested")

    def resume(self) -> None:
        """Resume paused orchestration."""
        self._pause_requested = False
        logger.info("[SubAgentOrchestrator] Resumed")

    def get_orchestration_history(self, limit: int = 20) -> List[OrchestrationResult]:
        """Get orchestration history.

        Args:
            limit: Maximum results

        Returns:
            List of OrchestrationResults
        """
        return self._orchestration_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "orchestrations": self._stats["orchestrations"],
            "successful_orchestrations": self._stats["successful_orchestrations"],
            "failed_orchestrations": self._stats["failed_orchestrations"],
            "total_tasks_executed": self._stats["total_tasks_executed"],
            "success_rate": (
                self._stats["successful_orchestrations"] / self._stats["orchestrations"]
                if self._stats["orchestrations"] > 0 else 0
            ),
            "active_agents": len(self._active_agents),
            "max_parallel_tasks": self.max_parallel_tasks
        }


def create_subagent_orchestrator(
    llm_client: Optional[Any] = None,
    tool_service: Optional[Any] = None,
    max_parallel_tasks: int = 10
) -> SubAgentOrchestrator:
    """Create a SubAgentOrchestrator.

    Args:
        llm_client: Optional LLM client
        tool_service: Optional tool service
        max_parallel_tasks: Maximum parallel tasks

    Returns:
        SubAgentOrchestrator instance
    """
    return SubAgentOrchestrator(
        llm_client=llm_client,
        tool_service=tool_service,
        max_parallel_tasks=max_parallel_tasks
    )
