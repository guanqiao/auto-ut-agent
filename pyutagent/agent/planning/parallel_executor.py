"""Parallel Execution Engine Module.

Provides parallel task execution with:
- Resource-aware scheduling
- Load balancing
- Fault isolation
- Progress tracking

This is part of Phase 3 Week 17-18: Task Planning Enhancement.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from pyutagent.agent.execution.execution_plan import SubTask

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources for task execution."""
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    LLM = "llm"
    MEMORY = "memory"


@dataclass
class ResourcePool:
    """Pool for managing a specific resource type."""
    resource_type: ResourceType
    max_concurrent: int
    current_usage: int = 0
    waiting_tasks: int = 0

    async def acquire(self, count: int = 1) -> bool:
        """Acquire resources.
        
        Args:
            count: Number of resources to acquire
            
        Returns:
            True if acquired, False if would block
        """
        if self.current_usage + count <= self.max_concurrent:
            self.current_usage += count
            return True
        return False

    async def release(self, count: int = 1) -> None:
        """Release resources."""
        self.current_usage = max(0, self.current_usage - count)

    @property
    def available(self) -> int:
        """Get available capacity."""
        return self.max_concurrent - self.current_usage

    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        if self.max_concurrent == 0:
            return 0.0
        return self.current_usage / self.max_concurrent


@dataclass
class SubTaskResult:
    """Result of a subtask execution."""
    subtask_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel execution."""
    max_concurrent_tasks: int = 4
    max_retries: int = 2
    retry_delay_ms: int = 1000
    timeout_ms: int = 300000
    resource_limits: Dict[ResourceType, int] = field(default_factory=lambda: {
        ResourceType.CPU: 4,
        ResourceType.IO: 8,
        ResourceType.NETWORK: 4,
        ResourceType.LLM: 2,
        ResourceType.MEMORY: 10,
    })


class ParallelExecutionEngine:
    """Engine for parallel task execution."""

    def __init__(self, config: Optional[ParallelExecutionConfig] = None):
        self.config = config or ParallelExecutionConfig()
        self._resource_pools: Dict[ResourceType, ResourcePool] = {}
        self._results: Dict[str, SubTaskResult] = {}
        self._running_tasks: Set[str] = set()
        self._completed_tasks: Set[str] = set()
        self._failed_tasks: Set[str] = set()
        self._lock = asyncio.Lock()
        
        self._init_resource_pools()

    def _init_resource_pools(self) -> None:
        """Initialize resource pools."""
        for resource_type, limit in self.config.resource_limits.items():
            self._resource_pools[resource_type] = ResourcePool(
                resource_type=resource_type,
                max_concurrent=limit,
            )

    async def execute(
        self,
        subtasks: List[SubTask],
        executor_func: Callable[[SubTask], Any],
        dependency_graph: Optional[Any] = None,
    ) -> Dict[str, SubTaskResult]:
        """Execute subtasks in parallel respecting dependencies.
        
        Args:
            subtasks: List of subtasks to execute
            executor_func: Function to execute each subtask
            dependency_graph: Optional dependency graph
            
        Returns:
            Dictionary mapping subtask IDs to results
        """
        if not subtasks:
            return {}
        
        if dependency_graph:
            return await self._execute_with_dependencies(
                subtasks, executor_func, dependency_graph
            )
        else:
            return await self._execute_parallel(subtasks, executor_func)

    async def _execute_parallel(
        self,
        subtasks: List[SubTask],
        executor_func: Callable[[SubTask], Any],
    ) -> Dict[str, SubTaskResult]:
        """Execute all subtasks in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        async def execute_with_limit(subtask: SubTask) -> SubTaskResult:
            async with semaphore:
                return await self._execute_single(subtask, executor_func)
        
        tasks = [execute_with_limit(st) for st in subtasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            subtask_id = subtasks[i].id
            if isinstance(result, Exception):
                self._results[subtask_id] = SubTaskResult(
                    subtask_id=subtask_id,
                    success=False,
                    error=str(result),
                )
            else:
                self._results[subtask_id] = result
        
        return self._results

    async def _execute_with_dependencies(
        self,
        subtasks: List[SubTask],
        executor_func: Callable[[SubTask], Any],
        dependency_graph: Any,
    ) -> Dict[str, SubTaskResult]:
        """Execute subtasks respecting dependencies."""
        pending = {st.id: st for st in subtasks}
        completed_ids: Set[str] = set()
        
        while pending:
            ready = self._get_ready_tasks(pending, completed_ids, dependency_graph)
            
            if not ready:
                remaining = list(pending.keys())
                logger.warning(f"No ready tasks, remaining: {remaining}")
                break
            
            semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
            
            async def execute_with_limit(subtask: SubTask) -> SubTaskResult:
                async with semaphore:
                    return await self._execute_single(subtask, executor_func)
            
            tasks = [execute_with_limit(st) for st in ready]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                subtask_id = ready[i].id
                if isinstance(result, Exception):
                    self._results[subtask_id] = SubTaskResult(
                        subtask_id=subtask_id,
                        success=False,
                        error=str(result),
                    )
                else:
                    self._results[subtask_id] = result
                
                completed_ids.add(subtask_id)
                del pending[subtask_id]
        
        return self._results

    def _get_ready_tasks(
        self,
        pending: Dict[str, SubTask],
        completed_ids: Set[str],
        dependency_graph: Any,
    ) -> List[SubTask]:
        """Get tasks that are ready to execute."""
        ready = []
        
        for subtask_id, subtask in pending.items():
            deps = dependency_graph.get_dependencies(subtask_id) if dependency_graph else []
            if deps.issubset(completed_ids):
                ready.append(subtask)
        
        return ready

    async def _execute_single(
        self,
        subtask: SubTask,
        executor_func: Callable[[SubTask], Any],
    ) -> SubTaskResult:
        """Execute a single subtask with retry logic."""
        start_time = time.time()
        started_at = datetime.now().isoformat()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._lock:
                    self._running_tasks.add(subtask.id)
                
                if asyncio.iscoroutinefunction(executor_func):
                    result = await executor_func(subtask)
                else:
                    result = executor_func(subtask)
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                async with self._lock:
                    self._running_tasks.discard(subtask.id)
                    self._completed_tasks.add(subtask.id)
                
                return SubTaskResult(
                    subtask_id=subtask.id,
                    success=True,
                    result=result,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    retry_count=attempt,
                )
                
            except Exception as e:
                async with self._lock:
                    self._running_tasks.discard(subtask.id)
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_ms / 1000)
                    continue
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                async with self._lock:
                    self._failed_tasks.add(subtask.id)
                
                return SubTaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms,
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    retry_count=attempt,
                )
        
        return SubTaskResult(
            subtask_id=subtask.id,
            success=False,
            error="Max retries exceeded",
        )

    async def acquire_resource(
        self,
        resource_type: ResourceType,
        count: int = 1,
    ) -> bool:
        """Acquire resources from a pool.
        
        Args:
            resource_type: Type of resource
            count: Number of resources
            
        Returns:
            True if acquired
        """
        if resource_type in self._resource_pools:
            return await self._resource_pools[resource_type].acquire(count)
        return True

    async def release_resource(
        self,
        resource_type: ResourceType,
        count: int = 1,
    ) -> None:
        """Release resources to a pool."""
        if resource_type in self._resource_pools:
            await self._resource_pools[resource_type].release(count)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len(self._failed_tasks),
            "resource_utilization": {
                rt.value: pool.utilization
                for rt, pool in self._resource_pools.items()
            },
        }

    def reset(self) -> None:
        """Reset execution state."""
        self._results.clear()
        self._running_tasks.clear()
        self._completed_tasks.clear()
        self._failed_tasks.clear()
        for pool in self._resource_pools.values():
            pool.current_usage = 0
