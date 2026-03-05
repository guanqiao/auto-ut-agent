"""Parallel Execution Engine Module.

Provides parallel task execution with:
- Resource-aware scheduling
- Load balancing
- Fault isolation
- Progress tracking
- Dynamic priority queue
- Task preemption

This is part of Phase 3 Week 17-18: Task Planning Enhancement.
"""

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pyutagent.agent.execution.execution_plan import SubTask

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Task type enumeration."""
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    LLM_BOUND = "llm_bound"
    MIXED = "mixed"


@dataclass
class PriorityTask:
    """Task wrapper with priority support.
    
    Attributes:
        id: Unique task identifier
        description: Task description
        priority: Priority score (0.0-1.0, higher is more important)
        created_at: Task creation timestamp
        deadline: Optional task deadline
        dependencies: Set of task IDs that must complete before this task
        resource_requirements: Resource requirements for this task
        estimated_duration: Estimated execution time in seconds
        actual_duration: Actual execution time in seconds
        status: Current task status
        result: Execution result
        error: Error message if failed
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts
        preemptible: Whether this task can be preempted
        paused: Whether this task is paused
        metadata: Additional task metadata
    """
    id: str
    description: str
    priority: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    dependencies: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    estimated_duration: float = 0.0
    actual_duration: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    preemptible: bool = True
    paused: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: 'PriorityTask') -> bool:
        """Comparison for heap sort (higher priority first).
        
        Note: heapq is a min-heap, so we invert the comparison.
        """
        return self.priority > other.priority
    
    def __eq__(self, other: object) -> bool:
        """Check equality by task ID."""
        if not isinstance(other, PriorityTask):
            return False
        return self.id == other.id
    
    def __hash__(self) -> int:
        """Hash by task ID."""
        return hash(self.id)


class ResourceType(Enum):
    """Types of resources for task execution."""
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    LLM = "llm"
    MEMORY = "memory"


@dataclass
class ResourcePool:
    """Pool for managing a specific resource type with prediction support.
    
    Attributes:
        resource_type: Type of resource
        max_concurrent: Maximum concurrent usage
        current_usage: Current resource usage
        waiting_tasks: Number of waiting tasks
        reserved: Reserved resource capacity
        usage_history: History of resource usage for prediction
    """
    resource_type: ResourceType
    max_concurrent: int
    current_usage: int = 0
    waiting_tasks: int = 0
    reserved: int = 0
    usage_history: List[Tuple[datetime, int]] = field(default_factory=list)
    
    async def acquire(self, count: int = 1, use_reserved: bool = False) -> bool:
        """Acquire resources.
        
        Args:
            count: Number of resources to acquire
            use_reserved: Whether to use reserved capacity
            
        Returns:
            True if acquired, False if would block
        """
        available = self.max_concurrent - self.current_usage - self.reserved
        if not use_reserved and available >= count:
            self.current_usage += count
            self._record_usage()
            return True
        elif use_reserved and self.reserved >= count:
            self.reserved -= count
            self.current_usage += count
            self._record_usage()
            return True
        return False
    
    async def release(self, count: int = 1) -> None:
        """Release resources."""
        self.current_usage = max(0, self.current_usage - count)
        self._record_usage()
    
    def reserve(self, count: int) -> bool:
        """Reserve resource capacity.
        
        Args:
            count: Amount to reserve
            
        Returns:
            True if reservation successful
        """
        available = self.max_concurrent - self.current_usage - self.reserved
        if available >= count:
            self.reserved += count
            return True
        return False
    
    def unreserve(self, count: int) -> None:
        """Release reserved capacity."""
        self.reserved = max(0, self.reserved - count)
    
    @property
    def available(self) -> int:
        """Get available capacity."""
        return self.max_concurrent - self.current_usage - self.reserved
    
    @property
    def utilization(self) -> float:
        """Get utilization percentage."""
        if self.max_concurrent == 0:
            return 0.0
        return self.current_usage / self.max_concurrent
    
    def _record_usage(self) -> None:
        """Record current usage to history."""
        now = datetime.now()
        self.usage_history.append((now, self.current_usage))
        # Keep last 100 records
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]
    
    def predict_availability(self, horizon: timedelta = timedelta(seconds=60)) -> int:
        """Predict future resource availability.
        
        Args:
            horizon: Prediction time horizon
            
        Returns:
            Predicted available capacity
        """
        if len(self.usage_history) < 2:
            return self.available
        
        # Simple linear trend prediction
        recent = self.usage_history[-10:]
        if len(recent) < 2:
            return self.available
        
        # Calculate trend
        time_diff = (recent[-1][0] - recent[0][0]).total_seconds()
        if time_diff == 0:
            return self.available
        
        usage_diff = recent[-1][1] - recent[0][1]
        trend_per_second = usage_diff / time_diff
        
        # Predict future usage
        predicted_change = trend_per_second * horizon.total_seconds()
        predicted_usage = int(self.current_usage + predicted_change)
        
        return max(0, self.max_concurrent - predicted_usage - self.reserved)


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
    task_timeout_seconds: float = 300.0
    enable_progress_tracking: bool = True
    progress_update_interval_seconds: float = 1.0
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    enable_resource_optimization: bool = True
    enable_priority_queue: bool = True
    preemption_enabled: bool = True
    resource_limits: Dict[ResourceType, int] = field(default_factory=lambda: {
        ResourceType.CPU: 4,
        ResourceType.IO: 8,
        ResourceType.NETWORK: 4,
        ResourceType.LLM: 2,
        ResourceType.MEMORY: 10,
    })


@dataclass
class PriorityExecutionConfig:
    """Extended configuration for priority-based execution.
    
    Attributes:
        max_concurrent_tasks: Maximum concurrent tasks
        task_timeout_seconds: Task timeout in seconds
        enable_progress_tracking: Enable progress tracking
        progress_update_interval_seconds: Progress update interval
        max_retries: Maximum retry attempts
        retry_delay_seconds: Delay between retries
        enable_resource_optimization: Enable resource optimization
        enable_priority_queue: Enable priority queue
        preemption_enabled: Enable task preemption
        priority_aging_enabled: Enable priority aging to prevent starvation
        priority_aging_interval_minutes: Interval for priority aging
        max_wait_time_seconds: Maximum wait time before priority boost
    """
    max_concurrent_tasks: int = 4
    task_timeout_seconds: float = 300.0
    enable_progress_tracking: bool = True
    progress_update_interval_seconds: float = 1.0
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    enable_resource_optimization: bool = True
    enable_priority_queue: bool = True
    preemption_enabled: bool = True
    priority_aging_enabled: bool = True
    priority_aging_interval_minutes: int = 1
    max_wait_time_seconds: float = 60.0


@dataclass
class ExecutionStats:
    """Statistics for execution monitoring."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    total_duration_ms: float = 0.0
    average_task_duration_ms: float = 0.0
    max_concurrent_tasks: int = 0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks + self.failed_tasks) / self.total_tasks * 100


class ParallelExecutionEngine:
    """Engine for parallel task execution with priority queue support."""

    def __init__(
        self,
        config: Optional[ParallelExecutionConfig] = None,
        priority_config: Optional[PriorityExecutionConfig] = None,
    ):
        self.config = config or ParallelExecutionConfig()
        self.priority_config = priority_config or PriorityExecutionConfig()
        self._resource_pools: Dict[ResourceType, ResourcePool] = {}
        self._results: Dict[str, SubTaskResult] = {}
        self._running_tasks: Set[str] = set()
        self._completed_tasks: Set[str] = set()
        self._failed_tasks: Set[str] = set()
        self._lock = asyncio.Lock()
        
        # Priority queue
        self._priority_queue: List[PriorityTask] = []
        self._task_map: Dict[str, PriorityTask] = {}
        self._paused_tasks: Set[str] = set()
        self._preempted_tasks: Set[str] = set()
        
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
    
    # Priority Queue Methods
    
    async def enqueue_task(self, task: PriorityTask) -> bool:
        """Add a task to the priority queue.
        
        Args:
            task: Task to enqueue
            
        Returns:
            True if successfully enqueued
        """
        async with self._lock:
            if task.id in self._task_map:
                logger.warning(f"Task {task.id} already in queue")
                return False
            
            heapq.heappush(self._priority_queue, task)
            self._task_map[task.id] = task
            task.status = TaskStatus.QUEUED
            logger.debug(f"Enqueued task {task.id} with priority {task.priority}")
            return True
    
    async def dequeue_task(self) -> Optional[PriorityTask]:
        """Remove and return the highest priority task.
        
        Returns:
            Highest priority task or None if queue is empty
        """
        async with self._lock:
            while self._priority_queue:
                task = heapq.heappop(self._priority_queue)
                if task.id in self._paused_tasks or task.id in self._preempted_tasks:
                    # Skip paused/preempted tasks
                    heapq.heappush(self._priority_queue, task)
                    break
                if task.id in self._task_map:
                    del self._task_map[task.id]
                    task.status = TaskStatus.RUNNING
                    logger.debug(f"Dequeued task {task.id}")
                    return task
            return None
    
    async def peek_task(self) -> Optional[PriorityTask]:
        """Return the highest priority task without removing it.
        
        Returns:
            Highest priority task or None if queue is empty
        """
        async with self._lock:
            if self._priority_queue:
                return self._priority_queue[0]
            return None
    
    async def update_priority(self, task_id: str, new_priority: float) -> bool:
        """Update the priority of a task.
        
        Args:
            task_id: Task identifier
            new_priority: New priority value
            
        Returns:
            True if updated successfully
        """
        async with self._lock:
            if task_id not in self._task_map:
                return False
            
            task = self._task_map[task_id]
            old_priority = task.priority
            task.priority = new_priority
            
            # Rebuild heap to maintain order
            heapq.heapify(self._priority_queue)
            
            logger.debug(
                f"Updated priority of task {task_id}: "
                f"{old_priority:.2f} -> {new_priority:.2f}"
            )
            return True
    
    async def preempt(self, task_id: str) -> bool:
        """Preempt a running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if preempted successfully
        """
        async with self._lock:
            if task_id not in self._task_map:
                return False
            
            task = self._task_map[task_id]
            if not task.preemptible:
                logger.warning(f"Task {task_id} is not preemptible")
                return False
            
            task.paused = True
            self._preempted_tasks.add(task_id)
            task.status = TaskStatus.PAUSED
            
            # Boost priority slightly to ensure it runs soon
            task.priority = min(1.0, task.priority + 0.05)
            heapq.heapify(self._priority_queue)
            
            logger.info(f"Preempted task {task_id}")
            return True
    
    async def pause(self, task_id: str) -> bool:
        """Pause a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if paused successfully
        """
        async with self._lock:
            if task_id not in self._task_map:
                return False
            
            task = self._task_map[task_id]
            task.paused = True
            self._paused_tasks.add(task_id)
            task.status = TaskStatus.PAUSED
            
            logger.info(f"Paused task {task_id}")
            return True
    
    async def resume(self, task_id: str) -> bool:
        """Resume a paused task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if resumed successfully
        """
        async with self._lock:
            if task_id not in self._task_map:
                return False
            
            task = self._task_map[task_id]
            task.paused = False
            self._paused_tasks.discard(task_id)
            self._preempted_tasks.discard(task_id)
            task.status = TaskStatus.QUEUED
            
            logger.info(f"Resumed task {task_id}")
            return True
    
    async def execute_with_priority(
        self,
        tasks: List[PriorityTask],
        executor_func: Callable[[PriorityTask], Any],
    ) -> Dict[str, SubTaskResult]:
        """Execute tasks with priority scheduling.
        
        Args:
            tasks: List of tasks to execute
            executor_func: Function to execute each task
            
        Returns:
            Dictionary mapping task IDs to results
        """
        if not tasks:
            return {}
        
        # Enqueue all tasks
        for task in tasks:
            await self.enqueue_task(task)
        
        # Execute tasks in priority order
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        async def execute_task(task: PriorityTask) -> SubTaskResult:
            async with semaphore:
                return await self._execute_priority_task(task, executor_func)
        
        # Process queue
        execution_tasks = []
        while self._priority_queue:
            task = await self.dequeue_task()
            if task:
                execution_tasks.append(execute_task(task))
        
        if not execution_tasks:
            return {}
        
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # Collect results
        for i, result in enumerate(results):
            task = tasks[i]
            if isinstance(result, Exception):
                self._results[task.id] = SubTaskResult(
                    subtask_id=task.id,
                    success=False,
                    error=str(result),
                )
            else:
                self._results[task.id] = result
        
        return self._results
    
    async def _execute_priority_task(
        self,
        task: PriorityTask,
        executor_func: Callable[[PriorityTask], Any],
    ) -> SubTaskResult:
        """Execute a single priority task.
        
        Args:
            task: Task to execute
            executor_func: Executor function
            
        Returns:
            Task result
        """
        start_time = time.time()
        started_at = datetime.now().isoformat()
        
        for attempt in range(task.max_retries + 1):
            try:
                async with self._lock:
                    self._running_tasks.add(task.id)
                
                # Check if task is paused
                if task.paused:
                    await asyncio.sleep(0.1)
                    continue
                
                if asyncio.iscoroutinefunction(executor_func):
                    result = await executor_func(task)
                else:
                    result = executor_func(task)
                
                duration_ms = int((time.time() - start_time) * 1000)
                task.actual_duration = duration_ms / 1000.0
                
                async with self._lock:
                    self._running_tasks.discard(task.id)
                    self._completed_tasks.add(task.id)
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                
                return SubTaskResult(
                    subtask_id=task.id,
                    success=True,
                    result=result,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    retry_count=attempt,
                )
                
            except Exception as e:
                async with self._lock:
                    self._running_tasks.discard(task.id)
                
                if attempt < task.max_retries:
                    task.retry_count = attempt + 1
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                
                duration_ms = int((time.time() - start_time) * 1000)
                
                async with self._lock:
                    self._failed_tasks.add(task.id)
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                
                return SubTaskResult(
                    subtask_id=task.id,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms,
                    started_at=started_at,
                    completed_at=datetime.now().isoformat(),
                    retry_count=attempt,
                )
        
        return SubTaskResult(
            subtask_id=task.id,
            success=False,
            error="Max retries exceeded",
        )
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get priority queue statistics.
        
        Returns:
            Queue statistics
        """
        return {
            "queue_size": len(self._priority_queue),
            "running_tasks": len(self._running_tasks),
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len(self._failed_tasks),
            "paused_tasks": len(self._paused_tasks),
            "preempted_tasks": len(self._preempted_tasks),
            "resource_utilization": {
                rt.value: pool.utilization
                for rt, pool in self._resource_pools.items()
            },
        }
