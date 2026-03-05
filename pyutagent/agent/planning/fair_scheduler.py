"""Fair Scheduler Module.

Provides fair task scheduling with:
- Round-robin time slicing
- Multi-level feedback queues
- Priority aging
- Starvation prevention

This is part of Phase 4 Week 6-7: Advanced Features.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Deque, Dict, List, Optional, Set

from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskStatus

logger = logging.getLogger(__name__)


class QueueLevel(Enum):
    """Multi-level feedback queue levels."""
    HIGH = auto()    # Level 0 - highest priority
    MEDIUM = auto()  # Level 1
    LOW = auto()     # Level 2 - lowest priority


@dataclass
class QueueConfig:
    """Configuration for fair scheduler.
    
    Attributes:
        time_slice_ms: Time slice per queue level in milliseconds
        aging_interval: Interval for priority aging in seconds
        aging_boost: Priority boost per aging interval
        starvation_threshold: Time after which task is considered starving
        max_queue_size: Maximum size per queue
    """
    time_slice_ms: Dict[QueueLevel, int] = field(default_factory=lambda: {
        QueueLevel.HIGH: 100,
        QueueLevel.MEDIUM: 200,
        QueueLevel.LOW: 400,
    })
    aging_interval: float = 5.0
    aging_boost: float = 0.1
    starvation_threshold: float = 30.0
    max_queue_size: int = 1000


@dataclass
class TaskQueueEntry:
    """Entry in the task queue.
    
    Attributes:
        task: Task to execute
        queue_level: Current queue level
        waiting_time: Time spent waiting in seconds
        last_run: Last execution timestamp
        time_slice_remaining: Remaining time slice in milliseconds
        total_wait_time: Total accumulated wait time
    """
    task: PriorityTask
    queue_level: QueueLevel = QueueLevel.HIGH
    waiting_time: float = 0.0
    last_run: Optional[datetime] = None
    time_slice_remaining: float = 0.0
    total_wait_time: float = 0.0
    added_at: datetime = field(default_factory=datetime.now)


@dataclass
class FairnessMetrics:
    """Metrics for measuring scheduling fairness.
    
    Attributes:
        jain_index: Jain's fairness index (0.0-1.0)
        avg_wait_time: Average wait time across tasks
        max_wait_time: Maximum wait time
        min_wait_time: Minimum wait time
        starvation_count: Number of starving tasks
        throughput: Tasks completed per second
    """
    jain_index: float
    avg_wait_time: float
    max_wait_time: float
    min_wait_time: float
    starvation_count: int
    throughput: float


class FairScheduler:
    """Fair scheduler for equitable task execution.
    
    Provides:
    - Multi-level feedback queues
    - Round-robin time slicing
    - Priority aging
    - Starvation prevention
    """
    
    def __init__(self, config: Optional[QueueConfig] = None):
        """Initialize FairScheduler.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config or QueueConfig()
        self._queues: Dict[QueueLevel, Deque[TaskQueueEntry]] = {
            level: deque() for level in QueueLevel
        }
        self._task_entries: Dict[str, TaskQueueEntry] = {}
        self._completed_tasks: List[TaskQueueEntry] = []
        self._last_aging_run = datetime.now()
        self._starving_tasks: Set[str] = set()
    
    def add_task(self, task: PriorityTask) -> None:
        """Add a task to the scheduler.
        
        Args:
            task: Task to add
        """
        if task.id in self._task_entries:
            logger.warning(f"Task {task.id} already in scheduler")
            return
        
        # Check queue size limit
        if len(self._queues[QueueLevel.HIGH]) >= self.config.max_queue_size:
            logger.warning(f"Queue full, rejecting task {task.id}")
            return
        
        entry = TaskQueueEntry(
            task=task,
            queue_level=QueueLevel.HIGH,
            time_slice_remaining=self.config.time_slice_ms[QueueLevel.HIGH],
        )
        
        self._queues[QueueLevel.HIGH].append(entry)
        self._task_entries[task.id] = entry
        
        logger.debug(f"Added task {task.id} to HIGH priority queue")
    
    def get_next_task(self) -> Optional[PriorityTask]:
        """Get next task to execute using multi-level feedback.
        
        Returns:
            Next task or None if no tasks available
        """
        # Apply aging before selecting
        self._apply_aging()
        
        # Check for starving tasks first
        starving_task = self._get_starving_task()
        if starving_task:
            return starving_task
        
        # Try queues in priority order
        for level in [QueueLevel.HIGH, QueueLevel.MEDIUM, QueueLevel.LOW]:
            if self._queues[level]:
                entry = self._queues[level][0]
                
                # Check if time slice expired
                if entry.time_slice_remaining <= 0:
                    # Move to lower queue
                    self._demote_task(entry)
                    continue
                
                # Return task
                self._queues[level].popleft()
                entry.last_run = datetime.now()
                return entry.task
        
        return None
    
    def task_yielded(self, task_id: str, completed: bool = False) -> None:
        """Handle task yield or completion.
        
        Args:
            task_id: Task that yielded
            completed: Whether task completed
        """
        if task_id not in self._task_entries:
            return
        
        entry = self._task_entries[task_id]
        
        if completed:
            # Remove from scheduler
            del self._task_entries[task_id]
            self._completed_tasks.append(entry)
            
            if task_id in self._starving_tasks:
                self._starving_tasks.remove(task_id)
            
            logger.debug(f"Task {task_id} completed")
        else:
            # Re-queue with updated time slice
            time_used = self.config.time_slice_ms[entry.queue_level] - \
                       entry.time_slice_remaining
            
            # Reduce time slice for next round
            entry.time_slice_remaining = max(
                10,
                entry.time_slice_remaining * 0.9
            )
            
            # Add back to same queue
            self._queues[entry.queue_level].append(entry)
            
            logger.debug(f"Task {task_id} yielded, requeued at {entry.queue_level.name}")
    
    def task_completed(self, task_id: str) -> None:
        """Mark a task as completed.
        
        Args:
            task_id: Task that completed
        """
        self.task_yielded(task_id, completed=True)
    
    def _apply_aging(self) -> None:
        """Apply priority aging to prevent starvation."""
        now = datetime.now()
        time_since_aging = (now - self._last_aging_run).total_seconds()
        
        if time_since_aging < self.config.aging_interval:
            return
        
        self._last_aging_run = now
        
        # Boost priority for tasks in lower queues
        for level in [QueueLevel.LOW, QueueLevel.MEDIUM]:
            for entry in self._queues[level]:
                entry.waiting_time += time_since_aging
                
                # Boost priority if waiting too long
                if entry.waiting_time > self.config.aging_interval * 2:
                    old_level = entry.queue_level
                    self._promote_task(entry)
                    
                    logger.debug(
                        f"Task {entry.task.id} aged from {old_level.name} "
                        f"to {entry.queue_level.name}"
                    )
    
    def _promote_task(self, entry: TaskQueueEntry) -> None:
        """Promote task to higher priority queue.
        
        Args:
            entry: Task entry to promote
        """
        if entry.queue_level == QueueLevel.HIGH:
            return
        
        # Remove from current queue
        self._queues[entry.queue_level] = deque(
            e for e in self._queues[entry.queue_level] if e != entry
        )
        
        # Promote
        if entry.queue_level == QueueLevel.MEDIUM:
            entry.queue_level = QueueLevel.HIGH
        elif entry.queue_level == QueueLevel.LOW:
            entry.queue_level = QueueLevel.MEDIUM
        
        # Reset waiting time
        entry.waiting_time = 0
        
        # Add to new queue
        self._queues[entry.queue_level].append(entry)
    
    def _demote_task(self, entry: TaskQueueEntry) -> None:
        """Demote task to lower priority queue.
        
        Args:
            entry: Task entry to demote
        """
        if entry.queue_level == QueueLevel.LOW:
            # Already at lowest, reset time slice
            entry.time_slice_remaining = self.config.time_slice_ms[QueueLevel.LOW]
            self._queues[QueueLevel.LOW].append(entry)
            return
        
        # Remove from current queue (already done in get_next_task)
        
        # Demote
        if entry.queue_level == QueueLevel.HIGH:
            entry.queue_level = QueueLevel.MEDIUM
        elif entry.queue_level == QueueLevel.MEDIUM:
            entry.queue_level = QueueLevel.LOW
        
        # Reset time slice
        entry.time_slice_remaining = self.config.time_slice_ms[entry.queue_level]
        
        # Add to new queue
        self._queues[entry.queue_level].append(entry)
    
    def _get_starving_task(self) -> Optional[PriorityTask]:
        """Get a starving task if any.
        
        Returns:
            Starving task or None
        """
        now = datetime.now()
        
        # Check all queues for starving tasks
        for level in QueueLevel:
            for entry in list(self._queues[level]):
                wait_time = (now - entry.added_at).total_seconds()
                
                if wait_time > self.config.starvation_threshold:
                    # This task is starving
                    if entry.task.id not in self._starving_tasks:
                        self._starving_tasks.add(entry.task.id)
                        logger.warning(
                            f"Task {entry.task.id} is starving "
                            f"(waited {wait_time:.1f}s)"
                        )
                    
                    # Boost to highest priority
                    if entry.queue_level != QueueLevel.HIGH:
                        self._promote_task(entry)
                    
                    return entry.task
        
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        stats = {}
        
        for level in QueueLevel:
            queue = self._queues[level]
            stats[level.name] = {
                "size": len(queue),
                "avg_wait_time": (
                    sum(e.waiting_time for e in queue) / len(queue)
                    if queue else 0.0
                ),
            }
        
        return stats
    
    def get_fairness_metrics(self) -> FairnessMetrics:
        """Calculate fairness metrics.
        
        Returns:
            FairnessMetrics
        """
        # Collect wait times
        wait_times = []
        
        for level in QueueLevel:
            for entry in self._queues[level]:
                wait_time = entry.waiting_time + entry.total_wait_time
                wait_times.append(wait_time)
        
        if not wait_times:
            return FairnessMetrics(
                jain_index=1.0,
                avg_wait_time=0.0,
                max_wait_time=0.0,
                min_wait_time=0.0,
                starvation_count=0,
                throughput=0.0,
            )
        
        # Calculate Jain's fairness index
        n = len(wait_times)
        sum_x = sum(wait_times)
        sum_x_sq = sum(x * x for x in wait_times)
        
        if sum_x_sq > 0:
            jain_index = (sum_x ** 2) / (n * sum_x_sq)
        else:
            jain_index = 1.0
        
        # Count starving tasks
        starving_count = len(self._starving_tasks)
        
        # Calculate throughput
        completed_count = len(self._completed_tasks)
        if completed_count > 0:
            first_completion = min(e.last_run for e in self._completed_tasks if e.last_run)
            last_completion = max(e.last_run for e in self._completed_tasks if e.last_run)
            time_span = (last_completion - first_completion).total_seconds() if first_completion and last_completion else 1.0
            throughput = completed_count / time_span
        else:
            throughput = 0.0
        
        return FairnessMetrics(
            jain_index=jain_index,
            avg_wait_time=sum(wait_times) / n,
            max_wait_time=max(wait_times),
            min_wait_time=min(wait_times),
            starvation_count=starving_count,
            throughput=throughput,
        )
    
    def get_task_wait_time(self, task_id: str) -> Optional[float]:
        """Get current wait time for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Wait time in seconds or None if not found
        """
        if task_id not in self._task_entries:
            return None
        
        entry = self._task_entries[task_id]
        return entry.waiting_time + entry.total_wait_time
    
    def clear(self) -> None:
        """Clear all queues."""
        for level in QueueLevel:
            self._queues[level].clear()
        
        self._task_entries.clear()
        self._starving_tasks.clear()
        
        logger.info("Cleared all scheduler queues")
