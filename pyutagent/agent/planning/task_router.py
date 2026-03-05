"""Task Router Module.

Provides task routing and classification with:
- Task type classification (CPU/IO/LLM bound)
- Priority calculation
- Routing decisions

This is part of Phase 1 Week 1-2: Core Engine Enhancement.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskType

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Routing decision enumeration."""
    EXECUTE_IMMEDIATE = "execute_immediate"
    EXECUTE_PARALLEL = "execute_parallel"
    WAIT_FOR_DEPENDENCIES = "wait_for_dependencies"
    QUEUE_LOW_PRIORITY = "queue_low_priority"
    DELAY_EXECUTION = "delay_execution"


@dataclass
class RoutingConfig:
    """Configuration for task routing.
    
    Attributes:
        high_priority_threshold: Priority threshold for high priority tasks
        low_priority_threshold: Priority threshold for low priority tasks
        max_wait_time_seconds: Maximum wait time before boosting priority
        dependency_check_enabled: Whether to check dependencies
        resource_check_enabled: Whether to check resource availability
    """
    high_priority_threshold: float = 0.7
    low_priority_threshold: float = 0.3
    max_wait_time_seconds: float = 60.0
    dependency_check_enabled: bool = True
    resource_check_enabled: bool = True


@dataclass
class RoutingResult:
    """Result of task routing decision.
    
    Attributes:
        task_id: Task identifier
        decision: Routing decision
        priority: Calculated priority
        task_type: Classified task type
        reason: Reason for the decision
        estimated_wait_time: Estimated wait time in seconds
        dependencies: List of dependency task IDs
    """
    task_id: str
    decision: RoutingDecision
    priority: float
    task_type: TaskType
    reason: str
    estimated_wait_time: float = 0.0
    dependencies: List[str] = field(default_factory=list)


class TaskRouter:
    """Task router for classifying and routing tasks.
    
    The TaskRouter provides:
    - Task classification based on keywords and patterns
    - Priority calculation based on multiple factors
    - Routing decisions based on priority and resources
    """
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        """Initialize TaskRouter.
        
        Args:
            config: Routing configuration
        """
        self.config = config or RoutingConfig()
        
        # Keyword patterns for task classification
        self._cpu_keywords = [
            "compute", "calculate", "process", "transform",
            "analyze", "compile", "build", "generate",
            "cpu", "algorithm", "math", "numeric"
        ]
        
        self._io_keywords = [
            "read", "write", "file", "save", "load",
            "download", "upload", "copy", "move",
            "io", "disk", "storage", "database"
        ]
        
        self._llm_keywords = [
            "generate", "analyze", "summarize", "explain",
            "translate", "classify", "extract", "llm",
            "ai", "model", "inference", "prompt"
        ]
        
        # Task type weights for priority calculation
        self._type_weights = {
            TaskType.CPU_BOUND: 0.7,
            TaskType.IO_BOUND: 0.5,
            TaskType.LLM_BOUND: 0.6,
            TaskType.MIXED: 0.5,
        }
    
    def classify_task(self, task: PriorityTask) -> TaskType:
        """Classify task type based on description and metadata.
        
        Args:
            task: Task to classify
            
        Returns:
            TaskType classification
        """
        # Extract text from task
        text = f"{task.description} {task.metadata.get('context', '')}".lower()
        
        # Count keyword matches
        scores = {
            TaskType.CPU_BOUND: 0,
            TaskType.IO_BOUND: 0,
            TaskType.LLM_BOUND: 0,
        }
        
        for keyword in self._cpu_keywords:
            if keyword in text:
                scores[TaskType.CPU_BOUND] += 1
        
        for keyword in self._io_keywords:
            if keyword in text:
                scores[TaskType.IO_BOUND] += 1
        
        for keyword in self._llm_keywords:
            if keyword in text:
                scores[TaskType.LLM_BOUND] += 1
        
        # Find dominant type
        max_score = max(scores.values())
        if max_score == 0:
            return TaskType.MIXED
        
        # Return the type with highest score
        dominant_type = max(scores, key=scores.get)
        
        logger.debug(f"Classified task {task.id} as {dominant_type.value} (score={max_score})")
        return dominant_type
    
    def calculate_priority(self, task: PriorityTask) -> float:
        """Calculate task priority based on multiple factors.
        
        Priority formula:
        priority = base_priority * 0.4 + 
                   deadline_factor * 0.3 + 
                   dependency_factor * 0.2 + 
                   type_factor * 0.1
        
        Args:
            task: Task to calculate priority for
            
        Returns:
            Priority score (0.0-1.0)
        """
        # Base priority from user (0.0-1.0)
        base_priority = task.priority
        
        # Deadline factor (0.0-1.0)
        deadline_factor = self._calculate_deadline_factor(task)
        
        # Dependency factor (0.0-1.0)
        dependency_factor = self._calculate_dependency_factor(task)
        
        # Type factor (0.0-1.0)
        type_factor = self._calculate_type_factor(task)
        
        # Calculate weighted priority
        priority = (
            base_priority * 0.4 +
            deadline_factor * 0.3 +
            dependency_factor * 0.2 +
            type_factor * 0.1
        )
        
        # Clamp to 0.0-1.0
        priority = min(1.0, max(0.0, priority))
        
        logger.debug(
            f"Calculated priority for task {task.id}: "
            f"{priority:.2f} (base={base_priority:.2f}, "
            f"deadline={deadline_factor:.2f}, dep={dependency_factor:.2f}, "
            f"type={type_factor:.2f})"
        )
        
        return priority
    
    def _calculate_deadline_factor(self, task: PriorityTask) -> float:
        """Calculate deadline urgency factor.
        
        Args:
            task: Task to calculate factor for
            
        Returns:
            Deadline factor (0.0-1.0)
        """
        if not task.deadline:
            return 0.5  # Default middle value
        
        now = datetime.now()
        time_remaining = (task.deadline - now).total_seconds()
        
        if time_remaining <= 0:
            return 1.0  # Already overdue
        
        # Calculate total time from creation to deadline
        total_time = (task.deadline - task.created_at).total_seconds()
        
        if total_time <= 0:
            return 1.0
        
        # Urgency increases as deadline approaches
        urgency = 1.0 - (time_remaining / total_time)
        
        return min(1.0, max(0.0, urgency))
    
    def _calculate_dependency_factor(self, task: PriorityTask) -> float:
        """Calculate dependency factor.
        
        Tasks with fewer dependencies have higher priority.
        
        Args:
            task: Task to calculate factor for
            
        Returns:
            Dependency factor (0.0-1.0)
        """
        num_dependencies = len(task.dependencies)
        
        # Factor = 1.0 / (1.0 + num_dependencies)
        # More dependencies = lower factor
        factor = 1.0 / (1.0 + num_dependencies)
        
        return factor
    
    def _calculate_type_factor(self, task: PriorityTask) -> float:
        """Calculate task type factor.
        
        Args:
            task: Task to calculate factor for
            
        Returns:
            Type factor (0.0-1.0)
        """
        task_type = self.classify_task(task)
        return self._type_weights.get(task_type, 0.5)
    
    def route(self, task: PriorityTask, resource_available: bool = True) -> RoutingResult:
        """Make routing decision for a task.
        
        Decision matrix:
        - Priority > 0.8, no deps, resources available -> EXECUTE_IMMEDIATE
        - Priority > 0.8, has deps -> WAIT_FOR_DEPENDENCIES
        - Priority 0.5-0.8, no deps -> EXECUTE_PARALLEL
        - Priority 0.5-0.8, resources unavailable -> QUEUE_LOW_PRIORITY
        - Priority < 0.5 -> DELAY_EXECUTION
        
        Args:
            task: Task to route
            resource_available: Whether resources are available
            
        Returns:
            RoutingResult with decision and metadata
        """
        # Calculate priority
        priority = self.calculate_priority(task)
        
        # Classify task type
        task_type = self.classify_task(task)
        
        # Check dependencies
        has_dependencies = len(task.dependencies) > 0
        
        # Make decision
        if priority > self.config.high_priority_threshold:
            if has_dependencies:
                decision = RoutingDecision.WAIT_FOR_DEPENDENCIES
                reason = "High priority task waiting for dependencies"
            elif resource_available:
                decision = RoutingDecision.EXECUTE_IMMEDIATE
                reason = "High priority task with resources available"
            else:
                decision = RoutingDecision.EXECUTE_PARALLEL
                reason = "High priority task, resources busy"
        
        elif priority > self.config.low_priority_threshold:
            if has_dependencies:
                decision = RoutingDecision.WAIT_FOR_DEPENDENCIES
                reason = "Medium priority task waiting for dependencies"
            elif resource_available:
                decision = RoutingDecision.EXECUTE_PARALLEL
                reason = "Medium priority task, executing in parallel"
            else:
                decision = RoutingDecision.QUEUE_LOW_PRIORITY
                reason = "Medium priority task, queuing due to resource constraints"
        
        else:
            decision = RoutingDecision.DELAY_EXECUTION
            reason = "Low priority task, delaying execution"
        
        # Estimate wait time
        estimated_wait = 0.0
        if decision == RoutingDecision.WAIT_FOR_DEPENDENCIES:
            estimated_wait = 5.0 * len(task.dependencies)  # Rough estimate
        elif decision == RoutingDecision.QUEUE_LOW_PRIORITY:
            estimated_wait = self.config.max_wait_time_seconds * 0.5
        elif decision == RoutingDecision.DELAY_EXECUTION:
            estimated_wait = self.config.max_wait_time_seconds
        
        result = RoutingResult(
            task_id=task.id,
            decision=decision,
            priority=priority,
            task_type=task_type,
            reason=reason,
            estimated_wait_time=estimated_wait,
            dependencies=list(task.dependencies),
        )
        
        logger.info(
            f"Routed task {task.id}: {decision.value} "
            f"(priority={priority:.2f}, type={task_type.value})"
        )
        
        return result
    
    def boost_priority(self, task: PriorityTask, wait_time: float) -> float:
        """Boost task priority based on wait time.
        
        Prevents starvation by increasing priority for long-waiting tasks.
        
        Args:
            task: Task to boost
            wait_time: Time waited in seconds
            
        Returns:
            New priority value
        """
        if wait_time < self.config.max_wait_time_seconds:
            return task.priority
        
        # Calculate boost factor
        boost = min(0.5, wait_time / (2 * self.config.max_wait_time_seconds))
        
        new_priority = min(1.0, task.priority + boost)
        
        logger.debug(
            f"Boosted priority for task {task.id}: "
            f"{task.priority:.2f} -> {new_priority:.2f} "
            f"(wait_time={wait_time:.1f}s)"
        )
        
        return new_priority
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics.
        
        Returns:
            Dictionary with routing statistics
        """
        return {
            "config": {
                "high_priority_threshold": self.config.high_priority_threshold,
                "low_priority_threshold": self.config.low_priority_threshold,
                "max_wait_time_seconds": self.config.max_wait_time_seconds,
            },
            "keyword_counts": {
                "cpu": len(self._cpu_keywords),
                "io": len(self._io_keywords),
                "llm": len(self._llm_keywords),
            },
        }


class PriorityManager:
    """Manager for task priority queues and aging.
    
    Provides:
    - Priority queue management
    - Priority aging to prevent starvation
    - Dynamic priority updates
    """
    
    def __init__(self, router: Optional[TaskRouter] = None):
        """Initialize PriorityManager.
        
        Args:
            router: TaskRouter for priority calculation
        """
        self.router = router or TaskRouter()
        self._task_priorities: Dict[str, float] = {}
        self._task_wait_times: Dict[str, float] = {}
    
    def update_priority(self, task: PriorityTask) -> float:
        """Update task priority based on current state.
        
        Args:
            task: Task to update
            
        Returns:
            New priority value
        """
        # Calculate base priority
        new_priority = self.router.calculate_priority(task)
        
        # Apply aging boost
        wait_time = self._task_wait_times.get(task.id, 0.0)
        if wait_time > 0:
            new_priority = self.router.boost_priority(task, wait_time)
        
        # Store updated priority
        self._task_priorities[task.id] = new_priority
        
        return new_priority
    
    def update_wait_time(self, task_id: str, wait_time: float) -> None:
        """Update wait time for a task.
        
        Args:
            task_id: Task identifier
            wait_time: Wait time in seconds
        """
        self._task_wait_times[task_id] = wait_time
    
    def get_priority(self, task_id: str) -> Optional[float]:
        """Get stored priority for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Priority value or None if not found
        """
        return self._task_priorities.get(task_id)
    
    def get_wait_time(self, task_id: str) -> float:
        """Get stored wait time for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Wait time in seconds
        """
        return self._task_wait_times.get(task_id, 0.0)
    
    def reset(self) -> None:
        """Reset all stored priorities and wait times."""
        self._task_priorities.clear()
        self._task_wait_times.clear()


@dataclass
class RoutingBatchResult:
    """Result of batch routing for multiple tasks.
    
    Attributes:
        total_tasks: Total number of tasks routed
        immediate_count: Number of tasks to execute immediately
        parallel_count: Number of tasks to execute in parallel
        waiting_count: Number of tasks waiting for dependencies
        delayed_count: Number of tasks delayed
        routing_results: List of individual routing results
    """
    total_tasks: int
    immediate_count: int
    parallel_count: int
    waiting_count: int
    delayed_count: int
    routing_results: List[RoutingResult]
    
    @classmethod
    def from_results(cls, results: List[RoutingResult]) -> 'RoutingBatchResult':
        """Create batch result from list of routing results.
        
        Args:
            results: List of routing results
            
        Returns:
            Batch result
        """
        immediate = sum(1 for r in results if r.decision == RoutingDecision.EXECUTE_IMMEDIATE)
        parallel = sum(1 for r in results if r.decision == RoutingDecision.EXECUTE_PARALLEL)
        waiting = sum(1 for r in results if r.decision == RoutingDecision.WAIT_FOR_DEPENDENCIES)
        delayed = sum(1 for r in results if r.decision == RoutingDecision.DELAY_EXECUTION)
        
        return cls(
            total_tasks=len(results),
            immediate_count=immediate,
            parallel_count=parallel,
            waiting_count=waiting,
            delayed_count=delayed,
            routing_results=results,
        )
