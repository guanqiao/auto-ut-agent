"""Progress Tracker Module.

Provides real-time progress tracking with:
- Task completion monitoring
- ETA prediction
- Bottleneck detection
- Performance metrics

This is part of Phase 2 Week 3-4: Load Balancing and Optimization.
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class TaskProgress:
    """Progress information for a task.
    
    Attributes:
        task_id: Task identifier
        status: Task status
        started_at: Task start time
        completed_at: Task completion time
        duration_ms: Task duration in milliseconds
        progress_percentage: Progress percentage (0-100)
        estimated_remaining_ms: Estimated remaining time
    """
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    progress_percentage: float = 0.0
    estimated_remaining_ms: float = 0.0


@dataclass
class ETAPrediction:
    """ETA prediction result.
    
    Attributes:
        estimated_completion: Estimated completion time
        remaining_tasks: Number of remaining tasks
        avg_speed: Average completion speed (tasks per second)
        confidence: Confidence level (0.0-1.0)
        confidence_interval: Confidence interval in seconds
        prediction_method: Method used for prediction
    """
    estimated_completion: datetime
    remaining_tasks: int
    avg_speed: float
    confidence: float
    confidence_interval: float
    prediction_method: str


@dataclass
class BottleneckInfo:
    """Information about a bottleneck.
    
    Attributes:
        bottleneck_type: Type of bottleneck (task/worker/resource)
        bottleneck_id: Identifier of the bottleneck
        severity: Severity level (0.0-1.0)
        description: Description of the bottleneck
        recommendation: Recommendation for resolution
        detected_at: Detection timestamp
    """
    bottleneck_type: str
    bottleneck_id: str
    severity: float
    description: str
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProgressTrackerConfig:
    """Configuration for progress tracker.
    
    Attributes:
        enable_eta_prediction: Enable ETA prediction
        enable_bottleneck_detection: Enable bottleneck detection
        eta_update_interval: ETA update interval in seconds
        bottleneck_threshold: Threshold for bottleneck detection
        slow_task_threshold: Threshold for slow task detection in seconds
    """
    enable_eta_prediction: bool = True
    enable_bottleneck_detection: bool = True
    eta_update_interval: float = 5.0
    bottleneck_threshold: float = 0.8
    slow_task_threshold: float = 30.0


class ProgressTracker:
    """Tracker for monitoring task progress in real-time.
    
    Provides:
    - Progress percentage calculation
    - ETA prediction
    - Bottleneck detection
    - Performance metrics
    """
    
    def __init__(self, config: Optional[ProgressTrackerConfig] = None):
        """Initialize ProgressTracker.
        
        Args:
            config: Progress tracker configuration
        """
        self.config = config or ProgressTrackerConfig()
        self._task_progress: Dict[str, TaskProgress] = {}
        self._completion_times: List[float] = []  # in milliseconds
        self._start_time: Optional[datetime] = None
        self._total_tasks: int = 0
        self._completed_tasks: int = 0
        self._failed_tasks: int = 0
        self._bottlenecks: List[BottleneckInfo] = []
        self._last_eta_update: Optional[datetime] = None
    
    def initialize(self, tasks: List[PriorityTask]) -> None:
        """Initialize tracker with task list.
        
        Args:
            tasks: List of tasks to track
        """
        self._total_tasks = len(tasks)
        self._start_time = datetime.now()
        
        for task in tasks:
            self._task_progress[task.id] = TaskProgress(
                task_id=task.id,
                status=task.status,
            )
        
        logger.info(f"Initialized progress tracker with {len(tasks)} tasks")
    
    def start_task(self, task_id: str) -> None:
        """Mark a task as started.
        
        Args:
            task_id: Task identifier
        """
        if task_id not in self._task_progress:
            self._task_progress[task_id] = TaskProgress(task_id=task_id)
        
        progress = self._task_progress[task_id]
        progress.status = TaskStatus.RUNNING
        progress.started_at = datetime.now()
        progress.progress_percentage = 0.0
        
        logger.debug(f"Task {task_id} started")
    
    def update_progress(self, task_id: str, percentage: float) -> None:
        """Update task progress percentage.
        
        Args:
            task_id: Task identifier
            percentage: Progress percentage (0-100)
        """
        if task_id not in self._task_progress:
            return
        
        progress = self._task_progress[task_id]
        progress.progress_percentage = min(100.0, max(0.0, percentage))
    
    def complete_task(self, task_id: str, duration_ms: float, success: bool = True) -> None:
        """Mark a task as completed.
        
        Args:
            task_id: Task identifier
            duration_ms: Task duration in milliseconds
            success: Whether task succeeded
        """
        if task_id not in self._task_progress:
            self._task_progress[task_id] = TaskProgress(task_id=task_id)
        
        progress = self._task_progress[task_id]
        progress.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        progress.completed_at = datetime.now()
        progress.duration_ms = duration_ms
        progress.progress_percentage = 100.0
        
        # Record completion time
        self._completion_times.append(duration_ms)
        
        # Update counters
        if success:
            self._completed_tasks += 1
        else:
            self._failed_tasks += 1
        
        logger.debug(f"Task {task_id} completed in {duration_ms:.0f}ms (success={success})")
        
        # Check for bottlenecks
        if self.config.enable_bottleneck_detection:
            self._check_bottlenecks(task_id, duration_ms)
    
    def get_overall_progress(self) -> float:
        """Get overall progress percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if self._total_tasks == 0:
            return 0.0
        
        # Calculate based on completed tasks
        return (self._completed_tasks + self._failed_tasks) / self._total_tasks * 100
    
    def get_task_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get progress for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task progress or None if not found
        """
        return self._task_progress.get(task_id)
    
    def predict_eta(self) -> Optional[ETAPrediction]:
        """Predict estimated time of arrival.
        
        Returns:
            ETA prediction or None if not enough data
        """
        if not self.config.enable_eta_prediction:
            return None
        
        remaining_tasks = self._total_tasks - self._completed_tasks - self._failed_tasks
        
        if remaining_tasks == 0:
            return ETAPrediction(
                estimated_completion=datetime.now(),
                remaining_tasks=0,
                avg_speed=0.0,
                confidence=1.0,
                confidence_interval=0.0,
                prediction_method="completed",
            )
        
        # Need at least some completion data
        if len(self._completion_times) < 3:
            return None
        
        # Calculate average speed
        avg_time_ms = statistics.mean(self._completion_times[-10:])  # Last 10 tasks
        avg_speed = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0  # tasks per second
        
        # Predict remaining time
        remaining_time_seconds = remaining_tasks / avg_speed if avg_speed > 0 else float('inf')
        
        # Calculate confidence based on sample size
        sample_size = len(self._completion_times)
        confidence = min(1.0, sample_size / 20.0)  # Saturate at 20 samples
        
        # Calculate confidence interval
        if len(self._completion_times) > 1:
            time_stddev = statistics.stdev(self._completion_times[-10:])
            confidence_interval = (time_stddev / avg_time_ms) * remaining_time_seconds if avg_time_ms > 0 else 0.0
        else:
            confidence_interval = remaining_time_seconds * 0.5  # 50% uncertainty
        
        estimated_completion = datetime.now() + timedelta(seconds=remaining_time_seconds)
        
        return ETAPrediction(
            estimated_completion=estimated_completion,
            remaining_tasks=remaining_tasks,
            avg_speed=avg_speed,
            confidence=confidence,
            confidence_interval=confidence_interval,
            prediction_method="moving_average",
        )
    
    def _check_bottlenecks(self, task_id: str, duration_ms: float) -> None:
        """Check for bottlenecks.
        
        Args:
            task_id: Task identifier
            duration_ms: Task duration in milliseconds
        """
        # Check for slow task
        if duration_ms > self.config.slow_task_threshold * 1000:
            bottleneck = BottleneckInfo(
                bottleneck_type="task",
                bottleneck_id=task_id,
                severity=min(1.0, duration_ms / (self.config.slow_task_threshold * 1000 * 2)),
                description=f"Task {task_id} took {duration_ms/1000:.1f}s (threshold: {self.config.slow_task_threshold}s)",
                recommendation="Consider optimizing task or splitting into smaller tasks",
            )
            self._bottlenecks.append(bottleneck)
            logger.warning(f"Bottleneck detected: {bottleneck.description}")
        
        # Check for worker bottlenecks (if we had worker tracking)
        # This would be integrated with LoadBalancer
    
    def get_bottlenecks(self) -> List[BottleneckInfo]:
        """Get detected bottlenecks.
        
        Returns:
            List of bottleneck information
        """
        return self._bottlenecks.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self._completion_times:
            return {}
        
        recent_times = self._completion_times[-10:]
        
        return {
            "total_tasks": self._total_tasks,
            "completed_tasks": self._completed_tasks,
            "failed_tasks": self._failed_tasks,
            "progress_percentage": self.get_overall_progress(),
            "avg_duration_ms": statistics.mean(self._completion_times),
            "recent_avg_duration_ms": statistics.mean(recent_times) if recent_times else 0.0,
            "min_duration_ms": min(self._completion_times),
            "max_duration_ms": max(self._completion_times),
            "stddev_duration_ms": statistics.stdev(self._completion_times) if len(self._completion_times) > 1 else 0.0,
            "throughput_tasks_per_second": 1000.0 / statistics.mean(recent_times) if recent_times else 0.0,
            "elapsed_time_seconds": (datetime.now() - self._start_time).total_seconds() if self._start_time else 0.0,
        }
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary.
        
        Returns:
            Dictionary with status summary
        """
        eta = self.predict_eta()
        
        summary = {
            "total_tasks": self._total_tasks,
            "completed": self._completed_tasks,
            "failed": self._failed_tasks,
            "in_progress": sum(1 for p in self._task_progress.values() if p.status == TaskStatus.RUNNING),
            "pending": sum(1 for p in self._task_progress.values() if p.status == TaskStatus.PENDING),
            "progress_percentage": self.get_overall_progress(),
        }
        
        if eta:
            summary["eta"] = {
                "estimated_completion": eta.estimated_completion.isoformat(),
                "remaining_tasks": eta.remaining_tasks,
                "avg_speed": eta.avg_speed,
                "confidence": eta.confidence,
                "confidence_interval": eta.confidence_interval,
            }
        
        summary["bottleneck_count"] = len(self._bottlenecks)
        
        return summary
    
    def reset(self) -> None:
        """Reset the tracker."""
        self._task_progress.clear()
        self._completion_times.clear()
        self._start_time = None
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._bottlenecks.clear()
        self._last_eta_update = None


class BottleneckAnalyzer:
    """Analyzer for detecting and diagnosing bottlenecks.
    
    Provides:
    - Bottleneck detection
    - Root cause analysis
    - Optimization recommendations
    """
    
    def __init__(self, tracker: Optional[ProgressTracker] = None):
        """Initialize BottleneckAnalyzer.
        
        Args:
            tracker: Progress tracker instance
        """
        self.tracker = tracker or ProgressTracker()
        self._task_dependencies: Dict[str, Set[str]] = defaultdict(set)
    
    def set_dependencies(self, task_id: str, dependencies: Set[str]) -> None:
        """Set task dependencies.
        
        Args:
            task_id: Task identifier
            dependencies: Set of dependency task IDs
        """
        self._task_dependencies[task_id] = dependencies
    
    def analyze_bottlenecks(self) -> List[BottleneckInfo]:
        """Analyze current bottlenecks.
        
        Returns:
            List of bottleneck information
        """
        bottlenecks = self.tracker.get_bottlenecks()
        
        # Analyze dependency bottlenecks
        for task_id, progress in self.tracker._task_progress.items():
            if progress.status == TaskStatus.PENDING:
                # Check if blocked by dependencies
                pending_deps = [
                    dep for dep in self._task_dependencies[task_id]
                    if self.tracker._task_progress.get(dep, TaskProgress(task_id=dep)).status not in 
                    [TaskStatus.COMPLETED, TaskStatus.FAILED]
                ]
                
                if pending_deps:
                    bottleneck = BottleneckInfo(
                        bottleneck_type="dependency",
                        bottleneck_id=task_id,
                        severity=0.5,
                        description=f"Task {task_id} blocked by {len(pending_deps)} pending dependencies",
                        recommendation=f"Wait for dependencies to complete: {', '.join(pending_deps)}",
                    )
                    bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        bottlenecks = self.analyze_bottlenecks()
        
        # Group by type
        task_bottlenecks = [b for b in bottlenecks if b.bottleneck_type == "task"]
        dependency_bottlenecks = [b for b in bottlenecks if b.bottleneck_type == "dependency"]
        
        if len(task_bottlenecks) > 3:
            recommendations.append("Consider adding more workers to improve parallelism")
        
        if len(dependency_bottlenecks) > 3:
            recommendations.append("Review task dependencies to identify opportunities for parallelization")
        
        metrics = self.tracker.get_performance_metrics()
        if metrics.get("stddev_duration_ms", 0) > metrics.get("avg_duration_ms", 1) * 0.5:
            recommendations.append("High variance in task duration - consider splitting long-running tasks")
        
        return recommendations
