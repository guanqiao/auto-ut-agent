"""Load Balancer Module.

Provides intelligent load balancing with:
- Real-time worker load monitoring
- Multiple load balancing algorithms
- Adaptive weight adjustment
- Hotspot detection and prevention

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


class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration."""
    ROUND_ROBIN = auto()
    LEAST_CONNECTIONS = auto()
    WEIGHTED_LEAST_CONNECTIONS = auto()
    SHORTEST_EXPECTED_TIME = auto()


@dataclass
class WorkerStats:
    """Statistics for a worker node.
    
    Attributes:
        worker_id: Worker identifier
        current_tasks: Number of currently assigned tasks
        max_capacity: Maximum task capacity
        completed_tasks: Total completed tasks
        failed_tasks: Total failed tasks
        avg_execution_time: Average execution time in milliseconds
        success_rate: Success rate (0.0-1.0)
        last_task_time: Timestamp of last task assignment
        resource_utilization: Resource utilization percentage
    """
    worker_id: str
    current_tasks: int = 0
    max_capacity: int = 10
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 1.0
    last_task_time: Optional[datetime] = None
    resource_utilization: float = 0.0
    
    @property
    def load(self) -> float:
        """Get current load percentage."""
        if self.max_capacity == 0:
            return 0.0
        return self.current_tasks / self.max_capacity
    
    @property
    def available(self) -> int:
        """Get available capacity."""
        return self.max_capacity - self.current_tasks
    
    def update_execution_time(self, execution_time: float) -> None:
        """Update average execution time using exponential moving average.
        
        Args:
            execution_time: New execution time in milliseconds
        """
        alpha = 0.3  # Smoothing factor
        self.avg_execution_time = (
            alpha * execution_time +
            (1 - alpha) * self.avg_execution_time
        )
    
    def update_success_rate(self, success: bool) -> None:
        """Update success rate.
        
        Args:
            success: Whether the task succeeded
        """
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            self.success_rate = 1.0
        else:
            self.success_rate = self.completed_tasks / total


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer.
    
    Attributes:
        strategy: Load balancing strategy
        enable_adaptive_weights: Enable adaptive weight adjustment
        enable_hotspot_detection: Enable hotspot detection
        hotspot_threshold: Load threshold for hotspot detection
        weight_adjustment_interval: Interval for weight adjustment in seconds
        max_history_size: Maximum size of history to keep
    """
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS
    enable_adaptive_weights: bool = True
    enable_hotspot_detection: bool = True
    hotspot_threshold: float = 0.8
    weight_adjustment_interval: float = 60.0
    max_history_size: int = 1000


@dataclass
class LoadBalanceDecision:
    """Result of load balancing decision.
    
    Attributes:
        worker_id: Selected worker ID
        strategy: Strategy used for decision
        score: Decision score
        reason: Reason for selection
        estimated_wait_time: Estimated wait time in seconds
    """
    worker_id: str
    strategy: LoadBalancingStrategy
    score: float
    reason: str
    estimated_wait_time: float = 0.0


class LoadMonitor:
    """Monitor for tracking worker load in real-time.
    
    Provides:
    - Real-time load tracking
    - Execution time statistics
    - Success rate monitoring
    - Hotspot detection
    """
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        """Initialize LoadMonitor.
        
        Args:
            config: Load balancer configuration
        """
        self.config = config or LoadBalancerConfig()
        self._worker_stats: Dict[str, WorkerStats] = {}
        self._task_history: List[Tuple[str, str, float, bool]] = []  # (worker_id, task_id, time, success)
        self._load_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    
    def register_worker(self, worker_id: str, max_capacity: int = 10) -> None:
        """Register a worker for monitoring.
        
        Args:
            worker_id: Worker identifier
            max_capacity: Maximum task capacity
        """
        if worker_id not in self._worker_stats:
            self._worker_stats[worker_id] = WorkerStats(
                worker_id=worker_id,
                max_capacity=max_capacity,
            )
            logger.debug(f"Registered worker {worker_id} with capacity {max_capacity}")
    
    def assign_task(self, worker_id: str, task_id: str) -> None:
        """Record task assignment to worker.
        
        Args:
            worker_id: Worker identifier
            task_id: Task identifier
        """
        if worker_id not in self._worker_stats:
            self.register_worker(worker_id)
        
        stats = self._worker_stats[worker_id]
        stats.current_tasks += 1
        stats.last_task_time = datetime.now()
        
        # Record load history
        self._load_history[worker_id].append((datetime.now(), stats.load))
        
        # Keep history size limited
        if len(self._load_history[worker_id]) > self.config.max_history_size:
            self._load_history[worker_id] = self._load_history[worker_id][-self.config.max_history_size:]
    
    def complete_task(self, worker_id: str, task_id: str, execution_time: float, success: bool) -> None:
        """Record task completion.
        
        Args:
            worker_id: Worker identifier
            task_id: Task identifier
            execution_time: Execution time in milliseconds
            success: Whether task succeeded
        """
        if worker_id not in self._worker_stats:
            return
        
        stats = self._worker_stats[worker_id]
        stats.current_tasks = max(0, stats.current_tasks - 1)
        
        if success:
            stats.completed_tasks += 1
        else:
            stats.failed_tasks += 1
        
        stats.update_execution_time(execution_time)
        stats.update_success_rate(success)
        
        # Record to history
        self._task_history.append((worker_id, task_id, execution_time, success))
        if len(self._task_history) > self.config.max_history_size:
            self._task_history = self._task_history[-self.config.max_history_size:]
    
    def get_worker_stats(self, worker_id: str) -> Optional[WorkerStats]:
        """Get statistics for a worker.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            Worker statistics or None if not found
        """
        return self._worker_stats.get(worker_id)
    
    def get_all_stats(self) -> Dict[str, WorkerStats]:
        """Get statistics for all workers.
        
        Returns:
            Dictionary of worker statistics
        """
        return self._worker_stats.copy()
    
    def is_hotspot(self, worker_id: str) -> bool:
        """Check if a worker is a hotspot (overloaded).
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            True if worker is a hotspot
        """
        if worker_id not in self._worker_stats:
            return False
        
        stats = self._worker_stats[worker_id]
        return stats.load > self.config.hotspot_threshold
    
    def get_hotspots(self) -> List[str]:
        """Get list of hotspot workers.
        
        Returns:
            List of hotspot worker IDs
        """
        return [
            worker_id for worker_id, stats in self._worker_stats.items()
            if self.is_hotspot(worker_id)
        ]
    
    def get_load_trend(self, worker_id: str, window: timedelta = timedelta(minutes=5)) -> float:
        """Get load trend for a worker.
        
        Args:
            worker_id: Worker identifier
            window: Time window for trend calculation
            
        Returns:
            Load trend (positive = increasing, negative = decreasing)
        """
        if worker_id not in self._load_history:
            return 0.0
        
        history = self._load_history[worker_id]
        if len(history) < 2:
            return 0.0
        
        cutoff = datetime.now() - window
        recent = [(ts, load) for ts, load in history if ts > cutoff]
        
        if len(recent) < 2:
            return 0.0
        
        # Simple linear trend
        loads = [load for _, load in recent]
        if len(loads) < 2:
            return 0.0
        
        return (loads[-1] - loads[0]) / len(loads)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self._worker_stats:
            return {}
        
        loads = [stats.load for stats in self._worker_stats.values()]
        success_rates = [stats.success_rate for stats in self._worker_stats.values()]
        
        return {
            "total_workers": len(self._worker_stats),
            "avg_load": statistics.mean(loads) if loads else 0.0,
            "max_load": max(loads) if loads else 0.0,
            "min_load": min(loads) if loads else 0.0,
            "load_stddev": statistics.stdev(loads) if len(loads) > 1 else 0.0,
            "avg_success_rate": statistics.mean(success_rates) if success_rates else 0.0,
            "hotspot_count": len(self.get_hotspots()),
        }


class LoadBalancer:
    """Load balancer for distributing tasks across workers.
    
    Provides:
    - Multiple load balancing strategies
    - Adaptive weight adjustment
    - Worker selection
    """
    
    def __init__(
        self,
        monitor: Optional[LoadMonitor] = None,
        config: Optional[LoadBalancerConfig] = None,
    ):
        """Initialize LoadBalancer.
        
        Args:
            monitor: Load monitor instance
            config: Load balancer configuration
        """
        self.monitor = monitor or LoadMonitor(config)
        self.config = config or LoadBalancerConfig()
        self._worker_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._round_robin_index = 0
    
    def select_worker(
        self,
        task: Optional[PriorityTask] = None,
        available_workers: Optional[List[str]] = None,
    ) -> LoadBalanceDecision:
        """Select the best worker for a task.
        
        Args:
            task: Task to assign (optional)
            available_workers: List of available worker IDs (optional)
            
        Returns:
            LoadBalanceDecision with selected worker
        """
        if available_workers is None:
            available_workers = list(self.monitor._worker_stats.keys())
        
        if not available_workers:
            raise ValueError("No available workers")
        
        # Select based on strategy
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            worker_id = self._select_round_robin(available_workers)
            return LoadBalanceDecision(
                worker_id=worker_id,
                strategy=self.config.strategy,
                score=0.0,
                reason="Round-robin selection",
            )
        
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            worker_id, score = self._select_least_connections(available_workers)
            return LoadBalanceDecision(
                worker_id=worker_id,
                strategy=self.config.strategy,
                score=score,
                reason="Least connections selection",
            )
        
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS:
            worker_id, score = self._select_weighted_least_connections(available_workers)
            return LoadBalanceDecision(
                worker_id=worker_id,
                strategy=self.config.strategy,
                score=score,
                reason="Weighted least connections selection",
            )
        
        elif self.config.strategy == LoadBalancingStrategy.SHORTEST_EXPECTED_TIME:
            worker_id, score = self._select_shortest_time(available_workers, task)
            return LoadBalanceDecision(
                worker_id=worker_id,
                strategy=self.config.strategy,
                score=score,
                reason="Shortest expected time selection",
            )
        
        raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def _select_round_robin(self, workers: List[str]) -> str:
        """Select worker using round-robin.
        
        Args:
            workers: List of worker IDs
            
        Returns:
            Selected worker ID
        """
        self._round_robin_index = self._round_robin_index % len(workers)
        selected = workers[self._round_robin_index]
        self._round_robin_index += 1
        return selected
    
    def _select_least_connections(self, workers: List[str]) -> Tuple[str, float]:
        """Select worker with least connections.
        
        Args:
            workers: List of worker IDs
            
        Returns:
            Tuple of (worker_id, score)
        """
        min_load = float('inf')
        selected = workers[0]
        
        for worker_id in workers:
            stats = self.monitor.get_worker_stats(worker_id)
            if not stats:
                continue
            
            if stats.load < min_load:
                min_load = stats.load
                selected = worker_id
        
        return selected, min_load
    
    def _select_weighted_least_connections(self, workers: List[str]) -> Tuple[str, float]:
        """Select worker using weighted least connections.
        
        Score = (load * weight) where weight is based on historical performance.
        
        Args:
            workers: List of worker IDs
            
        Returns:
            Tuple of (worker_id, score)
        """
        min_score = float('inf')
        selected = workers[0]
        
        for worker_id in workers:
            stats = self.monitor.get_worker_stats(worker_id)
            if not stats:
                continue
            
            # Calculate weight based on success rate and execution time
            weight = self._calculate_worker_weight(worker_id)
            
            # Score = load * weight (lower is better)
            score = stats.load * weight
            
            if score < min_score:
                min_score = score
                selected = worker_id
        
        return selected, min_score
    
    def _select_shortest_time(self, workers: List[str], task: Optional[PriorityTask]) -> Tuple[str, float]:
        """Select worker with shortest expected execution time.
        
        Args:
            workers: List of worker IDs
            task: Task to assign
            
        Returns:
            Tuple of (worker_id, score)
        """
        min_time = float('inf')
        selected = workers[0]
        
        for worker_id in workers:
            stats = self.monitor.get_worker_stats(worker_id)
            if not stats:
                continue
            
            # Estimated time = avg_execution_time * (1 + load)
            estimated_time = stats.avg_execution_time * (1 + stats.load)
            
            if estimated_time < min_time:
                min_time = estimated_time
                selected = worker_id
        
        return selected, min_time
    
    def _calculate_worker_weight(self, worker_id: str) -> float:
        """Calculate weight for a worker based on historical performance.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            Weight value (higher = better performance)
        """
        stats = self.monitor.get_worker_stats(worker_id)
        if not stats:
            return 1.0
        
        # Weight based on success rate and execution time
        success_weight = stats.success_rate
        
        # Normalize execution time (faster = higher weight)
        time_weight = 1.0
        if stats.avg_execution_time > 0:
            time_weight = min(2.0, 1000.0 / stats.avg_execution_time)  # Normalize to ~1.0
        
        # Combined weight
        weight = success_weight * time_weight
        
        # Apply adaptive adjustment if enabled
        if self.config.enable_adaptive_weights:
            weight *= self._worker_weights[worker_id]
        
        return max(0.1, weight)  # Minimum weight
    
    def update_weights(self) -> None:
        """Update worker weights based on recent performance."""
        if not self.config.enable_adaptive_weights:
            return
        
        for worker_id, stats in self.monitor._worker_stats.items():
            # Calculate new weight
            new_weight = self._calculate_worker_weight(worker_id)
            
            # Smooth adjustment
            old_weight = self._worker_weights[worker_id]
            self._worker_weights[worker_id] = 0.8 * old_weight + 0.2 * new_weight
    
    def record_assignment(self, worker_id: str, task_id: str) -> None:
        """Record task assignment.
        
        Args:
            worker_id: Worker identifier
            task_id: Task identifier
        """
        self.monitor.assign_task(worker_id, task_id)
    
    def record_completion(
        self,
        worker_id: str,
        task_id: str,
        execution_time: float,
        success: bool,
    ) -> None:
        """Record task completion.
        
        Args:
            worker_id: Worker identifier
            task_id: Task identifier
            execution_time: Execution time in milliseconds
            success: Whether task succeeded
        """
        self.monitor.complete_task(worker_id, task_id, execution_time, success)
    
    def get_balance_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics.
        
        Returns:
            Dictionary with statistics
        """
        summary = self.monitor.get_stats_summary()
        
        return {
            **summary,
            "strategy": self.config.strategy.name,
            "adaptive_weights_enabled": self.config.enable_adaptive_weights,
            "hotspot_detection_enabled": self.config.enable_hotspot_detection,
            "worker_weights": dict(self._worker_weights),
        }
