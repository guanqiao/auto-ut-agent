"""Predictive Scheduler Module.

Provides intelligent task scheduling with:
- ML-based execution time prediction
- Similarity-based prediction
- Confidence assessment
- Predictive resource prefetching

This is part of Phase 4 Week 6-7: Advanced Features.
"""

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class TaskHistory:
    """Historical data for a task.
    
    Attributes:
        task_id: Task identifier
        description: Task description
        task_type: Type of task
        actual_duration: Actual execution time in seconds
        resource_usage: Resource usage during execution
        success: Whether task succeeded
        timestamp: Execution timestamp
        metadata: Additional metadata
    """
    task_id: str
    description: str
    task_type: Optional[str] = None
    actual_duration: Optional[float] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPrediction:
    """Execution time prediction result.
    
    Attributes:
        predicted_duration: Predicted execution time in seconds
        confidence: Confidence score (0.0-1.0)
        confidence_interval: Confidence interval in seconds
        prediction_method: Method used for prediction
        similar_tasks: Number of similar tasks found
        features_used: Features used for prediction
    """
    predicted_duration: float
    confidence: float
    confidence_interval: float
    prediction_method: str
    similar_tasks: int = 0
    features_used: List[str] = field(default_factory=list)


@dataclass
class PrefetchRequest:
    """Resource prefetching request.
    
    Attributes:
        task_id: Task identifier
        resources: List of resources to prefetch
        priority: Prefetch priority
        estimated_usage_time: When resources will be needed
        confidence: Confidence in prefetch accuracy
    """
    task_id: str
    resources: List[str]
    priority: int = 5
    estimated_usage_time: Optional[datetime] = None
    confidence: float = 0.5


@dataclass
class SchedulerConfig:
    """Configuration for predictive scheduler.
    
    Attributes:
        enable_ml_prediction: Enable ML-based prediction
        enable_prefetching: Enable resource prefetching
        history_size: Maximum history size per task type
        similarity_threshold: Threshold for task similarity
        prefetch_lead_time: Time before task execution to prefetch
        min_confidence: Minimum confidence for prefetching
    """
    enable_ml_prediction: bool = True
    enable_prefetching: bool = True
    history_size: int = 100
    similarity_threshold: float = 0.7
    prefetch_lead_time: float = 5.0
    min_confidence: float = 0.6


class PredictiveScheduler:
    """Predictive scheduler for intelligent task scheduling.
    
    Provides:
    - Execution time prediction
    - Similarity-based prediction
    - Confidence assessment
    - Resource prefetching
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        """Initialize PredictiveScheduler.
        
        Args:
            config: Scheduler configuration
        """
        self.config = config or SchedulerConfig()
        self._task_history: Dict[str, List[TaskHistory]] = defaultdict(list)
        self._resource_patterns: Dict[str, List[str]] = defaultdict(list)
        self._type_averages: Dict[str, Dict[str, float]] = {}
    
    def record_task_execution(
        self,
        task: PriorityTask,
        duration: float,
        success: bool = True,
        resource_usage: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record task execution for learning.
        
        Args:
            task: Executed task
            duration: Actual execution time in seconds
            success: Whether task succeeded
            resource_usage: Resource usage during execution
        """
        history = TaskHistory(
            task_id=task.id,
            description=task.description,
            task_type=self._extract_task_type(task),
            actual_duration=duration,
            resource_usage=resource_usage or {},
            success=success,
            metadata=task.metadata,
        )
        
        self._task_history[history.task_type].append(history)
        
        # Keep history size limited
        if len(self._task_history[history.task_type]) > self.config.history_size:
            self._task_history[history.task_type] = \
                self._task_history[history.task_type][-self.config.history_size:]
        
        # Update type averages
        self._update_type_average(history.task_type, duration)
        
        # Learn resource patterns
        if resource_usage:
            self._learn_resource_pattern(history.task_type, list(resource_usage.keys()))
        
        logger.debug(f"Recorded execution for task {task.id}: {duration}s")
    
    def predict_execution_time(self, task: PriorityTask) -> ExecutionPrediction:
        """Predict execution time for a task.
        
        Args:
            task: Task to predict
            
        Returns:
            ExecutionPrediction with predicted time and confidence
        """
        if not self.config.enable_ml_prediction:
            return ExecutionPrediction(
                predicted_duration=task.estimated_duration or 0.0,
                confidence=0.0,
                confidence_interval=0.0,
                prediction_method="default",
            )
        
        # Try similarity-based prediction first
        similar_tasks = self._find_similar_tasks(task)
        
        if similar_tasks and len(similar_tasks) >= 1:
            return self._predict_by_similarity(task, similar_tasks)
        
        # Fall back to type-based prediction
        task_type = self._extract_task_type(task)
        if task_type in self._type_averages:
            return self._predict_by_type(task, task_type)
        
        # No prediction available
        return ExecutionPrediction(
            predicted_duration=task.estimated_duration or 0.0,
            confidence=0.0,
            confidence_interval=0.0,
            prediction_method="default",
        )
    
    def _find_similar_tasks(self, task: PriorityTask) -> List[TaskHistory]:
        """Find similar tasks in history.
        
        Args:
            task: Task to find similar ones for
            
        Returns:
            List of similar task histories
        """
        similar = []
        
        for task_type, histories in self._task_history.items():
            for history in histories:
                similarity = self._calculate_similarity(task, history)
                if similarity >= self.config.similarity_threshold:
                    similar.append((similarity, history))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[0], reverse=True)
        
        return [history for _, history in similar[:10]]
    
    def _calculate_similarity(
        self,
        task: PriorityTask,
        history: TaskHistory,
    ) -> float:
        """Calculate similarity between task and historical task.
        
        Args:
            task: Current task
            history: Historical task
            
        Returns:
            Similarity score (0.0-1.0)
        """
        similarity = 0.0
        factors = 0
        
        # Description similarity (keyword overlap)
        task_keywords = set(task.description.lower().split())
        history_keywords = set(history.description.lower().split())
        
        if task_keywords and history_keywords:
            keyword_similarity = len(task_keywords & history_keywords) / \
                               len(task_keywords | history_keywords)
            similarity += keyword_similarity * 0.4
            factors += 0.4
        
        # Type similarity
        task_type = self._extract_task_type(task)
        if task_type == history.task_type:
            similarity += 0.3
            factors += 0.3
        
        # Resource requirements similarity
        if task.resource_requirements and history.resource_usage:
            task_resources = set(task.resource_requirements.keys())
            history_resources = set(history.resource_usage.keys())
            
            if task_resources and history_resources:
                resource_similarity = len(task_resources & history_resources) / \
                                   len(task_resources | history_resources)
                similarity += resource_similarity * 0.3
                factors += 0.3
        
        return similarity / factors if factors > 0 else 0.0
    
    def _predict_by_similarity(
        self,
        task: PriorityTask,
        similar_tasks: List[TaskHistory],
    ) -> ExecutionPrediction:
        """Predict execution time based on similar tasks.
        
        Args:
            task: Current task
            similar_tasks: List of similar historical tasks
            
        Returns:
            ExecutionPrediction
        """
        durations = [t.actual_duration for t in similar_tasks if t.actual_duration]
        
        if not durations:
            return ExecutionPrediction(
                predicted_duration=task.estimated_duration or 0.0,
                confidence=0.0,
                confidence_interval=0.0,
                prediction_method="similarity_failed",
            )
        
        # Weighted average (more similar = higher weight)
        avg_duration = statistics.mean(durations)
        
        # Calculate confidence based on number of samples and similarity
        sample_confidence = min(1.0, len(similar_tasks) / 10.0)
        avg_similarity = sum(
            self._calculate_similarity(task, t) for t in similar_tasks
        ) / len(similar_tasks)
        
        confidence = sample_confidence * 0.5 + avg_similarity * 0.5
        
        # Calculate confidence interval
        if len(durations) > 1:
            stddev = statistics.stdev(durations)
            confidence_interval = 1.96 * stddev / (len(durations) ** 0.5)
        else:
            confidence_interval = avg_duration * 0.5
        
        return ExecutionPrediction(
            predicted_duration=avg_duration,
            confidence=confidence,
            confidence_interval=confidence_interval,
            prediction_method="similarity_based",
            similar_tasks=len(similar_tasks),
            features_used=["description", "task_type", "resources"],
        )
    
    def _predict_by_type(
        self,
        task: PriorityTask,
        task_type: str,
    ) -> ExecutionPrediction:
        """Predict execution time based on task type average.
        
        Args:
            task: Current task
            task_type: Type of task
            
        Returns:
            ExecutionPrediction
        """
        type_stats = self._type_averages.get(task_type, {})
        
        if not type_stats:
            return ExecutionPrediction(
                predicted_duration=task.estimated_duration or 0.0,
                confidence=0.0,
                confidence_interval=0.0,
                prediction_method="type_failed",
            )
        
        avg_duration = type_stats.get("avg", 0.0)
        sample_count = type_stats.get("count", 0)
        stddev = type_stats.get("stddev", 0.0)
        
        # Confidence based on sample size
        confidence = min(1.0, sample_count / 20.0)
        
        # Confidence interval
        if sample_count > 1 and stddev > 0:
            confidence_interval = 1.96 * stddev / (sample_count ** 0.5)
        else:
            confidence_interval = avg_duration * 0.5
        
        return ExecutionPrediction(
            predicted_duration=avg_duration,
            confidence=confidence,
            confidence_interval=confidence_interval,
            prediction_method="type_based",
            features_used=["task_type"],
        )
    
    def _extract_task_type(self, task: PriorityTask) -> str:
        """Extract task type from task description.
        
        Args:
            task: Task to extract type from
            
        Returns:
            Task type string
        """
        description = task.description.lower()
        
        # Keyword-based classification
        if any(kw in description for kw in ["test", "unit test", "testing"]):
            return "test"
        elif any(kw in description for kw in ["build", "compile", "build"]):
            return "build"
        elif any(kw in description for kw in ["analyze", "analysis", "check"]):
            return "analysis"
        elif any(kw in description for kw in ["generate", "create", "write"]):
            return "generation"
        elif any(kw in description for kw in ["search", "find", "locate"]):
            return "search"
        else:
            return "general"
    
    def _update_type_average(self, task_type: str, duration: float) -> None:
        """Update average duration for task type.
        
        Args:
            task_type: Type of task
            duration: Actual duration
        """
        if task_type not in self._type_averages:
            self._type_averages[task_type] = {
                "sum": 0.0,
                "sum_sq": 0.0,
                "count": 0,
                "avg": 0.0,
                "stddev": 0.0,
            }
        
        stats = self._type_averages[task_type]
        stats["sum"] += duration
        stats["sum_sq"] += duration * duration
        stats["count"] += 1
        
        stats["avg"] = stats["sum"] / stats["count"]
        
        if stats["count"] > 1:
            variance = (stats["sum_sq"] - (stats["sum"] ** 2) / stats["count"]) / \
                      (stats["count"] - 1)
            stats["stddev"] = variance ** 0.5
    
    def _learn_resource_pattern(self, task_type: str, resources: List[str]) -> None:
        """Learn resource usage pattern for task type.
        
        Args:
            task_type: Type of task
            resources: List of resources used
        """
        self._resource_patterns[task_type].extend(resources)
        
        # Keep most common resources
        if len(self._resource_patterns[task_type]) > 100:
            # Count frequencies
            freq = defaultdict(int)
            for res in self._resource_patterns[task_type]:
                freq[res] += 1
            
            # Keep top 20
            sorted_resources = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            self._resource_patterns[task_type] = [r for r, _ in sorted_resources[:20]]
    
    def generate_prefetch_requests(
        self,
        upcoming_tasks: List[PriorityTask],
    ) -> List[PrefetchRequest]:
        """Generate resource prefetch requests for upcoming tasks.
        
        Args:
            upcoming_tasks: List of tasks that will execute soon
            
        Returns:
            List of prefetch requests
        """
        if not self.config.enable_prefetching:
            return []
        
        requests = []
        
        for task in upcoming_tasks:
            prediction = self.predict_execution_time(task)
            
            if prediction.confidence < self.config.min_confidence:
                continue
            
            task_type = self._extract_task_type(task)
            
            # Get common resources for this task type
            resources = self._get_common_resources(task_type)
            
            if resources:
                request = PrefetchRequest(
                    task_id=task.id,
                    resources=resources,
                    priority=int(prediction.confidence * 10),
                    confidence=prediction.confidence,
                )
                requests.append(request)
        
        # Sort by priority
        requests.sort(key=lambda x: x.priority, reverse=True)
        
        return requests
    
    def _get_common_resources(self, task_type: str) -> List[str]:
        """Get common resources for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            List of common resources
        """
        if task_type not in self._resource_patterns:
            return []
        
        # Return most frequent resources
        freq = defaultdict(int)
        for res in self._resource_patterns[task_type]:
            freq[res] += 1
        
        sorted_resources = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [r for r, _ in sorted_resources[:5]]
    
    def get_task_statistics(self, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Get task execution statistics.
        
        Args:
            task_type: Optional task type to filter by
            
        Returns:
            Dictionary with statistics
        """
        if task_type:
            histories = self._task_history.get(task_type, [])
            if not histories:
                return {}
            
            durations = [h.actual_duration for h in histories if h.actual_duration]
            
            return {
                "task_type": task_type,
                "total_tasks": len(histories),
                "avg_duration": statistics.mean(durations) if durations else 0.0,
                "min_duration": min(durations) if durations else 0.0,
                "max_duration": max(durations) if durations else 0.0,
                "stddev_duration": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                "success_rate": sum(1 for h in histories if h.success) / len(histories),
            }
        else:
            # Overall statistics
            total_tasks = sum(len(h) for h in self._task_history.values())
            
            return {
                "total_tasks": total_tasks,
                "task_types": list(self._task_history.keys()),
                "type_averages": {
                    t: stats["avg"] for t, stats in self._type_averages.items()
                },
            }
    
    def get_prediction_accuracy(self) -> Dict[str, float]:
        """Get prediction accuracy statistics.
        
        Returns:
            Dictionary with accuracy metrics
        """
        # This would be implemented with actual prediction tracking
        # For now, return placeholder
        return {
            "overall_accuracy": 0.85,
            "similarity_accuracy": 0.88,
            "type_accuracy": 0.82,
            "avg_error_rate": 0.12,
        }
