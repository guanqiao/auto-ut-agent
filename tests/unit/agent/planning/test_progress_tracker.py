"""Unit tests for ProgressTracker module."""

import pytest
from datetime import datetime, timedelta
from typing import List

from pyutagent.agent.planning.progress_tracker import (
    ProgressTracker,
    ProgressTrackerConfig,
    TaskProgress,
    ETAPrediction,
    BottleneckInfo,
    BottleneckAnalyzer,
)
from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskStatus


class TestTaskProgress:
    """Test TaskProgress dataclass."""
    
    def test_task_progress_initialization(self):
        """Test TaskProgress initializes with default values."""
        progress = TaskProgress(task_id="task-1")
        
        assert progress.task_id == "task-1"
        assert progress.status == TaskStatus.PENDING
        assert progress.started_at is None
        assert progress.completed_at is None
        assert progress.duration_ms == 0.0
        assert progress.progress_percentage == 0.0
        assert progress.estimated_remaining_ms == 0.0


class TestETAPrediction:
    """Test ETAPrediction dataclass."""
    
    def test_eta_prediction_creation(self):
        """Test creating an ETA prediction."""
        eta = ETAPrediction(
            estimated_completion=datetime.now() + timedelta(minutes=10),
            remaining_tasks=5,
            avg_speed=0.5,
            confidence=0.8,
            confidence_interval=30.0,
            prediction_method="moving_average",
        )
        
        assert eta.remaining_tasks == 5
        assert eta.avg_speed == 0.5
        assert eta.confidence == 0.8
        assert eta.prediction_method == "moving_average"


class TestBottleneckInfo:
    """Test BottleneckInfo dataclass."""
    
    def test_bottleneck_info_creation(self):
        """Test creating bottleneck info."""
        bottleneck = BottleneckInfo(
            bottleneck_type="task",
            bottleneck_id="task-1",
            severity=0.7,
            description="Task is too slow",
            recommendation="Optimize the task",
        )
        
        assert bottleneck.bottleneck_type == "task"
        assert bottleneck.bottleneck_id == "task-1"
        assert bottleneck.severity == 0.7
        assert bottleneck.description == "Task is too slow"
        assert bottleneck.recommendation == "Optimize the task"
        assert isinstance(bottleneck.detected_at, datetime)


class TestProgressTrackerConfig:
    """Test ProgressTrackerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProgressTrackerConfig()
        
        assert config.enable_eta_prediction is True
        assert config.enable_bottleneck_detection is True
        assert config.eta_update_interval == 5.0
        assert config.bottleneck_threshold == 0.8
        assert config.slow_task_threshold == 30.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ProgressTrackerConfig(
            enable_eta_prediction=False,
            enable_bottleneck_detection=False,
            slow_task_threshold=60.0,
        )
        
        assert config.enable_eta_prediction is False
        assert config.enable_bottleneck_detection is False
        assert config.slow_task_threshold == 60.0


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initializes correctly."""
        tracker = ProgressTracker()
        
        assert tracker.config is not None
        assert tracker._task_progress == {}
        assert tracker._completion_times == []
        assert tracker._total_tasks == 0
        assert tracker._completed_tasks == 0
    
    def test_initialize_with_tasks(self):
        """Test initialization with task list."""
        tracker = ProgressTracker()
        tasks = [
            PriorityTask(id="task-1", description="Test task 1"),
            PriorityTask(id="task-2", description="Test task 2"),
            PriorityTask(id="task-3", description="Test task 3"),
        ]
        
        tracker.initialize(tasks)
        
        assert tracker._total_tasks == 3
        assert tracker._start_time is not None
        assert len(tracker._task_progress) == 3
        assert "task-1" in tracker._task_progress
    
    def test_start_task(self):
        """Test starting a task."""
        tracker = ProgressTracker()
        task = PriorityTask(id="task-1", description="Test task")
        tracker.initialize([task])
        
        tracker.start_task("task-1")
        
        progress = tracker.get_task_progress("task-1")
        assert progress is not None
        assert progress.status == TaskStatus.RUNNING
        assert progress.started_at is not None
        assert progress.progress_percentage == 0.0
    
    def test_start_task_auto_init(self):
        """Test that start_task auto-initializes task."""
        tracker = ProgressTracker()
        tracker.start_task("task-1")
        
        progress = tracker.get_task_progress("task-1")
        assert progress is not None
        assert progress.status == TaskStatus.RUNNING
    
    def test_update_progress(self):
        """Test updating task progress."""
        tracker = ProgressTracker()
        tracker.start_task("task-1")
        
        tracker.update_progress("task-1", 50.0)
        
        progress = tracker.get_task_progress("task-1")
        assert progress.progress_percentage == 50.0
    
    def test_update_progress_bounds(self):
        """Test progress percentage bounds."""
        tracker = ProgressTracker()
        tracker.start_task("task-1")
        
        tracker.update_progress("task-1", -10.0)
        progress = tracker.get_task_progress("task-1")
        assert progress.progress_percentage == 0.0
        
        tracker.update_progress("task-1", 150.0)
        progress = tracker.get_task_progress("task-1")
        assert progress.progress_percentage == 100.0
    
    def test_complete_task_success(self):
        """Test successful task completion."""
        tracker = ProgressTracker()
        tracker.start_task("task-1")
        
        tracker.complete_task("task-1", duration_ms=150.0, success=True)
        
        progress = tracker.get_task_progress("task-1")
        assert progress.status == TaskStatus.COMPLETED
        assert progress.completed_at is not None
        assert progress.duration_ms == 150.0
        assert progress.progress_percentage == 100.0
        assert tracker._completed_tasks == 1
    
    def test_complete_task_failure(self):
        """Test failed task completion."""
        tracker = ProgressTracker()
        tracker.start_task("task-1")
        
        tracker.complete_task("task-1", duration_ms=150.0, success=False)
        
        progress = tracker.get_task_progress("task-1")
        assert progress.status == TaskStatus.FAILED
        assert tracker._failed_tasks == 1
    
    def test_complete_task_auto_init(self):
        """Test that complete_task auto-initializes task."""
        tracker = ProgressTracker()
        tracker.complete_task("task-1", duration_ms=100.0, success=True)
        
        progress = tracker.get_task_progress("task-1")
        assert progress is not None
        assert tracker._completed_tasks == 1
    
    def test_get_overall_progress(self):
        """Test overall progress calculation."""
        tracker = ProgressTracker()
        tasks = [
            PriorityTask(id="task-1", description="Test task 1"),
            PriorityTask(id="task-2", description="Test task 2"),
            PriorityTask(id="task-3", description="Test task 3"),
            PriorityTask(id="task-4", description="Test task 4"),
        ]
        tracker.initialize(tasks)
        
        assert tracker.get_overall_progress() == 0.0
        
        tracker.complete_task("task-1", 100.0, success=True)
        tracker.complete_task("task-2", 100.0, success=True)
        
        assert tracker.get_overall_progress() == 50.0
        
        tracker.complete_task("task-3", 100.0, success=False)
        tracker.complete_task("task-4", 100.0, success=True)
        
        assert tracker.get_overall_progress() == 100.0
    
    def test_get_overall_progress_zero_tasks(self):
        """Test progress with no tasks."""
        tracker = ProgressTracker()
        assert tracker.get_overall_progress() == 0.0
    
    def test_predict_eta(self):
        """Test ETA prediction."""
        tracker = ProgressTracker()
        
        tasks = [PriorityTask(id=f"task-{i}", description=f"Test task {i}") for i in range(10)]
        tracker.initialize(tasks)
        
        for i in range(5):
            tracker.complete_task(f"task-{i}", duration_ms=100.0, success=True)
        
        eta = tracker.predict_eta()
        
        assert eta is not None
        assert eta.remaining_tasks == 5
        assert eta.avg_speed > 0
        assert eta.confidence > 0
        assert eta.prediction_method == "moving_average"
    
    def test_predict_eta_completed(self):
        """Test ETA prediction when all tasks completed."""
        tracker = ProgressTracker()
        tasks = [PriorityTask(id="task-1", description="Test task")]
        tracker.initialize(tasks)
        tracker.complete_task("task-1", 100.0, success=True)
        
        eta = tracker.predict_eta()
        
        assert eta is not None
        assert eta.remaining_tasks == 0
        assert eta.confidence == 1.0
        assert eta.prediction_method == "completed"
    
    def test_predict_eta_insufficient_data(self):
        """Test ETA prediction with insufficient data."""
        tracker = ProgressTracker()
        tasks = [PriorityTask(id=f"task-{i}", description=f"Test task {i}") for i in range(10)]
        tracker.initialize(tasks)
        
        tracker.complete_task("task-1", 100.0, success=True)
        tracker.complete_task("task-2", 100.0, success=True)
        
        eta = tracker.predict_eta()
        
        assert eta is None
    
    def test_predict_eta_disabled(self):
        """Test ETA prediction when disabled."""
        config = ProgressTrackerConfig(enable_eta_prediction=False)
        tracker = ProgressTracker(config=config)
        
        tasks = [PriorityTask(id=f"task-{i}", description=f"Test task {i}") for i in range(10)]
        tracker.initialize(tasks)
        
        for i in range(5):
            tracker.complete_task(f"task-{i}", 100.0, success=True)
        
        eta = tracker.predict_eta()
        
        assert eta is None
    
    def test_get_bottlenecks(self):
        """Test getting detected bottlenecks."""
        tracker = ProgressTracker()
        tracker.complete_task("task-1", duration_ms=100.0, success=True)
        
        bottlenecks = tracker.get_bottlenecks()
        assert isinstance(bottlenecks, list)
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        tracker = ProgressTracker()
        
        metrics = tracker.get_performance_metrics()
        assert metrics == {}
        
        tasks = [
            PriorityTask(id="task-1", description="Test task 1"),
            PriorityTask(id="task-2", description="Test task 2"),
            PriorityTask(id="task-3", description="Test task 3"),
        ]
        tracker.initialize(tasks)
        
        tracker.complete_task("task-1", duration_ms=100.0, success=True)
        tracker.complete_task("task-2", duration_ms=150.0, success=True)
        tracker.complete_task("task-3", duration_ms=200.0, success=True)
        
        metrics = tracker.get_performance_metrics()
        
        assert metrics["total_tasks"] == 3
        assert metrics["completed_tasks"] == 3
        assert metrics["failed_tasks"] == 0
        assert metrics["progress_percentage"] == 100.0
        assert "avg_duration_ms" in metrics
        assert "min_duration_ms" in metrics
        assert "max_duration_ms" in metrics
        assert "throughput_tasks_per_second" in metrics
    
    def test_get_status_summary(self):
        """Test getting status summary."""
        tracker = ProgressTracker()
        tasks = [
            PriorityTask(id="task-1", description="Test task 1"),
            PriorityTask(id="task-2", description="Test task 2"),
            PriorityTask(id="task-3", description="Test task 3"),
        ]
        tracker.initialize(tasks)
        
        tracker.complete_task("task-1", 100.0, success=True)
        tracker.start_task("task-2")
        
        summary = tracker.get_status_summary()
        
        assert summary["total_tasks"] == 3
        assert summary["completed"] == 1
        assert summary["failed"] == 0
        assert summary["in_progress"] == 1
        assert summary["pending"] == 1
        assert "progress_percentage" in summary
        assert "bottleneck_count" in summary
    
    def test_reset(self):
        """Test resetting the tracker."""
        tracker = ProgressTracker()
        tasks = [PriorityTask(id=f"task-{i}", description=f"Test task {i}") for i in range(5)]
        tracker.initialize(tasks)
        
        tracker.complete_task("task-1", 100.0, success=True)
        tracker.start_task("task-2")
        
        tracker.reset()
        
        assert tracker._task_progress == {}
        assert tracker._completion_times == []
        assert tracker._total_tasks == 0
        assert tracker._completed_tasks == 0
        assert tracker._bottlenecks == []


class TestBottleneckDetection:
    """Test bottleneck detection functionality."""
    
    def test_slow_task_detection(self):
        """Test detection of slow tasks."""
        config = ProgressTrackerConfig(
            enable_bottleneck_detection=True,
            slow_task_threshold=0.1,
        )
        tracker = ProgressTracker(config=config)
        
        tracker.complete_task("task-1", duration_ms=500.0, success=True)
        
        bottlenecks = tracker.get_bottlenecks()
        
        assert len(bottlenecks) > 0
        assert bottlenecks[0].bottleneck_type == "task"
        assert bottlenecks[0].bottleneck_id == "task-1"
        assert bottlenecks[0].severity > 0
    
    def test_bottleneck_with_recommendation(self):
        """Test that bottlenecks include recommendations."""
        config = ProgressTrackerConfig(
            enable_bottleneck_detection=True,
            slow_task_threshold=0.1,
        )
        tracker = ProgressTracker(config=config)
        
        tracker.complete_task("slow-task", duration_ms=1000.0, success=True)
        
        bottlenecks = tracker.get_bottlenecks()
        
        assert len(bottlenecks) > 0
        assert bottlenecks[0].recommendation is not None
        assert len(bottlenecks[0].recommendation) > 0


class TestBottleneckAnalyzer:
    """Test BottleneckAnalyzer class."""
    
    def test_bottleneck_analyzer_initialization(self):
        """Test BottleneckAnalyzer initializes correctly."""
        analyzer = BottleneckAnalyzer()
        
        assert analyzer.tracker is not None
        assert analyzer._task_dependencies == {}
    
    def test_set_dependencies(self):
        """Test setting task dependencies."""
        analyzer = BottleneckAnalyzer()
        
        analyzer.set_dependencies("task-3", {"task-1", "task-2"})
        
        assert "task-3" in analyzer._task_dependencies
        assert "task-1" in analyzer._task_dependencies["task-3"]
        assert "task-2" in analyzer._task_dependencies["task-3"]
    
    def test_analyze_bottlenecks(self):
        """Test bottleneck analysis."""
        analyzer = BottleneckAnalyzer()
        
        analyzer.tracker.start_task("task-1")
        analyzer.tracker.start_task("task-2")
        
        analyzer.set_dependencies("task-3", {"task-1", "task-2"})
        
        bottlenecks = analyzer.analyze_bottlenecks()
        
        dependency_bottlenecks = [b for b in bottlenecks if b.bottleneck_type == "dependency"]
        assert len(dependency_bottlenecks) >= 0
    
    def test_get_recommendations(self):
        """Test getting recommendations."""
        analyzer = BottleneckAnalyzer()
        
        config = ProgressTrackerConfig(slow_task_threshold=0.1)
        analyzer.tracker = ProgressTracker(config=config)
        
        for i in range(5):
            analyzer.tracker.complete_task(f"task-{i}", duration_ms=500.0, success=True)
        
        recommendations = analyzer.get_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_get_recommendations_no_bottlenecks(self):
        """Test recommendations with no bottlenecks."""
        analyzer = BottleneckAnalyzer()
        
        for i in range(3):
            analyzer.tracker.complete_task(f"task-{i}", duration_ms=50.0, success=True)
        
        recommendations = analyzer.get_recommendations()
        
        assert isinstance(recommendations, list)


class TestProgressTrackerIntegration:
    """Integration tests for ProgressTracker."""
    
    def test_full_task_lifecycle(self):
        """Test complete task lifecycle tracking."""
        tracker = ProgressTracker()
        tasks = [PriorityTask(id=f"task-{i}", description=f"Test task {i}") for i in range(5)]
        tracker.initialize(tasks)
        
        for i in range(5):
            tracker.start_task(f"task-{i}")
            tracker.update_progress(f"task-{i}", 50.0)
            tracker.complete_task(f"task-{i}", duration_ms=100.0 + i * 10, success=True)
        
        assert tracker.get_overall_progress() == 100.0
        
        metrics = tracker.get_performance_metrics()
        assert metrics["completed_tasks"] == 5
        
        eta = tracker.predict_eta()
        assert eta is not None
        assert eta.remaining_tasks == 0
    
    def test_mixed_success_failure(self):
        """Test tracking with mixed success and failure."""
        tracker = ProgressTracker()
        tasks = [PriorityTask(id=f"task-{i}", description=f"Test task {i}") for i in range(6)]
        tracker.initialize(tasks)
        
        for i in range(6):
            tracker.complete_task(f"task-{i}", duration_ms=100.0, success=(i % 2 == 0))
        
        assert tracker._completed_tasks == 3
        assert tracker._failed_tasks == 3
        assert tracker.get_overall_progress() == 100.0
        
        metrics = tracker.get_performance_metrics()
        assert metrics["completed_tasks"] == 3
        assert metrics["failed_tasks"] == 3
    
    def test_eta_accuracy(self):
        """Test ETA prediction accuracy with consistent task times."""
        tracker = ProgressTracker()
        tasks = [PriorityTask(id=f"task-{i}", description=f"Test task {i}") for i in range(20)]
        tracker.initialize(tasks)
        
        consistent_time = 100.0
        for i in range(15):
            tracker.complete_task(f"task-{i}", duration_ms=consistent_time, success=True)
        
        eta = tracker.predict_eta()
        
        assert eta is not None
        assert eta.remaining_tasks == 5
        
        expected_remaining_seconds = (5 * consistent_time) / 1000.0
        actual_eta_seconds = (eta.estimated_completion - datetime.now()).total_seconds()
        
        tolerance = expected_remaining_seconds * 0.3
        assert abs(actual_eta_seconds - expected_remaining_seconds) < tolerance
