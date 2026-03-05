"""Unit tests for Phase 4 - Advanced Features."""

import pytest
from datetime import datetime, timedelta
from typing import List

from pyutagent.agent.planning.predictive_scheduler import (
    PredictiveScheduler,
    SchedulerConfig,
    ExecutionPrediction,
    TaskHistory,
    PrefetchRequest,
)
from pyutagent.agent.planning.fair_scheduler import (
    FairScheduler,
    QueueConfig,
    QueueLevel,
    TaskQueueEntry,
    FairnessMetrics,
)
from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskStatus


class TestPredictiveScheduler:
    """Test PredictiveScheduler class."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly."""
        scheduler = PredictiveScheduler()
        
        assert scheduler.config is not None
        assert scheduler._task_history == {}
        assert scheduler._type_averages == {}
    
    def test_record_task_execution(self):
        """Test recording task execution."""
        scheduler = PredictiveScheduler()
        task = PriorityTask(id="task-1", description="Test task")
        
        scheduler.record_task_execution(task, duration=5.0, success=True)
        
        assert "test" in scheduler._task_history or "general" in scheduler._task_history
        
        # Check type averages updated
        assert len(scheduler._type_averages) > 0
    
    def test_record_multiple_executions(self):
        """Test recording multiple executions."""
        scheduler = PredictiveScheduler()
        
        for i in range(10):
            task = PriorityTask(id=f"task-{i}", description="Test task")
            scheduler.record_task_execution(task, duration=5.0 + i * 0.1, success=True)
        
        # Should have history
        assert len(scheduler._task_history) > 0
        
        # Check type averages
        for task_type, stats in scheduler._type_averages.items():
            assert stats["count"] > 0
            assert stats["avg"] > 0
    
    def test_predict_execution_time_no_history(self):
        """Test prediction with no history."""
        scheduler = PredictiveScheduler()
        task = PriorityTask(id="task-1", description="New task type xyz")
        
        prediction = scheduler.predict_execution_time(task)
        
        assert prediction.predicted_duration == 0.0
        assert prediction.confidence == 0.0
        assert prediction.prediction_method == "default"
    
    def test_predict_execution_time_with_history(self):
        """Test prediction with historical data."""
        scheduler = PredictiveScheduler()
        
        # Record some executions
        for i in range(5):
            task = PriorityTask(id=f"task-{i}", description="Test task execution")
            scheduler.record_task_execution(task, duration=10.0, success=True)
        
        # Predict for similar task
        new_task = PriorityTask(id="new-task", description="Test task execution")
        prediction = scheduler.predict_execution_time(new_task)
        
        assert prediction.predicted_duration > 0
        assert prediction.similar_tasks > 0 or prediction.prediction_method == "type_based"
    
    def test_predict_with_similarity(self):
        """Test similarity-based prediction."""
        scheduler = SchedulerConfig(similarity_threshold=0.5)
        predictive = PredictiveScheduler(config=scheduler)
        
        # Record similar tasks
        for i in range(3):
            task = PriorityTask(
                id=f"task-{i}",
                description="Build project with compilation"
            )
            predictive.record_task_execution(task, duration=15.0 + i, success=True)
        
        # Predict for similar task
        new_task = PriorityTask(
            id="new-task",
            description="Build project with compilation"
        )
        prediction = predictive.predict_execution_time(new_task)
        
        assert prediction.predicted_duration > 0
        assert prediction.confidence > 0
    
    def test_extract_task_type(self):
        """Test task type extraction."""
        scheduler = PredictiveScheduler()
        
        test_cases = [
            ("Run unit tests", "test"),
            ("Build the project", "build"),
            ("Analyze code quality", "analysis"),
            ("Generate report", "generation"),
            ("Search for files", "search"),
            ("Random task", "general"),
        ]
        
        for description, expected_type in test_cases:
            task = PriorityTask(id="task", description=description)
            task_type = scheduler._extract_task_type(task)
            assert task_type == expected_type, f"Failed for {description}"
    
    def test_generate_prefetch_requests(self):
        """Test prefetch request generation."""
        config = SchedulerConfig(enable_prefetching=True, min_confidence=0.0)
        scheduler = PredictiveScheduler(config=config)
        
        # Record some executions to learn patterns
        for i in range(5):
            task = PriorityTask(id=f"task-{i}", description="Test task")
            scheduler.record_task_execution(
                task,
                duration=5.0,
                success=True,
                resource_usage={"cpu": 0.5, "memory": 100},
            )
        
        # Generate prefetch requests
        upcoming = [PriorityTask(id="new-task", description="Test task")]
        requests = scheduler.generate_prefetch_requests(upcoming)
        
        assert isinstance(requests, list)
    
    def test_get_task_statistics(self):
        """Test getting task statistics."""
        scheduler = PredictiveScheduler()
        
        # Record executions
        for i in range(10):
            task = PriorityTask(id=f"task-{i}", description="Test task")
            scheduler.record_task_execution(task, duration=5.0 + i, success=(i % 2 == 0))
        
        stats = scheduler.get_task_statistics()
        
        assert stats["total_tasks"] == 10
        assert "task_types" in stats
    
    def test_get_prediction_accuracy(self):
        """Test getting prediction accuracy."""
        scheduler = PredictiveScheduler()
        
        accuracy = scheduler.get_prediction_accuracy()
        
        assert "overall_accuracy" in accuracy
        assert "avg_error_rate" in accuracy


class TestFairScheduler:
    """Test FairScheduler class."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly."""
        scheduler = FairScheduler()
        
        assert scheduler.config is not None
        assert len(scheduler._queues) == 3
        assert scheduler._task_entries == {}
    
    def test_add_task(self):
        """Test adding a task."""
        scheduler = FairScheduler()
        task = PriorityTask(id="task-1", description="Test task")
        
        scheduler.add_task(task)
        
        assert task.id in scheduler._task_entries
        assert len(scheduler._queues[QueueLevel.HIGH]) == 1
    
    def test_add_duplicate_task(self):
        """Test adding duplicate task."""
        scheduler = FairScheduler()
        task = PriorityTask(id="task-1", description="Test task")
        
        scheduler.add_task(task)
        scheduler.add_task(task)  # Should be ignored
        
        assert len(scheduler._queues[QueueLevel.HIGH]) == 1
    
    def test_get_next_task(self):
        """Test getting next task."""
        scheduler = FairScheduler()
        task1 = PriorityTask(id="task-1", description="Test task 1")
        task2 = PriorityTask(id="task-2", description="Test task 2")
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        
        next_task = scheduler.get_next_task()
        
        assert next_task is not None
        assert next_task.id in ["task-1", "task-2"]
    
    def test_task_yielded(self):
        """Test task yield."""
        scheduler = FairScheduler()
        task = PriorityTask(id="task-1", description="Test task")
        
        scheduler.add_task(task)
        scheduler.get_next_task()
        scheduler.task_yielded("task-1", completed=False)
        
        # Task should be re-queued
        assert task.id in scheduler._task_entries
    
    def test_task_completed(self):
        """Test task completion."""
        scheduler = FairScheduler()
        task = PriorityTask(id="task-1", description="Test task")
        
        scheduler.add_task(task)
        scheduler.get_next_task()
        scheduler.task_completed("task-1")
        
        # Task should be removed
        assert task.id not in scheduler._task_entries
    
    def test_multi_level_queues(self):
        """Test multi-level queue system."""
        config = QueueConfig(time_slice_ms={
            QueueLevel.HIGH: 10,
            QueueLevel.MEDIUM: 20,
            QueueLevel.LOW: 40,
        })
        scheduler = FairScheduler(config=config)
        
        # Add task
        task = PriorityTask(id="task-1", description="Test task")
        scheduler.add_task(task)
        
        # Get and yield multiple times to demote
        for _ in range(50):
            next_task = scheduler.get_next_task()
            if next_task:
                scheduler.task_yielded(next_task.id, completed=False)
        
        # Task should eventually be demoted or completed
        # (time slice reduces each yield)
        if task.id in scheduler._task_entries:
            entry = scheduler._task_entries[task.id]
            # Either demoted or time slice is minimal
            assert entry.queue_level in [QueueLevel.MEDIUM, QueueLevel.LOW] or \
                   entry.time_slice_remaining <= 10
    
    def test_aging_mechanism(self):
        """Test priority aging."""
        config = QueueConfig(aging_interval=0.1)  # Short interval for testing
        scheduler = FairScheduler(config=config)
        
        # Add task
        task = PriorityTask(id="task-1", description="Test task")
        scheduler.add_task(task)
        
        # Wait for aging
        import time
        time.sleep(0.2)
        
        # Trigger aging by getting task
        scheduler.get_next_task()
        
        # Task should still be in scheduler
        assert task.id in scheduler._task_entries
    
    def test_starvation_prevention(self):
        """Test starvation prevention."""
        config = QueueConfig(starvation_threshold=0.5)
        scheduler = FairScheduler(config=config)
        
        # Add task
        task = PriorityTask(id="task-1", description="Test task")
        scheduler.add_task(task)
        
        # Wait for starvation threshold
        import time
        time.sleep(0.6)
        
        # Get next task - should boost priority
        next_task = scheduler.get_next_task()
        
        assert next_task is not None
    
    def test_get_queue_stats(self):
        """Test getting queue statistics."""
        scheduler = FairScheduler()
        
        task1 = PriorityTask(id="task-1", description="Test task 1")
        task2 = PriorityTask(id="task-2", description="Test task 2")
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        
        stats = scheduler.get_queue_stats()
        
        assert "HIGH" in stats
        assert stats["HIGH"]["size"] == 2
    
    def test_get_fairness_metrics(self):
        """Test getting fairness metrics."""
        scheduler = FairScheduler()
        
        # Add and complete some tasks
        for i in range(5):
            task = PriorityTask(id=f"task-{i}", description=f"Task {i}")
            scheduler.add_task(task)
            
            next_task = scheduler.get_next_task()
            if next_task:
                scheduler.task_completed(next_task.id)
        
        metrics = scheduler.get_fairness_metrics()
        
        assert isinstance(metrics, FairnessMetrics)
        assert 0.0 <= metrics.jain_index <= 1.0
        assert metrics.avg_wait_time >= 0
    
    def test_get_task_wait_time(self):
        """Test getting task wait time."""
        scheduler = FairScheduler()
        task = PriorityTask(id="task-1", description="Test task")
        
        scheduler.add_task(task)
        
        wait_time = scheduler.get_task_wait_time("task-1")
        
        assert wait_time is not None
        assert wait_time >= 0
    
    def test_get_task_wait_time_not_found(self):
        """Test getting wait time for non-existent task."""
        scheduler = FairScheduler()
        
        wait_time = scheduler.get_task_wait_time("non-existent")
        
        assert wait_time is None
    
    def test_clear_queues(self):
        """Test clearing all queues."""
        scheduler = FairScheduler()
        
        for i in range(5):
            task = PriorityTask(id=f"task-{i}", description=f"Task {i}")
            scheduler.add_task(task)
        
        scheduler.clear()
        
        assert all(len(q) == 0 for q in scheduler._queues.values())
        assert len(scheduler._task_entries) == 0
    
    def test_queue_size_limit(self):
        """Test queue size limit."""
        config = QueueConfig(max_queue_size=2)
        scheduler = FairScheduler(config=config)
        
        for i in range(5):
            task = PriorityTask(id=f"task-{i}", description=f"Task {i}")
            scheduler.add_task(task)
        
        # Only 2 tasks should be added
        assert len(scheduler._queues[QueueLevel.HIGH]) == 2
    
    def test_round_robin_scheduling(self):
        """Test round-robin behavior."""
        scheduler = FairScheduler()
        
        task1 = PriorityTask(id="task-1", description="Test task 1")
        task2 = PriorityTask(id="task-2", description="Test task 2")
        
        scheduler.add_task(task1)
        scheduler.add_task(task2)
        
        # Get tasks multiple times
        tasks_executed = []
        for _ in range(4):
            next_task = scheduler.get_next_task()
            if next_task:
                tasks_executed.append(next_task.id)
                scheduler.task_yielded(next_task.id, completed=False)
        
        # Both tasks should get executed
        assert "task-1" in tasks_executed
        assert "task-2" in tasks_executed


class TestIntegration:
    """Integration tests for Phase 4 features."""
    
    def test_predictive_fair_scheduling(self):
        """Test predictive scheduler integrated with fair scheduler."""
        predictive = PredictiveScheduler()
        fair = FairScheduler()
        
        # Record some executions
        for i in range(5):
            task = PriorityTask(id=f"task-{i}", description="Test task")
            predictive.record_task_execution(task, duration=5.0, success=True)
            fair.add_task(task)
        
        # Get predictions and schedule
        while True:
            next_task = fair.get_next_task()
            if not next_task:
                break
            
            prediction = predictive.predict_execution_time(next_task)
            assert prediction is not None
            
            fair.task_completed(next_task.id)
    
    def test_full_task_lifecycle(self):
        """Test complete task lifecycle."""
        scheduler = FairScheduler()
        predictive = PredictiveScheduler()
        
        # Create and add task
        task = PriorityTask(id="task-1", description="Test task")
        scheduler.add_task(task)
        
        # Execute
        next_task = scheduler.get_next_task()
        assert next_task is not None
        
        # Record execution
        start_time = datetime.now()
        # Simulate execution
        predictive.record_task_execution(next_task, duration=0.1, success=True)
        
        # Complete
        scheduler.task_completed("task-1")
        
        # Verify
        assert "task-1" not in scheduler._task_entries


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_scheduler_prediction(self):
        """Test prediction with empty scheduler."""
        scheduler = PredictiveScheduler()
        task = PriorityTask(id="task-1", description="Test")
        
        prediction = scheduler.predict_execution_time(task)
        
        assert prediction.prediction_method == "default"
    
    def test_empty_fair_scheduler(self):
        """Test fair scheduler with no tasks."""
        scheduler = FairScheduler()
        
        next_task = scheduler.get_next_task()
        
        assert next_task is None
    
    def test_zero_duration_prediction(self):
        """Test prediction with zero duration history."""
        scheduler = PredictiveScheduler()
        
        task = PriorityTask(id="task-1", description="Test")
        scheduler.record_task_execution(task, duration=0.0, success=True)
        
        prediction = scheduler.predict_execution_time(task)
        
        assert prediction is not None
    
    def test_very_long_task_description(self):
        """Test with very long task description."""
        scheduler = PredictiveScheduler()
        
        long_desc = "Test " * 100
        task = PriorityTask(id="task-1", description=long_desc)
        
        scheduler.record_task_execution(task, duration=5.0, success=True)
        
        prediction = scheduler.predict_execution_time(task)
        
        assert prediction is not None
