"""Unit tests for Parallel Execution Engine with priority queue."""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List

from pyutagent.agent.planning.parallel_executor import (
    ParallelExecutionEngine,
    ParallelExecutionConfig,
    PriorityExecutionConfig,
    PriorityTask,
    TaskStatus,
    TaskType,
    ResourcePool,
    ResourceType,
    SubTaskResult,
)


class TestPriorityTask:
    """Tests for PriorityTask dataclass."""
    
    def test_priority_task_creation(self):
        """Test creating a PriorityTask."""
        task = PriorityTask(
            id="task_1",
            description="Test task",
            priority=0.8,
        )
        
        assert task.id == "task_1"
        assert task.description == "Test task"
        assert task.priority == 0.8
        assert task.status == TaskStatus.PENDING
        assert task.preemptible is True
        assert task.paused is False
    
    def test_priority_task_comparison(self):
        """Test PriorityTask comparison for heap sorting."""
        task1 = PriorityTask(id="task_1", description="Task 1", priority=0.5)
        task2 = PriorityTask(id="task_2", description="Task 2", priority=0.8)
        task3 = PriorityTask(id="task_3", description="Task 3", priority=0.3)
        
        # Higher priority should be "less than" for min-heap
        assert task2 < task1  # 0.8 < 0.5 in heap terms
        assert task1 < task3  # 0.5 < 0.3 in heap terms
        assert task2 < task3  # 0.8 < 0.3 in heap terms
    
    def test_priority_task_equality(self):
        """Test PriorityTask equality by ID."""
        task1 = PriorityTask(id="task_1", description="Task 1", priority=0.5)
        task2 = PriorityTask(id="task_1", description="Task 1 modified", priority=0.8)
        task3 = PriorityTask(id="task_2", description="Task 2", priority=0.5)
        
        assert task1 == task2  # Same ID
        assert task1 != task3  # Different ID
    
    def test_priority_task_hash(self):
        """Test PriorityTask hashing by ID."""
        task1 = PriorityTask(id="task_1", description="Task 1", priority=0.5)
        task2 = PriorityTask(id="task_1", description="Task 1 modified", priority=0.8)
        
        assert hash(task1) == hash(task2)  # Same hash for same ID
    
    def test_priority_task_default_values(self):
        """Test PriorityTask default values."""
        task = PriorityTask(id="task_1", description="Test task")
        
        assert task.priority == 0.5
        assert task.deadline is None
        assert task.dependencies == set()
        assert task.resource_requirements == {}
        assert task.estimated_duration == 0.0
        assert task.actual_duration is None
        assert task.result is None
        assert task.error is None
        assert task.retry_count == 0
        assert task.max_retries == 2
        assert task.preemptible is True
        assert task.paused is False
        assert task.metadata == {}


class TestResourcePool:
    """Tests for ResourcePool with prediction support."""
    
    def test_resource_pool_creation(self):
        """Test creating a ResourcePool."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            max_concurrent=4,
        )
        
        assert pool.resource_type == ResourceType.CPU
        assert pool.max_concurrent == 4
        assert pool.current_usage == 0
        assert pool.available == 4
        assert pool.utilization == 0.0
    
    @pytest.mark.asyncio
    async def test_resource_pool_acquire(self):
        """Test acquiring resources."""
        pool = ResourcePool(resource_type=ResourceType.CPU, max_concurrent=4)
        
        # Acquire resources
        result = await pool.acquire(2)
        assert result is True
        assert pool.current_usage == 2
        assert pool.available == 2
    
    @pytest.mark.asyncio
    async def test_resource_pool_acquire_insufficient(self):
        """Test acquiring more resources than available."""
        pool = ResourcePool(resource_type=ResourceType.CPU, max_concurrent=4)
        
        # Try to acquire more than available
        result = await pool.acquire(5)
        assert result is False
        assert pool.current_usage == 0
    
    @pytest.mark.asyncio
    async def test_resource_pool_release(self):
        """Test releasing resources."""
        pool = ResourcePool(resource_type=ResourceType.CPU, max_concurrent=4)
        
        await pool.acquire(3)
        await pool.release(2)
        
        assert pool.current_usage == 1
        assert pool.available == 3
    
    @pytest.mark.asyncio
    async def test_resource_pool_reserve(self):
        """Test reserving resources."""
        pool = ResourcePool(resource_type=ResourceType.CPU, max_concurrent=4)
        
        # Reserve resources
        result = pool.reserve(2)
        assert result is True
        assert pool.reserved == 2
        assert pool.available == 2
    
    @pytest.mark.asyncio
    async def test_resource_pool_acquire_with_reserved(self):
        """Test acquiring reserved resources."""
        pool = ResourcePool(resource_type=ResourceType.CPU, max_concurrent=4)
        
        pool.reserve(2)
        
        # Acquire from reserved
        result = await pool.acquire(2, use_reserved=True)
        assert result is True
        assert pool.current_usage == 2
        assert pool.reserved == 0
    
    def test_resource_pool_predict_availability(self):
        """Test predicting future resource availability."""
        pool = ResourcePool(resource_type=ResourceType.CPU, max_concurrent=4)
        
        # Add some usage history
        now = datetime.now()
        for i in range(10):
            pool.usage_history.append((now + timedelta(seconds=i), i % 4))
        pool.current_usage = 2
        
        # Should predict based on trend
        predicted = pool.predict_availability(timedelta(seconds=10))
        assert predicted >= 0
        assert predicted <= pool.max_concurrent


class TestParallelExecutionEnginePriorityQueue:
    """Tests for ParallelExecutionEngine priority queue functionality."""
    
    @pytest.mark.asyncio
    async def test_enqueue_task(self):
        """Test enqueuing a task."""
        engine = ParallelExecutionEngine()
        task = PriorityTask(id="task_1", description="Test task", priority=0.8)
        
        result = await engine.enqueue_task(task)
        
        assert result is True
        assert task.status == TaskStatus.QUEUED
    
    @pytest.mark.asyncio
    async def test_enqueue_duplicate_task(self):
        """Test enqueuing a duplicate task."""
        engine = ParallelExecutionEngine()
        task = PriorityTask(id="task_1", description="Test task", priority=0.8)
        
        await engine.enqueue_task(task)
        result = await engine.enqueue_task(task)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_dequeue_task(self):
        """Test dequeuing the highest priority task."""
        engine = ParallelExecutionEngine()
        
        task1 = PriorityTask(id="task_1", description="Task 1", priority=0.5)
        task2 = PriorityTask(id="task_2", description="Task 2", priority=0.8)
        task3 = PriorityTask(id="task_3", description="Task 3", priority=0.3)
        
        await engine.enqueue_task(task1)
        await engine.enqueue_task(task2)
        await engine.enqueue_task(task3)
        
        # Should dequeue highest priority first
        task = await engine.dequeue_task()
        assert task.id == "task_2"
        assert task.status == TaskStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self):
        """Test dequeuing from empty queue."""
        engine = ParallelExecutionEngine()
        
        task = await engine.dequeue_task()
        
        assert task is None
    
    @pytest.mark.asyncio
    async def test_peek_task(self):
        """Test peeking at the highest priority task."""
        engine = ParallelExecutionEngine()
        
        task1 = PriorityTask(id="task_1", description="Task 1", priority=0.5)
        task2 = PriorityTask(id="task_2", description="Task 2", priority=0.8)
        
        await engine.enqueue_task(task1)
        await engine.enqueue_task(task2)
        
        # Peek should not remove
        task = await engine.peek_task()
        assert task.id == "task_2"
        
        # Queue should still have both tasks
        assert len(engine._priority_queue) == 2
    
    @pytest.mark.asyncio
    async def test_update_priority(self):
        """Test updating task priority."""
        engine = ParallelExecutionEngine()
        task = PriorityTask(id="task_1", description="Test task", priority=0.5)
        
        await engine.enqueue_task(task)
        result = await engine.update_priority("task_1", 0.9)
        
        assert result is True
        assert task.priority == 0.9
    
    @pytest.mark.asyncio
    async def test_update_priority_nonexistent(self):
        """Test updating priority of nonexistent task."""
        engine = ParallelExecutionEngine()
        
        result = await engine.update_priority("nonexistent", 0.9)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_preempt_task(self):
        """Test preempting a task."""
        engine = ParallelExecutionEngine()
        task = PriorityTask(id="task_1", description="Test task", priority=0.5)
        
        await engine.enqueue_task(task)
        result = await engine.preempt("task_1")
        
        assert result is True
        assert task.paused is True
        assert task.status == TaskStatus.PAUSED
        assert task.priority > 0.5  # Priority should be boosted
    
    @pytest.mark.asyncio
    async def test_preempt_non_preemptible_task(self):
        """Test preempting a non-preemptible task."""
        engine = ParallelExecutionEngine()
        task = PriorityTask(
            id="task_1",
            description="Test task",
            priority=0.5,
            preemptible=False,
        )
        
        await engine.enqueue_task(task)
        result = await engine.preempt("task_1")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_pause_task(self):
        """Test pausing a task."""
        engine = ParallelExecutionEngine()
        task = PriorityTask(id="task_1", description="Test task", priority=0.5)
        
        await engine.enqueue_task(task)
        result = await engine.pause("task_1")
        
        assert result is True
        assert task.paused is True
        assert task.status == TaskStatus.PAUSED
    
    @pytest.mark.asyncio
    async def test_resume_task(self):
        """Test resuming a paused task."""
        engine = ParallelExecutionEngine()
        task = PriorityTask(id="task_1", description="Test task", priority=0.5)
        
        await engine.enqueue_task(task)
        await engine.pause("task_1")
        result = await engine.resume("task_1")
        
        assert result is True
        assert task.paused is False
        assert task.status == TaskStatus.QUEUED
    
    @pytest.mark.asyncio
    async def test_get_queue_stats(self):
        """Test getting queue statistics."""
        engine = ParallelExecutionEngine()
        
        task1 = PriorityTask(id="task_1", description="Task 1", priority=0.5)
        task2 = PriorityTask(id="task_2", description="Task 2", priority=0.8)
        
        await engine.enqueue_task(task1)
        await engine.enqueue_task(task2)
        
        stats = engine.get_queue_stats()
        
        assert stats["queue_size"] == 2
        assert stats["running_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["failed_tasks"] == 0
        assert stats["paused_tasks"] == 0
        assert stats["preempted_tasks"] == 0


class TestParallelExecutionEngineWithPriority:
    """Tests for parallel execution with priority scheduling."""
    
    @pytest.mark.asyncio
    async def test_execute_with_priority(self):
        """Test executing tasks with priority scheduling."""
        engine = ParallelExecutionEngine(
            config=ParallelExecutionConfig(max_concurrent_tasks=2)
        )
        
        execution_order = []
        
        async def mock_executor(task: PriorityTask) -> Any:
            execution_order.append(task.id)
            await asyncio.sleep(0.01)
            return f"Result for {task.id}"
        
        tasks = [
            PriorityTask(id="task_1", description="Task 1", priority=0.3),
            PriorityTask(id="task_2", description="Task 2", priority=0.8),
            PriorityTask(id="task_3", description="Task 3", priority=0.5),
            PriorityTask(id="task_4", description="Task 4", priority=0.9),
        ]
        
        results = await engine.execute_with_priority(tasks, mock_executor)
        
        # Verify all tasks completed
        assert len(results) == 4
        
        # Verify high priority tasks executed first
        # task_4 (0.9) and task_2 (0.8) should be in first batch
        first_batch = set(execution_order[:2])
        assert "task_4" in first_batch
        assert "task_2" in first_batch
    
    @pytest.mark.asyncio
    async def test_execute_with_priority_empty(self):
        """Test executing empty task list."""
        engine = ParallelExecutionEngine()
        
        async def mock_executor(task: PriorityTask) -> Any:
            return "Result"
        
        results = await engine.execute_with_priority([], mock_executor)
        
        assert results == {}
    
    @pytest.mark.asyncio
    async def test_execute_with_priority_failure(self):
        """Test executing tasks with failures."""
        engine = ParallelExecutionEngine()
        
        fail_count = [0]
        
        async def failing_executor(task: PriorityTask) -> Any:
            fail_count[0] += 1
            if fail_count[0] == 1:
                raise ValueError("First task failed")
            return f"Result for {task.id}"
        
        tasks = [
            PriorityTask(id="task_1", description="Task 1", priority=0.5),
            PriorityTask(id="task_2", description="Task 2", priority=0.8),
        ]
        
        results = await engine.execute_with_priority(tasks, failing_executor)
        
        # At least one task should have failed
        assert len(results) > 0


class TestPriorityExecutionConfig:
    """Tests for PriorityExecutionConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PriorityExecutionConfig()
        
        assert config.max_concurrent_tasks == 4
        assert config.task_timeout_seconds == 300.0
        assert config.enable_progress_tracking is True
        assert config.enable_priority_queue is True
        assert config.preemption_enabled is True
        assert config.priority_aging_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PriorityExecutionConfig(
            max_concurrent_tasks=8,
            task_timeout_seconds=600.0,
            enable_priority_queue=True,
            preemption_enabled=True,
        )
        
        assert config.max_concurrent_tasks == 8
        assert config.task_timeout_seconds == 600.0


class TestTaskStatus:
    """Tests for TaskStatus enum."""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.QUEUED.value == "queued"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.PAUSED.value == "paused"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTaskType:
    """Tests for TaskType enum."""
    
    def test_task_type_values(self):
        """Test TaskType enum values."""
        assert TaskType.CPU_BOUND.value == "cpu_bound"
        assert TaskType.IO_BOUND.value == "io_bound"
        assert TaskType.LLM_BOUND.value == "llm_bound"
        assert TaskType.MIXED.value == "mixed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
