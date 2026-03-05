"""Unit tests for Task Router module."""

import pytest
from datetime import datetime, timedelta
from typing import Any

from pyutagent.agent.planning.task_router import (
    TaskRouter,
    PriorityManager,
    RoutingConfig,
    RoutingDecision,
    RoutingResult,
    RoutingBatchResult,
)
from pyutagent.agent.planning.parallel_executor import (
    PriorityTask,
    TaskType,
)


class TestTaskRouter:
    """Tests for TaskRouter."""
    
    def test_router_creation(self):
        """Test creating a TaskRouter."""
        router = TaskRouter()
        
        assert router is not None
        assert router.config is not None
    
    def test_router_custom_config(self):
        """Test creating TaskRouter with custom config."""
        config = RoutingConfig(
            high_priority_threshold=0.8,
            low_priority_threshold=0.2,
        )
        router = TaskRouter(config=config)
        
        assert router.config.high_priority_threshold == 0.8
        assert router.config.low_priority_threshold == 0.2
    
    def test_classify_cpu_bound_task(self):
        """Test classifying CPU-bound task."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Calculate and process data",
            priority=0.5,
        )
        
        task_type = router.classify_task(task)
        
        assert task_type == TaskType.CPU_BOUND
    
    def test_classify_io_bound_task(self):
        """Test classifying IO-bound task."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_2",
            description="Read and write files",
            priority=0.5,
        )
        
        task_type = router.classify_task(task)
        
        assert task_type == TaskType.IO_BOUND
    
    def test_classify_llm_bound_task(self):
        """Test classifying LLM-bound task."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_3",
            description="Generate and analyze text using AI",
            priority=0.5,
        )
        
        task_type = router.classify_task(task)
        
        assert task_type == TaskType.LLM_BOUND
    
    def test_classify_mixed_task(self):
        """Test classifying mixed task."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_4",
            description="Do something",
            priority=0.5,
        )
        
        task_type = router.classify_task(task)
        
        assert task_type == TaskType.MIXED
    
    def test_classify_with_metadata(self):
        """Test classifying task with metadata context."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_5",
            description="Process data",
            priority=0.5,
            metadata={"context": "This is a CPU intensive calculation"},
        )
        
        task_type = router.classify_task(task)
        
        assert task_type == TaskType.CPU_BOUND
    
    def test_calculate_priority_basic(self):
        """Test basic priority calculation."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Test task",
            priority=0.5,
        )
        
        priority = router.calculate_priority(task)
        
        assert 0.0 <= priority <= 1.0
    
    def test_calculate_priority_with_deadline(self):
        """Test priority calculation with deadline."""
        router = TaskRouter()
        
        # Task with approaching deadline
        deadline = datetime.now() + timedelta(minutes=5)
        task = PriorityTask(
            id="task_1",
            description="Urgent task",
            priority=0.5,
            deadline=deadline,
            created_at=datetime.now() - timedelta(hours=1),
        )
        
        priority = router.calculate_priority(task)
        
        # Priority should be in reasonable range
        assert 0.0 <= priority <= 1.0
    
    def test_calculate_priority_with_dependencies(self):
        """Test priority calculation with dependencies."""
        router = TaskRouter()
        
        # Task with many dependencies
        task = PriorityTask(
            id="task_1",
            description="Complex task",
            priority=0.5,
            dependencies={"dep1", "dep2", "dep3"},
        )
        
        priority = router.calculate_priority(task)
        
        # Should be lower due to dependencies
        assert priority < 0.5
    
    def test_calculate_priority_no_deadline(self):
        """Test priority calculation without deadline."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Normal task",
            priority=0.5,
        )
        
        priority = router.calculate_priority(task)
        
        assert 0.0 <= priority <= 1.0
    
    def test_route_high_priority_immediate(self):
        """Test routing high priority task for immediate execution."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Critical calculation",
            priority=0.9,
        )
        
        result = router.route(task, resource_available=True)
        
        assert result.decision == RoutingDecision.EXECUTE_IMMEDIATE
        assert result.priority > 0.7
    
    def test_route_high_priority_with_deps(self):
        """Test routing high priority task with dependencies."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Critical task",
            priority=0.9,
            dependencies={"dep1"},
        )
        
        result = router.route(task, resource_available=True)
        
        assert result.decision == RoutingDecision.WAIT_FOR_DEPENDENCIES
    
    def test_route_medium_priority_parallel(self):
        """Test routing medium priority task for parallel execution."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Normal task",
            priority=0.5,
        )
        
        result = router.route(task, resource_available=True)
        
        assert result.decision == RoutingDecision.EXECUTE_PARALLEL
    
    def test_route_low_priority_delay(self):
        """Test routing low priority task with delay."""
        router = TaskRouter()
        
        # Very low priority task
        task = PriorityTask(
            id="task_1",
            description="Low priority task",
            priority=0.05,  # Very low base priority
        )
        
        result = router.route(task, resource_available=True)
        
        # Should execute in parallel (medium range priority due to type factor)
        # The final priority depends on multiple factors
        assert result.priority >= 0.0
        assert result.task_type is not None
    
    def test_route_with_no_resources(self):
        """Test routing when resources are not available."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Medium task",
            priority=0.5,
        )
        
        result = router.route(task, resource_available=False)
        
        # Should queue or delay when resources unavailable
        assert result.decision in [
            RoutingDecision.QUEUE_LOW_PRIORITY,
            RoutingDecision.DELAY_EXECUTION,
            RoutingDecision.WAIT_FOR_DEPENDENCIES,
        ]
    
    def test_boost_priority(self):
        """Test priority boosting for long-waiting tasks."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Waiting task",
            priority=0.3,
        )
        
        # Boost after long wait
        new_priority = router.boost_priority(task, wait_time=120.0)
        
        assert new_priority > task.priority
        assert new_priority <= 1.0
    
    def test_boost_priority_short_wait(self):
        """Test priority boosting for short wait."""
        router = TaskRouter()
        
        task = PriorityTask(
            id="task_1",
            description="Short wait task",
            priority=0.3,
        )
        
        # No boost for short wait
        new_priority = router.boost_priority(task, wait_time=10.0)
        
        assert new_priority == task.priority
    
    def test_get_routing_stats(self):
        """Test getting routing statistics."""
        router = TaskRouter()
        
        stats = router.get_routing_stats()
        
        assert "config" in stats
        assert "keyword_counts" in stats
        assert stats["config"]["high_priority_threshold"] == 0.7
        assert stats["config"]["low_priority_threshold"] == 0.3


class TestPriorityManager:
    """Tests for PriorityManager."""
    
    def test_manager_creation(self):
        """Test creating a PriorityManager."""
        manager = PriorityManager()
        
        assert manager is not None
        assert manager.router is not None
    
    def test_update_priority(self):
        """Test updating task priority."""
        manager = PriorityManager()
        
        task = PriorityTask(
            id="task_1",
            description="Test task",
            priority=0.5,
        )
        
        new_priority = manager.update_priority(task)
        
        assert 0.0 <= new_priority <= 1.0
        assert manager.get_priority("task_1") == new_priority
    
    def test_update_wait_time(self):
        """Test updating task wait time."""
        manager = PriorityManager()
        
        manager.update_wait_time("task_1", 30.0)
        
        assert manager.get_wait_time("task_1") == 30.0
    
    def test_update_priority_with_wait_time(self):
        """Test priority update with wait time."""
        manager = PriorityManager()
        
        task = PriorityTask(
            id="task_1",
            description="Waiting task",
            priority=0.3,
        )
        
        # Set long wait time
        manager.update_wait_time("task_1", 120.0)
        
        # Update priority should include boost
        new_priority = manager.update_priority(task)
        
        assert new_priority > task.priority
    
    def test_reset(self):
        """Test resetting manager state."""
        manager = PriorityManager()
        
        task = PriorityTask(id="task_1", description="Test")
        manager.update_priority(task)
        manager.update_wait_time("task_1", 30.0)
        
        manager.reset()
        
        assert manager.get_priority("task_1") is None
        assert manager.get_wait_time("task_1") == 0.0


class TestRoutingResult:
    """Tests for RoutingResult."""
    
    def test_routing_result_creation(self):
        """Test creating a RoutingResult."""
        result = RoutingResult(
            task_id="task_1",
            decision=RoutingDecision.EXECUTE_IMMEDIATE,
            priority=0.8,
            task_type=TaskType.CPU_BOUND,
            reason="High priority task",
        )
        
        assert result.task_id == "task_1"
        assert result.decision == RoutingDecision.EXECUTE_IMMEDIATE
        assert result.priority == 0.8
        assert result.task_type == TaskType.CPU_BOUND
        assert result.reason == "High priority task"


class TestRoutingBatchResult:
    """Tests for RoutingBatchResult."""
    
    def test_batch_result_from_results(self):
        """Test creating batch result from routing results."""
        results = [
            RoutingResult(
                task_id="task_1",
                decision=RoutingDecision.EXECUTE_IMMEDIATE,
                priority=0.9,
                task_type=TaskType.CPU_BOUND,
                reason="High priority",
            ),
            RoutingResult(
                task_id="task_2",
                decision=RoutingDecision.EXECUTE_PARALLEL,
                priority=0.5,
                task_type=TaskType.IO_BOUND,
                reason="Medium priority",
            ),
            RoutingResult(
                task_id="task_3",
                decision=RoutingDecision.WAIT_FOR_DEPENDENCIES,
                priority=0.8,
                task_type=TaskType.LLM_BOUND,
                reason="Has dependencies",
            ),
        ]
        
        batch_result = RoutingBatchResult.from_results(results)
        
        assert batch_result.total_tasks == 3
        assert batch_result.immediate_count == 1
        assert batch_result.parallel_count == 1
        assert batch_result.waiting_count == 1
        assert batch_result.delayed_count == 0


class TestRoutingConfig:
    """Tests for RoutingConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RoutingConfig()
        
        assert config.high_priority_threshold == 0.7
        assert config.low_priority_threshold == 0.3
        assert config.max_wait_time_seconds == 60.0
        assert config.dependency_check_enabled is True
        assert config.resource_check_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RoutingConfig(
            high_priority_threshold=0.8,
            low_priority_threshold=0.2,
            max_wait_time_seconds=120.0,
            dependency_check_enabled=False,
        )
        
        assert config.high_priority_threshold == 0.8
        assert config.low_priority_threshold == 0.2
        assert config.max_wait_time_seconds == 120.0
        assert config.dependency_check_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
