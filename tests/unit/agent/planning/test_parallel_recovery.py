"""Unit tests for Parallel Recovery module."""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any

from pyutagent.agent.planning.parallel_recovery import (
    ParallelRecoveryOrchestrator,
    RecoveryConfig,
    RecoveryStrategy,
    RecoveryStatus,
    RecoveryResult,
    SafeState,
    ErrorPattern,
)
from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskStatus


class TestRecoveryResult:
    """Test RecoveryResult dataclass."""
    
    def test_recovery_result_creation(self):
        """Test creating a recovery result."""
        result = RecoveryResult(
            strategy=RecoveryStrategy.RETRY,
            success=True,
            duration_ms=150.0,
            recovered_state={"status": "recovered"},
        )
        
        assert result.strategy == RecoveryStrategy.RETRY
        assert result.success is True
        assert result.duration_ms == 150.0
        assert result.recovered_state == {"status": "recovered"}
        assert isinstance(result.timestamp, datetime)


class TestSafeState:
    """Test SafeState dataclass."""
    
    def test_safe_state_creation(self):
        """Test creating a safe state."""
        state = SafeState(
            state_id="state-1",
            task_id="task-1",
            state_data={"data": "value"},
            is_consistent=True,
            checksum="abc123",
        )
        
        assert state.state_id == "state-1"
        assert state.task_id == "task-1"
        assert state.state_data == {"data": "value"}
        assert state.is_consistent is True
        assert state.checksum == "abc123"
        assert isinstance(state.created_at, datetime)


class TestErrorPattern:
    """Test ErrorPattern dataclass."""
    
    def test_error_pattern_creation(self):
        """Test creating an error pattern."""
        pattern = ErrorPattern(
            error_type="TimeoutError",
            error_message="Operation timed out",
            task_type="cpu_bound",
            occurrence_count=5,
        )
        
        assert pattern.error_type == "TimeoutError"
        assert pattern.error_message == "Operation timed out"
        assert pattern.task_type == "cpu_bound"
        assert pattern.occurrence_count == 5
        assert isinstance(pattern.last_occurrence, datetime)


class TestRecoveryConfig:
    """Test RecoveryConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RecoveryConfig()
        
        assert config.enable_parallel_recovery is True
        assert config.enable_automatic_rollback is True
        assert config.enable_error_learning is True
        assert config.max_parallel_strategies == 3
        assert config.rollback_timeout == 30.0
        assert config.max_recovery_attempts == 3
        assert config.strategy_timeout == 10.0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = RecoveryConfig(
            enable_parallel_recovery=False,
            max_parallel_strategies=2,
            max_recovery_attempts=5,
        )
        
        assert config.enable_parallel_recovery is False
        assert config.max_parallel_strategies == 2
        assert config.max_recovery_attempts == 5


class TestParallelRecoveryOrchestrator:
    """Test ParallelRecoveryOrchestrator class."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        orchestrator = ParallelRecoveryOrchestrator()
        
        assert orchestrator.config is not None
        assert orchestrator._safe_states == {}
        assert orchestrator._error_patterns == {}
        assert orchestrator._recovery_history == []
    
    @pytest.mark.asyncio
    async def test_recover_task_max_attempts_exceeded(self):
        """Test recovery when max attempts exceeded."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        
        orchestrator._task_recovery_count["task-1"] = 3
        
        result = await orchestrator.recover_task(task, "Test error")
        
        assert result.success is False
        assert "Max recovery attempts exceeded" in result.error
    
    @pytest.mark.asyncio
    async def test_recover_task_no_strategies(self):
        """Test recovery with no available strategies."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        
        async def mock_select(t, e):
            return []
        
        orchestrator._select_strategies = mock_select
        
        result = await orchestrator.recover_task(task, "Test error")
        
        assert result.success is False
    
    @pytest.mark.asyncio
    async def test_select_strategies_timeout_error(self):
        """Test strategy selection for timeout error."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        
        strategies = await orchestrator._select_strategies(task, "Operation timeout: deadline exceeded")
        
        assert RecoveryStrategy.RETRY in strategies
        assert RecoveryStrategy.ALTERNATIVE in strategies
    
    @pytest.mark.asyncio
    async def test_select_strategies_resource_error(self):
        """Test strategy selection for resource error."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        
        strategies = await orchestrator._select_strategies(task, "Resource error: out of memory")
        
        assert RecoveryStrategy.ROLLBACK in strategies
        assert RecoveryStrategy.RETRY in strategies
    
    @pytest.mark.asyncio
    async def test_select_strategies_dependency_error(self):
        """Test strategy selection for dependency error."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        
        strategies = await orchestrator._select_strategies(task, "Dependency error: missing module")
        
        assert RecoveryStrategy.SKIP in strategies
        assert RecoveryStrategy.ALTERNATIVE in strategies
    
    @pytest.mark.asyncio
    async def test_execute_retry_strategy(self):
        """Test retry strategy execution."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        result = await orchestrator._execute_retry(task, "Test error", datetime.now())
        
        assert result.success is True
        assert result.strategy == RecoveryStrategy.RETRY
        assert task.status == TaskStatus.PENDING
        assert task.retry_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_skip_strategy(self):
        """Test skip strategy execution."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        result = await orchestrator._execute_skip(task, "Test error", datetime.now())
        
        assert result.success is True
        assert result.strategy == RecoveryStrategy.SKIP
        assert task.status == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_execute_alternative_strategy(self):
        """Test alternative strategy execution."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        result = await orchestrator._execute_alternative(task, "Test error", datetime.now())
        
        assert result.success is True
        assert result.strategy == RecoveryStrategy.ALTERNATIVE
        assert task.metadata.get("alternative_approach") is True
    
    @pytest.mark.asyncio
    async def test_execute_rollback_no_safe_state(self):
        """Test rollback strategy with no safe state."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        
        result = await orchestrator._execute_rollback(task, "Test error", datetime.now())
        
        assert result.success is False
        assert "No safe state available" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_rollback_with_safe_state(self):
        """Test rollback strategy with safe state."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        safe_state = orchestrator.save_safe_state(
            "task-1",
            {"data": "original_value", "task_id": "task-1"},
        )
        
        result = await orchestrator._execute_rollback(task, "Test error", datetime.now())
        
        assert result.success is True
        assert result.strategy == RecoveryStrategy.ROLLBACK
        assert task.status == TaskStatus.PENDING
    
    def test_save_safe_state(self):
        """Test saving safe state."""
        orchestrator = ParallelRecoveryOrchestrator()
        
        state_data = {"key": "value", "counter": 42}
        safe_state = orchestrator.save_safe_state("task-1", state_data)
        
        assert safe_state.task_id == "task-1"
        assert safe_state.state_data == state_data
        assert safe_state.is_consistent is True
        assert safe_state.checksum is not None
    
    def test_get_safe_state(self):
        """Test retrieving safe state."""
        orchestrator = ParallelRecoveryOrchestrator()
        state_data = {"data": "test"}
        orchestrator.save_safe_state("task-1", state_data)
        
        safe_state = orchestrator.get_safe_state("task-1")
        
        assert safe_state is not None
        assert safe_state.task_id == "task-1"
        assert safe_state.state_data == state_data
    
    def test_get_safe_state_not_found(self):
        """Test retrieving non-existent safe state."""
        orchestrator = ParallelRecoveryOrchestrator()
        
        safe_state = orchestrator.get_safe_state("non-existent")
        
        assert safe_state is None
    
    def test_get_strategy_stats_empty(self):
        """Test getting strategy stats with no data."""
        orchestrator = ParallelRecoveryOrchestrator()
        
        stats = orchestrator.get_strategy_stats()
        
        assert len(stats) == len(RecoveryStrategy)
        for strategy_name, strategy_stats in stats.items():
            assert strategy_stats["success"] == 0
            assert strategy_stats["failure"] == 0
            assert strategy_stats["success_rate"] == 0.0
    
    def test_get_recovery_history_empty(self):
        """Test getting empty recovery history."""
        orchestrator = ParallelRecoveryOrchestrator()
        
        history = orchestrator.get_recovery_history()
        
        assert history == []
    
    def test_get_error_patterns_empty(self):
        """Test getting empty error patterns."""
        orchestrator = ParallelRecoveryOrchestrator()
        
        patterns = orchestrator.get_error_patterns()
        
        assert patterns == {}


class TestParallelRecoveryIntegration:
    """Integration tests for ParallelRecoveryOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_sequential_recovery_success(self):
        """Test sequential recovery with success."""
        config = RecoveryConfig(enable_parallel_recovery=False)
        orchestrator = ParallelRecoveryOrchestrator(config=config)
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        result = await orchestrator.recover_task(task, "Test error")
        
        assert result.success is True
        assert result.strategy in [RecoveryStrategy.RETRY, RecoveryStrategy.ALTERNATIVE]
    
    @pytest.mark.asyncio
    async def test_parallel_recovery_execution(self):
        """Test parallel recovery execution."""
        config = RecoveryConfig(
            enable_parallel_recovery=True,
            max_parallel_strategies=3,
        )
        orchestrator = ParallelRecoveryOrchestrator(config=config)
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        result = await orchestrator.recover_task(task, "Test error")
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_recovery_updates_stats(self):
        """Test that recovery updates strategy statistics."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        initial_stats = orchestrator.get_strategy_stats()
        
        result = await orchestrator.recover_task(task, "Test error")
        
        final_stats = orchestrator.get_strategy_stats()
        
        strategy_key = result.strategy.name
        assert final_stats[strategy_key]["success"] > initial_stats[strategy_key]["success"] or \
               final_stats[strategy_key]["failure"] > initial_stats[strategy_key]["failure"]
    
    @pytest.mark.asyncio
    async def test_error_learning(self):
        """Test error pattern learning."""
        config = RecoveryConfig(enable_error_learning=True)
        orchestrator = ParallelRecoveryOrchestrator(config=config)
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        error_message = "TimeoutError: Operation exceeded deadline"
        
        result = await orchestrator.recover_task(task, error_message)
        
        patterns = orchestrator.get_error_patterns()
        
        assert len(patterns) > 0
        
        pattern = list(patterns.values())[0]
        assert pattern.error_type == "TimeoutError"
        assert pattern.occurrence_count >= 1
        
        if result.success:
            assert result.strategy in pattern.successful_strategies
    
    @pytest.mark.asyncio
    async def test_multiple_recovery_attempts(self):
        """Test multiple recovery attempts for same task."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.FAILED)
        
        results = []
        for i in range(3):
            result = await orchestrator.recover_task(task, f"Error attempt {i+1}")
            results.append(result)
            task.status = TaskStatus.FAILED
        
        success_count = sum(1 for r in results if r.success)
        assert success_count > 0
    
    @pytest.mark.asyncio
    async def test_safe_state_rollback_flow(self):
        """Test complete flow of saving state and rolling back."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task", status=TaskStatus.RUNNING)
        
        initial_state = {"counter": 10, "status": "running"}
        orchestrator.save_safe_state("task-1", initial_state)
        
        task.status = TaskStatus.FAILED
        task.error = "Critical error"
        
        result = await orchestrator.recover_task(
            task,
            "Critical error",
            available_strategies=[RecoveryStrategy.ROLLBACK],
        )
        
        assert result.success is True
        assert result.strategy == RecoveryStrategy.ROLLBACK
        assert task.status == TaskStatus.PENDING


class TestRecoveryStrategies:
    """Test individual recovery strategies."""
    
    @pytest.mark.asyncio
    async def test_all_strategies_executable(self):
        """Test that all strategies can be executed."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        start_time = datetime.now()
        
        strategies = [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.SKIP,
            RecoveryStrategy.ALTERNATIVE,
            RecoveryStrategy.PARALLEL_RETRY,
        ]
        
        for strategy in strategies:
            result = await orchestrator._execute_single_strategy(task, "Test error", strategy)
            assert result is not None
            assert result.strategy == strategy
    
    @pytest.mark.asyncio
    async def test_rollback_requires_safe_state(self):
        """Test that rollback requires safe state."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        start_time = datetime.now()
        
        result = await orchestrator._execute_single_strategy(
            task, "Test error", RecoveryStrategy.ROLLBACK
        )
        
        assert result.success is False
        assert "No safe state" in result.error


class TestRecoveryEdgeCases:
    """Test edge cases in recovery."""
    
    @pytest.mark.asyncio
    async def test_recovery_with_unknown_error(self):
        """Test recovery with unknown error type."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        
        result = await orchestrator.recover_task(task, "UnknownError: xyz123")
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_recovery_with_empty_error_message(self):
        """Test recovery with empty error message."""
        orchestrator = ParallelRecoveryOrchestrator()
        task = PriorityTask(id="task-1", description="Test task")
        
        result = await orchestrator.recover_task(task, "")
        
        assert result is not None
    
    def test_save_inconsistent_state(self):
        """Test saving inconsistent state."""
        orchestrator = ParallelRecoveryOrchestrator()
        
        state_data = {"partial": "data"}
        safe_state = orchestrator.save_safe_state(
            "task-1",
            state_data,
            is_consistent=False,
        )
        
        assert safe_state.is_consistent is False
    
    def test_recovery_history_filtering(self):
        """Test filtering recovery history by task ID."""
        orchestrator = ParallelRecoveryOrchestrator()
        
        orchestrator._recovery_history = [
            RecoveryResult(
                strategy=RecoveryStrategy.RETRY,
                success=True,
                duration_ms=100.0,
                recovered_state={"task_id": "task-1"},
            ),
            RecoveryResult(
                strategy=RecoveryStrategy.ROLLBACK,
                success=True,
                duration_ms=150.0,
                recovered_state={"task_id": "task-2"},
            ),
        ]
        
        task1_history = orchestrator.get_recovery_history(task_id="task-1")
        assert len(task1_history) == 1
        assert task1_history[0].recovered_state["task_id"] == "task-1"
