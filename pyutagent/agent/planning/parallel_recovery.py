"""Parallel Recovery Orchestrator Module.

Provides intelligent failure recovery with:
- Multi-strategy parallel recovery
- Automatic rollback to safe state
- Error pattern learning
- Recovery success optimization

This is part of Phase 3 Week 5: Parallel Recovery Integration.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskStatus

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = auto()
    ROLLBACK = auto()
    SKIP = auto()
    ALTERNATIVE = auto()
    PARALLEL_RETRY = auto()


class RecoveryStatus(Enum):
    """Recovery status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt.
    
    Attributes:
        strategy: Strategy used for recovery
        success: Whether recovery succeeded
        duration_ms: Recovery duration in milliseconds
        error: Error message if failed
        recovered_state: Recovered state if successful
        timestamp: Recovery timestamp
    """
    strategy: RecoveryStrategy
    success: bool
    duration_ms: float
    error: Optional[str] = None
    recovered_state: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SafeState:
    """Safe state for rollback.
    
    Attributes:
        state_id: Unique state identifier
        task_id: Associated task ID
        state_data: State data snapshot
        created_at: State creation timestamp
        is_consistent: Whether state is consistent
        checksum: State checksum for validation
    """
    state_id: str
    task_id: str
    state_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    is_consistent: bool = True
    checksum: Optional[str] = None


@dataclass
class ErrorPattern:
    """Error pattern for learning.
    
    Attributes:
        error_type: Type of error
        error_message: Error message pattern
        task_type: Associated task type
        occurrence_count: Number of occurrences
        successful_strategies: Strategies that worked
        failed_strategies: Strategies that failed
        avg_recovery_time: Average recovery time
        last_occurrence: Last occurrence timestamp
    """
    error_type: str
    error_message: str
    task_type: Optional[str] = None
    occurrence_count: int = 0
    successful_strategies: List[RecoveryStrategy] = field(default_factory=list)
    failed_strategies: List[RecoveryStrategy] = field(default_factory=list)
    avg_recovery_time: float = 0.0
    last_occurrence: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryConfig:
    """Configuration for recovery orchestrator.
    
    Attributes:
        enable_parallel_recovery: Enable parallel strategy execution
        enable_automatic_rollback: Enable automatic rollback
        enable_error_learning: Enable error pattern learning
        max_parallel_strategies: Maximum parallel strategies to execute
        rollback_timeout: Rollback timeout in seconds
        max_recovery_attempts: Maximum recovery attempts per task
        strategy_timeout: Individual strategy timeout in seconds
    """
    enable_parallel_recovery: bool = True
    enable_automatic_rollback: bool = True
    enable_error_learning: bool = True
    max_parallel_strategies: int = 3
    rollback_timeout: float = 30.0
    max_recovery_attempts: int = 3
    strategy_timeout: float = 10.0


class ParallelRecoveryOrchestrator:
    """Orchestrator for parallel task recovery.
    
    Provides:
    - Multi-strategy parallel recovery
    - Automatic rollback
    - Error pattern learning
    - Recovery optimization
    """
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        """Initialize ParallelRecoveryOrchestrator.
        
        Args:
            config: Recovery configuration
        """
        self.config = config or RecoveryConfig()
        self._safe_states: Dict[str, SafeState] = {}
        self._error_patterns: Dict[str, ErrorPattern] = {}
        self._recovery_history: List[RecoveryResult] = []
        self._task_recovery_count: Dict[str, int] = {}
        self._strategy_stats: Dict[RecoveryStrategy, Dict[str, Any]] = {
            strategy: {"success": 0, "failure": 0, "total_time": 0.0}
            for strategy in RecoveryStrategy
        }
    
    async def recover_task(
        self,
        task: PriorityTask,
        error: str,
        available_strategies: Optional[List[RecoveryStrategy]] = None,
    ) -> RecoveryResult:
        """Attempt to recover a failed task.
        
        Args:
            task: Failed task to recover
            error: Error message that caused failure
            available_strategies: List of strategies to try
            
        Returns:
            RecoveryResult with recovery outcome
        """
        task_id = task.id
        
        # Check if we've exceeded max recovery attempts
        recovery_count = self._task_recovery_count.get(task_id, 0)
        if recovery_count >= self.config.max_recovery_attempts:
            logger.warning(f"Task {task_id} exceeded max recovery attempts")
            return RecoveryResult(
                strategy=RecoveryStrategy.RETRY,
                success=False,
                duration_ms=0.0,
                error="Max recovery attempts exceeded",
            )
        
        # Select strategies
        if available_strategies is None:
            available_strategies = await self._select_strategies(task, error)
        
        if not available_strategies:
            logger.warning(f"No recovery strategies available for task {task_id}")
            return RecoveryResult(
                strategy=RecoveryStrategy.RETRY,
                success=False,
                duration_ms=0.0,
                error="No recovery strategies available",
            )
        
        # Execute recovery
        if self.config.enable_parallel_recovery and len(available_strategies) > 1:
            result = await self._execute_parallel_recovery(
                task, error, available_strategies
            )
        else:
            result = await self._execute_sequential_recovery(
                task, error, available_strategies
            )
        
        # Update statistics
        if result:
            self._update_strategy_stats(result)
            self._recovery_history.append(result)
            
            if result.success:
                self._task_recovery_count[task_id] = 0
            else:
                self._task_recovery_count[task_id] = recovery_count + 1
            
            # Learn from error pattern
            if self.config.enable_error_learning:
                await self._learn_from_error(task, error, result)
        
        return result or RecoveryResult(
            strategy=available_strategies[0],
            success=False,
            duration_ms=0.0,
            error="All recovery strategies failed",
        )
    
    async def _select_strategies(
        self,
        task: PriorityTask,
        error: str,
    ) -> List[RecoveryStrategy]:
        """Select recovery strategies based on error pattern.
        
        Args:
            task: Failed task
            error: Error message
            
        Returns:
            List of recommended strategies
        """
        # Try to match error pattern
        error_pattern = await self._match_error_pattern(error, task)
        
        if error_pattern and error_pattern.successful_strategies:
            # Use historically successful strategies
            return error_pattern.successful_strategies[:self.config.max_parallel_strategies]
        
        # Default strategy selection based on error type
        strategies = []
        
        if "timeout" in error.lower() or "deadline" in error.lower():
            strategies.append(RecoveryStrategy.RETRY)
            strategies.append(RecoveryStrategy.ALTERNATIVE)
        elif "resource" in error.lower() or "memory" in error.lower():
            strategies.append(RecoveryStrategy.ROLLBACK)
            strategies.append(RecoveryStrategy.RETRY)
        elif "dependency" in error.lower():
            strategies.append(RecoveryStrategy.SKIP)
            strategies.append(RecoveryStrategy.ALTERNATIVE)
        else:
            strategies.append(RecoveryStrategy.RETRY)
            strategies.append(RecoveryStrategy.ROLLBACK)
            strategies.append(RecoveryStrategy.ALTERNATIVE)
        
        return strategies[:self.config.max_parallel_strategies]
    
    async def _match_error_pattern(
        self,
        error: str,
        task: PriorityTask,
    ) -> Optional[ErrorPattern]:
        """Match error to known patterns.
        
        Args:
            error: Error message
            task: Failed task
            
        Returns:
            Matching error pattern or None
        """
        error_type = error.split(":")[0].strip() if ":" in error else "unknown"
        
        for pattern in self._error_patterns.values():
            if (pattern.error_type == error_type and
                pattern.error_message.lower() in error.lower()):
                return pattern
        
        return None
    
    async def _execute_parallel_recovery(
        self,
        task: PriorityTask,
        error: str,
        strategies: List[RecoveryStrategy],
    ) -> Optional[RecoveryResult]:
        """Execute multiple recovery strategies in parallel.
        
        Args:
            task: Failed task
            error: Error message
            strategies: Strategies to execute
            
        Returns:
            Best recovery result
        """
        logger.info(f"Executing parallel recovery for task {task.id} with {len(strategies)} strategies")
        
        # Create tasks for parallel execution
        recovery_tasks = [
            self._execute_single_strategy(task, error, strategy)
            for strategy in strategies
        ]
        
        # Execute with timeout
        try:
            done, pending = await asyncio.wait(
                [asyncio.create_task(task) for task in recovery_tasks],
                timeout=self.config.strategy_timeout * len(strategies),
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Get results
            results = []
            for task in done:
                try:
                    result = task.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Recovery task failed: {e}")
            
            if not results:
                return None
            
            # Select best result (first success or least failed)
            success_results = [r for r in results if r.success]
            if success_results:
                return success_results[0]
            
            return results[0]
            
        except asyncio.TimeoutError:
            logger.error(f"Parallel recovery timed out for task {task.id}")
            return None
    
    async def _execute_sequential_recovery(
        self,
        task: PriorityTask,
        error: str,
        strategies: List[RecoveryStrategy],
    ) -> Optional[RecoveryResult]:
        """Execute recovery strategies sequentially.
        
        Args:
            task: Failed task
            error: Error message
            strategies: Strategies to execute
            
        Returns:
            First successful result or last failed result
        """
        last_result = None
        
        for strategy in strategies:
            logger.debug(f"Trying recovery strategy {strategy.name} for task {task.id}")
            
            result = await self._execute_single_strategy(task, error, strategy)
            
            if result and result.success:
                return result
            
            if result:
                last_result = result
        
        return last_result
    
    async def _execute_single_strategy(
        self,
        task: PriorityTask,
        error: str,
        strategy: RecoveryStrategy,
    ) -> RecoveryResult:
        """Execute a single recovery strategy.
        
        Args:
            task: Failed task
            error: Error message
            strategy: Strategy to execute
            
        Returns:
            Recovery result
        """
        start_time = datetime.now()
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._execute_retry(task, error, start_time)
            elif strategy == RecoveryStrategy.ROLLBACK:
                return await self._execute_rollback(task, error, start_time)
            elif strategy == RecoveryStrategy.SKIP:
                return await self._execute_skip(task, error, start_time)
            elif strategy == RecoveryStrategy.ALTERNATIVE:
                return await self._execute_alternative(task, error, start_time)
            elif strategy == RecoveryStrategy.PARALLEL_RETRY:
                return await self._execute_parallel_retry(task, error, start_time)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Strategy {strategy.name} failed: {e}")
            return RecoveryResult(
                strategy=strategy,
                success=False,
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error=str(e),
            )
    
    async def _execute_retry(
        self,
        task: PriorityTask,
        error: str,
        start_time: datetime,
    ) -> RecoveryResult:
        """Execute retry strategy.
        
        Args:
            task: Failed task
            error: Error message
            start_time: Strategy start time
            
        Returns:
            Recovery result
        """
        logger.debug(f"Retrying task {task.id}")
        
        # Reset task status
        task.status = TaskStatus.PENDING
        task.retry_count += 1
        task.error = None
        
        # Simulate retry (in real implementation, this would re-execute the task)
        await asyncio.sleep(0.01)
        
        return RecoveryResult(
            strategy=RecoveryStrategy.RETRY,
            success=True,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            recovered_state={"retry_count": task.retry_count},
        )
    
    async def _execute_rollback(
        self,
        task: PriorityTask,
        error: str,
        start_time: datetime,
    ) -> RecoveryResult:
        """Execute rollback strategy.
        
        Args:
            task: Failed task
            error: Error message
            start_time: Strategy start time
            
        Returns:
            Recovery result
        """
        logger.debug(f"Rolling back task {task.id}")
        
        # Find safe state
        safe_state = self._safe_states.get(task.id)
        
        if not safe_state:
            return RecoveryResult(
                strategy=RecoveryStrategy.ROLLBACK,
                success=False,
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                error="No safe state available for rollback",
            )
        
        # Restore state
        task.status = TaskStatus.PENDING
        task.result = None
        task.error = None
        
        return RecoveryResult(
            strategy=RecoveryStrategy.ROLLBACK,
            success=True,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            recovered_state=safe_state.state_data,
        )
    
    async def _execute_skip(
        self,
        task: PriorityTask,
        error: str,
        start_time: datetime,
    ) -> RecoveryResult:
        """Execute skip strategy.
        
        Args:
            task: Failed task
            error: Error message
            start_time: Strategy start time
            
        Returns:
            Recovery result
        """
        logger.debug(f"Skipping task {task.id}")
        
        task.status = TaskStatus.CANCELLED
        task.error = f"Skipped due to: {error}"
        
        return RecoveryResult(
            strategy=RecoveryStrategy.SKIP,
            success=True,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )
    
    async def _execute_alternative(
        self,
        task: PriorityTask,
        error: str,
        start_time: datetime,
    ) -> RecoveryResult:
        """Execute alternative strategy.
        
        Args:
            task: Failed task
            error: Error message
            start_time: Strategy start time
            
        Returns:
            Recovery result
        """
        logger.debug(f"Trying alternative approach for task {task.id}")
        
        # In real implementation, this would try an alternative implementation
        task.status = TaskStatus.PENDING
        task.metadata["alternative_approach"] = True
        
        return RecoveryResult(
            strategy=RecoveryStrategy.ALTERNATIVE,
            success=True,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )
    
    async def _execute_parallel_retry(
        self,
        task: PriorityTask,
        error: str,
        start_time: datetime,
    ) -> RecoveryResult:
        """Execute parallel retry strategy.
        
        Args:
            task: Failed task
            error: Error message
            start_time: Strategy start time
            
        Returns:
            Recovery result
        """
        logger.debug(f"Executing parallel retry for task {task.id}")
        
        task.status = TaskStatus.PENDING
        task.metadata["parallel_retry"] = True
        
        return RecoveryResult(
            strategy=RecoveryStrategy.PARALLEL_RETRY,
            success=True,
            duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )
    
    def save_safe_state(
        self,
        task_id: str,
        state_data: Dict[str, Any],
        is_consistent: bool = True,
    ) -> SafeState:
        """Save a safe state for potential rollback.
        
        Args:
            task_id: Task identifier
            state_data: State data snapshot
            is_consistent: Whether state is consistent
            
        Returns:
            Saved safe state
        """
        import hashlib
        
        state_id = f"{task_id}_{datetime.now().timestamp()}"
        checksum = hashlib.md5(str(state_data).encode()).hexdigest()
        
        safe_state = SafeState(
            state_id=state_id,
            task_id=task_id,
            state_data=state_data,
            is_consistent=is_consistent,
            checksum=checksum,
        )
        
        self._safe_states[task_id] = safe_state
        logger.debug(f"Saved safe state for task {task_id}")
        
        return safe_state
    
    def get_safe_state(self, task_id: str) -> Optional[SafeState]:
        """Get safe state for a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Safe state or None if not found
        """
        return self._safe_states.get(task_id)
    
    async def _learn_from_error(
        self,
        task: PriorityTask,
        error: str,
        result: RecoveryResult,
    ) -> None:
        """Learn from error and recovery outcome.
        
        Args:
            task: Failed task
            error: Error message
            result: Recovery result
        """
        error_type = error.split(":")[0].strip() if ":" in error else "unknown"
        pattern_key = f"{error_type}_{task.id}"
        
        if pattern_key not in self._error_patterns:
            self._error_patterns[pattern_key] = ErrorPattern(
                error_type=error_type,
                error_message=error,
                task_type=str(task.status),
            )
        
        pattern = self._error_patterns[pattern_key]
        pattern.occurrence_count += 1
        pattern.last_occurrence = datetime.now()
        
        if result.success:
            if result.strategy not in pattern.successful_strategies:
                pattern.successful_strategies.append(result.strategy)
        else:
            if result.strategy not in pattern.failed_strategies:
                pattern.failed_strategies.append(result.strategy)
        
        # Update average recovery time
        total_time = pattern.avg_recovery_time * (pattern.occurrence_count - 1)
        pattern.avg_recovery_time = (total_time + result.duration_ms) / pattern.occurrence_count
        
        logger.debug(f"Updated error pattern for {pattern_key}")
    
    def _update_strategy_stats(self, result: RecoveryResult) -> None:
        """Update strategy statistics.
        
        Args:
            result: Recovery result
        """
        stats = self._strategy_stats[result.strategy]
        
        if result.success:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        
        stats["total_time"] += result.duration_ms
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """Get strategy statistics.
        
        Returns:
            Dictionary with strategy statistics
        """
        return {
            strategy.name: {
                "success": stats["success"],
                "failure": stats["failure"],
                "success_rate": stats["success"] / (stats["success"] + stats["failure"]) 
                    if (stats["success"] + stats["failure"]) > 0 else 0.0,
                "avg_time_ms": stats["total_time"] / (stats["success"] + stats["failure"])
                    if (stats["success"] + stats["failure"]) > 0 else 0.0,
            }
            for strategy, stats in self._strategy_stats.items()
        }
    
    def get_recovery_history(self, task_id: Optional[str] = None) -> List[RecoveryResult]:
        """Get recovery history.
        
        Args:
            task_id: Optional task ID to filter by
            
        Returns:
            List of recovery results
        """
        if task_id:
            return [r for r in self._recovery_history if r.recovered_state and 
                    str(r.recovered_state.get("task_id", "")) == task_id]
        return self._recovery_history.copy()
    
    def get_error_patterns(self) -> Dict[str, ErrorPattern]:
        """Get learned error patterns.
        
        Returns:
            Dictionary of error patterns
        """
        return self._error_patterns.copy()
