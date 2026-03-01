"""Parallel recovery strategies for faster error resolution.

This module provides parallel recovery capabilities:
- Parallel strategy execution
- Result aggregation
- Best result selection
- Timeout handling
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

from .error_recovery import (
    ErrorCategory,
    RecoveryStrategy,
    RecoveryResult,
    RecoveryContext,
)

logger = logging.getLogger(__name__)


@dataclass
class ParallelRecoveryResult:
    """Result of parallel recovery execution."""
    success: bool
    best_strategy: RecoveryStrategy
    best_result: RecoveryResult
    all_results: Dict[RecoveryStrategy, RecoveryResult]
    total_attempts: int
    successful_strategies: List[RecoveryStrategy]
    failed_strategies: List[RecoveryStrategy]
    elapsed_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyTask:
    """A task for executing a recovery strategy."""
    strategy: RecoveryStrategy
    coroutine: Any
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[RecoveryResult] = None
    error: Optional[Exception] = None


class ParallelRecoveryManager:
    """Manages parallel recovery strategy execution.
    
    Features:
    - Execute multiple strategies in parallel
    - Select best result
    - Cancel remaining tasks on success
    - Timeout handling
    """
    
    def __init__(
        self,
        max_parallel: int = 3,
        default_timeout: float = 60.0,
        cancel_on_success: bool = True
    ):
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self.cancel_on_success = cancel_on_success
        
        self._active_tasks: Dict[RecoveryStrategy, asyncio.Task] = {}
        self._results: Dict[RecoveryStrategy, RecoveryResult] = {}
        self._cancel_event = asyncio.Event()
    
    async def recover_with_parallel_strategies(
        self,
        error: Exception,
        strategies: List[RecoveryStrategy],
        context: RecoveryContext,
        strategy_executor: Callable[[RecoveryStrategy, RecoveryContext], RecoveryResult],
        timeout: Optional[float] = None
    ) -> ParallelRecoveryResult:
        """Execute multiple recovery strategies in parallel.
        
        Args:
            error: The error to recover from
            strategies: List of strategies to try in parallel
            context: Recovery context
            strategy_executor: Function to execute a strategy
            timeout: Optional timeout for all strategies
            
        Returns:
            ParallelRecoveryResult with the best result
        """
        self._cancel_event.clear()
        self._results.clear()
        self._active_tasks.clear()
        
        strategies = strategies[:self.max_parallel]
        
        if not strategies:
            return ParallelRecoveryResult(
                success=False,
                best_strategy=RecoveryStrategy.ANALYZE_AND_FIX,
                best_result=RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.ANALYZE_AND_FIX,
                    attempts_made=0,
                    error_message="No strategies provided"
                ),
                all_results={},
                total_attempts=0,
                successful_strategies=[],
                failed_strategies=[],
                elapsed_time=0.0
            )
        
        logger.info(
            f"[ParallelRecovery] Starting parallel recovery with {len(strategies)} strategies: "
            f"{[s.name for s in strategies]}"
        )
        
        start_time = time.time()
        
        tasks = []
        for strategy in strategies:
            task = asyncio.create_task(
                self._execute_strategy(
                    strategy,
                    context,
                    strategy_executor
                )
            )
            self._active_tasks[strategy] = task
            tasks.append((strategy, task))
        
        try:
            timeout_val = timeout or self.default_timeout
            
            done, pending = await asyncio.wait(
                [t for _, t in tasks],
                timeout=timeout_val,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for strategy, task in tasks:
                if task in done:
                    try:
                        result = task.result()
                        self._results[strategy] = result
                        
                        if result.success and self.cancel_on_success:
                            logger.info(
                                f"[ParallelRecovery] Strategy {strategy.name} succeeded, "
                                f"cancelling remaining tasks"
                            )
                            self._cancel_event.set()
                            for _, pending_task in tasks:
                                if pending_task in pending:
                                    pending_task.cancel()
                            break
                    except Exception as e:
                        logger.warning(
                            f"[ParallelRecovery] Strategy {strategy.name} raised exception: {e}"
                        )
                        self._results[strategy] = RecoveryResult(
                            success=False,
                            strategy_used=strategy,
                            attempts_made=1,
                            error_message=str(e)
                        )
            
            if pending and not self._cancel_event.is_set():
                done_more, still_pending = await asyncio.wait(
                    pending,
                    timeout=timeout_val / 2,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                for strategy, task in tasks:
                    if task in done_more:
                        try:
                            result = task.result()
                            self._results[strategy] = result
                        except Exception as e:
                            self._results[strategy] = RecoveryResult(
                                success=False,
                                strategy_used=strategy,
                                attempts_made=1,
                                error_message=str(e)
                            )
                
                for task in still_pending:
                    task.cancel()
        
        except asyncio.TimeoutError:
            logger.warning("[ParallelRecovery] Parallel recovery timed out")
            for _, task in tasks:
                task.cancel()
        
        elapsed = time.time() - start_time
        
        best_strategy, best_result = self._select_best_result()
        
        successful = [s for s, r in self._results.items() if r.success]
        failed = [s for s, r in self._results.items() if not r.success]
        
        logger.info(
            f"[ParallelRecovery] Parallel recovery complete - "
            f"Best: {best_strategy.name}, "
            f"Successful: {[s.name for s in successful]}, "
            f"Failed: {[s.name for s in failed]}, "
            f"Time: {elapsed:.2f}s"
        )
        
        return ParallelRecoveryResult(
            success=best_result.success if best_result else False,
            best_strategy=best_strategy,
            best_result=best_result,
            all_results=self._results.copy(),
            total_attempts=len(self._results),
            successful_strategies=successful,
            failed_strategies=failed,
            elapsed_time=elapsed
        )
    
    async def _execute_strategy(
        self,
        strategy: RecoveryStrategy,
        context: RecoveryContext,
        executor: Callable
    ) -> RecoveryResult:
        """Execute a single strategy."""
        if self._cancel_event.is_set():
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                attempts_made=0,
                error_message="Cancelled"
            )
        
        try:
            result = await executor(strategy, context)
            return result
        except Exception as e:
            logger.exception(f"[ParallelRecovery] Strategy {strategy.name} failed: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                attempts_made=1,
                error_message=str(e)
            )
    
    def _select_best_result(
        self
    ) -> Tuple[RecoveryStrategy, RecoveryResult]:
        """Select the best result from all results."""
        if not self._results:
            return (
                RecoveryStrategy.ANALYZE_AND_FIX,
                RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.ANALYZE_AND_FIX,
                    attempts_made=0,
                    error_message="No results"
                )
            )
        
        for strategy, result in self._results.items():
            if result.success:
                return (strategy, result)
        
        strategy_priority = [
            RecoveryStrategy.ANALYZE_AND_FIX,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
            RecoveryStrategy.RESET_AND_REGENERATE,
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.RETRY_IMMEDIATE,
            RecoveryStrategy.SKIP_AND_CONTINUE,
            RecoveryStrategy.ESCALATE_TO_USER,
        ]
        
        for strategy in strategy_priority:
            if strategy in self._results:
                return (strategy, self._results[strategy])
        
        return next(iter(self._results.items()))
    
    def cancel_all(self):
        """Cancel all active recovery tasks."""
        self._cancel_event.set()
        for task in self._active_tasks.values():
            task.cancel()
        logger.info("[ParallelRecovery] Cancelled all active tasks")


class StrategyPrioritizer:
    """Prioritizes recovery strategies based on error type."""
    
    STRATEGY_PRIORITY_MAP: Dict[ErrorCategory, List[RecoveryStrategy]] = {
        ErrorCategory.COMPILATION_ERROR: [
            RecoveryStrategy.ANALYZE_AND_FIX,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
            RecoveryStrategy.RESET_AND_REGENERATE,
        ],
        ErrorCategory.TEST_FAILURE: [
            RecoveryStrategy.ANALYZE_AND_FIX,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
            RecoveryStrategy.RESET_AND_REGENERATE,
        ],
        ErrorCategory.LLM_API_ERROR: [
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.RETRY_IMMEDIATE,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
        ],
        ErrorCategory.NETWORK: [
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.RETRY_IMMEDIATE,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
        ],
        ErrorCategory.TIMEOUT: [
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
            RecoveryStrategy.SKIP_AND_CONTINUE,
        ],
        ErrorCategory.TOOL_EXECUTION_ERROR: [
            RecoveryStrategy.ANALYZE_AND_FIX,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
            RecoveryStrategy.SKIP_AND_CONTINUE,
        ],
        ErrorCategory.PARSING_ERROR: [
            RecoveryStrategy.ANALYZE_AND_FIX,
            RecoveryStrategy.RESET_AND_REGENERATE,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
        ],
        ErrorCategory.GENERATION_ERROR: [
            RecoveryStrategy.RESET_AND_REGENERATE,
            RecoveryStrategy.ANALYZE_AND_FIX,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
        ],
    }
    
    DEFAULT_STRATEGIES = [
        RecoveryStrategy.ANALYZE_AND_FIX,
        RecoveryStrategy.FALLBACK_ALTERNATIVE,
        RecoveryStrategy.RESET_AND_REGENERATE,
    ]
    
    @classmethod
    def get_prioritized_strategies(
        cls,
        error_category: ErrorCategory,
        max_strategies: int = 3
    ) -> List[RecoveryStrategy]:
        """Get prioritized strategies for an error category.
        
        Args:
            error_category: Category of the error
            max_strategies: Maximum number of strategies to return
            
        Returns:
            List of prioritized strategies
        """
        strategies = cls.STRATEGY_PRIORITY_MAP.get(
            error_category,
            cls.DEFAULT_STRATEGIES
        )
        return strategies[:max_strategies]


def create_parallel_recovery_manager(
    max_parallel: int = 3,
    timeout: float = 60.0
) -> ParallelRecoveryManager:
    """Create a parallel recovery manager.
    
    Args:
        max_parallel: Maximum parallel strategies
        timeout: Default timeout
        
    Returns:
        Configured ParallelRecoveryManager
    """
    return ParallelRecoveryManager(
        max_parallel=max_parallel,
        default_timeout=timeout
    )
