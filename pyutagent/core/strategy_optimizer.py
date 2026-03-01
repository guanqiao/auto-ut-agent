"""Adaptive strategy optimization for error recovery.

This module provides adaptive strategy adjustment:
- Strategy effectiveness tracking
- Dynamic weight adjustment
- Context-aware strategy selection
- Performance optimization
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .error_recovery import ErrorCategory, RecoveryStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy: RecoveryStrategy
    total_uses: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    total_attempts: int = 0
    last_used: Optional[datetime] = None
    last_success: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.total_uses if self.total_uses > 0 else 0.0
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.total_uses if self.total_uses > 0 else 0.0
    
    @property
    def avg_attempts(self) -> float:
        return self.total_attempts / self.total_uses if self.total_uses > 0 else 0.0


@dataclass
class StrategyWeights:
    """Weights for strategy selection."""
    base_weight: float = 1.0
    success_weight: float = 0.4
    time_weight: float = 0.2
    attempt_weight: float = 0.2
    recency_weight: float = 0.2


@dataclass
class OptimizationResult:
    """Result of strategy optimization."""
    recommended_strategy: RecoveryStrategy
    confidence: float
    expected_success_rate: float
    expected_time: float
    alternatives: List[Tuple[RecoveryStrategy, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyOptimizer:
    """Optimizes strategy selection based on performance history.
    
    Features:
    - Performance tracking
    - Dynamic weight adjustment
    - Context-aware selection
    - Continuous learning
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        decay_rate: float = 0.95,
        min_samples: int = 5
    ):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.min_samples = min_samples
        
        self._performance: Dict[Tuple[ErrorCategory, RecoveryStrategy], StrategyPerformance] = {}
        self._weights = StrategyWeights()
        self._category_preferences: Dict[ErrorCategory, List[RecoveryStrategy]] = {}
        
        self._initialize_default_preferences()
    
    def _initialize_default_preferences(self):
        """Initialize default strategy preferences per category."""
        self._category_preferences = {
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
                RecoveryStrategy.SKIP_AND_CONTINUE,
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
    
    def record_result(
        self,
        error_category: ErrorCategory,
        strategy: RecoveryStrategy,
        success: bool,
        time_taken: float,
        attempts: int
    ):
        """Record a strategy execution result.
        
        Args:
            error_category: Category of the error
            strategy: Strategy used
            success: Whether recovery succeeded
            time_taken: Time taken for recovery
            attempts: Number of attempts needed
        """
        key = (error_category, strategy)
        
        if key not in self._performance:
            self._performance[key] = StrategyPerformance(strategy=strategy)
        
        perf = self._performance[key]
        perf.total_uses += 1
        perf.total_time += time_taken
        perf.total_attempts += attempts
        perf.last_used = datetime.now()
        
        if success:
            perf.successes += 1
            perf.last_success = datetime.now()
        else:
            perf.failures += 1
        
        self._update_weights(error_category, strategy, success, time_taken)
        
        logger.debug(
            f"[StrategyOptimizer] Recorded {strategy.name} for {error_category.name}: "
            f"{'success' if success else 'failure'}, time={time_taken:.2f}s"
        )
    
    def _update_weights(
        self,
        error_category: ErrorCategory,
        strategy: RecoveryStrategy,
        success: bool,
        time_taken: float
    ):
        """Update weights based on result."""
        pass
    
    def optimize_strategy_selection(
        self,
        error_category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Optimize strategy selection for an error category.
        
        Args:
            error_category: Category of the error
            context: Additional context
            
        Returns:
            OptimizationResult with recommended strategy
        """
        candidates = self._get_candidate_strategies(error_category)
        
        if not candidates:
            return OptimizationResult(
                recommended_strategy=RecoveryStrategy.ANALYZE_AND_FIX,
                confidence=0.3,
                expected_success_rate=0.5,
                expected_time=30.0,
                alternatives=[],
                metadata={"reason": "No candidates available"}
            )
        
        scored_strategies: List[Tuple[RecoveryStrategy, float, Dict[str, float]]] = []
        
        for strategy in candidates:
            score, components = self._score_strategy(error_category, strategy, context)
            scored_strategies.append((strategy, score, components))
        
        scored_strategies.sort(key=lambda x: x[1], reverse=True)
        
        best_strategy, best_score, best_components = scored_strategies[0]
        
        key = (error_category, best_strategy)
        perf = self._performance.get(key)
        
        expected_success = perf.success_rate if perf else 0.5
        expected_time = perf.avg_time if perf else 30.0
        
        confidence = self._calculate_confidence(error_category, best_strategy)
        
        alternatives = [
            (s, score)
            for s, score, _ in scored_strategies[1:4]
        ]
        
        logger.info(
            f"[StrategyOptimizer] Recommended {best_strategy.name} for {error_category.name} "
            f"(score={best_score:.2f}, confidence={confidence:.2f})"
        )
        
        return OptimizationResult(
            recommended_strategy=best_strategy,
            confidence=confidence,
            expected_success_rate=expected_success,
            expected_time=expected_time,
            alternatives=alternatives,
            metadata={
                "score_components": best_components,
                "total_candidates": len(candidates)
            }
        )
    
    def _get_candidate_strategies(
        self,
        error_category: ErrorCategory
    ) -> List[RecoveryStrategy]:
        """Get candidate strategies for an error category."""
        if error_category in self._category_preferences:
            return self._category_preferences[error_category]
        
        return [
            RecoveryStrategy.ANALYZE_AND_FIX,
            RecoveryStrategy.FALLBACK_ALTERNATIVE,
            RecoveryStrategy.RESET_AND_REGENERATE,
        ]
    
    def _score_strategy(
        self,
        error_category: ErrorCategory,
        strategy: RecoveryStrategy,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        """Score a strategy for selection.
        
        Returns:
            Tuple of (total_score, score_components)
        """
        key = (error_category, strategy)
        perf = self._performance.get(key)
        
        components = {}
        
        if perf and perf.total_uses >= self.min_samples:
            components["success"] = perf.success_rate * self._weights.success_weight
            
            time_score = 1.0 / (1.0 + perf.avg_time / 60.0)
            components["time"] = time_score * self._weights.time_weight
            
            attempt_score = 1.0 / perf.avg_attempts if perf.avg_attempts > 0 else 0.5
            components["attempt"] = attempt_score * self._weights.attempt_weight
            
            if perf.last_success:
                hours_ago = (datetime.now() - perf.last_success).total_seconds() / 3600
                recency_score = math.exp(-hours_ago / 24)
                components["recency"] = recency_score * self._weights.recency_weight
            else:
                components["recency"] = 0.0
        else:
            components["success"] = 0.5 * self._weights.success_weight
            components["time"] = 0.5 * self._weights.time_weight
            components["attempt"] = 0.5 * self._weights.attempt_weight
            components["recency"] = 0.5 * self._weights.recency_weight
        
        candidates = self._get_candidate_strategies(error_category)
        if strategy in candidates:
            position = candidates.index(strategy)
            position_score = 1.0 - (position / len(candidates))
            components["position"] = position_score * 0.2
        else:
            components["position"] = 0.0
        
        total = sum(components.values()) + self._weights.base_weight
        
        return total, components
    
    def _calculate_confidence(
        self,
        error_category: ErrorCategory,
        strategy: RecoveryStrategy
    ) -> float:
        """Calculate confidence in strategy recommendation."""
        key = (error_category, strategy)
        perf = self._performance.get(key)
        
        if not perf or perf.total_uses < self.min_samples:
            return 0.3
        
        sample_confidence = min(1.0, perf.total_uses / (self.min_samples * 2))
        
        success_confidence = abs(perf.success_rate - 0.5) * 2
        
        return (sample_confidence + success_confidence) / 2
    
    def get_performance_stats(
        self,
        error_category: Optional[ErrorCategory] = None
    ) -> Dict[str, Any]:
        """Get performance statistics.
        
        Args:
            error_category: Optional category filter
            
        Returns:
            Performance statistics
        """
        stats = {}
        
        for (cat, strategy), perf in self._performance.items():
            if error_category and cat != error_category:
                continue
            
            key = f"{cat.name}:{strategy.name}"
            stats[key] = {
                "total_uses": perf.total_uses,
                "successes": perf.successes,
                "failures": perf.failures,
                "success_rate": perf.success_rate,
                "avg_time": perf.avg_time,
                "avg_attempts": perf.avg_attempts,
                "last_used": perf.last_used.isoformat() if perf.last_used else None,
            }
        
        return stats
    
    def update_weights(self, new_weights: StrategyWeights):
        """Update strategy weights."""
        self._weights = new_weights
        logger.info(f"[StrategyOptimizer] Updated weights: {new_weights}")
    
    def reset_performance(self):
        """Reset all performance data."""
        self._performance.clear()
        logger.info("[StrategyOptimizer] Reset all performance data")


def create_strategy_optimizer(
    learning_rate: float = 0.1,
    min_samples: int = 5
) -> StrategyOptimizer:
    """Create a StrategyOptimizer instance.
    
    Args:
        learning_rate: Learning rate for weight updates
        min_samples: Minimum samples before optimization
        
    Returns:
        Configured StrategyOptimizer
    """
    return StrategyOptimizer(
        learning_rate=learning_rate,
        min_samples=min_samples
    )
