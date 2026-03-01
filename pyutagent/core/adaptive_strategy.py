"""Adaptive strategy manager for dynamic strategy optimization.

This module provides adaptive strategy selection capabilities:
- Strategy effectiveness tracking
- Dynamic weight adjustment
- Context-aware strategy selection
- Multi-strategy ensemble
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict

from ..core.error_recovery import RecoveryStrategy, ErrorCategory

logger = logging.getLogger(__name__)


class StrategyEffectiveness(Enum):
    """Effectiveness rating for strategies."""
    EXCELLENT = auto()  # > 90% success
    GOOD = auto()       # 70-90% success
    MODERATE = auto()   # 50-70% success
    POOR = auto()       # 30-50% success
    BAD = auto()        # < 30% success


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_name: str
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    avg_execution_time_ms: float = 0.0
    last_used: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts
    
    @property
    def effectiveness(self) -> StrategyEffectiveness:
        rate = self.success_rate
        if rate > 0.9:
            return StrategyEffectiveness.EXCELLENT
        elif rate > 0.7:
            return StrategyEffectiveness.GOOD
        elif rate > 0.5:
            return StrategyEffectiveness.MODERATE
        elif rate > 0.3:
            return StrategyEffectiveness.POOR
        else:
            return StrategyEffectiveness.BAD
    
    def record_attempt(self, success: bool, execution_time_ms: float):
        """Record a strategy attempt."""
        self.total_attempts += 1
        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
        
        # Update average execution time
        self.avg_execution_time_ms = (
            (self.avg_execution_time_ms * (self.total_attempts - 1) + execution_time_ms)
            / self.total_attempts
        )
        
        self.last_used = datetime.now().isoformat()


@dataclass
class ContextFeatures:
    """Features extracted from error context."""
    error_category: ErrorCategory
    error_message_length: int
    has_stack_trace: bool
    code_complexity: int  # Lines of code
    iteration_count: int
    previous_strategies_tried: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_category": self.error_category.name,
            "error_message_length": self.error_message_length,
            "has_stack_trace": self.has_stack_trace,
            "code_complexity": self.code_complexity,
            "iteration_count": self.iteration_count,
            "previous_strategies_tried": self.previous_strategies_tried
        }


@dataclass
class StrategySelection:
    """Result of strategy selection."""
    selected_strategy: RecoveryStrategy
    confidence: float
    alternative_strategies: List[Tuple[RecoveryStrategy, float]]
    reasoning: str


class AdaptiveStrategyManager:
    """Adaptive strategy manager for dynamic strategy optimization.
    
    Features:
    - Tracks strategy effectiveness over time
    - Dynamically adjusts strategy weights
    - Context-aware strategy selection
    - Supports multi-strategy ensemble
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize adaptive strategy manager.
        
        Args:
            db_path: Path to SQLite database for persistence
        """
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "adaptive_strategy.db"
        
        self.db_path = str(db_path)
        self._init_database()
        
        # In-memory cache
        self._performance_cache: Dict[str, StrategyPerformance] = {}
        self._context_weights: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Exploration vs exploitation
        self.exploration_rate = 0.2  # 20% exploration
        self.min_attempts_before_exploit = 5
        
        logger.info(f"[AdaptiveStrategyManager] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy_name TEXT PRIMARY KEY,
                    total_attempts INTEGER DEFAULT 0,
                    successful_attempts INTEGER DEFAULT 0,
                    failed_attempts INTEGER DEFAULT 0,
                    avg_execution_time_ms REAL DEFAULT 0.0,
                    last_used TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy attempts table (detailed history)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    error_category TEXT,
                    success BOOLEAN,
                    execution_time_ms REAL,
                    context_features TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Context weights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_weights (
                    context_key TEXT PRIMARY KEY,
                    weights TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def select_strategy(
        self,
        error_category: ErrorCategory,
        available_strategies: List[RecoveryStrategy],
        context: Optional[ContextFeatures] = None,
        allow_exploration: bool = True
    ) -> StrategySelection:
        """Select the best strategy based on historical performance and context.
        
        Args:
            error_category: Category of the error
            available_strategies: List of available strategies
            context: Optional context features
            allow_exploration: Whether to allow exploration
            
        Returns:
            StrategySelection with selected strategy and alternatives
        """
        if not available_strategies:
            raise ValueError("No strategies available")
        
        # Get performance scores for each strategy
        strategy_scores = []
        
        for strategy in available_strategies:
            perf = self._get_performance(strategy.name)
            
            # Base score from historical performance
            base_score = perf.success_rate
            
            # Adjust score based on context
            context_score = self._calculate_context_score(strategy, error_category, context)
            
            # Combine scores
            combined_score = (base_score * 0.6) + (context_score * 0.4)
            
            strategy_scores.append((strategy, combined_score, perf))
        
        # Sort by score
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Exploration: occasionally try less effective strategies
        if allow_exploration and len(strategy_scores) > 1:
            import random
            if random.random() < self.exploration_rate:
                # Swap top strategy with a random one
                idx = random.randint(1, len(strategy_scores) - 1)
                strategy_scores[0], strategy_scores[idx] = strategy_scores[idx], strategy_scores[0]
                logger.debug(f"[AdaptiveStrategyManager] Exploration: trying {strategy_scores[0][0].name}")
        
        # Select top strategy
        selected = strategy_scores[0]
        alternatives = [(s, score) for s, score, _ in strategy_scores[1:4]]
        
        # Build reasoning
        reasoning = self._build_reasoning(selected, strategy_scores, context)
        
        selection = StrategySelection(
            selected_strategy=selected[0],
            confidence=selected[1],
            alternative_strategies=alternatives,
            reasoning=reasoning
        )
        
        logger.info(f"[AdaptiveStrategyManager] Selected strategy: {selected[0].name} "
                   f"(confidence: {selected[1]:.2f})")
        
        return selection
    
    def select_strategies_ensemble(
        self,
        error_category: ErrorCategory,
        available_strategies: List[RecoveryStrategy],
        context: Optional[ContextFeatures] = None,
        top_k: int = 3
    ) -> List[Tuple[RecoveryStrategy, float]]:
        """Select top-k strategies for ensemble execution.
        
        Args:
            error_category: Category of the error
            available_strategies: List of available strategies
            context: Optional context features
            top_k: Number of strategies to select
            
        Returns:
            List of (strategy, confidence) tuples
        """
        strategy_scores = []
        
        for strategy in available_strategies:
            perf = self._get_performance(strategy.name)
            context_score = self._calculate_context_score(strategy, error_category, context)
            combined_score = (perf.success_rate * 0.6) + (context_score * 0.4)
            
            strategy_scores.append((strategy, combined_score))
        
        # Sort and return top-k
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        return strategy_scores[:top_k]
    
    def record_attempt(
        self,
        strategy: RecoveryStrategy,
        error_category: ErrorCategory,
        success: bool,
        execution_time_ms: float,
        context: Optional[ContextFeatures] = None
    ):
        """Record the result of a strategy attempt.
        
        Args:
            strategy: The strategy that was attempted
            error_category: Category of the error
            success: Whether the attempt was successful
            execution_time_ms: Execution time in milliseconds
            context: Optional context features
        """
        # Update in-memory cache
        perf = self._get_performance(strategy.name)
        perf.record_attempt(success, execution_time_ms)
        
        # Persist to database
        self._persist_performance(strategy.name, perf)
        
        # Record detailed attempt
        self._persist_attempt(strategy, error_category, success, execution_time_ms, context)
        
        # Update context weights
        if context:
            self._update_context_weights(strategy, error_category, context, success)
        
        logger.debug(f"[AdaptiveStrategyManager] Recorded attempt for {strategy.name}: "
                    f"success={success}, rate={perf.success_rate:.2f}")
    
    def get_strategy_stats(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for strategies.
        
        Args:
            strategy_name: Specific strategy or None for all
            
        Returns:
            Statistics dictionary
        """
        if strategy_name:
            perf = self._get_performance(strategy_name)
            return {
                "strategy_name": strategy_name,
                "total_attempts": perf.total_attempts,
                "success_rate": perf.success_rate,
                "effectiveness": perf.effectiveness.name,
                "avg_execution_time_ms": perf.avg_execution_time_ms,
                "last_used": perf.last_used
            }
        
        # Get all strategies
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT strategy_name FROM strategy_performance')
            strategies = [row[0] for row in cursor.fetchall()]
        
        return {
            name: self.get_strategy_stats(name)
            for name in strategies
        }
    
    def get_recommended_strategies(
        self,
        error_category: ErrorCategory,
        min_success_rate: float = 0.5
    ) -> List[str]:
        """Get list of recommended strategies for an error category.
        
        Args:
            error_category: Category of error
            min_success_rate: Minimum success rate threshold
            
        Returns:
            List of recommended strategy names
        """
        recommended = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT strategy_name, 
                       CAST(successful_attempts AS REAL) / NULLIF(total_attempts, 0) as rate
                FROM strategy_performance
                WHERE total_attempts >= ?
                ORDER BY rate DESC
            ''', (self.min_attempts_before_exploit,))
            
            for row in cursor.fetchall():
                if row[1] >= min_success_rate:
                    recommended.append(row[0])
        
        return recommended
    
    def reset_strategy_weights(self, strategy_name: Optional[str] = None):
        """Reset weights for a strategy or all strategies.
        
        Args:
            strategy_name: Specific strategy or None for all
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if strategy_name:
                cursor.execute('''
                    UPDATE strategy_performance
                    SET total_attempts = 0,
                        successful_attempts = 0,
                        failed_attempts = 0,
                        avg_execution_time_ms = 0.0
                    WHERE strategy_name = ?
                ''', (strategy_name,))
                
                if strategy_name in self._performance_cache:
                    del self._performance_cache[strategy_name]
            else:
                cursor.execute('''
                    UPDATE strategy_performance
                    SET total_attempts = 0,
                        successful_attempts = 0,
                        failed_attempts = 0,
                        avg_execution_time_ms = 0.0
                ''')
                
                self._performance_cache.clear()
            
            conn.commit()
        
        logger.info(f"[AdaptiveStrategyManager] Reset weights for {strategy_name or 'all strategies'}")
    
    def _get_performance(self, strategy_name: str) -> StrategyPerformance:
        """Get performance metrics for a strategy (from cache or DB)."""
        if strategy_name in self._performance_cache:
            return self._performance_cache[strategy_name]
        
        # Load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT total_attempts, successful_attempts, failed_attempts,
                       avg_execution_time_ms, last_used, created_at
                FROM strategy_performance
                WHERE strategy_name = ?
            ''', (strategy_name,))
            
            row = cursor.fetchone()
            
            if row:
                perf = StrategyPerformance(
                    strategy_name=strategy_name,
                    total_attempts=row[0],
                    successful_attempts=row[1],
                    failed_attempts=row[2],
                    avg_execution_time_ms=row[3],
                    last_used=row[4],
                    created_at=row[5]
                )
            else:
                # Create new performance record
                perf = StrategyPerformance(strategy_name=strategy_name)
                self._persist_performance(strategy_name, perf)
            
            self._performance_cache[strategy_name] = perf
            return perf
    
    def _persist_performance(self, strategy_name: str, perf: StrategyPerformance):
        """Persist performance metrics to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO strategy_performance
                (strategy_name, total_attempts, successful_attempts, failed_attempts,
                 avg_execution_time_ms, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                strategy_name,
                perf.total_attempts,
                perf.successful_attempts,
                perf.failed_attempts,
                perf.avg_execution_time_ms,
                perf.last_used
            ))
            conn.commit()
    
    def _persist_attempt(
        self,
        strategy: RecoveryStrategy,
        error_category: ErrorCategory,
        success: bool,
        execution_time_ms: float,
        context: Optional[ContextFeatures]
    ):
        """Persist detailed attempt to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO strategy_attempts
                (strategy_name, error_category, success, execution_time_ms, context_features)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                strategy.name,
                error_category.name,
                success,
                execution_time_ms,
                json.dumps(context.to_dict()) if context else None
            ))
            conn.commit()
    
    def _calculate_context_score(
        self,
        strategy: RecoveryStrategy,
        error_category: ErrorCategory,
        context: Optional[ContextFeatures]
    ) -> float:
        """Calculate context-adjusted score for a strategy."""
        if not context:
            return 0.5  # Neutral score
        
        # Build context key
        context_key = f"{error_category.name}_{strategy.name}"
        
        # Get context-specific weight
        if context_key in self._context_weights:
            weights = self._context_weights[context_key]
            
            # Calculate score based on context features
            score = 0.5  # Base score
            
            # Adjust based on iteration count (prefer different strategies in later iterations)
            if context.iteration_count > 2:
                score -= 0.1 * min(context.iteration_count - 2, 3)
            
            # Adjust based on previously tried strategies
            if strategy.name in context.previous_strategies_tried:
                score -= 0.2  # Penalty for retrying same strategy
            
            # Apply learned weight
            learned_weight = weights.get('weight', 0.5)
            score = (score * 0.5) + (learned_weight * 0.5)
            
            return max(0.0, min(1.0, score))
        
        return 0.5  # Default neutral score
    
    def _update_context_weights(
        self,
        strategy: RecoveryStrategy,
        error_category: ErrorCategory,
        context: ContextFeatures,
        success: bool
    ):
        """Update context-specific weights based on outcome."""
        context_key = f"{error_category.name}_{strategy.name}"
        
        # Get current weight
        current_weight = self._context_weights[context_key].get('weight', 0.5)
        
        # Update weight using simple moving average
        learning_rate = 0.1
        new_weight = current_weight + learning_rate * (1.0 if success else -1.0)
        new_weight = max(0.0, min(1.0, new_weight))
        
        self._context_weights[context_key]['weight'] = new_weight
        
        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO context_weights
                (context_key, weights, updated_at)
                VALUES (?, ?, ?)
            ''', (
                context_key,
                json.dumps(self._context_weights[context_key]),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def _build_reasoning(
        self,
        selected: Tuple[RecoveryStrategy, float, StrategyPerformance],
        all_scores: List[Tuple[RecoveryStrategy, float, StrategyPerformance]],
        context: Optional[ContextFeatures]
    ) -> str:
        """Build human-readable reasoning for strategy selection."""
        strategy, score, perf = selected
        
        lines = [
            f"Selected '{strategy.name}' based on:",
            f"  - Historical success rate: {perf.success_rate:.1%} ({perf.successful_attempts}/{perf.total_attempts} attempts)",
            f"  - Effectiveness rating: {perf.effectiveness.name}",
            f"  - Combined confidence score: {score:.2f}",
        ]
        
        if context:
            lines.append(f"  - Error category: {context.error_category.name}")
            lines.append(f"  - Iteration: {context.iteration_count}")
        
        if len(all_scores) > 1:
            lines.append(f"\nAlternatives considered:")
            for alt_strategy, alt_score, alt_perf in all_scores[1:3]:
                lines.append(f"  - {alt_strategy.name}: {alt_score:.2f} (success rate: {alt_perf.success_rate:.1%})")
        
        return "\n".join(lines)


def create_adaptive_strategy_manager(db_path: Optional[str] = None) -> AdaptiveStrategyManager:
    """Create an adaptive strategy manager instance.
    
    Args:
        db_path: Optional path to database
        
    Returns:
        Configured AdaptiveStrategyManager
    """
    return AdaptiveStrategyManager(db_path=db_path)
