"""Enhanced Feedback Loop for intelligent learning and adaptation.

This module provides advanced feedback mechanisms including execution analysis,
failure learning, and adaptive strategy adjustment.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict, Counter
import uuid

from .component_registry import SimpleComponent, component


logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback."""
    COMPILATION_SUCCESS = "compilation_success"
    COMPILATION_FAILURE = "compilation_failure"
    TEST_PASS = "test_pass"
    TEST_FAILURE = "test_failure"
    COVERAGE_IMPROVEMENT = "coverage_improvement"
    COVERAGE_REGRESSION = "coverage_regression"
    ERROR_FIXED = "error_fixed"
    ERROR_RECURRING = "error_recurring"
    STRATEGY_EFFECTIVE = "strategy_effective"
    STRATEGY_INEFFECTIVE = "strategy_ineffective"


class LearningCategory(Enum):
    """Categories of learning."""
    ERROR_PATTERN = "error_pattern"
    SUCCESSFUL_PATTERN = "successful_pattern"
    OPTIMAL_STRATEGY = "optimal_strategy"
    AVOIDED_APPROACH = "avoided_approach"
    CONTEXT_MATCH = "context_match"


@dataclass
class FeedbackEvent:
    """A feedback event from test execution."""
    event_id: str
    feedback_type: FeedbackType
    context: Dict[str, Any]
    outcome: str
    details: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "feedback_type": self.feedback_type.value,
            "context": self.context,
            "outcome": self.outcome,
            "details": self.details,
            "timestamp": self.timestamp,
            "session_id": self.session_id
        }


@dataclass
class LearningInsight:
    """An insight learned from feedback."""
    insight_id: str
    category: LearningCategory
    pattern: str
    conditions: List[str]
    recommendation: str
    confidence: float
    occurrence_count: int = 1
    success_rate: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveAdjustment:
    """An adjustment to make based on learning."""
    adjustment_id: str
    target: str
    action: str
    reason: str
    priority: int
    confidence: float
    applied: bool = False


@component(
    component_id="enhanced_feedback_loop",
    dependencies=[],
    description="Enhanced feedback loop for intelligent learning and adaptation"
)
class EnhancedFeedbackLoop(SimpleComponent):
    """Enhanced feedback loop for intelligent learning.
    
    Features:
    - Event collection and analysis
    - Pattern learning from successes and failures
    - Adaptive strategy adjustment
    - Knowledge persistence
    - Insight generation
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize enhanced feedback loop.
        
        Args:
            db_path: Path to SQLite database
        """
        super().__init__()
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "feedback_loop.db"
        
        self.db_path = str(db_path)
        self._init_database()
        
        self._event_buffer: List[FeedbackEvent] = []
        self._insight_cache: Dict[str, LearningInsight] = {}
        self._pattern_counts: Counter = Counter()
        
        logger.info(f"[EnhancedFeedbackLoop] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback_events (
                    event_id TEXT PRIMARY KEY,
                    feedback_type TEXT NOT NULL,
                    context TEXT,
                    outcome TEXT,
                    details TEXT,
                    timestamp TEXT,
                    session_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_insights (
                    insight_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    conditions TEXT,
                    recommendation TEXT,
                    confidence REAL,
                    occurrence_count INTEGER DEFAULT 1,
                    success_rate REAL DEFAULT 0.0,
                    last_updated TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    error_type TEXT NOT NULL,
                    error_message_pattern TEXT,
                    solution TEXT,
                    occurrence_count INTEGER DEFAULT 1,
                    success_count INTEGER DEFAULT 0,
                    last_seen TEXT
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON feedback_events(feedback_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_session ON feedback_events(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_insight_category ON learning_insights(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type)')
            
            conn.commit()
    
    def record_feedback(
        self,
        feedback_type: FeedbackType,
        context: Dict[str, Any],
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Record a feedback event.
        
        Args:
            feedback_type: Type of feedback
            context: Context information
            outcome: Outcome description
            details: Additional details
            session_id: Optional session ID
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        event = FeedbackEvent(
            event_id=event_id,
            feedback_type=feedback_type,
            context=context,
            outcome=outcome,
            details=details or {},
            session_id=session_id
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO feedback_events
                (event_id, feedback_type, context, outcome, details, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                feedback_type.value,
                json.dumps(context),
                outcome,
                json.dumps(details or {}),
                event.timestamp,
                session_id
            ))
            conn.commit()
        
        self._event_buffer.append(event)
        
        if len(self._event_buffer) >= 10:
            self._process_buffer()
        
        logger.debug(f"[EnhancedFeedbackLoop] Recorded feedback: {feedback_type.value}")
        return event_id
    
    def record_compilation_result(
        self,
        success: bool,
        errors: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> None:
        """Record compilation result."""
        feedback_type = FeedbackType.COMPILATION_SUCCESS if success else FeedbackType.COMPILATION_FAILURE
        
        self.record_feedback(
            feedback_type=feedback_type,
            context=context,
            outcome="success" if success else "failed",
            details={"errors": errors}
        )
        
        if not success:
            for error in errors:
                self._learn_error_pattern(error, context)
    
    def record_test_result(
        self,
        test_name: str,
        passed: bool,
        failure_reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record test execution result."""
        feedback_type = FeedbackType.TEST_PASS if passed else FeedbackType.TEST_FAILURE
        
        self.record_feedback(
            feedback_type=feedback_type,
            context=context or {},
            outcome=test_name,
            details={"failure_reason": failure_reason} if failure_reason else {}
        )
        
        if not passed and failure_reason:
            self._learn_from_failure(failure_reason, context or {})
    
    def record_coverage_change(
        self,
        old_coverage: float,
        new_coverage: float,
        context: Dict[str, Any]
    ) -> None:
        """Record coverage change."""
        if new_coverage > old_coverage:
            feedback_type = FeedbackType.COVERAGE_IMPROVEMENT
        else:
            feedback_type = FeedbackType.COVERAGE_REGRESSION
        
        self.record_feedback(
            feedback_type=feedback_type,
            context=context,
            outcome=f"coverage: {old_coverage:.2%} -> {new_coverage:.2%}",
            details={"old_coverage": old_coverage, "new_coverage": new_coverage}
        )
    
    def get_adaptive_adjustments(
        self,
        context: Dict[str, Any]
    ) -> List[AdaptiveAdjustment]:
        """Get adaptive adjustments based on context and learning.
        
        Args:
            context: Current context
            
        Returns:
            List of recommended adjustments
        """
        adjustments = []
        
        insights = self._get_relevant_insights(context)
        
        for insight in insights:
            if insight.category == LearningCategory.ERROR_PATTERN:
                adjustments.append(AdaptiveAdjustment(
                    adjustment_id=str(uuid.uuid4()),
                    target="error_prevention",
                    action=insight.recommendation,
                    reason=f"Pattern: {insight.pattern}",
                    priority=1,
                    confidence=insight.confidence
                ))
            
            elif insight.category == LearningCategory.OPTIMAL_STRATEGY:
                adjustments.append(AdaptiveAdjustment(
                    adjustment_id=str(uuid.uuid4()),
                    target="strategy_selection",
                    action=insight.recommendation,
                    reason=f"Optimal for: {', '.join(insight.conditions)}",
                    priority=2,
                    confidence=insight.confidence
                ))
            
            elif insight.category == LearningCategory.AVOIDED_APPROACH:
                adjustments.append(AdaptiveAdjustment(
                    adjustment_id=str(uuid.uuid4()),
                    target="approach_avoidance",
                    action=f"Avoid: {insight.recommendation}",
                    reason=f"Pattern: {insight.pattern}",
                    priority=1,
                    confidence=insight.confidence
                ))
        
        error_patterns = self._get_relevant_error_patterns(context)
        for pattern in error_patterns:
            if pattern.get("solution"):
                adjustments.append(AdaptiveAdjustment(
                    adjustment_id=str(uuid.uuid4()),
                    target="error_fix",
                    action=pattern["solution"],
                    reason=f"Known error pattern: {pattern['error_type']}",
                    priority=1,
                    confidence=pattern.get("success_rate", 0.5)
                ))
        
        adjustments.sort(key=lambda a: (a.priority, -a.confidence))
        
        return adjustments[:5]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM feedback_events')
            total_events = cursor.fetchone()[0]
            
            cursor.execute('SELECT feedback_type, COUNT(*) FROM feedback_events GROUP BY feedback_type')
            event_counts = dict(cursor.fetchall())
            
            cursor.execute('SELECT COUNT(*) FROM learning_insights')
            total_insights = cursor.fetchone()[0]
            
            cursor.execute('SELECT category, COUNT(*) FROM learning_insights GROUP BY category')
            insight_counts = dict(cursor.fetchall())
            
            cursor.execute('SELECT COUNT(*) FROM error_patterns')
            total_patterns = cursor.fetchone()[0]
            
            success_rate = 0.0
            if total_events > 0:
                success_events = event_counts.get("test_pass", 0) + event_counts.get("compilation_success", 0)
                total_test_events = success_events + event_counts.get("test_failure", 0) + event_counts.get("compilation_failure", 0)
                if total_test_events > 0:
                    success_rate = success_events / total_test_events
            
            return {
                "total_events": total_events,
                "event_distribution": event_counts,
                "total_insights": total_insights,
                "insight_distribution": insight_counts,
                "total_error_patterns": total_patterns,
                "overall_success_rate": success_rate,
                "buffer_size": len(self._event_buffer)
            }
    
    def _process_buffer(self):
        """Process buffered events for learning."""
        if not self._event_buffer:
            return
        
        for event in self._event_buffer:
            self._extract_insights(event)
        
        self._event_buffer.clear()
        
        logger.debug("[EnhancedFeedbackLoop] Processed event buffer")
    
    def _extract_insights(self, event: FeedbackEvent):
        """Extract learning insights from an event."""
        if event.feedback_type == FeedbackType.TEST_FAILURE:
            self._learn_from_failure(event.outcome, event.context)
        
        elif event.feedback_type == FeedbackType.TEST_PASS:
            self._learn_from_success(event.context)
        
        elif event.feedback_type == FeedbackType.STRATEGY_EFFECTIVE:
            self._learn_strategy_effectiveness(event.context, True)
        
        elif event.feedback_type == FeedbackType.STRATEGY_INEFFECTIVE:
            self._learn_strategy_effectiveness(event.context, False)
    
    def _learn_from_failure(self, failure_reason: str, context: Dict[str, Any]):
        """Learn from a failure."""
        error_type = self._classify_error(failure_reason)
        
        pattern_key = f"{error_type}:{failure_reason[:50]}"
        self._pattern_counts[pattern_key] += 1
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT pattern_id, occurrence_count FROM error_patterns WHERE error_type = ?',
                (error_type,)
            )
            row = cursor.fetchone()
            
            if row:
                cursor.execute('''
                    UPDATE error_patterns 
                    SET occurrence_count = ?, last_seen = ?
                    WHERE pattern_id = ?
                ''', (row[1] + 1, datetime.now().isoformat(), row[0]))
            else:
                pattern_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO error_patterns
                    (pattern_id, error_type, error_message_pattern, last_seen)
                    VALUES (?, ?, ?, ?)
                ''', (pattern_id, error_type, failure_reason[:200], datetime.now().isoformat()))
            
            conn.commit()
    
    def _learn_from_success(self, context: Dict[str, Any]):
        """Learn from a success."""
        strategy = context.get("strategy", "unknown")
        approach = context.get("approach", "unknown")
        
        insight_key = f"success:{strategy}:{approach}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT insight_id, occurrence_count, success_rate FROM learning_insights WHERE pattern = ?',
                (insight_key,)
            )
            row = cursor.fetchone()
            
            if row:
                new_count = row[1] + 1
                new_rate = row[2] + (1.0 - row[2]) / new_count
                cursor.execute('''
                    UPDATE learning_insights 
                    SET occurrence_count = ?, success_rate = ?, last_updated = ?
                    WHERE insight_id = ?
                ''', (new_count, new_rate, datetime.now().isoformat(), row[0]))
            else:
                insight_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO learning_insights
                    (insight_id, category, pattern, conditions, recommendation, confidence, success_rate, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    insight_id,
                    LearningCategory.SUCCESSFUL_PATTERN.value,
                    insight_key,
                    json.dumps(context.get("conditions", [])),
                    f"Use {strategy} approach for similar contexts",
                    0.8,
                    1.0,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
    
    def _learn_error_pattern(self, error: Dict[str, Any], context: Dict[str, Any]):
        """Learn from an error pattern."""
        error_type = error.get("type", "unknown")
        error_message = error.get("message", "")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT pattern_id, occurrence_count FROM error_patterns WHERE error_type = ? AND error_message_pattern = ?',
                (error_type, error_message[:200])
            )
            row = cursor.fetchone()
            
            if row:
                cursor.execute('''
                    UPDATE error_patterns 
                    SET occurrence_count = ?, last_seen = ?
                    WHERE pattern_id = ?
                ''', (row[1] + 1, datetime.now().isoformat(), row[0]))
            else:
                pattern_id = str(uuid.uuid4())
                cursor.execute('''
                    INSERT INTO error_patterns
                    (pattern_id, error_type, error_message_pattern, last_seen)
                    VALUES (?, ?, ?, ?)
                ''', (pattern_id, error_type, error_message[:200], datetime.now().isoformat()))
            
            conn.commit()
    
    def _learn_strategy_effectiveness(self, context: Dict[str, Any], effective: bool):
        """Learn about strategy effectiveness."""
        strategy = context.get("strategy", "unknown")
        conditions = context.get("conditions", [])
        
        insight_key = f"strategy:{strategy}"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT insight_id, occurrence_count, success_rate FROM learning_insights WHERE pattern = ?',
                (insight_key,)
            )
            row = cursor.fetchone()
            
            if row:
                new_count = row[1] + 1
                current_rate = row[2]
                new_rate = current_rate + ((1.0 if effective else 0.0) - current_rate) / new_count
                
                cursor.execute('''
                    UPDATE learning_insights 
                    SET occurrence_count = ?, success_rate = ?, confidence = ?, last_updated = ?
                    WHERE insight_id = ?
                ''', (new_count, new_rate, min(new_rate + 0.2, 1.0), datetime.now().isoformat(), row[0]))
            else:
                insight_id = str(uuid.uuid4())
                category = LearningCategory.OPTIMAL_STRATEGY if effective else LearningCategory.AVOIDED_APPROACH
                
                cursor.execute('''
                    INSERT INTO learning_insights
                    (insight_id, category, pattern, conditions, recommendation, confidence, success_rate, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    insight_id,
                    category.value,
                    insight_key,
                    json.dumps(conditions),
                    f"{'Use' if effective else 'Avoid'} {strategy} strategy",
                    0.7 if effective else 0.6,
                    1.0 if effective else 0.0,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
    
    def _get_relevant_insights(self, context: Dict[str, Any]) -> List[LearningInsight]:
        """Get insights relevant to the current context."""
        insights = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM learning_insights WHERE confidence >= 0.5 ORDER BY success_rate DESC, occurrence_count DESC LIMIT 20'
            )
            
            for row in cursor.fetchall():
                insight = LearningInsight(
                    insight_id=row[0],
                    category=LearningCategory(row[1]),
                    pattern=row[2],
                    conditions=json.loads(row[3]) if row[3] else [],
                    recommendation=row[4],
                    confidence=row[5],
                    occurrence_count=row[6],
                    success_rate=row[7],
                    last_updated=row[8]
                )
                
                if self._is_insight_relevant(insight, context):
                    insights.append(insight)
        
        return insights
    
    def _is_insight_relevant(self, insight: LearningInsight, context: Dict[str, Any]) -> bool:
        """Check if an insight is relevant to the context."""
        if not insight.conditions:
            return True
        
        for condition in insight.conditions:
            if condition in str(context):
                return True
        
        return False
    
    def _get_relevant_error_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get error patterns relevant to the context."""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM error_patterns ORDER BY occurrence_count DESC LIMIT 10'
            )
            
            for row in cursor.fetchall():
                success_rate = row[4] / row[3] if row[3] > 0 else 0.0
                patterns.append({
                    "pattern_id": row[0],
                    "error_type": row[1],
                    "error_message_pattern": row[2],
                    "solution": row[3],
                    "success_rate": success_rate
                })
        
        return patterns
    
    def _classify_error(self, error_message: str) -> str:
        """Classify an error message into a type."""
        error_lower = error_message.lower()
        
        if "nullpointer" in error_lower or "null" in error_lower:
            return "NullPointerException"
        elif "classnotfound" in error_lower:
            return "ClassNotFoundException"
        elif "illegalargument" in error_lower:
            return "IllegalArgumentException"
        elif "indexoutofbound" in error_lower:
            return "IndexOutOfBoundsException"
        elif "assertion" in error_lower:
            return "AssertionError"
        elif "compilation" in error_lower or "cannot find symbol" in error_lower:
            return "CompilationError"
        elif "timeout" in error_lower:
            return "TimeoutError"
        else:
            return "UnknownError"
    
    def update_error_solution(self, error_type: str, solution: str):
        """Update the solution for an error pattern."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE error_patterns 
                SET solution = ?, success_count = success_count + 1
                WHERE error_type = ?
            ''', (solution, error_type))
            conn.commit()
    
    def clear_old_events(self, days: int = 30):
        """Clear events older than specified days."""
        cutoff = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM feedback_events WHERE datetime(timestamp) < datetime(?, ?)',
                (cutoff, f'-{days} days')
            )
            conn.commit()
        
        logger.info(f"[EnhancedFeedbackLoop] Cleared events older than {days} days")
