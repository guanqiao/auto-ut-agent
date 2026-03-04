"""Error pattern learning for improved recovery strategies.

This module provides learning capabilities for error recovery:
- Pattern extraction from errors
- Persistent storage of error patterns
- Strategy recommendation based on history
- Learning from recovery attempts
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Type

from ..core.error_recovery import ErrorCategory, RecoveryStrategy

logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """Represents an error pattern for learning."""
    pattern_id: str
    category: ErrorCategory
    signature: str
    keywords: List[str]
    regex_patterns: List[str]
    created_at: str
    updated_at: str
    occurrence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "category": self.category.name,
            "signature": self.signature,
            "keywords": self.keywords,
            "regex_patterns": self.regex_patterns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "occurrence_count": self.occurrence_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorPattern":
        return cls(
            pattern_id=data["pattern_id"],
            category=ErrorCategory[data["category"]],
            signature=data["signature"],
            keywords=data["keywords"],
            regex_patterns=data["regex_patterns"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            occurrence_count=data.get("occurrence_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0)
        )


@dataclass
class StrategyRecord:
    """Record of a strategy used for an error pattern."""
    strategy: RecoveryStrategy
    success: bool
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)
    time_to_recover: float = 0.0
    attempts_needed: int = 1


@dataclass
class PatternMatch:
    """Result of matching an error to a pattern."""
    pattern: ErrorPattern
    confidence: float
    matched_keywords: List[str]
    matched_regex: List[str]


class PatternStorage:
    """Persistent storage for error patterns.
    
    Features:
    - JSON-based storage
    - Atomic writes
    - Automatic backup
    - Migration support
    """
    
    def __init__(self, persist_path: str):
        self.persist_path = Path(persist_path)
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._patterns: Dict[str, ErrorPattern] = {}
        self._strategy_history: Dict[str, List[StrategyRecord]] = {}
        
        self._load()
    
    def _load(self):
        """Load patterns from storage."""
        if not self.persist_path.exists():
            logger.info(f"[PatternStorage] Creating new storage at {self.persist_path}")
            self._save()
            return
        
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for pattern_data in data.get("patterns", []):
                pattern = ErrorPattern.from_dict(pattern_data)
                self._patterns[pattern.pattern_id] = pattern
            
            for pattern_id, records in data.get("strategy_history", {}).items():
                self._strategy_history[pattern_id] = [
                    StrategyRecord(
                        strategy=RecoveryStrategy[r["strategy"]],
                        success=r["success"],
                        timestamp=r["timestamp"],
                        context=r.get("context", {}),
                        time_to_recover=r.get("time_to_recover", 0.0),
                        attempts_needed=r.get("attempts_needed", 1)
                    )
                    for r in records
                ]
            
            logger.info(f"[PatternStorage] Loaded {len(self._patterns)} patterns from storage")
            
        except Exception as e:
            logger.exception(f"[PatternStorage] Failed to load: {e}")
            self._backup_corrupted()
    
    def _save(self):
        """Save patterns to storage."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "patterns": [p.to_dict() for p in self._patterns.values()],
            "strategy_history": {
                pid: [
                    {
                        "strategy": r.strategy.name,
                        "success": r.success,
                        "timestamp": r.timestamp,
                        "context": self._serialize_context(r.context),
                        "time_to_recover": r.time_to_recover,
                        "attempts_needed": r.attempts_needed
                    }
                    for r in records
                ]
                for pid, records in self._strategy_history.items()
            }
        }

        temp_path = self.persist_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            temp_path.replace(self.persist_path)
            logger.debug(f"[PatternStorage] Saved {len(self._patterns)} patterns")

        except Exception as e:
            logger.exception(f"[PatternStorage] Failed to save: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _serialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize context to JSON-compatible format.

        Args:
            context: The context dictionary to serialize

        Returns:
            JSON-serializable dictionary
        """
        serialized = {}
        for key, value in context.items():
            serialized[key] = self._serialize_value(value)
        return serialized

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value to JSON-compatible format.

        Args:
            value: The value to serialize

        Returns:
            JSON-serializable value
        """
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
            # Handle objects with to_dict method
            return self._serialize_value(value.to_dict())
        elif hasattr(value, '__dict__'):
            # Handle dataclasses and regular objects
            return self._serialize_value(value.__dict__)
        elif hasattr(value, 'name') and hasattr(value, 'value'):
            # Handle enum-like objects
            return value.name
        elif hasattr(value, 'name'):
            # Handle objects with name attribute (like AgentState)
            return value.name
        else:
            # Fallback: convert to string
            return str(value)
    
    def _backup_corrupted(self):
        """Backup corrupted storage file."""
        if self.persist_path.exists():
            backup_path = self.persist_path.with_suffix(
                f'.corrupted.{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            self.persist_path.rename(backup_path)
            logger.warning(f"[PatternStorage] Backed up corrupted file to {backup_path}")
    
    def get_pattern(self, pattern_id: str) -> Optional[ErrorPattern]:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)
    
    def save_pattern(self, pattern: ErrorPattern):
        """Save or update a pattern."""
        self._patterns[pattern.pattern_id] = pattern
        self._save()
    
    def record_strategy(
        self,
        pattern_id: str,
        strategy: RecoveryStrategy,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
        time_to_recover: float = 0.0,
        attempts_needed: int = 1
    ):
        """Record a strategy usage for a pattern."""
        if pattern_id not in self._strategy_history:
            self._strategy_history[pattern_id] = []
        
        record = StrategyRecord(
            strategy=strategy,
            success=success,
            timestamp=datetime.now().isoformat(),
            context=context or {},
            time_to_recover=time_to_recover,
            attempts_needed=attempts_needed
        )
        
        self._strategy_history[pattern_id].append(record)
        
        if pattern_id in self._patterns:
            pattern = self._patterns[pattern_id]
            pattern.occurrence_count += 1
            if success:
                pattern.success_count += 1
            else:
                pattern.failure_count += 1
            pattern.updated_at = datetime.now().isoformat()
        
        self._save()
    
    def get_strategy_history(
        self,
        pattern_id: str
    ) -> List[StrategyRecord]:
        """Get strategy history for a pattern."""
        return self._strategy_history.get(pattern_id, [])
    
    def get_all_patterns(self) -> List[ErrorPattern]:
        """Get all patterns."""
        return list(self._patterns.values())
    
    def clear(self):
        """Clear all patterns."""
        self._patterns.clear()
        self._strategy_history.clear()
        self._save()


class ErrorPatternLearner:
    """Learns from error patterns to improve recovery.
    
    Features:
    - Pattern extraction from errors
    - Strategy effectiveness tracking
    - Recommendation based on history
    - Continuous learning
    """
    
    def __init__(
        self,
        persist_path: str = ".pyutagent/error_patterns.json",
        min_samples: int = 3,
        confidence_threshold: float = 0.6
    ):
        self.storage = PatternStorage(persist_path)
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        
        self._keyword_extractors = {
            ErrorCategory.COMPILATION_ERROR: self._extract_compilation_keywords,
            ErrorCategory.TEST_FAILURE: self._extract_test_failure_keywords,
            ErrorCategory.TOOL_EXECUTION_ERROR: self._extract_tool_keywords,
            ErrorCategory.LLM_API_ERROR: self._extract_llm_keywords,
            ErrorCategory.NETWORK: self._extract_network_keywords,
        }
        
        self._session_stats = {
            "patterns_learned": 0,
            "strategies_recorded": 0,
            "recommendations_made": 0
        }
    
    def extract_pattern(
        self,
        error: Exception,
        error_category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorPattern:
        """Extract a pattern from an error.
        
        Args:
            error: The exception that occurred
            error_category: Category of the error
            context: Additional context about the error
            
        Returns:
            ErrorPattern extracted from the error
        """
        error_message = str(error)
        error_type = type(error).__name__
        
        signature = self._create_signature(error_message, error_type, error_category)
        
        extractor = self._keyword_extractors.get(
            error_category,
            self._extract_generic_keywords
        )
        keywords = extractor(error_message, context or {})
        
        regex_patterns = self._extract_regex_patterns(error_message, error_category)
        
        now = datetime.now().isoformat()
        
        pattern = ErrorPattern(
            pattern_id=self._generate_pattern_id(signature),
            category=error_category,
            signature=signature,
            keywords=keywords,
            regex_patterns=regex_patterns,
            created_at=now,
            updated_at=now,
            occurrence_count=0,
            success_count=0,
            failure_count=0
        )
        
        return pattern
    
    def _create_signature(
        self,
        error_message: str,
        error_type: str,
        category: ErrorCategory
    ) -> str:
        """Create a signature for the error."""
        normalized = error_message.lower()
        
        normalized = re.sub(r'\b\d+\b', 'NUM', normalized)
        normalized = re.sub(r'\b0x[0-9a-f]+\b', 'HEX', normalized)
        normalized = re.sub(r'\b[a-f0-9]{8,}\b', 'HASH', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return f"{category.name}:{error_type}:{normalized[:200]}"
    
    def _generate_pattern_id(self, signature: str) -> str:
        """Generate a unique ID for a pattern."""
        return hashlib.sha256(signature.encode()).hexdigest()[:16]
    
    def _extract_compilation_keywords(
        self,
        error_message: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract keywords from compilation error."""
        keywords = []
        
        patterns = [
            (r'cannot find symbol[:\s]+(\w+)', 'missing_symbol'),
            (r'package\s+([\w.]+)\s+does not exist', 'missing_package'),
            (r'class\s+(\w+)\s+not found', 'missing_class'),
            (r'incompatible types[:\s]+(.+?)(?:\n|$)', 'type_mismatch'),
            (r"method\s+(\w+)\s+cannot be applied", 'method_mismatch'),
            (r"';' expected", 'missing_semicolon'),
            (r"'}' expected", 'missing_brace'),
            (r"'{' expected", 'missing_brace'),
            (r"illegal start of (?:expression|type)", 'syntax_error'),
            (r"variable\s+(\w+)\s+might not have been initialized", 'uninitialized'),
        ]
        
        for pattern, keyword in patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                keywords.append(keyword)
        
        if 'cannot find symbol' in error_message.lower():
            match = re.search(r'cannot find symbol[:\s]+symbol\s*:\s*(?:class|method|variable)\s+(\w+)', 
                            error_message, re.IGNORECASE)
            if match:
                keywords.append(f'symbol_{match.group(1).lower()}')
        
        if re.search(r"package\s+[\w.]+\s+does not exist", error_message):
            keywords.append("missing_dependency")
            
            packages = re.findall(r"package\s+([\w.]+)\s+does not exist", error_message)
            for pkg in packages:
                if "junit" in pkg.lower():
                    keywords.append("missing_junit")
                elif "mockito" in pkg.lower():
                    keywords.append("missing_mockito")
                elif "assertj" in pkg.lower():
                    keywords.append("missing_assertj")
                elif "hamcrest" in pkg.lower():
                    keywords.append("missing_hamcrest")
                else:
                    keywords.append(f"missing_pkg_{pkg.split('.')[-1].lower()}")
        
        return keywords
    
    def _extract_test_failure_keywords(
        self,
        error_message: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract keywords from test failure."""
        keywords = []
        
        patterns = [
            (r'AssertionError', 'assertion_failed'),
            (r'NullPointerException', 'null_pointer'),
            (r'expected:\s*(.+?)\s*but was:\s*(.+?)(?:\n|$)', 'assertion_mismatch'),
            (r'Wanted but not invoked', 'mock_not_invoked'),
            (r'Too many actual invocations', 'mock_over_invoked'),
            (r'Timeout', 'timeout'),
            (r'Exception\s+:\s*(\w+)', 'exception_thrown'),
        ]
        
        for pattern, keyword in patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                keywords.append(keyword)
        
        failures = context.get("failures", [])
        for failure in failures[:3]:
            test_name = failure.get("test_name", "")
            if test_name:
                keywords.append(f"test_{test_name.lower()[:20]}")
        
        return keywords
    
    def _extract_tool_keywords(
        self,
        error_message: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract keywords from tool execution error."""
        keywords = []
        
        tool_name = context.get("tool", "unknown")
        keywords.append(f"tool_{tool_name}")
        
        patterns = [
            (r'command not found', 'command_not_found'),
            (r'permission denied', 'permission_denied'),
            (r'timeout', 'timeout'),
            (r'exit code[:\s]+(\d+)', 'exit_code'),
            (r'not recognized', 'not_recognized'),
        ]
        
        for pattern, keyword in patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                keywords.append(keyword)
        
        return keywords
    
    def _extract_llm_keywords(
        self,
        error_message: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract keywords from LLM API error."""
        keywords = []
        
        patterns = [
            (r'rate limit', 'rate_limit'),
            (r'timeout', 'timeout'),
            (r'api key', 'api_key_error'),
            (r'authentication', 'auth_error'),
            (r'context length', 'context_too_long'),
            (r'token limit', 'token_limit'),
            (r'model not found', 'model_not_found'),
        ]
        
        for pattern, keyword in patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                keywords.append(keyword)
        
        return keywords
    
    def _extract_network_keywords(
        self,
        error_message: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract keywords from network error."""
        keywords = []
        
        patterns = [
            (r'connection refused', 'connection_refused'),
            (r'connection reset', 'connection_reset'),
            (r'connection timeout', 'connection_timeout'),
            (r'dns', 'dns_error'),
            (r'ssl|tls', 'ssl_error'),
            (r'socket', 'socket_error'),
        ]
        
        for pattern, keyword in patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                keywords.append(keyword)
        
        return keywords
    
    def _extract_generic_keywords(
        self,
        error_message: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Extract generic keywords from error."""
        keywords = []
        
        words = re.findall(r'\b[a-z]{4,}\b', error_message.lower())
        word_freq: Dict[str, int] = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [w for w, _ in sorted_words[:5]]
        
        return keywords
    
    def _extract_regex_patterns(
        self,
        error_message: str,
        category: ErrorCategory
    ) -> List[str]:
        """Extract regex patterns for matching similar errors."""
        patterns = []
        
        escaped = re.escape(error_message[:100])
        escaped = re.sub(r'\\ \d+', r'\\d+', escaped)
        escaped = re.sub(r'\\ [a-f0-9]+', r'[a-f0-9]+', escaped)
        
        patterns.append(escaped)
        
        return patterns
    
    def learn_from_recovery(
        self,
        error: Exception,
        error_category: ErrorCategory,
        strategy: RecoveryStrategy,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
        time_to_recover: float = 0.0,
        attempts_needed: int = 1
    ):
        """Learn from a recovery attempt.
        
        Args:
            error: The error that occurred
            error_category: Category of the error
            strategy: Strategy used for recovery
            success: Whether recovery was successful
            context: Additional context
            time_to_recover: Time taken to recover
            attempts_needed: Number of attempts needed
        """
        pattern = self.extract_pattern(error, error_category, context)
        
        existing = self.storage.get_pattern(pattern.pattern_id)
        if existing:
            pattern = existing
        else:
            self.storage.save_pattern(pattern)
            self._session_stats["patterns_learned"] += 1
            logger.info(f"[ErrorLearner] New pattern learned: {pattern.pattern_id}")
        
        self.storage.record_strategy(
            pattern.pattern_id,
            strategy,
            success,
            context,
            time_to_recover,
            attempts_needed
        )
        
        self._session_stats["strategies_recorded"] += 1
        
        logger.debug(
            f"[ErrorLearner] Recorded strategy {strategy.name} for pattern {pattern.pattern_id}: "
            f"{'success' if success else 'failure'}"
        )
    
    def suggest_strategy(
        self,
        error: Exception,
        error_category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[RecoveryStrategy, float]]:
        """Suggest the best recovery strategy based on history.
        
        Args:
            error: The error that occurred
            error_category: Category of the error
            context: Additional context
            
        Returns:
            Tuple of (recommended strategy, confidence) or None
        """
        pattern = self.extract_pattern(error, error_category, context)
        
        existing = self.storage.get_pattern(pattern.pattern_id)
        if not existing:
            logger.debug(f"[ErrorLearner] No history for pattern {pattern.pattern_id}")
            return None
        
        history = self.storage.get_strategy_history(pattern.pattern_id)
        
        if len(history) < self.min_samples:
            logger.debug(
                f"[ErrorLearner] Not enough samples ({len(history)} < {self.min_samples}) "
                f"for pattern {pattern.pattern_id}"
            )
            return None
        
        strategy_stats: Dict[RecoveryStrategy, Dict[str, float]] = {}
        
        for record in history:
            if record.strategy not in strategy_stats:
                strategy_stats[record.strategy] = {
                    "success": 0.0,
                    "total": 0.0,
                    "total_time": 0.0,
                    "total_attempts": 0.0
                }
            
            strategy_stats[record.strategy]["total"] += 1
            if record.success:
                strategy_stats[record.strategy]["success"] += 1
            strategy_stats[record.strategy]["total_time"] += record.time_to_recover
            strategy_stats[record.strategy]["total_attempts"] += record.attempts_needed
        
        best_strategy = None
        best_score = 0.0
        
        for strategy, stats in strategy_stats.items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            avg_time = stats["total_time"] / stats["total"] if stats["total"] > 0 else float('inf')
            avg_attempts = stats["total_attempts"] / stats["total"] if stats["total"] > 0 else float('inf')
            
            time_score = 1.0 / (1.0 + avg_time / 60.0)
            attempt_score = 1.0 / avg_attempts
            
            score = (
                success_rate * 0.6 +
                time_score * 0.2 +
                attempt_score * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        if best_strategy and best_score >= self.confidence_threshold:
            self._session_stats["recommendations_made"] += 1
            logger.info(
                f"[ErrorLearner] Recommending {best_strategy.name} for pattern "
                f"{pattern.pattern_id} with confidence {best_score:.2f}"
            )
            return (best_strategy, best_score)
        
        return None
    
    def find_similar_patterns(
        self,
        error: Exception,
        error_category: ErrorCategory,
        context: Optional[Dict[str, Any]] = None
    ) -> List[PatternMatch]:
        """Find patterns similar to the given error.
        
        Args:
            error: The error to match
            error_category: Category of the error
            context: Additional context
            
        Returns:
            List of PatternMatch objects sorted by confidence
        """
        pattern = self.extract_pattern(error, error_category, context)
        matches = []
        
        for stored_pattern in self.storage.get_all_patterns():
            if stored_pattern.category != error_category:
                continue
            
            confidence = 0.0
            matched_keywords = []
            matched_regex = []
            
            for keyword in pattern.keywords:
                if keyword in stored_pattern.keywords:
                    confidence += 0.2
                    matched_keywords.append(keyword)
            
            for regex in pattern.regex_patterns:
                try:
                    if re.search(regex, stored_pattern.signature, re.IGNORECASE):
                        confidence += 0.1
                        matched_regex.append(regex)
                except re.error:
                    pass
            
            if confidence > 0:
                matches.append(PatternMatch(
                    pattern=stored_pattern,
                    confidence=min(confidence, 1.0),
                    matched_keywords=matched_keywords,
                    matched_regex=matched_regex
                ))
        
        return sorted(matches, key=lambda m: m.confidence, reverse=True)
    
    def get_pattern_stats(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific pattern."""
        pattern = self.storage.get_pattern(pattern_id)
        if not pattern:
            return None
        
        history = self.storage.get_strategy_history(pattern_id)
        
        strategy_stats: Dict[str, Dict[str, Any]] = {}
        for record in history:
            strategy_name = record.strategy.name
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {
                    "success": 0,
                    "failure": 0,
                    "avg_time": 0.0,
                    "avg_attempts": 0.0,
                    "times": [],
                    "attempts": []
                }
            
            if record.success:
                strategy_stats[strategy_name]["success"] += 1
            else:
                strategy_stats[strategy_name]["failure"] += 1
            strategy_stats[strategy_name]["times"].append(record.time_to_recover)
            strategy_stats[strategy_name]["attempts"].append(record.attempts_needed)
        
        for stats in strategy_stats.values():
            if stats["times"]:
                stats["avg_time"] = sum(stats["times"]) / len(stats["times"])
            if stats["attempts"]:
                stats["avg_attempts"] = sum(stats["attempts"]) / len(stats["attempts"])
            del stats["times"]
            del stats["attempts"]
        
        return {
            "pattern": pattern.to_dict(),
            "strategy_stats": strategy_stats,
            "total_occurrences": pattern.occurrence_count,
            "success_rate": pattern.success_count / pattern.occurrence_count if pattern.occurrence_count > 0 else 0
        }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session."""
        return self._session_stats.copy()
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global statistics across all patterns."""
        patterns = self.storage.get_all_patterns()
        
        total_patterns = len(patterns)
        total_occurrences = sum(p.occurrence_count for p in patterns)
        total_successes = sum(p.success_count for p in patterns)
        total_failures = sum(p.failure_count for p in patterns)
        
        category_stats: Dict[str, int] = {}
        for pattern in patterns:
            cat_name = pattern.category.name
            category_stats[cat_name] = category_stats.get(cat_name, 0) + pattern.occurrence_count
        
        return {
            "total_patterns": total_patterns,
            "total_occurrences": total_occurrences,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "overall_success_rate": total_successes / total_occurrences if total_occurrences > 0 else 0,
            "by_category": category_stats
        }
    
    def clear_history(self):
        """Clear all learning history."""
        self.storage.clear()
        self._session_stats = {
            "patterns_learned": 0,
            "strategies_recorded": 0,
            "recommendations_made": 0
        }
        logger.info("[ErrorLearner] Cleared all learning history")


def create_error_learner(
    persist_path: str = ".pyutagent/error_patterns.json",
    min_samples: int = 3
) -> ErrorPatternLearner:
    """Create an ErrorPatternLearner instance.
    
    Args:
        persist_path: Path to persist patterns
        min_samples: Minimum samples before recommending
        
    Returns:
        Configured ErrorPatternLearner
    """
    return ErrorPatternLearner(
        persist_path=persist_path,
        min_samples=min_samples
    )
