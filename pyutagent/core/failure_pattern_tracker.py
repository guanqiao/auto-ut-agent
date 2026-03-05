"""Failure Pattern Tracker - Prevents meaningless retries by tracking failure patterns.

This module provides intelligent failure pattern detection and strategy adjustment
to avoid wasting time on repeated failures with the same root cause.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """Represents a detected failure pattern."""

    error_type: str
    error_message: str
    step_name: str
    count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    contexts: List[Dict[str, Any]] = field(default_factory=list)

    def record_occurrence(self, context: Optional[Dict[str, Any]] = None):
        """Record a new occurrence of this failure pattern."""
        self.count += 1
        self.last_seen = datetime.now()
        if context:
            self.contexts.append({
                "timestamp": datetime.now().isoformat(),
                "context": context
            })

    def get_time_span(self) -> timedelta:
        """Get the time span between first and last occurrence."""
        return self.last_seen - self.first_seen

    def is_recent(self, window_seconds: float = 300) -> bool:
        """Check if the failure occurred recently."""
        return (datetime.now() - self.last_seen).total_seconds() < window_seconds


class FailurePatternTracker:
    """Tracks failure patterns to prevent meaningless retries.

    This class identifies repeated failures and provides recommendations
    for how to handle them (retry, skip, escalate, etc.).
    """

    def __init__(
        self,
        max_repeated_failures: int = 3,
        pattern_expiry_seconds: float = 600,
        similarity_threshold: float = 0.8
    ):
        """Initialize the failure pattern tracker.

        Args:
            max_repeated_failures: Maximum failures before recommending stop
            pattern_expiry_seconds: How long to remember patterns (seconds)
            similarity_threshold: Threshold for considering errors similar
        """
        self.patterns: Dict[str, FailurePattern] = {}
        self.max_repeated_failures = max_repeated_failures
        self.pattern_expiry_seconds = pattern_expiry_seconds
        self.similarity_threshold = similarity_threshold

    def _generate_pattern_key(self, error: Exception, step_name: str) -> str:
        """Generate a unique key for a failure pattern.

        The key includes error type, step name, and a simplified error message
        to group similar errors together.
        """
        error_type = type(error).__name__
        # Simplify error message by removing variable parts (file paths, line numbers, etc.)
        error_msg = str(error)
        # Remove common variable parts
        import re
        # Remove file paths
        error_msg = re.sub(r'[\w/\\.-]+\.(java|py|class):\d+', '<FILE>', error_msg)
        # Remove line numbers
        error_msg = re.sub(r'line \d+', 'line <N>', error_msg)
        # Remove specific identifiers (keep first 100 chars)
        simplified_msg = error_msg[:100]

        return f"{step_name}:{error_type}:{simplified_msg}"

    def record_failure(
        self,
        error: Exception,
        step_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> FailurePattern:
        """Record a failure and return the updated pattern.

        Args:
            error: The exception that occurred
            step_name: Name of the step where failure occurred
            context: Additional context about the failure

        Returns:
            The FailurePattern for this error
        """
        pattern_key = self._generate_pattern_key(error, step_name)

        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = FailurePattern(
                error_type=type(error).__name__,
                error_message=str(error)[:200],
                step_name=step_name
            )
            logger.info(f"[FailurePatternTracker] New failure pattern detected: {pattern_key}")

        pattern = self.patterns[pattern_key]
        pattern.record_occurrence(context)

        logger.info(f"[FailurePatternTracker] Failure recorded - Step: {step_name}, "
                   f"Type: {type(error).__name__}, Count: {pattern.count}")

        return pattern

    def should_stop_retrying(self, error: Exception, step_name: str) -> bool:
        """Check if we should stop retrying this operation.

        Args:
            error: The exception that occurred
            step_name: Name of the step

        Returns:
            True if retry should stop, False otherwise
        """
        pattern_key = self._generate_pattern_key(error, step_name)
        pattern = self.patterns.get(pattern_key)

        if not pattern:
            return False

        # Stop if we've seen this failure too many times
        if pattern.count >= self.max_repeated_failures:
            logger.warning(f"[FailurePatternTracker] Max repeated failures reached - "
                          f"Step: {step_name}, Count: {pattern.count}")
            return True

        # Stop if the pattern is old but still failing (indicates persistent issue)
        if pattern.count >= 2 and not pattern.is_recent(self.pattern_expiry_seconds):
            logger.warning(f"[FailurePatternTracker] Persistent failure detected - "
                          f"Step: {step_name}, First seen: {pattern.first_seen}")
            return True

        return False

    def get_recommendation(self, error: Exception, step_name: str) -> str:
        """Get a recommendation for how to handle this failure.

        Args:
            error: The exception that occurred
            step_name: Name of the step

        Returns:
            Recommendation: "retry", "retry_with_modification", "skip_file",
                           "accept_partial", "escalate", or "reset"
        """
        pattern_key = self._generate_pattern_key(error, step_name)
        pattern = self.patterns.get(pattern_key)

        if not pattern:
            return "retry"

        # If we haven't seen this many times, try again
        if pattern.count < self.max_repeated_failures // 2:
            return "retry"

        # If we've seen this several times, try with modification
        if pattern.count < self.max_repeated_failures:
            return "retry_with_modification"

        # For repeated failures, recommend based on step type
        if step_name in ["compilation", "compile", "compiling"]:
            return "skip_file"
        elif step_name in ["test", "testing", "test_execution"]:
            return "accept_partial"
        elif step_name in ["generate", "generating", "generation"]:
            return "reset"
        else:
            return "escalate"

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about tracked failure patterns.

        Returns:
            Dict with pattern statistics
        """
        total_patterns = len(self.patterns)
        active_patterns = sum(1 for p in self.patterns.values() if p.is_recent())
        critical_patterns = sum(
            1 for p in self.patterns.values()
            if p.count >= self.max_repeated_failures
        )

        return {
            "total_patterns": total_patterns,
            "active_patterns": active_patterns,
            "critical_patterns": critical_patterns,
            "max_repeated_failures": self.max_repeated_failures,
            "patterns": [
                {
                    "error_type": p.error_type,
                    "step_name": p.step_name,
                    "count": p.count,
                    "first_seen": p.first_seen.isoformat(),
                    "last_seen": p.last_seen.isoformat(),
                    "is_recent": p.is_recent()
                }
                for p in self.patterns.values()
            ]
        }

    def clear_expired_patterns(self):
        """Clear patterns that haven't occurred recently."""
        expired_keys = [
            key for key, pattern in self.patterns.items()
            if not pattern.is_recent(self.pattern_expiry_seconds * 2)
        ]
        for key in expired_keys:
            del self.patterns[key]
            logger.debug(f"[FailurePatternTracker] Expired pattern cleared: {key}")

    def reset(self):
        """Clear all tracked patterns."""
        self.patterns.clear()
        logger.info("[FailurePatternTracker] All patterns cleared")


class SharedFailureKnowledge:
    """Shared failure knowledge for batch processing.

    This class allows multiple file processing tasks to share
    failure information and avoid repeating the same mistakes.
    """

    def __init__(self, max_entries: int = 100):
        """Initialize shared knowledge.

        Args:
            max_entries: Maximum number of entries to keep
        """
        self.knowledge: Dict[str, Dict[str, Any]] = {}
        self.max_entries = max_entries
        self.access_times: Dict[str, datetime] = {}

    def record_failure(
        self,
        file_pattern: str,
        error: Exception,
        step_name: str,
        skip_recommended: bool = False
    ):
        """Record a failure for a file pattern.

        Args:
            file_pattern: Pattern to match files (e.g., "*.java" or class name)
            error: The exception that occurred
            step_name: Name of the step
            skip_recommended: Whether to recommend skipping similar files
        """
        # Evict old entries if at capacity
        if len(self.knowledge) >= self.max_entries:
            self._evict_oldest()

        self.knowledge[file_pattern] = {
            "error_type": type(error).__name__,
            "error_message": str(error)[:200],
            "step_name": step_name,
            "timestamp": datetime.now().isoformat(),
            "skip_recommended": skip_recommended
        }
        self.access_times[file_pattern] = datetime.now()

    def should_skip_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Check if a file should be skipped based on shared knowledge.

        Args:
            file_path: Path to the file to check

        Returns:
            Tuple of (should_skip, reason)
        """
        from pathlib import Path

        file_name = Path(file_path).name

        # Check for exact match
        if file_name in self.knowledge:
            entry = self.knowledge[file_name]
            if entry.get("skip_recommended", False):
                return True, f"Known issue: {entry['error_message']}"

        # Check for pattern match (simplified)
        for pattern, entry in self.knowledge.items():
            if pattern in file_name or file_name in pattern:
                if entry.get("skip_recommended", False):
                    return True, f"Similar file pattern has known issue: {entry['error_message']}"

        return False, None

    def _evict_oldest(self):
        """Evict the oldest entry from knowledge."""
        if not self.access_times:
            return

        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.knowledge[oldest_key]
        del self.access_times[oldest_key]
        logger.debug(f"[SharedFailureKnowledge] Evicted oldest entry: {oldest_key}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about shared knowledge."""
        return {
            "total_entries": len(self.knowledge),
            "max_entries": self.max_entries,
            "entries": list(self.knowledge.keys())
        }

    def clear(self):
        """Clear all shared knowledge."""
        self.knowledge.clear()
        self.access_times.clear()
        logger.info("[SharedFailureKnowledge] All entries cleared")
