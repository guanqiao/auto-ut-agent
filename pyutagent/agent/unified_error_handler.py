"""Unified Error Handler for consistent error handling across the agent.

This module provides:
- ErrorHandler: Centralized error handling
- Error categories and severity levels
- Recovery strategies
- Error logging and reporting
"""

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type

from pyutagent.core.error_recovery import ErrorCategory, RecoveryStrategy

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class ErrorContext:
    """Error context information."""
    error: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    agent_state: str
    task: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoverySuggestion:
    """Recovery suggestion for an error."""
    action: str
    description: str
    confidence: float
    steps: List[str] = field(default_factory=list)


@dataclass
class ErrorReport:
    """Error report for logging and analysis."""
    error_id: str
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    context: Dict[str, Any]
    recovery_suggestion: Optional[RecoverySuggestion]
    stack_trace: str


class ErrorHandler:
    """Unified error handler.

    Features:
    - Categorize errors
    - Determine severity
    - Generate recovery suggestions
    - Log and report errors
    """

    ERROR_CATEGORIES = {
        SyntaxError: ErrorCategory.SYNTAX,
        ValueError: ErrorCategory.VALIDATION,
        TypeError: ErrorCategory.VALIDATION,
        ImportError: ErrorCategory.PARSING_ERROR,
        FileNotFoundError: ErrorCategory.FILE_IO_ERROR,
        TimeoutError: ErrorCategory.TIMEOUT,
        ConnectionError: ErrorCategory.NETWORK,
        PermissionError: ErrorCategory.VALIDATION,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize error handler.

        Args:
            config: Configuration options
        """
        self.config = config or {}
        self._error_history: List[ErrorReport] = []
        self._error_counts: Dict[ErrorCategory, int] = {}
        self._recovery_strategies: Dict[ErrorCategory, Callable] = {}

    def categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error.

        Args:
            error: The error to categorize

        Returns:
            Error category
        """
        error_type = type(error)

        if error_type in self.ERROR_CATEGORIES:
            return self.ERROR_CATEGORIES[error_type]

        error_name = error_type.__name__.lower()
        if "parse" in error_name or "syntax" in error_name:
            return ErrorCategory.SYNTAX
        if "compile" in error_name or "build" in error_name:
            return ErrorCategory.COMPILATION_ERROR
        if "test" in error_name:
            return ErrorCategory.TEST_FAILURE
        if "network" in error_name or "connection" in error_name:
            return ErrorCategory.NETWORK
        if "auth" in error_name or "permission" in error_name:
            return ErrorCategory.VALIDATION
        if "validate" in error_name or "invalid" in error_name:
            return ErrorCategory.VALIDATION
        if "timeout" in error_name:
            return ErrorCategory.TIMEOUT

        return ErrorCategory.UNKNOWN

    def determine_severity(
        self,
        error: Exception,
        category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity.

        Args:
            error: The error
            category: Error category

        Returns:
            Severity level
        """
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return ErrorSeverity.CRITICAL

        error_type = type(error).__name__.lower()

        if "outofmemory" in error_type or "memory" in error_type:
            return ErrorSeverity.CRITICAL
        if "permission" in error_type or "denied" in error_type:
            return ErrorSeverity.HIGH
        if "timeout" in error_type:
            return ErrorSeverity.MEDIUM

        if category in (ErrorCategory.COMPILATION_ERROR, ErrorCategory.TEST_FAILURE):
            return ErrorSeverity.HIGH

        return ErrorSeverity.MEDIUM

    def create_error_context(
        self,
        error: Exception,
        agent_state: str,
        task: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Create error context.

        Args:
            error: The error
            agent_state: Current agent state
            task: Current task
            metadata: Additional metadata

        Returns:
            Error context
        """
        category = self.categorize_error(error)
        severity = self.determine_severity(error, category)

        return ErrorContext(
            error=error,
            category=category,
            severity=severity,
            timestamp=datetime.now(),
            agent_state=agent_state,
            task=task,
            metadata=metadata or {}
        )

    def generate_recovery_suggestion(
        self,
        error: Exception,
        context: ErrorContext
    ) -> RecoverySuggestion:
        """Generate recovery suggestion.

        Args:
            error: The error
            context: Error context

        Returns:
            Recovery suggestion
        """
        category = context.category

        suggestions = {
            ErrorCategory.SYNTAX: RecoverySuggestion(
                action="fix_syntax",
                description="Fix syntax error in code",
                confidence=0.9,
                steps=[
                    "Review error message for location",
                    "Check for missing brackets or semicolons",
                    "Validate code structure"
                ]
            ),
            ErrorCategory.PARSING_ERROR: RecoverySuggestion(
                action="fix_parsing",
                description="Fix parsing error",
                confidence=0.9,
                steps=[
                    "Review error message for location",
                    "Check for syntax issues",
                    "Validate code structure"
                ]
            ),
            ErrorCategory.COMPILATION_ERROR: RecoverySuggestion(
                action="fix_compilation",
                description="Fix compilation error",
                confidence=0.85,
                steps=[
                    "Check import statements",
                    "Verify class and method names",
                    "Ensure dependencies are available"
                ]
            ),
            ErrorCategory.TEST_FAILURE: RecoverySuggestion(
                action="fix_test",
                description="Fix test execution error",
                confidence=0.8,
                steps=[
                    "Check test setup",
                    "Verify test data",
                    "Review assertion logic"
                ]
            ),
            ErrorCategory.NETWORK: RecoverySuggestion(
                action="retry",
                description="Retry network operation",
                confidence=0.7,
                steps=[
                    "Check network connection",
                    "Retry the operation",
                    "Use fallback service if available"
                ]
            ),
            ErrorCategory.TIMEOUT: RecoverySuggestion(
                action="increase_timeout",
                description="Increase timeout and retry",
                confidence=0.75,
                steps=[
                    "Increase timeout value",
                    "Retry the operation",
                    "Consider async implementation"
                ]
            ),
        }

        return suggestions.get(
            category,
            RecoverySuggestion(
                action="analyze",
                description="Analyze error manually",
                confidence=0.5,
                steps=["Review error details", "Check logs", "Debug step by step"]
            )
        )

    def handle_error(
        self,
        error: Exception,
        agent_state: str,
        task: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ErrorReport:
        """Handle an error.

        Args:
            error: The error
            agent_state: Current agent state
            task: Current task
            metadata: Additional metadata

        Returns:
            Error report
        """
        import uuid

        context = self.create_error_context(error, agent_state, task, metadata)
        suggestion = self.generate_recovery_suggestion(error, context)

        error_type = type(error).__name__
        stack_trace = traceback.format_exc()

        report = ErrorReport(
            error_id=str(uuid.uuid4()),
            error_type=error_type,
            message=str(error),
            category=context.category,
            severity=context.severity,
            timestamp=context.timestamp,
            context={
                "agent_state": agent_state,
                "task": task,
                "metadata": metadata or {}
            },
            recovery_suggestion=suggestion,
            stack_trace=stack_trace
        )

        self._error_history.append(report)
        self._error_counts[context.category] = self._error_counts.get(context.category, 0) + 1

        logger.error(
            f"[ErrorHandler] {error_type}: {error} | "
            f"Category: {context.category.name} | "
            f"Severity: {context.severity.name} | "
            f"Task: {task}"
        )

        return report

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics.

        Returns:
            Error statistics
        """
        total = len(self._error_history)
        if total == 0:
            return {"total": 0, "by_category": {}, "by_severity": {}}

        by_category = {
            cat.name: count
            for cat, count in self._error_counts.items()
        }

        by_severity = {}
        for report in self._error_history:
            severity = report.severity.name
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total": total,
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_errors": [
                {
                    "error_id": r.error_id,
                    "type": r.error_type,
                    "category": r.category.name,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self._error_history[-10:]
            ]
        }

    def clear_history(self):
        """Clear error history."""
        self._error_history.clear()
        self._error_counts.clear()
        logger.info("[ErrorHandler] Error history cleared")


def create_error_handler(config: Optional[Dict[str, Any]] = None) -> ErrorHandler:
    """Create an error handler.

    Args:
        config: Configuration

    Returns:
        ErrorHandler instance
    """
    return ErrorHandler(config)
