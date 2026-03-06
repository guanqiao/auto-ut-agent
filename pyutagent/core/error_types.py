"""Unified error types for the entire system.

This module provides centralized error classification and recovery strategies
to ensure consistent error handling across all modules.

This module consolidates error types from:
- error_recovery.py (detailed error categories)
- error_handling.py (basic error categories)
- root_cause_analyzer.py (compilation-specific categories)
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Type
import traceback


class ErrorCategory(Enum):
    """Unified error categories for the entire system.
    
    This enum consolidates error categories from multiple modules to provide
    a single source of truth for error classification.
    """
    TRANSIENT = auto()
    PERMANENT = auto()
    RESOURCE = auto()
    NETWORK = auto()
    TIMEOUT = auto()
    VALIDATION = auto()
    SYNTAX = auto()
    LOGIC = auto()
    COMPILATION_ERROR = auto()
    COMPILATION = auto()
    RUNTIME = auto()
    TEST_FAILURE = auto()
    TOOL_EXECUTION_ERROR = auto()
    PARSING_ERROR = auto()
    GENERATION_ERROR = auto()
    FILE_IO_ERROR = auto()
    LLM_API_ERROR = auto()
    CONFIGURATION = auto()
    DEPENDENCY_ERROR = auto()
    ENVIRONMENT_ERROR = auto()
    UNKNOWN = auto()


class RecoveryStrategy(Enum):
    """Unified recovery strategies for the entire system.
    
    This enum consolidates recovery strategies from multiple modules to provide
    a single source of truth for recovery actions.
    """
    RETRY = auto()
    RETRY_IMMEDIATE = auto()
    RETRY_WITH_BACKOFF = auto()
    BACKOFF = auto()
    FALLBACK = auto()
    RESET = auto()
    SKIP = auto()
    SKIP_AND_CONTINUE = auto()
    ABORT = auto()
    MANUAL = auto()
    ANALYZE_AND_FIX = auto()
    RESET_AND_REGENERATE = auto()
    FALLBACK_ALTERNATIVE = auto()
    ESCALATE_TO_USER = auto()
    INSTALL_DEPENDENCIES = auto()
    RESOLVE_DEPENDENCIES = auto()
    FIX_ENVIRONMENT = auto()


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    error_type: Type[Exception]
    error_message: str
    stack_trace: str
    category: ErrorCategory
    timestamp: datetime
    operation: str
    attempt: int
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_exception(
        cls,
        error: Exception,
        operation: str = "unknown",
        attempt: int = 1,
        context_data: Optional[Dict[str, Any]] = None,
        category: Optional[ErrorCategory] = None
    ) -> "ErrorContext":
        """Create ErrorContext from an exception.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            attempt: Current attempt number
            context_data: Additional context data
            category: Optional pre-determined category
            
        Returns:
            ErrorContext instance
        """
        return cls(
            error=error,
            error_type=type(error),
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            category=category or ErrorCategory.UNKNOWN,
            timestamp=datetime.now(),
            operation=operation,
            attempt=attempt,
            context_data=context_data or {}
        )


@dataclass
class PyUTError:
    """Unified error definition for the system."""
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[Exception] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"[{self.category.name}:{self.severity.name}] {self.error_type} - {self.message}"


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    timestamp: str
    error_category: ErrorCategory
    error_message: str
    strategy_used: RecoveryStrategy
    attempt_number: int
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempts_made: int
    error_context: Optional[ErrorContext] = None
    recovered_data: Any = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


def classify_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
    """Classify an error into a category.
    
    This is a utility function that provides basic error classification.
    For more sophisticated classification, use ErrorClassifier from error_recovery.py.
    
    Args:
        error: The exception to classify
        context: Optional context for classification
        
    Returns:
        ErrorCategory for the error
    """
    error_name = type(error).__name__.lower()
    error_msg = str(error).lower()
    context = context or {}
    
    if 'timeout' in error_name or 'timeout' in error_msg:
        return ErrorCategory.TIMEOUT
    
    if 'network' in error_name or 'connection' in error_msg:
        return ErrorCategory.NETWORK
    
    if 'compilation' in error_msg or 'compile' in error_msg:
        return ErrorCategory.COMPILATION_ERROR
    
    if 'test' in context.get('step', '').lower():
        return ErrorCategory.TEST_FAILURE
    
    return ErrorCategory.UNKNOWN
