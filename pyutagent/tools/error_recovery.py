"""Error recovery module for enhanced fault tolerance.

This module provides comprehensive error recovery strategies including:
- Error classification and analysis
- Recovery strategy selection
- State preservation and rollback
- Graceful degradation
"""

import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Type, Union
from datetime import datetime
from contextlib import contextmanager
import asyncio

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for recovery strategy selection."""
    TRANSIENT = auto()      # Temporary errors that may succeed on retry
    PERMANENT = auto()      # Permanent errors that won't succeed on retry
    RESOURCE = auto()       # Resource-related errors (OOM, disk full, etc.)
    NETWORK = auto()        # Network-related errors
    TIMEOUT = auto()        # Timeout errors
    VALIDATION = auto()     # Validation errors
    SYNTAX = auto()         # Syntax errors
    LOGIC = auto()          # Logic errors
    UNKNOWN = auto()        # Unknown errors


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY = auto()          # Simple retry
    BACKOFF = auto()        # Exponential backoff retry
    FALLBACK = auto()       # Use fallback method
    RESET = auto()          # Reset state and retry
    SKIP = auto()           # Skip current operation
    ABORT = auto()          # Abort operation
    MANUAL = auto()         # Require manual intervention


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
        operation: str,
        attempt: int = 1,
        context_data: Optional[Dict[str, Any]] = None
    ) -> "ErrorContext":
        """Create ErrorContext from an exception.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            attempt: Attempt number
            context_data: Additional context data
            
        Returns:
            ErrorContext instance
        """
        return cls(
            error=error,
            error_type=type(error),
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            category=ErrorClassifier.classify(error),
            timestamp=datetime.now(),
            operation=operation,
            attempt=attempt,
            context_data=context_data or {}
        )


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempts_made: int
    error_context: Optional[ErrorContext] = None
    recovered_data: Optional[Any] = None
    error_message: str = ""
    next_action: str = ""


class ErrorClassifier:
    """Classifies errors into categories for recovery strategy selection."""
    
    # Mapping of error types to categories
    ERROR_MAPPINGS: Dict[Type[Exception], ErrorCategory] = {
        # Transient errors
        ConnectionError: ErrorCategory.NETWORK,
        TimeoutError: ErrorCategory.TIMEOUT,
        asyncio.TimeoutError: ErrorCategory.TIMEOUT,
        
        # Resource errors
        MemoryError: ErrorCategory.RESOURCE,
        OSError: ErrorCategory.RESOURCE,
        
        # Validation errors
        ValueError: ErrorCategory.VALIDATION,
        TypeError: ErrorCategory.VALIDATION,
        
        # Syntax errors
        SyntaxError: ErrorCategory.SYNTAX,
    }
    
    # Keywords in error messages that indicate categories
    MESSAGE_MAPPINGS: Dict[ErrorCategory, List[str]] = {
        ErrorCategory.NETWORK: [
            "connection", "network", "socket", "timeout",
            "unreachable", "refused", "reset"
        ],
        ErrorCategory.RESOURCE: [
            "memory", "disk", "space", "resource",
            "quota", "limit exceeded"
        ],
        ErrorCategory.TRANSIENT: [
            "temporary", "transient", "retry", "unavailable",
            "rate limit", "throttled"
        ],
        ErrorCategory.PERMANENT: [
            "not found", "invalid", "unauthorized",
            "forbidden", "authentication"
        ],
    }
    
    @classmethod
    def classify(cls, error: Exception) -> ErrorCategory:
        """Classify an error into a category.
        
        Args:
            error: The exception to classify
            
        Returns:
            ErrorCategory
        """
        error_type = type(error)
        error_message = str(error).lower()
        
        # Check direct type mapping
        for exc_type, category in cls.ERROR_MAPPINGS.items():
            if isinstance(error, exc_type):
                return category
        
        # Check message content
        for category, keywords in cls.MESSAGE_MAPPINGS.items():
            for keyword in keywords:
                if keyword in error_message:
                    return category
        
        return ErrorCategory.UNKNOWN
    
    @classmethod
    def is_retryable(cls, error: Exception) -> bool:
        """Check if an error is retryable.
        
        Args:
            error: The exception to check
            
        Returns:
            True if the error is retryable
        """
        category = cls.classify(error)
        return category in [
            ErrorCategory.TRANSIENT,
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RESOURCE
        ]


class RecoveryManager:
    """Manages error recovery with multiple strategies."""
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0
    ):
        """Initialize recovery manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_base: Base delay for exponential backoff (seconds)
            backoff_max: Maximum backoff delay (seconds)
        """
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.error_history: List[ErrorContext] = []
        self.recovery_stats: Dict[str, Dict[str, int]] = {}
    
    async def execute_with_recovery(
        self,
        operation: Callable,
        operation_name: str,
        fallback: Optional[Callable] = None,
        context_data: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> RecoveryResult:
        """Execute an operation with error recovery.
        
        Args:
            operation: The operation to execute
            operation_name: Name of the operation
            fallback: Optional fallback operation
            context_data: Additional context data
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            RecoveryResult
        """
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Executing {operation_name} (attempt {attempt}/{self.max_retries})")
                
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Success
                self._record_success(operation_name, attempt)
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RETRY if attempt > 1 else RecoveryStrategy.RETRY,
                    attempts_made=attempt,
                    recovered_data=result
                )
                
            except Exception as e:
                last_error = e
                error_context = ErrorContext.from_exception(
                    e, operation_name, attempt, context_data
                )
                self.error_history.append(error_context)
                
                logger.warning(
                    f"{operation_name} failed (attempt {attempt}): {e}"
                )
                
                # Check if we should retry
                if not ErrorClassifier.is_retryable(e):
                    logger.error(f"Error not retryable: {e}")
                    break
                
                # Check if we have more retries
                if attempt < self.max_retries:
                    # Calculate backoff delay
                    delay = min(
                        self.backoff_base * (2 ** (attempt - 1)),
                        self.backoff_max
                    )
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
        
        # All retries exhausted, try fallback
        if fallback:
            logger.info(f"Attempting fallback for {operation_name}")
            try:
                if asyncio.iscoroutinefunction(fallback):
                    result = await fallback(*args, **kwargs)
                else:
                    result = fallback(*args, **kwargs)
                
                self._record_fallback_success(operation_name)
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.FALLBACK,
                    attempts_made=self.max_retries,
                    recovered_data=result
                )
            except Exception as e:
                logger.error(f"Fallback also failed: {e}")
        
        # Recovery failed
        self._record_failure(operation_name)
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ABORT,
            attempts_made=self.max_retries,
            error_context=ErrorContext.from_exception(
                last_error, operation_name, self.max_retries, context_data
            ),
            error_message=str(last_error),
            next_action="Manual intervention required"
        )
    
    def _record_success(self, operation: str, attempts: int):
        """Record successful recovery."""
        if operation not in self.recovery_stats:
            self.recovery_stats[operation] = {
                "success": 0,
                "failure": 0,
                "fallback_success": 0,
                "total_attempts": 0
            }
        self.recovery_stats[operation]["success"] += 1
        self.recovery_stats[operation]["total_attempts"] += attempts
    
    def _record_failure(self, operation: str):
        """Record failed recovery."""
        if operation not in self.recovery_stats:
            self.recovery_stats[operation] = {
                "success": 0,
                "failure": 0,
                "fallback_success": 0,
                "total_attempts": 0
            }
        self.recovery_stats[operation]["failure"] += 1
    
    def _record_fallback_success(self, operation: str):
        """Record successful fallback."""
        if operation not in self.recovery_stats:
            self.recovery_stats[operation] = {
                "success": 0,
                "failure": 0,
                "fallback_success": 0,
                "total_attempts": 0
            }
        self.recovery_stats[operation]["fallback_success"] += 1
    
    def get_recovery_stats(self) -> Dict[str, Dict[str, int]]:
        """Get recovery statistics."""
        return self.recovery_stats.copy()
    
    def get_error_history(self) -> List[ErrorContext]:
        """Get error history."""
        return self.error_history.copy()
    
    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()


class StatePreserver:
    """Preserves and restores state for rollback capability."""
    
    def __init__(self):
        """Initialize state preserver."""
        self.state_stack: List[Dict[str, Any]] = []
        self.max_stack_size = 10
    
    def save_state(self, state: Dict[str, Any], label: str = "") -> int:
        """Save current state.
        
        Args:
            state: State dictionary to save
            label: Optional label for the state
            
        Returns:
            State version number
        """
        state_copy = {
            "_label": label,
            "_timestamp": datetime.now(),
            "_version": len(self.state_stack),
            **{k: v for k, v in state.items()}
        }
        
        self.state_stack.append(state_copy)
        
        # Limit stack size
        if len(self.state_stack) > self.max_stack_size:
            self.state_stack.pop(0)
        
        logger.debug(f"State saved: {label} (version {state_copy['_version']})")
        return state_copy["_version"]
    
    def restore_state(self, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Restore state to a previous version.
        
        Args:
            version: Version to restore (None for last state)
            
        Returns:
            Restored state or None if not found
        """
        if not self.state_stack:
            return None
        
        if version is None:
            state = self.state_stack[-1]
            logger.info(f"Restored to last state: {state.get('_label', 'unnamed')}")
            return {k: v for k, v in state.items() if not k.startswith('_')}
        
        for state in reversed(self.state_stack):
            if state.get("_version") == version:
                logger.info(f"Restored to state version {version}: {state.get('_label', 'unnamed')}")
                return {k: v for k, v in state.items() if not k.startswith('_')}
        
        logger.warning(f"State version {version} not found")
        return None
    
    def clear_history(self):
        """Clear all saved states."""
        self.state_stack.clear()
        logger.info("State history cleared")


class GracefulDegradation:
    """Provides graceful degradation for operations."""
    
    def __init__(self):
        """Initialize graceful degradation."""
        self.degradation_levels: Dict[str, List[Callable]] = {}
        self.current_level: Dict[str, int] = {}
    
    def register_degradation_chain(
        self,
        operation_name: str,
        methods: List[Callable]
    ):
        """Register a chain of degradation methods.
        
        Args:
            operation_name: Name of the operation
            methods: List of methods to try in order (best to worst)
        """
        self.degradation_levels[operation_name] = methods
        self.current_level[operation_name] = 0
    
    async def execute_with_degradation(
        self,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute with graceful degradation.
        
        Args:
            operation_name: Name of the operation
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from the best available method
            
        Raises:
            Exception: If all methods fail
        """
        if operation_name not in self.degradation_levels:
            raise ValueError(f"No degradation chain registered for {operation_name}")
        
        methods = self.degradation_levels[operation_name]
        start_level = self.current_level.get(operation_name, 0)
        
        for i in range(start_level, len(methods)):
            method = methods[i]
            try:
                logger.info(f"Trying {operation_name} at degradation level {i}")
                
                if asyncio.iscoroutinefunction(method):
                    result = await method(*args, **kwargs)
                else:
                    result = method(*args, **kwargs)
                
                # Success, update current level
                self.current_level[operation_name] = i
                return result
                
            except Exception as e:
                logger.warning(f"Degradation level {i} failed: {e}")
                continue
        
        # All levels failed
        raise Exception(f"All degradation levels failed for {operation_name}")
    
    def reset_level(self, operation_name: str):
        """Reset degradation level to best quality.
        
        Args:
            operation_name: Name of the operation
        """
        self.current_level[operation_name] = 0
        logger.info(f"Degradation level reset for {operation_name}")


# Context manager for safe execution
@contextmanager
def safe_execution_context(
    operation_name: str,
    recovery_manager: Optional[RecoveryManager] = None,
    on_error: Optional[Callable] = None
):
    """Context manager for safe execution with error handling.
    
    Args:
        operation_name: Name of the operation
        recovery_manager: Optional recovery manager
        on_error: Optional callback on error
        
    Yields:
        None
    """
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {e}")
        
        if on_error:
            try:
                on_error(e)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")
        
        raise


# Convenience functions
def create_recovery_manager(
    max_retries: int = 3,
    backoff_base: float = 1.0
) -> RecoveryManager:
    """Create a recovery manager with common settings.
    
    Args:
        max_retries: Maximum retry attempts
        backoff_base: Base backoff delay
        
    Returns:
        RecoveryManager instance
    """
    return RecoveryManager(
        max_retries=max_retries,
        backoff_base=backoff_base
    )


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an error.
    
    Args:
        error: Exception to classify
        
    Returns:
        ErrorCategory
    """
    return ErrorClassifier.classify(error)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable.
    
    Args:
        error: Exception to check
        
    Returns:
        True if retryable
    """
    return ErrorClassifier.is_retryable(error)
