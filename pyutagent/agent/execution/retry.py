"""Unified Retry Mechanism.

This module provides a unified retry mechanism with:
- Decorator-based retry
- Multiple backoff strategies
- Smart retry policies
- Error classification-based retry

This module integrates with the core RetryConfig for consistent behavior.
"""

import asyncio
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ...core.retry_config import (
    RetryConfig as CoreRetryConfig,
    RetryStrategy as CoreRetryStrategy,
    get_default_retry_config,
)

logger = logging.getLogger(__name__)


class BackoffStrategy(Enum):
    """Backoff strategies for retry (maps to core RetryStrategy)."""
    FIXED = auto()
    LINEAR = auto()
    EXPONENTIAL = auto()
    EXPONENTIAL_JITTER = auto()
    CUSTOM = auto()
    
    @classmethod
    def from_core(cls, strategy: CoreRetryStrategy) -> 'BackoffStrategy':
        """Convert core RetryStrategy to BackoffStrategy."""
        mapping = {
            CoreRetryStrategy.IMMEDIATE: cls.FIXED,
            CoreRetryStrategy.FIXED_DELAY: cls.FIXED,
            CoreRetryStrategy.LINEAR_BACKOFF: cls.LINEAR,
            CoreRetryStrategy.EXPONENTIAL_BACKOFF: cls.EXPONENTIAL,
            CoreRetryStrategy.EXPONENTIAL_JITTER: cls.EXPONENTIAL_JITTER,
            CoreRetryStrategy.ADAPTIVE: cls.CUSTOM,
        }
        return mapping.get(strategy, cls.EXPONENTIAL)
    
    def to_core(self) -> CoreRetryStrategy:
        """Convert to core RetryStrategy."""
        mapping = {
            self.FIXED: CoreRetryStrategy.FIXED_DELAY,
            self.LINEAR: CoreRetryStrategy.LINEAR_BACKOFF,
            self.EXPONENTIAL: CoreRetryStrategy.EXPONENTIAL_BACKOFF,
            self.EXPONENTIAL_JITTER: CoreRetryStrategy.EXPONENTIAL_JITTER,
            self.CUSTOM: CoreRetryStrategy.ADAPTIVE,
        }
        return mapping[self]


@dataclass
class RetryConfig:
    """Configuration for retry behavior.
    
    This is an adapter that wraps core RetryConfig for backward compatibility
    while providing a simpler interface for decorator-based retry.
    """
    max_attempts: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    
    _core_config: Optional[CoreRetryConfig] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self._core_config is None:
            self._core_config = self._to_core_config()
    
    def _to_core_config(self) -> CoreRetryConfig:
        """Convert to core RetryConfig."""
        strategy_mapping = {
            BackoffStrategy.FIXED: CoreRetryStrategy.FIXED_DELAY,
            BackoffStrategy.LINEAR: CoreRetryStrategy.LINEAR_BACKOFF,
            BackoffStrategy.EXPONENTIAL: CoreRetryStrategy.EXPONENTIAL_BACKOFF,
            BackoffStrategy.EXPONENTIAL_JITTER: CoreRetryStrategy.EXPONENTIAL_JITTER,
            BackoffStrategy.CUSTOM: CoreRetryStrategy.ADAPTIVE,
        }
        return CoreRetryConfig(
            max_step_attempts=self.max_attempts,
            backoff_base=self.base_delay,
            backoff_max=self.max_delay,
            backoff_strategy=strategy_mapping[self.backoff_strategy],
        )
    
    @classmethod
    def from_core(cls, core_config: CoreRetryConfig) -> 'RetryConfig':
        """Create RetryConfig from core RetryConfig."""
        return cls(
            max_attempts=core_config.max_step_attempts,
            backoff_strategy=BackoffStrategy.from_core(core_config.backoff_strategy),
            base_delay=core_config.backoff_base,
            max_delay=core_config.backoff_max,
            _core_config=core_config,
        )
    
    @property
    def core_config(self) -> CoreRetryConfig:
        """Get the core RetryConfig."""
        if self._core_config is None:
            self._core_config = self._to_core_config()
        return self._core_config
    
    @property
    def max_total_attempts(self) -> int:
        """Get max total attempts from core config."""
        return self.core_config.max_total_attempts
    
    @property
    def max_reset_count(self) -> int:
        """Get max reset count from core config."""
        return self.core_config.max_reset_count
    
    @property
    def enable_smart_retry(self) -> bool:
        """Get enable_smart_retry from core config."""
        return self.core_config.enable_smart_retry
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        if self.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.base_delay
        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.base_delay * attempt
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER:
            delay = self.base_delay * (self.exponential_base ** (attempt - 1))
            if self.jitter:
                delay = delay * (0.5 + random.random())
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)
    
    def get_max_attempts(self, step_name: Optional[str] = None) -> int:
        """Get maximum attempts for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Maximum attempts allowed
        """
        return self.core_config.get_max_attempts(step_name)
    
    def should_stop(self, attempt: int, step_name: Optional[str] = None) -> bool:
        """Check if should stop based on attempt count.
        
        Args:
            attempt: Current attempt number
            step_name: Name of the current step (for step-specific limits)
            
        Returns:
            True if should stop
        """
        return self.core_config.should_stop(attempt, step_name)
    
    def can_reset(self, reset_count: int) -> bool:
        """Check if reset operation is allowed.
        
        Args:
            reset_count: Current reset count
            
        Returns:
            True if reset is allowed
        """
        return self.core_config.can_reset(reset_count)
    
    def should_stop_reset(self, reset_count: int) -> bool:
        """Check if should stop due to too many resets.

        Args:
            reset_count: Current reset count

        Returns:
            True if should stop
        """
        return self.core_config.should_stop_reset(reset_count)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if should retry based on exception and attempt.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number
            
        Returns:
            True if should retry
        """
        if attempt >= self.max_attempts:
            return False
        
        if self.non_retryable_exceptions:
            for exc_type in self.non_retryable_exceptions:
                if isinstance(exception, exc_type):
                    return False
        
        if self.retryable_exceptions:
            for exc_type in self.retryable_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False
        
        return True


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""
    
    @abstractmethod
    def should_retry(
        self,
        error: Exception,
        attempt: int,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if should retry.
        
        Args:
            error: The error that occurred
            attempt: Current attempt number
            context: Optional context information
            
        Returns:
            True if should retry
        """
        pass
    
    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry.
        
        Args:
            attempt: Current attempt number
            
        Returns:
            Delay in seconds
        """
        pass


class SmartRetryPolicy(RetryPolicy):
    """Smart retry policy based on error classification.
    
    This policy:
    - Retries network errors multiple times
    - Uses LLM analysis for code errors
    - Skips retry for certain error types
    """
    
    def __init__(
        self,
        max_network_retries: int = 5,
        max_code_retries: int = 2,
        base_delay: float = 1.0
    ):
        self._max_network_retries = max_network_retries
        self._max_code_retries = max_code_retries
        self._base_delay = base_delay
        self._error_classifier = None
    
    def _get_error_classifier(self):
        """Get error classifier lazily."""
        if self._error_classifier is None:
            try:
                from ...core.error_classifier import ErrorClassifier
                self._error_classifier = ErrorClassifier()
            except Exception:
                pass
        return self._error_classifier
    
    def _is_network_error(self, error: Exception) -> bool:
        """Check if error is a network error."""
        error_msg = str(error).lower()
        network_indicators = [
            'timeout', 'connection', 'network', 'socket',
            'api', 'rate limit', 'throttl', 'unavailable'
        ]
        return any(indicator in error_msg for indicator in network_indicators)
    
    def _is_code_error(self, error: Exception) -> bool:
        """Check if error is a code error."""
        error_msg = str(error).lower()
        code_indicators = [
            'syntax', 'compilation', 'import', 'undefined',
            'type error', 'nullpointer', 'assertion'
        ]
        return any(indicator in error_msg for indicator in code_indicators)
    
    def should_retry(
        self,
        error: Exception,
        attempt: int,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if should retry based on error type."""
        if self._is_network_error(error):
            return attempt < self._max_network_retries
        elif self._is_code_error(error):
            return attempt < self._max_code_retries
        
        classifier = self._get_error_classifier()
        if classifier:
            try:
                error_info = classifier.get_detailed_error_info(error, context or {})
                if error_info.get("is_environment_issue"):
                    return attempt < 2
            except Exception:
                pass
        
        return attempt < 3
    
    def get_delay(self, attempt: int) -> float:
        """Get delay with exponential backoff."""
        return self._base_delay * (2 ** (attempt - 1))


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_time: float = 0.0
    errors: List[Exception] = field(default_factory=list)


def with_retry(
    max_attempts: int = 3,
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    policy: Optional[RetryPolicy] = None
):
    """Decorator for automatic retry on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        backoff_strategy: Backoff strategy to use
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        retryable_exceptions: List of exceptions to retry on
        on_retry: Callback called on each retry
        policy: Optional retry policy (overrides other settings)
    
    Returns:
        Decorated function
    
    Example:
        @with_retry(max_attempts=3, backoff_strategy=BackoffStrategy.EXPONENTIAL)
        async def fetch_data():
            ...
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        backoff_strategy=backoff_strategy,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions or []
    )
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await _execute_with_retry(
                func, args, kwargs, config, policy, on_retry
            )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return _execute_with_retry_sync(
                func, args, kwargs, config, policy, on_retry
            )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


async def _execute_with_retry(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: RetryConfig,
    policy: Optional[RetryPolicy],
    on_retry: Optional[Callable]
) -> Any:
    """Execute async function with retry."""
    start_time = time.time()
    attempt = 0
    errors = []
    
    while True:
        attempt += 1
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            errors.append(e)
            
            should_retry = False
            if policy:
                should_retry = policy.should_retry(e, attempt)
            else:
                should_retry = config.should_retry(e, attempt)
            
            if not should_retry:
                raise
            
            if on_retry:
                on_retry(e, attempt)
            
            delay = policy.get_delay(attempt) if policy else config.get_delay(attempt)
            logger.warning(
                f"[Retry] Attempt {attempt}/{config.max_attempts} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            
            await asyncio.sleep(delay)


def _execute_with_retry_sync(
    func: Callable,
    args: tuple,
    kwargs: dict,
    config: RetryConfig,
    policy: Optional[RetryPolicy],
    on_retry: Optional[Callable]
) -> Any:
    """Execute sync function with retry."""
    attempt = 0
    errors = []
    
    while True:
        attempt += 1
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            errors.append(e)
            
            should_retry = False
            if policy:
                should_retry = policy.should_retry(e, attempt)
            else:
                should_retry = config.should_retry(e, attempt)
            
            if not should_retry:
                raise
            
            if on_retry:
                on_retry(e, attempt)
            
            delay = policy.get_delay(attempt) if policy else config.get_delay(attempt)
            logger.warning(
                f"[Retry] Attempt {attempt}/{config.max_attempts} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            
            time.sleep(delay)


class RetryExecutor:
    """Executor for running operations with retry logic.
    
    This class provides a programmatic interface for retry logic,
    useful when decorator-based retry is not suitable.
    
    Example:
        executor = RetryExecutor(config=RetryConfig(max_attempts=3))
        result = await executor.execute(some_async_function, arg1, arg2)
    """
    
    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        policy: Optional[RetryPolicy] = None,
        on_retry: Optional[Callable[[Exception, int], None]] = None
    ):
        self._config = config or RetryConfig()
        self._policy = policy
        self._on_retry = on_retry
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> RetryResult:
        """Execute function with retry.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            RetryResult with execution result
        """
        start_time = time.time()
        attempt = 0
        errors = []
        
        while True:
            attempt += 1
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_time=time.time() - start_time,
                    errors=errors
                )
            except Exception as e:
                errors.append(e)
                
                should_retry = False
                if self._policy:
                    should_retry = self._policy.should_retry(e, attempt)
                else:
                    should_retry = self._config.should_retry(e, attempt)
                
                if not should_retry:
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=attempt,
                        total_time=time.time() - start_time,
                        errors=errors
                    )
                
                if self._on_retry:
                    self._on_retry(e, attempt)
                
                delay = (
                    self._policy.get_delay(attempt)
                    if self._policy
                    else self._config.get_delay(attempt)
                )
                
                logger.warning(
                    f"[RetryExecutor] Attempt {attempt}/{self._config.max_attempts} "
                    f"failed: {e}. Retrying in {delay:.1f}s..."
                )
                
                await asyncio.sleep(delay)
    
    def execute_sync(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> RetryResult:
        """Execute sync function with retry.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            RetryResult with execution result
        """
        start_time = time.time()
        attempt = 0
        errors = []
        
        while True:
            attempt += 1
            try:
                result = func(*args, **kwargs)
                
                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_time=time.time() - start_time,
                    errors=errors
                )
            except Exception as e:
                errors.append(e)
                
                should_retry = False
                if self._policy:
                    should_retry = self._policy.should_retry(e, attempt)
                else:
                    should_retry = self._config.should_retry(e, attempt)
                
                if not should_retry:
                    return RetryResult(
                        success=False,
                        error=e,
                        attempts=attempt,
                        total_time=time.time() - start_time,
                        errors=errors
                    )
                
                if self._on_retry:
                    self._on_retry(e, attempt)
                
                delay = (
                    self._policy.get_delay(attempt)
                    if self._policy
                    else self._config.get_delay(attempt)
                )
                
                logger.warning(
                    f"[RetryExecutor] Attempt {attempt}/{self._config.max_attempts} "
                    f"failed: {e}. Retrying in {delay:.1f}s..."
                )
                
                time.sleep(delay)
