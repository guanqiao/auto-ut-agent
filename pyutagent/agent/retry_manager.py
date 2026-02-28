"""Retry manager for handling operation retries with various strategies."""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any, TypeVar, Generic
import functools

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategies."""
    IMMEDIATE = auto()  # Retry immediately
    FIXED_DELAY = auto()  # Fixed delay between retries
    EXPONENTIAL_BACKOFF = auto()  # Exponential backoff
    LINEAR_BACKOFF = auto()  # Linear increasing delay
    RANDOM_JITTER = auto()  # Random jitter to prevent thundering herd
    ADAPTIVE = auto()  # Adaptive based on error type


class RetryCondition(Enum):
    """When to retry."""
    ALWAYS = auto()  # Always retry
    ON_EXCEPTION = auto()  # Retry on specific exceptions
    ON_RESULT = auto()  # Retry based on result
    CUSTOM = auto()  # Custom condition


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 10
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0
    jitter_range: tuple = (0.0, 1.0)  # Random jitter range
    retry_condition: RetryCondition = RetryCondition.ON_EXCEPTION
    exceptions_to_retry: tuple = (Exception,)
    exceptions_to_ignore: tuple = (KeyboardInterrupt, SystemExit)
    should_retry_result: Optional[Callable[[Any], bool]] = None
    on_retry_callback: Optional[Callable[[int, Exception, float], None]] = None
    on_success_callback: Optional[Callable[[int, Any], None]] = None
    on_failure_callback: Optional[Callable[[int, Exception], None]] = None


@dataclass
class RetryAttempt:
    """Record of a retry attempt."""
    attempt_number: int
    timestamp: datetime
    exception: Optional[Exception]
    delay_before: float
    success: bool
    result: Any = None


@dataclass
class RetryResult(Generic[T]):
    """Result of a retry operation."""
    success: bool
    result: Optional[T]
    final_exception: Optional[Exception]
    total_attempts: int
    total_time: timedelta
    attempts: List[RetryAttempt]
    
    @property
    def is_success(self) -> bool:
        """Check if operation was successful."""
        return self.success and self.result is not None


class RetryManager:
    """Manages retry logic for operations.
    
    Supports multiple retry strategies and can be used as a context manager
    or decorator.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry manager.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self._stop_requested = False
        
    def stop(self):
        """Request to stop retrying."""
        self._stop_requested = True
        
    def reset(self):
        """Reset stop flag."""
        self._stop_requested = False
        
    def should_stop(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested
    
    def calculate_delay(self, attempt_number: int) -> float:
        """Calculate delay before next retry.
        
        Args:
            attempt_number: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        strategy = self.config.strategy
        base = self.config.base_delay
        max_delay = self.config.max_delay
        
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        
        elif strategy == RetryStrategy.FIXED_DELAY:
            return base
        
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base * (self.config.exponential_base ** (attempt_number - 1))
            return min(delay, max_delay)
        
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base * attempt_number
            return min(delay, max_delay)
        
        elif strategy == RetryStrategy.RANDOM_JITTER:
            delay = base * (self.config.exponential_base ** (attempt_number - 1))
            jitter = random.uniform(*self.config.jitter_range)
            return min(delay + jitter, max_delay)
        
        elif strategy == RetryStrategy.ADAPTIVE:
            # Adaptive: exponential with random jitter
            delay = base * (self.config.exponential_base ** (attempt_number - 1))
            jitter = random.uniform(0, delay * 0.1)  # 10% jitter
            return min(delay + jitter, max_delay)
        
        return base
    
    def should_retry_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry.
        
        Args:
            exception: The exception that occurred
            
        Returns:
            True if should retry
        """
        # Check if it's an exception we should ignore
        if isinstance(exception, self.config.exceptions_to_ignore):
            return False
        
        # Check if it's an exception we should retry
        if isinstance(exception, self.config.exceptions_to_retry):
            return True
        
        return False
    
    def should_retry_result(self, result: Any) -> bool:
        """Check if result should trigger a retry.
        
        Args:
            result: The operation result
            
        Returns:
            True if should retry
        """
        if self.config.should_retry_result:
            return self.config.should_retry_result(result)
        return False
    
    async def execute(
        self,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> RetryResult[T]:
        """Execute an operation with retry logic.
        
        Args:
            operation: The operation to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            RetryResult with execution details
        """
        start_time = datetime.now()
        attempts: List[RetryAttempt] = []
        last_exception: Optional[Exception] = None
        
        for attempt_number in range(1, self.config.max_attempts + 1):
            # Check if stop was requested
            if self._stop_requested:
                return RetryResult(
                    success=False,
                    result=None,
                    final_exception=last_exception,
                    total_attempts=attempt_number - 1,
                    total_time=datetime.now() - start_time,
                    attempts=attempts
                )
            
            delay = 0.0
            if attempt_number > 1:
                delay = self.calculate_delay(attempt_number - 1)
                if delay > 0:
                    if self.config.on_retry_callback:
                        self.config.on_retry_callback(attempt_number - 1, last_exception, delay)
                    await asyncio.sleep(delay)
            
            try:
                # Execute operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Check if result indicates failure
                if self.should_retry_result(result):
                    attempt = RetryAttempt(
                        attempt_number=attempt_number,
                        timestamp=datetime.now(),
                        exception=None,
                        delay_before=delay,
                        success=False,
                        result=result
                    )
                    attempts.append(attempt)
                    
                    if attempt_number < self.config.max_attempts:
                        continue
                    else:
                        # Max attempts reached
                        if self.config.on_failure_callback:
                            self.config.on_failure_callback(attempt_number, Exception("Max retries exceeded"))
                        return RetryResult(
                            success=False,
                            result=result,
                            final_exception=None,
                            total_attempts=attempt_number,
                            total_time=datetime.now() - start_time,
                            attempts=attempts
                        )
                
                # Success!
                if self.config.on_success_callback:
                    self.config.on_success_callback(attempt_number, result)
                
                attempt = RetryAttempt(
                    attempt_number=attempt_number,
                    timestamp=datetime.now(),
                    exception=None,
                    delay_before=delay,
                    success=True,
                    result=result
                )
                attempts.append(attempt)
                
                return RetryResult(
                    success=True,
                    result=result,
                    final_exception=None,
                    total_attempts=attempt_number,
                    total_time=datetime.now() - start_time,
                    attempts=attempts
                )
                
            except Exception as e:
                last_exception = e
                
                attempt = RetryAttempt(
                    attempt_number=attempt_number,
                    timestamp=datetime.now(),
                    exception=e,
                    delay_before=delay,
                    success=False
                )
                attempts.append(attempt)
                
                # Check if we should retry this exception
                if not self.should_retry_exception(e):
                    if self.config.on_failure_callback:
                        self.config.on_failure_callback(attempt_number, e)
                    return RetryResult(
                        success=False,
                        result=None,
                        final_exception=e,
                        total_attempts=attempt_number,
                        total_time=datetime.now() - start_time,
                        attempts=attempts
                    )
                
                # Check if we have more attempts
                if attempt_number >= self.config.max_attempts:
                    if self.config.on_failure_callback:
                        self.config.on_failure_callback(attempt_number, e)
                    return RetryResult(
                        success=False,
                        result=None,
                        final_exception=e,
                        total_attempts=attempt_number,
                        total_time=datetime.now() - start_time,
                        attempts=attempts
                    )
        
        # Should not reach here, but just in case
        return RetryResult(
            success=False,
            result=None,
            final_exception=last_exception,
            total_attempts=len(attempts),
            total_time=datetime.now() - start_time,
            attempts=attempts
        )
    
    def execute_sync(
        self,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> RetryResult[T]:
        """Synchronous version of execute."""
        return asyncio.run(self.execute(operation, *args, **kwargs))
    
    def __call__(self, operation: Callable[..., T]) -> Callable[..., RetryResult[T]]:
        """Use as a decorator."""
        @functools.wraps(operation)
        async def wrapper(*args, **kwargs):
            return await self.execute(operation, *args, **kwargs)
        return wrapper


class InfiniteRetryManager(RetryManager):
    """Retry manager that retries indefinitely until success or stop requested.
    
    This is useful for the Agent's main loop where we want to keep trying
    until the user explicitly stops the operation.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize infinite retry manager."""
        super().__init__(config)
        # Override max_attempts to effectively infinite
        if self.config:
            self.config.max_attempts = 999999
    
    async def execute_with_recovery(
        self,
        operation: Callable[..., T],
        recovery_manager,
        *args,
        **kwargs
    ) -> RetryResult[T]:
        """Execute with automatic recovery on failure.
        
        Args:
            operation: The operation to execute
            recovery_manager: ErrorRecoveryManager instance
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            RetryResult
        """
        attempt = 0
        start_time = datetime.now()
        attempts: List[RetryAttempt] = []
        
        while not self._stop_requested:
            attempt += 1
            delay = self.calculate_delay(attempt)
            
            if delay > 0 and attempt > 1:
                await asyncio.sleep(delay)
            
            try:
                # Try to execute
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Success!
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    timestamp=datetime.now(),
                    exception=None,
                    delay_before=delay,
                    success=True,
                    result=result
                )
                attempts.append(attempt_record)
                
                return RetryResult(
                    success=True,
                    result=result,
                    final_exception=None,
                    total_attempts=attempt,
                    total_time=datetime.now() - start_time,
                    attempts=attempts
                )
                
            except Exception as e:
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    timestamp=datetime.now(),
                    exception=e,
                    delay_before=delay,
                    success=False
                )
                attempts.append(attempt_record)
                
                # Try to recover
                recovery_result = await recovery_manager.recover(
                    e,
                    error_context={"operation": operation.__name__, "attempt": attempt},
                    current_test_code=kwargs.get("current_test_code"),
                    target_class_info=kwargs.get("target_class_info")
                )
                
                if not recovery_result.get("should_continue", True):
                    # Recovery manager says we should stop
                    return RetryResult(
                        success=False,
                        result=None,
                        final_exception=e,
                        total_attempts=attempt,
                        total_time=datetime.now() - start_time,
                        attempts=attempts
                    )
                
                # Apply recovery action
                action = recovery_result.get("action", "retry")
                
                if action == "retry":
                    continue  # Just retry
                elif action == "fix":
                    # Apply the fix and retry
                    fixed_code = recovery_result.get("fixed_code")
                    if fixed_code and "current_test_code" in kwargs:
                        kwargs["current_test_code"] = fixed_code
                    continue
                elif action == "skip":
                    # Skip this operation and return partial success
                    return RetryResult(
                        success=True,  # Considered success (skipped)
                        result=None,
                        final_exception=None,
                        total_attempts=attempt,
                        total_time=datetime.now() - start_time,
                        attempts=attempts
                    )
                elif action == "reset":
                    # Clear history and start fresh
                    recovery_manager.clear_history()
                    continue
                elif action == "fallback":
                    # Use alternative approach
                    continue
                else:
                    # Unknown action, just retry
                    continue
        
        # Stop requested
        return RetryResult(
            success=False,
            result=None,
            final_exception=None,
            total_attempts=attempt,
            total_time=datetime.now() - start_time,
            attempts=attempts
        )


def with_retry(
    max_attempts: int = 10,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for adding retry logic to functions.
    
    Args:
        max_attempts: Maximum number of attempts
        strategy: Retry strategy
        base_delay: Base delay between retries
        exceptions: Exceptions to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., RetryResult[T]]:
        config = RetryConfig(
            max_attempts=max_attempts,
            strategy=strategy,
            base_delay=base_delay,
            exceptions_to_retry=exceptions
        )
        manager = RetryManager(config)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await manager.execute(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return manager.execute_sync(func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator