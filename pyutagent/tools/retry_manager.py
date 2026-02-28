"""Retry manager with advanced retry strategies.

This module provides enhanced retry capabilities using tenacity and backoff,
with support for:
- Exponential backoff with jitter
- Circuit breaker pattern
- Conditional retry based on error types
- Timeout handling
- Async support
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Any, Type, List, Union, Coroutine
from functools import wraps
import time

from tenacity import (
    retry as tenacity_retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random,
    wait_fixed,
    retry_if_exception_type,
    retry_if_result,
    before_sleep_log,
    after_log,
    RetryCallState
)
import backoff

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""
    EXPONENTIAL = auto()      # Exponential backoff
    FIXED = auto()            # Fixed interval
    LINEAR = auto()           # Linear increase
    RANDOM = auto()           # Random interval
    EXPONENTIAL_JITTER = auto()  # Exponential with jitter


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing if recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    max_delay: float = 60.0
    base_delay: float = 1.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    retryable_result: Optional[Callable[[Any], bool]] = None
    stop_on_result: Optional[Callable[[Any], bool]] = None
    on_retry: Optional[Callable[[RetryCallState], None]] = None
    on_success: Optional[Callable[[RetryCallState], None]] = None
    on_failure: Optional[Callable[[RetryCallState], None]] = None
    
    def __post_init__(self):
        """Set default retryable exceptions if not specified."""
        if not self.retryable_exceptions:
            self.retryable_exceptions = [
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                OSError,
            ]


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit {self.name} entering HALF_OPEN state")
                else:
                    raise Exception(f"Circuit {self.name} is OPEN")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise Exception(f"Circuit {self.name} HALF_OPEN limit reached")
                self.half_open_calls += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info(f"Circuit {self.name} CLOSED (recovered)")
            else:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name} OPEN (half-open test failed)")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name} OPEN (threshold reached)")
    
    def _reset(self):
        """Reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "half_open_calls": self.half_open_calls,
            "last_failure_time": self.last_failure_time
        }


class RetryManager:
    """Manager for retry operations with multiple strategies."""
    
    def __init__(self):
        """Initialize retry manager."""
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.default_config = RetryConfig()
    
    def get_wait_strategy(self, config: RetryConfig):
        """Get wait strategy based on configuration.
        
        Args:
            config: Retry configuration
            
        Returns:
            Wait strategy for tenacity
        """
        if config.strategy == RetryStrategy.EXPONENTIAL:
            return wait_exponential(
                multiplier=config.base_delay,
                max=config.max_delay
            )
        elif config.strategy == RetryStrategy.EXPONENTIAL_JITTER:
            return wait_exponential(
                multiplier=config.base_delay,
                max=config.max_delay
            ) + wait_random(0, 1)
        elif config.strategy == RetryStrategy.FIXED:
            return wait_fixed(config.base_delay)
        elif config.strategy == RetryStrategy.RANDOM:
            return wait_random(config.base_delay, config.max_delay)
        elif config.strategy == RetryStrategy.LINEAR:
            return wait_fixed(config.base_delay)
        else:
            return wait_exponential(
                multiplier=config.base_delay,
                max=config.max_delay
            )
    
    def get_stop_strategy(self, config: RetryConfig):
        """Get stop strategy based on configuration.
        
        Args:
            config: Retry configuration
            
        Returns:
            Stop strategy for tenacity
        """
        return stop_after_attempt(config.max_attempts)
    
    def get_retry_strategy(self, config: RetryConfig):
        """Get retry strategy based on configuration.
        
        Args:
            config: Retry configuration
            
        Returns:
            Retry strategy for tenacity
        """
        if config.retryable_exceptions:
            return retry_if_exception_type(tuple(config.retryable_exceptions))
        elif config.retryable_result:
            return retry_if_result(config.retryable_result)
        else:
            return retry_if_exception_type((Exception,))
    
    def create_retry_decorator(self, config: Optional[RetryConfig] = None):
        """Create a retry decorator.
        
        Args:
            config: Retry configuration
            
        Returns:
            Retry decorator
        """
        config = config or self.default_config
        
        return tenacity_retry(
            stop=self.get_stop_strategy(config),
            wait=self.get_wait_strategy(config),
            retry=self.get_retry_strategy(config),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.INFO),
            reraise=True
        )
    
    def retry(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """Decorator for retry functionality.
        
        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            strategy: Retry strategy
            retryable_exceptions: List of exceptions to retry on
            
        Returns:
            Decorator function
        """
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=strategy,
            retryable_exceptions=retryable_exceptions or [
                ConnectionError, TimeoutError, asyncio.TimeoutError
            ]
        )
        
        return self.create_retry_decorator(config)
    
    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
            failure_threshold: Failure threshold (used if config not provided)
            recovery_timeout: Recovery timeout in seconds (used if config not provided)
            
        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout
                )
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    async def call_with_circuit_breaker(
        self,
        circuit_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Call a function with circuit breaker protection.
        
        Args:
            circuit_name: Circuit breaker name
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        breaker = self.get_circuit_breaker(circuit_name)
        return await breaker.call(func, *args, **kwargs)


# Async retry with backoff
class AsyncRetryWithBackoff:
    """Async retry with exponential backoff and jitter."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """Initialize async retry.
        
        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential calculation
            jitter: Whether to add random jitter
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    async def execute(
        self,
        func: Callable,
        *args,
        should_retry: Optional[Callable[[Exception], bool]] = None,
        **kwargs
    ) -> Any:
        """Execute function with retry.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            should_retry: Function to determine if error is retryable
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_error = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_error = e
                
                if should_retry and not should_retry(e):
                    logger.warning(f"Error not retryable: {e}")
                    raise
                
                if attempt < self.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_attempts} attempts failed")
        
        raise last_error
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt.
        
        Args:
            attempt: Attempt number
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)
        
        return delay


# Timeout wrapper
class TimeoutManager:
    """Manager for timeout handling."""
    
    @staticmethod
    async def with_timeout(
        coro: Coroutine,
        timeout: float,
        timeout_message: str = "Operation timed out"
    ) -> Any:
        """Execute coroutine with timeout.
        
        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds
            timeout_message: Message for timeout exception
            
        Returns:
            Coroutine result
            
        Raises:
            asyncio.TimeoutError: If timeout is reached
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(timeout_message)
            raise asyncio.TimeoutError(timeout_message)
    
    @staticmethod
    def timeout(
        seconds: float,
        timeout_message: str = "Operation timed out"
    ):
        """Decorator for timeout.
        
        Args:
            seconds: Timeout in seconds
            timeout_message: Message for timeout exception
            
        Returns:
            Decorator
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await TimeoutManager.with_timeout(
                    func(*args, **kwargs),
                    seconds,
                    timeout_message
                )
            return wrapper
        return decorator


# Convenience functions
def create_retry_manager() -> RetryManager:
    """Create a retry manager.
    
    Returns:
        RetryManager instance
    """
    return RetryManager()


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Decorator for retry with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay
        max_delay: Maximum delay
        
    Returns:
        Decorator
    """
    manager = RetryManager()
    return manager.retry(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=RetryStrategy.EXPONENTIAL_JITTER
    )


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator for circuit breaker.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Failure threshold
        recovery_timeout: Recovery timeout
        
    Returns:
        Decorator
    """
    manager = RetryManager()
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = manager.get_circuit_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


# Global retry manager instance
_default_retry_manager: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """Get global retry manager instance.
    
    Returns:
        RetryManager instance
    """
    global _default_retry_manager
    if _default_retry_manager is None:
        _default_retry_manager = RetryManager()
    return _default_retry_manager
