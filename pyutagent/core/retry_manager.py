"""Unified retry manager with advanced retry strategies.

This module provides enhanced retry capabilities with support for:
- Multiple retry strategies (immediate, fixed, exponential, adaptive)
- Circuit breaker pattern for fault tolerance
- Conditional retry based on error types
- Timeout handling
- Async and sync support
- Infinite retry for agent loops

This module consolidates the functionality from:
- agent/retry_manager.py (InfiniteRetryManager, adaptive strategies)
- tools/retry_manager.py (tenacity integration, circuit breaker)
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Any,
    TypeVar,
    Generic,
    Tuple,
    Type,
    Union,
    Coroutine,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategies."""
    IMMEDIATE = auto()
    FIXED = auto()
    FIXED_DELAY = auto()
    EXPONENTIAL = auto()
    EXPONENTIAL_BACKOFF = auto()
    LINEAR_BACKOFF = auto()
    RANDOM = auto()
    RANDOM_JITTER = auto()
    ADAPTIVE = auto()
    EXPONENTIAL_JITTER = auto()
    STOP = auto()


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class RetryCondition(Enum):
    """When to retry."""
    ALWAYS = auto()
    ON_EXCEPTION = auto()
    ON_RESULT = auto()
    CUSTOM = auto()


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 10
    strategy: RetryStrategy = RetryStrategy.ADAPTIVE
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_range: Tuple[float, float] = (0.0, 1.0)
    retry_condition: RetryCondition = RetryCondition.ON_EXCEPTION
    exceptions_to_retry: Tuple[Type[Exception], ...] = (Exception,)
    exceptions_to_ignore: Tuple[Type[Exception], ...] = (KeyboardInterrupt, SystemExit)
    should_retry_result: Optional[Callable[[Any], bool]] = None
    on_retry_callback: Optional[Callable[[int, Exception, float], None]] = None
    on_success_callback: Optional[Callable[[int, Any], None]] = None
    on_failure_callback: Optional[Callable[[int, Exception], None]] = None
    retryable_exceptions: Optional[List[Type[Exception]]] = None

    def __post_init__(self):
        if self.retryable_exceptions and not self.exceptions_to_retry:
            self.exceptions_to_retry = tuple(self.retryable_exceptions)
        elif not self.exceptions_to_retry or self.exceptions_to_retry == (Exception,):
            self.exceptions_to_retry = (
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                OSError,
            )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


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
        return self.success and self.result is not None


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
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
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    async def _on_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info(f"Circuit {self.name} CLOSED (recovered)")
            else:
                self.failure_count = max(0, self.failure_count - 1)

    async def _on_failure(self):
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
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None

    def get_state(self) -> CircuitState:
        return self.state

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "half_open_calls": self.half_open_calls,
            "last_failure_time": self.last_failure_time
        }


class RetryManager:
    """Unified retry manager with multiple strategies.

    Supports both sync and async operations, circuit breaker pattern,
    and various retry strategies.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._stop_requested = False
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

    def stop(self):
        self._stop_requested = True

    def reset(self):
        self._stop_requested = False

    def should_stop(self) -> bool:
        return self._stop_requested

    def calculate_delay(self, attempt_number: int) -> float:
        strategy = self.config.strategy
        base = self.config.base_delay
        max_delay = self.config.max_delay

        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0

        elif strategy in (RetryStrategy.FIXED, RetryStrategy.FIXED_DELAY):
            return base

        elif strategy in (RetryStrategy.EXPONENTIAL, RetryStrategy.EXPONENTIAL_BACKOFF):
            delay = base * (self.config.exponential_base ** (attempt_number - 1))
            return min(delay, max_delay)

        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base * attempt_number
            return min(delay, max_delay)

        elif strategy == RetryStrategy.RANDOM_JITTER or strategy == RetryStrategy.RANDOM:
            delay = base * (self.config.exponential_base ** (attempt_number - 1))
            jitter = random.uniform(*self.config.jitter_range)
            return min(delay + jitter, max_delay)

        elif strategy == RetryStrategy.EXPONENTIAL_JITTER:
            delay = base * (self.config.exponential_base ** (attempt_number - 1))
            jitter = random.uniform(0, 1)
            return min(delay + jitter, max_delay)

        elif strategy == RetryStrategy.ADAPTIVE:
            delay = base * (self.config.exponential_base ** (attempt_number - 1))
            jitter = random.uniform(0, delay * 0.1)
            return min(delay + jitter, max_delay)

        elif strategy == RetryStrategy.STOP:
            return 0.0

        return base

    def should_retry_exception(self, exception: Exception) -> bool:
        if isinstance(exception, self.config.exceptions_to_ignore):
            return False

        if isinstance(exception, self.config.exceptions_to_retry):
            return True

        return False

    def should_retry_result(self, result: Any) -> bool:
        if self.config.should_retry_result:
            return self.config.should_retry_result(result)
        return False

    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout
                )
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]

    def get_wait_strategy(self, config: Optional[RetryConfig] = None) -> Callable[[int], float]:
        """Get wait strategy function for backward compatibility.

        Args:
            config: RetryConfig to use (defaults to self.config)

        Returns:
            Function that calculates delay for a given attempt number
        """
        cfg = config or self.config
        return lambda attempt: self.calculate_delay(attempt)

    def get_stop_strategy(self, config: Optional[RetryConfig] = None) -> Callable[[int, Exception], bool]:
        """Get stop strategy function for backward compatibility.

        Args:
            config: RetryConfig to use (defaults to self.config)

        Returns:
            Function that returns True if should stop
        """
        cfg = config or self.config
        def stop_strategy(attempt: int, exception: Exception) -> bool:
            return attempt >= cfg.max_attempts
        return stop_strategy

    def get_retry_strategy(self, config: Optional[RetryConfig] = None) -> Callable[[Exception], bool]:
        """Get retry strategy function for backward compatibility.

        Args:
            config: RetryConfig to use (defaults to self.config)

        Returns:
            Function that returns True if exception is retryable
        """
        cfg = config or self.config
        return lambda exc: isinstance(exc, cfg.exceptions_to_retry)

    async def call_with_circuit_breaker(
        self,
        circuit_name: str,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute operation with circuit breaker for backward compatibility.

        Args:
            circuit_name: Name for the circuit breaker
            operation: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from the operation
        """
        return await self.execute_with_circuit_breaker(circuit_name, operation, *args, **kwargs)

    async def execute(
        self,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> RetryResult[T]:
        start_time = datetime.now()
        attempts: List[RetryAttempt] = []
        last_exception: Optional[Exception] = None

        for attempt_number in range(1, self.config.max_attempts + 1):
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
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)

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

        return RetryResult(
            success=False,
            result=None,
            final_exception=last_exception,
            total_attempts=len(attempts),
            total_time=datetime.now() - start_time,
            attempts=attempts
        )

    async def execute_with_circuit_breaker(
        self,
        circuit_name: str,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        breaker = self.get_circuit_breaker(circuit_name)
        return await breaker.call(operation, *args, **kwargs)

    def execute_sync(
        self,
        operation: Callable[..., T],
        *args,
        **kwargs
    ) -> RetryResult[T]:
        return asyncio.run(self.execute(operation, *args, **kwargs))

    def __call__(self, operation: Callable[..., T]) -> Callable[..., RetryResult[T]]:
        @wraps(operation)
        async def wrapper(*args, **kwargs):
            return await self.execute(operation, *args, **kwargs)
        return wrapper


class InfiniteRetryManager(RetryManager):
    """Retry manager that retries indefinitely until success or stop requested.

    This is useful for the Agent's main loop where we want to keep trying
    until the user explicitly stops the operation.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        super().__init__(config)
        if self.config:
            self.config.max_attempts = 999999

    async def execute_with_recovery(
        self,
        operation: Callable[..., T],
        recovery_manager: Any,
        *args,
        **kwargs
    ) -> RetryResult[T]:
        attempt = 0
        start_time = datetime.now()
        attempts: List[RetryAttempt] = []

        while not self._stop_requested:
            attempt += 1
            delay = self.calculate_delay(attempt)

            if delay > 0 and attempt > 1:
                await asyncio.sleep(delay)

            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)

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

                recovery_result = await recovery_manager.recover(
                    e,
                    error_context={"operation": operation.__name__, "attempt": attempt},
                    current_test_code=kwargs.get("current_test_code"),
                    target_class_info=kwargs.get("target_class_info")
                )

                if not recovery_result.get("should_continue", True):
                    return RetryResult(
                        success=False,
                        result=None,
                        final_exception=e,
                        total_attempts=attempt,
                        total_time=datetime.now() - start_time,
                        attempts=attempts
                    )

                action = recovery_result.get("action", "retry")

                if action == "retry":
                    continue
                elif action == "fix":
                    fixed_code = recovery_result.get("fixed_code")
                    if fixed_code and "current_test_code" in kwargs:
                        kwargs["current_test_code"] = fixed_code
                    continue
                elif action == "skip":
                    return RetryResult(
                        success=True,
                        result=None,
                        final_exception=None,
                        total_attempts=attempt,
                        total_time=datetime.now() - start_time,
                        attempts=attempts
                    )
                elif action == "reset":
                    recovery_manager.clear_history()
                    continue
                elif action == "fallback":
                    continue
                else:
                    continue

        return RetryResult(
            success=False,
            result=None,
            final_exception=None,
            total_attempts=attempt,
            total_time=datetime.now() - start_time,
            attempts=attempts
        )


class TimeoutManager:
    """Manager for timeout handling."""

    @staticmethod
    async def with_timeout(
        coro: Coroutine,
        timeout: float,
        timeout_message: str = "Operation timed out"
    ) -> Any:
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


def with_retry(
    max_attempts: int = 10,
    strategy: RetryStrategy = RetryStrategy.ADAPTIVE,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """Decorator for adding retry logic to functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., RetryResult[T]]:
        config = RetryConfig(
            max_attempts=max_attempts,
            strategy=strategy,
            base_delay=base_delay,
            max_delay=max_delay,
            exceptions_to_retry=exceptions
        )
        manager = RetryManager(config)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await manager.execute(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return manager.execute_sync(func, *args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator for circuit breaker pattern."""
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


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
):
    """Decorator for retry with exponential backoff.

    Returns the result directly (not RetryResult) for backward compatibility.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            strategy=RetryStrategy.EXPONENTIAL_JITTER
        )
        manager = RetryManager(config)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await manager.execute(func, *args, **kwargs)
            if result.success:
                return result.result
            raise result.final_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = manager.execute_sync(func, *args, **kwargs)
            if result.success:
                return result.result
            raise result.final_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


_default_retry_manager: Optional[RetryManager] = None


def get_retry_manager() -> RetryManager:
    """Get global retry manager instance."""
    global _default_retry_manager
    if _default_retry_manager is None:
        _default_retry_manager = RetryManager()
    return _default_retry_manager


def create_retry_manager(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.ADAPTIVE
) -> RetryManager:
    """Create a retry manager with common settings.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay for backoff
        max_delay: Maximum delay
        strategy: Retry strategy

    Returns:
        Configured RetryManager instance
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy
    )
    return RetryManager(config)


class AsyncRetryWithBackoff:
    """Async retry with exponential backoff.

    This class provides a simple interface for retrying async operations
    with exponential backoff and optional jitter.

    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter to delays
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self._attempt_count = 0

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Args:
            attempt: The attempt number (1-based)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay

    async def execute(
        self,
        func: Callable,
        should_retry: Optional[Callable[[Exception], bool]] = None,
        on_retry: Optional[Callable[[int, Exception, float], None]] = None
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            func: Async function to execute
            should_retry: Optional function to determine if retry should occur
            on_retry: Optional callback for retry events

        Returns:
            Result from the function

        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None

        for attempt in range(1, self.max_attempts + 1):
            self._attempt_count = attempt

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()
                return result

            except Exception as e:
                last_exception = e

                if should_retry and not should_retry(e):
                    raise

                if attempt < self.max_attempts:
                    delay = self._calculate_delay(attempt)
                    if on_retry:
                        on_retry(attempt, e, delay)
                    await asyncio.sleep(delay)

        raise last_exception

    @property
    def attempt_count(self) -> int:
        """Get current attempt count."""
        return self._attempt_count
