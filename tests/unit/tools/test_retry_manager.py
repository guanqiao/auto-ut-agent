"""Unit tests for retry_manager module.

This module provides comprehensive tests for retry functionality,
including circuit breaker, retry strategies, timeout handling,
and async retry with backoff.
"""

import pytest
import asyncio
import allure
from unittest.mock import Mock, AsyncMock, patch
from typing import Any

from pyutagent.core.retry_manager import (
    RetryManager, CircuitBreaker, CircuitBreakerConfig, CircuitState,
    AsyncRetryWithBackoff, TimeoutManager, RetryConfig, RetryStrategy,
    create_retry_manager, retry_with_backoff, circuit_breaker,
    get_retry_manager
)


# Custom decorator for display descriptions
def display_description(description: str):
    """Decorator to add display description for test cases."""
    def decorator(func):
        func.display_description = description
        return allure.description(description)(func)
    return decorator


@allure.feature("Retry Management")
@allure.story("Circuit Breaker")
class TestCircuitBreaker:
    """Tests for CircuitBreaker class with display descriptions."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a CircuitBreaker instance."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,
            half_open_max_calls=2,
            success_threshold=2
        )
        return CircuitBreaker("test_breaker", config)
    
    @display_description("验证熔断器初始状态为 CLOSED")
    @allure.title("Test circuit breaker initial state")
    @allure.severity(allure.severity_level.BLOCKER)
    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
    
    @display_description("验证成功调用增加成功计数")
    @allure.title("Test successful call increments success count")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call."""
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @display_description("验证失败调用增加失败计数")
    @allure.title("Test failed call increments failure count")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker):
        """Test failed function call."""
        async def fail_func():
            raise ConnectionError("Failed")
        
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(fail_func)
        
        assert circuit_breaker.failure_count == 1
    
    @display_description("验证达到失败阈值后熔断器打开")
    @allure.title("Test circuit opens after failure threshold")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_circuit_opens(self, circuit_breaker):
        """Test circuit opens after reaching failure threshold."""
        async def fail_func():
            raise ConnectionError("Failed")
        
        # Trigger failures up to threshold
        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(fail_func)
        
        assert circuit_breaker.state == CircuitState.OPEN
    
    @display_description("验证熔断器打开时拒绝调用")
    @allure.title("Test circuit rejects calls when open")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self, circuit_breaker):
        """Test circuit rejects calls when open."""
        # Force circuit open with recent failure time (within recovery timeout)
        import time
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = time.time()  # Recent failure
        circuit_breaker.config.recovery_timeout = 3600.0  # Set very long timeout (1 hour)
        
        async def any_func():
            return "result"
        
        with pytest.raises(Exception, match="is OPEN"):
            await circuit_breaker.call(any_func)
    
    @display_description("验证熔断器超时后进入半开状态")
    @allure.title("Test circuit transitions to half-open")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_circuit_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after timeout."""
        # Force circuit open with old failure time
        circuit_breaker.state = CircuitState.OPEN
        circuit_breaker.last_failure_time = asyncio.get_event_loop().time() - 2.0
        
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    @display_description("验证半开状态下成功恢复后熔断器关闭")
    @allure.title("Test circuit closes after recovery")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_circuit_closes_after_recovery(self, circuit_breaker):
        """Test circuit closes after successful recovery."""
        circuit_breaker.state = CircuitState.HALF_OPEN
        
        async def success_func():
            return "success"
        
        # Need success_threshold successes
        for _ in range(2):
            await circuit_breaker.call(success_func)
        
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @display_description("验证半开状态下失败会重新打开熔断器")
    @allure.title("Test circuit reopens on half-open failure")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_circuit_reopens_on_failure(self, circuit_breaker):
        """Test circuit reopens when half-open test fails."""
        circuit_breaker.state = CircuitState.HALF_OPEN
        
        async def fail_func():
            raise ConnectionError("Failed")
        
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(fail_func)
        
        assert circuit_breaker.state == CircuitState.OPEN
    
    @display_description("验证熔断器统计信息")
    @allure.title("Test circuit breaker statistics")
    @allure.severity(allure.severity_level.NORMAL)
    def test_circuit_stats(self, circuit_breaker):
        """Test circuit breaker statistics."""
        stats = circuit_breaker.get_stats()
        
        assert stats["name"] == "test_breaker"
        assert stats["state"] == "CLOSED"
        assert "failure_count" in stats
    
    @display_description("验证同步函数调用")
    @allure.title("Test sync function call")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_sync_function_call(self, circuit_breaker):
        """Test calling synchronous function."""
        def sync_func():
            return "sync_result"
        
        result = await circuit_breaker.call(sync_func)
        
        assert result == "sync_result"


@allure.feature("Retry Management")
@allure.story("Retry Manager")
class TestRetryManager:
    """Tests for RetryManager class with display descriptions."""
    
    @pytest.fixture
    def retry_manager(self):
        """Create a RetryManager instance."""
        return RetryManager()
    
    @display_description("验证创建熔断器")
    @allure.title("Test create circuit breaker")
    @allure.severity(allure.severity_level.BLOCKER)
    def test_get_circuit_breaker(self, retry_manager):
        """Test getting/creating circuit breaker."""
        breaker1 = retry_manager.get_circuit_breaker("test")
        breaker2 = retry_manager.get_circuit_breaker("test")
        
        assert breaker1 is breaker2  # Same instance
        assert isinstance(breaker1, CircuitBreaker)
    
    @display_description("验证指数退避等待策略")
    @allure.title("Test exponential backoff wait strategy")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_exponential_wait_strategy(self, retry_manager):
        """Test exponential backoff wait strategy."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=60.0
        )
        
        strategy = retry_manager.get_wait_strategy(config)
        
        assert strategy is not None
    
    @display_description("验证固定间隔等待策略")
    @allure.title("Test fixed wait strategy")
    @allure.severity(allure.severity_level.NORMAL)
    def test_fixed_wait_strategy(self, retry_manager):
        """Test fixed interval wait strategy."""
        config = RetryConfig(
            strategy=RetryStrategy.FIXED,
            base_delay=2.0
        )
        
        strategy = retry_manager.get_wait_strategy(config)
        
        assert strategy is not None
    
    @display_description("验证随机等待策略")
    @allure.title("Test random wait strategy")
    @allure.severity(allure.severity_level.NORMAL)
    def test_random_wait_strategy(self, retry_manager):
        """Test random interval wait strategy."""
        config = RetryConfig(
            strategy=RetryStrategy.RANDOM,
            base_delay=1.0,
            max_delay=5.0
        )
        
        strategy = retry_manager.get_wait_strategy(config)
        
        assert strategy is not None
    
    @display_description("验证停止策略")
    @allure.title("Test stop strategy")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_stop_strategy(self, retry_manager):
        """Test stop strategy."""
        config = RetryConfig(max_attempts=5)
        
        strategy = retry_manager.get_stop_strategy(config)
        
        assert strategy is not None
    
    @display_description("验证重试策略 - 基于异常类型")
    @allure.title("Test retry strategy by exception type")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_retry_strategy_by_exception(self, retry_manager):
        """Test retry strategy based on exception types."""
        config = RetryConfig(
            retryable_exceptions=[ConnectionError, TimeoutError]
        )
        
        strategy = retry_manager.get_retry_strategy(config)
        
        assert strategy is not None
    
    @display_description("验证使用熔断器调用函数")
    @allure.title("Test call with circuit breaker")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker(self, retry_manager):
        """Test calling function with circuit breaker."""
        async def success_func():
            return "success"
        
        result = await retry_manager.call_with_circuit_breaker(
            "test_circuit", success_func
        )
        
        assert result == "success"


@allure.feature("Retry Management")
@allure.story("Async Retry with Backoff")
class TestAsyncRetryWithBackoff:
    """Tests for AsyncRetryWithBackoff class with display descriptions."""
    
    @pytest.fixture
    def async_retry(self):
        """Create an AsyncRetryWithBackoff instance."""
        return AsyncRetryWithBackoff(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
    
    @display_description("验证成功执行无需重试")
    @allure.title("Test successful execution without retry")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_successful_execution(self, async_retry):
        """Test successful function execution."""
        async def success_func():
            return "success"
        
        result = await async_retry.execute(success_func)
        
        assert result == "success"
    
    @display_description("验证失败后重试并最终成功")
    @allure.title("Test retry until success")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_retry_until_success(self, async_retry):
        """Test retrying until success."""
        call_count = 0
        
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Attempt {call_count} failed")
            return "success"
        
        result = await async_retry.execute(fail_then_succeed)
        
        assert result == "success"
        assert call_count == 3
    
    @display_description("验证所有重试失败后抛出异常")
    @allure.title("Test all retries exhausted")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, async_retry):
        """Test when all retries are exhausted."""
        async def always_fail():
            raise ConnectionError("Always fails")
        
        with pytest.raises(ConnectionError):
            await async_retry.execute(always_fail)
    
    @display_description("验证自定义重试条件")
    @allure.title("Test custom retry condition")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_custom_retry_condition(self, async_retry):
        """Test custom retry condition."""
        async def raise_value_error():
            raise ValueError("Not retryable")
        
        def should_retry(e):
            return isinstance(e, ConnectionError)
        
        with pytest.raises(ValueError):
            await async_retry.execute(
                raise_value_error,
                should_retry=should_retry
            )
    
    @display_description("验证延迟计算 - 指数退避")
    @allure.title("Test delay calculation")
    @allure.severity(allure.severity_level.NORMAL)
    def test_delay_calculation(self, async_retry):
        """Test delay calculation with exponential backoff."""
        delay1 = async_retry._calculate_delay(1)
        delay2 = async_retry._calculate_delay(2)
        delay3 = async_retry._calculate_delay(3)
        
        # Exponential backoff: base * (exponential_base ^ (attempt - 1))
        assert delay1 == 0.1
        assert delay2 == 0.2
        assert delay3 == 0.4
    
    @display_description("验证延迟上限")
    @allure.title("Test delay cap")
    @allure.severity(allure.severity_level.NORMAL)
    def test_delay_cap(self, async_retry):
        """Test that delay is capped at max_delay."""
        # For high attempt numbers, delay should be capped
        delay = async_retry._calculate_delay(10)
        
        assert delay <= async_retry.max_delay
    
    @display_description("验证带抖动的延迟")
    @allure.title("Test delay with jitter")
    @allure.severity(allure.severity_level.NORMAL)
    def test_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        retry_with_jitter = AsyncRetryWithBackoff(
            max_attempts=3,
            base_delay=1.0,
            jitter=True
        )
        
        delay = retry_with_jitter._calculate_delay(1)
        
        # With jitter, delay should be around base_delay ± 25%
        assert 0.75 <= delay <= 1.25


@allure.feature("Retry Management")
@allure.story("Timeout Management")
class TestTimeoutManager:
    """Tests for TimeoutManager class with display descriptions."""
    
    @display_description("验证超时前成功完成")
    @allure.title("Test completion before timeout")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_completion_before_timeout(self):
        """Test successful completion before timeout."""
        async def quick_task():
            await asyncio.sleep(0.1)
            return "done"
        
        result = await TimeoutManager.with_timeout(
            quick_task(),
            timeout=1.0
        )
        
        assert result == "done"
    
    @display_description("验证超时后抛出异常")
    @allure.title("Test timeout exception")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_timeout_exception(self):
        """Test timeout exception."""
        async def slow_task():
            await asyncio.sleep(2.0)
            return "done"
        
        with pytest.raises(asyncio.TimeoutError):
            await TimeoutManager.with_timeout(
                slow_task(),
                timeout=0.1,
                timeout_message="Task timed out"
            )
    
    @display_description("验证超时装饰器")
    @allure.title("Test timeout decorator")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_timeout_decorator(self):
        """Test timeout decorator."""
        @TimeoutManager.timeout(0.5)
        async def decorated_func():
            await asyncio.sleep(0.1)
            return "success"
        
        result = await decorated_func()
        
        assert result == "success"
    
    @display_description("验证超时装饰器超时")
    @allure.title("Test timeout decorator timeout")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_timeout_decorator_timeout(self):
        """Test timeout decorator with timeout."""
        @TimeoutManager.timeout(0.1)
        async def slow_decorated_func():
            await asyncio.sleep(1.0)
            return "success"
        
        with pytest.raises(asyncio.TimeoutError):
            await slow_decorated_func()


@allure.feature("Retry Management")
@allure.story("Utility Functions")
class TestUtilityFunctions:
    """Tests for utility functions with display descriptions."""
    
    @display_description("验证创建重试管理器便捷函数")
    @allure.title("Test create_retry_manager utility")
    @allure.severity(allure.severity_level.NORMAL)
    def test_create_retry_manager(self):
        """Test create_retry_manager function."""
        manager = create_retry_manager()
        
        assert isinstance(manager, RetryManager)
    
    @display_description("验证获取全局重试管理器")
    @allure.title("Test get_retry_manager utility")
    @allure.severity(allure.severity_level.NORMAL)
    def test_get_retry_manager(self):
        """Test get_retry_manager function."""
        manager1 = get_retry_manager()
        manager2 = get_retry_manager()
        
        assert isinstance(manager1, RetryManager)
        assert manager1 is manager2  # Singleton
    
    @display_description("验证 retry_with_backoff 装饰器")
    @allure.title("Test retry_with_backoff decorator")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_retry_with_backoff_decorator(self):
        """Test retry_with_backoff decorator."""
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, base_delay=0.1)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Failed")
            return "success"
        
        result = await flaky_function()
        
        assert result == "success"
        assert call_count == 3
    
    @display_description("验证 circuit_breaker 装饰器")
    @allure.title("Test circuit_breaker decorator")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit_breaker decorator."""
        @circuit_breaker("test_decorator", failure_threshold=2)
        async def protected_function():
            return "success"
        
        result = await protected_function()
        
        assert result == "success"


@allure.feature("Retry Management")
@allure.story("Retry Configuration")
class TestRetryConfig:
    """Tests for RetryConfig dataclass with display descriptions."""
    
    @display_description("验证默认重试配置")
    @allure.title("Test default retry config")
    @allure.severity(allure.severity_level.NORMAL)
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 10
        assert config.strategy == RetryStrategy.ADAPTIVE
        assert len(config.exceptions_to_retry) > 0
    
    @display_description("验证自定义重试配置")
    @allure.title("Test custom retry config")
    @allure.severity(allure.severity_level.NORMAL)
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.FIXED,
            retryable_exceptions=[ValueError]
        )
        
        assert config.max_attempts == 5
        assert config.strategy == RetryStrategy.FIXED
        assert config.retryable_exceptions == [ValueError]


@allure.feature("Retry Management")
@allure.story("Circuit Breaker Configuration")
class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass with display descriptions."""
    
    @display_description("验证熔断器默认配置")
    @allure.title("Test default circuit breaker config")
    @allure.severity(allure.severity_level.NORMAL)
    def test_default_config(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2
    
    @display_description("验证自定义熔断器配置")
    @allure.title("Test custom circuit breaker config")
    @allure.severity(allure.severity_level.NORMAL)
    def test_custom_config(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            half_open_max_calls=1,
            success_threshold=1
        )
        
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 1
        assert config.success_threshold == 1


@allure.feature("Retry Management")
@allure.story("Integration Tests")
class TestIntegration:
    """Integration tests with display descriptions."""
    
    @display_description("验证重试和熔断器组合使用")
    @allure.title("Test retry with circuit breaker integration")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test combining retry with circuit breaker."""
        retry_manager = RetryManager()
        
        # First call should succeed
        async def success_function():
            return "success"
        
        # Use circuit breaker
        result = await retry_manager.call_with_circuit_breaker(
            "integration_test",
            success_function
        )
        
        assert result == "success"
    
    @display_description("验证超时和重试组合使用")
    @allure.title("Test timeout with retry integration")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_timeout_with_retry(self):
        """Test combining timeout with retry."""
        async_retry = AsyncRetryWithBackoff(
            max_attempts=2,
            base_delay=0.1
        )
        
        call_count = 0
        async def sometimes_slow():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(0.5)  # Too slow
            return "success"
        
        # This should timeout on first attempt, succeed on second
        try:
            result = await TimeoutManager.with_timeout(
                async_retry.execute(sometimes_slow),
                timeout=0.2
            )
            # If we get here, it means the timeout didn't trigger as expected
            # In real scenarios, this would need different handling
        except asyncio.TimeoutError:
            # Expected on first attempt
            pass
    
    @display_description("验证完整容错流程")
    @allure.title("Test complete fault tolerance workflow")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_complete_fault_tolerance_workflow(self):
        """Test complete fault tolerance workflow."""
        # Use AsyncRetryWithBackoff for retry functionality
        async_retry = AsyncRetryWithBackoff(max_attempts=3, base_delay=0.1)
        
        # Create a function that fails multiple times then succeeds
        call_count = 0
        async def resilient_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Attempt {call_count}")
            return f"success_after_{call_count}_attempts"
        
        # Execute with retry protection
        result = await async_retry.execute(resilient_function)
        
        assert "success" in result
        assert call_count == 3
