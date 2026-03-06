"""Tests for Unified Retry Mechanism."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
import time

from pyutagent.agent.execution.retry import (
    BackoffStrategy,
    RetryConfig,
    RetryPolicy,
    SmartRetryPolicy,
    RetryResult,
    RetryExecutor,
    with_retry,
)


class TestRetryConfig:
    """Tests for RetryConfig."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.backoff_strategy == BackoffStrategy.EXPONENTIAL
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
    
    def test_get_delay_fixed(self):
        """Test fixed backoff delay."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.FIXED,
            base_delay=2.0
        )
        
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 2.0
        assert config.get_delay(3) == 2.0
    
    def test_get_delay_linear(self):
        """Test linear backoff delay."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.LINEAR,
            base_delay=1.0
        )
        
        assert config.get_delay(1) == 1.0
        assert config.get_delay(2) == 2.0
        assert config.get_delay(3) == 3.0
    
    def test_get_delay_exponential(self):
        """Test exponential backoff delay."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=1.0,
            exponential_base=2.0
        )
        
        assert config.get_delay(1) == 1.0
        assert config.get_delay(2) == 2.0
        assert config.get_delay(3) == 4.0
    
    def test_get_delay_max_delay(self):
        """Test max delay cap."""
        config = RetryConfig(
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            base_delay=1.0,
            max_delay=5.0
        )
        
        assert config.get_delay(10) == 5.0
    
    def test_should_retry_max_attempts(self):
        """Test max attempts limit."""
        config = RetryConfig(max_attempts=3)
        error = Exception("test error")
        
        assert config.should_retry(error, 1) == True
        assert config.should_retry(error, 2) == True
        assert config.should_retry(error, 3) == False
    
    def test_should_retry_retryable_exceptions(self):
        """Test retryable exceptions."""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[ValueError, TypeError]
        )
        
        assert config.should_retry(ValueError("test"), 1) == True
        assert config.should_retry(TypeError("test"), 1) == True
        assert config.should_retry(RuntimeError("test"), 1) == False
    
    def test_should_retry_non_retryable_exceptions(self):
        """Test non-retryable exceptions."""
        config = RetryConfig(
            max_attempts=3,
            non_retryable_exceptions=[KeyboardInterrupt]
        )
        
        assert config.should_retry(Exception("test"), 1) == True
        assert config.should_retry(KeyboardInterrupt(), 1) == False


class TestSmartRetryPolicy:
    """Tests for SmartRetryPolicy."""
    
    def test_network_error_retry(self):
        """Test network error retry behavior."""
        policy = SmartRetryPolicy(max_network_retries=5)
        
        network_error = Exception("Connection timeout")
        assert policy.should_retry(network_error, 1) == True
        assert policy.should_retry(network_error, 4) == True
        assert policy.should_retry(network_error, 5) == False
    
    def test_code_error_retry(self):
        """Test code error retry behavior."""
        policy = SmartRetryPolicy(max_code_retries=2)
        
        code_error = Exception("Compilation error: syntax error")
        assert policy.should_retry(code_error, 1) == True
        assert policy.should_retry(code_error, 2) == False
    
    def test_unknown_error_retry(self):
        """Test unknown error retry behavior."""
        policy = SmartRetryPolicy()
        
        unknown_error = Exception("Unknown error")
        assert policy.should_retry(unknown_error, 1) == True
        assert policy.should_retry(unknown_error, 2) == True
        assert policy.should_retry(unknown_error, 3) == False
    
    def test_get_delay(self):
        """Test delay calculation."""
        policy = SmartRetryPolicy(base_delay=1.0)
        
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 4.0


class TestRetryExecutor:
    """Tests for RetryExecutor."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        executor = RetryExecutor()
        
        async def success_func():
            return "success"
        
        result = await executor.execute(success_func)
        
        assert result.success == True
        assert result.result == "success"
        assert result.attempts == 1
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_execute_with_retry(self):
        """Test execution with retry."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        executor = RetryExecutor(config=config)
        
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await executor.execute(failing_func)
        
        assert result.success == True
        assert result.result == "success"
        assert result.attempts == 3
        assert len(result.errors) == 2
    
    @pytest.mark.asyncio
    async def test_execute_max_attempts_reached(self):
        """Test max attempts reached."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        executor = RetryExecutor(config=config)
        
        async def always_fail():
            raise ValueError("Always fails")
        
        result = await executor.execute(always_fail)
        
        assert result.success == False
        assert result.attempts == 2
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_execute_with_policy(self):
        """Test execution with custom policy."""
        policy = SmartRetryPolicy(max_code_retries=1)
        executor = RetryExecutor(policy=policy)
        
        call_count = 0
        
        async def code_error_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Compilation error")
        
        result = await executor.execute(code_error_func)
        
        assert result.success == False
        assert result.attempts == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_callback(self):
        """Test execution with retry callback."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        retry_calls = []
        
        def on_retry(error, attempt):
            retry_calls.append((str(error), attempt))
        
        executor = RetryExecutor(config=config, on_retry=on_retry)
        
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Error")
            return "success"
        
        await executor.execute(failing_func)
        
        assert len(retry_calls) == 2
        assert retry_calls[0] == ("Error", 1)
        assert retry_calls[1] == ("Error", 2)
    
    def test_execute_sync_success(self):
        """Test successful sync execution."""
        executor = RetryExecutor()
        
        def success_func():
            return "success"
        
        result = executor.execute_sync(success_func)
        
        assert result.success == True
        assert result.result == "success"
    
    def test_execute_sync_with_retry(self):
        """Test sync execution with retry."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        executor = RetryExecutor(config=config)
        
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Error")
            return "success"
        
        result = executor.execute_sync(failing_func)
        
        assert result.success == True
        assert result.attempts == 3


class TestWithRetryDecorator:
    """Tests for @with_retry decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_async_success(self):
        """Test decorator on async function success."""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.01)
        async def async_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await async_func()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_decorator_async_retry(self):
        """Test decorator on async function with retry."""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.01)
        async def async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Error")
            return "success"
        
        result = await async_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_decorator_sync_success(self):
        """Test decorator on sync function success."""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.01)
        def sync_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = sync_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_decorator_sync_retry(self):
        """Test decorator on sync function with retry."""
        call_count = 0
        
        @with_retry(max_attempts=3, base_delay=0.01)
        def sync_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Error")
            return "success"
        
        result = sync_func()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_decorator_with_retryable_exceptions(self):
        """Test decorator with retryable exceptions."""
        call_count = 0
        
        @with_retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=[ValueError]
        )
        async def async_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable")
            return "success"
        
        result = await async_func()
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_decorator_non_retryable(self):
        """Test decorator with non-retryable exception."""
        call_count = 0
        
        @with_retry(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=[ValueError]
        )
        async def async_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")
        
        with pytest.raises(TypeError):
            await async_func()
        
        assert call_count == 1


class TestRetryResult:
    """Tests for RetryResult."""
    
    def test_result_creation(self):
        """Test creating retry result."""
        result = RetryResult(
            success=True,
            result="data",
            attempts=3,
            total_time=1.5
        )
        
        assert result.success == True
        assert result.result == "data"
        assert result.attempts == 3
        assert result.total_time == 1.5
        assert result.error is None
        assert result.errors == []
    
    def test_result_with_error(self):
        """Test result with error."""
        error = ValueError("test error")
        result = RetryResult(
            success=False,
            error=error,
            attempts=2,
            errors=[error]
        )
        
        assert result.success == False
        assert result.error == error
        assert len(result.errors) == 1
