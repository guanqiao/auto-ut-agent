"""Tests for RetryConfig."""

import pytest
from pyutagent.core.retry_config import (
    RetryConfig,
    RetryStrategy,
    DEFAULT_RETRY_CONFIG,
    get_default_retry_config,
    create_retry_config,
)


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""
    
    def test_retry_strategy_values(self):
        assert RetryStrategy.IMMEDIATE.value == 1
        assert RetryStrategy.LINEAR_BACKOFF.value == 2
        assert RetryStrategy.EXPONENTIAL_BACKOFF.value == 3
        assert RetryStrategy.FIXED_DELAY.value == 4


class TestRetryConfig:
    """Tests for RetryConfig class."""
    
    def test_default_values(self):
        config = RetryConfig()
        
        assert config.max_total_attempts == 50
        assert config.max_step_attempts == 2
        assert config.max_compilation_attempts == 2
        assert config.max_test_attempts == 2
        assert config.backoff_base == 2.0
        assert config.backoff_max == 60.0
        assert config.backoff_strategy == RetryStrategy.EXPONENTIAL_BACKOFF
    
    def test_get_delay_immediate(self):
        config = RetryConfig(backoff_strategy=RetryStrategy.IMMEDIATE)
        
        assert config.get_delay(1) == 0
        assert config.get_delay(5) == 0
        assert config.get_delay(10) == 0
    
    def test_get_delay_linear_backoff(self):
        config = RetryConfig(
            backoff_strategy=RetryStrategy.LINEAR_BACKOFF,
            backoff_base=2.0,
            backoff_max=60.0
        )
        
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 6.0
        assert config.get_delay(30) == 60.0
    
    def test_get_delay_exponential_backoff(self):
        config = RetryConfig(
            backoff_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            backoff_base=2.0,
            backoff_max=60.0
        )
        
        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 4.0
        assert config.get_delay(3) == 8.0
        assert config.get_delay(4) == 16.0
        assert config.get_delay(5) == 32.0
        assert config.get_delay(6) == 60.0
    
    def test_get_delay_fixed_delay(self):
        config = RetryConfig(
            backoff_strategy=RetryStrategy.FIXED_DELAY,
            backoff_base=5.0
        )
        
        assert config.get_delay(1) == 5.0
        assert config.get_delay(5) == 5.0
        assert config.get_delay(10) == 5.0
    
    def test_is_retryable(self):
        config = RetryConfig()
        
        assert config.is_retryable("network") is True
        assert config.is_retryable("timeout") is True
        assert config.is_retryable("compilation") is True
        assert config.is_retryable("unknown_category") is False
    
    def test_should_stop_total_attempts(self):
        config = RetryConfig(max_total_attempts=10)
        
        assert config.should_stop(5) is False
        assert config.should_stop(10) is True
        assert config.should_stop(15) is True
    
    def test_should_stop_step_attempts(self):
        config = RetryConfig(max_step_attempts=3, max_compilation_attempts=5)
        
        assert config.should_stop(3, "compilation") is False
        assert config.should_stop(5, "compilation") is True
        assert config.should_stop(3, "unknown") is True
    
    def test_get_max_attempts(self):
        config = RetryConfig(
            max_step_attempts=3,
            max_compilation_attempts=5,
            max_test_attempts=4
        )
        
        assert config.get_max_attempts("compilation") == 5
        assert config.get_max_attempts("test") == 4
        assert config.get_max_attempts("unknown") == 3
        assert config.get_max_attempts() == 3
    
    def test_with_overrides(self):
        original = RetryConfig(max_total_attempts=50, max_step_attempts=2)
        modified = original.with_overrides(max_total_attempts=100, max_step_attempts=10)
        
        assert original.max_total_attempts == 50
        assert original.max_step_attempts == 2
        assert modified.max_total_attempts == 100
        assert modified.max_step_attempts == 10


class TestGlobalFunctions:
    """Tests for global functions."""
    
    def test_get_default_retry_config(self):
        config = get_default_retry_config()
        
        assert isinstance(config, RetryConfig)
        assert config is DEFAULT_RETRY_CONFIG
    
    def test_create_retry_config(self):
        config = create_retry_config(
            max_total_attempts=100,
            max_step_attempts=10,
            backoff_base=3.0
        )
        
        assert config.max_total_attempts == 100
        assert config.max_step_attempts == 10
        assert config.backoff_base == 3.0
