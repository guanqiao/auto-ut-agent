"""Unified retry configuration for the entire system.

This module provides a centralized retry configuration to ensure consistent
behavior across all retry mechanisms and prevent infinite loops.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


class RetryStrategy(Enum):
    """Retry strategies for different scenarios."""
    IMMEDIATE = auto()
    LINEAR_BACKOFF = auto()
    EXPONENTIAL_BACKOFF = auto()
    FIXED_DELAY = auto()


@dataclass
class RetryConfig:
    """Unified retry configuration.
    
    This configuration is used across all retry mechanisms to ensure
    consistent behavior and prevent infinite loops.
    
    Attributes:
        max_total_attempts: Maximum total attempts across all operations
        max_step_attempts: Maximum attempts for a single step
        max_compilation_attempts: Maximum compilation attempts
        max_test_attempts: Maximum test execution attempts
        backoff_base: Base delay for backoff strategies (seconds)
        backoff_max: Maximum delay for backoff strategies (seconds)
        backoff_strategy: Strategy for calculating retry delays
        retryable_errors: Mapping of error categories to retryability
    """
    
    max_total_attempts: int = 50
    max_step_attempts: int = 2
    max_compilation_attempts: int = 2
    max_test_attempts: int = 2
    
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    backoff_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    retryable_errors: Dict[str, bool] = field(default_factory=lambda: {
        "network": True,
        "timeout": True,
        "transient": True,
        "resource": True,
        "compilation": True,
        "test_failure": True,
        "llm_api": True,
    })
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        if self.backoff_strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.backoff_strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(self.backoff_base * attempt, self.backoff_max)
        elif self.backoff_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
        elif self.backoff_strategy == RetryStrategy.FIXED_DELAY:
            return self.backoff_base
        else:
            return self.backoff_base
    
    def is_retryable(self, error_category: str) -> bool:
        """Check if an error category is retryable.
        
        Args:
            error_category: Category of the error
            
        Returns:
            True if the error category is retryable
        """
        return self.retryable_errors.get(error_category.lower(), False)
    
    def should_stop(self, attempt: int, step_name: Optional[str] = None) -> bool:
        """Check if should stop based on attempt count.
        
        Args:
            attempt: Current attempt number
            step_name: Name of the current step (for step-specific limits)
            
        Returns:
            True if should stop
        """
        if attempt >= self.max_total_attempts:
            return True
        
        if step_name:
            step_limits = {
                "compilation": self.max_compilation_attempts,
                "compile": self.max_compilation_attempts,
                "test_execution": self.max_test_attempts,
                "test": self.max_test_attempts,
                "testing": self.max_test_attempts,
            }
            
            step_limit = step_limits.get(step_name.lower(), self.max_step_attempts)
            if attempt >= step_limit:
                return True
        
        return False
    
    def get_max_attempts(self, step_name: Optional[str] = None) -> int:
        """Get maximum attempts for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Maximum attempts allowed
        """
        if step_name:
            step_limits = {
                "compilation": self.max_compilation_attempts,
                "compile": self.max_compilation_attempts,
                "test_execution": self.max_test_attempts,
                "test": self.max_test_attempts,
                "testing": self.max_test_attempts,
            }
            return step_limits.get(step_name.lower(), self.max_step_attempts)
        return self.max_step_attempts
    
    def with_overrides(self, **kwargs) -> 'RetryConfig':
        """Create a new RetryConfig with overridden values.
        
        Args:
            **kwargs: Values to override
            
        Returns:
            New RetryConfig instance
        """
        current_values = {
            'max_total_attempts': self.max_total_attempts,
            'max_step_attempts': self.max_step_attempts,
            'max_compilation_attempts': self.max_compilation_attempts,
            'max_test_attempts': self.max_test_attempts,
            'backoff_base': self.backoff_base,
            'backoff_max': self.backoff_max,
            'backoff_strategy': self.backoff_strategy,
            'retryable_errors': self.retryable_errors.copy(),
        }
        current_values.update(kwargs)
        return RetryConfig(**current_values)


DEFAULT_RETRY_CONFIG = RetryConfig()


def get_default_retry_config() -> RetryConfig:
    """Get the default retry configuration.
    
    Returns:
        Default RetryConfig instance
    """
    return DEFAULT_RETRY_CONFIG


def create_retry_config(
    max_total_attempts: int = 50,
    max_step_attempts: int = 2,
    backoff_base: float = 2.0,
    **kwargs
) -> RetryConfig:
    """Create a custom retry configuration.
    
    Args:
        max_total_attempts: Maximum total attempts
        max_step_attempts: Maximum attempts per step
        backoff_base: Base delay for backoff
        **kwargs: Additional configuration options
        
    Returns:
        Configured RetryConfig instance
    """
    return RetryConfig(
        max_total_attempts=max_total_attempts,
        max_step_attempts=max_step_attempts,
        backoff_base=backoff_base,
        **kwargs
    )
