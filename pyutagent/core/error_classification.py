"""Unified error classification service.

This module provides a single point for all error classification logic,
eliminating duplication across the codebase.
"""

import logging
from typing import Any, Dict, Optional

from pyutagent.core.error_recovery import ErrorCategory, ErrorClassifier

logger = logging.getLogger(__name__)


class ErrorClassificationService:
    """Unified service for error classification.
    
    This service provides a single point for all error classification
    logic, eliminating duplication across the codebase.
    
    Features:
    - Error classification by type and message
    - Retryability checking
    - Recovery strategy recommendation
    - Singleton pattern for global access
    
    Example:
        >>> service = get_error_classification_service()
        >>> category = service.classify(ValueError("test"))
        >>> print(category)
        ErrorCategory.VALIDATION
    """
    
    _instance: Optional['ErrorClassificationService'] = None
    
    def __new__(cls) -> 'ErrorClassificationService':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def classify(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """Classify an error into a category.
        
        Args:
            error: The error to classify
            context: Optional context information (step name, etc.)
            
        Returns:
            ErrorCategory enum value
        """
        category = ErrorClassifier.classify(error)
        
        if context:
            step = context.get("step", "").lower()
            if "compile" in step:
                return ErrorCategory.COMPILATION_ERROR
            elif "test" in step and "fail" in str(error).lower():
                return ErrorCategory.TEST_FAILURE
        
        return category
    
    def categorize_by_message(self, error_message: str, error_details: Dict[str, Any]) -> ErrorCategory:
        """Categorize error by message and details.
        
        Args:
            error_message: Error message string
            error_details: Additional error details
            
        Returns:
            ErrorCategory enum value
        """
        return ErrorClassifier.categorize_error(error_message, error_details)
    
    def is_retryable(self, error: Exception) -> bool:
        """Check if an error is retryable.
        
        Args:
            error: The error to check
            
        Returns:
            True if the error is retryable
        """
        return ErrorClassifier.is_retryable(error)
    
    def is_retryable_category(self, category: ErrorCategory) -> bool:
        """Check if an error category is retryable.
        
        Args:
            category: Error category to check
            
        Returns:
            True if the category is retryable
        """
        retryable_categories = {
            ErrorCategory.TRANSIENT,
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RESOURCE,
            ErrorCategory.COMPILATION_ERROR,
            ErrorCategory.TEST_FAILURE,
            ErrorCategory.LLM_API_ERROR,
        }
        return category in retryable_categories
    
    def get_recovery_strategy(self, error: Exception, attempt_count: int = 0) -> str:
        """Get recommended recovery strategy for an error.
        
        Args:
            error: The error
            attempt_count: Number of previous attempts
            
        Returns:
            Strategy name string
        """
        category = self.classify(error)
        return self.get_strategy_for_category(category, attempt_count)
    
    def get_strategy_for_category(self, category: ErrorCategory, attempt_count: int = 0) -> str:
        """Get recommended recovery strategy for an error category.
        
        Args:
            category: Error category
            attempt_count: Number of previous attempts
            
        Returns:
            Strategy name string
        """
        if category == ErrorCategory.NETWORK:
            return "RETRY_WITH_BACKOFF"
        elif category == ErrorCategory.TIMEOUT:
            return "RETRY_WITH_BACKOFF"
        elif category == ErrorCategory.COMPILATION_ERROR:
            return "ANALYZE_AND_FIX"
        elif category == ErrorCategory.TEST_FAILURE:
            if attempt_count < 3:
                return "ANALYZE_AND_FIX"
            else:
                return "RESET_AND_REGENERATE"
        elif category == ErrorCategory.LLM_API_ERROR:
            if attempt_count < 2:
                return "RETRY_WITH_BACKOFF"
            else:
                return "FALLBACK_ALTERNATIVE"
        elif category in (ErrorCategory.TRANSIENT, ErrorCategory.RESOURCE):
            return "RETRY_IMMEDIATE"
        elif category == ErrorCategory.PARSING_ERROR:
            return "RESET_AND_REGENERATE"
        elif category == ErrorCategory.FILE_IO_ERROR:
            return "SKIP_AND_CONTINUE"
        else:
            return "ANALYZE_AND_FIX"
    
    def get_error_severity(self, error: Exception) -> str:
        """Get severity level for an error.
        
        Args:
            error: The error
            
        Returns:
            Severity level string: "low", "medium", "high", "critical"
        """
        category = self.classify(error)
        
        severity_mapping = {
            ErrorCategory.TRANSIENT: "low",
            ErrorCategory.NETWORK: "medium",
            ErrorCategory.TIMEOUT: "medium",
            ErrorCategory.RESOURCE: "high",
            ErrorCategory.VALIDATION: "medium",
            ErrorCategory.SYNTAX: "high",
            ErrorCategory.LOGIC: "medium",
            ErrorCategory.COMPILATION_ERROR: "medium",
            ErrorCategory.TEST_FAILURE: "medium",
            ErrorCategory.TOOL_EXECUTION_ERROR: "high",
            ErrorCategory.PARSING_ERROR: "high",
            ErrorCategory.GENERATION_ERROR: "high",
            ErrorCategory.FILE_IO_ERROR: "high",
            ErrorCategory.LLM_API_ERROR: "high",
            ErrorCategory.PERMANENT: "critical",
            ErrorCategory.UNKNOWN: "medium",
        }
        
        return severity_mapping.get(category, "medium")
    
    def should_skip_recovery(self, error: Exception) -> bool:
        """Check if recovery should be skipped for this error.
        
        Args:
            error: The error
            
        Returns:
            True if recovery should be skipped
        """
        error_message = str(error).lower()
        
        skip_patterns = [
            "no compilation errors",
            "no test failures",
            "all tests passed",
            "compilation successful",
            "tests passed",
        ]
        
        for pattern in skip_patterns:
            if pattern in error_message:
                return True
        
        return False
    
    def get_error_info(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get comprehensive error information.
        
        Args:
            error: The error
            context: Optional context information
            
        Returns:
            Dictionary with error information
        """
        category = self.classify(error, context)
        
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "category": category.name,
            "is_retryable": self.is_retryable_category(category),
            "recommended_strategy": self.get_strategy_for_category(category),
            "severity": self.get_error_severity(error),
            "should_skip_recovery": self.should_skip_recovery(error),
        }


error_classification_service = ErrorClassificationService()


def get_error_classification_service() -> ErrorClassificationService:
    """Get the global error classification service instance.
    
    Returns:
        ErrorClassificationService singleton instance
    """
    return error_classification_service


def classify_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
    """Convenience function to classify an error.
    
    Args:
        error: The error to classify
        context: Optional context information
        
    Returns:
        ErrorCategory enum value
    """
    return get_error_classification_service().classify(error, context)


def is_retryable_error(error: Exception) -> bool:
    """Convenience function to check if an error is retryable.
    
    Args:
        error: The error to check
        
    Returns:
        True if the error is retryable
    """
    return get_error_classification_service().is_retryable(error)


def get_recovery_strategy(error: Exception, attempt_count: int = 0) -> str:
    """Convenience function to get recovery strategy.
    
    Args:
        error: The error
        attempt_count: Number of previous attempts
        
    Returns:
        Strategy name string
    """
    return get_error_classification_service().get_recovery_strategy(error, attempt_count)
