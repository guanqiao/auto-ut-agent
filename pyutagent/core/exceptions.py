"""Custom exceptions for PyUT Agent.

This module defines a hierarchy of exceptions for better error handling
and debugging throughout the application.
"""


class PyUTAgentException(Exception):
    """Base exception for all PyUT Agent errors."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# LLM-related exceptions
class LLMException(PyUTAgentException):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMException):
    """Raised when connection to LLM service fails."""
    pass


class LLMGenerationError(LLMException):
    """Raised when LLM generation fails."""
    pass


class LLMTimeoutError(LLMException):
    """Raised when LLM request times out."""
    pass


class LLMRateLimitError(LLMException):
    """Raised when LLM rate limit is exceeded."""
    pass


# Compilation-related exceptions
class CompilationException(PyUTAgentException):
    """Base exception for compilation errors."""
    pass


class JavaCompilationError(CompilationException):
    """Raised when Java compilation fails."""
    pass


class MavenExecutionError(CompilationException):
    """Raised when Maven execution fails."""
    pass


# Test-related exceptions
class TestExecutionException(PyUTAgentException):
    """Base exception for test execution errors."""
    pass


class TestFailureError(TestExecutionException):
    """Raised when tests fail."""
    pass


class CoverageAnalysisError(TestExecutionException):
    """Raised when coverage analysis fails."""
    pass


# Configuration exceptions
class ConfigurationException(PyUTAgentException):
    """Base exception for configuration errors."""
    pass


class InvalidConfigurationError(ConfigurationException):
    """Raised when configuration is invalid."""
    pass


class MissingConfigurationError(ConfigurationException):
    """Raised when required configuration is missing."""
    pass


# File-related exceptions
class FileOperationException(PyUTAgentException):
    """Base exception for file operation errors."""
    pass


class FileNotFoundError(FileOperationException):
    """Raised when a required file is not found."""
    pass


class FileReadError(FileOperationException):
    """Raised when file read operation fails."""
    pass


class FileWriteError(FileOperationException):
    """Raised when file write operation fails."""
    pass


# Recovery-related exceptions
class RecoveryException(PyUTAgentException):
    """Base exception for recovery operation errors."""
    pass


class RecoveryFailedError(RecoveryException):
    """Raised when all recovery strategies fail."""
    pass


class MaxRetriesExceededError(RecoveryException):
    """Raised when maximum retry attempts are exceeded."""
    pass


# Agent-related exceptions
class AgentException(PyUTAgentException):
    """Base exception for agent operation errors."""
    pass


class AgentStateError(AgentException):
    """Raised when agent is in an invalid state."""
    pass


class TaskCancelledError(AgentException):
    """Raised when a task is cancelled."""
    pass


# Tool-related exceptions
class ToolExecutionException(PyUTAgentException):
    """Base exception for tool execution errors."""
    pass


class ToolNotFoundError(ToolExecutionException):
    """Raised when a required tool is not found."""
    pass


class ToolExecutionError(ToolExecutionException):
    """Raised when tool execution fails."""
    pass


# Validation exceptions
class ValidationException(PyUTAgentException):
    """Base exception for validation errors."""
    pass


class CodeValidationError(ValidationException):
    """Raised when code validation fails."""
    pass


class EditValidationError(ValidationException):
    """Raised when edit validation fails."""
    pass
