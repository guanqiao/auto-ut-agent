"""Core module for PyUT Agent.

This module contains the core components that are shared across the application:
- Unified retry management
- Unified error recovery
- Unified configuration management
- Dependency injection container
- Abstract base classes and protocols
"""

from .retry_manager import (
    RetryManager,
    RetryConfig,
    RetryStrategy,
    RetryResult,
    InfiniteRetryManager,
    with_retry,
)
from .error_recovery import (
    ErrorRecoveryManager,
    ErrorCategory,
    RecoveryStrategy,
    RecoveryResult,
    ErrorClassifier,
    StatePreserver,
    is_retryable_error,
)
from .config import (
    Settings,
    LLMConfig,
    LLMConfigCollection,
    AiderConfig,
    get_settings,
    get_data_dir,
    load_llm_config,
    save_llm_config,
    load_aider_config,
    save_aider_config,
)
from .container import Container, get_container, configure_container
from .protocols import (
    LLMClientProtocol,
    CodeEditorProtocol,
    TestRunnerProtocol,
    CodeParserProtocol,
    MemoryProtocol,
    RecoveryHandlerProtocol,
    AgentState,
    AgentResult,
    TestResult,
    CoverageResult,
    ParsedCode,
    ClassInfo,
    MethodInfo,
    FieldInfo,
    ParameterInfo,
    ErrorAnalysis,
    RecoveryAction,
    BaseAgent,
    BaseTool,
)

__all__ = [
    "RetryManager",
    "RetryConfig",
    "RetryStrategy",
    "RetryResult",
    "InfiniteRetryManager",
    "with_retry",
    "ErrorRecoveryManager",
    "ErrorCategory",
    "RecoveryStrategy",
    "RecoveryResult",
    "ErrorClassifier",
    "StatePreserver",
    "is_retryable_error",
    "Settings",
    "LLMConfig",
    "LLMConfigCollection",
    "AiderConfig",
    "get_settings",
    "get_data_dir",
    "load_llm_config",
    "save_llm_config",
    "load_aider_config",
    "save_aider_config",
    "Container",
    "get_container",
    "configure_container",
    "LLMClientProtocol",
    "CodeEditorProtocol",
    "TestRunnerProtocol",
    "CodeParserProtocol",
    "MemoryProtocol",
    "RecoveryHandlerProtocol",
    "AgentState",
    "AgentResult",
    "TestResult",
    "CoverageResult",
    "ParsedCode",
    "ClassInfo",
    "MethodInfo",
    "FieldInfo",
    "ParameterInfo",
    "ErrorAnalysis",
    "RecoveryAction",
    "BaseAgent",
    "BaseTool",
]
