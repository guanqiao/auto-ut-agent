"""Agent module for UT generation with self-feedback loop."""

from .base_agent import BaseAgent, AgentState, AgentResult, StepResult
from .react_agent import ReActAgent
from .actions import Action, ActionRegistry
from .prompts import PromptBuilder
from .error_recovery import (
    ErrorRecoveryManager,
    ErrorCategory,
    RecoveryStrategy,
    RecoveryAttempt
)
from .retry_manager import (
    RetryManager,
    InfiniteRetryManager,
    RetryConfig,
    RetryStrategy,
    RetryResult
)

from .handlers import (
    CompilationHandler,
    CoverageHandler,
    TestExecutionHandler,
)

from .generators import (
    BaseTestGenerator,
    LLMTestGenerator,
    AiderTestGenerator,
)

from .utils import (
    TestFileManager,
    StateManager,
)

__all__ = [
    "BaseAgent",
    "AgentState",
    "AgentResult",
    "StepResult",
    "ReActAgent",
    "Action",
    "ActionRegistry",
    "PromptBuilder",
    "ErrorRecoveryManager",
    "ErrorCategory",
    "RecoveryStrategy",
    "RecoveryAttempt",
    "RetryManager",
    "InfiniteRetryManager",
    "RetryConfig",
    "RetryStrategy",
    "RetryResult",
    "CompilationHandler",
    "CoverageHandler",
    "TestExecutionHandler",
    "BaseTestGenerator",
    "LLMTestGenerator",
    "AiderTestGenerator",
    "TestFileManager",
    "StateManager",
]
