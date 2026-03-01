"""Agent module for UT generation with self-feedback loop."""

from .base_agent import BaseAgent, StepResult
from .react_agent import ReActAgent
from .actions import Action, ActionRegistry
from .prompts import PromptBuilder
from ..core.protocols import AgentState, AgentResult
from ..core.error_recovery import (
    ErrorRecoveryManager,
    ErrorCategory,
    RecoveryStrategy,
    RecoveryAttempt
)
from ..core.retry_manager import (
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

from .services import (
    TestGenerationService,
    TestExecutionService,
    CoverageAnalysisService,
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
    "TestGenerationService",
    "TestExecutionService",
    "CoverageAnalysisService",
]
