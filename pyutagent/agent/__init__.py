"""Agent module for UT generation with self-feedback loop."""

from .base_agent import BaseAgent, StepResult
from .react_agent import ReActAgent
from .enhanced_agent import EnhancedAgent, EnhancedAgentConfig
from .integration_manager import (
    IntegrationManager,
    get_integration_manager,
    reset_integration_manager,
    ComponentStatus
)
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

# P0 Components
from .context_manager import (
    ContextManager,
    CompressionStrategy,
    SnippetExtractor
)
from .generation_evaluator import (
    GenerationEvaluator,
    EvaluationResult,
    EvaluationDimension
)
from .partial_success_handler import (
    PartialSuccessHandler,
    PartialTestResult,
    TestFragment
)

# P1 Components
from .prompt_optimizer import (
    PromptOptimizer,
    ModelCharacteristics,
    ModelType,
    PromptStrategy,
    PromptTemplate,
    FewShotExample,
    ABTest,
    ABTestVariant,
    optimize_prompt,
    get_few_shot_prompt
)

# P2 Components
from .multi_agent import (
    AgentCoordinator,
    AgentCapability,
    AgentRole,
    SpecializedAgent,
    AgentTask,
    TaskResult,
    MessageBus,
    AgentMessage,
    MessageType,
    SharedKnowledgeBase,
    KnowledgeItem,
    ExperienceReplay,
    Experience
)

__all__ = [
    # Base Components
    "BaseAgent",
    "AgentState",
    "AgentResult",
    "StepResult",
    "ReActAgent",
    "EnhancedAgent",
    "EnhancedAgentConfig",
    "IntegrationManager",
    "get_integration_manager",
    "reset_integration_manager",
    "ComponentStatus",
    
    # Core Components
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
    
    # Handlers
    "CompilationHandler",
    "CoverageHandler",
    "TestExecutionHandler",
    
    # Generators
    "BaseTestGenerator",
    "LLMTestGenerator",
    "AiderTestGenerator",
    
    # Utils
    "TestFileManager",
    "StateManager",
    
    # Services
    "TestGenerationService",
    "TestExecutionService",
    "CoverageAnalysisService",
    
    # P0 Components
    "ContextManager",
    "CompressionStrategy",
    "SnippetExtractor",
    "GenerationEvaluator",
    "EvaluationResult",
    "EvaluationDimension",
    "PartialSuccessHandler",
    "PartialTestResult",
    "TestFragment",
    
    # P1 Components
    "PromptOptimizer",
    "ModelCharacteristics",
    "ModelType",
    "PromptStrategy",
    "PromptTemplate",
    "FewShotExample",
    "ABTest",
    "ABTestVariant",
    "optimize_prompt",
    "get_few_shot_prompt",
    
    # P2 Components
    "AgentCoordinator",
    "AgentCapability",
    "AgentRole",
    "SpecializedAgent",
    "AgentTask",
    "TaskResult",
    "MessageBus",
    "AgentMessage",
    "MessageType",
    "SharedKnowledgeBase",
    "KnowledgeItem",
    "ExperienceReplay",
    "Experience",
]
