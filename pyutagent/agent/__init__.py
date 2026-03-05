"""Agent module for UT generation with self-feedback loop."""

from .base_agent import BaseAgent, StepResult
from .unified_agent_base import (
    UnifiedAgentBase,
    AgentConfig,
    AgentState as UnifiedAgentState,
    AgentCapability as UnifiedAgentCapability,
    AgentResult as UnifiedAgentResult,
    AgentMixin,
    ProgressUpdate,
    create_agent_config,
)
from .react_agent import ReActAgent
from .enhanced_agent import EnhancedAgent, EnhancedAgentConfig
from .test_generator import TestGeneratorAgent
from .unified_autonomous_loop import (
    UnifiedAutonomousLoop,
    LoopConfig,
    LoopState as UnifiedLoopState,
    LoopFeature,
    DecisionStrategy,
    Observation,
    Thought,
    Action as LoopAction,
    Verification,
    LearningEntry,
    LoopResult,
    create_loop_config,
)
from .integration_manager import (
    IntegrationManager,
    get_integration_manager,
    reset_integration_manager,
    ComponentStatus
)
from .actions import Action, ActionRegistry
from .prompts import PromptBuilder, ToolUsagePromptBuilder
from .tool_service import AgentToolService, create_agent_tool_service
from .unified_tool_service import (
    UnifiedToolService,
    ToolServiceConfig,
    ToolCall as UnifiedToolCall,
    ToolState,
    ExecutionPlan as ToolExecutionPlan,
    PlanState,
    OrchestrationResult,
    DependencyGraph as ToolDependencyGraph,
    create_unified_tool_service,
)
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
    RetryManagerConfig,
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
    CompressionStrategy
)
from .generation_evaluator import (
    GenerationEvaluator,
    EvaluationResult,
    QualityDimension
)
from .partial_success_handler import (
    PartialSuccessHandler,
    PartialTestResult
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
    MessageBus,
    AgentMessage,
    MessageType,
    SharedKnowledgeBase,
    ExperienceReplay
)

# P4 Intelligent Enhancement Components
from .self_reflection import (
    SelfReflection,
    CritiqueResult,
    ImprovementResult,
    QualityDimension,
    IssueSeverity
)

# P4 Intelligent Enhancement Components - Additional
from .intelligent_selector import (
    IntelligentToolSelector,
    ToolScore,
    create_intelligent_selector
)
from .tool_use_agent import (
    ToolUseAgent,
    ToolUseTurn,
    ToolUseResult,
    create_tool_use_agent
)
from .intelligence_enhancer import IntelligenceEnhancer

# Universal Agent Components (P0 Gap Fill)
from .universal_planner import (
    UniversalTaskPlanner,
    TaskType,
    TaskUnderstanding,
    Subtask,
    SubtaskResult,
    ExecutionPlan,
    TaskHandler,
    ExecutionResult,
)
from .task_understanding import (
    TaskPriority,
    TaskComplexity,
    TargetScope,
    Constraint,
    SuccessCriterion,
    TaskClassifier,
    EnhancedTaskClassifier,
    Intent,
    EntityExtractionResult,
)
from .task_planner import (
    SubTaskStatus,
    SubTaskType,
    SubTask,
    TaskPlanner,
    PlanExecutor,
    EnhancedTaskPlanner,
    EnhancedPlanExecutor,
)
from .universal_agent import (
    AgentMode,
    TaskResult,
    UniversalCodingAgent,
)

# Skills Framework (P0 Gap Fill)
from .skills import (
    SkillCategory,
    SkillMetadata,
    SkillInput,
    SkillOutput,
    Skill,
    SkillRegistry,
    SkillLoader,
    EnhancedSkillExecutor,
    SkillStep,
    SkillExample,
)
from .builtin_skills import (
    GenerateUnitTestSkill,
    FixCompilationErrorSkill,
    AnalyzeCodeSkill,
    RefactorCodeSkill,
    GenerateDocSkill,
    ExplainCodeSkill,
    DebugTestSkill,
)

# Voice Interaction (P1 Gap Fill)
from .voice import (
    VoiceConfig,
    VoiceProvider,
    TTSProvider,
    VoiceInputHandler,
    VoiceOutputHandler,
    VoiceCommandParser,
    VoiceCommand,
    create_voice_handlers,
)

# Safety & Security (P1 Gap Fill)
from .safety import (
    SafetyPolicy,
    SafetyValidator,
    UserInterventionHandler,
    ValidationResult,
    UserDecision,
    InterruptType,
    ProposedAction,
    UserResponse,
    UserInterrupt,
    ValidationContext,
    ValidationResponse,
    create_default_policy,
)

# IDE Integration (P1 Gap Fill)
from .acp_client import (
    ACPClient,
    ACPClientConfig,
    ACPMessage,
    ACPMessageType,
    create_acp_client,
)

# SubAgent Enhancement (P2 Gap Fill)
from .delegating_subagent import (
    DelegatingSubAgent,
    DelegationContext,
    DelegationMode,
    DelegationResult,
    ProgressUpdate as DelegationProgressUpdate,
    create_delegating_subagent,
)
from .subagent_factory import (
    SubAgentFactory,
    AgentType,
    AgentTemplate,
    AgentPoolConfig,
    AgentInfo,
    create_subagent_factory,
)
from .subagent_orchestrator import (
    SubAgentOrchestrator,
    OrchestrationMode,
    OrchestrationResult,
    OrchestrationStatus,
    create_subagent_orchestrator,
)

# Coordination (P2 Gap Fill)
from .hierarchical_planner import (
    HierarchicalTaskPlanner,
    TaskTree,
    DependencyGraph,
    ExecutionPlan,
    Subtask,
    SubtaskType,
    SubtaskStatus,
    create_hierarchical_planner,
)
from .task_router import (
    IntelligentTaskRouter,
    RoutingStrategy,
    RoutingDecision,
    RoutingScore,
    AgentProfile,
    create_task_router,
)
from .conflict_resolver import (
    ConflictResolver,
    ConflictType,
    ConflictSeverity,
    ConflictStatus,
    ResolutionStrategy,
    Conflict,
    ConflictParty,
    ConflictResource,
    Resolution,
    ConflictRecord,
    create_conflict_resolver,
)

# Context & Results (P2 Gap Fill)
from .shared_context import (
    SharedContextManager,
    AgentContext,
    ContextSnapshot,
    ContextEntry,
    ContextScope,
    ContextVisibility,
    create_shared_context_manager,
)
from .result_aggregator import (
    ResultAggregator,
    AggregationStrategy,
    AggregatedResult,
    ValidationResult,
    Inconsistency,
    InconsistencyType,
    SummaryReport,
    create_result_aggregator,
)
from .delegation_mixin import (
    AgentDelegationMixin,
    DelegationOptions,
    DelegationMode as MixinDelegationMode,
    DelegationRecord,
    create_delegation_mixin,
)

# Collaboration (P2 Gap Fill)
from .collaboration import (
    CollaborationMode,
    UserResponse,
    ProposedAction,
    ActionResult,
    UserDecision,
    ContentPreview,
    UserInteractionHandler,
    CollaborationManager,
    create_collaboration_handler,
)

__all__ = [
    # Base Components
    "BaseAgent",
    "AgentState",
    "AgentResult",
    "StepResult",
    
    # Unified Agent Base
    "UnifiedAgentBase",
    "AgentConfig",
    "UnifiedAgentState",
    "UnifiedAgentCapability",
    "UnifiedAgentResult",
    "AgentMixin",
    "ProgressUpdate",
    "create_agent_config",
    
    # Unified Autonomous Loop
    "UnifiedAutonomousLoop",
    "LoopConfig",
    "UnifiedLoopState",
    "LoopFeature",
    "DecisionStrategy",
    "Observation",
    "Thought",
    "LoopAction",
    "Verification",
    "LearningEntry",
    "LoopResult",
    "create_loop_config",
    
    "ReActAgent",
    "EnhancedAgent",
    "EnhancedAgentConfig",
    "TestGeneratorAgent",
    "IntegrationManager",
    "get_integration_manager",
    "reset_integration_manager",
    "ComponentStatus",
    
    # Core Components
    "Action",
    "ActionRegistry",
    "PromptBuilder",
    "ToolUsagePromptBuilder",
    "AgentToolService",
    "create_agent_tool_service",
    "UnifiedToolService",
    "ToolServiceConfig",
    "UnifiedToolCall",
    "ToolState",
    "ToolExecutionPlan",
    "PlanState",
    "OrchestrationResult",
    "ToolDependencyGraph",
    "create_unified_tool_service",
    "ErrorRecoveryManager",
    "ErrorCategory",
    "RecoveryStrategy",
    "RecoveryAttempt",
    "RetryManager",
    "InfiniteRetryManager",
    "RetryManagerConfig",
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
    "GenerationEvaluator",
    "EvaluationResult",
    "QualityDimension",
    "PartialSuccessHandler",
    "PartialTestResult",
    
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
    "MessageBus",
    "AgentMessage",
    "MessageType",
    "SharedKnowledgeBase",
    "ExperienceReplay",
    
    # P4 Intelligent Enhancement Components
    "SelfReflection",
    "CritiqueResult",
    "ImprovementResult",
    "QualityDimension",
    "IssueSeverity",
    
    # P4 Intelligent Enhancement Components - Additional
    "IntelligentToolSelector",
    "ToolScore",
    "create_intelligent_selector",
    "ToolUseAgent",
    "ToolUseTurn",
    "ToolUseResult",
    "create_tool_use_agent",
    "IntelligenceEnhancer",
    
    # Universal Task Planner
    "UniversalTaskPlanner",
    "TaskType",
    "TaskUnderstanding",
    "Subtask",
    "SubtaskResult",
    "ExecutionPlan",
    "TaskHandler",
    "ExecutionResult",

    # Universal Agent Components (P0 Gap Fill)
    "TaskPriority",
    "TaskComplexity",
    "TargetScope",
    "Constraint",
    "SuccessCriterion",
    "TaskClassifier",
    "SubTaskStatus",
    "SubTaskType",
    "SubTask",
    "TaskPlanner",
    "PlanExecutor",
    "EnhancedTaskPlanner",
    "EnhancedPlanExecutor",
    "AgentMode",
    "TaskResult",
    "UniversalCodingAgent",

    # Skills Framework
    "SkillCategory",
    "SkillMetadata",
    "SkillInput",
    "SkillOutput",
    "Skill",
    "SkillRegistry",
    "SkillLoader",
    "EnhancedSkillExecutor",
    "SkillStep",
    "SkillExample",
    "GenerateUnitTestSkill",
    "FixCompilationErrorSkill",
    "AnalyzeCodeSkill",
    "RefactorCodeSkill",
    "GenerateDocSkill",
    "ExplainCodeSkill",
    "DebugTestSkill",

    # Voice Interaction
    "VoiceConfig",
    "VoiceProvider",
    "TTSProvider",
    "VoiceInputHandler",
    "VoiceOutputHandler",
    "VoiceCommandParser",
    "VoiceCommand",
    "create_voice_handlers",

    # Safety & Security
    "SafetyPolicy",
    "SafetyValidator",
    "UserInterventionHandler",
    "ValidationResult",
    "UserDecision",
    "InterruptType",
    "ProposedAction",
    "UserResponse",
    "UserInterrupt",
    "ValidationContext",
    "ValidationResponse",
    "create_default_policy",

    # IDE Integration
    "ACPClient",
    "ACPClientConfig",
    "ACPMessage",
    "ACPMessageType",
    "create_acp_client",

    # SubAgent Enhancement
    "DelegatingSubAgent",
    "DelegationContext",
    "DelegationMode",
    "DelegationResult",
    "ProgressUpdate",
    "create_delegating_subagent",
    "SubAgentFactory",
    "AgentType",
    "AgentTemplate",
    "AgentPoolConfig",
    "AgentInfo",
    "create_subagent_factory",
    "SubAgentOrchestrator",
    "OrchestrationMode",
    "OrchestrationResult",
    "OrchestrationStatus",
    "create_subagent_orchestrator",

    # Coordination
    "HierarchicalTaskPlanner",
    "TaskTree",
    "DependencyGraph",
    "ExecutionPlan",
    "Subtask",
    "SubtaskType",
    "SubtaskStatus",
    "create_hierarchical_planner",
    "IntelligentTaskRouter",
    "RoutingStrategy",
    "RoutingDecision",
    "RoutingScore",
    "AgentProfile",
    "create_task_router",
    "ConflictResolver",
    "ConflictType",
    "ConflictSeverity",
    "ConflictStatus",
    "ResolutionStrategy",
    "Conflict",
    "ConflictParty",
    "ConflictResource",
    "Resolution",
    "ConflictRecord",
    "create_conflict_resolver",

    # Context & Results
    "SharedContextManager",
    "AgentContext",
    "ContextSnapshot",
    "ContextEntry",
    "ContextScope",
    "ContextVisibility",
    "create_shared_context_manager",
    "ResultAggregator",
    "AggregationStrategy",
    "AggregatedResult",
    "ValidationResult",
    "Inconsistency",
    "InconsistencyType",
    "SummaryReport",
    "create_result_aggregator",
    "AgentDelegationMixin",
    "DelegationOptions",
    "DelegationMode as MixinDelegationMode",
    "DelegationRecord",
    "create_delegation_mixin",

    # Collaboration
    "CollaborationMode",
    "CollaborationManager",
    "UserInteractionHandler",
    "UserResponse",
    "ProposedAction",
    "ActionResult",
    "UserDecision",
    "ContentPreview",
    "create_collaboration_handler",
]


