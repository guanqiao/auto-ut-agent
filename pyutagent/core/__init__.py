"""Core module - 核心基础设施"""
import warnings
from typing import Any

# Unified messaging (recommended)
from pyutagent.core.messaging import (
    UnifiedMessageBus,
    Message,
    MessageType,
    MessagePriority,
)

# Unified event bus (recommended for event-driven patterns)
from pyutagent.core.event_bus import (
    EventBus,
    Event,
    Subscription,
    create_event_bus,
    publish_event,
)

# Legacy event bus adapters (deprecated, for backward compatibility)
from pyutagent.core.messaging.adapters import (
    EventBusAdapter as _EventBusAdapter,
    AsyncEventBusAdapter as _AsyncEventBusAdapter,
)

# Provide backward compatible aliases with deprecation warnings
class _LegacyEventBus(_EventBusAdapter):
    """Legacy EventBus - Deprecated, use EventBus or UnifiedMessageBus instead."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "core.EventBus is deprecated. Use core.event_bus.EventBus or core.messaging.UnifiedMessageBus instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

class _LegacyAsyncEventBus(_AsyncEventBusAdapter):
    """Legacy AsyncEventBus - Deprecated, use EventBus or UnifiedMessageBus instead."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "core.AsyncEventBus is deprecated. Use core.event_bus.EventBus or core.messaging.UnifiedMessageBus instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

# Keep old names for backward compatibility
EventBusAdapter = _LegacyEventBus
AsyncEventBusAdapter = _LegacyAsyncEventBus

from pyutagent.core.test_strategy_selector import (
    TestStrategySelector,
    TestStrategy,
    CodeCharacteristic,
    StrategyRecommendation
)
from pyutagent.core.boundary_analyzer import (
    BoundaryAnalyzer,
    BoundaryType,
    ParameterType,
    BoundaryAnalysisResult
)
from pyutagent.core.enhanced_feedback_loop import (
    EnhancedFeedbackLoop,
    FeedbackType,
    LearningCategory,
    AdaptiveAdjustment
)
from pyutagent.core.smart_mock_generator import (
    SmartMockGenerator,
    MockConfig,
    GeneratedMock,
    GenerationContext
)
from pyutagent.core.cache import (
    get_global_cache,
    init_global_cache,
    get_file_cache,
    MultiLevelCache,
    L1MemoryCache,
    L2DiskCache
)
from pyutagent.core.project_config import (
    BuildTool,
    TestFramework,
    MockFramework,
    BuildConfig,
    TestConfig,
    AgentPreferences,
    CodeStyle,
    DependencyInfo,
    ProjectContext,
    ProjectConfig,
    ProjectConfigLoader,
    load_project_config,
    create_config_template,
)
from pyutagent.core.context_compactor import (
    ContextCompactor,
    AutoCompactManager,
    CompactedContext,
    CompactionStrategy,
    CompactionEvent,
)
from pyutagent.core.checkpoint import (
    Checkpoint,
    CheckpointMetadata,
    CheckpointManager,
)

__all__ = [
    # Unified messaging (recommended)
    'UnifiedMessageBus',
    'Message',
    'MessageType',
    'MessagePriority',
    # Unified event bus (recommended for event-driven patterns)
    'EventBus',
    'Event',
    'Subscription',
    'create_event_bus',
    'publish_event',
    # Legacy event bus adapters (deprecated)
    'EventBusAdapter',
    'AsyncEventBusAdapter',
    'TestStrategySelector',
    'TestStrategy',
    'CodeCharacteristic',
    'StrategyRecommendation',
    'BoundaryAnalyzer',
    'BoundaryType',
    'ParameterType',
    'BoundaryAnalysisResult',
    'EnhancedFeedbackLoop',
    'FeedbackType',
    'LearningCategory',
    'AdaptiveAdjustment',
    'SmartMockGenerator',
    'MockConfig',
    'GeneratedMock',
    'GenerationContext',
    'get_global_cache',
    'init_global_cache',
    'get_file_cache',
    'MultiLevelCache',
    'L1MemoryCache',
    'L2DiskCache',
    # Project Configuration
    'BuildTool',
    'TestFramework',
    'MockFramework',
    'BuildConfig',
    'TestConfig',
    'AgentPreferences',
    'CodeStyle',
    'DependencyInfo',
    'ProjectContext',
    'ProjectConfig',
    'ProjectConfigLoader',
    'load_project_config',
    'create_config_template',
    # Context Compaction
    'ContextCompactor',
    'AutoCompactManager',
    'CompactedContext',
    'CompactionStrategy',
    'CompactionEvent',
    # Checkpoint
    'Checkpoint',
    'CheckpointMetadata',
    'CheckpointManager',
]

# Unified interfaces (recommended for new code)
from pyutagent.core.interfaces import (
    # Enums
    AgentState,
    ExecutionStatus,
    CapabilityType,
    # Data Classes
    ExecutionResult,
    Capability,
    Task,
    # Core Protocols
    IExecutable,
    IInitializable,
    IStateful,
    ICapable,
    # Agent Protocols
    IAgent,
    ISubAgent,
    # Tool Protocols
    ITool,
    IToolRegistry,
    # Context Protocols
    IContext,
    IProjectContext,
    # Memory Protocols
    IMemory,
    IWorkingMemory,
    ILongTermMemory,
    # LLM Protocols
    ILLMClient,
    # Event Protocols
    IEvent,
    IEventBus,
    # Skill Protocols
    ISkill,
    # Abstract Base Classes
    AbstractAgent,
    AbstractTool,
)

# Add unified interfaces to __all__
__all__.extend([
    # Enums
    'AgentState',
    'ExecutionStatus',
    'CapabilityType',
    # Data Classes
    'ExecutionResult',
    'Capability',
    'Task',
    # Core Protocols
    'IExecutable',
    'IInitializable',
    'IStateful',
    'ICapable',
    # Agent Protocols
    'IAgent',
    'ISubAgent',
    # Tool Protocols
    'ITool',
    'IToolRegistry',
    # Context Protocols
    'IContext',
    'IProjectContext',
    # Memory Protocols
    'IMemory',
    'IWorkingMemory',
    'ILongTermMemory',
    # LLM Protocols
    'ILLMClient',
    # Event Protocols
    'IEvent',
    'IEventBus',
    # Skill Protocols
    'ISkill',
    # Abstract Base Classes
    'AbstractAgent',
    'AbstractTool',
])
