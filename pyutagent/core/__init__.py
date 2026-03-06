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

# Legacy event bus (deprecated, for backward compatibility)
from pyutagent.core.event_bus import EventBus as _EventBus, AsyncEventBus as _AsyncEventBus

warnings.warn(
    "pyutagent.core.EventBus and AsyncEventBus are deprecated. "
    "Use pyutagent.core.messaging.UnifiedMessageBus instead.",
    DeprecationWarning,
    stacklevel=2
)

# Provide backward compatible aliases
class EventBus(_EventBus):
    """EventBus - Deprecated, use UnifiedMessageBus instead."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "EventBus is deprecated. Use UnifiedMessageBus from pyutagent.core.messaging instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

class AsyncEventBus(_AsyncEventBus):
    """AsyncEventBus - Deprecated, use UnifiedMessageBus instead."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "AsyncEventBus is deprecated. Use UnifiedMessageBus from pyutagent.core.messaging instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)

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
    # Legacy event bus (deprecated)
    'EventBus',
    'AsyncEventBus',
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
