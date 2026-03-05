"""Task Planning Module.

This module provides intelligent task planning capabilities:
- Task decomposition (atomic, sequential, composite)
- Dependency analysis
- Parallel execution

This is part of Phase 3 Week 17-18: Task Planning Enhancement.
"""

from .decomposer import (
    TaskDecomposer,
    SimpleTaskDecomposer,
    LLMTaskDecomposer,
    TemplateTaskDecomposer,
    DecompositionStrategy,
    DecompositionContext,
    get_task_decomposer,
)
from .dependency_analyzer import (
    DependencyAnalyzer,
    AdvancedDependencyAnalyzer,
    DependencyGraph,
    DependencyNode,
)
from .parallel_executor import (
    ParallelExecutionEngine,
    ParallelExecutionConfig,
    SubTaskResult,
    ResourceType,
    ResourcePool,
)

__all__ = [
    "TaskDecomposer",
    "SimpleTaskDecomposer",
    "LLMTaskDecomposer",
    "TemplateTaskDecomposer",
    "DecompositionStrategy",
    "DecompositionContext",
    "get_task_decomposer",
    "DependencyAnalyzer",
    "AdvancedDependencyAnalyzer",
    "DependencyGraph",
    "DependencyNode",
    "ParallelExecutionEngine",
    "ParallelExecutionConfig",
    "SubTaskResult",
    "ResourceType",
    "ResourcePool",
]
