"""Model router for selecting appropriate LLM based on task.

Enhanced with:
- Task complexity analysis
- Token budget optimization
- Cost-aware routing
- Multi-model pool management
- Fallback strategies
"""

import logging
import re
from enum import Enum, auto
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tier enumeration."""
    LITE = "lite"               
    EFFICIENT = "efficient"     
    PERFORMANCE = "performance" 
    AUTO = "auto"               


class TaskType(str, Enum):
    """Task type enumeration."""
    CODE_COMPLETION = "code_completion"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    COVERAGE_OPTIMIZATION = "coverage_optimization"
    EXPLANATION = "explanation"
    ERROR_ANALYSIS = "error_analysis"
    REFACTORING = "refactoring"
    MOCK_GENERATION = "mock_generation"
    BOUNDARY_ANALYSIS = "boundary_analysis"
    SELF_REFLECTION = "self_reflection"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = auto()      
    MODERATE = auto()    
    COMPLEX = auto()     
    VERY_COMPLEX = auto() 


class SelectionStrategy(Enum):
    """Model selection strategies."""
    COST_OPTIMIZED = "cost_optimized"        
    QUALITY_OPTIMIZED = "quality_optimized"  
    BALANCED = "balanced"                    
    SPEED_OPTIMIZED = "speed_optimized"      


@dataclass
class ModelInfo:
    """Model information."""
    name: str
    provider: str
    max_tokens: int
    context_window: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    capabilities: Dict[str, float]
    supports_streaming: bool = True
    supports_function_calling: bool = True
    avg_latency_ms: float = 1000.0  
    reliability_score: float = 0.95  


@dataclass
class TaskContext:
    """Context for task execution."""
    task_type: TaskType
    complexity: TaskComplexity = TaskComplexity.MODERATE
    estimated_input_tokens: int = 1000
    estimated_output_tokens: int = 2000
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[float] = None
    requires_streaming: bool = False
    requires_function_calling: bool = False
    quality_threshold: float = 0.8
    preferred_provider: Optional[str] = None
    fallback_allowed: bool = True
    

@dataclass
class ModelSelection:
    """Result of model selection."""
    model_name: str
    model_info: ModelInfo
    estimated_cost: float
    estimated_latency_ms: float
    quality_score: float
    selection_reason: str
    fallback_models: List[str] = field(default_factory=list)


@dataclass
class UsageRecord:
    """Record of model usage."""
    model_name: str
    task_type: TaskType
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


MODELS = {
    "gpt-4": ModelInfo(
        name="gpt-4",
        provider="openai",
        max_tokens=8192,
        context_window=8192,
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
        capabilities={
            "reasoning": 0.95,
            "coding": 0.92,
            "speed": 0.6,
            "context_length": 0.7,
            "error_analysis": 0.93,
            "test_design": 0.90,
        },
        avg_latency_ms=3000.0,
        reliability_score=0.98,
    ),
    "gpt-4-turbo": ModelInfo(
        name="gpt-4-turbo",
        provider="openai",
        max_tokens=4096,
        context_window=128000,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        capabilities={
            "reasoning": 0.92,
            "coding": 0.90,
            "speed": 0.85,
            "context_length": 0.95,
            "error_analysis": 0.90,
            "test_design": 0.88,
        },
        avg_latency_ms=2000.0,
        reliability_score=0.97,
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        provider="openai",
        max_tokens=4096,
        context_window=16385,
        cost_per_1k_input=0.0015,
        cost_per_1k_output=0.002,
        capabilities={
            "reasoning": 0.75,
            "coding": 0.75,
            "speed": 0.95,
            "context_length": 0.5,
            "error_analysis": 0.70,
            "test_design": 0.72,
        },
        avg_latency_ms=800.0,
        reliability_score=0.95,
    ),
    "claude-3-opus": ModelInfo(
        name="claude-3-opus-20240229",
        provider="anthropic",
        max_tokens=4096,
        context_window=200000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        capabilities={
            "reasoning": 0.95,
            "coding": 0.90,
            "speed": 0.5,
            "context_length": 0.98,
            "error_analysis": 0.94,
            "test_design": 0.92,
        },
        avg_latency_ms=4000.0,
        reliability_score=0.97,
    ),
    "claude-3-sonnet": ModelInfo(
        name="claude-3-sonnet-20240229",
        provider="anthropic",
        max_tokens=4096,
        context_window=200000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        capabilities={
            "reasoning": 0.85,
            "coding": 0.82,
            "speed": 0.8,
            "context_length": 0.95,
            "error_analysis": 0.83,
            "test_design": 0.80,
        },
        avg_latency_ms=1500.0,
        reliability_score=0.96,
    ),
    "deepseek-chat": ModelInfo(
        name="deepseek-chat",
        provider="deepseek",
        max_tokens=4096,
        context_window=32000,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        capabilities={
            "reasoning": 0.80,
            "coding": 0.78,
            "speed": 0.85,
            "context_length": 0.85,
            "error_analysis": 0.75,
            "test_design": 0.76,
        },
        avg_latency_ms=1000.0,
        reliability_score=0.93,
    ),
    "deepseek-coder": ModelInfo(
        name="deepseek-coder",
        provider="deepseek",
        max_tokens=4096,
        context_window=16000,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        capabilities={
            "reasoning": 0.78,
            "coding": 0.85,
            "speed": 0.85,
            "context_length": 0.70,
            "error_analysis": 0.72,
            "test_design": 0.82,
        },
        avg_latency_ms=1000.0,
        reliability_score=0.93,
    ),
}

# Tier to models mapping
TIER_MODELS = {
    ModelTier.LITE: ["gpt-3.5-turbo", "deepseek-chat"],
    ModelTier.EFFICIENT: ["gpt-4-turbo", "claude-3-sonnet", "deepseek-coder"],
    ModelTier.PERFORMANCE: ["gpt-4", "claude-3-opus"],
}

# Task to preferred capabilities
TASK_CAPABILITIES = {
    TaskType.CODE_COMPLETION: ["coding", "speed"],
    TaskType.TEST_GENERATION: ["coding", "reasoning", "test_design"],
    TaskType.CODE_REVIEW: ["reasoning", "context_length"],
    TaskType.COVERAGE_OPTIMIZATION: ["coding", "reasoning"],
    TaskType.EXPLANATION: ["reasoning", "context_length"],
    TaskType.ERROR_ANALYSIS: ["reasoning", "error_analysis"],
    TaskType.REFACTORING: ["coding", "reasoning"],
    TaskType.MOCK_GENERATION: ["coding", "test_design"],
    TaskType.BOUNDARY_ANALYSIS: ["reasoning", "test_design"],
    TaskType.SELF_REFLECTION: ["reasoning", "error_analysis"],
}

TASK_COMPLEXITY_WEIGHTS = {
    TaskComplexity.SIMPLE: {"cost": 0.5, "quality": 0.3, "speed": 0.2},
    TaskComplexity.MODERATE: {"cost": 0.3, "quality": 0.4, "speed": 0.3},
    TaskComplexity.COMPLEX: {"cost": 0.2, "quality": 0.5, "speed": 0.3},
    TaskComplexity.VERY_COMPLEX: {"cost": 0.1, "quality": 0.6, "speed": 0.3},
}

COMPLEXITY_INDICATORS = {
    TaskComplexity.SIMPLE: {
        "max_lines": 50,
        "max_methods": 3,
        "max_nested_depth": 2,
    },
    TaskComplexity.MODERATE: {
        "max_lines": 150,
        "max_methods": 8,
        "max_nested_depth": 4,
    },
    TaskComplexity.COMPLEX: {
        "max_lines": 400,
        "max_methods": 15,
        "max_nested_depth": 6,
    },
    TaskComplexity.VERY_COMPLEX: {
        "max_lines": float('inf'),
        "max_methods": float('inf'),
        "max_nested_depth": float('inf'),
    },
}


class ModelRouter:
    """Router for selecting appropriate LLM model.
    
    Enhanced Features:
    - Task complexity analysis
    - Cost-aware routing
    - Token budget optimization
    - Fallback strategies
    - Usage tracking and analytics
    - Performance-based selection
    """
    
    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        db_path: Optional[str] = None
    ):
        """Initialize model router.
        
        Args:
            strategy: Default selection strategy
            db_path: Optional path to store usage history
        """
        self._strategy = strategy
        self._usage_stats: Dict[str, int] = {}
        self._usage_history: List[UsageRecord] = []
        self._model_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success_rate": 0.9, "avg_latency": 1000.0}
        )
        
        if db_path:
            self._db_path = Path(db_path)
            self._load_history()
        else:
            self._db_path = None
        
        logger.info(f"[ModelRouter] Initialized with strategy: {strategy.value}")
    
    def _load_history(self):
        """Load usage history from disk."""
        if self._db_path and self._db_path.exists():
            try:
                with open(self._db_path, 'r') as f:
                    data = json.load(f)
                    self._usage_history = [
                        UsageRecord(**record) for record in data.get("history", [])
                    ]
                logger.info(f"[ModelRouter] Loaded {len(self._usage_history)} usage records")
            except Exception as e:
                logger.warning(f"[ModelRouter] Failed to load history: {e}")
    
    def _save_history(self):
        """Save usage history to disk."""
        if self._db_path:
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "history": [
                        {
                            "model_name": r.model_name,
                            "task_type": r.task_type.value,
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                            "cost_usd": r.cost_usd,
                            "latency_ms": r.latency_ms,
                            "success": r.success,
                            "timestamp": r.timestamp,
                        }
                        for r in self._usage_history[-1000:]  # Keep last 1000 records
                    ]
                }
                with open(self._db_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"[ModelRouter] Failed to save history: {e}")
    
    def analyze_complexity(
        self,
        code: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> TaskComplexity:
        """Analyze task complexity based on code.
        
        Args:
            code: Source code to analyze
            class_info: Optional class information
            
        Returns:
            TaskComplexity level
        """
        lines = code.count('\n') + 1
        methods = len(re.findall(r'\b(public|private|protected)\s+\w+\s+\w+\s*\(', code))
        nested_depth = self._count_nested_depth(code)
        
        for complexity, thresholds in COMPLEXITY_INDICATORS.items():
            if (lines <= thresholds["max_lines"] and
                methods <= thresholds["max_methods"] and
                nested_depth <= thresholds["max_nested_depth"]):
                return complexity
        
        return TaskComplexity.VERY_COMPLEX
    
    def _count_nested_depth(self, code: str) -> int:
        """Count maximum nesting depth in code."""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def select_model(
        self,
        task_type: TaskType,
        tier: ModelTier = ModelTier.AUTO,
        preferred_provider: Optional[str] = None,
        available_models: Optional[List[str]] = None,
        context: Optional[TaskContext] = None
    ) -> str:
        """Select appropriate model for task.
        
        Args:
            task_type: Type of task
            tier: Model tier preference
            preferred_provider: Preferred provider
            available_models: List of available models
            context: Optional task context for advanced selection
            
        Returns:
            Selected model name
        """
        if available_models is None:
            available_models = list(MODELS.keys())
        
        if tier != ModelTier.AUTO:
            tier_models = TIER_MODELS.get(tier, [])
            available_models = [m for m in available_models if m in tier_models]
        
        if preferred_provider:
            available_models = [
                m for m in available_models 
                if MODELS.get(m, ModelInfo("", "", 0, 0, 0, 0, {})).provider == preferred_provider
            ]
        
        if not available_models:
            return "gpt-3.5-turbo"
        
        if context:
            selection = self._select_with_context(task_type, available_models, context)
            return selection.model_name
        
        if tier == ModelTier.AUTO:
            return self._score_models_for_task(task_type, available_models)
        
        return available_models[0]
    
    def select_model_advanced(
        self,
        task_type: TaskType,
        code: str = "",
        class_info: Optional[Dict[str, Any]] = None,
        strategy: Optional[SelectionStrategy] = None,
        max_cost_usd: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        preferred_provider: Optional[str] = None,
        available_models: Optional[List[str]] = None,
    ) -> ModelSelection:
        """Advanced model selection with full context.
        
        Args:
            task_type: Type of task
            code: Source code for complexity analysis
            class_info: Optional class information
            strategy: Selection strategy
            max_cost_usd: Maximum cost budget
            max_latency_ms: Maximum latency requirement
            preferred_provider: Preferred provider
            available_models: List of available models
            
        Returns:
            ModelSelection with detailed information
        """
        strategy = strategy or self._strategy
        available_models = available_models or list(MODELS.keys())
        
        complexity = self.analyze_complexity(code, class_info) if code else TaskComplexity.MODERATE
        
        estimated_input_tokens = len(code) // 4 if code else 1000
        estimated_output_tokens = max(2000, estimated_input_tokens // 2)
        
        context = TaskContext(
            task_type=task_type,
            complexity=complexity,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            max_cost_usd=max_cost_usd,
            max_latency_ms=max_latency_ms,
            preferred_provider=preferred_provider,
        )
        
        return self._select_with_context(task_type, available_models, context, strategy)
    
    def _select_with_context(
        self,
        task_type: TaskType,
        models: List[str],
        context: TaskContext,
        strategy: Optional[SelectionStrategy] = None
    ) -> ModelSelection:
        """Select model with full context consideration.
        
        Args:
            task_type: Task type
            models: Available models
            context: Task context
            strategy: Selection strategy
            
        Returns:
            ModelSelection
        """
        strategy = strategy or self._strategy
        
        candidates = []
        
        for model_name in models:
            model_info = MODELS.get(model_name)
            if not model_info:
                continue
            
            if context.preferred_provider and model_info.provider != context.preferred_provider:
                continue
            
            if context.requires_streaming and not model_info.supports_streaming:
                continue
            
            if context.requires_function_calling and not model_info.supports_function_calling:
                continue
            
            estimated_cost = self.estimate_cost(
                model_name,
                context.estimated_input_tokens,
                context.estimated_output_tokens
            )
            
            if context.max_cost_usd and estimated_cost > context.max_cost_usd:
                continue
            
            if context.max_latency_ms and model_info.avg_latency_ms > context.max_latency_ms:
                continue
            
            quality_score = self._calculate_quality_score(model_info, task_type)
            
            if quality_score < context.quality_threshold:
                continue
            
            candidates.append({
                "model_name": model_name,
                "model_info": model_info,
                "cost": estimated_cost,
                "latency": model_info.avg_latency_ms,
                "quality": quality_score,
            })
        
        if not candidates:
            fallback_model = models[0] if models else "gpt-3.5-turbo"
            fallback_info = MODELS.get(fallback_model, ModelInfo("", "", 0, 0, 0, 0, {}))
            return ModelSelection(
                model_name=fallback_model,
                model_info=fallback_info,
                estimated_cost=0.0,
                estimated_latency_ms=1000.0,
                quality_score=0.5,
                selection_reason="Fallback: No models matched criteria",
                fallback_models=[],
            )
        
        weights = TASK_COMPLEXITY_WEIGHTS[context.complexity]
        
        if strategy == SelectionStrategy.COST_OPTIMIZED:
            weights = {"cost": 0.6, "quality": 0.3, "speed": 0.1}
        elif strategy == SelectionStrategy.QUALITY_OPTIMIZED:
            weights = {"cost": 0.1, "quality": 0.7, "speed": 0.2}
        elif strategy == SelectionStrategy.SPEED_OPTIMIZED:
            weights = {"cost": 0.1, "quality": 0.3, "speed": 0.6}
        
        scored_candidates = []
        for c in candidates:
            score = (
                weights["quality"] * c["quality"] +
                weights["speed"] * (1 - c["latency"] / 5000) +  
                weights["cost"] * (1 - min(c["cost"] / 0.1, 1.0))  
            )
            
            perf = self._model_performance[c["model_name"]]
            score *= perf["success_rate"]
            
            scored_candidates.append((score, c))
        
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        best = scored_candidates[0][1]
        fallbacks = [c[1]["model_name"] for c in scored_candidates[1:4]]
        
        return ModelSelection(
            model_name=best["model_name"],
            model_info=best["model_info"],
            estimated_cost=best["cost"],
            estimated_latency_ms=best["latency"],
            quality_score=best["quality"],
            selection_reason=f"Selected based on {strategy.value} strategy for {context.complexity.name} task",
            fallback_models=fallbacks,
        )
    
    def _calculate_quality_score(
        self,
        model_info: ModelInfo,
        task_type: TaskType
    ) -> float:
        """Calculate quality score for a model on a task.
        
        Args:
            model_info: Model information
            task_type: Task type
            
        Returns:
            Quality score (0-1)
        """
        capabilities = TASK_CAPABILITIES.get(task_type, ["coding"])
        
        capability_score = sum(
            model_info.capabilities.get(cap, 0.5)
            for cap in capabilities
        ) / len(capabilities)
        
        reliability_score = model_info.reliability_score
        
        return (capability_score * 0.7 + reliability_score * 0.3)
    
    def _score_models_for_task(
        self,
        task_type: TaskType,
        models: List[str]
    ) -> str:
        """Score and select best model for task.
        
        Args:
            task_type: Task type
            models: Available models
            
        Returns:
            Best model name
        """
        capabilities = TASK_CAPABILITIES.get(task_type, ["coding"])
        
        best_model = models[0]
        best_score = 0.0
        
        for model_name in models:
            model_info = MODELS.get(model_name)
            if not model_info:
                continue
            
            score = sum(
                model_info.capabilities.get(cap, 0.0) 
                for cap in capabilities
            ) / len(capabilities)
            
            perf = self._model_performance.get(model_name, {})
            score *= perf.get("success_rate", 0.9)
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def estimate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Estimate cost for operation.
        
        Args:
            model_name: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        model_info = MODELS.get(model_name)
        if not model_info:
            return 0.0
        
        input_cost = (input_tokens / 1000) * model_info.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model_info.cost_per_1k_output
        
        return input_cost + output_cost
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get model information.
        
        Args:
            model_name: Model name
            
        Returns:
            ModelInfo if found, None otherwise
        """
        return MODELS.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all available models.
        
        Returns:
            List of model names
        """
        return list(MODELS.keys())
    
    def get_models_by_provider(self, provider: str) -> List[str]:
        """Get models by provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of model names
        """
        return [
            name for name, info in MODELS.items()
            if info.provider == provider
        ]
    
    def record_usage(self, model_name: str, tokens: int):
        """Record model usage.
        
        Args:
            model_name: Model name
            tokens: Number of tokens used
        """
        if model_name not in self._usage_stats:
            self._usage_stats[model_name] = 0
        self._usage_stats[model_name] += tokens
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics.
        
        Returns:
            Dictionary of model -> token count
        """
        return self._usage_stats.copy()
