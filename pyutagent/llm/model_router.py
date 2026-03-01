"""Model router for selecting appropriate LLM based on task."""

import logging
from enum import Enum
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tier enumeration."""
    LITE = "lite"               # Lightweight - fast response
    EFFICIENT = "efficient"     # Cost efficient - balanced
    PERFORMANCE = "performance" # High performance - deep understanding
    AUTO = "auto"               # Auto select based on task


class TaskType(str, Enum):
    """Task type enumeration."""
    CODE_COMPLETION = "code_completion"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    COVERAGE_OPTIMIZATION = "coverage_optimization"
    EXPLANATION = "explanation"


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


# Predefined model information
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
        }
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
        }
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
        }
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
        }
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
        }
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
        }
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
        }
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
    TaskType.TEST_GENERATION: ["coding", "reasoning"],
    TaskType.CODE_REVIEW: ["reasoning", "context_length"],
    TaskType.COVERAGE_OPTIMIZATION: ["coding", "reasoning"],
    TaskType.EXPLANATION: ["reasoning", "context_length"],
}


class ModelRouter:
    """Router for selecting appropriate LLM model.
    
    Features:
    - Select model by tier or task type
    - Estimate cost for operations
    - Get model information
    """
    
    def __init__(self):
        """Initialize model router."""
        self._usage_stats: Dict[str, int] = {}
    
    def select_model(
        self,
        task_type: TaskType,
        tier: ModelTier = ModelTier.AUTO,
        preferred_provider: Optional[str] = None,
        available_models: Optional[List[str]] = None
    ) -> str:
        """Select appropriate model for task.
        
        Args:
            task_type: Type of task
            tier: Model tier preference
            preferred_provider: Preferred provider
            available_models: List of available models
            
        Returns:
            Selected model name
        """
        if available_models is None:
            available_models = list(MODELS.keys())
        
        # Filter by tier if specified
        if tier != ModelTier.AUTO:
            tier_models = TIER_MODELS.get(tier, [])
            available_models = [m for m in available_models if m in tier_models]
        
        # Filter by provider if specified
        if preferred_provider:
            available_models = [
                m for m in available_models 
                if MODELS.get(m, ModelInfo("", "", 0, 0, 0, 0, {})).provider == preferred_provider
            ]
        
        if not available_models:
            # Fallback to default
            return "gpt-3.5-turbo"
        
        # Score models based on task capabilities
        if tier == ModelTier.AUTO:
            return self._score_models_for_task(task_type, available_models)
        
        # Return first available model in tier
        return available_models[0]
    
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
            
            # Calculate score based on capabilities
            score = sum(
                model_info.capabilities.get(cap, 0.0) 
                for cap in capabilities
            ) / len(capabilities)
            
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
