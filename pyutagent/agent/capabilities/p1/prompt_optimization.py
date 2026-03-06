"""Prompt Optimization Capability (P1).

Provides model-specific prompt optimization and A/B testing
for improved test generation quality.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container
    from ....prompt_optimizer import PromptOptimizer

logger = __import__('logging').getLogger(__name__)


class PromptOptimizationCapability(Capability):
    """Prompt optimization capability.
    
    This capability provides:
    - Model-specific prompt optimization
    - A/B testing for prompt variants
    - Prompt template management
    - Optimization result tracking
    
    Configuration:
        enable_ab_testing: Enable A/B testing (default: False)
        model_name: Default model name (default: "gpt-4")
    """
    
    _optimizer: Any = None
    _enable_ab_testing: bool = False
    _model_name: str = "gpt-4"
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="prompt_optimization",
            description="Model-specific prompt optimization and A/B testing",
            priority=CapabilityPriority.NORMAL,
            provides={"prompt_optimizer"},
            tags={"p1", "optimization", "llm"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize prompt optimizer.
        
        Args:
            container: Dependency injection container
        """
        self._container = container
        
        try:
            from ....prompt_optimizer import PromptOptimizer, ModelType
            
            config = self._get_config()
            self._enable_ab_testing = config.get("enable_ab_testing", False)
            self._model_name = config.get("model_name", "gpt-4")
            
            model_type = self._detect_model_type(self._model_name)
            
            self._optimizer = PromptOptimizer(
                model_type=model_type,
                enable_ab_testing=self._enable_ab_testing
            )
            
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info(
                f"[PromptOptimizationCapability] Initialized - "
                f"model={self._model_name}, ab_testing={self._enable_ab_testing}"
            )
            
        except Exception as e:
            self._set_error(e)
            raise
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration from container or defaults."""
        if self._container:
            try:
                settings = self._container.resolve(type('Settings', (), {}))
                if hasattr(settings, 'prompt'):
                    return {
                        "enable_ab_testing": getattr(
                            settings.prompt, 'enable_ab_testing', False
                        ),
                        "model_name": getattr(
                            settings.llm, 'model', 'gpt-4'
                        ),
                    }
            except Exception:
                pass
        return {}
    
    def _detect_model_type(self, model_name: str) -> Any:
        """Detect model type from model name."""
        try:
            from ....prompt_optimizer import ModelType
            
            model_lower = model_name.lower()
            if 'gpt-4' in model_lower:
                return ModelType.GPT4
            elif 'gpt-3.5' in model_lower:
                return ModelType.GPT35
            elif 'claude' in model_lower:
                return ModelType.CLAUDE
            elif 'deepseek' in model_lower:
                return ModelType.DEEPSEEK
            else:
                return ModelType.GPT4
        except Exception:
            return None
    
    @property
    def optimizer(self) -> Any:
        """Get the optimizer instance."""
        return self._optimizer
    
    def optimize_for_model(
        self,
        base_prompt: str,
        model_name: str,
        task_type: str
    ) -> str:
        """Optimize prompt for a specific model.
        
        Args:
            base_prompt: The base prompt to optimize
            model_name: Target model name
            task_type: Type of task (test_generation, error_fix, etc.)
            
        Returns:
            Optimized prompt
        """
        if not self._optimizer:
            return base_prompt
        
        return self._optimizer.optimize_for_model(
            base_prompt=base_prompt,
            model_name=model_name,
            task_type=task_type
        )
    
    def get_prompt_for_test(
        self,
        test_id: str,
        **kwargs
    ) -> tuple:
        """Get prompt variant for A/B testing.
        
        Args:
            test_id: A/B test ID
            **kwargs: Template variables
            
        Returns:
            Tuple of (variant_id, prompt)
        """
        if not self._optimizer:
            return None, None
        
        return self._optimizer.get_prompt_for_test(test_id, **kwargs)
    
    def record_ab_test_result(
        self,
        test_id: str,
        variant_id: str,
        success: bool,
        response_time_ms: int = 0
    ) -> None:
        """Record A/B test result.
        
        Args:
            test_id: A/B test ID
            variant_id: Variant ID
            success: Whether the test succeeded
            response_time_ms: Response time in milliseconds
        """
        if self._optimizer:
            self._optimizer.record_ab_test_result(
                test_id=test_id,
                variant_id=variant_id,
                success=success,
                response_time_ms=response_time_ms
            )
    
    def shutdown(self) -> None:
        """Shutdown optimizer."""
        self._optimizer = None
        super().shutdown()
