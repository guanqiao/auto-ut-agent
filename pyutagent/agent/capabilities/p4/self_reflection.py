"""Self Reflection Capability (P4).

Provides self-reflection and quality assessment for generated code.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority
from ...execution.retry import RetryConfig

if TYPE_CHECKING:
    from ....core.container import Container
    from ....self_reflection import SelfReflection

logger = __import__('logging').getLogger(__name__)


class SelfReflectionCapability(Capability):
    """Self-reflection capability.
    
    This capability provides:
    - Quality assessment of generated code
    - Issue identification
    - Improvement suggestions
    - Regeneration recommendations
    
    Configuration:
        quality_threshold: Minimum quality score (default: 0.7)
        enable_deep_analysis: Enable deep analysis (default: True)
    """
    
    _reflection: Any = None
    _quality_threshold: float = 0.7
    _enable_deep_analysis: bool = True
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="self_reflection",
            description="Self-reflection and quality assessment",
            priority=CapabilityPriority.NORMAL,
            provides={"self_reflection"},
            dependencies={"generation_evaluation"},
            tags={"p4", "quality", "reflection"}
        )
    
    def _create_default_retry_config(self) -> "RetryConfig":
        """Create default retry config for self-reflection.
        
        Self-reflection involves LLM calls and should have smart retry
        with moderate backoff.
        """
        return RetryConfig(
            max_attempts=3,
            base_delay=2.0,
            max_delay=30.0,
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize self-reflection.
        
        Args:
            container: Dependency injection container
        """
        self._container = container
        
        try:
            from ....self_reflection import SelfReflection
            
            config = self._get_config()
            self._quality_threshold = config.get("quality_threshold", 0.7)
            self._enable_deep_analysis = config.get("enable_deep_analysis", True)
            
            self._reflection = SelfReflection(
                quality_threshold=self._quality_threshold,
                enable_deep_analysis=self._enable_deep_analysis
            )
            
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info(
                f"[SelfReflectionCapability] Initialized - "
                f"threshold={self._quality_threshold}, deep={self._enable_deep_analysis}"
            )
            
        except Exception as e:
            self._set_error(e)
            raise
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration from container or defaults."""
        if self._container:
            try:
                settings = self._container.resolve(type('Settings', (), {}))
                if hasattr(settings, 'self_reflection'):
                    return {
                        "quality_threshold": getattr(
                            settings.self_reflection, 'quality_threshold', 0.7
                        ),
                        "enable_deep_analysis": getattr(
                            settings.self_reflection, 'enable_deep_analysis', True
                        ),
                    }
            except Exception:
                pass
        return {}
    
    @property
    def reflection(self) -> Any:
        """Get the reflection instance."""
        return self._reflection
    
    async def critique_generated_test(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        class_info: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Critique generated test code.
        
        Args:
            test_code: Generated test code
            source_code: Original source code
            class_info: Class information
            
        Returns:
            CritiqueResult with quality metrics
        """
        if not self._reflection:
            return None
        
        return await self._reflection.critique_generated_test(
            test_code=test_code,
            source_code=source_code,
            class_info=class_info
        )
    
    def get_critique_stats(self) -> Dict[str, Any]:
        """Get critique statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self._reflection:
            return {"enabled": False}
        
        return self._reflection.get_critique_stats()
    
    def shutdown(self) -> None:
        """Shutdown reflection."""
        self._reflection = None
        super().shutdown()
