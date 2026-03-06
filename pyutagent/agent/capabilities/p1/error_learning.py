"""Error Learning Capability (P1).

Provides learning from errors for improved recovery strategies.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class ErrorLearningCapability(Capability):
    """Error learning capability.
    
    This capability provides:
    - Error pattern learning
    - Strategy recommendation
    - Knowledge base integration
    """
    
    _learner: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="error_learning",
            description="Learn from errors for improved recovery",
            priority=CapabilityPriority.NORMAL,
            provides={"error_learner"},
            dependencies={"context_management"},
            tags={"p1", "learning", "recovery"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize error learner."""
        self._container = container
        
        try:
            from ....core.error_learner import ErrorLearner
            
            self._learner = ErrorLearner()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[ErrorLearningCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def learner(self) -> Any:
        """Get the learner instance."""
        return self._learner
    
    def shutdown(self) -> None:
        """Shutdown learner."""
        self._learner = None
        super().shutdown()
