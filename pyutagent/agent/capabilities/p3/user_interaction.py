"""User Interaction Capability (P3).

Provides user interaction for confirmations and feedback.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class UserInteractionCapability(Capability):
    """User interaction capability.
    
    This capability provides:
    - User confirmation requests
    - Feedback collection
    - Auto-decision for high confidence
    """
    
    _handler: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="user_interaction",
            description="User interaction for confirmations and feedback",
            priority=CapabilityPriority.NORMAL,
            provides={"user_interaction_handler"},
            tags={"p3", "user", "interaction"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize user interaction handler."""
        self._container = container
        
        try:
            from ....user_interaction import UserInteractionHandler
            
            self._handler = UserInteractionHandler()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[UserInteractionCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def handler(self) -> Any:
        """Get the handler instance."""
        return self._handler
    
    def shutdown(self) -> None:
        """Shutdown handler."""
        self._handler = None
        super().shutdown()
