"""Knowledge Sharing Capability (P2).

Provides knowledge sharing between agents.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class KnowledgeSharingCapability(Capability):
    """Knowledge sharing capability.
    
    This capability provides:
    - Shared knowledge base
    - Experience replay
    - Knowledge synchronization
    """
    
    _knowledge_base: Any = None
    _experience_replay: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="knowledge_sharing",
            description="Knowledge sharing between agents",
            priority=CapabilityPriority.NORMAL,
            provides={"shared_knowledge", "experience_replay"},
            dependencies={"multi_agent"},
            tags={"p2", "knowledge", "sharing"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize knowledge sharing."""
        self._container = container
        
        try:
            from ...multi_agent import SharedKnowledgeBase, ExperienceReplay
            
            self._knowledge_base = SharedKnowledgeBase()
            self._experience_replay = ExperienceReplay()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[KnowledgeSharingCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def knowledge_base(self) -> Any:
        """Get the knowledge base instance."""
        return self._knowledge_base
    
    @property
    def experience_replay(self) -> Any:
        """Get the experience replay instance."""
        return self._experience_replay
    
    def shutdown(self) -> None:
        """Shutdown knowledge sharing."""
        self._knowledge_base = None
        self._experience_replay = None
        super().shutdown()
