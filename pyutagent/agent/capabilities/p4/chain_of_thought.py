"""Chain of Thought Capability (P4).

Provides chain-of-thought reasoning for complex problems.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class ChainOfThoughtCapability(Capability):
    """Chain of thought capability.
    
    This capability provides:
    - Structured reasoning
    - Step-by-step analysis
    - Thought chain management
    """
    
    _engine: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="chain_of_thought",
            description="Chain-of-thought reasoning for complex problems",
            priority=CapabilityPriority.NORMAL,
            provides={"cot_engine"},
            tags={"p4", "reasoning", "cot"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize chain-of-thought engine."""
        self._container = container
        
        try:
            from ....llm.chain_of_thought import ChainOfThoughtEngine
            
            self._engine = ChainOfThoughtEngine()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[ChainOfThoughtCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def engine(self) -> Any:
        """Get the engine instance."""
        return self._engine
    
    def shutdown(self) -> None:
        """Shutdown engine."""
        self._engine = None
        super().shutdown()
