"""Pattern Library Capability (P4).

Provides test pattern library for reusable patterns.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class PatternLibraryCapability(Capability):
    """Pattern library capability.
    
    This capability provides:
    - Test pattern storage
    - Pattern matching
    - Pattern recommendation
    """
    
    _library: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="pattern_library",
            description="Test pattern library for reusable patterns",
            priority=CapabilityPriority.NORMAL,
            provides={"pattern_library"},
            tags={"p4", "pattern", "library"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize pattern library."""
        self._container = container
        
        try:
            from ....memory.pattern_library import PatternLibrary
            
            self._library = PatternLibrary()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[PatternLibraryCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def library(self) -> Any:
        """Get the library instance."""
        return self._library
    
    def shutdown(self) -> None:
        """Shutdown library."""
        self._library = None
        super().shutdown()
