"""Boundary Analysis Capability (P4).

Provides boundary value analysis for test case generation.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class BoundaryAnalysisCapability(Capability):
    """Boundary analysis capability.
    
    This capability provides:
    - Boundary value detection
    - Edge case identification
    - Test case suggestion
    """
    
    _analyzer: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="boundary_analysis",
            description="Boundary value analysis for test case generation",
            priority=CapabilityPriority.NORMAL,
            provides={"boundary_analyzer"},
            tags={"p4", "boundary", "analysis"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize boundary analyzer."""
        self._container = container
        
        try:
            from ....core.boundary_analyzer import BoundaryAnalyzer
            
            self._analyzer = BoundaryAnalyzer()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[BoundaryAnalysisCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def analyzer(self) -> Any:
        """Get the analyzer instance."""
        return self._analyzer
    
    def shutdown(self) -> None:
        """Shutdown analyzer."""
        self._analyzer = None
        super().shutdown()
