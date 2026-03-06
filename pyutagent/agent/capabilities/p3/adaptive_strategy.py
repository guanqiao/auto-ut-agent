"""Adaptive Strategy Capability (P3).

Provides adaptive strategy selection for error recovery.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class AdaptiveStrategyCapability(Capability):
    """Adaptive strategy capability.
    
    This capability provides:
    - Strategy selection based on context
    - Performance tracking
    - Strategy optimization
    """
    
    _manager: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="adaptive_strategy",
            description="Adaptive strategy selection for recovery",
            priority=CapabilityPriority.NORMAL,
            provides={"strategy_manager"},
            dependencies={"error_prediction"},
            tags={"p3", "strategy", "optimization"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize strategy manager."""
        self._container = container
        
        try:
            from ....core.adaptive_strategy import AdaptiveStrategyManager
            
            self._manager = AdaptiveStrategyManager()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[AdaptiveStrategyCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def manager(self) -> Any:
        """Get the manager instance."""
        return self._manager
    
    def shutdown(self) -> None:
        """Shutdown manager."""
        self._manager = None
        super().shutdown()
