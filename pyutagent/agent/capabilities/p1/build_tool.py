"""Build Tool Capability (P1).

Provides build tool management and execution.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class BuildToolCapability(Capability):
    """Build tool capability.
    
    This capability provides:
    - Maven/Gradle detection
    - Build command execution
    - Dependency management
    """
    
    _manager: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="build_tool",
            description="Build tool management and execution",
            priority=CapabilityPriority.NORMAL,
            provides={"build_tool_manager"},
            tags={"p1", "build", "tools"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize build tool manager."""
        self._container = container
        
        try:
            from ....tools.build_tool_manager import BuildToolManager
            
            self._manager = BuildToolManager()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[BuildToolCapability] Initialized")
            
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
