"""Sandbox Execution Capability (P3).

Provides sandboxed code execution for safety.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class SandboxExecutionCapability(Capability):
    """Sandbox execution capability.
    
    This capability provides:
    - Sandboxed code execution
    - Security level management
    - Resource limiting
    """
    
    _executor: Any = None
    _security_level: str = "moderate"
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="sandbox_execution",
            description="Sandboxed code execution for safety",
            priority=CapabilityPriority.NORMAL,
            provides={"sandbox_executor"},
            tags={"p3", "sandbox", "security"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize sandbox executor."""
        self._container = container
        
        try:
            from ....core.sandbox_executor import SandboxExecutor, SecurityLevel
            
            level = SecurityLevel.MODERATE
            self._executor = SandboxExecutor(security_level=level)
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[SandboxExecutionCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def executor(self) -> Any:
        """Get the executor instance."""
        return self._executor
    
    def shutdown(self) -> None:
        """Shutdown executor."""
        self._executor = None
        super().shutdown()
