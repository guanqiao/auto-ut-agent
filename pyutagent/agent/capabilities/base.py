"""Base classes for Agent Capabilities.

This module defines the abstract base class for all capabilities
and the metadata that describes them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.container import Container

logger = __import__('logging').getLogger(__name__)


class CapabilityPriority(Enum):
    """Priority levels for capability initialization."""
    CRITICAL = auto()
    HIGH = auto()
    NORMAL = auto()
    LOW = auto()
    OPTIONAL = auto()


class CapabilityState(Enum):
    """State of a capability."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    ERROR = auto()
    DISABLED = auto()


@dataclass
class CapabilityMetadata:
    """Metadata describing a capability."""
    name: str
    description: str
    priority: CapabilityPriority = CapabilityPriority.NORMAL
    dependencies: Set[str] = field(default_factory=set)
    provides: Set[str] = field(default_factory=set)
    config_key: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if not isinstance(other, CapabilityMetadata):
            return False
        return self.name == other.name


class Capability(ABC):
    """Abstract base class for all agent capabilities.
    
    Capabilities are modular features that can be loaded on-demand
    based on configuration. Each capability:
    
    1. Has metadata describing its name, dependencies, and what it provides
    2. Can be initialized with a dependency injection container
    3. Can be enabled/disabled via configuration
    4. Provides specific functionality to the agent
    
    Example:
        class ContextManagementCapability(Capability):
            @classmethod
            def metadata(cls) -> CapabilityMetadata:
                return CapabilityMetadata(
                    name="context_management",
                    description="Context compression and management",
                    priority=CapabilityPriority.HIGH,
                    provides={"context_manager", "context_compressor"}
                )
            
            def initialize(self, container: "Container") -> None:
                self.context_manager = container.resolve(ContextManager)
                self.context_compressor = container.resolve(ContextCompressor)
    """
    
    def __init__(self):
        self._state: CapabilityState = CapabilityState.UNINITIALIZED
        self._error: Optional[Exception] = None
        self._container: Optional["Container"] = None
    
    @classmethod
    @abstractmethod
    def metadata(cls) -> CapabilityMetadata:
        """Return metadata describing this capability.
        
        Returns:
            CapabilityMetadata with name, description, dependencies, etc.
        """
        pass
    
    @property
    def name(self) -> str:
        """Get capability name from metadata."""
        return self.metadata().name
    
    @property
    def state(self) -> CapabilityState:
        """Get current capability state."""
        return self._state
    
    @property
    def error(self) -> Optional[Exception]:
        """Get error if capability failed to initialize."""
        return self._error
    
    @property
    def is_ready(self) -> bool:
        """Check if capability is ready to use."""
        return self._state == CapabilityState.READY
    
    def initialize(self, container: "Container") -> None:
        """Initialize the capability with dependencies.
        
        Override this method to set up the capability with required
        dependencies from the container. The default implementation
        just marks the capability as ready.
        
        Args:
            container: Dependency injection container
            
        Raises:
            Exception: If initialization fails
        """
        self._container = container
        self._state = CapabilityState.READY
        logger.info(f"[Capability] {self.name} initialized successfully")
    
    def shutdown(self) -> None:
        """Shutdown the capability and release resources.
        
        Override this method to clean up resources when the capability
        is no longer needed.
        """
        self._state = CapabilityState.UNINITIALIZED
        logger.info(f"[Capability] {self.name} shutdown complete")
    
    def enable(self) -> None:
        """Enable the capability."""
        if self._state == CapabilityState.DISABLED:
            self._state = CapabilityState.UNINITIALIZED
            logger.info(f"[Capability] {self.name} enabled")
    
    def disable(self) -> None:
        """Disable the capability."""
        self._state = CapabilityState.DISABLED
        logger.info(f"[Capability] {self.name} disabled")
    
    def _set_error(self, error: Exception) -> None:
        """Set error state."""
        self._error = error
        self._state = CapabilityState.ERROR
        logger.error(f"[Capability] {self.name} error: {error}")
    
    def resolve(self, interface_type: type) -> Any:
        """Resolve a dependency from the container.
        
        Args:
            interface_type: Type of the dependency to resolve
            
        Returns:
            The resolved instance
            
        Raises:
            RuntimeError: If container is not set
        """
        if self._container is None:
            raise RuntimeError(f"Container not set for capability {self.name}")
        return self._container.resolve(interface_type)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, state={self._state.name})"


class CompositeCapability(Capability):
    """A capability composed of multiple sub-capabilities.
    
    Use this when you want to group related capabilities together.
    """
    
    def __init__(self, sub_capabilities: List[Capability]):
        super().__init__()
        self._sub_capabilities = sub_capabilities
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="composite",
            description="Composite capability",
            priority=CapabilityPriority.NORMAL
        )
    
    def initialize(self, container: "Container") -> None:
        self._container = container
        errors = []
        
        for cap in self._sub_capabilities:
            try:
                cap.initialize(container)
            except Exception as e:
                errors.append((cap.name, e))
        
        if errors:
            self._set_error(Exception(f"Failed to initialize sub-capabilities: {errors}"))
        else:
            self._state = CapabilityState.READY
    
    def shutdown(self) -> None:
        for cap in self._sub_capabilities:
            try:
                cap.shutdown()
            except Exception as e:
                logger.warning(f"[Capability] Error shutting down {cap.name}: {e}")
        super().shutdown()
    
    def get_sub_capability(self, name: str) -> Optional[Capability]:
        """Get a sub-capability by name."""
        for cap in self._sub_capabilities:
            if cap.name == name:
                return cap
        return None
