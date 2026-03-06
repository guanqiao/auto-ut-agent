"""Capability Registry for managing and loading capabilities.

This module provides the CapabilityRegistry class that handles:
- Capability registration
- Dependency resolution
- Lifecycle management
- Configuration-based loading
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Type, TYPE_CHECKING

from .base import (
    Capability,
    CapabilityMetadata,
    CapabilityPriority,
    CapabilityState,
)

if TYPE_CHECKING:
    from ...core.container import Container

logger = logging.getLogger(__name__)


@dataclass
class CapabilityConfig:
    """Configuration for capability loading."""
    enabled: bool = True
    priority: Optional[CapabilityPriority] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)


class CapabilityRegistry:
    """Registry for managing agent capabilities.
    
    The CapabilityRegistry handles:
    - Registration of capability classes
    - Dependency resolution and ordering
    - Lifecycle management (initialize, shutdown)
    - Configuration-based loading
    
    Example:
        registry = CapabilityRegistry(container)
        
        # Register capabilities
        registry.register(ContextManagementCapability)
        registry.register(ErrorLearningCapability)
        
        # Load all enabled capabilities
        registry.load_all(config)
        
        # Get a capability
        context_cap = registry.get("context_management")
    """
    
    def __init__(self, container: Optional["Container"] = None):
        """Initialize the registry.
        
        Args:
            container: Optional dependency injection container
        """
        self._container = container
        self._capabilities: Dict[str, Capability] = {}
        self._metadata: Dict[str, CapabilityMetadata] = {}
        self._configs: Dict[str, CapabilityConfig] = {}
        self._load_order: List[str] = []
        self._initialized = False
    
    @property
    def container(self) -> Optional["Container"]:
        """Get the container."""
        return self._container
    
    @container.setter
    def container(self, value: "Container") -> None:
        """Set the container."""
        self._container = value
    
    @property
    def capabilities(self) -> Dict[str, Capability]:
        """Get all registered capabilities."""
        return self._capabilities.copy()
    
    @property
    def is_initialized(self) -> bool:
        """Check if registry has been initialized."""
        return self._initialized
    
    def register(
        self,
        capability_class: Type[Capability],
        config: Optional[CapabilityConfig] = None
    ) -> str:
        """Register a capability class.
        
        Args:
            capability_class: The capability class to register
            config: Optional configuration for the capability
            
        Returns:
            The name of the registered capability
            
        Raises:
            ValueError: If capability with same name already registered
        """
        metadata = capability_class.metadata()
        name = metadata.name
        
        if name in self._metadata:
            logger.warning(f"[CapabilityRegistry] Overwriting existing capability: {name}")
        
        self._metadata[name] = metadata
        self._configs[name] = config or CapabilityConfig()
        
        instance = capability_class()
        self._capabilities[name] = instance
        
        logger.debug(f"[CapabilityRegistry] Registered capability: {name}")
        return name
    
    def unregister(self, name: str) -> bool:
        """Unregister a capability.
        
        Args:
            name: Name of the capability to unregister
            
        Returns:
            True if capability was unregistered, False if not found
        """
        if name not in self._capabilities:
            return False
        
        cap = self._capabilities[name]
        if cap.state == CapabilityState.READY:
            cap.shutdown()
        
        del self._capabilities[name]
        del self._metadata[name]
        if name in self._configs:
            del self._configs[name]
        
        logger.debug(f"[CapabilityRegistry] Unregistered capability: {name}")
        return True
    
    def get(self, name: str) -> Optional[Capability]:
        """Get a capability by name.
        
        Args:
            name: Name of the capability
            
        Returns:
            The capability instance, or None if not found
        """
        return self._capabilities.get(name)
    
    def get_metadata(self, name: str) -> Optional[CapabilityMetadata]:
        """Get capability metadata by name.
        
        Args:
            name: Name of the capability
            
        Returns:
            The capability metadata, or None if not found
        """
        return self._metadata.get(name)
    
    def get_all_ready(self) -> List[Capability]:
        """Get all capabilities that are ready to use.
        
        Returns:
            List of ready capabilities
        """
        return [
            cap for cap in self._capabilities.values()
            if cap.state == CapabilityState.READY
        ]
    
    def get_by_priority(self, priority: CapabilityPriority) -> List[Capability]:
        """Get all capabilities with a specific priority.
        
        Args:
            priority: The priority level to filter by
            
        Returns:
            List of capabilities with the specified priority
        """
        return [
            cap for name, cap in self._capabilities.items()
            if self._metadata[name].priority == priority
        ]
    
    def get_by_tag(self, tag: str) -> List[Capability]:
        """Get all capabilities with a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of capabilities with the specified tag
        """
        return [
            cap for name, cap in self._capabilities.items()
            if tag in self._metadata[name].tags
        ]
    
    def resolve_dependencies(self) -> List[str]:
        """Resolve dependency order for initialization.
        
        Returns:
            List of capability names in initialization order
            
        Raises:
            ValueError: If circular dependencies detected
        """
        visited: Set[str] = set()
        visiting: Set[str] = set()
        order: List[str] = []
        
        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"Circular dependency detected: {name}")
            
            visiting.add(name)
            
            metadata = self._metadata.get(name)
            if metadata:
                for dep in metadata.dependencies:
                    if dep in self._metadata:
                        visit(dep)
            
            visiting.remove(name)
            visited.add(name)
            order.append(name)
        
        sorted_names = sorted(
            self._metadata.keys(),
            key=lambda n: (self._metadata[n].priority.value, n)
        )
        
        for name in sorted_names:
            visit(name)
        
        return order
    
    def load_all(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """Load and initialize all registered capabilities.
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            Dictionary mapping capability names to success status
        """
        if self._initialized:
            logger.warning("[CapabilityRegistry] Already initialized, skipping load_all")
            return {name: cap.state == CapabilityState.READY 
                    for name, cap in self._capabilities.items()}
        
        if config:
            self._apply_config(config)
        
        load_order = self.resolve_dependencies()
        results: Dict[str, bool] = {}
        
        logger.info(f"[CapabilityRegistry] Loading {len(load_order)} capabilities")
        
        for name in load_order:
            cap = self._capabilities.get(name)
            cap_config = self._configs.get(name, CapabilityConfig())
            
            if not cap_config.enabled:
                cap.disable()
                results[name] = False
                logger.info(f"[CapabilityRegistry] Capability {name} is disabled")
                continue
            
            try:
                if self._container:
                    cap.initialize(self._container)
                else:
                    cap._state = CapabilityState.READY
                
                results[name] = True
                logger.info(f"[CapabilityRegistry] Loaded capability: {name}")
            except Exception as e:
                cap._set_error(e)
                results[name] = False
                logger.error(f"[CapabilityRegistry] Failed to load {name}: {e}")
        
        self._load_order = load_order
        self._initialized = True
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"[CapabilityRegistry] Loaded {success_count}/{len(results)} capabilities")
        
        return results
    
    def shutdown_all(self) -> None:
        """Shutdown all capabilities in reverse initialization order."""
        for name in reversed(self._load_order):
            cap = self._capabilities.get(name)
            if cap and cap.state == CapabilityState.READY:
                try:
                    cap.shutdown()
                except Exception as e:
                    logger.warning(f"[CapabilityRegistry] Error shutting down {name}: {e}")
        
        self._initialized = False
        logger.info("[CapabilityRegistry] All capabilities shutdown")
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration to capabilities.
        
        Args:
            config: Configuration dictionary
        """
        for name, cap_config in config.items():
            if name not in self._configs:
                continue
            
            if isinstance(cap_config, bool):
                self._configs[name].enabled = cap_config
            elif isinstance(cap_config, dict):
                if "enabled" in cap_config:
                    self._configs[name].enabled = cap_config["enabled"]
                if "priority" in cap_config:
                    self._configs[name].priority = cap_config["priority"]
                if "custom_config" in cap_config:
                    self._configs[name].custom_config.update(cap_config["custom_config"])
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all capabilities.
        
        Returns:
            Dictionary with capability status information
        """
        return {
            "initialized": self._initialized,
            "total_count": len(self._capabilities),
            "ready_count": len(self.get_all_ready()),
            "capabilities": {
                name: {
                    "state": cap.state.name,
                    "priority": self._metadata[name].priority.name,
                    "enabled": self._configs.get(name, CapabilityConfig()).enabled,
                    "error": str(cap.error) if cap.error else None,
                }
                for name, cap in self._capabilities.items()
            }
        }
    
    def check_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all capabilities.
        
        Returns:
            Dictionary mapping capability names to health status
        """
        return {
            name: cap.check_health()
            for name, cap in self._capabilities.items()
            if cap.is_ready
        }
    
    def auto_disable_unhealthy(self) -> List[str]:
        """Automatically disable unhealthy capabilities.
        
        This method checks the health of all ready capabilities
        and disables those that are unhealthy with 'disable' recommendation.
        
        Returns:
            List of disabled capability names
        """
        disabled = []
        for name, cap in self._capabilities.items():
            if not cap.is_ready:
                continue
            
            health = cap.check_health()
            if not health.get("healthy") and health.get("recommendation") == "disable":
                cap.disable()
                disabled.append(name)
                logger.warning(
                    f"[CapabilityRegistry] Auto-disabled unhealthy capability: {name} "
                    f"(reason: {health.get('reason')})"
                )
        
        if disabled:
            logger.info(f"[CapabilityRegistry] Auto-disabled {len(disabled)} unhealthy capabilities")
        
        return disabled
    
    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry statistics from all capabilities.
        
        Returns:
            Dictionary with retry statistics for all ready capabilities
        """
        return {
            name: cap.get_retry_stats()
            for name, cap in self._capabilities.items()
            if cap.is_ready
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of all capability health.
        
        Returns:
            Dictionary with health summary
        """
        health_status = self.check_all_health()
        
        healthy_count = sum(1 for h in health_status.values() if h.get("healthy"))
        unhealthy_count = len(health_status) - healthy_count
        monitor_count = sum(
            1 for h in health_status.values()
            if h.get("recommendation") == "monitor"
        )
        
        return {
            "total_capabilities": len(self._capabilities),
            "ready_capabilities": len(self.get_all_ready()),
            "healthy_count": healthy_count,
            "unhealthy_count": unhealthy_count,
            "monitor_recommended_count": monitor_count,
            "health_details": health_status,
        }
    
    def __contains__(self, name: str) -> bool:
        """Check if a capability is registered."""
        return name in self._capabilities
    
    def __len__(self) -> int:
        """Get number of registered capabilities."""
        return len(self._capabilities)
    
    def __repr__(self) -> str:
        return f"CapabilityRegistry(count={len(self._capabilities)}, initialized={self._initialized})"
