"""Tests for Capability Registry."""

import pytest
from unittest.mock import Mock, MagicMock

from pyutagent.agent.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityPriority,
    CapabilityState,
)
from pyutagent.agent.capabilities.registry import (
    CapabilityRegistry,
    CapabilityConfig,
)


class CapabilityA(Capability):
    """Test capability A."""
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="capability_a",
            description="Capability A",
            priority=CapabilityPriority.HIGH
        )
    
    def initialize(self, container):
        self._state = CapabilityState.READY


class CapabilityB(Capability):
    """Test capability B with dependency on A."""
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="capability_b",
            description="Capability B",
            priority=CapabilityPriority.NORMAL,
            dependencies={"capability_a"}
        )
    
    def initialize(self, container):
        self._state = CapabilityState.READY


class CapabilityC(Capability):
    """Test capability C with dependency on B."""
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="capability_c",
            description="Capability C",
            priority=CapabilityPriority.LOW,
            dependencies={"capability_b"}
        )
    
    def initialize(self, container):
        self._state = CapabilityState.READY


class TestCapabilityConfig:
    """Tests for CapabilityConfig."""
    
    def test_config_defaults(self):
        """Test default config values."""
        config = CapabilityConfig()
        
        assert config.enabled == True
        assert config.priority is None
        assert config.custom_config == {}
    
    def test_config_custom_values(self):
        """Test config with custom values."""
        config = CapabilityConfig(
            enabled=False,
            priority=CapabilityPriority.HIGH,
            custom_config={"key": "value"}
        )
        
        assert config.enabled == False
        assert config.priority == CapabilityPriority.HIGH
        assert config.custom_config == {"key": "value"}


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry."""
    
    def test_registry_creation(self):
        """Test creating a registry."""
        registry = CapabilityRegistry()
        
        assert registry.container is None
        assert registry.is_initialized == False
        assert len(registry.capabilities) == 0
    
    def test_registry_with_container(self):
        """Test creating registry with container."""
        container = Mock()
        registry = CapabilityRegistry(container=container)
        
        assert registry.container == container
    
    def test_register_capability(self):
        """Test registering a capability."""
        registry = CapabilityRegistry()
        
        name = registry.register(CapabilityA)
        
        assert name == "capability_a"
        assert "capability_a" in registry
        assert len(registry) == 1
    
    def test_register_with_config(self):
        """Test registering with config."""
        registry = CapabilityRegistry()
        config = CapabilityConfig(enabled=False)
        
        registry.register(CapabilityA, config=config)
        
        cap = registry.get("capability_a")
        assert cap is not None
    
    def test_unregister_capability(self):
        """Test unregistering a capability."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        
        result = registry.unregister("capability_a")
        
        assert result == True
        assert "capability_a" not in registry
        
        result = registry.unregister("nonexistent")
        assert result == False
    
    def test_get_capability(self):
        """Test getting a capability."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        
        cap = registry.get("capability_a")
        
        assert cap is not None
        assert cap.name == "capability_a"
        
        assert registry.get("nonexistent") is None
    
    def test_get_metadata(self):
        """Test getting capability metadata."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        
        meta = registry.get_metadata("capability_a")
        
        assert meta is not None
        assert meta.name == "capability_a"
        assert meta.priority == CapabilityPriority.HIGH
    
    def test_get_all_ready(self):
        """Test getting all ready capabilities."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        registry.register(CapabilityB)
        
        ready = registry.get_all_ready()
        
        assert len(ready) == 0  # Not initialized yet
        
        registry.capabilities["capability_a"]._state = CapabilityState.READY
        ready = registry.get_all_ready()
        
        assert len(ready) == 1
    
    def test_get_by_priority(self):
        """Test getting capabilities by priority."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        registry.register(CapabilityB)
        registry.register(CapabilityC)
        
        high_priority = registry.get_by_priority(CapabilityPriority.HIGH)
        
        assert len(high_priority) == 1
        assert high_priority[0].name == "capability_a"
    
    def test_resolve_dependencies(self):
        """Test resolving dependency order."""
        registry = CapabilityRegistry()
        registry.register(CapabilityC)
        registry.register(CapabilityA)
        registry.register(CapabilityB)
        
        order = registry.resolve_dependencies()
        
        assert order.index("capability_a") < order.index("capability_b")
        assert order.index("capability_b") < order.index("capability_c")
    
    def test_resolve_dependencies_circular(self):
        """Test detecting circular dependencies."""
        class CircularA(Capability):
            @classmethod
            def metadata(cls) -> CapabilityMetadata:
                return CapabilityMetadata(
                    name="circular_a",
                    description="Circular A",
                    dependencies={"circular_b"}
                )
            def initialize(self, container): pass
        
        class CircularB(Capability):
            @classmethod
            def metadata(cls) -> CapabilityMetadata:
                return CapabilityMetadata(
                    name="circular_b",
                    description="Circular B",
                    dependencies={"circular_a"}
                )
            def initialize(self, container): pass
        
        registry = CapabilityRegistry()
        registry.register(CircularA)
        registry.register(CircularB)
        
        with pytest.raises(ValueError, match="Circular dependency"):
            registry.resolve_dependencies()
    
    def test_load_all(self):
        """Test loading all capabilities."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        registry.register(CapabilityB)
        
        results = registry.load_all()
        
        assert registry.is_initialized == True
        assert results["capability_a"] == True
        assert results["capability_b"] == True
    
    def test_load_all_with_disabled(self):
        """Test loading with disabled capability."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA, CapabilityConfig(enabled=False))
        registry.register(CapabilityB)
        
        results = registry.load_all()
        
        assert results["capability_a"] == False
        assert results["capability_b"] == True
    
    def test_load_all_with_config(self):
        """Test loading with configuration."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        registry.register(CapabilityB)
        
        config = {
            "capability_a": {"enabled": False}
        }
        
        results = registry.load_all(config)
        
        assert results["capability_a"] == False
        assert results["capability_b"] == True
    
    def test_shutdown_all(self):
        """Test shutting down all capabilities."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        registry.register(CapabilityB)
        registry.load_all()
        
        registry.shutdown_all()
        
        assert registry.is_initialized == False
    
    def test_get_status(self):
        """Test getting registry status."""
        registry = CapabilityRegistry()
        registry.register(CapabilityA)
        registry.load_all()
        
        status = registry.get_status()
        
        assert status["initialized"] == True
        assert status["total_count"] == 1
        assert status["ready_count"] == 1
        assert "capabilities" in status
    
    def test_len_and_contains(self):
        """Test __len__ and __contains__."""
        registry = CapabilityRegistry()
        
        assert len(registry) == 0
        assert "capability_a" not in registry
        
        registry.register(CapabilityA)
        
        assert len(registry) == 1
        assert "capability_a" in registry
    
    def test_repr(self):
        """Test string representation."""
        registry = CapabilityRegistry()
        
        repr_str = repr(registry)
        
        assert "CapabilityRegistry" in repr_str
        assert "count=0" in repr_str
        assert "initialized=False" in repr_str
