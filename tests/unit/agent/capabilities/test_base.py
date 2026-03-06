"""Tests for Capability Base Classes."""

import pytest
from unittest.mock import Mock, MagicMock

from pyutagent.agent.capabilities.base import (
    Capability,
    CapabilityMetadata,
    CapabilityPriority,
    CapabilityState,
)


class TestCapabilityMetadata:
    """Tests for CapabilityMetadata."""
    
    def test_metadata_creation(self):
        """Test creating metadata."""
        meta = CapabilityMetadata(
            name="test_capability",
            description="Test capability description",
            priority=CapabilityPriority.HIGH,
            dependencies={"dep1", "dep2"},
            provides={"provider1"}
        )
        
        assert meta.name == "test_capability"
        assert meta.description == "Test capability description"
        assert meta.priority == CapabilityPriority.HIGH
        assert meta.dependencies == {"dep1", "dep2"}
        assert meta.provides == {"provider1"}
    
    def test_metadata_hash(self):
        """Test metadata hash based on name."""
        meta1 = CapabilityMetadata(name="test", description="desc1")
        meta2 = CapabilityMetadata(name="test", description="desc2")
        meta3 = CapabilityMetadata(name="other", description="desc1")
        
        assert hash(meta1) == hash(meta2)
        assert hash(meta1) != hash(meta3)
    
    def test_metadata_equality(self):
        """Test metadata equality."""
        meta1 = CapabilityMetadata(name="test", description="desc1")
        meta2 = CapabilityMetadata(name="test", description="desc2")
        meta3 = CapabilityMetadata(name="other", description="desc1")
        
        assert meta1 == meta2
        assert meta1 != meta3
        assert meta1 != "not a metadata"


class MockCapability(Capability):
    """Mock capability for testing."""
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="mock_capability",
            description="Mock capability for testing",
            priority=CapabilityPriority.NORMAL
        )
    
    def initialize(self, container):
        self._state = CapabilityState.READY


class TestCapability:
    """Tests for Capability base class."""
    
    def test_capability_creation(self):
        """Test creating a capability."""
        cap = MockCapability()
        
        assert cap.name == "mock_capability"
        assert cap.state == CapabilityState.UNINITIALIZED
        assert cap.is_ready == False
    
    def test_capability_initialize(self):
        """Test initializing a capability."""
        cap = MockCapability()
        container = Mock()
        
        cap.initialize(container)
        
        assert cap.state == CapabilityState.READY
        assert cap.is_ready == True
    
    def test_capability_shutdown(self):
        """Test shutting down a capability."""
        cap = MockCapability()
        container = Mock()
        cap.initialize(container)
        
        cap.shutdown()
        
        assert cap.state == CapabilityState.UNINITIALIZED
    
    def test_capability_enable_disable(self):
        """Test enabling and disabling a capability."""
        cap = MockCapability()
        
        cap.disable()
        assert cap.state == CapabilityState.DISABLED
        
        cap.enable()
        assert cap.state == CapabilityState.UNINITIALIZED
    
    def test_capability_error_handling(self):
        """Test error handling in capability."""
        cap = MockCapability()
        error = Exception("Test error")
        
        cap._set_error(error)
        
        assert cap.state == CapabilityState.ERROR
        assert cap.error == error
    
    def test_capability_resolve_without_container(self):
        """Test resolving dependency without container raises error."""
        cap = MockCapability()
        
        with pytest.raises(RuntimeError, match="Container not set"):
            cap.resolve(str)
    
    def test_capability_repr(self):
        """Test capability string representation."""
        cap = MockCapability()
        
        repr_str = repr(cap)
        
        assert "MockCapability" in repr_str
        assert "mock_capability" in repr_str
        assert "UNINITIALIZED" in repr_str
