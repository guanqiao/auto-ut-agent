"""Tests for DIContainer - Unified Dependency Injection Container."""

import pytest
from typing import Protocol

from pyutagent.core.container import (
    DIContainer,
    RegistrationType,
    register_singleton,
    register_factory,
    resolve,
    try_resolve,
)


# Test fixtures and protocols
class IMessageBus(Protocol):
    """Test message bus interface."""
    def send(self, message: str) -> bool:
        ...


class IStateManager(Protocol):
    """Test state manager interface."""
    def get_state(self) -> str:
        ...


class MockMessageBus:
    """Mock implementation of IMessageBus."""
    def __init__(self):
        self.messages = []
    
    def send(self, message: str) -> bool:
        self.messages.append(message)
        return True


class MockStateManager:
    """Mock implementation of IStateManager."""
    def __init__(self, initial_state: str = "idle"):
        self.state = initial_state
    
    def get_state(self) -> str:
        return self.state


class TestDIContainer:
    """Test cases for DIContainer."""
    
    def setup_method(self):
        """Clear container before each test."""
        DIContainer.clear()
    
    def teardown_method(self):
        """Clear container after each test."""
        DIContainer.clear()
    
    def test_register_singleton(self):
        """Test registering a singleton instance."""
        # Arrange
        bus = MockMessageBus()
        
        # Act
        DIContainer.register_singleton(IMessageBus, bus)
        
        # Assert
        resolved_bus = DIContainer.resolve(IMessageBus)
        assert resolved_bus is bus
        assert DIContainer.is_registered(IMessageBus)
    
    def test_singleton_returns_same_instance(self):
        """Test that singleton returns the same instance."""
        # Arrange
        bus = MockMessageBus()
        DIContainer.register_singleton(IMessageBus, bus)
        
        # Act
        instance1 = DIContainer.resolve(IMessageBus)
        instance2 = DIContainer.resolve(IMessageBus)
        
        # Assert
        assert instance1 is instance2
        assert instance1 is bus
    
    def test_register_factory(self):
        """Test registering a factory function."""
        # Arrange
        call_count = 0
        
        def create_state_manager():
            nonlocal call_count
            call_count += 1
            return MockStateManager()
        
        # Act
        DIContainer.register_factory(IStateManager, create_state_manager)
        
        # Assert
        instance = DIContainer.resolve(IStateManager)
        assert isinstance(instance, MockStateManager)
        assert call_count == 1
    
    def test_transient_factory_creates_new_instances(self):
        """Test that transient factory creates new instances."""
        # Arrange
        DIContainer.register_factory(
            IStateManager,
            lambda: MockStateManager(),
            RegistrationType.TRANSIENT
        )
        
        # Act
        instance1 = DIContainer.resolve(IStateManager)
        instance2 = DIContainer.resolve(IStateManager)
        
        # Assert
        assert instance1 is not instance2
        assert isinstance(instance1, MockStateManager)
        assert isinstance(instance2, MockStateManager)
    
    def test_register_implementation(self):
        """Test registering an implementation type."""
        # Act
        DIContainer.register_implementation(IMessageBus, MockMessageBus)
        
        # Assert
        instance = DIContainer.resolve(IMessageBus)
        assert isinstance(instance, MockMessageBus)
    
    def test_resolve_unregistered_raises_keyerror(self):
        """Test that resolving unregistered interface raises KeyError."""
        # Act & Assert
        with pytest.raises(KeyError) as exc_info:
            DIContainer.resolve(IMessageBus)
        
        assert "IMessageBus" in str(exc_info.value)
    
    def test_try_resolve_unregistered_returns_none(self):
        """Test that try_resolve returns None for unregistered interface."""
        # Act
        result = DIContainer.try_resolve(IMessageBus)
        
        # Assert
        assert result is None
    
    def test_try_resolve_registered_returns_instance(self):
        """Test that try_resolve returns instance for registered interface."""
        # Arrange
        bus = MockMessageBus()
        DIContainer.register_singleton(IMessageBus, bus)
        
        # Act
        result = DIContainer.try_resolve(IMessageBus)
        
        # Assert
        assert result is bus
    
    def test_is_registered(self):
        """Test checking if interface is registered."""
        # Arrange
        bus = MockMessageBus()
        
        # Act & Assert
        assert not DIContainer.is_registered(IMessageBus)
        
        DIContainer.register_singleton(IMessageBus, bus)
        assert DIContainer.is_registered(IMessageBus)
    
    def test_unregister(self):
        """Test unregistering an interface."""
        # Arrange
        bus = MockMessageBus()
        DIContainer.register_singleton(IMessageBus, bus)
        
        # Act
        result = DIContainer.unregister(IMessageBus)
        
        # Assert
        assert result is True
        assert not DIContainer.is_registered(IMessageBus)
    
    def test_unregister_not_registered_returns_false(self):
        """Test unregistering an interface that is not registered."""
        # Act
        result = DIContainer.unregister(IMessageBus)
        
        # Assert
        assert result is False
    
    def test_clear(self):
        """Test clearing all registrations."""
        # Arrange
        DIContainer.register_singleton(IMessageBus, MockMessageBus())
        DIContainer.register_singleton(IStateManager, MockStateManager())
        
        # Act
        DIContainer.clear()
        
        # Assert
        assert not DIContainer.is_registered(IMessageBus)
        assert not DIContainer.is_registered(IStateManager)
        assert len(DIContainer.get_registered_interfaces()) == 0
    
    def test_get_registered_interfaces(self):
        """Test getting list of registered interfaces."""
        # Arrange
        DIContainer.register_singleton(IMessageBus, MockMessageBus())
        DIContainer.register_singleton(IStateManager, MockStateManager())
        
        # Act
        interfaces = DIContainer.get_registered_interfaces()
        
        # Assert
        assert len(interfaces) == 2
        assert IMessageBus in interfaces
        assert IStateManager in interfaces
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Arrange
        bus = MockMessageBus()
        
        # Act
        register_singleton(IMessageBus, bus)
        resolved = resolve(IMessageBus)
        
        # Assert
        assert resolved is bus
    
    def test_convenience_try_resolve(self):
        """Test convenience try_resolve function."""
        # Act
        result = try_resolve(IMessageBus)
        
        # Assert
        assert result is None
        
        # Arrange
        bus = MockMessageBus()
        register_singleton(IMessageBus, bus)
        
        # Act
        result = try_resolve(IMessageBus)
        
        # Assert
        assert result is bus
    
    def test_factory_with_dependencies(self):
        """Test factory that creates instances with dependencies."""
        # Arrange
        bus = MockMessageBus()
        DIContainer.register_singleton(IMessageBus, bus)
        
        def create_service():
            message_bus = DIContainer.resolve(IMessageBus)
            return MockService(message_bus)
        
        # Act
        DIContainer.register_factory(MockService, create_service)
        service = DIContainer.resolve(MockService)
        
        # Assert
        assert service.message_bus is bus
    
    def test_multiple_registrations(self):
        """Test multiple registrations and resolutions."""
        # Arrange
        DIContainer.register_singleton(IMessageBus, MockMessageBus())
        DIContainer.register_singleton(IStateManager, MockStateManager())
        
        # Act
        bus = DIContainer.resolve(IMessageBus)
        state = DIContainer.resolve(IStateManager)
        
        # Assert
        assert isinstance(bus, MockMessageBus)
        assert isinstance(state, MockStateManager)


class MockService:
    """Mock service with dependencies."""
    def __init__(self, message_bus: IMessageBus):
        self.message_bus = message_bus


class TestRegistrationType:
    """Test cases for RegistrationType enum."""
    
    def test_registration_type_values(self):
        """Test RegistrationType enum values."""
        assert RegistrationType.SINGLETON.name == "SINGLETON"
        assert RegistrationType.TRANSIENT.name == "TRANSIENT"
        assert RegistrationType.SCOPED.name == "SCOPED"


class TestContainerEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Clear container before each test."""
        DIContainer.clear()
    
    def teardown_method(self):
        """Clear container after each test."""
        DIContainer.clear()
    
    def test_factory_raises_exception(self):
        """Test factory that raises exception."""
        # Arrange
        def failing_factory():
            raise ValueError("Factory failed")
        
        DIContainer.register_factory(IMessageBus, failing_factory)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Factory failed"):
            DIContainer.resolve(IMessageBus)
    
    def test_implementation_without_default_constructor(self):
        """Test implementation type without default constructor."""
        # Arrange
        class ComplexService:
            def __init__(self, required_param: str):
                self.param = required_param
        
        DIContainer.register_implementation(ComplexService, ComplexService)
        
        # Act & Assert
        with pytest.raises(TypeError):
            DIContainer.resolve(ComplexService)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
