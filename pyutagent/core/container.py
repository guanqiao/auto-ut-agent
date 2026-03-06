"""Unified Dependency Injection Container.

This module provides a centralized dependency injection system to replace
dispersed Manager class instances and improve testability and maintainability.

Example:
    >>> from pyutagent.core.container import DIContainer
    >>> 
    >>> # Register a singleton
    >>> DIContainer.register_singleton(MessageBus, UnifiedMessageBus())
    >>> 
    >>> # Register a factory
    >>> DIContainer.register_factory(StateManager, lambda: StateManager())
    >>> 
    >>> # Resolve dependencies
    >>> bus = DIContainer.resolve(MessageBus)
    >>> state = DIContainer.resolve(StateManager)
"""

import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RegistrationType(Enum):
    """Type of dependency registration."""
    SINGLETON = auto()  # Single instance shared across all resolutions
    TRANSIENT = auto()  # New instance created for each resolution
    SCOPED = auto()     # Instance shared within a scope


class Registration:
    """Dependency registration entry."""
    
    def __init__(
        self,
        interface: Type,
        implementation: Optional[Type] = None,
        factory: Optional[Callable[..., Any]] = None,
        instance: Optional[Any] = None,
        registration_type: RegistrationType = RegistrationType.SINGLETON
    ):
        self.interface = interface
        self.implementation = implementation
        self.factory = factory
        self.instance = instance
        self.registration_type = registration_type
        # Track if instance was explicitly set (even to None)
        self._instance_set = 'instance' in locals() and instance is not None
    
    def get_instance(self) -> Any:
        """Get or create instance based on registration type."""
        if self.registration_type == RegistrationType.SINGLETON:
            # If instance was explicitly set (even to None), return it
            if hasattr(self, '_instance_set') and self._instance_set:
                return self.instance
            # Otherwise, create and cache the instance
            if self.instance is None:
                self.instance = self._create_instance()
                self._instance_set = True
            return self.instance
        elif self.registration_type == RegistrationType.TRANSIENT:
            return self._create_instance()
        else:  # SCOPED
            # For scoped, we would need scope context
            # For now, treat as transient
            return self._create_instance()
    
    def _create_instance(self) -> Any:
        """Create a new instance."""
        if self.factory:
            return self.factory()
        elif self.implementation:
            return self.implementation()
        else:
            raise ValueError(f"No factory or implementation for {self.interface}")


class DIContainer:
    """Unified Dependency Injection Container.
    
    Provides centralized dependency management with support for:
    - Singleton registration
    - Factory registration
    - Transient and scoped lifetimes
    - Interface-to-implementation mapping
    
    This replaces the dispersed Manager class instances throughout the codebase.
    """
    
    _registrations: Dict[Type, Registration] = {}
    _instance_cache: Dict[Type, Any] = {}
    
    @classmethod
    def register_singleton(
        cls,
        interface: Type[T],
        instance: T
    ) -> None:
        """Register a singleton instance.
        
        Args:
            interface: Interface type (usually abstract base class or protocol)
            instance: Singleton instance to register
            
        Example:
            >>> DIContainer.register_singleton(MessageBus, UnifiedMessageBus())
        """
        cls._registrations[interface] = Registration(
            interface=interface,
            instance=instance,
            registration_type=RegistrationType.SINGLETON
        )
        logger.debug(f"Registered singleton for {interface.__name__}")
    
    @classmethod
    def register_factory(
        cls,
        interface: Type[T],
        factory: Callable[..., T],
        registration_type: RegistrationType = RegistrationType.TRANSIENT
    ) -> None:
        """Register a factory function.
        
        Args:
            interface: Interface type
            factory: Factory function to create instances
            registration_type: Lifetime of created instances
            
        Example:
            >>> DIContainer.register_factory(StateManager, lambda: StateManager())
        """
        cls._registrations[interface] = Registration(
            interface=interface,
            factory=factory,
            registration_type=registration_type
        )
        logger.debug(f"Registered factory for {interface.__name__}")
    
    @classmethod
    def register_implementation(
        cls,
        interface: Type[T],
        implementation: Type[T],
        registration_type: RegistrationType = RegistrationType.SINGLETON
    ) -> None:
        """Register an implementation type for an interface.
        
        Args:
            interface: Interface type
            implementation: Concrete implementation type
            registration_type: Lifetime of created instances
            
        Example:
            >>> DIContainer.register_implementation(MessageBus, UnifiedMessageBus)
        """
        cls._registrations[interface] = Registration(
            interface=interface,
            implementation=implementation,
            registration_type=registration_type
        )
        logger.debug(f"Registered implementation {implementation.__name__} for {interface.__name__}")
    
    @classmethod
    def resolve(cls, interface: Type[T]) -> T:
        """Resolve a dependency.
        
        Args:
            interface: Interface type to resolve
            
        Returns:
            Instance of the registered implementation
            
        Raises:
            KeyError: If interface is not registered
            
        Example:
            >>> bus = DIContainer.resolve(MessageBus)
        """
        if interface not in cls._registrations:
            raise KeyError(f"No registration found for {interface.__name__}")
        
        registration = cls._registrations[interface]
        instance = registration.get_instance()
        return cast(T, instance)
    
    @classmethod
    def try_resolve(cls, interface: Type[T]) -> Optional[T]:
        """Try to resolve a dependency, returning None if not registered.
        
        Args:
            interface: Interface type to resolve
            
        Returns:
            Instance or None if not registered
        """
        try:
            return cls.resolve(interface)
        except KeyError:
            return None
    
    @classmethod
    def is_registered(cls, interface: Type) -> bool:
        """Check if an interface is registered.
        
        Args:
            interface: Interface type to check
            
        Returns:
            True if registered, False otherwise
        """
        return interface in cls._registrations
    
    @classmethod
    def unregister(cls, interface: Type) -> bool:
        """Unregister an interface.
        
        Args:
            interface: Interface type to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if interface in cls._registrations:
            del cls._registrations[interface]
            cls._instance_cache.pop(interface, None)
            logger.debug(f"Unregistered {interface.__name__}")
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Useful for testing."""
        cls._registrations.clear()
        cls._instance_cache.clear()
        logger.debug("Cleared all registrations")
    
    @classmethod
    def get_registered_interfaces(cls) -> list[Type]:
        """Get list of all registered interfaces.
        
        Returns:
            List of registered interface types
        """
        return list(cls._registrations.keys())


# Convenience functions for common patterns

def register_singleton(interface: Type[T], instance: T) -> None:
    """Convenience function to register a singleton."""
    DIContainer.register_singleton(interface, instance)


def register_factory(interface: Type[T], factory: Callable[..., T]) -> None:
    """Convenience function to register a factory."""
    DIContainer.register_factory(interface, factory)


def resolve(interface: Type[T]) -> T:
    """Convenience function to resolve a dependency."""
    return DIContainer.resolve(interface)


def try_resolve(interface: Type[T]) -> Optional[T]:
    """Convenience function to try resolving a dependency."""
    return DIContainer.try_resolve(interface)


# Backward compatibility aliases
# These are provided for compatibility with existing code
Container = DIContainer


def get_container() -> DIContainer:
    """Get the global DIContainer instance.
    
    This is a convenience function for backward compatibility.
    
    Returns:
        DIContainer instance
    """
    return DIContainer


def configure_container() -> None:
    """Configure the container with default registrations.
    
    This function registers the default implementations for common interfaces.
    Call this at application startup.
    """
    from pyutagent.core.messaging import UnifiedMessageBus
    
    # Register default implementations if not already registered
    if not DIContainer.is_registered(UnifiedMessageBus):
        DIContainer.register_singleton(UnifiedMessageBus, UnifiedMessageBus())
    
    logger.debug("Container configured with default registrations")
