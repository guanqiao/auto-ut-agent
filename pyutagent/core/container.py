"""Dependency injection container for PyUT Agent.

This module provides a simple dependency injection container that manages
the lifecycle of components and their dependencies.

Features:
- Singleton and transient lifecycle management
- Lazy initialization
- Factory-based component creation
- Configuration injection
- Easy testing with mock injection
- Thread-safe singleton initialization
"""

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Lifecycle(Enum):
    """Component lifecycle types."""
    SINGLETON = auto()
    TRANSIENT = auto()


@dataclass
class ComponentRegistration(Generic[T]):
    """Registration info for a component."""
    component_type: Type[T]
    factory: Optional[Callable[[], T]] = None
    instance: Optional[T] = None
    lifecycle: Lifecycle = Lifecycle.SINGLETON
    dependencies: Dict[str, str] = field(default_factory=dict)


class Container:
    """Dependency injection container.

    Manages component registration, lifecycle, and dependency resolution.

    Example:
        container = Container()

        # Register with factory
        container.register_factory(LLMClient, create_llm_client)

        # Register singleton instance
        container.register_instance(Settings, settings)

        # Resolve dependencies
        client = container.resolve(LLMClient)
    """

    def __init__(self):
        self._registrations: Dict[Type, ComponentRegistration] = {}
        self._named_instances: Dict[str, Any] = {}
        self._resolving: set = set()

    def register_singleton(
        self,
        component_type: Type[T],
        factory: Optional[Callable[[], T]] = None
    ) -> None:
        """Register a singleton component.

        Args:
            component_type: The type to register
            factory: Optional factory function (default: component_type())
        """
        self._registrations[component_type] = ComponentRegistration(
            component_type=component_type,
            factory=factory,
            lifecycle=Lifecycle.SINGLETON
        )
        logger.debug(f"[Container] Registered singleton: {component_type.__name__}")

    def register_transient(
        self,
        component_type: Type[T],
        factory: Optional[Callable[[], T]] = None
    ) -> None:
        """Register a transient component (new instance each time).

        Args:
            component_type: The type to register
            factory: Optional factory function
        """
        self._registrations[component_type] = ComponentRegistration(
            component_type=component_type,
            factory=factory,
            lifecycle=Lifecycle.TRANSIENT
        )
        logger.debug(f"[Container] Registered transient: {component_type.__name__}")

    def register_factory(
        self,
        component_type: Type[T],
        factory: Callable[[], T],
        lifecycle: Lifecycle = Lifecycle.SINGLETON
    ) -> None:
        """Register a component with a factory function.

        Args:
            component_type: The type to register
            factory: Factory function to create instances
            lifecycle: Component lifecycle
        """
        self._registrations[component_type] = ComponentRegistration(
            component_type=component_type,
            factory=factory,
            lifecycle=lifecycle
        )
        logger.debug(f"[Container] Registered factory for {component_type.__name__} with {lifecycle.name} lifecycle")

    def register_instance(self, component_type: Type[T], instance: T) -> None:
        """Register an existing instance as a singleton.

        Args:
            component_type: The type to register
            instance: The instance to register
        """
        self._registrations[component_type] = ComponentRegistration(
            component_type=component_type,
            instance=instance,
            lifecycle=Lifecycle.SINGLETON
        )
        logger.debug(f"[Container] Registered instance of {component_type.__name__}")

    def register_named(self, name: str, instance: Any) -> None:
        """Register a named instance.

        Args:
            name: Name for the instance
            instance: The instance to register
        """
        self._named_instances[name] = instance
        logger.debug(f"[Container] Registered named instance: {name}")

    def resolve(self, component_type: Type[T]) -> T:
        """Resolve a component by type.

        Args:
            component_type: The type to resolve

        Returns:
            The resolved instance

        Raises:
            KeyError: If component is not registered
            RuntimeError: If circular dependency is detected
        """
        if component_type not in self._registrations:
            raise KeyError(f"Component {component_type.__name__} is not registered")

        registration = self._registrations[component_type]

        if registration.lifecycle == Lifecycle.SINGLETON:
            if registration.instance is not None:
                return registration.instance

            if component_type in self._resolving:
                raise RuntimeError(f"Circular dependency detected for {component_type.__name__}")

            self._resolving.add(component_type)
            try:
                instance = self._create_instance(registration)
                registration.instance = instance
                return instance
            finally:
                self._resolving.discard(component_type)

        elif registration.lifecycle == Lifecycle.TRANSIENT:
            return self._create_instance(registration)

        raise RuntimeError(f"Unknown lifecycle: {registration.lifecycle}")

    def resolve_named(self, name: str) -> Any:
        """Resolve a named instance.

        Args:
            name: Name of the instance

        Returns:
            The resolved instance

        Raises:
            KeyError: If named instance is not registered
        """
        if name not in self._named_instances:
            raise KeyError(f"Named instance '{name}' is not registered")
        return self._named_instances[name]

    def try_resolve(self, component_type: Type[T]) -> Optional[T]:
        """Try to resolve a component, returning None if not registered.

        Args:
            component_type: The type to resolve

        Returns:
            The resolved instance or None
        """
        try:
            return self.resolve(component_type)
        except KeyError:
            return None

    def _create_instance(self, registration: ComponentRegistration[T]) -> T:
        """Create an instance using the registration info.

        Args:
            registration: The component registration

        Returns:
            The created instance
        """
        if registration.factory:
            return registration.factory()

        component_type = registration.component_type

        try:
            import inspect
            sig = inspect.signature(component_type.__init__)
            params = {}

            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                if param_name in registration.dependencies:
                    dep_name = registration.dependencies[param_name]
                    params[param_name] = self.resolve_named(dep_name)
                elif param.annotation != inspect.Parameter.empty:
                    if isinstance(param.annotation, type):
                        try:
                            params[param_name] = self.resolve(param.annotation)
                        except KeyError:
                            if param.default != inspect.Parameter.empty:
                                continue
                            raise

            return component_type(**params)

        except Exception as e:
            logger.warning(f"[Container] Failed to resolve dependencies for {component_type.__name__}: {e}, using default constructor")
            return component_type()

    def is_registered(self, component_type: Type) -> bool:
        """Check if a component type is registered.

        Args:
            component_type: The type to check

        Returns:
            True if registered
        """
        return component_type in self._registrations

    def is_named_registered(self, name: str) -> bool:
        """Check if a named instance is registered.

        Args:
            name: Name to check

        Returns:
            True if registered
        """
        return name in self._named_instances

    def clear(self) -> None:
        """Clear all registrations."""
        self._registrations.clear()
        self._named_instances.clear()
        self._resolving.clear()
        logger.info("[Container] Cleared all registrations")

    def get_registrations(self) -> Dict[Type, ComponentRegistration]:
        """Get all registrations (for debugging)."""
        return self._registrations.copy()

    def get_named_instances(self) -> Dict[str, Any]:
        """Get all named instances (for debugging)."""
        return self._named_instances.copy()


_global_container: Optional[Container] = None
_container_lock = threading.Lock()


def get_container() -> Container:
    """Get the global container instance (thread-safe).

    Returns:
        The global Container instance
    """
    global _global_container
    if _global_container is None:
        with _container_lock:
            # Double-checked locking pattern
            if _global_container is None:
                _global_container = Container()
                logger.info("[Container] Created global container")
    return _global_container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _global_container
    with _container_lock:
        if _global_container is not None:
            _global_container.clear()
        _global_container = None
        logger.info("[Container] Reset global container")


def configure_container(
    settings: Optional[Any] = None,
    llm_config_collection: Optional[Any] = None,
    aider_config: Optional[Any] = None
) -> Container:
    """Configure the global container with application components.

    Args:
        settings: Application settings
        llm_config_collection: LLM configuration collection
        aider_config: Aider configuration

    Returns:
        The configured container
    """
    container = get_container()

    if settings is not None:
        from .config import Settings
        container.register_instance(Settings, settings)

    if llm_config_collection is not None:
        from .config import LLMConfigCollection
        container.register_instance(LLMConfigCollection, llm_config_collection)

    if aider_config is not None:
        from .config import AiderConfig
        container.register_instance(AiderConfig, aider_config)

    from .retry_manager import RetryManager, RetryConfig
    container.register_singleton(RetryManager, lambda: RetryManager())

    from .error_recovery import RecoveryManager
    container.register_singleton(RecoveryManager, lambda: RecoveryManager())

    logger.info("[Container] Configured with application components")
    return container
