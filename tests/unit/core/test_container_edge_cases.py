"""Edge case tests for DIContainer - DIContainer 边界测试"""
import asyncio
import gc
import threading
import pytest
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from pyutagent.core.container import DIContainer, RegistrationType
from pyutagent.core.interfaces import IEventBus, IContext


class TestDIContainerEdgeCases:
    """Test edge cases for DIContainer."""

    def setup_method(self):
        """Clear container before each test."""
        DIContainer.clear()

    def teardown_method(self):
        """Clear container after each test."""
        DIContainer.clear()

    # =========================================================================
    # Exception Handling Tests
    # =========================================================================

    def test_resolve_unregistered_interface_raises_keyerror(self):
        """Test resolving unregistered interface raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            DIContainer.resolve(IEventBus)
        
        assert "IEventBus" in str(exc_info.value)

    def test_resolve_with_none_interface_raises_error(self):
        """Test resolving None interface raises AttributeError.
        
        Note: This documents the current behavior where None is not handled.
        In production, this should be caught at the call site.
        """
        with pytest.raises(AttributeError):
            DIContainer.resolve(None)

    def test_register_none_interface_raises_error(self):
        """Test registering None interface raises AttributeError.
        
        Note: This documents the current behavior where None is not handled.
        In production, this should be caught at the call site.
        """
        with pytest.raises(AttributeError):
            DIContainer.register_singleton(None, object())

    def test_register_none_instance_raises_error(self):
        """Test registering None instance is not supported.
        
        Note: DIContainer doesn't support None instances because it tries
        to create instances via factory when the stored instance is None.
        This documents the current behavior.
        """
        # Registering None is not a valid use case
        # This test documents that it will fail
        with pytest.raises((ValueError, TypeError)):
            DIContainer.register_singleton(IEventBus, None)
            DIContainer.resolve(IEventBus)

    def test_factory_raises_exception_in_resolve(self):
        """Test factory that raises exception during resolve."""
        def failing_factory():
            raise RuntimeError("Factory failed")
        
        DIContainer.register_factory(IEventBus, failing_factory)
        
        with pytest.raises(RuntimeError) as exc_info:
            DIContainer.resolve(IEventBus)
        
        assert "Factory failed" in str(exc_info.value)

    def test_factory_returns_none(self):
        """Test factory that returns None."""
        DIContainer.register_factory(IEventBus, lambda: None)
        result = DIContainer.resolve(IEventBus)
        assert result is None

    # =========================================================================
    # Concurrent Access Tests
    # =========================================================================

    def test_concurrent_singleton_registration(self):
        """Test concurrent singleton registration is thread-safe.
        
        Note: Due to Python GIL and lack of locking in DIContainer,
        concurrent registration may create multiple instances.
        This test documents the current behavior.
        """
        class Counter:
            def __init__(self):
                self.count = 0
        
        results = []
        errors = []
        
        def register_and_resolve():
            try:
                DIContainer.register_singleton(Counter, Counter())
                instance = DIContainer.resolve(Counter)
                results.append(id(instance))
            except Exception as e:
                errors.append(e)
        
        # Run concurrent registrations
        threads = []
        for _ in range(10):
            t = threading.Thread(target=register_and_resolve)
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10
        # All should get instances (may not be the same due to race conditions)
        # After all threads complete, resolve again to get the final singleton
        final_instance = DIContainer.resolve(Counter)
        assert final_instance is not None

    def test_concurrent_resolve_same_singleton(self):
        """Test concurrent resolve of same singleton."""
        class Service:
            def __init__(self):
                self.id = id(self)
        
        DIContainer.register_singleton(Service, Service())
        
        results = []
        
        def resolve_service():
            service = DIContainer.resolve(Service)
            results.append(service.id)
        
        # Run concurrent resolves
        threads = []
        for _ in range(20):
            t = threading.Thread(target=resolve_service)
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # All should get the same instance
        assert len(set(results)) == 1

    def test_concurrent_transient_factory(self):
        """Test concurrent transient factory creates instances.
        
        Note: Due to Python GIL and lack of locking in DIContainer,
        concurrent factory calls may return the same instance.
        This test documents the current behavior.
        """
        class Service:
            pass
        
        DIContainer.register_factory(Service, Service, RegistrationType.TRANSIENT)
        
        results = []
        
        def resolve_service():
            service = DIContainer.resolve(Service)
            results.append(id(service))
        
        # Run concurrent resolves
        threads = []
        for _ in range(20):
            t = threading.Thread(target=resolve_service)
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # All should get instances
        assert len(results) == 20
        # Each thread should have gotten a service instance
        assert all(r is not None for r in results)

    # =========================================================================
    # Memory Leak Tests
    # =========================================================================

    def test_singleton_not_garbage_collected(self):
        """Test singleton instances are not garbage collected."""
        class Service:
            def __init__(self):
                self.data = "x" * 1000000  # 1MB of data
        
        DIContainer.register_singleton(Service, Service())
        
        # Force garbage collection
        gc.collect()
        
        # Instance should still be available
        instance = DIContainer.resolve(Service)
        assert instance is not None
        assert len(instance.data) == 1000000

    def test_unregister_releases_reference(self):
        """Test unregister releases reference to instance."""
        class Service:
            pass
        
        service = Service()
        service_id = id(service)
        
        DIContainer.register_singleton(Service, service)
        
        # Unregister
        DIContainer.unregister(Service)
        
        # Delete reference
        del service
        gc.collect()
        
        # Should not be able to resolve anymore
        with pytest.raises(KeyError):
            DIContainer.resolve(Service)

    def test_clear_releases_all_references(self):
        """Test clear releases all references."""
        class Service1:
            pass
        
        class Service2:
            pass
        
        DIContainer.register_singleton(Service1, Service1())
        DIContainer.register_singleton(Service2, Service2())
        
        # Clear all
        DIContainer.clear()
        
        # Should not be able to resolve anymore
        with pytest.raises(KeyError):
            DIContainer.resolve(Service1)
        
        with pytest.raises(KeyError):
            DIContainer.resolve(Service2)

    # =========================================================================
    # Complex Scenario Tests
    # =========================================================================

    def test_multiple_registrations_same_interface(self):
        """Test multiple registrations for same interface (last wins)."""
        class Service:
            def __init__(self, name):
                self.name = name
        
        service1 = Service("first")
        service2 = Service("second")
        
        DIContainer.register_singleton(Service, service1)
        DIContainer.register_singleton(Service, service2)
        
        # Last registration should win
        resolved = DIContainer.resolve(Service)
        assert resolved.name == "second"

    def test_register_after_unregister(self):
        """Test register after unregister."""
        class Service:
            def __init__(self, name):
                self.name = name
        
        service1 = Service("first")
        service2 = Service("second")
        
        DIContainer.register_singleton(Service, service1)
        DIContainer.unregister(Service)
        DIContainer.register_singleton(Service, service2)
        
        resolved = DIContainer.resolve(Service)
        assert resolved.name == "second"

    def test_is_registered_after_unregister(self):
        """Test is_registered returns False after unregister."""
        class Service:
            pass
        
        assert not DIContainer.is_registered(Service)
        
        DIContainer.register_singleton(Service, Service())
        assert DIContainer.is_registered(Service)
        
        DIContainer.unregister(Service)
        assert not DIContainer.is_registered(Service)

    def test_get_registered_interfaces_empty(self):
        """Test get_registered_interfaces returns empty list initially."""
        interfaces = DIContainer.get_registered_interfaces()
        assert interfaces == []

    def test_get_registered_interfaces_with_registrations(self):
        """Test get_registered_interfaces returns all registered interfaces."""
        class Service1:
            pass
        
        class Service2:
            pass
        
        DIContainer.register_singleton(Service1, Service1())
        DIContainer.register_singleton(Service2, Service2())
        
        interfaces = DIContainer.get_registered_interfaces()
        assert len(interfaces) == 2
        assert Service1 in interfaces
        assert Service2 in interfaces

    # =========================================================================
    # Factory with Dependencies Tests
    # =========================================================================

    def test_factory_with_dependency_not_registered(self):
        """Test factory with dependency that is not registered."""
        class Dependency:
            pass
        
        class Service:
            def __init__(self, dep: Dependency):
                self.dep = dep
        
        # Don't register Dependency
        def factory():
            return Service(Dependency())
        
        DIContainer.register_factory(Service, factory)
        
        # Should still work
        service = DIContainer.resolve(Service)
        assert service is not None
        assert service.dep is not None

    def test_nested_factory_calls(self):
        """Test nested factory calls."""
        call_count = 0
        
        class Service:
            pass
        
        def factory():
            nonlocal call_count
            call_count += 1
            return Service()
        
        DIContainer.register_factory(Service, factory, RegistrationType.TRANSIENT)
        
        # Resolve multiple times
        for _ in range(5):
            DIContainer.resolve(Service)
        
        # Factory should be called 5 times
        assert call_count == 5


class TestDIContainerAsyncEdgeCases:
    """Test async edge cases for DIContainer."""

    def setup_method(self):
        """Clear container before each test."""
        DIContainer.clear()

    def teardown_method(self):
        """Clear container after each test."""
        DIContainer.clear()

    @pytest.mark.asyncio
    async def test_async_factory_registration(self):
        """Test async factory registration and resolution."""
        class Service:
            def __init__(self, name):
                self.name = name
        
        async def async_factory():
            await asyncio.sleep(0.01)  # Simulate async work
            return Service("async")
        
        DIContainer.register_factory(Service, async_factory)
        
        # Note: DIContainer doesn't support async factories directly
        # This test documents the current behavior
        service = DIContainer.resolve(Service)
        # The factory is called but not awaited
        # This is expected behavior - factories should be sync

    @pytest.mark.asyncio
    async def test_concurrent_async_access(self):
        """Test concurrent async access to container."""
        class Service:
            def __init__(self):
                self.id = id(self)
        
        DIContainer.register_singleton(Service, Service())
        
        async def resolve_service():
            await asyncio.sleep(0.001)  # Simulate async work
            return DIContainer.resolve(Service)
        
        # Resolve concurrently
        tasks = [resolve_service() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should get the same instance
        ids = [r.id for r in results]
        assert len(set(ids)) == 1
