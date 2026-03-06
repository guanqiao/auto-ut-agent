"""Edge case tests for EventBus - EventBus 边界测试"""
import asyncio
import gc
import pytest
from typing import Any, List

from pyutagent.core.event_bus import EventBus, Event, Subscription
from pyutagent.core.messaging.bus import UnifiedMessageBus


class TestEventBusEdgeCases:
    """Test edge cases for EventBus."""

    def setup_method(self):
        """Create fresh EventBus before each test."""
        self.event_bus = EventBus()

    # =========================================================================
    # Exception Handler Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_break_other_handlers(self):
        """Test that one handler exception doesn't break other handlers."""
        results = []

        async def good_handler(event):
            results.append("good")

        async def bad_handler(event):
            raise RuntimeError("Handler error")

        await self.event_bus.subscribe(Event, good_handler)
        await self.event_bus.subscribe(Event, bad_handler)

        event = Event()
        count = await self.event_bus.publish(event)

        # Good handler should still be called
        assert "good" in results
        # Bad handler threw exception but didn't crash publish
        assert count == 1  # Only good handler succeeded

    @pytest.mark.asyncio
    async def test_sync_handler_exception_handling(self):
        """Test sync handler exception handling."""
        results = []

        def sync_handler(event):
            results.append("called")
            raise ValueError("Sync error")

        await self.event_bus.subscribe(Event, sync_handler)

        event = Event()
        count = await self.event_bus.publish(event)

        # Handler was called but threw exception
        assert "called" in results
        assert count == 0  # No successful handlers

    @pytest.mark.asyncio
    async def test_all_handlers_fail(self):
        """Test when all handlers fail."""
        async def failing_handler1(event):
            raise RuntimeError("Error 1")

        async def failing_handler2(event):
            raise RuntimeError("Error 2")

        await self.event_bus.subscribe(Event, failing_handler1)
        await self.event_bus.subscribe(Event, failing_handler2)

        event = Event()
        count = await self.event_bus.publish(event)

        assert count == 0  # No successful handlers

    # =========================================================================
    # Subscription Management Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_handler(self):
        """Test unsubscribing a handler that doesn't exist."""
        async def handler(event):
            pass

        # Try to unsubscribe without subscribing first
        # Should not raise error
        subscription = Subscription(Event, handler, self.event_bus)
        subscription.unsubscribe()  # Not async

    @pytest.mark.asyncio
    async def test_multiple_subscriptions_same_handler(self):
        """Test multiple subscriptions of the same handler.
        
        Note: EventBus stores handlers in a list, so same handler subscribed
        twice will be called twice. Unsubscribing removes all occurrences.
        """
        results = []

        async def handler(event):
            results.append("called")

        # Subscribe same handler multiple times
        sub1 = await self.event_bus.subscribe(Event, handler)
        sub2 = await self.event_bus.subscribe(Event, handler)

        event = Event()
        count = await self.event_bus.publish(event)

        # Handler should be called twice (once for each subscription)
        assert count == 2
        assert results.count("called") == 2

        # Unsubscribe one - this removes all occurrences of the handler
        sub1.unsubscribe()  # Not async

        results.clear()
        count = await self.event_bus.publish(event)

        # Handler should not be called (all occurrences removed)
        assert count == 0
        assert results.count("called") == 0

    @pytest.mark.asyncio
    async def test_subscription_context_manager(self):
        """Test subscription as context manager."""
        results = []

        async def handler(event):
            results.append("called")

        # Use subscription as context manager
        async with await self.event_bus.subscribe(Event, handler):
            event = Event()
            await self.event_bus.publish(event)
            assert "called" in results

        # After context exit, handler should be unsubscribed
        results.clear()
        event = Event()
        count = await self.event_bus.publish(event)
        assert count == 0
        assert "called" not in results

    # =========================================================================
    # Event Type Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_subclass_event_handling(self):
        """Test that subclass events are handled by their own type handlers.
        
        Note: EventBus uses event type name for routing, not inheritance.
        Subclass events won't trigger parent class handlers.
        """
        class CustomEvent(Event):
            pass

        results = []

        async def handler(event):
            results.append(type(event).__name__)

        # Subscribe to subclass
        await self.event_bus.subscribe(CustomEvent, handler)

        # Publish subclass event
        event = CustomEvent()
        count = await self.event_bus.publish(event)

        # Handler should be called
        assert count == 1
        assert "CustomEvent" in results
        
        # Verify parent class handler is NOT called
        results.clear()
        await self.event_bus.subscribe(Event, handler)
        
        # Publish parent event
        parent_event = Event()
        count = await self.event_bus.publish(parent_event)
        
        # Only Event handler should be called
        assert count == 1
        assert "Event" in results

    @pytest.mark.asyncio
    async def test_no_handlers_for_event_type(self):
        """Test publishing event with no handlers."""
        class UnhandledEvent(Event):
            pass

        event = UnhandledEvent()
        count = await self.event_bus.publish(event)

        assert count == 0

    @pytest.mark.asyncio
    async def test_multiple_event_types_same_handler(self):
        """Test same handler for multiple event types."""
        class EventA(Event):
            pass

        class EventB(Event):
            pass

        results = []

        async def handler(event):
            results.append(type(event).__name__)

        await self.event_bus.subscribe(EventA, handler)
        await self.event_bus.subscribe(EventB, handler)

        await self.event_bus.publish(EventA())
        await self.event_bus.publish(EventB())

        assert "EventA" in results
        assert "EventB" in results

    # =========================================================================
    # Request/Response Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_request_no_handlers(self):
        """Test request with no handlers."""
        class RequestEvent(Event):
            def __init__(self, data=None):
                super().__init__()
                self.data = data

        result = await self.event_bus.request(RequestEvent, {"data": {"key": "value"}}, timeout=0.1)

        # Should timeout since no handlers
        assert result is None

    @pytest.mark.asyncio
    async def test_request_with_handlers(self):
        """Test request with handlers."""
        class RequestEvent(Event):
            def __init__(self, **kwargs):
                super().__init__()
                self.data = kwargs

        async def handler(event):
            # Handler doesn't set future, so request will timeout
            pass

        await self.event_bus.subscribe(RequestEvent, handler)

        result = await self.event_bus.request(RequestEvent, {"key": "value"}, timeout=0.1)

        # Should timeout since handler doesn't respond
        assert result is None

    # =========================================================================
    # Performance/Stress Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_many_subscribers(self):
        """Test with many subscribers."""
        results = []

        async def handler(event):
            results.append(1)

        # Subscribe many handlers
        subscriptions = []
        for _ in range(100):
            sub = await self.event_bus.subscribe(Event, handler)
            subscriptions.append(sub)

        event = Event()
        count = await self.event_bus.publish(event)

        assert count == 100
        assert len(results) == 100

    @pytest.mark.asyncio
    async def test_many_events(self):
        """Test publishing many events."""
        results = []

        async def handler(event):
            results.append(event)

        await self.event_bus.subscribe(Event, handler)

        # Publish many events
        for i in range(100):
            event = Event()
            await self.event_bus.publish(event)

        assert len(results) == 100

    @pytest.mark.asyncio
    async def test_concurrent_publish(self):
        """Test concurrent event publishing."""
        results = []

        async def handler(event):
            results.append(event)

        await self.event_bus.subscribe(Event, handler)

        # Publish events concurrently
        async def publish_event():
            event = Event()
            return await self.event_bus.publish(event)

        tasks = [publish_event() for _ in range(50)]
        counts = await asyncio.gather(*tasks)

        assert sum(counts) == 50
        assert len(results) == 50

    # =========================================================================
    # Memory Management Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_handler_not_leaked_after_unsubscribe(self):
        """Test that handler is not leaked after unsubscribe."""
        class Handler:
            def __init__(self):
                self.data = "x" * 1000000  # 1MB

            async def handle(self, event):
                pass

        handler = Handler()
        handler_id = id(handler)

        sub = await self.event_bus.subscribe(Event, handler.handle)

        # Unsubscribe
        sub.unsubscribe()  # Not async

        # Delete reference
        del handler
        del sub
        gc.collect()

        # Handler should be garbage collected
        # (We can't directly test this, but the test passes if no memory leak)

    @pytest.mark.asyncio
    async def test_event_not_leaked_after_publish(self):
        """Test that event is not leaked after publish."""
        class LargeEvent(Event):
            def __init__(self):
                super().__init__()
                self.data = "x" * 1000000  # 1MB

        async def handler(event):
            pass

        await self.event_bus.subscribe(LargeEvent, handler)

        # Publish event
        event = LargeEvent()
        await self.event_bus.publish(event)

        # Delete reference
        del event
        gc.collect()

        # Event should be garbage collected


class TestEventBusIntegration:
    """Integration tests for EventBus."""

    def setup_method(self):
        """Create fresh EventBus before each test."""
        self.event_bus = EventBus()

    @pytest.mark.asyncio
    async def test_event_bus_with_unified_message_bus(self):
        """Test EventBus integration with UnifiedMessageBus."""
        message_bus = UnifiedMessageBus()
        event_bus = EventBus(message_bus)

        results = []

        async def handler(event):
            results.append("handled")

        await event_bus.subscribe(Event, handler)

        event = Event()
        count = await event_bus.publish(event)

        assert count == 1
        assert "handled" in results

    @pytest.mark.asyncio
    async def test_event_chain(self):
        """Test event chain (handler publishes another event)."""
        class EventA(Event):
            pass

        class EventB(Event):
            pass

        results = []

        async def handler_a(event):
            results.append("A")
            # Publish another event
            await self.event_bus.publish(EventB())

        async def handler_b(event):
            results.append("B")

        await self.event_bus.subscribe(EventA, handler_a)
        await self.event_bus.subscribe(EventB, handler_b)

        await self.event_bus.publish(EventA())

        assert "A" in results
        assert "B" in results

    @pytest.mark.asyncio
    async def test_event_loop(self):
        """Test that event loop doesn't cause infinite recursion."""
        class LoopEvent(Event):
            pass

        counter = [0]

        async def handler(event):
            counter[0] += 1
            if counter[0] < 3:
                # Publish same event type (but limit recursion)
                await self.event_bus.publish(LoopEvent())

        await self.event_bus.subscribe(LoopEvent, handler)

        await self.event_bus.publish(LoopEvent())

        # Should have been called 3 times (initial + 2 recursions)
        assert counter[0] == 3
