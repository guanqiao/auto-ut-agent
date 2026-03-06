"""Tests for EventBus - Unified Event Bus."""

import pytest
import asyncio
from dataclasses import dataclass
from datetime import datetime

from pyutagent.core.event_bus import (
    EventBus,
    Event,
    Subscription,
    create_event_bus,
    publish_event,
)


# Test event classes
@dataclass
class TestEvent(Event):
    """Test event."""
    data: str = ""


@dataclass
class DataEvent(Event):
    """Data event with payload."""
    payload: dict = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.payload is None:
            self.payload = {}


class TestEventBus:
    """Test cases for EventBus."""
    
    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for each test."""
        return EventBus()
    
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test subscribing to events and publishing."""
        # Arrange
        received_events = []
        
        async def handler(event):
            received_events.append(event)
        
        # Act
        sub = await event_bus.subscribe(TestEvent, handler)
        count = await event_bus.publish(TestEvent(data="hello"))
        
        # Assert
        assert count == 1
        assert len(received_events) == 1
        assert received_events[0].data == "hello"
        
        # Cleanup
        sub.unsubscribe()
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers for same event type."""
        # Arrange
        handler1_calls = []
        handler2_calls = []
        
        async def handler1(event):
            handler1_calls.append(event)
        
        async def handler2(event):
            handler2_calls.append(event)
        
        # Act
        sub1 = await event_bus.subscribe(TestEvent, handler1)
        sub2 = await event_bus.subscribe(TestEvent, handler2)
        count = await event_bus.publish(TestEvent(data="test"))
        
        # Assert
        assert count == 2
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1
        
        # Cleanup
        sub1.unsubscribe()
        sub2.unsubscribe()
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        # Arrange
        received_events = []
        
        async def handler(event):
            received_events.append(event)
        
        sub = await event_bus.subscribe(TestEvent, handler)
        
        # Act - publish before unsubscribe
        await event_bus.publish(TestEvent(data="before"))
        
        # Unsubscribe
        sub.unsubscribe()
        
        # Publish after unsubscribe
        count = await event_bus.publish(TestEvent(data="after"))
        
        # Assert
        assert count == 0
        assert len(received_events) == 1
        assert received_events[0].data == "before"
    
    @pytest.mark.asyncio
    async def test_sync_handler(self, event_bus):
        """Test synchronous event handler."""
        # Arrange
        received_events = []
        
        def sync_handler(event):
            received_events.append(event)
        
        # Act
        sub = await event_bus.subscribe(TestEvent, sync_handler)
        count = await event_bus.publish(TestEvent(data="sync"))
        
        # Assert
        assert count == 1
        assert len(received_events) == 1
        
        # Cleanup
        sub.unsubscribe()
    
    @pytest.mark.asyncio
    async def test_no_subscribers(self, event_bus):
        """Test publishing with no subscribers."""
        # Act
        count = await event_bus.publish(TestEvent(data="no one listening"))
        
        # Assert
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_different_event_types(self, event_bus):
        """Test different event types don't interfere."""
        # Arrange
        test_events = []
        data_events = []
        
        async def test_handler(event):
            test_events.append(event)
        
        async def data_handler(event):
            data_events.append(event)
        
        # Act
        sub1 = await event_bus.subscribe(TestEvent, test_handler)
        sub2 = await event_bus.subscribe(DataEvent, data_handler)
        
        await event_bus.publish(TestEvent(data="test"))
        await event_bus.publish(DataEvent(payload={"key": "value"}))
        
        # Assert
        assert len(test_events) == 1
        assert len(data_events) == 1
        assert test_events[0].data == "test"
        assert data_events[0].payload == {"key": "value"}
        
        # Cleanup
        sub1.unsubscribe()
        sub2.unsubscribe()
    
    @pytest.mark.asyncio
    async def test_handler_error(self, event_bus):
        """Test handler that raises exception."""
        # Arrange
        good_calls = []
        
        async def bad_handler(event):
            raise ValueError("Handler error")
        
        async def good_handler(event):
            good_calls.append(event)
        
        # Act
        sub1 = await event_bus.subscribe(TestEvent, bad_handler)
        sub2 = await event_bus.subscribe(TestEvent, good_handler)
        count = await event_bus.publish(TestEvent(data="test"))
        
        # Assert - good handler should still be called
        assert count == 1  # Only good handler succeeds
        assert len(good_calls) == 1
        
        # Cleanup
        sub1.unsubscribe()
        sub2.unsubscribe()
    
    @pytest.mark.asyncio
    async def test_get_subscriber_count(self, event_bus):
        """Test getting subscriber count."""
        # Arrange
        async def handler(event):
            pass
        
        # Act & Assert
        assert event_bus.get_subscriber_count(TestEvent) == 0
        
        sub = await event_bus.subscribe(TestEvent, handler)
        assert event_bus.get_subscriber_count(TestEvent) == 1
        
        sub.unsubscribe()
        assert event_bus.get_subscriber_count(TestEvent) == 0
    
    @pytest.mark.asyncio
    async def test_get_all_subscriber_counts(self, event_bus):
        """Test getting all subscriber counts."""
        # Arrange
        async def handler(event):
            pass
        
        # Act
        sub1 = await event_bus.subscribe(TestEvent, handler)
        sub2 = await event_bus.subscribe(DataEvent, handler)
        
        counts = event_bus.get_all_subscriber_counts()
        
        # Assert
        assert counts["TestEvent"] == 1
        assert counts["DataEvent"] == 1
        
        # Cleanup
        sub1.unsubscribe()
        sub2.unsubscribe()
    
    @pytest.mark.asyncio
    async def test_clear(self, event_bus):
        """Test clearing all subscriptions."""
        # Arrange
        async def handler(event):
            pass
        
        sub1 = await event_bus.subscribe(TestEvent, handler)
        sub2 = await event_bus.subscribe(DataEvent, handler)
        
        # Act
        await event_bus.clear()
        count = await event_bus.publish(TestEvent(data="test"))
        
        # Assert
        assert count == 0
        assert event_bus.get_subscriber_count(TestEvent) == 0
    
    @pytest.mark.asyncio
    async def test_subscription_context_manager(self, event_bus):
        """Test subscription as context manager."""
        # Arrange
        received_events = []
        
        async def handler(event):
            received_events.append(event)
        
        # Act
        async with await event_bus.subscribe(TestEvent, handler):
            await event_bus.publish(TestEvent(data="inside"))
        
        # After context exit
        await event_bus.publish(TestEvent(data="outside"))
        
        # Assert
        assert len(received_events) == 1
        assert received_events[0].data == "inside"
    
    @pytest.mark.asyncio
    async def test_event_timestamp(self, event_bus):
        """Test event timestamp is set."""
        # Arrange
        before = datetime.now()
        
        received_event = None
        async def handler(event):
            nonlocal received_event
            received_event = event
        
        # Act
        sub = await event_bus.subscribe(TestEvent, handler)
        await event_bus.publish(TestEvent(data="test"))
        after = datetime.now()
        
        # Assert
        assert received_event is not None
        assert received_event.timestamp is not None
        assert before <= received_event.timestamp <= after
        
        # Cleanup
        sub.unsubscribe()
    
    @pytest.mark.asyncio
    async def test_event_get_event_type(self, event_bus):
        """Test getting event type name."""
        # Arrange
        event = TestEvent(data="test")
        
        # Act & Assert
        assert event.get_event_type() == "TestEvent"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_create_event_bus(self):
        """Test create_event_bus function."""
        # Act
        bus = create_event_bus()
        
        # Assert
        assert isinstance(bus, EventBus)
    
    @pytest.mark.asyncio
    async def test_publish_event(self):
        """Test publish_event convenience function."""
        # Arrange
        received = []
        
        async def handler(event):
            received.append(event)
        
        bus = EventBus()
        await bus.subscribe(TestEvent, handler)
        
        # Act
        count = await publish_event(TestEvent(data="test"), bus)
        
        # Assert
        assert count == 1
        assert len(received) == 1


class TestEventBusEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.asyncio
    async def test_double_unsubscribe(self):
        """Test unsubscribing twice."""
        # Arrange
        bus = EventBus()
        
        async def handler(event):
            pass
        
        sub = await bus.subscribe(TestEvent, handler)
        
        # Act - unsubscribe twice (should not raise)
        sub.unsubscribe()
        sub.unsubscribe()  # Should be safe
        
        # Assert
        assert bus.get_subscriber_count(TestEvent) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_event_buses(self):
        """Test multiple independent event buses."""
        # Arrange
        bus1 = EventBus()
        bus2 = EventBus()
        
        bus1_events = []
        bus2_events = []
        
        async def handler1(event):
            bus1_events.append(event)
        
        async def handler2(event):
            bus2_events.append(event)
        
        # Act
        await bus1.subscribe(TestEvent, handler1)
        await bus2.subscribe(TestEvent, handler2)
        
        await bus1.publish(TestEvent(data="bus1"))
        await bus2.publish(TestEvent(data="bus2"))
        
        # Assert
        assert len(bus1_events) == 1
        assert len(bus2_events) == 1
        assert bus1_events[0].data == "bus1"
        assert bus2_events[0].data == "bus2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
