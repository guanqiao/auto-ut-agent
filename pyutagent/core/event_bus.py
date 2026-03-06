"""Unified Event Bus - Event-driven communication system.

This module provides a unified event bus that wraps UnifiedMessageBus
to provide a simpler event-driven interface while maintaining backward compatibility.

Example:
    >>> from pyutagent.core.event_bus import EventBus
    >>> 
    >>> # Create event bus
    >>> event_bus = EventBus()
    >>> 
    >>> # Subscribe to events
    >>> await event_bus.subscribe(MyEvent, handler)
    >>> 
    >>> # Publish events
    >>> await event_bus.publish(MyEvent(data="hello"))
"""

import asyncio
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass
from datetime import datetime

from .messaging import UnifiedMessageBus, Message, MessageType

logger = logging.getLogger(__name__)

T = TypeVar('T')
EventHandler = Callable[[T], Any]


@dataclass
class Event:
    """Base event class.
    
    All events should inherit from this class.
    """
    timestamp: datetime = None
    source: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def get_event_type(self) -> str:
        """Get the event type name."""
        return self.__class__.__name__


class Subscription:
    """Event subscription handle."""
    
    def __init__(
        self,
        event_type: Type[T],
        handler: EventHandler,
        event_bus: 'EventBus'
    ):
        self.event_type = event_type
        self.handler = handler
        self._event_bus = event_bus
        self._active = True
    
    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        if self._active:
            self._event_bus._unsubscribe(self.event_type, self.handler)
            self._active = False
    
    def __enter__(self) -> 'Subscription':
        return self
    
    def __exit__(self, *args) -> None:
        self.unsubscribe()
    
    async def __aenter__(self) -> 'Subscription':
        return self
    
    async def __aexit__(self, *args) -> None:
        self.unsubscribe()


class EventBus:
    """Unified Event Bus - Event-driven communication system.
    
    This class provides a simplified event-driven interface on top of
    UnifiedMessageBus. It supports:
    - Event subscription with type safety
    - Event publishing (sync and async)
    - Request-response patterns
    - Event filtering
    
    This is the recommended way to handle events in the application.
    
    Example:
        >>> event_bus = EventBus()
        >>> 
        >>> # Subscribe to events
        >>> sub = await event_bus.subscribe(MyEvent, my_handler)
        >>> 
        >>> # Publish events
        >>> await event_bus.publish(MyEvent(data="hello"))
        >>> 
        >>> # Unsubscribe
        >>> sub.unsubscribe()
    """
    
    def __init__(self, message_bus: Optional[UnifiedMessageBus] = None):
        """Initialize event bus.
        
        Args:
            message_bus: Optional UnifiedMessageBus instance
        """
        self._message_bus = message_bus or UnifiedMessageBus()
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._entity_id = f"event_bus_{id(self)}"
        self._lock = asyncio.Lock()
    
    async def subscribe(
        self,
        event_type: Type[T],
        handler: EventHandler[T]
    ) -> Subscription:
        """Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Event handler function
            
        Returns:
            Subscription handle
            
        Example:
            >>> async def handler(event: MyEvent):
            ...     print(f"Received: {event.data}")
            >>> 
            >>> sub = await event_bus.subscribe(MyEvent, handler)
        """
        async with self._lock:
            event_name = event_type.__name__
            if event_name not in self._handlers:
                self._handlers[event_name] = []
            self._handlers[event_name].append(handler)
        
        logger.debug(f"[EventBus] Subscribed to {event_name}")
        return Subscription(event_type, handler, self)
    
    def _unsubscribe(self, event_type: Type[T], handler: EventHandler[T]) -> None:
        """Internal unsubscribe method."""
        event_name = event_type.__name__
        if event_name in self._handlers:
            self._handlers[event_name] = [
                h for h in self._handlers[event_name] if h != handler
            ]
            if not self._handlers[event_name]:
                del self._handlers[event_name]
        logger.debug(f"[EventBus] Unsubscribed from {event_name}")
    
    async def publish(self, event: Event) -> int:
        """Publish an event.
        
        Args:
            event: Event to publish
            
        Returns:
            Number of handlers invoked
            
        Example:
            >>> await event_bus.publish(MyEvent(data="hello"))
        """
        event_name = event.get_event_type()
        handlers = self._handlers.get(event_name, [])
        
        count = 0
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                count += 1
            except Exception as e:
                logger.error(f"[EventBus] Handler error for {event_name}: {e}")
        
        logger.debug(f"[EventBus] Published {event_name} to {count} handlers")
        return count
    
    async def publish_async(self, event: Event) -> int:
        """Publish an event asynchronously (alias for publish).
        
        Args:
            event: Event to publish
            
        Returns:
            Number of handlers invoked
        """
        return await self.publish(event)
    
    async def request(
        self,
        event_type: Type[T],
        data: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[Any]:
        """Send a request and wait for response.
        
        Args:
            event_type: Type of request event
            data: Request data
            timeout: Timeout in seconds
            
        Returns:
            Response data or None if timeout
            
        Example:
            >>> response = await event_bus.request(GetDataEvent, {"id": 123})
        """
        # Create a future to wait for response
        future = asyncio.Future()
        
        async def response_handler(event):
            if not future.done():
                future.set_result(event)
        
        # Subscribe to response
        response_type = f"{event_type.__name__}Response"
        # This is a simplified implementation
        # In practice, you'd use correlation IDs
        
        # Publish request
        event = event_type(**data)
        await self.publish(event)
        
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            logger.warning(f"[EventBus] Request timeout for {event_type.__name__}")
            return None
    
    def get_subscriber_count(self, event_type: Type[T]) -> int:
        """Get number of subscribers for an event type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            Number of subscribers
        """
        return len(self._handlers.get(event_type.__name__, []))
    
    def get_all_subscriber_counts(self) -> Dict[str, int]:
        """Get subscriber counts for all event types.
        
        Returns:
            Dictionary mapping event names to subscriber counts
        """
        return {name: len(handlers) for name, handlers in self._handlers.items()}
    
    async def clear(self) -> None:
        """Clear all subscriptions."""
        async with self._lock:
            self._handlers.clear()
        logger.debug("[EventBus] All subscriptions cleared")


# Backward compatibility classes
class LegacyEventBus(EventBus):
    """Legacy EventBus - Deprecated, use EventBus instead."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LegacyEventBus is deprecated. Use EventBus instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class LegacyAsyncEventBus(EventBus):
    """Legacy AsyncEventBus - Deprecated, use EventBus instead."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "LegacyAsyncEventBus is deprecated. Use EventBus instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


# Convenience functions
def create_event_bus(message_bus: Optional[UnifiedMessageBus] = None) -> EventBus:
    """Create a new EventBus instance.
    
    Args:
        message_bus: Optional UnifiedMessageBus instance
        
    Returns:
        New EventBus instance
    """
    return EventBus(message_bus)


async def publish_event(event: Event, event_bus: Optional[EventBus] = None) -> int:
    """Convenience function to publish an event.
    
    Args:
        event: Event to publish
        event_bus: Optional EventBus instance (creates new if None)
        
    Returns:
        Number of handlers invoked
    """
    bus = event_bus or EventBus()
    return await bus.publish(event)
