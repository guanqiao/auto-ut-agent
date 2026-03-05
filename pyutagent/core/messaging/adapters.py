"""Message Bus Adapters - 兼容旧接口的适配器.

This module provides adapters to maintain backward compatibility
with legacy message bus interfaces while using UnifiedMessageBus.
"""

import asyncio
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from .bus import UnifiedMessageBus
from .message import Message, MessageType, MessagePriority

T = TypeVar('T')


class EventBusAdapter:
    """Adapter for legacy EventBus interface.
    
    Provides compatibility with the old EventBus/AsyncEventBus API
    while internally using UnifiedMessageBus.
    
    Deprecated: Use UnifiedMessageBus directly instead.
    """
    
    def __init__(self, unified_bus: Optional[UnifiedMessageBus] = None):
        """Initialize adapter.
        
        Args:
            unified_bus: UnifiedMessageBus instance (creates new if None)
        """
        warnings.warn(
            "EventBusAdapter is deprecated. Use UnifiedMessageBus directly.",
            DeprecationWarning,
            stacklevel=2
        )
        self._bus = unified_bus or UnifiedMessageBus()
        self._subscriptions: Dict[str, List[Dict[str, Any]]] = {}
    
    def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], Any]
    ) -> 'Subscription':
        """Subscribe to events (legacy interface).
        
        Args:
            event_type: Type of event to subscribe to
            handler: Event handler callback
            
        Returns:
            Subscription object
        """
        sub_id = f"{event_type.__name__}_{id(handler)}"
        
        if event_type.__name__ not in self._subscriptions:
            self._subscriptions[event_type.__name__] = []
        
        subscription = {
            "id": sub_id,
            "type": event_type,
            "handler": handler
        }
        self._subscriptions[event_type.__name__].append(subscription)
        
        # Create a wrapper class for compatibility
        return Subscription(sub_id, event_type, handler, self)
    
    def unsubscribe(self, subscription: 'Subscription') -> None:
        """Unsubscribe from events.
        
        Args:
            subscription: Subscription to remove
        """
        event_type_name = subscription.event_type.__name__
        if event_type_name in self._subscriptions:
            self._subscriptions[event_type_name] = [
                s for s in self._subscriptions[event_type_name]
                if s["id"] != subscription.id
            ]
    
    def publish(self, event: Any) -> None:
        """Publish event (legacy synchronous interface).
        
        Args:
            event: Event to publish
        """
        event_type = type(event).__name__
        
        if event_type in self._subscriptions:
            for sub in self._subscriptions[event_type]:
                try:
                    handler = sub["handler"]
                    if asyncio.iscoroutinefunction(handler):
                        # Schedule async handler
                        asyncio.create_task(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Event handler error: {e}")
    
    async def publish_async(self, event: Any) -> None:
        """Publish event asynchronously.
        
        Args:
            event: Event to publish
        """
        self.publish(event)


class AsyncEventBusAdapter(EventBusAdapter):
    """Adapter for legacy AsyncEventBus interface.
    
    Deprecated: Use UnifiedMessageBus directly instead.
    """
    
    def __init__(self, unified_bus: Optional[UnifiedMessageBus] = None):
        """Initialize adapter."""
        warnings.warn(
            "AsyncEventBusAdapter is deprecated. Use UnifiedMessageBus directly.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(unified_bus)
    
    async def publish(self, event: Any) -> None:
        """Publish event asynchronously.
        
        Args:
            event: Event to publish
        """
        await super().publish_async(event)


class MessageBusAdapter:
    """Adapter for legacy MessageBus interface.
    
    Provides compatibility with the old MessageBus API
    while internally using UnifiedMessageBus.
    
    Deprecated: Use UnifiedMessageBus directly instead.
    """
    
    def __init__(self, unified_bus: Optional[UnifiedMessageBus] = None):
        """Initialize adapter.
        
        Args:
            unified_bus: UnifiedMessageBus instance (creates new if None)
        """
        warnings.warn(
            "MessageBusAdapter is deprecated. Use UnifiedMessageBus directly.",
            DeprecationWarning,
            stacklevel=2
        )
        self._bus = unified_bus or UnifiedMessageBus()
        self._entity_id = "legacy_adapter"
        
        # Register self as an entity
        asyncio.create_task(self._register())
    
    async def _register(self) -> None:
        """Register with the unified bus."""
        await self._bus.register(self._entity_id)
    
    def create_queue(self, queue_name: str, max_size: int = 1000) -> None:
        """Create a message queue (legacy interface).
        
        In UnifiedMessageBus, queues are created automatically on registration.
        This method is kept for compatibility but does nothing.
        
        Args:
            queue_name: Name of the queue
            max_size: Maximum queue size (ignored)
        """
        # UnifiedMessageBus creates queues automatically
        pass
    
    async def publish(self, queue_name: str, message: Any) -> bool:
        """Publish message to queue (legacy interface).
        
        Args:
            queue_name: Target queue/entity name
            message: Message to publish
            
        Returns:
            True if published successfully
        """
        # Convert legacy message to unified message if needed
        if hasattr(message, 'message_id'):
            # Legacy Message object
            unified_message = Message.create(
                sender=self._entity_id,
                recipient=queue_name,
                message_type=MessageType.COMPONENT_REQUEST,
                payload=message.content if hasattr(message, 'content') else {"data": message},
                priority=MessagePriority.NORMAL
            )
        else:
            # Generic object
            unified_message = Message.create(
                sender=self._entity_id,
                recipient=queue_name,
                message_type=MessageType.COMPONENT_REQUEST,
                payload={"data": message},
                priority=MessagePriority.NORMAL
            )
        
        return await self._bus.send(unified_message)
    
    async def subscribe(
        self,
        queue_name: str,
        callback: Callable[[Any], Any]
    ) -> None:
        """Subscribe to queue messages (legacy interface).
        
        Args:
            queue_name: Queue to subscribe to
            callback: Message handler callback
        """
        # Register callback as message handler
        async def handler(message: Message) -> None:
            await callback(message)
        
        await self._bus.register(queue_name, handler)
    
    async def consume(self, queue_name: str) -> Optional[Any]:
        """Consume message from queue (legacy interface).
        
        Args:
            queue_name: Queue to consume from
            
        Returns:
            Message or None
        """
        return await self._bus.receive(queue_name)


class Subscription:
    """Subscription object for backward compatibility.
    
    This class mimics the old Subscription dataclass.
    """
    
    def __init__(
        self,
        sub_id: str,
        event_type: Type[Any],
        handler: Callable,
        adapter: EventBusAdapter
    ):
        """Initialize subscription.
        
        Args:
            sub_id: Subscription ID
            event_type: Event type
            handler: Event handler
            adapter: Parent adapter
        """
        self.id = sub_id
        self.event_type = event_type
        self._handler = handler
        self._adapter = adapter
    
    def unsubscribe(self) -> None:
        """Unsubscribe from events."""
        self._adapter.unsubscribe(self)


# Convenience functions for migration

def create_event_bus_adapter(
    unified_bus: Optional[UnifiedMessageBus] = None
) -> EventBusAdapter:
    """Create EventBus adapter (for migration).
    
    Args:
        unified_bus: Optional UnifiedMessageBus instance
        
    Returns:
        EventBusAdapter instance
        
    Deprecated: Use UnifiedMessageBus directly.
    """
    warnings.warn(
        "create_event_bus_adapter is deprecated. Use UnifiedMessageBus directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return EventBusAdapter(unified_bus)


def create_message_bus_adapter(
    unified_bus: Optional[UnifiedMessageBus] = None
) -> MessageBusAdapter:
    """Create MessageBus adapter (for migration).
    
    Args:
        unified_bus: Optional UnifiedMessageBus instance
        
    Returns:
        MessageBusAdapter instance
        
    Deprecated: Use UnifiedMessageBus directly.
    """
    warnings.warn(
        "create_message_bus_adapter is deprecated. Use UnifiedMessageBus directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return MessageBusAdapter(unified_bus)
