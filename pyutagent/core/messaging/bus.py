"""Unified Message Bus.

This module provides a unified message bus that supports:
- Component communication
- Agent communication
- Request-Response pattern
- Broadcast pattern
- Priority-based delivery
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set, Awaitable

from .message import Message, MessageType, MessagePriority

logger = logging.getLogger(__name__)


@dataclass
class MessageBusStats:
    """Statistics for the message bus."""
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_broadcasts: int = 0
    total_errors: int = 0
    messages_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    messages_by_priority: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def record_sent(self, message: Message) -> None:
        """Record a sent message."""
        self.total_messages_sent += 1
        self.messages_by_type[message.type.name] += 1
        self.messages_by_priority[message.priority.name] += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "total_broadcasts": self.total_broadcasts,
            "total_errors": self.total_errors,
            "messages_by_type": dict(self.messages_by_type),
            "messages_by_priority": dict(self.messages_by_priority),
        }


class UnifiedMessageBus:
    """Unified message bus for all communication patterns.
    
    Features:
    - Point-to-point messaging
    - Broadcast messaging
    - Request-Response pattern with timeout
    - Priority-based delivery
    - Message filtering and routing
    - Subscription management
    - Message history
    - Statistics tracking
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        history_limit: int = 1000,
    ):
        """Initialize the message bus.
        
        Args:
            max_queue_size: Maximum messages per queue
            history_limit: Maximum messages in history
        """
        self.max_queue_size = max_queue_size
        self.history_limit = history_limit
        
        self._queues: Dict[str, asyncio.Queue] = {}
        self._subscribers: Dict[MessageType, Set[str]] = defaultdict(set)
        self._broadcast_subscribers: Set[str] = set()
        self._message_handlers: Dict[str, Callable[[Message], Awaitable[None]]] = {}
        
        self._history: List[Message] = []
        self._stats = MessageBusStats()
        self._lock = asyncio.Lock()
        
        logger.info(f"[UnifiedMessageBus] Initialized with max_queue_size={max_queue_size}")
    
    async def register(
        self,
        entity_id: str,
        handler: Optional[Callable[[Message], Awaitable[None]]] = None,
    ) -> asyncio.Queue:
        """Register an entity (component or agent) to receive messages.
        
        Args:
            entity_id: Unique entity identifier
            handler: Optional async handler for messages
            
        Returns:
            Message queue for the entity
        """
        async with self._lock:
            if entity_id not in self._queues:
                self._queues[entity_id] = asyncio.Queue(maxsize=self.max_queue_size)
                self._broadcast_subscribers.add(entity_id)
                logger.debug(f"[UnifiedMessageBus] Registered: {entity_id}")
            
            if handler:
                self._message_handlers[entity_id] = handler
        
        return self._queues[entity_id]
    
    async def unregister(self, entity_id: str) -> None:
        """Unregister an entity.
        
        Args:
            entity_id: Entity to unregister
        """
        async with self._lock:
            if entity_id in self._queues:
                del self._queues[entity_id]
            
            self._broadcast_subscribers.discard(entity_id)
            self._message_handlers.pop(entity_id, None)
            
            for subscribers in self._subscribers.values():
                subscribers.discard(entity_id)
            
            logger.debug(f"[UnifiedMessageBus] Unregistered: {entity_id}")
    
    async def subscribe(
        self,
        entity_id: str,
        message_type: MessageType,
    ) -> None:
        """Subscribe an entity to a specific message type.
        
        Args:
            entity_id: Entity to subscribe
            message_type: Type of messages to receive
        """
        async with self._lock:
            self._subscribers[message_type].add(entity_id)
            logger.debug(f"[UnifiedMessageBus] {entity_id} subscribed to {message_type.name}")
    
    async def unsubscribe(
        self,
        entity_id: str,
        message_type: MessageType,
    ) -> None:
        """Unsubscribe an entity from a message type.
        
        Args:
            entity_id: Entity to unsubscribe
            message_type: Type to unsubscribe from
        """
        async with self._lock:
            self._subscribers[message_type].discard(entity_id)
    
    async def send(self, message: Message) -> bool:
        """Send a message.
        
        Args:
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        if message.is_expired():
            logger.warning(f"[UnifiedMessageBus] Message expired: {message.id[:8]}")
            return False
        
        self._stats.record_sent(message)
        
        self._history.append(message)
        if len(self._history) > self.history_limit:
            self._history.pop(0)
        
        if message.is_broadcast():
            self._stats.total_broadcasts += 1
            return await self._broadcast(message)
        else:
            return await self._send_to_entity(message, message.recipient)
    
    async def _send_to_entity(self, message: Message, entity_id: str) -> bool:
        """Send message to a specific entity.
        
        Args:
            message: Message to send
            entity_id: Target entity
            
        Returns:
            True if sent successfully
        """
        async with self._lock:
            if entity_id not in self._queues:
                logger.warning(f"[UnifiedMessageBus] Entity not found: {entity_id}")
                return False
            queue = self._queues[entity_id]
        
        try:
            await asyncio.wait_for(queue.put(message), timeout=1.0)
            logger.debug(f"[UnifiedMessageBus] Message {message.id[:8]} sent to {entity_id}")
            
            handler = self._message_handlers.get(entity_id)
            if handler:
                asyncio.create_task(self._handle_message(handler, message))
            
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"[UnifiedMessageBus] Queue full for: {entity_id}")
            self._stats.total_errors += 1
            return False
    
    async def _handle_message(
        self,
        handler: Callable[[Message], Awaitable[None]],
        message: Message,
    ) -> None:
        """Handle a message with a handler.
        
        Args:
            handler: Async handler function
            message: Message to handle
        """
        try:
            await handler(message)
        except Exception as e:
            logger.error(f"[UnifiedMessageBus] Handler error: {e}")
            self._stats.total_errors += 1
    
    async def _broadcast(self, message: Message) -> bool:
        """Broadcast message to all subscribers.
        
        Args:
            message: Message to broadcast
            
        Returns:
            True if sent to at least one entity
        """
        async with self._lock:
            type_subscribers = self._subscribers.get(message.type, set())
            all_recipients = type_subscribers | self._broadcast_subscribers
        
        if not all_recipients:
            logger.debug(f"[UnifiedMessageBus] No subscribers for broadcast")
            return False
        
        tasks = [
            self._send_to_entity(message, entity_id)
            for entity_id in all_recipients
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        logger.debug(f"[UnifiedMessageBus] Broadcast to {success_count}/{len(all_recipients)} entities")
        
        return success_count > 0
    
    async def receive(
        self,
        entity_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Message]:
        """Receive a message for an entity.
        
        Args:
            entity_id: Entity to receive for
            timeout: Optional timeout in seconds
            
        Returns:
            Message or None if timeout
        """
        async with self._lock:
            if entity_id not in self._queues:
                logger.warning(f"[UnifiedMessageBus] Entity not registered: {entity_id}")
                return None
            queue = self._queues[entity_id]
        
        try:
            if timeout:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                message = await queue.get()
            
            self._stats.total_messages_received += 1
            logger.debug(f"[UnifiedMessageBus] {entity_id} received message {message.id[:8]}")
            return message
            
        except asyncio.TimeoutError:
            return None
    
    async def request(
        self,
        sender: str,
        recipient: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.COMPONENT_REQUEST,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """Send a request and wait for response.
        
        Args:
            sender: Sender ID
            recipient: Recipient ID
            payload: Request payload
            message_type: Type of request
            timeout: Timeout in seconds
            
        Returns:
            Response message or None if timeout
        """
        message = Message.create(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
        )
        
        if not await self.send(message):
            return None
        
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            response = await self.receive(sender, timeout=1.0)
            
            if response and response.correlation_id == message.id:
                return response
        
        logger.warning(f"[UnifiedMessageBus] Request timeout: {message.id[:8]}")
        return None
    
    async def respond(
        self,
        original_message: Message,
        response_payload: Dict[str, Any],
        sender: Optional[str] = None,
    ) -> bool:
        """Send a response to a message.
        
        Args:
            original_message: Message to respond to
            response_payload: Response payload
            sender: Optional sender override
            
        Returns:
            True if sent successfully
        """
        response = original_message.create_response(
            sender=sender or original_message.recipient,
            payload=response_payload,
        )
        
        return await self.send(response)
    
    def get_history(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100,
    ) -> List[Message]:
        """Get filtered message history.
        
        Args:
            sender: Filter by sender
            recipient: Filter by recipient
            message_type: Filter by type
            limit: Maximum results
            
        Returns:
            Filtered messages
        """
        filtered = self._history
        
        if sender:
            filtered = [m for m in filtered if m.sender == sender]
        if recipient:
            filtered = [m for m in filtered if m.recipient == recipient or m.is_broadcast()]
        if message_type:
            filtered = [m for m in filtered if m.type == message_type]
        
        return filtered[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self._stats.to_dict()
        stats.update({
            "registered_entities": len(self._queues),
            "broadcast_subscribers": len(self._broadcast_subscribers),
            "history_size": len(self._history),
            "queue_sizes": {
                entity_id: queue.qsize()
                for entity_id, queue in self._queues.items()
            },
        })
        return stats
    
    async def clear(self) -> None:
        """Clear all queues and history."""
        async with self._lock:
            for queue in self._queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            self._history.clear()
            self._stats = MessageBusStats()
        
        logger.info("[UnifiedMessageBus] Cleared all queues and history")
    
    def get_queue_size(self, entity_id: str) -> int:
        """Get the queue size for an entity.
        
        Args:
            entity_id: Entity to check
            
        Returns:
            Queue size or 0 if not found
        """
        queue = self._queues.get(entity_id)
        return queue.qsize() if queue else 0
