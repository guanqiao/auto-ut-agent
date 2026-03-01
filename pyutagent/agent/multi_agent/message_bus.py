"""Message bus for inter-agent communication.

Provides asynchronous messaging infrastructure for multi-agent collaboration.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages between agents."""
    TASK_ASSIGNMENT = auto()      # Task assigned to agent
    TASK_RESULT = auto()          # Task completion result
    TASK_FAILED = auto()          # Task failure notification
    COORDINATION = auto()         # Coordination messages
    KNOWLEDGE_SHARE = auto()      # Knowledge sharing
    QUERY = auto()                # Query/request for information
    RESPONSE = auto()             # Response to query
    BROADCAST = auto()            # Broadcast to all agents
    HEARTBEAT = auto()            # Agent health check
    ERROR = auto()                # Error notification


@dataclass
class AgentMessage:
    """Message exchanged between agents."""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None  # For request-response pairing
    priority: int = 5  # 1-10, lower is higher priority
    
    @classmethod
    def create(
        cls,
        sender_id: str,
        recipient_id: Optional[str],
        message_type: MessageType,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: int = 5
    ) -> 'AgentMessage':
        """Create a new message."""
        return cls(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority
        )


class MessageBus:
    """Asynchronous message bus for agent communication.
    
    Features:
    - Point-to-point messaging
    - Broadcast messaging
    - Message filtering and routing
    - Priority-based delivery
    - Async/await support
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """Initialize message bus.
        
        Args:
            max_queue_size: Maximum messages per agent queue
        """
        self.max_queue_size = max_queue_size
        self._queues: Dict[str, asyncio.Queue] = {}
        self._subscribers: Dict[str, Set[str]] = defaultdict(set)  # message_type -> agent_ids
        self._broadcast_subscribers: Set[str] = set()
        self._message_history: List[AgentMessage] = []
        self._history_limit = 1000
        self._lock = asyncio.Lock()
        
        logger.info(f"[MessageBus] Initialized with max_queue_size={max_queue_size}")
    
    async def register_agent(self, agent_id: str) -> asyncio.Queue:
        """Register an agent to receive messages.
        
        Args:
            agent_id: Unique agent identifier
            
        Returns:
            Message queue for the agent
        """
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
                self._broadcast_subscribers.add(agent_id)
                logger.debug(f"[MessageBus] Registered agent: {agent_id}")
            return self._queues[agent_id]
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent.
        
        Args:
            agent_id: Agent to unregister
        """
        async with self._lock:
            if agent_id in self._queues:
                del self._queues[agent_id]
            self._broadcast_subscribers.discard(agent_id)
            for subscribers in self._subscribers.values():
                subscribers.discard(agent_id)
            logger.debug(f"[MessageBus] Unregistered agent: {agent_id}")
    
    async def subscribe(self, agent_id: str, message_type: MessageType):
        """Subscribe agent to specific message type.
        
        Args:
            agent_id: Agent to subscribe
            message_type: Type of messages to receive
        """
        async with self._lock:
            self._subscribers[message_type].add(agent_id)
            logger.debug(f"[MessageBus] Agent {agent_id} subscribed to {message_type.name}")
    
    async def unsubscribe(self, agent_id: str, message_type: MessageType):
        """Unsubscribe agent from message type.
        
        Args:
            agent_id: Agent to unsubscribe
            message_type: Type to unsubscribe from
        """
        async with self._lock:
            self._subscribers[message_type].discard(agent_id)
            logger.debug(f"[MessageBus] Agent {agent_id} unsubscribed from {message_type.name}")
    
    async def send(self, message: AgentMessage) -> bool:
        """Send a message.
        
        Args:
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        # Store in history
        self._message_history.append(message)
        if len(self._message_history) > self._history_limit:
            self._message_history.pop(0)
        
        if message.recipient_id:
            # Point-to-point message
            return await self._send_to_agent(message, message.recipient_id)
        else:
            # Broadcast message
            return await self._broadcast(message)
    
    async def _send_to_agent(self, message: AgentMessage, agent_id: str) -> bool:
        """Send message to specific agent.
        
        Args:
            message: Message to send
            agent_id: Target agent
            
        Returns:
            True if sent successfully
        """
        async with self._lock:
            if agent_id not in self._queues:
                logger.warning(f"[MessageBus] Agent not found: {agent_id}")
                return False
            
            queue = self._queues[agent_id]
        
        try:
            # Use timeout to avoid blocking
            await asyncio.wait_for(queue.put(message), timeout=1.0)
            logger.debug(f"[MessageBus] Message {message.message_id[:8]} sent to {agent_id}")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"[MessageBus] Queue full for agent: {agent_id}")
            return False
    
    async def _broadcast(self, message: AgentMessage) -> bool:
        """Broadcast message to all subscribers.
        
        Args:
            message: Message to broadcast
            
        Returns:
            True if sent to at least one agent
        """
        async with self._lock:
            # Get subscribers for this message type
            type_subscribers = self._subscribers.get(message.message_type, set())
            # Also send to broadcast subscribers
            all_recipients = type_subscribers | self._broadcast_subscribers
        
        if not all_recipients:
            logger.debug(f"[MessageBus] No subscribers for broadcast")
            return False
        
        # Send to all recipients
        tasks = [self._send_to_agent(message, agent_id) for agent_id in all_recipients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        logger.debug(f"[MessageBus] Broadcast to {success_count}/{len(all_recipients)} agents")
        
        return success_count > 0
    
    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[AgentMessage]:
        """Receive message for agent.
        
        Args:
            agent_id: Agent to receive for
            timeout: Optional timeout in seconds
            
        Returns:
            Message or None if timeout
        """
        async with self._lock:
            if agent_id not in self._queues:
                logger.warning(f"[MessageBus] Agent not registered: {agent_id}")
                return None
            queue = self._queues[agent_id]
        
        try:
            if timeout:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                message = await queue.get()
            
            logger.debug(f"[MessageBus] Agent {agent_id} received message {message.message_id[:8]}")
            return message
        except asyncio.TimeoutError:
            return None
    
    async def request_response(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Send request and wait for response.
        
        Args:
            sender_id: Sending agent
            recipient_id: Target agent
            message_type: Type of request
            payload: Request payload
            timeout: Timeout in seconds
            
        Returns:
            Response message or None if timeout
        """
        correlation_id = str(uuid.uuid4())
        
        # Create request message
        request = AgentMessage.create(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
            priority=3  # Higher priority for requests
        )
        
        # Send request
        if not await self.send(request):
            return None
        
        # Wait for response
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            message = await self.receive(sender_id, timeout=1.0)
            
            if message and message.correlation_id == correlation_id:
                return message
        
        logger.warning(f"[MessageBus] Request timeout - correlation_id: {correlation_id[:8]}")
        return None
    
    def get_message_history(
        self,
        sender_id: Optional[str] = None,
        recipient_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get filtered message history.
        
        Args:
            sender_id: Filter by sender
            recipient_id: Filter by recipient
            message_type: Filter by type
            limit: Maximum results
            
        Returns:
            Filtered messages
        """
        filtered = self._message_history
        
        if sender_id:
            filtered = [m for m in filtered if m.sender_id == sender_id]
        if recipient_id:
            filtered = [m for m in filtered if m.recipient_id == recipient_id]
        if message_type:
            filtered = [m for m in filtered if m.message_type == message_type]
        
        return filtered[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "registered_agents": len(self._queues),
            "broadcast_subscribers": len(self._broadcast_subscribers),
            "message_history_size": len(self._message_history),
            "subscriptions": {
                msg_type.name: len(subscribers)
                for msg_type, subscribers in self._subscribers.items()
            },
            "queue_sizes": {
                agent_id: queue.qsize()
                for agent_id, queue in self._queues.items()
            }
        }
