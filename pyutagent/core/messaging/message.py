"""Unified Message Definition.

This module provides a unified message protocol that supports:
- Component communication
- Agent communication
- Request-Response pattern
- Broadcast pattern
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any
import uuid
import asyncio


class MessageType(Enum):
    """Unified message types for all communication patterns."""
    
    COMPONENT_REQUEST = auto()
    COMPONENT_RESPONSE = auto()
    COMPONENT_NOTIFICATION = auto()
    
    AGENT_TASK = auto()
    AGENT_RESULT = auto()
    AGENT_COORDINATION = auto()
    AGENT_QUERY = auto()
    AGENT_RESPONSE = auto()
    
    BROADCAST = auto()
    HEARTBEAT = auto()
    
    ERROR = auto()
    CONTROL = auto()
    
    @property
    def is_request(self) -> bool:
        """Check if this is a request-type message."""
        return self in (
            MessageType.COMPONENT_REQUEST,
            MessageType.AGENT_TASK,
            MessageType.AGENT_QUERY,
        )
    
    @property
    def is_response(self) -> bool:
        """Check if this is a response-type message."""
        return self in (
            MessageType.COMPONENT_RESPONSE,
            MessageType.AGENT_RESULT,
            MessageType.AGENT_RESPONSE,
        )
    
    @property
    def is_agent_message(self) -> bool:
        """Check if this is an agent-related message."""
        return self in (
            MessageType.AGENT_TASK,
            MessageType.AGENT_RESULT,
            MessageType.AGENT_COORDINATION,
            MessageType.AGENT_QUERY,
            MessageType.AGENT_RESPONSE,
        )
    
    @property
    def is_component_message(self) -> bool:
        """Check if this is a component-related message."""
        return self in (
            MessageType.COMPONENT_REQUEST,
            MessageType.COMPONENT_RESPONSE,
            MessageType.COMPONENT_NOTIFICATION,
        )


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5
    
    def __lt__(self, other: "MessagePriority") -> bool:
        return self.value < other.value
    
    def __le__(self, other: "MessagePriority") -> bool:
        return self.value <= other.value
    
    def __gt__(self, other: "MessagePriority") -> bool:
        return self.value > other.value
    
    def __ge__(self, other: "MessagePriority") -> bool:
        return self.value >= other.value


@dataclass
class Message:
    """Unified message definition.
    
    Supports:
    - Component communication (sender/recipient are component IDs)
    - Agent communication (sender/recipient are agent IDs)
    - Request-Response (using correlation_id)
    - Broadcast (recipient is None)
    - Priority-based delivery
    - TTL-based expiration
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.COMPONENT_REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    
    sender: str = ""
    recipient: Optional[str] = None
    
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    correlation_id: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: float = 60.0
    
    @classmethod
    def create(
        cls,
        sender: str,
        recipient: Optional[str],
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: float = 60.0,
    ) -> "Message":
        """Create a new message.
        
        Args:
            sender: Sender ID (component or agent)
            recipient: Recipient ID (None for broadcast)
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            correlation_id: For request-response pairing
            metadata: Additional metadata
            ttl: Time-to-live in seconds
            
        Returns:
            New Message instance
        """
        return cls(
            id=str(uuid.uuid4()),
            type=message_type,
            priority=priority,
            sender=sender,
            recipient=recipient,
            payload=payload,
            metadata=metadata or {},
            correlation_id=correlation_id,
            ttl=ttl,
        )
    
    @classmethod
    def request(
        cls,
        sender: str,
        recipient: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> "Message":
        """Create a request message.
        
        Args:
            sender: Sender ID
            recipient: Recipient ID
            payload: Request payload
            priority: Message priority
            
        Returns:
            Request Message
        """
        return cls.create(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.COMPONENT_REQUEST,
            payload=payload,
            priority=priority,
        )
    
    @classmethod
    def response(
        cls,
        sender: str,
        recipient: str,
        payload: Dict[str, Any],
        correlation_id: str,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> "Message":
        """Create a response message.
        
        Args:
            sender: Sender ID
            recipient: Recipient ID
            payload: Response payload
            correlation_id: ID of the request being responded to
            priority: Message priority
            
        Returns:
            Response Message
        """
        return cls.create(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.COMPONENT_RESPONSE,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
        )
    
    @classmethod
    def broadcast(
        cls,
        sender: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> "Message":
        """Create a broadcast message.
        
        Args:
            sender: Sender ID
            payload: Broadcast payload
            priority: Message priority
            
        Returns:
            Broadcast Message
        """
        return cls.create(
            sender=sender,
            recipient=None,
            message_type=MessageType.BROADCAST,
            payload=payload,
            priority=priority,
        )
    
    @classmethod
    def agent_task(
        cls,
        sender: str,
        recipient: str,
        task_type: str,
        task_data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> "Message":
        """Create an agent task message.
        
        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            task_type: Type of task
            task_data: Task data
            priority: Message priority
            
        Returns:
            Agent task Message
        """
        return cls.create(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.AGENT_TASK,
            payload={"task_type": task_type, "task_data": task_data},
            priority=priority,
        )
    
    @classmethod
    def agent_result(
        cls,
        sender: str,
        recipient: str,
        result: Dict[str, Any],
        correlation_id: str,
        success: bool = True,
    ) -> "Message":
        """Create an agent result message.
        
        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            result: Result data
            correlation_id: ID of the task being responded to
            success: Whether the task succeeded
            
        Returns:
            Agent result Message
        """
        return cls.create(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.AGENT_RESULT,
            payload={"success": success, "result": result},
            correlation_id=correlation_id,
        )
    
    @classmethod
    def error(
        cls,
        sender: str,
        recipient: Optional[str],
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> "Message":
        """Create an error message.
        
        Args:
            sender: Sender ID
            recipient: Recipient ID (None for broadcast error)
            error_message: Error message
            error_details: Error details
            correlation_id: ID of the message that caused the error
            
        Returns:
            Error Message
        """
        return cls.create(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.ERROR,
            payload={"error": error_message, "details": error_details or {}},
            priority=MessagePriority.HIGH,
            correlation_id=correlation_id,
        )
    
    def is_expired(self) -> bool:
        """Check if the message has expired.
        
        Returns:
            True if message has exceeded TTL
        """
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.ttl
    
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message.
        
        Returns:
            True if recipient is None
        """
        return self.recipient is None or self.type == MessageType.BROADCAST
    
    def create_response(
        self,
        sender: str,
        payload: Dict[str, Any],
        message_type: Optional[MessageType] = None,
    ) -> "Message":
        """Create a response to this message.
        
        Args:
            sender: Response sender ID
            payload: Response payload
            message_type: Optional override for response type
            
        Returns:
            Response Message
        """
        if message_type is None:
            if self.type.is_agent_message:
                message_type = MessageType.AGENT_RESPONSE
            else:
                message_type = MessageType.COMPONENT_RESPONSE
        
        return Message.create(
            sender=sender,
            recipient=self.sender,
            message_type=message_type,
            payload=payload,
            correlation_id=self.id,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "type": self.type.name,
            "priority": self.priority.name,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Message instance
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType[data.get("type", "COMPONENT_REQUEST")],
            priority=MessagePriority[data.get("priority", "NORMAL")],
            sender=data.get("sender", ""),
            recipient=data.get("recipient"),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            correlation_id=data.get("correlation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            ttl=data.get("ttl", 60.0),
        )
    
    def __repr__(self) -> str:
        """String representation."""
        recipient = self.recipient or "BROADCAST"
        return f"Message(id={self.id[:8]}, type={self.type.name}, {self.sender}->{recipient})"
