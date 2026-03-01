"""Conversation management for Agent interaction."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message role enumeration."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """Conversation message."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


class ConversationManager:
    """Manages conversation history between user and agent.
    
    Features:
    - Store and retrieve conversation messages
    - Limit context window for LLM
    - Export/import conversations
    - Callback support for real-time updates
    """
    
    def __init__(self, max_history: int = 50):
        """Initialize conversation manager.
        
        Args:
            max_history: Maximum number of messages to keep
        """
        self.messages: List[Message] = []
        self.max_history = max_history
        self.on_message: Optional[Callable[[Message], None]] = None
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Add a message to the conversation.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created message
        """
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(msg)
        
        # Limit history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        # Trigger callback
        if self.on_message:
            self.on_message(msg)
        
        return msg
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add user message."""
        return self.add_message(MessageRole.USER, content, metadata)
    
    def add_agent_message(self, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add agent message."""
        return self.add_message(MessageRole.AGENT, content, metadata)
    
    def add_system_message(self, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add system message."""
        return self.add_message(MessageRole.SYSTEM, content, metadata)
    
    def add_tool_message(self, content: str, tool_name: str, metadata: Optional[Dict] = None) -> Message:
        """Add tool execution message."""
        meta = metadata or {}
        meta["tool_name"] = tool_name
        return self.add_message(MessageRole.TOOL, content, meta)
    
    def get_context(self, n: Optional[int] = None) -> List[Message]:
        """Get recent messages for context.
        
        Args:
            n: Number of messages to return (default: all)
            
        Returns:
            List of messages
        """
        if n is None:
            return self.messages.copy()
        return self.messages[-n:]
    
    def get_context_as_text(self, n: Optional[int] = None) -> str:
        """Get context as formatted text.
        
        Args:
            n: Number of messages to include
            
        Returns:
            Formatted conversation text
        """
        messages = self.get_context(n)
        lines = []
        for msg in messages:
            role_label = {
                MessageRole.USER: "User",
                MessageRole.AGENT: "Agent",
                MessageRole.SYSTEM: "System",
                MessageRole.TOOL: f"Tool({msg.metadata.get('tool_name', 'unknown')})",
            }.get(msg.role, msg.role.value)
            lines.append(f"[{role_label}]: {msg.content}")
        return "\n".join(lines)
    
    def to_llm_messages(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Convert to LangChain message format.
        
        Args:
            n: Number of messages to include
            
        Returns:
            List of message dictionaries
        """
        messages = self.get_context(n)
        return [
            {
                "role": msg.role.value,
                "content": msg.content,
            }
            for msg in messages
        ]
    
    def clear(self):
        """Clear all messages."""
        self.messages.clear()
    
    def export_to_file(self, filepath: str):
        """Export conversation to file.
        
        Args:
            filepath: Output file path
        """
        data = {
            "session_id": self.session_id,
            "export_time": datetime.now().isoformat(),
            "messages": [msg.to_dict() for msg in self.messages],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def import_from_file(self, filepath: str):
        """Import conversation from file.
        
        Args:
            filepath: Input file path
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.session_id = data.get("session_id", self.session_id)
        self.messages = [
            Message.from_dict(msg_data)
            for msg_data in data.get("messages", [])
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get conversation summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "user_messages": len([m for m in self.messages if m.role == MessageRole.USER]),
            "agent_messages": len([m for m in self.messages if m.role == MessageRole.AGENT]),
            "tool_calls": len([m for m in self.messages if m.role == MessageRole.TOOL]),
        }
