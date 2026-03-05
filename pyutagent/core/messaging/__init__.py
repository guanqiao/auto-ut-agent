"""Unified Messaging Module.

This module provides:
- Message: Unified message definition
- UnifiedMessageBus: Unified message bus
- MessageRouter: Message routing
- Adapters: Backward compatibility adapters (deprecated)
"""

from .message import Message, MessageType, MessagePriority
from .bus import UnifiedMessageBus
from .router import MessageRouter

# Backward compatibility adapters (deprecated)
from .adapters import (
    EventBusAdapter,
    AsyncEventBusAdapter,
    MessageBusAdapter,
    Subscription,
    create_event_bus_adapter,
    create_message_bus_adapter,
)

__all__ = [
    # Core components
    "Message",
    "MessageType",
    "MessagePriority",
    "UnifiedMessageBus",
    "MessageRouter",
    # Adapters (deprecated, for backward compatibility)
    "EventBusAdapter",
    "AsyncEventBusAdapter",
    "MessageBusAdapter",
    "Subscription",
    "create_event_bus_adapter",
    "create_message_bus_adapter",
]
