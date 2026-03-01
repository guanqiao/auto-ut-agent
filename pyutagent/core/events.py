"""Event system for PyUT Agent.

This module provides a structured event system for decoupled communication
between components, replacing simple callback functions.

Features:
- Type-safe event definitions
- Subscriber pattern for loose coupling
- Event filtering and routing
- Async event handling support
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
    Generic,
)
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EventType(Enum):
    """Types of events in the system."""
    STATE_CHANGE = auto()
    PROGRESS = auto()
    ERROR = auto()
    COVERAGE_UPDATE = auto()
    TEST_RESULT = auto()
    COMPILATION_RESULT = auto()
    GENERATION_COMPLETE = auto()
    RECOVERY_ACTION = auto()
    LOG = auto()
    CUSTOM = auto()


@dataclass
class AgentEvent(Generic[T]):
    """Structured event for agent communication.
    
    Attributes:
        type: The type of event
        data: Event payload
        timestamp: When the event occurred
        source: Source component that emitted the event
        metadata: Additional event metadata
    """
    type: EventType
    data: T
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "type": self.type.name,
            "data": self.data if not isinstance(self.data, (dict, list, str, int, float, bool, type(None))) else self.data,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


EventHandler = Callable[[AgentEvent], None]
AsyncEventHandler = Callable[[AgentEvent], Any]


class EventEmitter:
    """Event emitter for publishing events.
    
    Provides methods for emitting events and managing subscribers.
    """
    
    def __init__(self, name: str = "default"):
        """Initialize event emitter.
        
        Args:
            name: Name of the emitter for logging
        """
        self.name = name
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._async_subscribers: Dict[EventType, List[AsyncEventHandler]] = {}
        self._wildcard_subscribers: List[EventHandler] = []
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
    
    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler
    ) -> None:
        """Subscribe to a specific event type.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Function to call when event is emitted
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"[EventEmitter:{self.name}] Subscribed to {event_type.name}")
    
    def subscribe_async(
        self,
        event_type: EventType,
        handler: AsyncEventHandler
    ) -> None:
        """Subscribe to a specific event type with async handler.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async function to call when event is emitted
        """
        if event_type not in self._async_subscribers:
            self._async_subscribers[event_type] = []
        self._async_subscribers[event_type].append(handler)
        logger.debug(f"[EventEmitter:{self.name}] Subscribed async to {event_type.name}")
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe to all events.
        
        Args:
            handler: Function to call for any event
        """
        self._wildcard_subscribers.append(handler)
        logger.debug(f"[EventEmitter:{self.name}] Subscribed to all events")
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler
    ) -> bool:
        """Unsubscribe from a specific event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
            
        Returns:
            True if handler was removed
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.debug(f"[EventEmitter:{self.name}] Unsubscribed from {event_type.name}")
                return True
            except ValueError:
                pass
        return False
    
    def emit(self, event: AgentEvent) -> None:
        """Emit an event to all subscribers.
        
        Args:
            event: The event to emit
        """
        logger.debug(f"[EventEmitter:{self.name}] Emitting {event.type.name} from {event.source}")
        
        handlers = self._subscribers.get(event.type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"[EventEmitter:{self.name}] Handler error: {e}")
        
        for handler in self._wildcard_subscribers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"[EventEmitter:{self.name}] Wildcard handler error: {e}")
    
    async def emit_async(self, event: AgentEvent) -> None:
        """Emit an event to all subscribers (async).
        
        Args:
            event: The event to emit
        """
        logger.debug(f"[EventEmitter:{self.name}] Emitting async {event.type.name} from {event.source}")
        
        handlers = self._subscribers.get(event.type, [])
        async_handlers = self._async_subscribers.get(event.type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"[EventEmitter:{self.name}] Handler error: {e}")
        
        for handler in async_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"[EventEmitter:{self.name}] Async handler error: {e}")
        
        for handler in self._wildcard_subscribers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"[EventEmitter:{self.name}] Wildcard handler error: {e}")
    
    def emit_state_change(
        self,
        old_state: str,
        new_state: str,
        message: str = "",
        source: str = ""
    ) -> None:
        """Emit a state change event.
        
        Args:
            old_state: Previous state
            new_state: New state
            message: Optional message
            source: Source component
        """
        event = AgentEvent(
            type=EventType.STATE_CHANGE,
            data={
                "old_state": old_state,
                "new_state": new_state,
                "message": message,
            },
            source=source,
        )
        self.emit(event)
    
    def emit_progress(
        self,
        progress: float,
        message: str = "",
        source: str = ""
    ) -> None:
        """Emit a progress event.
        
        Args:
            progress: Progress value (0.0 to 1.0)
            message: Optional message
            source: Source component
        """
        event = AgentEvent(
            type=EventType.PROGRESS,
            data={
                "progress": progress,
                "message": message,
            },
            source=source,
        )
        self.emit(event)
    
    def emit_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        source: str = ""
    ) -> None:
        """Emit an error event.
        
        Args:
            error: The error that occurred
            context: Optional context information
            source: Source component
        """
        event = AgentEvent(
            type=EventType.ERROR,
            data={
                "error": str(error),
                "error_type": type(error).__name__,
                "context": context or {},
            },
            source=source,
        )
        self.emit(event)
    
    def emit_coverage_update(
        self,
        line_coverage: float,
        branch_coverage: float,
        source: str = ""
    ) -> None:
        """Emit a coverage update event.
        
        Args:
            line_coverage: Line coverage percentage
            branch_coverage: Branch coverage percentage
            source: Source component
        """
        event = AgentEvent(
            type=EventType.COVERAGE_UPDATE,
            data={
                "line_coverage": line_coverage,
                "branch_coverage": branch_coverage,
            },
            source=source,
        )
        self.emit(event)
    
    def clear_subscribers(self) -> None:
        """Clear all subscribers."""
        self._subscribers.clear()
        self._async_subscribers.clear()
        self._wildcard_subscribers.clear()
        logger.info(f"[EventEmitter:{self.name}] Cleared all subscribers")
    
    def get_subscriber_count(self, event_type: Optional[EventType] = None) -> int:
        """Get count of subscribers.
        
        Args:
            event_type: Optional specific event type to count
            
        Returns:
            Number of subscribers
        """
        if event_type:
            return len(self._subscribers.get(event_type, []))
        total = sum(len(handlers) for handlers in self._subscribers.values())
        total += len(self._wildcard_subscribers)
        return total


class EventSubscriber:
    """Base class for event subscribers.
    
    Provides convenient methods for subscribing to events
    and handling them in a structured way.
    """
    
    def __init__(self, emitter: EventEmitter):
        """Initialize subscriber.
        
        Args:
            emitter: The event emitter to subscribe to
        """
        self.emitter = emitter
        self._subscriptions: List[tuple] = []
    
    def on(self, event_type: EventType, handler: Optional[EventHandler] = None) -> Callable:
        """Subscribe to an event type.
        
        Can be used as a decorator or method call.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Optional handler function
            
        Returns:
            Decorator function or None
        """
        def decorator(func: EventHandler) -> EventHandler:
            self.emitter.subscribe(event_type, func)
            self._subscriptions.append((event_type, func))
            return func
        
        if handler:
            return decorator(handler)
        return decorator
    
    def on_async(self, event_type: EventType, handler: Optional[AsyncEventHandler] = None) -> Callable:
        """Subscribe to an event type with async handler.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Optional async handler function
            
        Returns:
            Decorator function or None
        """
        def decorator(func: AsyncEventHandler) -> AsyncEventHandler:
            self.emitter.subscribe_async(event_type, func)
            self._subscriptions.append((event_type, func))
            return func
        
        if handler:
            return decorator(handler)
        return decorator
    
    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events."""
        for event_type, handler in self._subscriptions:
            self.emitter.unsubscribe(event_type, handler)
        self._subscriptions.clear()


_global_emitter: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Get the global event emitter.
    
    Returns:
        The global EventEmitter instance
    """
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter("global")
        logger.info("[EventSystem] Created global event emitter")
    return _global_emitter


def reset_event_emitter() -> None:
    """Reset the global event emitter."""
    global _global_emitter
    if _global_emitter is not None:
        _global_emitter.clear_subscribers()
    _global_emitter = None
    logger.info("[EventSystem] Reset global event emitter")
