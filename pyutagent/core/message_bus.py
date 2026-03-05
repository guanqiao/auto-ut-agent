"""组件通信优化 - 消息总线和消息队列

.. deprecated::
    Use pyutagent.core.messaging.UnifiedMessageBus instead.
    This module is kept for backward compatibility.
"""
import warnings
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from collections import deque

logger = logging.getLogger(__name__)

# Emit deprecation warning when module is imported
warnings.warn(
    "pyutagent.core.message_bus is deprecated. "
    "Use pyutagent.core.messaging.UnifiedMessageBus instead.",
    DeprecationWarning,
    stacklevel=2
)


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageType(Enum):
    """消息类型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """消息定义"""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    sender: str
    recipients: List[str]
    content: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    correlation_id: Optional[str] = None
    ttl: float = 60.0  # 生存时间（秒）
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        current_time = asyncio.get_event_loop().time()
        return (current_time - self.timestamp) > self.ttl


class MessageQueue:
    """消息队列"""
    
    def __init__(self, max_size: int = 1000):
        self._queue: deque[Message] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def put(self, message: Message):
        """添加消息到队列"""
        async with self._lock:
            self._queue.append(message)
    
    async def get(self) -> Optional[Message]:
        """从队列获取消息"""
        async with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None
    
    async def get_by_priority(self, min_priority: MessagePriority) -> Optional[Message]:
        """按优先级获取消息"""
        async with self._lock:
            for i, message in enumerate(self._queue):
                if message.priority.value >= min_priority.value:
                    del self._queue[i]
                    return message
            return None
    
    def size(self) -> int:
        """获取队列大小"""
        return len(self._queue)
    
    def clear(self):
        """清空队列"""
        self._queue.clear()


class MessageBus:
    """消息总线"""
    
    def __init__(self):
        self._queues: Dict[str, MessageQueue] = {}
        self._subscribers: Dict[str, List[Callable[[Message], Any]]] = {}
        self._lock = asyncio.Lock()
    
    def create_queue(self, queue_name: str, max_size: int = 1000):
        """创建消息队列"""
        self._queues[queue_name] = MessageQueue(max_size)
    
    async def publish(self, queue_name: str, message: Message):
        """发布消息到队列"""
        if queue_name not in self._queues:
            self.create_queue(queue_name)
        
        await self._queues[queue_name].put(message)
        
        # 通知订阅者
        await self._notify_subscribers(queue_name, message)
    
    async def subscribe(
        self,
        queue_name: str,
        callback: Callable[[Message], Any]
    ):
        """订阅队列消息"""
        async with self._lock:
            if queue_name not in self._subscribers:
                self._subscribers[queue_name] = []
            self._subscribers[queue_name].append(callback)
    
    async def _notify_subscribers(self, queue_name: str, message: Message):
        """通知订阅者"""
        if queue_name in self._subscribers:
            for callback in self._subscribers[queue_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")
    
    async def consume(self, queue_name: str) -> Optional[Message]:
        """消费队列消息"""
        if queue_name not in self._queues:
            return None
        
        return await self._queues[queue_name].get()


class ComponentRouter:
    """组件路由器"""
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self._routes: Dict[str, str] = {}  # component_id -> queue_name
    
    def register_component(self, component_id: str, queue_name: str):
        """注册组件路由"""
        self._routes[component_id] = queue_name
    
    def get_queue(self, component_id: str) -> Optional[str]:
        """获取组件队列"""
        return self._routes.get(component_id)
    
    async def route_message(self, message: Message):
        """路由消息到目标组件"""
        for recipient in message.recipients:
            queue_name = self.get_queue(recipient)
            if queue_name:
                await self.message_bus.publish(queue_name, message)
            else:
                logger.warning(f"No route found for component: {recipient}")
