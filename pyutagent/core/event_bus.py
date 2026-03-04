"""事件总线 - 实现组件解耦"""
from typing import Any, Callable, Dict, List, Type, TypeVar
from dataclasses import dataclass, field
import uuid
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Subscription:
    """订阅"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Type[Any] = None
    handler: Callable[[Any], None] = None


class EventBus:
    """同步事件总线"""
    
    def __init__(self):
        self._subscriptions: Dict[Type[Any], List[Subscription]] = {}
    
    def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], None]
    ) -> Subscription:
        """订阅事件"""
        subscription = Subscription(event_type=event_type, handler=handler)
        
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []
        
        self._subscriptions[event_type].append(subscription)
        return subscription
    
    def unsubscribe(self, subscription: Subscription):
        """取消订阅"""
        event_type = subscription.event_type
        if event_type in self._subscriptions:
            self._subscriptions[event_type] = [
                s for s in self._subscriptions[event_type]
                if s.id != subscription.id
            ]
    
    def publish(self, event: Any):
        """发布事件"""
        event_type = type(event)
        
        if event_type in self._subscriptions:
            for subscription in self._subscriptions[event_type]:
                try:
                    subscription.handler(event)
                except Exception as e:
                    # 记录错误但不影响其他处理器
                    logger.error(f"Event handler error: {e}")


class AsyncEventBus:
    """异步事件总线"""
    
    def __init__(self):
        self._subscriptions: Dict[Type[Any], List[Subscription]] = {}
    
    def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], Any]  # 可以是同步或异步函数
    ) -> Subscription:
        """订阅事件"""
        subscription = Subscription(event_type=event_type, handler=handler)
        
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []
        
        self._subscriptions[event_type].append(subscription)
        return subscription
    
    def unsubscribe(self, subscription: Subscription):
        """取消订阅"""
        event_type = subscription.event_type
        if event_type in self._subscriptions:
            self._subscriptions[event_type] = [
                s for s in self._subscriptions[event_type]
                if s.id != subscription.id
            ]
    
    async def publish(self, event: Any):
        """异步发布事件"""
        event_type = type(event)
        
        if event_type in self._subscriptions:
            tasks = []
            for subscription in self._subscriptions[event_type]:
                task = self._invoke_handler(subscription.handler, event)
                tasks.append(task)
            
            # 并发执行所有处理器
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 记录错误
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Async event handler error: {result}")
    
    async def _invoke_handler(
        self,
        handler: Callable,
        event: Any
    ) -> Any:
        """调用处理器（支持同步和异步）"""
        if asyncio.iscoroutinefunction(handler):
            return await handler(event)
        else:
            return handler(event)
