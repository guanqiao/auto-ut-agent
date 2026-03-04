"""测试事件总线基础功能"""
import pytest
import asyncio
from pyutagent.core.event_bus import EventBus, AsyncEventBus


class TestEventBusBasic:
    """同步事件总线基础测试"""
    
    def test_create_event_bus(self):
        """测试创建事件总线"""
        bus = EventBus()
        assert bus is not None
    
    def test_subscribe_and_publish(self):
        """测试订阅和发布"""
        bus = EventBus()
        received_events = []
        
        # 定义事件处理器
        def handler(event):
            received_events.append(event)
        
        # 订阅事件
        bus.subscribe(str, handler)
        
        # 发布事件
        test_event = "test_message"
        bus.publish(test_event)
        
        # 验证
        assert len(received_events) == 1
        assert received_events[0] == test_event
    
    def test_multiple_subscribers(self):
        """测试多个订阅者"""
        bus = EventBus()
        received_by_handler1 = []
        received_by_handler2 = []
        
        def handler1(event):
            received_by_handler1.append(event)
        
        def handler2(event):
            received_by_handler2.append(event)
        
        bus.subscribe(str, handler1)
        bus.subscribe(str, handler2)
        
        bus.publish("test")
        
        assert len(received_by_handler1) == 1
        assert len(received_by_handler2) == 1
    
    def test_unsubscribe(self):
        """测试取消订阅"""
        bus = EventBus()
        received = []
        
        def handler(event):
            received.append(event)
        
        subscription = bus.subscribe(str, handler)
        bus.unsubscribe(subscription)
        bus.publish("test")
        
        assert len(received) == 0
    
    def test_publish_without_subscribers(self):
        """测试没有订阅者时发布事件"""
        bus = EventBus()
        # 不应该抛出异常
        bus.publish("test")


class TestAsyncEventBus:
    """异步事件总线测试"""
    
    @pytest.mark.asyncio
    async def test_async_subscribe_and_publish(self):
        """测试异步订阅和发布"""
        bus = AsyncEventBus()
        received_events = []
        
        async def handler(event: str):
            received_events.append(event)
        
        bus.subscribe(str, handler)
        await bus.publish("test")
        
        assert len(received_events) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_publish(self):
        """测试并发发布"""
        bus = AsyncEventBus()
        received = []
        
        async def handler(event: str):
            await asyncio.sleep(0.01)
            received.append(event)
        
        bus.subscribe(str, handler)
        
        # 并发发布 10 个事件
        tasks = [bus.publish(f"event_{i}") for i in range(10)]
        await asyncio.gather(*tasks)
        
        assert len(received) == 10
    
    @pytest.mark.asyncio
    async def test_handler_error_isolation(self):
        """测试处理器错误隔离"""
        bus = AsyncEventBus()
        successful_handler_called = False
        
        async def failing_handler(event: str):
            raise ValueError("Test error")
        
        async def successful_handler(event: str):
            nonlocal successful_handler_called
            successful_handler_called = True
        
        bus.subscribe(str, failing_handler)
        bus.subscribe(str, successful_handler)
        
        # 不应该因为一个处理器失败而影响其他处理器
        await bus.publish("test")
        assert successful_handler_called
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_handlers(self):
        """测试混合同步和异步处理器"""
        bus = AsyncEventBus()
        sync_called = False
        async_called = False
        
        def sync_handler(event: str):
            nonlocal sync_called
            sync_called = True
        
        async def async_handler(event: str):
            nonlocal async_called
            async_called = True
        
        bus.subscribe(str, sync_handler)
        bus.subscribe(str, async_handler)
        
        await bus.publish("test")
        
        assert sync_called
        assert async_called
    
    @pytest.mark.asyncio
    async def test_async_unsubscribe(self):
        """测试异步取消订阅"""
        bus = AsyncEventBus()
        received = []
        
        async def handler(event: str):
            received.append(event)
        
        subscription = bus.subscribe(str, handler)
        bus.unsubscribe(subscription)
        await bus.publish("test")
        
        assert len(received) == 0
