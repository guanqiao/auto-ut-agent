"""测试消息总线和消息队列"""
import pytest
import asyncio
from pyutagent.core.message_bus import (
    MessagePriority,
    MessageType,
    Message,
    MessageQueue,
    MessageBus,
    ComponentRouter
)


class TestMessage:
    """测试消息定义"""
    
    def test_message_creation(self):
        """测试创建消息"""
        message = Message(
            message_id="msg_001",
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            sender="component_a",
            recipients=["component_b"],
            content={"action": "test"}
        )
        
        assert message.message_id == "msg_001"
        assert message.message_type == MessageType.REQUEST
        assert message.priority == MessagePriority.NORMAL
        assert len(message.recipients) == 1
    
    def test_message_with_correlation(self):
        """测试带关联 ID 的消息"""
        message = Message(
            message_id="msg_002",
            message_type=MessageType.RESPONSE,
            priority=MessagePriority.HIGH,
            sender="component_b",
            recipients=["component_a"],
            content={"result": "success"},
            correlation_id="msg_001"
        )
        
        assert message.correlation_id == "msg_001"
        assert message.priority == MessagePriority.HIGH
    
    def test_message_expiration(self):
        """测试消息过期"""
        message = Message(
            message_id="msg_003",
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.LOW,
            sender="system",
            recipients=["all"],
            content={},
            ttl=0.1  # 100ms 过期
        )
        
        assert message.is_expired() is False
        
        # 等待过期
        import time
        time.sleep(0.2)
        assert message.is_expired() is True


class TestMessageQueue:
    """测试消息队列"""
    
    @pytest.mark.asyncio
    async def test_message_queue_basic(self):
        """测试消息队列基本操作"""
        queue = MessageQueue(max_size=100)
        
        message = Message(
            message_id="msg_001",
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            sender="test",
            recipients=["receiver"],
            content={}
        )
        
        await queue.put(message)
        assert queue.size() == 1
        
        retrieved = await queue.get()
        assert retrieved == message
        assert queue.size() == 0
    
    @pytest.mark.asyncio
    async def test_message_queue_fifo(self):
        """测试 FIFO 顺序"""
        queue = MessageQueue(max_size=100)
        
        messages = [
            Message(f"msg_{i}", MessageType.REQUEST, MessagePriority.NORMAL,
                   "sender", ["receiver"], {})
            for i in range(5)
        ]
        
        for msg in messages:
            await queue.put(msg)
        
        # 验证 FIFO
        for i in range(5):
            retrieved = await queue.get()
            assert retrieved.message_id == f"msg_{i}"
    
    @pytest.mark.asyncio
    async def test_message_queue_priority(self):
        """测试优先级获取"""
        queue = MessageQueue(max_size=100)
        
        low_priority = Message("low", MessageType.REQUEST, MessagePriority.LOW,
                              "sender", ["receiver"], {})
        high_priority = Message("high", MessageType.REQUEST, MessagePriority.HIGH,
                               "sender", ["receiver"], {})
        
        await queue.put(low_priority)
        await queue.put(high_priority)
        
        # 按优先级获取
        retrieved = await queue.get_by_priority(MessagePriority.HIGH)
        assert retrieved == high_priority
    
    @pytest.mark.asyncio
    async def test_message_queue_max_size(self):
        """测试最大容量限制"""
        queue = MessageQueue(max_size=3)
        
        # 添加超过容量的消息
        for i in range(5):
            msg = Message(f"msg_{i}", MessageType.REQUEST, MessagePriority.NORMAL,
                         "sender", ["receiver"], {})
            await queue.put(msg)
        
        # 应该只保留最近的 3 个
        assert queue.size() == 3


class TestMessageBus:
    """测试消息总线"""
    
    @pytest.mark.asyncio
    async def test_message_bus_publish_subscribe(self):
        """测试发布订阅"""
        bus = MessageBus()
        received_messages = []
        
        async def subscriber(message: Message):
            received_messages.append(message)
        
        await bus.subscribe("test_queue", subscriber)
        
        message = Message(
            message_id="msg_001",
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            sender="test",
            recipients=["receiver"],
            content={}
        )
        
        await bus.publish("test_queue", message)
        
        assert len(received_messages) == 1
        assert received_messages[0] == message
    
    @pytest.mark.asyncio
    async def test_message_bus_multiple_subscribers(self):
        """测试多个订阅者"""
        bus = MessageBus()
        received_by_a = []
        received_by_b = []
        
        async def subscriber_a(message: Message):
            received_by_a.append(message)
        
        async def subscriber_b(message: Message):
            received_by_b.append(message)
        
        await bus.subscribe("queue", subscriber_a)
        await bus.subscribe("queue", subscriber_b)
        
        message = Message("msg", MessageType.REQUEST, MessagePriority.NORMAL,
                         "sender", ["receiver"], {})
        
        await bus.publish("queue", message)
        
        assert len(received_by_a) == 1
        assert len(received_by_b) == 1
    
    @pytest.mark.asyncio
    async def test_message_bus_consume(self):
        """测试消费消息"""
        bus = MessageBus()
        
        message = Message("msg", MessageType.REQUEST, MessagePriority.NORMAL,
                         "sender", ["receiver"], {})
        
        await bus.publish("queue", message)
        consumed = await bus.consume("queue")
        
        assert consumed == message
        
        # 再次消费应该为空
        consumed_again = await bus.consume("queue")
        assert consumed_again is None
    
    @pytest.mark.asyncio
    async def test_message_bus_error_handling(self):
        """测试错误处理"""
        bus = MessageBus()
        
        async def failing_subscriber(message: Message):
            raise ValueError("Test error")
        
        async def normal_subscriber(message: Message):
            pass
        
        await bus.subscribe("queue", failing_subscriber)
        await bus.subscribe("queue", normal_subscriber)
        
        # 不应该因为一个订阅者失败而影响其他
        message = Message("msg", MessageType.REQUEST, MessagePriority.NORMAL,
                         "sender", ["receiver"], {})
        
        await bus.publish("queue", message)  # 不应该抛出异常


class TestComponentRouter:
    """测试组件路由器"""
    
    @pytest.mark.asyncio
    async def test_component_router_registration(self):
        """测试组件注册"""
        bus = MessageBus()
        router = ComponentRouter(bus)
        
        router.register_component("component_a", "queue_a")
        router.register_component("component_b", "queue_b")
        
        assert router.get_queue("component_a") == "queue_a"
        assert router.get_queue("component_b") == "queue_b"
    
    @pytest.mark.asyncio
    async def test_component_router_routing(self):
        """测试消息路由"""
        bus = MessageBus()
        router = ComponentRouter(bus)
        
        router.register_component("receiver", "receiver_queue")
        
        received = []
        async def receiver_callback(message: Message):
            received.append(message)
        
        await bus.subscribe("receiver_queue", receiver_callback)
        
        message = Message(
            message_id="msg_001",
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            sender="sender",
            recipients=["receiver"],
            content={}
        )
        
        await router.route_message(message)
        
        assert len(received) == 1
    
    @pytest.mark.asyncio
    async def test_component_router_unknown_component(self):
        """测试未知组件路由"""
        bus = MessageBus()
        router = ComponentRouter(bus)
        
        message = Message(
            message_id="msg_001",
            message_type=MessageType.REQUEST,
            priority=MessagePriority.NORMAL,
            sender="sender",
            recipients=["unknown_component"],
            content={}
        )
        
        # 不应该抛出异常
        await router.route_message(message)
