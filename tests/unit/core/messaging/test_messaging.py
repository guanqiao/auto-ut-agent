"""Tests for Unified Messaging Module."""

import pytest
import asyncio
from datetime import datetime, timedelta

from pyutagent.core.messaging import (
    Message,
    MessageType,
    MessagePriority,
    UnifiedMessageBus,
    MessageRouter,
)
from pyutagent.core.messaging.router import RoutingRule


class TestMessageType:
    """Tests for MessageType enum."""
    
    def test_is_request(self):
        """Test request type detection."""
        assert MessageType.COMPONENT_REQUEST.is_request
        assert MessageType.AGENT_TASK.is_request
        assert MessageType.AGENT_QUERY.is_request
        assert not MessageType.COMPONENT_RESPONSE.is_request
    
    def test_is_response(self):
        """Test response type detection."""
        assert MessageType.COMPONENT_RESPONSE.is_response
        assert MessageType.AGENT_RESULT.is_response
        assert MessageType.AGENT_RESPONSE.is_response
        assert not MessageType.COMPONENT_REQUEST.is_response
    
    def test_is_agent_message(self):
        """Test agent message detection."""
        assert MessageType.AGENT_TASK.is_agent_message
        assert MessageType.AGENT_RESULT.is_agent_message
        assert MessageType.AGENT_COORDINATION.is_agent_message
        assert not MessageType.COMPONENT_REQUEST.is_agent_message
    
    def test_is_component_message(self):
        """Test component message detection."""
        assert MessageType.COMPONENT_REQUEST.is_component_message
        assert MessageType.COMPONENT_RESPONSE.is_component_message
        assert MessageType.COMPONENT_NOTIFICATION.is_component_message
        assert not MessageType.AGENT_TASK.is_component_message


class TestMessage:
    """Tests for Message class."""
    
    def test_create(self):
        """Test message creation."""
        message = Message.create(
            sender="sender1",
            recipient="recipient1",
            message_type=MessageType.COMPONENT_REQUEST,
            payload={"key": "value"},
        )
        
        assert message.sender == "sender1"
        assert message.recipient == "recipient1"
        assert message.type == MessageType.COMPONENT_REQUEST
        assert message.payload["key"] == "value"
        assert message.id is not None
    
    def test_request_factory(self):
        """Test request factory method."""
        message = Message.request(
            sender="sender",
            recipient="recipient",
            payload={"action": "test"},
        )
        
        assert message.type == MessageType.COMPONENT_REQUEST
        assert message.recipient == "recipient"
    
    def test_response_factory(self):
        """Test response factory method."""
        message = Message.response(
            sender="sender",
            recipient="recipient",
            payload={"result": "success"},
            correlation_id="corr-123",
        )
        
        assert message.type == MessageType.COMPONENT_RESPONSE
        assert message.correlation_id == "corr-123"
    
    def test_broadcast_factory(self):
        """Test broadcast factory method."""
        message = Message.broadcast(
            sender="sender",
            payload={"event": "update"},
        )
        
        assert message.type == MessageType.BROADCAST
        assert message.recipient is None
        assert message.is_broadcast()
    
    def test_agent_task_factory(self):
        """Test agent task factory method."""
        message = Message.agent_task(
            sender="coordinator",
            recipient="worker",
            task_type="generate_tests",
            task_data={"file": "test.java"},
        )
        
        assert message.type == MessageType.AGENT_TASK
        assert message.payload["task_type"] == "generate_tests"
    
    def test_agent_result_factory(self):
        """Test agent result factory method."""
        message = Message.agent_result(
            sender="worker",
            recipient="coordinator",
            result={"coverage": 0.8},
            correlation_id="task-123",
            success=True,
        )
        
        assert message.type == MessageType.AGENT_RESULT
        assert message.payload["success"] is True
        assert message.correlation_id == "task-123"
    
    def test_error_factory(self):
        """Test error factory method."""
        message = Message.error(
            sender="component",
            recipient="handler",
            error_message="Test error",
            error_details={"code": 500},
        )
        
        assert message.type == MessageType.ERROR
        assert message.priority == MessagePriority.HIGH
        assert message.payload["error"] == "Test error"
    
    def test_is_expired(self):
        """Test message expiration check."""
        message = Message(ttl=1.0)
        assert not message.is_expired()
        
        message.timestamp = datetime.now() - timedelta(seconds=2)
        assert message.is_expired()
    
    def test_create_response(self):
        """Test creating a response to a message."""
        original = Message.request(
            sender="sender",
            recipient="recipient",
            payload={"request": "data"},
        )
        
        response = original.create_response(
            sender="recipient",
            payload={"response": "data"},
        )
        
        assert response.type == MessageType.COMPONENT_RESPONSE
        assert response.correlation_id == original.id
        assert response.recipient == original.sender
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = Message.create(
            sender="sender",
            recipient="recipient",
            message_type=MessageType.AGENT_TASK,
            payload={"key": "value"},
            priority=MessagePriority.HIGH,
        )
        
        data = original.to_dict()
        restored = Message.from_dict(data)
        
        assert restored.id == original.id
        assert restored.sender == original.sender
        assert restored.type == original.type
        assert restored.priority == original.priority
        assert restored.payload == original.payload


class TestUnifiedMessageBus:
    """Tests for UnifiedMessageBus."""
    
    @pytest.fixture
    async def bus(self):
        """Create a message bus."""
        bus = UnifiedMessageBus()
        yield bus
        await bus.clear()
    
    @pytest.mark.asyncio
    async def test_register_unregister(self, bus):
        """Test entity registration and unregistration."""
        queue = await bus.register("entity1")
        assert queue is not None
        
        stats = bus.get_stats()
        assert stats["registered_entities"] == 1
        
        await bus.unregister("entity1")
        stats = bus.get_stats()
        assert stats["registered_entities"] == 0
    
    @pytest.mark.asyncio
    async def test_send_and_receive(self, bus):
        """Test sending and receiving messages."""
        await bus.register("sender")
        await bus.register("receiver")
        
        message = Message.request(
            sender="sender",
            recipient="receiver",
            payload={"test": "data"},
        )
        
        sent = await bus.send(message)
        assert sent is True
        
        received = await bus.receive("receiver", timeout=1.0)
        assert received is not None
        assert received.id == message.id
    
    @pytest.mark.asyncio
    async def test_broadcast(self, bus):
        """Test broadcast messaging."""
        await bus.register("entity1")
        await bus.register("entity2")
        await bus.register("entity3")
        
        message = Message.broadcast(
            sender="broadcaster",
            payload={"event": "update"},
        )
        
        sent = await bus.send(message)
        assert sent is True
        
        for entity_id in ["entity1", "entity2", "entity3"]:
            received = await bus.receive(entity_id, timeout=1.0)
            assert received is not None
    
    @pytest.mark.asyncio
    async def test_request_response(self, bus):
        """Test request-response pattern."""
        await bus.register("client")
        await bus.register("server")
        
        async def server_handler():
            request = await bus.receive("server", timeout=5.0)
            if request:
                await bus.respond(request, {"result": "success"})
        
        server_task = asyncio.create_task(server_handler())
        
        response = await bus.request(
            sender="client",
            recipient="server",
            payload={"action": "test"},
            timeout=5.0,
        )
        
        await server_task
        
        assert response is not None
        assert response.payload["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_subscribe_to_type(self, bus):
        """Test subscribing to specific message types."""
        await bus.register("subscriber")
        await bus.subscribe("subscriber", MessageType.AGENT_TASK)
        
        task_message = Message.agent_task(
            sender="coordinator",
            recipient="subscriber",
            task_type="test",
            task_data={},
        )
        
        other_message = Message.request(
            sender="other",
            recipient="subscriber",
            payload={},
        )
        
        await bus.send(task_message)
        
        received = await bus.receive("subscriber", timeout=1.0)
        assert received is not None
        assert received.type == MessageType.AGENT_TASK
    
    @pytest.mark.asyncio
    async def test_message_handler(self, bus):
        """Test message handler callback."""
        received_messages = []
        
        async def handler(message: Message):
            received_messages.append(message)
        
        await bus.register("handler_entity", handler=handler)
        
        message = Message.request(
            sender="sender",
            recipient="handler_entity",
            payload={},
        )
        
        await bus.send(message)
        
        await asyncio.sleep(0.1)
        
        assert len(received_messages) == 1
    
    @pytest.mark.asyncio
    async def test_get_history(self, bus):
        """Test message history retrieval."""
        await bus.register("sender")
        await bus.register("receiver")
        
        for i in range(5):
            message = Message.request(
                sender="sender",
                recipient="receiver",
                payload={"index": i},
            )
            await bus.send(message)
        
        history = bus.get_history(sender="sender")
        assert len(history) == 5
        
        history = bus.get_history(limit=2)
        assert len(history) == 2
    
    @pytest.mark.asyncio
    async def test_stats(self, bus):
        """Test statistics tracking."""
        await bus.register("sender")
        await bus.register("receiver")
        
        message = Message.request(
            sender="sender",
            recipient="receiver",
            payload={},
        )
        
        await bus.send(message)
        
        stats = bus.get_stats()
        assert stats["total_messages_sent"] == 1
        assert "COMPONENT_REQUEST" in stats["messages_by_type"]


class TestMessageRouter:
    """Tests for MessageRouter."""
    
    def test_register_route(self):
        """Test registering a direct route."""
        router = MessageRouter()
        router.register_route("entity1", "queue1")
        
        assert router.get_route("entity1") == "queue1"
    
    def test_unregister_route(self):
        """Test unregistering a route."""
        router = MessageRouter()
        router.register_route("entity1", "queue1")
        
        result = router.unregister_route("entity1")
        assert result is True
        assert router.get_route("entity1") is None
    
    def test_add_remove_rule(self):
        """Test adding and removing routing rules."""
        router = MessageRouter()
        
        rule = RoutingRule(
            name="test_rule",
            target="target_queue",
            message_types={MessageType.AGENT_TASK},
        )
        
        router.add_rule(rule)
        rules = router.get_rules()
        assert len(rules) == 1
        
        result = router.remove_rule("test_rule")
        assert result is True
        assert len(router.get_rules()) == 0
    
    def test_route_by_recipient(self):
        """Test routing by recipient."""
        router = MessageRouter()
        router.register_route("agent1", "agent1_queue")
        
        message = Message.request(
            sender="client",
            recipient="agent1",
            payload={},
        )
        
        target = router.route(message)
        assert target == "agent1_queue"
    
    def test_route_by_rule(self):
        """Test routing by rule."""
        router = MessageRouter()
        
        router.add_rule(RoutingRule(
            name="agent_tasks",
            target="task_queue",
            message_types={MessageType.AGENT_TASK},
        ))
        
        message = Message.agent_task(
            sender="coordinator",
            recipient=None,
            task_type="test",
            task_data={},
        )
        
        target = router.route(message)
        assert target == "task_queue"
    
    def test_default_target(self):
        """Test default target for unroutable messages."""
        router = MessageRouter()
        router.set_default_target("default_queue")
        
        message = Message.broadcast(
            sender="sender",
            payload={},
        )
        
        target = router.route(message)
        assert target == "default_queue"
    
    def test_rule_priority(self):
        """Test rule priority ordering."""
        router = MessageRouter()
        
        router.add_rule(RoutingRule(
            name="low_priority",
            target="low_queue",
            message_types={MessageType.BROADCAST},
            priority=1,
        ))
        
        router.add_rule(RoutingRule(
            name="high_priority",
            target="high_queue",
            message_types={MessageType.BROADCAST},
            priority=10,
        ))
        
        message = Message.broadcast(sender="sender", payload={})
        
        target = router.route(message)
        assert target == "high_queue"
    
    def test_create_agent_router(self):
        """Test creating agent-configured router."""
        router = MessageRouter.create_agent_router()
        
        task_message = Message.agent_task(
            sender="coord",
            recipient=None,
            task_type="test",
            task_data={},
        )
        
        target = router.route(task_message)
        assert target == "agent_task_queue"
    
    def test_create_component_router(self):
        """Test creating component-configured router."""
        router = MessageRouter.create_component_router()
        
        request_message = Message.request(
            sender="comp1",
            recipient=None,
            payload={},
        )
        
        target = router.route(request_message)
        assert target == "component_request_queue"
    
    def test_get_stats(self):
        """Test getting router statistics."""
        router = MessageRouter()
        router.register_route("entity1", "queue1")
        router.add_rule(RoutingRule(
            name="rule1",
            target="target1",
        ))
        
        stats = router.get_stats()
        
        assert stats["total_routes"] == 1
        assert stats["total_rules"] == 1
