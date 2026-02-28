"""Tests for conversation manager."""

import pytest
import tempfile
from pathlib import Path


class TestConversationManager:
    """Test suite for ConversationManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a conversation manager."""
        from pyutagent.agent.conversation import ConversationManager, MessageRole
        return ConversationManager()
    
    def test_add_message(self, manager):
        """Test adding a message."""
        from pyutagent.agent.conversation import MessageRole
        
        msg = manager.add_message(MessageRole.USER, "Hello")
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert len(manager.messages) == 1
    
    def test_add_user_message(self, manager):
        """Test adding user message."""
        from pyutagent.agent.conversation import MessageRole
        
        msg = manager.add_user_message("Test message")
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Test message"
    
    def test_add_agent_message(self, manager):
        """Test adding agent message."""
        from pyutagent.agent.conversation import MessageRole
        
        msg = manager.add_agent_message("Response")
        
        assert msg.role == MessageRole.AGENT
        assert msg.content == "Response"
    
    def test_add_tool_message(self, manager):
        """Test adding tool message."""
        from pyutagent.agent.conversation import MessageRole
        
        msg = manager.add_tool_message("Tool result", "parse_java")
        
        assert msg.role == MessageRole.TOOL
        assert msg.metadata["tool_name"] == "parse_java"
    
    def test_get_context(self, manager):
        """Test getting context."""
        from pyutagent.agent.conversation import MessageRole
        
        manager.add_message(MessageRole.USER, "Message 1")
        manager.add_message(MessageRole.AGENT, "Message 2")
        manager.add_message(MessageRole.USER, "Message 3")
        
        context = manager.get_context(2)
        
        assert len(context) == 2
        assert context[0].content == "Message 2"
        assert context[1].content == "Message 3"
    
    def test_max_history_limit(self, manager):
        """Test max history limit."""
        from pyutagent.agent.conversation import MessageRole
        
        manager.max_history = 3
        
        for i in range(5):
            manager.add_message(MessageRole.USER, f"Message {i}")
        
        assert len(manager.messages) == 3
        assert manager.messages[0].content == "Message 2"
        assert manager.messages[-1].content == "Message 4"
    
    def test_to_llm_messages(self, manager):
        """Test converting to LLM format."""
        from pyutagent.agent.conversation import MessageRole
        
        manager.add_user_message("Hello")
        manager.add_agent_message("Hi there")
        
        llm_msgs = manager.to_llm_messages()
        
        assert len(llm_msgs) == 2
        assert llm_msgs[0]["role"] == "user"
        assert llm_msgs[1]["role"] == "agent"
    
    def test_clear(self, manager):
        """Test clearing messages."""
        from pyutagent.agent.conversation import MessageRole
        
        manager.add_message(MessageRole.USER, "Test")
        assert len(manager.messages) == 1
        
        manager.clear()
        assert len(manager.messages) == 0
    
    def test_export_import(self, manager, tmp_path):
        """Test export and import."""
        from pyutagent.agent.conversation import MessageRole
        
        manager.add_user_message("User message")
        manager.add_agent_message("Agent response")
        
        export_path = tmp_path / "conversation.json"
        manager.export_to_file(str(export_path))
        
        assert export_path.exists()
        
        # Create new manager and import
        from pyutagent.agent.conversation import ConversationManager
        new_manager = ConversationManager()
        new_manager.import_from_file(str(export_path))
        
        assert len(new_manager.messages) == 2
        assert new_manager.messages[0].content == "User message"
        assert new_manager.messages[1].content == "Agent response"
    
    def test_get_summary(self, manager):
        """Test getting summary."""
        from pyutagent.agent.conversation import MessageRole
        
        manager.add_user_message("User 1")
        manager.add_agent_message("Agent 1")
        manager.add_tool_message("Tool result", "test_tool")
        
        summary = manager.get_summary()
        
        assert summary["total_messages"] == 3
        assert summary["user_messages"] == 1
        assert summary["agent_messages"] == 1
        assert summary["tool_calls"] == 1
    
    def test_callback(self, manager):
        """Test message callback."""
        from pyutagent.agent.conversation import MessageRole
        
        received_messages = []
        
        def callback(msg):
            received_messages.append(msg)
        
        manager.on_message = callback
        manager.add_message(MessageRole.USER, "Test")
        
        assert len(received_messages) == 1
        assert received_messages[0].content == "Test"


class TestMessage:
    """Test suite for Message."""
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        from pyutagent.agent.conversation import Message, MessageRole
        
        msg = Message(
            role=MessageRole.USER,
            content="Test",
            metadata={"key": "value"}
        )
        
        data = msg.to_dict()
        
        assert data["role"] == "user"
        assert data["content"] == "Test"
        assert data["metadata"]["key"] == "value"
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        from pyutagent.agent.conversation import Message, MessageRole
        
        data = {
            "role": "agent",
            "content": "Response",
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {}
        }
        
        msg = Message.from_dict(data)
        
        assert msg.role == MessageRole.AGENT
        assert msg.content == "Response"
