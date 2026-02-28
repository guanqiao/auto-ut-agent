"""Tests for short term memory."""

import pytest
import tempfile
from pathlib import Path


class TestShortTermMemory:
    """Test suite for ShortTermMemory."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"
            yield str(db_path)
    
    def test_init_creates_tables(self, temp_db):
        """Test that initialization creates necessary tables."""
        from pyutagent.memory.short_term_memory import ShortTermMemory
        
        mem = ShortTermMemory(temp_db)
        mem.close()
        
        assert Path(temp_db).exists()
    
    def test_save_and_load_conversation(self, temp_db):
        """Test saving and loading conversation."""
        from pyutagent.memory.short_term_memory import ShortTermMemory
        
        mem = ShortTermMemory(temp_db)
        
        # Save messages
        session_id = "test-session-1"
        mem.add_message(session_id, "user", "生成 UserService 的测试")
        mem.add_message(session_id, "agent", "好的，我来分析这个文件...")
        mem.add_message(session_id, "user", "暂停")
        
        # Load conversation
        messages = mem.get_conversation(session_id)
        mem.close()
        
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "生成 UserService 的测试"
        assert messages[1]["role"] == "agent"
    
    def test_get_conversation_limit(self, temp_db):
        """Test conversation with limit."""
        from pyutagent.memory.short_term_memory import ShortTermMemory
        
        mem = ShortTermMemory(temp_db)
        
        session_id = "test-session"
        for i in range(10):
            mem.add_message(session_id, "user", f"Message {i}")
        
        # Get only last 5
        messages = mem.get_conversation(session_id, limit=5)
        mem.close()
        
        assert len(messages) == 5
        assert messages[0]["content"] == "Message 5"
    
    def test_multiple_sessions(self, temp_db):
        """Test multiple concurrent sessions."""
        from pyutagent.memory.short_term_memory import ShortTermMemory
        
        mem = ShortTermMemory(temp_db)
        
        # Session 1
        mem.add_message("session-1", "user", "Session 1 message")
        
        # Session 2
        mem.add_message("session-2", "user", "Session 2 message")
        
        # Verify separation
        conv1 = mem.get_conversation("session-1")
        conv2 = mem.get_conversation("session-2")
        mem.close()
        
        assert len(conv1) == 1
        assert len(conv2) == 1
        assert conv1[0]["content"] == "Session 1 message"
        assert conv2[0]["content"] == "Session 2 message"
    
    def test_clear_conversation(self, temp_db):
        """Test clearing conversation."""
        from pyutagent.memory.short_term_memory import ShortTermMemory
        
        mem = ShortTermMemory(temp_db)
        
        session_id = "test-session"
        mem.add_message(session_id, "user", "Message 1")
        mem.add_message(session_id, "agent", "Message 2")
        
        mem.clear_conversation(session_id)
        
        messages = mem.get_conversation(session_id)
        mem.close()
        
        assert len(messages) == 0
    
    def test_get_all_sessions(self, temp_db):
        """Test getting all session IDs."""
        from pyutagent.memory.short_term_memory import ShortTermMemory
        
        mem = ShortTermMemory(temp_db)
        
        mem.add_message("session-a", "user", "Message")
        mem.add_message("session-b", "user", "Message")
        mem.add_message("session-c", "user", "Message")
        
        sessions = mem.get_all_sessions()
        mem.close()
        
        assert len(sessions) == 3
        assert "session-a" in sessions
        assert "session-b" in sessions
        assert "session-c" in sessions
