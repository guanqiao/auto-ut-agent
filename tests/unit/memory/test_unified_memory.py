"""Unit tests for UnifiedMemory."""

import json
import pytest
from datetime import datetime
from pathlib import Path

from pyutagent.memory.unified_memory import (
    BasePattern,
    KnowledgeEntry,
    MemoryPriority,
    PatternCategory,
    SQLiteMemoryStore,
    InMemoryStore,
    ToolCallRecord,
    UnifiedMemory,
    get_unified_memory,
)


class TestBasePattern:
    """Tests for BasePattern dataclass."""

    def test_create_pattern(self):
        """Test creating a pattern."""
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test Pattern",
            category=PatternCategory.TEST,
            description="A test pattern",
        )
        
        assert pattern.pattern_id == "test_1"
        assert pattern.name == "Test Pattern"
        assert pattern.category == PatternCategory.TEST
        assert pattern.description == "A test pattern"
        assert pattern.success_count == 0
        assert pattern.failure_count == 0

    def test_pattern_success_rate(self):
        """Test success rate calculation."""
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
            success_count=8,
            failure_count=2,
        )
        
        assert pattern.success_rate == 0.8

    def test_pattern_success_rate_zero(self):
        """Test success rate with no attempts."""
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
        )
        
        assert pattern.success_rate == 0.0

    def test_record_success(self):
        """Test recording a success."""
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
        )
        
        pattern.record_success()
        
        assert pattern.success_count == 1
        assert pattern.failure_count == 0

    def test_record_failure(self):
        """Test recording a failure."""
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
        )
        
        pattern.record_failure()
        
        assert pattern.success_count == 0
        assert pattern.failure_count == 1

    def test_pattern_to_dict(self):
        """Test converting to dictionary."""
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
            tags=["tag1", "tag2"],
        )
        
        data = pattern.to_dict()
        
        assert data["pattern_id"] == "test_1"
        assert data["name"] == "Test"
        assert data["category"] == "test"
        assert data["tags"] == ["tag1", "tag2"]

    def test_pattern_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "pattern_id": "test_1",
            "name": "Test",
            "category": "test",
            "description": "A test",
            "success_count": 5,
        }
        
        pattern = BasePattern.from_dict(data)
        
        assert pattern.pattern_id == "test_1"
        assert pattern.category == PatternCategory.TEST
        assert pattern.success_count == 5

    def test_pattern_with_template(self):
        """Test pattern with template."""
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
            template="def test_{method_name}(): pass",
        )
        
        assert pattern.template == "def test_{method_name}(): pass"


class TestToolCallRecord:
    """Tests for ToolCallRecord dataclass."""

    def test_create_record(self):
        """Test creating a tool call record."""
        record = ToolCallRecord(
            tool_name="read_file",
            parameters={"path": "/test.py"},
            result="file content",
            success=True,
        )
        
        assert record.tool_name == "read_file"
        assert record.parameters == {"path": "/test.py"}
        assert record.result == "file content"
        assert record.success is True
        assert record.error_message is None

    def test_record_with_error(self):
        """Test record with error."""
        record = ToolCallRecord(
            tool_name="read_file",
            parameters={"path": "/test.py"},
            success=False,
            error_message="File not found",
        )
        
        assert record.success is False
        assert record.error_message == "File not found"

    def test_record_to_dict(self):
        """Test converting to dictionary."""
        record = ToolCallRecord(
            tool_name="read_file",
            parameters={"path": "/test.py"},
            success=True,
            duration_ms=100,
        )
        
        data = record.to_dict()
        
        assert data["tool_name"] == "read_file"
        assert data["parameters"] == {"path": "/test.py"}
        assert data["duration_ms"] == 100

    def test_record_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "tool_name": "write_file",
            "parameters": {"path": "/test.py", "content": "test"},
            "result": "OK",
            "success": True,
            "error_message": None,
            "duration_ms": 50,
            "timestamp": "2024-01-01T00:00:00",
        }
        
        record = ToolCallRecord.from_dict(data)
        
        assert record.tool_name == "write_file"
        assert record.duration_ms == 50


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry dataclass."""

    def test_create_entry(self):
        """Test creating a knowledge entry."""
        entry = KnowledgeEntry(
            entry_id="kb_1",
            title="Java Best Practices",
            content="Use try-with-resources for AutoCloseable objects",
            category="java",
        )
        
        assert entry.entry_id == "kb_1"
        assert entry.title == "Java Best Practices"
        assert entry.category == "java"

    def test_entry_to_dict(self):
        """Test converting to dictionary."""
        entry = KnowledgeEntry(
            entry_id="kb_1",
            title="Test",
            content="Content",
            tags=["tag1"],
        )
        
        data = entry.to_dict()
        
        assert data["entry_id"] == "kb_1"
        assert data["tags"] == ["tag1"]

    def test_entry_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "entry_id": "kb_1",
            "title": "Test",
            "content": "Content",
            "category": "general",
            "tags": [],
            "confidence": 0.9,
        }
        
        entry = KnowledgeEntry.from_dict(data)
        
        assert entry.entry_id == "kb_1"
        assert entry.confidence == 0.9


class TestInMemoryStore:
    """Tests for InMemoryStore."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving values."""
        store = InMemoryStore()
        
        store.store("key1", "value1")
        
        assert store.retrieve("key1") == "value1"

    def test_retrieve_nonexistent(self):
        """Test retrieving nonexistent key."""
        store = InMemoryStore()
        
        assert store.retrieve("nonexistent") is None

    def test_delete(self):
        """Test deleting a key."""
        store = InMemoryStore()
        store.store("key1", "value1")
        
        result = store.delete("key1")
        
        assert result is True
        assert store.retrieve("key1") is None

    def test_delete_nonexistent(self):
        """Test deleting nonexistent key."""
        store = InMemoryStore()
        
        result = store.delete("nonexistent")
        
        assert result is False

    def test_list_all(self):
        """Test listing all keys."""
        store = InMemoryStore()
        store.store("key1", "value1")
        store.store("key2", "value2")
        
        keys = store.list_all()
        
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    def test_clear(self):
        """Test clearing all values."""
        store = InMemoryStore()
        store.store("key1", "value1")
        store.store("key2", "value2")
        
        store.clear()
        
        assert len(store.list_all()) == 0


class TestSQLiteMemoryStore:
    """Tests for SQLiteMemoryStore."""

    def test_store_and_retrieve(self, tmp_path):
        """Test storing and retrieving values."""
        db_path = tmp_path / "test.db"
        store = SQLiteMemoryStore(
            db_path=db_path,
            table_name="test_table",
            serializer=json.dumps,
            deserializer=json.loads,
        )
        
        store.store("key1", {"value": "data1"})
        
        result = store.retrieve("key1")
        
        assert result == {"value": "data1"}

    def test_retrieve_nonexistent(self, tmp_path):
        """Test retrieving nonexistent key."""
        db_path = tmp_path / "test.db"
        store = SQLiteMemoryStore(
            db_path=db_path,
            table_name="test_table",
            serializer=json.dumps,
            deserializer=json.loads,
        )
        
        assert store.retrieve("nonexistent") is None

    def test_delete(self, tmp_path):
        """Test deleting a key."""
        db_path = tmp_path / "test.db"
        store = SQLiteMemoryStore(
            db_path=db_path,
            table_name="test_table",
            serializer=json.dumps,
            deserializer=json.loads,
        )
        store.store("key1", {"value": "data1"})
        
        result = store.delete("key1")
        
        assert result is True
        assert store.retrieve("key1") is None

    def test_list_all(self, tmp_path):
        """Test listing all keys."""
        db_path = tmp_path / "test.db"
        store = SQLiteMemoryStore(
            db_path=db_path,
            table_name="test_table",
            serializer=json.dumps,
            deserializer=json.loads,
        )
        store.store("key1", {"value": "data1"})
        store.store("key2", {"value": "data2"})
        
        keys = store.list_all()
        
        assert len(keys) == 2

    def test_search(self, tmp_path):
        """Test searching for values."""
        db_path = tmp_path / "test.db"
        store = SQLiteMemoryStore(
            db_path=db_path,
            table_name="test_table",
            serializer=json.dumps,
            deserializer=json.loads,
        )
        store.store("key1", {"name": "test_pattern"})
        store.store("key2", {"name": "other_thing"})
        
        results = store.search("pattern")
        
        assert len(results) == 1
        assert results[0]["name"] == "test_pattern"

    def test_count(self, tmp_path):
        """Test counting entries."""
        db_path = tmp_path / "test.db"
        store = SQLiteMemoryStore(
            db_path=db_path,
            table_name="test_table",
            serializer=json.dumps,
            deserializer=json.loads,
        )
        store.store("key1", {"value": "data1"})
        store.store("key2", {"value": "data2"})
        
        assert store.count() == 2

    def test_clear(self, tmp_path):
        """Test clearing all values."""
        db_path = tmp_path / "test.db"
        store = SQLiteMemoryStore(
            db_path=db_path,
            table_name="test_table",
            serializer=json.dumps,
            deserializer=json.loads,
        )
        store.store("key1", {"value": "data1"})
        store.store("key2", {"value": "data2"})
        
        store.clear()
        
        assert store.count() == 0


class TestUnifiedMemory:
    """Tests for UnifiedMemory."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedMemory.reset_instance()

    def test_singleton_instance(self, tmp_path):
        """Test singleton pattern."""
        memory1 = UnifiedMemory(data_dir=tmp_path)
        UnifiedMemory._instance = memory1
        memory2 = UnifiedMemory.get_instance()
        
        assert memory1 is memory2

    def test_store_and_get_pattern(self, tmp_path):
        """Test storing and getting patterns."""
        UnifiedMemory.reset_instance()
        memory = UnifiedMemory(data_dir=tmp_path)
        
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test Pattern",
            category=PatternCategory.TEST,
        )
        
        memory.store_pattern(pattern)
        
        result = memory.get_pattern("test_1")
        
        assert result is not None
        assert result.name == "Test Pattern"

    def test_delete_pattern(self, tmp_path):
        """Test deleting a pattern."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        pattern = BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
        )
        memory.store_pattern(pattern)
        
        result = memory.delete_pattern("test_1")
        
        assert result is True
        assert memory.get_pattern("test_1") is None

    def test_list_patterns(self, tmp_path):
        """Test listing patterns."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        memory.store_pattern(BasePattern(
            pattern_id="test_1",
            name="Test 1",
            category=PatternCategory.TEST,
        ))
        memory.store_pattern(BasePattern(
            pattern_id="code_1",
            name="Code 1",
            category=PatternCategory.CODE,
        ))
        
        all_patterns = memory.list_patterns()
        test_patterns = memory.list_patterns(PatternCategory.TEST)
        
        assert len(all_patterns) == 2
        assert len(test_patterns) == 1

    def test_get_best_patterns(self, tmp_path):
        """Test getting best patterns by success rate."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        pattern1 = BasePattern(
            pattern_id="test_1",
            name="High Success",
            category=PatternCategory.TEST,
            success_count=9,
            failure_count=1,
        )
        pattern2 = BasePattern(
            pattern_id="test_2",
            name="Low Success",
            category=PatternCategory.TEST,
            success_count=1,
            failure_count=9,
        )
        
        memory.store_pattern(pattern1)
        memory.store_pattern(pattern2)
        
        best = memory.get_best_patterns(min_success_rate=0.5)
        
        assert len(best) == 1
        assert best[0].name == "High Success"

    def test_record_tool_call(self, tmp_path):
        """Test recording tool calls."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        record_id = memory.record_tool_call(
            tool_name="read_file",
            parameters={"path": "/test.py"},
            result="content",
            success=True,
            duration_ms=100,
        )
        
        assert record_id.startswith("read_file_")
        
        history = memory.get_tool_call_history()
        assert len(history) == 1
        assert history[0].tool_name == "read_file"

    def test_get_tool_call_history_filtered(self, tmp_path):
        """Test getting tool call history filtered by tool name."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        memory.record_tool_call("read_file", {}, success=True)
        memory.record_tool_call("write_file", {}, success=True)
        memory.record_tool_call("read_file", {}, success=True)
        
        read_history = memory.get_tool_call_history(tool_name="read_file")
        
        assert len(read_history) == 2

    def test_get_tool_success_rate(self, tmp_path):
        """Test getting tool success rate."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        memory.record_tool_call("read_file", {}, success=True)
        memory.record_tool_call("read_file", {}, success=True)
        memory.record_tool_call("read_file", {}, success=False)
        
        rate = memory.get_tool_success_rate("read_file")
        
        assert rate == pytest.approx(2/3)

    def test_store_and_get_knowledge(self, tmp_path):
        """Test storing and getting knowledge."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        entry = KnowledgeEntry(
            entry_id="kb_1",
            title="Test Knowledge",
            content="Some content",
        )
        
        memory.store_knowledge(entry)
        
        result = memory.get_knowledge("kb_1")
        
        assert result is not None
        assert result.title == "Test Knowledge"

    def test_delete_knowledge(self, tmp_path):
        """Test deleting knowledge."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        entry = KnowledgeEntry(
            entry_id="kb_1",
            title="Test",
            content="Content",
        )
        memory.store_knowledge(entry)
        
        result = memory.delete_knowledge("kb_1")
        
        assert result is True
        assert memory.get_knowledge("kb_1") is None

    def test_search_knowledge(self, tmp_path):
        """Test searching knowledge."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        memory.store_knowledge(KnowledgeEntry(
            entry_id="kb_1",
            title="Java Testing",
            content="JUnit is a testing framework",
        ))
        memory.store_knowledge(KnowledgeEntry(
            entry_id="kb_2",
            title="Python",
            content="Python is a language",
        ))
        
        results = memory.search_knowledge("testing")
        
        assert len(results) == 1
        assert results[0].entry_id == "kb_1"

    def test_working_memory(self, tmp_path):
        """Test working memory operations."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        memory.set_working("current_task", "generate tests")
        memory.set_working("target_class", "UserService")
        
        assert memory.get_working("current_task") == "generate tests"
        assert memory.get_working("nonexistent", "default") == "default"

    def test_clear_working(self, tmp_path):
        """Test clearing working memory."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        memory.set_working("key1", "value1")
        memory.set_working("key2", "value2")
        
        memory.clear_working()
        
        assert memory.get_working("key1") is None

    def test_get_stats(self, tmp_path):
        """Test getting statistics."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        memory.store_pattern(BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
        ))
        memory.record_tool_call("read_file", {}, success=True)
        memory.store_knowledge(KnowledgeEntry(
            entry_id="kb_1",
            title="Test",
            content="Content",
        ))
        
        stats = memory.get_stats()
        
        assert stats["patterns_count"] == 1
        assert stats["tool_calls_count"] == 1
        assert stats["knowledge_count"] == 1

    def test_clear_all(self, tmp_path):
        """Test clearing all memory."""
        memory = UnifiedMemory(data_dir=tmp_path)
        
        memory.store_pattern(BasePattern(
            pattern_id="test_1",
            name="Test",
            category=PatternCategory.TEST,
        ))
        memory.record_tool_call("read_file", {}, success=True)
        
        memory.clear_all()
        
        stats = memory.get_stats()
        assert stats["patterns_count"] == 0
        assert stats["tool_calls_count"] == 0


class TestPatternCategory:
    """Tests for PatternCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert PatternCategory.TEST.value == "test"
        assert PatternCategory.CODE.value == "code"
        assert PatternCategory.TOOL.value == "tool"
        assert PatternCategory.ERROR_SOLUTION.value == "error_solution"
        assert PatternCategory.BEST_PRACTICE.value == "best_practice"


class TestMemoryPriority:
    """Tests for MemoryPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert MemoryPriority.LOW.value == 1
        assert MemoryPriority.NORMAL.value == 5
        assert MemoryPriority.HIGH.value == 10
        assert MemoryPriority.CRITICAL.value == 20


class TestGetUnifiedMemory:
    """Tests for get_unified_memory helper."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedMemory.reset_instance()

    def test_get_unified_memory(self):
        """Test getting unified memory instance."""
        memory = get_unified_memory()
        
        assert isinstance(memory, UnifiedMemory)
