"""Tests for Agent Context Management."""

import pytest
from datetime import datetime

from pyutagent.agent.core.agent_context import (
    ContextKey,
    ContextEntry,
    AgentContext,
)


class TestContextKey:
    """Tests for ContextKey enum."""
    
    def test_key_property(self):
        """Test key string access."""
        assert ContextKey.PROJECT_PATH.key == "project_path"
        assert ContextKey.TARGET_FILE.key == "target_file"
    
    def test_value_type_property(self):
        """Test value type access."""
        assert ContextKey.PROJECT_PATH.value_type == str
        assert ContextKey.CURRENT_ITERATION.value_type == int
        assert ContextKey.CLASS_INFO.value_type == dict


class TestContextEntry:
    """Tests for ContextEntry."""
    
    def test_creation(self):
        """Test entry creation."""
        entry = ContextEntry(
            key="test_key",
            value="test_value",
            source="user",
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.source == "user"
        assert isinstance(entry.created_at, datetime)
    
    def test_update(self):
        """Test entry update."""
        entry = ContextEntry(key="test", value="old")
        
        entry.update("new", "agent")
        
        assert entry.value == "new"
        assert entry.source == "agent"
    
    def test_to_dict(self):
        """Test entry serialization."""
        entry = ContextEntry(
            key="test",
            value="value",
            source="user",
            metadata={"meta": "data"},
        )
        
        result = entry.to_dict()
        
        assert result["key"] == "test"
        assert result["value"] == "value"
        assert result["source"] == "user"
        assert result["metadata"]["meta"] == "data"


class TestAgentContext:
    """Tests for AgentContext."""
    
    def test_get_set(self):
        """Test basic get/set operations."""
        context = AgentContext()
        
        context.set("key1", "value1")
        assert context.get("key1") == "value1"
        
        context.set(ContextKey.PROJECT_PATH, "/path/to/project")
        assert context.get(ContextKey.PROJECT_PATH) == "/path/to/project"
    
    def test_get_default(self):
        """Test get with default value."""
        context = AgentContext()
        
        assert context.get("nonexistent") is None
        assert context.get("nonexistent", "default") == "default"
    
    def test_has(self):
        """Test key existence check."""
        context = AgentContext()
        context.set("key1", "value1")
        
        assert context.has("key1")
        assert not context.has("nonexistent")
    
    def test_delete(self):
        """Test key deletion."""
        context = AgentContext()
        context.set("key1", "value1")
        
        assert context.delete("key1") is True
        assert not context.has("key1")
        assert context.delete("nonexistent") is False
    
    def test_keys(self):
        """Test getting all keys."""
        context = AgentContext()
        context.set("key1", "value1")
        context.set("key2", "value2")
        
        keys = context.keys()
        
        assert "key1" in keys
        assert "key2" in keys
        assert len(keys) == 2
    
    def test_parent_inheritance(self):
        """Test context inheritance from parent."""
        parent = AgentContext()
        parent.set("parent_key", "parent_value")
        
        child = parent.create_child({"child_key": "child_value"})
        
        assert child.get("parent_key") == "parent_value"
        assert child.get("child_key") == "child_value"
        assert parent.get("child_key") is None
    
    def test_child_override(self):
        """Test child can override parent values."""
        parent = AgentContext()
        parent.set("key", "parent_value")
        
        child = parent.create_child()
        child.set("key", "child_value")
        
        assert child.get("key") == "child_value"
        assert parent.get("key") == "parent_value"
    
    def test_to_dict(self):
        """Test context serialization."""
        context = AgentContext()
        context.set("key1", "value1")
        context.set("key2", "value2")
        
        result = context.to_dict()
        
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
    
    def test_from_dict(self):
        """Test context deserialization."""
        data = {"key1": "value1", "key2": "value2"}
        
        context = AgentContext.from_dict(data)
        
        assert context.get("key1") == "value1"
        assert context.get("key2") == "value2"
    
    def test_snapshot(self):
        """Test snapshot creation and restore."""
        context = AgentContext()
        context.set("key1", "value1")
        
        snapshot_id = context.create_snapshot()
        
        context.set("key1", "value2")
        context.set("key3", "value3")
        
        assert context.get("key1") == "value2"
        
        context.restore_snapshot(snapshot_id)
        
        assert context.get("key1") == "value1"
        assert not context.has("key3")
    
    def test_changed_keys(self):
        """Test changed keys tracking."""
        context = AgentContext()
        context.set("key1", "value1")
        context.set("key2", "value2")
        
        assert "key1" in context.get_changed_keys()
        assert "key2" in context.get_changed_keys()
        
        context.clear_changed_keys()
        
        assert len(context.get_changed_keys()) == 0
    
    def test_item_access(self):
        """Test [] access syntax."""
        context = AgentContext()
        
        context["key"] = "value"
        assert context["key"] == "value"
        assert "key" in context
        
        with pytest.raises(KeyError):
            _ = context["nonexistent"]
    
    def test_clear(self):
        """Test context clearing."""
        context = AgentContext()
        context.set("key1", "value1")
        context.set("key2", "value2")
        
        context.clear()
        
        assert len(context.keys()) == 0
    
    def test_get_entry(self):
        """Test getting full entry."""
        context = AgentContext()
        context.set("key", "value", source="test", metadata={"meta": "data"})
        
        entry = context.get_entry("key")
        
        assert entry is not None
        assert entry.value == "value"
        assert entry.source == "test"
        assert entry.metadata["meta"] == "data"
