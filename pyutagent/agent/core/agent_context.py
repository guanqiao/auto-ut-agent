"""Agent Context Management.

This module provides unified context management for agents with:
- Type-safe context keys
- Context validation
- Context serialization
- Context inheritance
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, TypeVar, Generic
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ContextKey(Enum):
    """Standard context keys with type hints."""
    
    PROJECT_PATH = ("project_path", str)
    TARGET_FILE = ("target_file", str)
    TEST_FILE = ("test_file", str)
    CLASS_INFO = ("class_info", dict)
    SOURCE_CODE = ("source_code", str)
    TEST_CODE = ("test_code", str)
    
    CURRENT_ITERATION = ("current_iteration", int)
    MAX_ITERATIONS = ("max_iterations", int)
    CURRENT_COVERAGE = ("current_coverage", float)
    TARGET_COVERAGE = ("target_coverage", float)
    
    LLM_MODEL = ("llm_model", str)
    LLM_PROVIDER = ("llm_provider", str)
    TEMPERATURE = ("temperature", float)
    
    COMPILATION_ERRORS = ("compilation_errors", list)
    TEST_FAILURES = ("test_failures", list)
    COVERAGE_REPORT = ("coverage_report", dict)
    
    USER_MESSAGE = ("user_message", str)
    AGENT_MESSAGE = ("agent_message", str)
    
    START_TIME = ("start_time", datetime)
    END_TIME = ("end_time", datetime)
    ELAPSED_TIME = ("elapsed_time", float)
    
    CUSTOM_DATA = ("custom_data", dict)
    
    def __init__(self, key: str, value_type: type):
        self._key = key
        self._value_type = value_type
    
    @property
    def key(self) -> str:
        """Get the string key."""
        return self._key
    
    @property
    def value_type(self) -> type:
        """Get the expected value type."""
        return self._value_type


@dataclass
class ContextEntry:
    """A single context entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, new_value: Any, source: str = "agent") -> None:
        """Update the entry value."""
        self.value = new_value
        self.updated_at = datetime.now()
        self.source = source
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


class AgentContext:
    """Unified context management for agents.
    
    Features:
    - Type-safe key access
    - Context inheritance
    - Serialization support
    - Change tracking
    - Context snapshots
    """
    
    def __init__(
        self,
        parent: Optional["AgentContext"] = None,
        initial_data: Optional[Dict[str, Any]] = None,
    ):
        """Initialize agent context.
        
        Args:
            parent: Optional parent context for inheritance
            initial_data: Initial context data
        """
        self._data: Dict[str, ContextEntry] = {}
        self._parent = parent
        self._changed_keys: Set[str] = set()
        self._snapshots: List[Dict[str, Any]] = []
        
        if initial_data:
            for key, value in initial_data.items():
                self.set(key, value)
    
    def get(
        self,
        key: str | ContextKey,
        default: Any = None,
        include_parent: bool = True,
    ) -> Any:
        """Get a context value.
        
        Args:
            key: Context key (string or ContextKey enum)
            default: Default value if key not found
            include_parent: Whether to check parent context
            
        Returns:
            The context value or default
        """
        key_str = key.key if isinstance(key, ContextKey) else key
        
        if key_str in self._data:
            return self._data[key_str].value
        
        if include_parent and self._parent:
            return self._parent.get(key_str, default, include_parent=True)
        
        return default
    
    def set(
        self,
        key: str | ContextKey,
        value: Any,
        source: str = "agent",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a context value.
        
        Args:
            key: Context key (string or ContextKey enum)
            value: Value to set
            source: Source of the value
            metadata: Optional metadata
        """
        key_str = key.key if isinstance(key, ContextKey) else key
        
        if key_str in self._data:
            self._data[key_str].update(value, source)
            if metadata:
                self._data[key_str].metadata.update(metadata)
        else:
            self._data[key_str] = ContextEntry(
                key=key_str,
                value=value,
                source=source,
                metadata=metadata or {},
            )
        
        self._changed_keys.add(key_str)
    
    def delete(self, key: str | ContextKey) -> bool:
        """Delete a context value.
        
        Args:
            key: Context key to delete
            
        Returns:
            True if key was deleted
        """
        key_str = key.key if isinstance(key, ContextKey) else key
        
        if key_str in self._data:
            del self._data[key_str]
            self._changed_keys.add(key_str)
            return True
        
        return False
    
    def has(self, key: str | ContextKey, include_parent: bool = True) -> bool:
        """Check if a key exists.
        
        Args:
            key: Context key to check
            include_parent: Whether to check parent context
            
        Returns:
            True if key exists
        """
        key_str = key.key if isinstance(key, ContextKey) else key
        
        if key_str in self._data:
            return True
        
        if include_parent and self._parent:
            return self._parent.has(key_str, include_parent=True)
        
        return False
    
    def keys(self, include_parent: bool = True) -> Set[str]:
        """Get all context keys.
        
        Args:
            include_parent: Whether to include parent keys
            
        Returns:
            Set of all keys
        """
        keys = set(self._data.keys())
        
        if include_parent and self._parent:
            keys.update(self._parent.keys(include_parent=True))
        
        return keys
    
    def get_changed_keys(self) -> Set[str]:
        """Get keys that have been changed since last snapshot.
        
        Returns:
            Set of changed keys
        """
        return self._changed_keys.copy()
    
    def clear_changed_keys(self) -> None:
        """Clear the changed keys tracking."""
        self._changed_keys.clear()
    
    def create_snapshot(self) -> str:
        """Create a snapshot of current context.
        
        Returns:
            Snapshot ID
        """
        snapshot = self.to_dict()
        snapshot_id = f"snapshot_{len(self._snapshots)}"
        self._snapshots.append(snapshot)
        self.clear_changed_keys()
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore context from a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to restore
            
        Returns:
            True if snapshot was restored
        """
        try:
            snapshot_idx = int(snapshot_id.split("_")[1])
            if 0 <= snapshot_idx < len(self._snapshots):
                snapshot = self._snapshots[snapshot_idx]
                self._data.clear()
                for key, value in snapshot.items():
                    self.set(key, value)
                return True
        except (IndexError, ValueError):
            pass
        
        return False
    
    def to_dict(self, include_parent: bool = True) -> Dict[str, Any]:
        """Convert context to dictionary.
        
        Args:
            include_parent: Whether to include parent data
            
        Returns:
            Dictionary of context data
        """
        result = {}
        
        if include_parent and self._parent:
            result.update(self._parent.to_dict(include_parent=True))
        
        for key, entry in self._data.items():
            result[key] = entry.value
        
        return result
    
    def to_json(self, include_parent: bool = True) -> str:
        """Convert context to JSON string.
        
        Args:
            include_parent: Whether to include parent data
            
        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(include_parent), default=str, indent=2)
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        parent: Optional["AgentContext"] = None,
    ) -> "AgentContext":
        """Create context from dictionary.
        
        Args:
            data: Dictionary of context data
            parent: Optional parent context
            
        Returns:
            New AgentContext instance
        """
        context = cls(parent=parent)
        for key, value in data.items():
            context.set(key, value, source="deserialization")
        return context
    
    def create_child(self, initial_data: Optional[Dict[str, Any]] = None) -> "AgentContext":
        """Create a child context that inherits from this one.
        
        Args:
            initial_data: Initial data for child context
            
        Returns:
            New child AgentContext
        """
        return AgentContext(parent=self, initial_data=initial_data)
    
    def get_entry(self, key: str | ContextKey) -> Optional[ContextEntry]:
        """Get the full context entry for a key.
        
        Args:
            key: Context key
            
        Returns:
            ContextEntry or None
        """
        key_str = key.key if isinstance(key, ContextKey) else key
        return self._data.get(key_str)
    
    def get_all_entries(self) -> Dict[str, ContextEntry]:
        """Get all context entries.
        
        Returns:
            Dictionary of all entries
        """
        return self._data.copy()
    
    def clear(self) -> None:
        """Clear all context data."""
        self._data.clear()
        self._changed_keys.clear()
    
    def __contains__(self, key: str | ContextKey) -> bool:
        """Check if key exists using 'in' operator."""
        return self.has(key)
    
    def __getitem__(self, key: str | ContextKey) -> Any:
        """Get value using [] operator."""
        result = self.get(key)
        if result is None:
            raise KeyError(f"Context key not found: {key}")
        return result
    
    def __setitem__(self, key: str | ContextKey, value: Any) -> None:
        """Set value using [] operator."""
        self.set(key, value)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AgentContext(keys={len(self._data)}, parent={self._parent is not None})"
