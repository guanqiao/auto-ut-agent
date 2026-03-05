"""Unified Memory System - Simplified Architecture.

This module provides a simplified, unified memory system that consolidates:
- Pattern storage (from pattern_library, semantic_memory, knowledge_transfer)
- Tool memory (from tool_memory, enhanced_memory)
- Knowledge storage (from domain_knowledge, project_knowledge_graph)

This is part of Phase 2 Week 15-16: Memory System Simplification.
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PatternCategory(Enum):
    """Categories for patterns."""
    TEST = "test"
    CODE = "code"
    TOOL = "tool"
    ERROR_SOLUTION = "error_solution"
    BEST_PRACTICE = "best_practice"


class MemoryPriority(Enum):
    """Priority levels for memory entries."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class BasePattern:
    """Unified pattern base class.
    
    Consolidates:
    - TestPattern from pattern_library
    - CodePattern from semantic_memory
    - Pattern from knowledge_transfer
    - ToolPattern from enhanced_memory
    """
    pattern_id: str
    name: str
    category: PatternCategory
    description: str = ""
    template: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def record_success(self) -> None:
        """Record a successful application."""
        self.success_count += 1
        self.updated_at = datetime.now().isoformat()
    
    def record_failure(self) -> None:
        """Record a failed application."""
        self.failure_count += 1
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["category"] = self.category.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasePattern":
        """Create from dictionary."""
        if isinstance(data.get("category"), str):
            data["category"] = PatternCategory(data["category"])
        return cls(**data)


@dataclass
class ToolCallRecord:
    """Record of a tool call."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class KnowledgeEntry:
    """A knowledge entry in the knowledge base."""
    entry_id: str
    title: str
    content: str
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Create from dictionary."""
        return cls(**data)


class MemoryStore(ABC, Generic[T]):
    """Abstract base class for memory storage."""
    
    @abstractmethod
    def store(self, key: str, value: T) -> None:
        """Store a value."""
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> Optional[T]:
        """Retrieve a value."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value."""
        pass
    
    @abstractmethod
    def list_all(self) -> List[str]:
        """List all keys."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all values."""
        pass


class SQLiteMemoryStore(MemoryStore[T]):
    """SQLite-based memory storage.
    
    Provides persistent storage with search capabilities.
    """
    
    def __init__(
        self,
        db_path: Path,
        table_name: str,
        serializer: callable = json.dumps,
        deserializer: callable = json.loads,
    ):
        """Initialize SQLite store.
        
        Args:
            db_path: Path to SQLite database
            table_name: Table name for this store
            serializer: Function to serialize values
            deserializer: Function to deserialize values
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._serializer = serializer
        self._deserializer = deserializer
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database table."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def store(self, key: str, value: T) -> None:
        """Store a value."""
        serialized = self._serializer(value)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"""
                INSERT OR REPLACE INTO {self.table_name} (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (key, serialized))
            conn.commit()
    
    def retrieve(self, key: str) -> Optional[T]:
        """Retrieve a value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT value FROM {self.table_name} WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._deserializer(row[0])
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"DELETE FROM {self.table_name} WHERE key = ?",
                (key,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def list_all(self) -> List[str]:
        """List all keys."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT key FROM {self.table_name}")
            return [row[0] for row in cursor.fetchall()]
    
    def search(self, query: str, limit: int = 10) -> List[T]:
        """Search for values containing query."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT value FROM {self.table_name} WHERE value LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            )
            return [self._deserializer(row[0]) for row in cursor.fetchall()]
    
    def clear(self) -> None:
        """Clear all values."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
            conn.commit()
    
    def count(self) -> int:
        """Count total entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cursor.fetchone()[0]


class InMemoryStore(MemoryStore[T]):
    """In-memory storage for temporary data."""
    
    def __init__(self):
        """Initialize in-memory store."""
        self._data: Dict[str, T] = {}
    
    def store(self, key: str, value: T) -> None:
        """Store a value."""
        self._data[key] = value
    
    def retrieve(self, key: str) -> Optional[T]:
        """Retrieve a value."""
        return self._data.get(key)
    
    def delete(self, key: str) -> bool:
        """Delete a value."""
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    def list_all(self) -> List[str]:
        """List all keys."""
        return list(self._data.keys())
    
    def clear(self) -> None:
        """Clear all values."""
        self._data.clear()


class UnifiedMemory:
    """Unified memory system.
    
    Consolidates all memory types into a single, simplified interface:
    - Patterns: Test patterns, code patterns, tool patterns
    - Tool calls: Tool usage history
    - Knowledge: Domain knowledge, project knowledge
    
    Usage:
        memory = UnifiedMemory.get_instance()
        
        # Store a pattern
        pattern = BasePattern(
            pattern_id="test_1",
            name="Basic Test Pattern",
            category=PatternCategory.TEST,
            template="test template"
        )
        memory.store_pattern(pattern)
        
        # Record a tool call
        memory.record_tool_call("read_file", {"path": "/test.py"}, success=True)
        
        # Search patterns
        patterns = memory.search_patterns("test")
    """
    
    _instance: Optional["UnifiedMemory"] = None
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        app_name: str = "pyutagent",
    ):
        """Initialize unified memory.
        
        Args:
            data_dir: Data directory for storage
            app_name: Application name
        """
        self.data_dir = data_dir or Path.home() / f".{app_name}"
        self.db_path = self.data_dir / "memory.db"
        
        self._init_stores()
    
    def _init_stores(self) -> None:
        """Initialize all stores."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._patterns = SQLiteMemoryStore(
            db_path=self.db_path,
            table_name="patterns",
            serializer=lambda p: json.dumps(p.to_dict() if hasattr(p, 'to_dict') else p),
            deserializer=lambda d: BasePattern.from_dict(json.loads(d)),
        )
        
        self._tool_calls = SQLiteMemoryStore(
            db_path=self.db_path,
            table_name="tool_calls",
            serializer=lambda r: json.dumps(r.to_dict() if hasattr(r, 'to_dict') else r),
            deserializer=lambda d: ToolCallRecord.from_dict(json.loads(d)),
        )
        
        self._knowledge = SQLiteMemoryStore(
            db_path=self.db_path,
            table_name="knowledge",
            serializer=lambda k: json.dumps(k.to_dict() if hasattr(k, 'to_dict') else k),
            deserializer=lambda d: KnowledgeEntry.from_dict(json.loads(d)),
        )
        
        self._working = InMemoryStore()
    
    @classmethod
    def get_instance(cls) -> "UnifiedMemory":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
    
    # Pattern operations
    
    def store_pattern(self, pattern: BasePattern) -> None:
        """Store a pattern."""
        self._patterns.store(pattern.pattern_id, pattern)
        logger.debug(f"Stored pattern: {pattern.pattern_id}")
    
    def get_pattern(self, pattern_id: str) -> Optional[BasePattern]:
        """Get a pattern by ID."""
        return self._patterns.retrieve(pattern_id)
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a pattern."""
        return self._patterns.delete(pattern_id)
    
    def list_patterns(self, category: Optional[PatternCategory] = None) -> List[BasePattern]:
        """List all patterns, optionally filtered by category."""
        all_patterns = [
            self._patterns.retrieve(key)
            for key in self._patterns.list_all()
        ]
        
        if category:
            return [p for p in all_patterns if p and p.category == category]
        
        return [p for p in all_patterns if p]
    
    def search_patterns(self, query: str, limit: int = 10) -> List[BasePattern]:
        """Search patterns by query."""
        return self._patterns.search(query, limit)
    
    def get_best_patterns(
        self,
        category: Optional[PatternCategory] = None,
        min_success_rate: float = 0.5,
        limit: int = 10,
    ) -> List[BasePattern]:
        """Get best patterns by success rate."""
        patterns = self.list_patterns(category)
        
        filtered = [p for p in patterns if p.success_rate >= min_success_rate]
        
        sorted_patterns = sorted(
            filtered,
            key=lambda p: p.success_rate,
            reverse=True,
        )
        
        return sorted_patterns[:limit]
    
    # Tool call operations
    
    def record_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        duration_ms: int = 0,
    ) -> str:
        """Record a tool call.
        
        Returns:
            Record ID
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            success=success,
            error_message=error_message,
            duration_ms=duration_ms,
        )
        
        record_id = f"{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self._tool_calls.store(record_id, record)
        
        logger.debug(f"Recorded tool call: {record_id}")
        return record_id
    
    def get_tool_call_history(
        self,
        tool_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[ToolCallRecord]:
        """Get tool call history."""
        all_calls = [
            self._tool_calls.retrieve(key)
            for key in self._tool_calls.list_all()
        ]
        
        calls = [c for c in all_calls if c]
        
        if tool_name:
            calls = [c for c in calls if c.tool_name == tool_name]
        
        sorted_calls = sorted(
            calls,
            key=lambda c: c.timestamp,
            reverse=True,
        )
        
        return sorted_calls[:limit]
    
    def get_tool_success_rate(self, tool_name: str) -> float:
        """Get success rate for a tool."""
        calls = self.get_tool_call_history(tool_name)
        
        if not calls:
            return 0.0
        
        successes = sum(1 for c in calls if c.success)
        return successes / len(calls)
    
    # Knowledge operations
    
    def store_knowledge(self, entry: KnowledgeEntry) -> None:
        """Store a knowledge entry."""
        self._knowledge.store(entry.entry_id, entry)
        logger.debug(f"Stored knowledge: {entry.entry_id}")
    
    def get_knowledge(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry."""
        return self._knowledge.retrieve(entry_id)
    
    def delete_knowledge(self, entry_id: str) -> bool:
        """Delete a knowledge entry."""
        return self._knowledge.delete(entry_id)
    
    def search_knowledge(self, query: str, limit: int = 10) -> List[KnowledgeEntry]:
        """Search knowledge entries."""
        return self._knowledge.search(query, limit)
    
    # Working memory operations
    
    def set_working(self, key: str, value: Any) -> None:
        """Set a working memory value."""
        self._working.store(key, value)
    
    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a working memory value."""
        value = self._working.retrieve(key)
        return value if value is not None else default
    
    def clear_working(self) -> None:
        """Clear working memory."""
        self._working.clear()
    
    # Statistics
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "patterns_count": self._patterns.count(),
            "tool_calls_count": self._tool_calls.count(),
            "knowledge_count": self._knowledge.count(),
            "working_memory_keys": len(self._working.list_all()),
        }
    
    def clear_all(self) -> None:
        """Clear all persistent memory."""
        self._patterns.clear()
        self._tool_calls.clear()
        self._knowledge.clear()
        self._working.clear()
        logger.info("Cleared all memory")


def get_unified_memory() -> UnifiedMemory:
    """Get the global unified memory instance."""
    return UnifiedMemory.get_instance()
