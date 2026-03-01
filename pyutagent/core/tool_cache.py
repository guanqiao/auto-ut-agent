"""Tool result caching for improved performance.

This module provides caching capabilities for tool results:
- LRU cache with configurable size
- Content-based cache keys
- Cache invalidation
- Cache statistics
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry."""
    key: str
    value: T
    created_at: datetime
    expires_at: Optional[datetime]
    hit_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class CacheStats:
    """Statistics for the cache."""
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_size: int = 0
    current_entries: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """LRU (Least Recently Used) cache implementation.
    
    Features:
    - Configurable max size
    - TTL (Time To Live) support
    - Thread-safe operations
    - Statistics tracking
    """
    
    def __init__(
        self,
        maxsize: int = 100,
        ttl_seconds: Optional[int] = None
    ):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._stats = CacheStats()
        self._lock = None
    
    def _get_lock(self):
        if self._lock is None:
            import threading
            self._lock = threading.Lock()
        return self._lock
    
    def _compute_key(
        self,
        tool_name: str,
        args: tuple,
        kwargs: dict
    ) -> str:
        """Compute a cache key from tool name and arguments."""
        key_data = {
            "tool": tool_name,
            "args": self._serialize_args(args),
            "kwargs": self._serialize_kwargs(kwargs)
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def _serialize_args(self, args: tuple) -> List[Any]:
        """Serialize positional arguments for hashing."""
        result = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                result.append(arg)
            elif isinstance(arg, (list, tuple, set)):
                result.append(list(arg))
            elif isinstance(arg, dict):
                result.append(dict(sorted(arg.items())))
            else:
                result.append(str(arg))
        return result
    
    def _serialize_kwargs(self, kwargs: dict) -> Dict[str, Any]:
        """Serialize keyword arguments for hashing."""
        return {
            k: self._serialize_args((v,))[0]
            for k, v in sorted(kwargs.items())
        }
    
    def get(
        self,
        tool_name: str,
        args: tuple,
        kwargs: dict
    ) -> Optional[T]:
        """Get a cached result.
        
        Args:
            tool_name: Name of the tool
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cached result or None
        """
        key = self._compute_key(tool_name, args, kwargs)
        
        with self._get_lock():
            if key not in self._cache:
                self._stats.total_misses += 1
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired:
                del self._cache[key]
                self._stats.total_misses += 1
                self._stats.total_evictions += 1
                return None
            
            self._cache.move_to_end(key)
            entry.hit_count += 1
            entry.last_accessed = datetime.now()
            self._stats.total_hits += 1
            
            logger.debug(f"[Cache] Hit for {tool_name} - Key: {key[:8]}...")
            return entry.value
    
    def set(
        self,
        tool_name: str,
        args: tuple,
        kwargs: dict,
        value: T,
        ttl_seconds: Optional[int] = None
    ):
        """Set a cached result.
        
        Args:
            tool_name: Name of the tool
            args: Positional arguments
            kwargs: Keyword arguments
            value: Value to cache
            ttl_seconds: Optional TTL override
        """
        key = self._compute_key(tool_name, args, kwargs)
        
        ttl = ttl_seconds or self.ttl_seconds
        expires_at = None
        if ttl:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
            last_accessed=datetime.now()
        )
        
        with self._get_lock():
            if key in self._cache:
                del self._cache[key]
            
            self._cache[key] = entry
            self._stats.current_entries = len(self._cache)
            
            while len(self._cache) > self.maxsize:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.total_evictions += 1
            
            self._stats.current_entries = len(self._cache)
        
        logger.debug(f"[Cache] Set for {tool_name} - Key: {key[:8]}...")
    
    def invalidate(
        self,
        tool_name: str,
        args: tuple,
        kwargs: dict
    ) -> bool:
        """Invalidate a specific cache entry.
        
        Args:
            tool_name: Name of the tool
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            True if entry was invalidated
        """
        key = self._compute_key(tool_name, args, kwargs)
        
        with self._get_lock():
            if key in self._cache:
                del self._cache[key]
                self._stats.current_entries = len(self._cache)
                logger.debug(f"[Cache] Invalidated - Key: {key[:8]}...")
                return True
        return False
    
    def invalidate_tool(self, tool_name: str) -> int:
        """Invalidate all cache entries for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        with self._get_lock():
            keys_to_remove = []
            for key, entry in self._cache.items():
                if entry.metadata.get("tool_name") == tool_name:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
            
            self._stats.current_entries = len(self._cache)
        
        logger.debug(f"[Cache] Invalidated {count} entries for {tool_name}")
        return count
    
    def clear(self):
        """Clear all cache entries."""
        with self._get_lock():
            count = len(self._cache)
            self._cache.clear()
            self._stats.current_entries = 0
        
        logger.info(f"[Cache] Cleared {count} entries")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def get_entries(self) -> List[Dict[str, Any]]:
        """Get all cache entries info."""
        with self._get_lock():
            return [
                {
                    "key": entry.key[:8] + "...",
                    "created_at": entry.created_at.isoformat(),
                    "hit_count": entry.hit_count,
                    "is_expired": entry.is_expired
                }
                for entry in self._cache.values()
            ]


class ToolResultCache:
    """Cache for tool results with smart invalidation.
    
    Features:
    - Content-based cache keys
    - Dependency-aware invalidation
    - File modification tracking
    - Cache warming
    """
    
    def __init__(
        self,
        maxsize: int = 100,
        default_ttl: int = 300
    ):
        self.cache = LRUCache[Dict[str, Any]](maxsize=maxsize, ttl_seconds=default_ttl)
        self.default_ttl = default_ttl
        
        self._file_hashes: Dict[str, str] = {}
        self._tool_dependencies: Dict[str, List[str]] = {}
    
    def compute_content_hash(self, content: str) -> str:
        """Compute hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def compute_file_hash(self, file_path: str) -> str:
        """Compute hash for a file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return ""
    
    def register_file_dependency(
        self,
        tool_name: str,
        file_path: str
    ):
        """Register a file dependency for a tool."""
        if tool_name not in self._tool_dependencies:
            self._tool_dependencies[tool_name] = []
        if file_path not in self._tool_dependencies[tool_name]:
            self._tool_dependencies[tool_name].append(file_path)
        
        self._file_hashes[file_path] = self.compute_file_hash(file_path)
    
    def check_file_changed(self, file_path: str) -> bool:
        """Check if a file has changed since last hash."""
        current_hash = self.compute_file_hash(file_path)
        stored_hash = self._file_hashes.get(file_path, "")
        return current_hash != stored_hash
    
    async def get_or_execute(
        self,
        tool_name: str,
        executor: Callable,
        *args,
        use_cache: bool = True,
        ttl: Optional[int] = None,
        **kwargs
    ) -> Tuple[Any, bool]:
        """Get cached result or execute tool.
        
        Args:
            tool_name: Name of the tool
            executor: Async function to execute
            *args: Positional arguments
            use_cache: Whether to use cache
            ttl: Cache TTL override
            **kwargs: Keyword arguments
            
        Returns:
            Tuple of (result, from_cache)
        """
        if use_cache:
            cached = self.cache.get(tool_name, args, kwargs)
            if cached is not None:
                return cached, True
        
        if tool_name in self._tool_dependencies:
            for dep_file in self._tool_dependencies[tool_name]:
                if self.check_file_changed(dep_file):
                    self.cache.invalidate_tool(tool_name)
                    break
        
        import asyncio
        if asyncio.iscoroutinefunction(executor):
            result = await executor(*args, **kwargs)
        else:
            result = executor(*args, **kwargs)
        
        if use_cache:
            self.cache.set(tool_name, args, kwargs, result, ttl)
        
        return result, False
    
    def invalidate_file(self, file_path: str) -> int:
        """Invalidate all cache entries that depend on a file.
        
        Args:
            file_path: Path to the changed file
            
        Returns:
            Number of tools invalidated
        """
        count = 0
        for tool_name, deps in self._tool_dependencies.items():
            if file_path in deps:
                count += self.cache.invalidate_tool(tool_name)
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        return {
            "hits": stats.total_hits,
            "misses": stats.total_misses,
            "hit_rate": stats.hit_rate,
            "evictions": stats.total_evictions,
            "current_entries": stats.current_entries,
            "file_dependencies": len(self._file_hashes)
        }
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self._file_hashes.clear()


def create_tool_cache(
    maxsize: int = 100,
    ttl: int = 300
) -> ToolResultCache:
    """Create a tool result cache.
    
    Args:
        maxsize: Maximum cache entries
        ttl: Default TTL in seconds
        
    Returns:
        Configured ToolResultCache
    """
    return ToolResultCache(maxsize=maxsize, default_ttl=ttl)
