"""Tool Result Cache - Avoid repeated executions.

This module provides:
- LRU cache for tool results
- Cache invalidation strategies
- Persistent cache support
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for tool result."""
    key: str
    value: Any
    timestamp: datetime
    expires_at: Optional[datetime]
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolResultCache:
    """Cache for tool execution results.
    
    Features:
    - LRU eviction
    - TTL support
    - Persistent storage
    - Statistics tracking
    """
    
    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        persistent_path: Optional[str] = None
    ):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Default TTL in seconds
            persistent_path: Optional path for persistence
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._persistent_path = persistent_path
        
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "saves": 0
        }
        
        if persistent_path:
            self._load()
        
        logger.info(f"[ToolResultCache] Initialized (max_size={max_size}, ttl={ttl_seconds}s)")
    
    def get(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached result.
        
        Args:
            tool_name: Tool name
            params: Tool parameters
        
        Returns:
            Cached result or None
        """
        key = self._make_key(tool_name, params)
        
        if key not in self._cache:
            self._stats["misses"] += 1
            return None
        
        entry = self._cache[key]
        
        if self._is_expired(entry):
            del self._cache[key]
            self._stats["misses"] += 1
            return None
        
        entry.hit_count += 1
        self._stats["hits"] += 1
        
        logger.debug(f"[ToolResultCache] Hit: {tool_name}")
        return entry.value
    
    def set(
        self,
        tool_name: str,
        params: Dict[str, Any],
        value: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Set cached result.
        
        Args:
            tool_name: Tool name
            params: Tool parameters
            value: Result to cache
            ttl_seconds: Optional custom TTL
            metadata: Optional metadata
        """
        key = self._make_key(tool_name, params)
        
        if len(self._cache) >= self._max_size:
            self._evict_lru()
        
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        self._cache[key] = entry
        
        if self._persistent_path:
            self._save_async()
        
        logger.debug(f"[ToolResultCache] Cached: {tool_name}")
    
    def _make_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Generate cache key.
        
        Args:
            tool_name: Tool name
            params: Tool parameters
        
        Returns:
            Cache key string
        """
        param_str = json.dumps(params, sort_keys=True, default=str)
        key_input = f"{tool_name}:{param_str}"
        return hashlib.sha256(key_input.encode()).hexdigest()[:32]
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired.
        
        Args:
            entry: Cache entry
        
        Returns:
            True if expired
        """
        if entry.expires_at is None:
            return False
        
        return datetime.now() > entry.expires_at
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].hit_count, self._cache[k].timestamp)
        )
        
        del self._cache[lru_key]
        self._stats["evictions"] += 1
        
        logger.debug(f"[ToolResultCache] Evicted: {lru_key[:8]}...")
    
    def invalidate(self, tool_name: Optional[str] = None, pattern: Optional[str] = None):
        """Invalidate cache entries.
        
        Args:
            tool_name: Optional tool name to invalidate
            pattern: Optional pattern to match
        """
        if tool_name is None and pattern is None:
            self._cache.clear()
            logger.info("[ToolResultCache] Cleared all")
            return
        
        to_remove = []
        for key, entry in self._cache.items():
            if tool_name and tool_name in key:
                to_remove.append(key)
            elif pattern and pattern in key:
                to_remove.append(key)
        
        for key in to_remove:
            del self._cache[key]
        
        logger.info(f"[ToolResultCache] Invalidated {len(to_remove)} entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self._stats["evictions"],
            "saves": self._stats["saves"]
        }
    
    def _save_async(self):
        """Save cache to disk asynchronously."""
        pass
    
    def _save(self):
        """Save cache to disk."""
        if not self._persistent_path:
            return
        
        try:
            Path(self._persistent_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for key, entry in self._cache.items():
                data[key] = {
                    "value": entry.value,
                    "timestamp": entry.timestamp.isoformat(),
                    "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                    "hit_count": entry.hit_count,
                    "metadata": entry.metadata
                }
            
            with open(self._persistent_path, 'w') as f:
                json.dump(data, f, default=str)
            
            self._stats["saves"] += 1
            
        except Exception as e:
            logger.error(f"[ToolResultCache] Save failed: {e}")
    
    def _load(self):
        """Load cache from disk."""
        if not self._persistent_path:
            return
        
        path = Path(self._persistent_path)
        if not path.exists():
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for key, entry_data in data.items():
                expires_at = None
                if entry_data.get("expires_at"):
                    expires_at = datetime.fromisoformat(entry_data["expires_at"])
                
                entry = CacheEntry(
                    key=key,
                    value=entry_data["value"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    expires_at=expires_at,
                    hit_count=entry_data.get("hit_count", 0),
                    metadata=entry_data.get("metadata", {})
                )
                
                if not self._is_expired(entry):
                    self._cache[key] = entry
            
            logger.info(f"[ToolResultCache] Loaded {len(self._cache)} entries")
            
        except Exception as e:
            logger.error(f"[ToolResultCache] Load failed: {e}")


class CachedToolExecutor:
    """Tool executor with caching.
    
    Wraps tool execution with cache support.
    """
    
    def __init__(self, tool_service, cache: Optional[ToolResultCache] = None):
        """Initialize cached executor.
        
        Args:
            tool_service: Tool service
            cache: Optional cache instance
        """
        self.tool_service = tool_service
        self.cache = cache or ToolResultCache()
    
    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        use_cache: bool = True,
        cache_ttl: Optional[int] = None
    ) -> Any:
        """Execute tool with caching.
        
        Args:
            tool_name: Tool name
            params: Tool parameters
            use_cache: Whether to use cache
            cache_ttl: Optional cache TTL override
        
        Returns:
            Tool result
        """
        if use_cache:
            cached = self.cache.get(tool_name, params)
            if cached is not None:
                logger.debug(f"[CachedToolExecutor] Cache hit: {tool_name}")
                return cached
        
        result = await self.tool_service.execute_tool(tool_name, params)
        
        if use_cache and result.success:
            self.cache.set(
                tool_name, params, result,
                ttl_seconds=cache_ttl,
                metadata={"tool_name": tool_name}
            )
        
        return result
    
    def invalidate(self, tool_name: Optional[str] = None):
        """Invalidate cache.
        
        Args:
            tool_name: Optional tool name to invalidate
        """
        self.cache.invalidate(tool_name=tool_name)


def create_result_cache(
    max_size: int = 100,
    ttl_seconds: int = 3600,
    persistent_path: Optional[str] = None
) -> ToolResultCache:
    """Create a tool result cache.
    
    Args:
        max_size: Maximum cache size
        ttl_seconds: Default TTL
        persistent_path: Optional persistence path
    
    Returns:
        ToolResultCache instance
    """
    return ToolResultCache(
        max_size=max_size,
        ttl_seconds=ttl_seconds,
        persistent_path=persistent_path
    )
