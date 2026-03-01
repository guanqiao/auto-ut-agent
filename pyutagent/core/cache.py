"""Caching utilities for PyUT Agent.

This module provides caching mechanisms to improve performance
by avoiding redundant computations and I/O operations.
"""

import hashlib
import logging
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FileCache:
    """Cache for file contents with modification time checking."""
    
    def __init__(self, max_size: int = 100):
        """Initialize file cache.
        
        Args:
            max_size: Maximum number of files to cache
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._access_count: Dict[str, int] = {}
    
    def get(self, file_path: str) -> Optional[str]:
        """Get cached file content if valid.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cached content or None if not cached or stale
        """
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        cache_key = str(path.resolve())
        cached = self._cache.get(cache_key)
        
        if cached is None:
            return None
        
        # Check if file has been modified
        current_mtime = path.stat().st_mtime
        if current_mtime != cached['mtime']:
            logger.debug(f"[FileCache] Cache stale for: {file_path}")
            del self._cache[cache_key]
            del self._access_count[cache_key]
            return None
        
        self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
        logger.debug(f"[FileCache] Cache hit for: {file_path}")
        return cached['content']
    
    def set(self, file_path: str, content: str) -> None:
        """Cache file content.
        
        Args:
            file_path: Path to the file
            content: File content to cache
        """
        path = Path(file_path)
        cache_key = str(path.resolve())
        
        # Evict least recently used if cache is full
        if len(self._cache) >= self._max_size and cache_key not in self._cache:
            self._evict_lru()
        
        self._cache[cache_key] = {
            'content': content,
            'mtime': path.stat().st_mtime,
            'cached_at': datetime.now()
        }
        self._access_count[cache_key] = 1
        logger.debug(f"[FileCache] Cached: {file_path}")
    
    def invalidate(self, file_path: str) -> None:
        """Invalidate cache for a specific file.
        
        Args:
            file_path: Path to the file
        """
        path = Path(file_path)
        cache_key = str(path.resolve())
        
        if cache_key in self._cache:
            del self._cache[cache_key]
            del self._access_count[cache_key]
            logger.debug(f"[FileCache] Invalidated: {file_path}")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_count.clear()
        logger.info("[FileCache] Cache cleared")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_count:
            return
        
        lru_key = min(self._access_count, key=self._access_count.get)
        del self._cache[lru_key]
        del self._access_count[lru_key]
        logger.debug(f"[FileCache] Evicted LRU: {lru_key}")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'total_accesses': sum(self._access_count.values())
        }


class Memoize:
    """Decorator for memoizing function results."""
    
    def __init__(self, max_size: int = 128, ttl: Optional[int] = None):
        """Initialize memoization decorator.
        
        Args:
            max_size: Maximum cache size
            ttl: Time to live in seconds (None for no expiration)
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._max_size = max_size
        self._ttl = ttl
    
    def __call__(self, func: Callable) -> Callable:
        """Apply memoization to function.
        
        Args:
            func: Function to memoize
            
        Returns:
            Memoized function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = self._make_key(args, kwargs)
            
            # Check if result is cached and not expired
            if key in self._cache:
                if self._ttl is not None:
                    cached_time = self._timestamps.get(key)
                    if cached_time and datetime.now() - cached_time < timedelta(seconds=self._ttl):
                        logger.debug(f"[Memoize] Cache hit for {func.__name__}")
                        return self._cache[key]
                    else:
                        # Expired
                        del self._cache[key]
                        del self._timestamps[key]
                else:
                    logger.debug(f"[Memoize] Cache hit for {func.__name__}")
                    return self._cache[key]
            
            # Compute and cache result
            result = func(*args, **kwargs)
            
            # Evict oldest if cache is full
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._timestamps, key=self._timestamps.get)
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = result
            self._timestamps[key] = datetime.now()
            
            return result
        
        # Attach cache management methods
        wrapper.cache_clear = self._clear
        wrapper.cache_info = self._get_info
        
        return wrapper
    
    def _make_key(self, args: tuple, kwargs: dict) -> str:
        """Create cache key from arguments.
        
        Args:
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._timestamps.clear()
    
    def _get_info(self) -> Dict[str, int]:
        """Get cache information.
        
        Returns:
            Dictionary with cache info
        """
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'ttl': self._ttl
        }


# Global file cache instance
_global_file_cache: Optional[FileCache] = None


def get_file_cache() -> FileCache:
    """Get global file cache instance.
    
    Returns:
        FileCache instance
    """
    global _global_file_cache
    if _global_file_cache is None:
        _global_file_cache = FileCache()
    return _global_file_cache


def clear_file_cache() -> None:
    """Clear global file cache."""
    global _global_file_cache
    if _global_file_cache is not None:
        _global_file_cache.clear()
