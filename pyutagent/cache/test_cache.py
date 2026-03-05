"""Cache mechanism for test generation results."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with TTL support."""
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600  # Default 1 hour
    ):
        """Initialize cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            ttl_seconds: Time to live in seconds
        """
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.ttl_seconds = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if entry is expired.
        
        Returns:
            True if expired
        """
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'ttl_seconds': self.ttl_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            CacheEntry instance
        """
        entry = cls(
            key=data['key'],
            value=data['value'],
            ttl_seconds=data['ttl_seconds']
        )
        entry.created_at = datetime.fromisoformat(data['created_at'])
        return entry


class TestCache:
    """Cache for test generation results.
    
    Features:
    - Memory cache with TTL
    - Persistent cache (JSON file)
    - LRU eviction
    - File hash-based cache keys
    """
    
    def __init__(
        self,
        cache_dir: str,
        max_size: int = 1000,
        default_ttl: int = 3600
    ):
        """Initialize test cache.
        
        Args:
            cache_dir: Directory to store persistent cache
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Memory cache
        self._cache: Dict[str, CacheEntry] = {}
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persistent cache
        self._load_persistent_cache()
        
        logger.info(f"TestCache initialized: {cache_dir}, max_size={max_size}")
    
    def _generate_cache_key(
        self,
        source_file: str,
        source_hash: str,
        test_framework: str,
        mock_framework: str,
        target_coverage: float
    ) -> str:
        """Generate cache key.
        
        Args:
            source_file: Source file path
            source_hash: Source file hash
            test_framework: Test framework (junit5, testng, etc.)
            mock_framework: Mock framework (mockito, etc.)
            target_coverage: Target coverage ratio
            
        Returns:
            Cache key
        """
        key_string = (
            f"{source_file}:{source_hash}:{test_framework}:"
            f"{mock_framework}:{target_coverage}"
        )
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute file hash.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash
        """
        if not file_path.exists():
            return ""
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute hash for {file_path}: {e}")
            return ""
    
    def get(
        self,
        source_file: str,
        source_path: Path,
        test_framework: str = 'junit5',
        mock_framework: str = 'mockito',
        target_coverage: float = 0.8
    ) -> Optional[str]:
        """Get cached test code.
        
        Args:
            source_file: Source file path
            source_path: Source file path (absolute)
            test_framework: Test framework
            mock_framework: Mock framework
            target_coverage: Target coverage
            
        Returns:
            Cached test code or None
        """
        source_hash = self._compute_file_hash(source_path)
        key = self._generate_cache_key(
            source_file, source_hash, test_framework, mock_framework, target_coverage
        )
        
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                logger.debug(f"Cache hit for {source_file}")
                return entry.value
            else:
                # Remove expired entry
                del self._cache[key]
                logger.debug(f"Cache entry expired for {source_file}")
        
        logger.debug(f"Cache miss for {source_file}")
        return None
    
    def set(
        self,
        source_file: str,
        source_path: Path,
        test_framework: str,
        mock_framework: str,
        target_coverage: float,
        test_code: str
    ):
        """Cache test code.
        
        Args:
            source_file: Source file path
            source_path: Source file path (absolute)
            test_framework: Test framework
            mock_framework: Mock framework
            target_coverage: Target coverage
            test_code: Test code to cache
        """
        source_hash = self._compute_file_hash(source_path)
        key = self._generate_cache_key(
            source_file, source_hash, test_framework, mock_framework, target_coverage
        )
        
        # Evict oldest if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        # Create and store entry
        entry = CacheEntry(key=key, value=test_code, ttl_seconds=self.default_ttl)
        self._cache[key] = entry
        
        # Save to persistent storage
        self._save_persistent_cache()
        
        logger.debug(f"Cached test for {source_file}")
    
    def _evict_oldest(self):
        """Evict oldest cache entry (LRU)."""
        if not self._cache:
            return
        
        # Find oldest entry
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        
        # Remove it
        del self._cache[oldest_key]
        logger.debug(f"Evicted oldest cache entry: {oldest_key}")
    
    def _save_persistent_cache(self):
        """Save cache to persistent storage."""
        cache_file = self.cache_dir / "cache.json"
        
        # Filter out expired entries
        data = {
            key: entry.to_dict()
            for key, entry in self._cache.items()
            if not entry.is_expired()
        }
        
        try:
            cache_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
            logger.debug(f"Saved {len(data)} cache entries to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        cache_file = self.cache_dir / "cache.json"
        if not cache_file.exists():
            logger.debug("No persistent cache found")
            return
        
        try:
            data = json.loads(cache_file.read_text(encoding='utf-8'))
            loaded = 0
            
            for key, entry_data in data.items():
                entry = CacheEntry.from_dict(entry_data)
                if not entry.is_expired():
                    self._cache[key] = entry
                    loaded += 1
            
            logger.info(f"Loaded {loaded} cache entries from {cache_file}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            cache_file.unlink()
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        now = datetime.now()
        valid_entries = sum(
            1 for entry in self._cache.values()
            if not entry.is_expired()
        )
        
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self._cache) - valid_entries,
            'max_size': self.max_size,
            'usage_percent': (len(self._cache) / self.max_size * 100) if self.max_size > 0 else 0,
            'cache_dir': str(self.cache_dir)
        }
    
    def remove(self, source_file: str, source_path: Path) -> bool:
        """Remove cache entry for a file.
        
        Args:
            source_file: Source file path
            source_path: Source file path (absolute)
            
        Returns:
            True if removed
        """
        source_hash = self._compute_file_hash(source_path)
        
        # Find and remove matching keys
        removed = False
        keys_to_remove = []
        
        for key in self._cache.keys():
            if source_hash in key:
                keys_to_remove.append(key)
                removed = True
        
        for key in keys_to_remove:
            del self._cache[key]
        
        if removed:
            self._save_persistent_cache()
            logger.debug(f"Removed {len(keys_to_remove)} cache entries for {source_file}")
        
        return removed
    
    def invalidate_all(self):
        """Invalidate all cache entries without deleting them."""
        for entry in self._cache.values():
            entry.ttl_seconds = 0  # Mark as expired
        
        self._save_persistent_cache()
        logger.info("All cache entries invalidated")
