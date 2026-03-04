"""多级缓存 - L1（内存）+ L2（磁盘）"""
import asyncio
import hashlib
import json
import os
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    ttl: int = -1  # -1 表示永不过期
    created_at: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl < 0:
            return False
        if self.ttl == 0:
            return True  # TTL 为 0 表示立即过期
        return time.time() - self.created_at > self.ttl


@dataclass
class CacheStats:
    """缓存统计"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l1_evictions: int = 0
    l2_evictions: int = 0
    
    @property
    def total_hits(self) -> int:
        """总命中次数"""
        return self.l1_hits + self.l2_hits
    
    @property
    def total_requests(self) -> int:
        """总请求次数"""
        return self.l1_hits + self.l1_misses
    
    @property
    def hit_rate(self) -> float:
        """命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests


@dataclass
class CacheConfig:
    """缓存配置"""
    l1_capacity: int = 1000
    l2_storage_path: Optional[str] = None
    enable_compression: bool = False
    default_ttl: int = 3600


class CacheLevel:
    """缓存级别枚举"""
    L1 = "L1"
    L2 = "L2"
    NONE = "NONE"


class L1Cache:
    """L1 缓存 - 内存缓存"""
    
    def __init__(self, capacity: int = 1000):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._capacity = capacity
        self._lock = asyncio.Lock()
    
    @property
    def capacity(self) -> int:
        """缓存容量"""
        return self._capacity
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """放入缓存"""
        async with self._lock:
            entry = CacheEntry(key=key, value=value, ttl=ttl if ttl is not None else -1)
            
            if key in self._cache:
                self._cache.move_to_end(key)
            
            self._cache[key] = entry
            
            # LRU 淘汰
            while len(self._cache) > self._capacity:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
    
    async def get(self, key: str) -> Optional[Any]:
        """从缓存获取"""
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # 检查过期
            if entry.is_expired():
                del self._cache[key]
                return None
            
            # 移到末尾（LRU）
            self._cache.move_to_end(key)
            return entry.value
    
    async def contains(self, key: str) -> bool:
        """检查是否包含"""
        async with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return False
            
            return True
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """清空缓存"""
        async with self._lock:
            self._cache.clear()
    
    async def size(self) -> int:
        """获取缓存大小"""
        async with self._lock:
            return len(self._cache)


class L2Cache:
    """L2 缓存 - 磁盘缓存"""
    
    def __init__(self, storage_path: str, enable_compression: bool = False):
        self.storage_path = storage_path
        self.enable_compression = enable_compression
        self._lock = asyncio.Lock()
        
        if storage_path:
            os.makedirs(storage_path, exist_ok=True)
        
        self._index_file = os.path.join(storage_path, "cache_index.json") if storage_path else None
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
    
    def _get_cache_file(self, key: str) -> str:
        """获取缓存文件路径"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.storage_path, f"{key_hash}.cache")
    
    def _load_index(self):
        """加载索引"""
        if self._index_file and os.path.exists(self._index_file):
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
                logger.debug(f"Loaded L2 cache index with {len(self._index)} entries")
            except Exception as e:
                logger.error(f"Failed to load L2 cache index: {e}")
                self._index = {}
    
    def _save_index(self):
        """保存索引"""
        if self._index_file:
            try:
                with open(self._index_file, 'w', encoding='utf-8') as f:
                    json.dump(self._index, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save L2 cache index: {e}")
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """放入缓存"""
        async with self._lock:
            cache_file = self._get_cache_file(key)
            
            # 序列化值
            value_data = json.dumps(value)
            
            # 压缩
            if self.enable_compression:
                value_data = zlib.compress(value_data.encode('utf-8'))
                value_bytes = value_data if isinstance(value_data, bytes) else value_data.encode('utf-8')
            else:
                value_bytes = value_data.encode('utf-8')
            
            # 写入文件
            with open(cache_file, 'wb') as f:
                f.write(value_bytes)
            
            # 更新索引
            self._index[key] = {
                'file': cache_file,
                'ttl': ttl if ttl is not None else -1,
                'created_at': time.time(),
                'size': len(value_bytes)
            }
            self._save_index()
            
            logger.debug(f"L2 cache put: {key[:16]}...")
    
    async def get(self, key: str) -> Optional[Any]:
        """从缓存获取"""
        async with self._lock:
            if key not in self._index:
                return None
            
            meta = self._index[key]
            
            # 检查过期
            if meta['ttl'] >= 0:
                if time.time() - meta['created_at'] > meta['ttl']:
                    # 在释放锁后删除
                    key_to_delete = key
                    should_delete = True
                else:
                    should_delete = False
            else:
                should_delete = False
            
            if should_delete:
                # 释放锁后再删除，避免死锁
                pass
            else:
                cache_file = meta['file']
                
                if not os.path.exists(cache_file):
                    del self._index[key]
                    self._save_index()
                    return None
                
                try:
                    with open(cache_file, 'rb') as f:
                        data = f.read()
                    
                    # 解压缩
                    if self.enable_compression:
                        data = zlib.decompress(data).decode('utf-8')
                    else:
                        data = data.decode('utf-8')
                    
                    value = json.loads(data)
                    logger.debug(f"L2 cache get: {key[:16]}...")
                    return value
                except Exception as e:
                    logger.error(f"Failed to read L2 cache: {e}")
                    return None
        
        # 在锁外执行删除操作
        if should_delete:
            await self.delete(key_to_delete)
            return None
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        async with self._lock:
            if key not in self._index:
                return False
            
            meta = self._index[key]
            cache_file = meta['file']
            
            # 删除文件
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            # 删除索引
            del self._index[key]
            self._save_index()
            
            logger.debug(f"L2 cache delete: {key[:16]}...")
            return True
    
    async def clear(self) -> None:
        """清空缓存"""
        async with self._lock:
            # 删除所有缓存文件
            for meta in self._index.values():
                cache_file = meta['file']
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            
            self._index.clear()
            self._save_index()
            
            logger.debug("L2 cache cleared")
    
    async def size(self) -> int:
        """获取缓存大小"""
        async with self._lock:
            return len(self._index)
    
    async def cleanup(self) -> None:
        """清理资源"""
        self._save_index()
        logger.debug("L2 cache cleanup completed")


class MultiLevelCache:
    """多级缓存 - L1（内存）+ L2（磁盘）"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        
        # L1 缓存
        self._l1_cache = L1Cache(capacity=self.config.l1_capacity)
        
        # L2 缓存
        self._l2_cache = None
        if self.config.l2_storage_path:
            self._l2_cache = L2Cache(
                storage_path=self.config.l2_storage_path,
                enable_compression=self.config.enable_compression
            )
        
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        async with self._lock:
            # 先查 L1
            value = await self._l1_cache.get(key)
            if value is not None:
                self.stats.l1_hits += 1
                logger.debug(f"MultiLevelCache L1 hit: {key[:16]}...")
                return value
            
            self.stats.l1_misses += 1
            
            # 再查 L2
            if self._l2_cache:
                value = await self._l2_cache.get(key)
                if value is not None:
                    self.stats.l2_hits += 1
                    
                    # 回写到 L1
                    await self._l1_cache.put(key, value)
                    
                    logger.debug(f"MultiLevelCache L2 hit: {key[:16]}...")
                    return value
            
            self.stats.l2_misses += 1
            logger.debug(f"MultiLevelCache miss: {key[:16]}...")
            return None
    
    async def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """放入缓存"""
        async with self._lock:
            # 同时写入 L1 和 L2
            effective_ttl = ttl if ttl is not None else self.config.default_ttl
            
            await self._l1_cache.put(key, value, ttl=effective_ttl)
            
            if self._l2_cache:
                await self._l2_cache.put(key, value, ttl=effective_ttl)
            
            logger.debug(f"MultiLevelCache put: {key[:16]}...")
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        async with self._lock:
            deleted = False
            
            if await self._l1_cache.delete(key):
                deleted = True
            
            if self._l2_cache and await self._l2_cache.delete(key):
                deleted = True
            
            if deleted:
                logger.debug(f"MultiLevelCache delete: {key[:16]}...")
            
            return deleted
    
    async def clear(self) -> None:
        """清空缓存"""
        async with self._lock:
            await self._l1_cache.clear()
            
            if self._l2_cache:
                await self._l2_cache.clear()
            
            logger.debug("MultiLevelCache cleared")
    
    async def size(self) -> int:
        """获取缓存大小"""
        async with self._lock:
            l1_size = await self._l1_cache.size()
            l2_size = await self._l2_cache.size() if self._l2_cache else 0
            return l1_size + l2_size
    
    async def warmup(self, data: Dict[str, Any]) -> None:
        """预热缓存"""
        logger.debug(f"Warming up cache with {len(data)} entries")
        
        for key, value in data.items():
            await self.put(key, value)
        
        logger.debug("Cache warmup completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "l1_capacity": self.config.l1_capacity,
            "l1_size": len(self._l1_cache._cache),
            "l2_size": len(self._l2_cache._index) if self._l2_cache else 0,
            "l1_hits": self.stats.l1_hits,
            "l1_misses": self.stats.l1_misses,
            "l2_hits": self.stats.l2_hits,
            "l2_misses": self.stats.l2_misses,
            "total_hits": self.stats.total_hits,
            "total_requests": self.stats.total_requests,
            "hit_rate": self.stats.hit_rate
        }
    
    async def cleanup(self) -> None:
        """清理资源"""
        if self._l2_cache:
            await self._l2_cache.cleanup()
        logger.debug("MultiLevelCache cleanup completed")
