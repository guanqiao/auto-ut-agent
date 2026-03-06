"""多级缓存系统 - 用于智能化分析结果缓存

提供 L1(内存) + L2(磁盘) 两级缓存，支持:
- 快速内存访问
- 持久化磁盘存储
- TTL 过期策略
- LRU 淘汰策略
- 缓存统计和监控
"""

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float  # 时间戳
    expires_at: Optional[float]  # 过期时间戳
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """更新访问时间"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """从字典创建"""
        return cls(**data)


class L1MemoryCache:
    """L1 内存缓存 - 快速访问"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认 TTL(秒)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"[L1MemoryCache] Initialized with max_size={max_size}, default_ttl={default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                self._remove(key)
                self.misses += 1
                return None
            
            # 更新访问统计
            entry.touch()
            self.cache.move_to_end(key)  # LRU: 最近访问的移到末尾
            self.hits += 1
            
            logger.debug(f"[L1MemoryCache] Hit for key={key}")
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        with self.lock:
            # 如果已存在，先删除
            if key in self.cache:
                self._remove(key)
            
            # 如果缓存已满，删除最久未使用的
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # 计算过期时间
            expires_at = None
            if ttl is not None or self.default_ttl > 0:
                ttl = ttl if ttl is not None else self.default_ttl
                expires_at = time.time() + ttl
            
            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=expires_at
            )
            
            self.cache[key] = entry
            logger.debug(f"[L1MemoryCache] Set key={key}, ttl={ttl}")
    
    def remove(self, key: str):
        """删除缓存"""
        with self.lock:
            self._remove(key)
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            logger.info("[L1MemoryCache] Cache cleared")
    
    def _remove(self, key: str):
        """内部删除方法 (需要持有锁)"""
        if key in self.cache:
            del self.cache[key]
    
    def _evict_lru(self):
        """淘汰最久未使用的条目 (需要持有锁)"""
        if self.cache:
            # OrderedDict 第一个是最久未使用的
            oldest_key = next(iter(self.cache))
            self._remove(oldest_key)
            logger.debug(f"[L1MemoryCache] Evicted LRU key={oldest_key}")
    
    def contains(self, key: str) -> bool:
        """检查键是否存在且未过期"""
        with self.lock:
            if key not in self.cache:
                return False
            
            entry = self.cache[key]
            if entry.is_expired():
                self._remove(key)
                return False
            
            return True
    
    def size(self) -> int:
        """获取缓存大小"""
        with self.lock:
            return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.2f}%",
                "default_ttl": self.default_ttl
            }


class L2DiskCache:
    """L2 磁盘缓存 - 持久化存储"""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 100, default_ttl: int = 86400):
        """
        Args:
            cache_dir: 缓存目录
            max_size_mb: 最大缓存大小 (MB)
            default_ttl: 默认 TTL(秒)
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.db_path = self.cache_dir / "cache.db"
        self.lock = threading.RLock()
        
        # 确保目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_db()
        
        logger.info(f"[L2DiskCache] Initialized at {cache_dir}, max_size={max_size_mb}MB")
    
    def _init_db(self):
        """初始化数据库"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 创建缓存表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed REAL NOT NULL
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON cache(last_accessed)")
            
            conn.commit()
            conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 查询缓存
            cursor.execute("""
                SELECT value, expires_at FROM cache 
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (key, time.time()))
            
            row = cursor.fetchone()
            
            if row is None:
                conn.close()
                logger.debug(f"[L2DiskCache] Miss for key={key}")
                return None
            
            value_str, expires_at = row
            
            # 更新访问统计
            cursor.execute("""
                UPDATE cache 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE key = ?
            """, (time.time(), key))
            
            conn.commit()
            conn.close()
            
            # 反序列化
            try:
                value = json.loads(value_str)
                logger.debug(f"[L2DiskCache] Hit for key={key}")
                return value
            except json.JSONDecodeError as e:
                logger.error(f"[L2DiskCache] Failed to deserialize: {e}")
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 计算过期时间
            expires_at = None
            if ttl is not None or self.default_ttl > 0:
                ttl = ttl if ttl is not None else self.default_ttl
                expires_at = time.time() + ttl
            
            # 序列化
            value_str = json.dumps(value, ensure_ascii=False)
            
            # 检查磁盘空间
            self._enforce_size_limit(conn)
            
            # 插入或更新
            cursor.execute("""
                INSERT OR REPLACE INTO cache 
                (key, value, created_at, expires_at, last_accessed)
                VALUES (?, ?, ?, ?, ?)
            """, (key, value_str, time.time(), expires_at, time.time()))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"[L2DiskCache] Set key={key}, ttl={ttl}")
    
    def remove(self, key: str):
        """删除缓存"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
            
            conn.commit()
            conn.close()
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM cache")
            
            conn.commit()
            conn.close()
            
            logger.info("[L2DiskCache] Cache cleared")
    
    def contains(self, key: str) -> bool:
        """检查键是否存在且未过期"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 1 FROM cache 
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
            """, (key, time.time()))
            
            exists = cursor.fetchone() is not None
            
            conn.close()
            return exists
    
    def size(self) -> int:
        """获取缓存条目数"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM cache")
            count = cursor.fetchone()[0]
            
            conn.close()
            return count
    
    def size_mb(self) -> float:
        """获取缓存大小 (MB)"""
        with self.lock:
            if self.db_path.exists():
                return self.db_path.stat().st_size / (1024 * 1024)
            return 0.0
    
    def _enforce_size_limit(self, conn: sqlite3.Connection):
        """执行大小限制 (需要持有锁)"""
        current_size_mb = self.size_mb()
        
        if current_size_mb > self.max_size_mb:
            # 删除过期的
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM cache 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """, (time.time(),))
            
            # 如果还是太大，删除最久未使用的
            while self.size_mb() > self.max_size_mb * 0.9:  # 保留 10% 缓冲
                cursor.execute("""
                    DELETE FROM cache 
                    WHERE rowid IN (
                        SELECT rowid FROM cache 
                        ORDER BY last_accessed ASC 
                        LIMIT 10
                    )
                """)
                
                logger.info(f"[L2DiskCache] Evicted old entries, current size={self.size_mb():.2f}MB")
    
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 总条目数
            cursor.execute("SELECT COUNT(*) FROM cache")
            total_count = cursor.fetchone()[0]
            
            # 过期条目数
            cursor.execute("""
                SELECT COUNT(*) FROM cache 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """, (time.time(),))
            expired_count = cursor.fetchone()[0]
            
            # 总访问次数
            cursor.execute("SELECT SUM(access_count) FROM cache")
            total_accesses = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "size_mb": f"{self.size_mb():.2f}",
                "max_size_mb": self.max_size_mb,
                "entries": total_count,
                "expired": expired_count,
                "total_accesses": total_accesses,
                "default_ttl": self.default_ttl
            }
    
    def cleanup_expired(self) -> int:
        """清理过期条目"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM cache 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """, (time.time(),))
            
            deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted > 0:
                logger.info(f"[L2DiskCache] Cleaned up {deleted} expired entries")
            
            return deleted


class MultiLevelCache:
    """多级缓存系统 - L1(内存) + L2(磁盘)"""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        l1_max_size: int = 1000,
        l2_max_size_mb: int = 100,
        l1_ttl: int = 3600,
        l2_ttl: int = 86400
    ):
        """
        Args:
            cache_dir: L2 缓存目录 (默认：~/.pyutagent/cache)
            l1_max_size: L1 最大条目数
            l2_max_size_mb: L2 最大大小 (MB)
            l1_ttl: L1 默认 TTL(秒)
            l2_ttl: L2 默认 TTL(秒)
        """
        # 默认缓存目录
        if cache_dir is None:
            cache_dir = str(Path.home() / ".pyutagent" / "cache")
        
        # 初始化两级缓存
        self.l1_cache = L1MemoryCache(max_size=l1_max_size, default_ttl=l1_ttl)
        self.l2_cache = L2DiskCache(cache_dir=cache_dir, max_size_mb=l2_max_size_mb, default_ttl=l2_ttl)
        
        self.cache_dir = cache_dir
        logger.info(f"[MultiLevelCache] Initialized with L1(max={l1_max_size}) + L2(dir={cache_dir})")
    
    def get(self, key: str, use_l2: bool = True) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            use_l2: 是否查找 L2 缓存
            
        Returns:
            缓存值，不存在则返回 None
        """
        # 先查 L1
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # 再查 L2
        if use_l2:
            value = self.l2_cache.get(key)
            
            # 如果 L2 命中，回填 L1
            if value is not None:
                self.l1_cache.set(key, value)
                logger.debug(f"[MultiLevelCache] L2 hit, backfilled L1 for key={key}")
            
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl_l1: Optional[int] = None, ttl_l2: Optional[int] = None):
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl_l1: L1 TTL(秒)
            ttl_l2: L2 TTL(秒)
        """
        # 同时写入 L1 和 L2
        self.l1_cache.set(key, value, ttl=ttl_l1)
        self.l2_cache.set(key, value, ttl=ttl_l2)
        
        logger.debug(f"[MultiLevelCache] Set key={key}")
    
    def remove(self, key: str):
        """删除缓存"""
        self.l1_cache.remove(key)
        self.l2_cache.remove(key)
    
    def clear(self):
        """清空所有缓存"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        logger.info("[MultiLevelCache] All caches cleared")
    
    def contains(self, key: str, check_l2: bool = True) -> bool:
        """检查键是否存在"""
        if self.l1_cache.contains(key):
            return True
        
        if check_l2:
            return self.l2_cache.contains(key)
        
        return False
    
    def get_or_compute(self, key: str, compute_fn, ttl_l1: Optional[int] = None, ttl_l2: Optional[int] = None) -> Any:
        """
        获取或计算缓存值
        
        Args:
            key: 缓存键
            compute_fn: 计算函数 (无参数)
            ttl_l1: L1 TTL(秒)
            ttl_l2: L2 TTL(秒)
            
        Returns:
            缓存值或计算结果
        """
        # 尝试获取缓存
        value = self.get(key)
        if value is not None:
            return value
        
        # 计算并缓存
        value = compute_fn()
        self.set(key, value, ttl_l1=ttl_l1, ttl_l2=ttl_l2)
        
        return value
    
    def generate_key(self, *args, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            MD5 哈希键
        """
        # 序列化参数
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        
        # 生成 MD5 哈希
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "l1_cache": self.l1_cache.stats(),
            "l2_cache": self.l2_cache.stats(),
            "cache_dir": self.cache_dir
        }
    
    def cleanup(self, expired_only: bool = True) -> Dict[str, int]:
        """
        清理缓存
        
        Args:
            expired_only: 只清理过期条目
            
        Returns:
            清理统计
        """
        stats = {}
        
        # 清理 L2 过期条目
        if expired_only:
            deleted = self.l2_cache.cleanup_expired()
            stats["l2_expired_deleted"] = deleted
        
        logger.info(f"[MultiLevelCache] Cleanup completed: {stats}")
        return stats


# 全局缓存实例
_global_cache: Optional[MultiLevelCache] = None


def get_global_cache() -> MultiLevelCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = MultiLevelCache()
    return _global_cache


def init_global_cache(cache_dir: Optional[str] = None, **kwargs) -> MultiLevelCache:
    """初始化全局缓存"""
    global _global_cache
    _global_cache = MultiLevelCache(cache_dir=cache_dir, **kwargs)
    return _global_cache


def get_file_cache() -> MultiLevelCache:
    """获取文件缓存实例 (兼容旧代码)"""
    return get_global_cache()


class ToolResultCache(MultiLevelCache):
    """工具结果缓存 - 专用于工具执行结果
    
    Features:
    - Content-based cache keys
    - Dependency-aware invalidation
    - File modification tracking
    """
    
    def __init__(
        self,
        max_memory_entries: int = 500,
        cache_dir: Optional[str] = None,
        default_ttl: int = 3600
    ):
        """
        初始化工具结果缓存
        
        Args:
            max_memory_entries: 内存缓存最大条目数
            cache_dir: 磁盘缓存目录（可选）
            default_ttl: 默认过期时间（秒）
        """
        super().__init__(
            cache_dir=cache_dir,
            l1_max_size=max_memory_entries,
            l1_ttl=default_ttl,
            l2_ttl=default_ttl * 24
        )
        
        self._file_hashes: Dict[str, str] = {}
        self._tool_dependencies: Dict[str, List[str]] = {}
        
        logger.info(f"[ToolResultCache] Initialized (max={max_memory_entries}, ttl={default_ttl}s)")
    
    @staticmethod
    def make_key(tool_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键
        
        Args:
            tool_name: 工具名称
            args: 位置参数
            kwargs: 关键字参数
            
        Returns:
            缓存键
        """
        key_data = {
            "tool": tool_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def compute_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def compute_file_hash(self, file_path: str) -> str:
        """计算文件哈希"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return ""
    
    def register_file_dependency(self, tool_name: str, file_path: str):
        """注册文件依赖"""
        if tool_name not in self._tool_dependencies:
            self._tool_dependencies[tool_name] = []
        if file_path not in self._tool_dependencies[tool_name]:
            self._tool_dependencies[tool_name].append(file_path)
        
        self._file_hashes[file_path] = self.compute_file_hash(file_path)
    
    def check_file_changed(self, file_path: str) -> bool:
        """检查文件是否已更改"""
        current_hash = self.compute_file_hash(file_path)
        stored_hash = self._file_hashes.get(file_path, "")
        return current_hash != stored_hash
    
    def get_cached(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """获取缓存结果（兼容旧接口）"""
        key = self.make_key(tool_name, (), params)
        return self.get(key)
    
    def set_cached(
        self,
        tool_name: str,
        params: Dict[str, Any],
        value: Any,
        ttl_seconds: Optional[int] = None
    ):
        """设置缓存结果（兼容旧接口）"""
        key = self.make_key(tool_name, (), params)
        self.set(key, value, ttl_l1=ttl_seconds)
    
    def invalidate(self, tool_name: Optional[str] = None, pattern: Optional[str] = None):
        """使缓存失效"""
        if tool_name is None and pattern is None:
            self.clear()
            return
        
        if tool_name:
            self.l1_cache.clear()
            if tool_name in self._tool_dependencies:
                del self._tool_dependencies[tool_name]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        l1_stats = self.l1_cache.stats()
        return {
            "size": l1_stats.get("size", 0),
            "max_size": l1_stats.get("max_size", 0),
            "hits": l1_stats.get("hits", 0),
            "misses": l1_stats.get("misses", 0),
            "hit_rate": l1_stats.get("hit_rate", "0.00%"),
            "file_dependencies": len(self._file_hashes)
        }


class PromptCache:
    """Prompt 缓存 - 专用于 LLM Prompt 结果
    
    Features:
    - LRU 缓存策略
    - 基于 prompt + model 的键生成
    - 命中率统计
    """
    
    def __init__(
        self,
        max_memory_entries: int = 1000,
        default_ttl: int = 7200
    ):
        """
        初始化 Prompt 缓存
        
        Args:
            max_memory_entries: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
        """
        self._cache = L1MemoryCache(max_size=max_memory_entries, default_ttl=default_ttl)
        self._hits = 0
        self._misses = 0
        
        logger.info(f"[PromptCache] Initialized (max={max_memory_entries}, ttl={default_ttl}s)")
    
    @staticmethod
    def make_key(prompt: str, model: str, **kwargs) -> str:
        """生成缓存键
        
        Args:
            prompt: Prompt 文本
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            缓存键
        """
        key_data = {
            "prompt": prompt,
            "model": model,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """获取缓存结果
        
        Args:
            prompt: Prompt 文本
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            缓存的响应或 None
        """
        key = self.make_key(prompt, model, **kwargs)
        result = self._cache.get(key)
        
        if result is not None:
            self._hits += 1
            logger.debug(f"[PromptCache] Hit for model={model}")
        else:
            self._misses += 1
        
        return result
    
    def set(
        self,
        prompt: str,
        model: str,
        response: str,
        ttl: Optional[int] = None,
        **kwargs
    ):
        """设置缓存结果
        
        Args:
            prompt: Prompt 文本
            model: 模型名称
            response: 响应文本
            ttl: 过期时间（秒）
            **kwargs: 额外参数
        """
        key = self.make_key(prompt, model, **kwargs)
        self._cache.set(key, response, ttl=ttl)
        logger.debug(f"[PromptCache] Set for model={model}")
    
    async def get_or_generate(
        self,
        prompt: str,
        model: str,
        llm_client,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """获取或生成响应
        
        Args:
            prompt: Prompt 文本
            model: 模型名称
            llm_client: LLM 客户端
            system_prompt: 系统提示词
            **kwargs: 额外参数
            
        Returns:
            响应文本
        """
        cached = self.get(prompt, model, **kwargs)
        if cached is not None:
            return cached
        
        if system_prompt:
            response = await llm_client.agenerate(prompt, system_prompt)
        else:
            response = await llm_client.agenerate(prompt)
        
        self.set(prompt, model, response, **kwargs)
        return response
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("[PromptCache] Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "capacity": self._cache.max_size,
            "current_size": self._cache.size(),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2%}"
        }


class CachedToolExecutor:
    """带缓存的工具执行器
    
    Wraps tool execution with cache support.
    """
    
    def __init__(self, tool_service, cache: Optional[ToolResultCache] = None):
        """初始化缓存执行器
        
        Args:
            tool_service: 工具服务
            cache: 可选的缓存实例
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
        """执行工具（带缓存）
        
        Args:
            tool_name: 工具名称
            params: 工具参数
            use_cache: 是否使用缓存
            cache_ttl: 可选的缓存 TTL 覆盖
            
        Returns:
            工具结果
        """
        if use_cache:
            cached = self.cache.get_cached(tool_name, params)
            if cached is not None:
                logger.debug(f"[CachedToolExecutor] Cache hit: {tool_name}")
                return cached
        
        result = await self.tool_service.execute_tool(tool_name, params)
        
        if use_cache and result:
            self.cache.set_cached(
                tool_name, params, result,
                ttl_seconds=cache_ttl
            )
        
        return result
    
    def invalidate(self, tool_name: Optional[str] = None):
        """使缓存失效
        
        Args:
            tool_name: 可选的工具名称
        """
        self.cache.invalidate(tool_name=tool_name)


def create_tool_cache(
    max_size: int = 100,
    ttl: int = 3600
) -> ToolResultCache:
    """创建工具结果缓存
    
    Args:
        max_size: 最大缓存条目数
        ttl: 默认 TTL（秒）
        
    Returns:
        配置好的 ToolResultCache
    """
    return ToolResultCache(max_memory_entries=max_size, default_ttl=ttl)


def create_result_cache(
    max_size: int = 100,
    ttl_seconds: int = 3600,
    persistent_path: Optional[str] = None
) -> ToolResultCache:
    """创建工具结果缓存（兼容旧接口）
    
    Args:
        max_size: 最大缓存大小
        ttl_seconds: 默认 TTL
        persistent_path: 可选的持久化路径
        
    Returns:
        ToolResultCache 实例
    """
    return ToolResultCache(
        max_memory_entries=max_size,
        cache_dir=persistent_path,
        default_ttl=ttl_seconds
    )
