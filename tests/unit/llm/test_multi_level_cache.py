"""多级缓存测试"""
import pytest
import asyncio
import tempfile
import os
import time
from pathlib import Path
from pyutagent.llm.multi_level_cache import (
    MultiLevelCache,
    CacheLevel,
    L1Cache,
    L2Cache,
    CacheEntry,
    CacheConfig
)


class TestCacheEntry:
    """缓存条目测试"""
    
    def test_create_cache_entry(self):
        """测试创建缓存条目"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=3600
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 3600
        assert entry.created_at is not None
        assert entry.is_expired() is False
    
    def test_cache_entry_expiration(self):
        """测试缓存条目过期"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=0  # 立即过期
        )
        
        assert entry.is_expired() is True
    
    def test_cache_entry_no_expiration(self):
        """测试缓存条目永不过期"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=-1  # 永不过期
        )
        
        assert entry.is_expired() is False


class TestL1Cache:
    """L1 缓存（内存）测试"""
    
    def test_create_l1_cache(self):
        """测试创建 L1 缓存"""
        cache = L1Cache(capacity=100)
        assert cache is not None
        assert cache.capacity == 100
    
    @pytest.mark.asyncio
    async def test_l1_cache_put_get(self):
        """测试 L1 缓存的存取"""
        cache = L1Cache(capacity=100)
        
        await cache.put("key1", "value1")
        value = await cache.get("key1")
        
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_l1_cache_miss(self):
        """测试 L1 缓存未命中"""
        cache = L1Cache(capacity=100)
        
        value = await cache.get("nonexistent")
        
        assert value is None
    
    @pytest.mark.asyncio
    async def test_l1_cache_lru_eviction(self):
        """测试 L1 缓存 LRU 淘汰"""
        cache = L1Cache(capacity=3)
        
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.put("key3", "value3")
        await cache.put("key4", "value4")  # 应该淘汰 key1
        
        value1 = await cache.get("key1")
        value4 = await cache.get("key4")
        
        assert value1 is None  # 已被淘汰
        assert value4 == "value4"
    
    @pytest.mark.asyncio
    async def test_l1_cache_clear(self):
        """测试 L1 缓存清空"""
        cache = L1Cache(capacity=100)
        
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        await cache.clear()
        
        size = await cache.size()
        assert size == 0
    
    @pytest.mark.asyncio
    async def test_l1_cache_size(self):
        """测试 L1 缓存大小"""
        cache = L1Cache(capacity=100)
        
        await cache.put("key1", "value1")
        await cache.put("key2", "value2")
        
        size = await cache.size()
        assert size == 2
    
    @pytest.mark.asyncio
    async def test_l1_cache_contains(self):
        """测试 L1 缓存包含检查"""
        cache = L1Cache(capacity=100)
        
        await cache.put("key1", "value1")
        
        assert await cache.contains("key1") is True
        assert await cache.contains("key2") is False


class TestL2Cache:
    """L2 缓存（磁盘）测试"""
    
    def test_create_l2_cache(self):
        """测试创建 L2 缓存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = L2Cache(storage_path=temp_dir)
            assert cache is not None
    
    @pytest.mark.asyncio
    async def test_l2_cache_put_get(self):
        """测试 L2 缓存的存取"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = L2Cache(storage_path=temp_dir)
            
            await cache.put("key1", "value1")
            value = await cache.get("key1")
            
            assert value == "value1"
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_l2_cache_persistence(self):
        """测试 L2 缓存持久化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache1 = L2Cache(storage_path=temp_dir)
            await cache1.put("persistent_key", "persistent_value")
            await cache1.cleanup()
            
            cache2 = L2Cache(storage_path=temp_dir)
            value = await cache2.get("persistent_key")
            
            assert value == "persistent_value"
            
            await cache2.cleanup()
    
    @pytest.mark.asyncio
    async def test_l2_cache_delete(self):
        """测试 L2 缓存删除"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = L2Cache(storage_path=temp_dir)
            
            await cache.put("key1", "value1")
            await cache.delete("key1")
            
            value = await cache.get("key1")
            assert value is None
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_l2_cache_clear(self):
        """测试 L2 缓存清空"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = L2Cache(storage_path=temp_dir)
            
            await cache.put("key1", "value1")
            await cache.put("key2", "value2")
            await cache.clear()
            
            size = await cache.size()
            assert size == 0
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_l2_cache_size(self):
        """测试 L2 缓存大小"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = L2Cache(storage_path=temp_dir)
            
            await cache.put("key1", "value1")
            await cache.put("key2", "value2")
            await cache.put("key3", "value3")
            
            size = await cache.size()
            assert size == 3
            
            await cache.cleanup()


class TestMultiLevelCache:
    """多级缓存测试"""
    
    @pytest.mark.asyncio
    async def test_create_multi_level_cache(self):
        """测试创建多级缓存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                l1_capacity=100,
                l2_storage_path=temp_dir
            )
            cache = MultiLevelCache(config)
            assert cache is not None
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_get_hit_l1(self):
        """测试多级缓存 L1 命中"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            await cache.put("key1", "value1")
            value = await cache.get("key1")
            
            assert value == "value1"
            assert cache.stats.l1_hits == 1
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_get_hit_l2(self):
        """测试多级缓存 L2 命中"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            # 直接写入 L2
            await cache._l2_cache.put("key1", "value1")
            
            # 从多级缓存读取（应该从 L2 加载到 L1）
            value = await cache.get("key1")
            
            assert value == "value1"
            assert cache.stats.l2_hits == 1
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_miss(self):
        """测试多级缓存未命中"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            value = await cache.get("nonexistent")
            
            assert value is None
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_put(self):
        """测试多级缓存写入"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            await cache.put("key1", "value1")
            
            l1_value = await cache._l1_cache.get("key1")
            l2_value = await cache._l2_cache.get("key1")
            
            assert l1_value == "value1"
            assert l2_value == "value1"
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_delete(self):
        """测试多级缓存删除"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            await cache.put("key1", "value1")
            await cache.delete("key1")
            
            value = await cache.get("key1")
            
            assert value is None
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_clear(self):
        """测试多级缓存清空"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            await cache.put("key1", "value1")
            await cache.put("key2", "value2")
            await cache.clear()
            
            size = await cache.size()
            assert size == 0
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_stats(self):
        """测试多级缓存统计"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            await cache.put("key1", "value1")
            await cache.get("key1")  # L1 hit
            await cache.get("key1")  # L1 hit
            
            stats = cache.get_stats()
            
            assert stats["l1_hits"] == 2
            assert stats["total_hits"] == 2
            assert "hit_rate" in stats
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_warmup(self):
        """测试多级缓存预热"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            warmup_data = {
                "key1": "value1",
                "key2": "value2",
                "key3": "value3"
            }
            
            await cache.warmup(warmup_data)
            
            for key, value in warmup_data.items():
                cached_value = await cache.get(key)
                assert cached_value == value
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_concurrent_access(self):
        """测试多级缓存并发访问"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            async def writer(key: str, value: str):
                await cache.put(key, value)
            
            async def reader(key: str):
                return await cache.get(key)
            
            # 并发写入
            await asyncio.gather(
                writer("key1", "value1"),
                writer("key2", "value2"),
                writer("key3", "value3")
            )
            
            # 并发读取
            values = await asyncio.gather(
                reader("key1"),
                reader("key2"),
                reader("key3")
            )
            
            assert values == ["value1", "value2", "value3"]
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_ttl(self):
        """测试多级缓存 TTL"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(l1_capacity=100, l2_storage_path=temp_dir)
            cache = MultiLevelCache(config)
            
            await cache.put("key1", "value1", ttl=0)  # TTL 为 0，立即过期
            
            value_immediate = await cache.get("key1")
            assert value_immediate is None  # TTL 为 0 应该立即过期
            
            await cache.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_compression(self):
        """测试多级缓存压缩"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CacheConfig(
                l1_capacity=100,
                l2_storage_path=temp_dir,
                enable_compression=True
            )
            cache = MultiLevelCache(config)
            
            large_value = "x" * 10000  # 10KB 数据
            await cache.put("large_key", large_value)
            
            value = await cache.get("large_key")
            assert value == large_value
            
            await cache.cleanup()


class TestCacheConfig:
    """缓存配置测试"""
    
    def test_create_default_config(self):
        """测试创建默认配置"""
        config = CacheConfig()
        
        assert config.l1_capacity == 1000
        assert config.l2_storage_path is None
        assert config.enable_compression is False
        assert config.default_ttl == 3600
    
    def test_create_custom_config(self):
        """测试创建自定义配置"""
        config = CacheConfig(
            l1_capacity=500,
            l2_storage_path="/tmp/cache",
            enable_compression=True,
            default_ttl=7200
        )
        
        assert config.l1_capacity == 500
        assert config.l2_storage_path == "/tmp/cache"
        assert config.enable_compression is True
        assert config.default_ttl == 7200
