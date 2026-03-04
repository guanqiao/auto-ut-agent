"""缓存系统单元测试

测试多级缓存系统的功能，包括:
- L1MemoryCache (内存缓存)
- L2DiskCache (磁盘缓存)
- MultiLevelCache (多级缓存)
"""

import json
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import pytest

from pyutagent.core.cache import (
    CacheEntry,
    L1MemoryCache,
    L2DiskCache,
    MultiLevelCache,
    get_global_cache,
    init_global_cache,
)


class TestCacheEntry:
    """测试 CacheEntry 类"""

    def test_create_entry(self):
        """测试创建缓存条目"""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test_value"},
            created_at=time.time(),
            expires_at=time.time() + 3600,
        )

        assert entry.key == "test_key"
        assert entry.value == {"data": "test_value"}
        assert entry.access_count == 0
        assert not entry.is_expired()

    def test_entry_expiration(self):
        """测试缓存条目过期"""
        # 创建已过期的条目
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time() - 100,
            expires_at=time.time() - 50,  # 已过期
        )

        assert entry.is_expired()

    def test_entry_no_expiration(self):
        """测试缓存条目永不过期"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            expires_at=None,  # 永不过期
        )

        assert not entry.is_expired()

    def test_entry_touch(self):
        """测试更新访问时间"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            expires_at=time.time() + 3600,
        )

        initial_access = entry.access_count
        initial_last_accessed = entry.last_accessed

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == initial_access + 1
        assert entry.last_accessed > initial_last_accessed

    def test_entry_to_dict(self):
        """测试转换为字典"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=1234567890.0,
            expires_at=1234567890.0 + 3600,
            access_count=5,
            last_accessed=1234567895.0,
        )

        entry_dict = entry.to_dict()

        assert entry_dict["key"] == "test_key"
        assert entry_dict["value"] == "test_value"
        assert entry_dict["access_count"] == 5

    def test_entry_from_dict(self):
        """测试从字典创建"""
        data = {
            "key": "test_key",
            "value": "test_value",
            "created_at": 1234567890.0,
            "expires_at": 1234567890.0 + 3600,
            "access_count": 5,
            "last_accessed": 1234567895.0,
        }

        entry = CacheEntry.from_dict(data)

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.access_count == 5


class TestL1MemoryCache:
    """测试 L1 内存缓存"""

    def setup_method(self):
        """每个测试前的设置"""
        self.cache = L1MemoryCache(max_size=100, default_ttl=3600)

    def teardown_method(self):
        """每个测试后的清理"""
        self.cache.clear()

    def test_basic_set_get(self):
        """测试基本的设置和获取"""
        self.cache.set("key1", "value1")
        result = self.cache.get("key1")

        assert result == "value1"

    def test_set_complex_value(self):
        """测试设置复杂值"""
        value = {
            "name": "test",
            "data": [1, 2, 3],
            "nested": {"key": "value"},
        }
        self.cache.set("complex_key", value)
        result = self.cache.get("complex_key")

        assert result == value

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        result = self.cache.get("nonexistent")
        assert result is None

    def test_contains(self):
        """测试检查键是否存在"""
        self.cache.set("key1", "value1")

        assert self.cache.contains("key1") is True
        assert self.cache.contains("nonexistent") is False

    def test_remove(self):
        """测试删除缓存"""
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

        self.cache.remove("key1")
        assert self.cache.get("key1") is None

    def test_clear(self):
        """测试清空缓存"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        self.cache.clear()

        assert self.cache.size() == 0
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None

    def test_ttl_expiration(self):
        """测试 TTL 过期"""
        # 设置一个很短的 TTL
        self.cache.set("key1", "value1", ttl=1)

        # 立即获取应该成功
        assert self.cache.get("key1") == "value1"

        # 等待过期
        time.sleep(1.1)

        # 获取应该返回 None
        assert self.cache.get("key1") is None

    def test_no_expiration_without_ttl(self):
        """测试没有 TTL 时不过期"""
        # 设置 default_ttl=0 的缓存
        cache = L1MemoryCache(max_size=100, default_ttl=0)
        cache.set("key1", "value1")

        # 等待一小段时间
        time.sleep(0.1)

        # 仍然应该能获取
        assert cache.get("key1") == "value1"

    def test_lru_eviction(self):
        """测试 LRU 淘汰机制"""
        cache = L1MemoryCache(max_size=3, default_ttl=0)

        # 添加 3 个条目
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # 访问 key1，使其成为最近使用的
        cache.get("key1")

        # 添加第 4 个条目，应该淘汰 key2 (最久未使用)
        cache.set("key4", "value4")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None  # 被淘汰
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_update_existing_key(self):
        """测试更新已存在的键"""
        self.cache.set("key1", "value1")
        self.cache.set("key1", "updated_value")

        result = self.cache.get("key1")
        assert result == "updated_value"

    def test_stats(self):
        """测试统计信息"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        # 访问 key1 两次
        self.cache.get("key1")
        self.cache.get("key1")

        # 访问 key2 一次
        self.cache.get("key2")

        # 访问不存在的键
        self.cache.get("nonexistent")

        stats = self.cache.stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert "hit_rate" in stats

    def test_size_limit(self):
        """测试大小限制"""
        cache = L1MemoryCache(max_size=5, default_ttl=0)

        # 添加超过限制的条目
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # 缓存大小不应该超过 max_size
        assert cache.size() <= 5

    def test_thread_safety(self):
        """测试线程安全性"""
        cache = L1MemoryCache(max_size=1000, default_ttl=3600)
        results: List[bool] = []

        def worker(worker_id: int):
            try:
                # 写入
                cache.set(f"key_{worker_id}", f"value_{worker_id}")

                # 读取
                value = cache.get(f"key_{worker_id}")

                # 验证
                results.append(value == f"value_{worker_id}")
            except Exception as e:
                results.append(False)

        # 创建多个线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 所有操作都应该成功
        assert all(results)
        assert len(results) == 10

    def test_concurrent_access(self):
        """测试并发访问"""
        cache = L1MemoryCache(max_size=100, default_ttl=3600)
        error_count = 0

        def access_cache():
            nonlocal error_count
            try:
                for i in range(100):
                    cache.set(f"key_{i}", f"value_{i}")
                    cache.get(f"key_{i}")
            except Exception:
                error_count += 1

        # 使用线程池并发访问
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_cache) for _ in range(10)]
            for future in futures:
                future.result()

        # 不应该有错误
        assert error_count == 0


class TestL2DiskCache:
    """测试 L2 磁盘缓存"""

    def setup_method(self):
        """每个测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = L2DiskCache(
            cache_dir=self.temp_dir, max_size_mb=100, default_ttl=3600
        )

    def teardown_method(self):
        """每个测试后的清理"""
        self.cache.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_set_get(self):
        """测试基本的设置和获取"""
        self.cache.set("key1", "value1")
        result = self.cache.get("key1")

        assert result == "value1"

    def test_persistence(self):
        """测试持久化"""
        self.cache.set("key1", "persistent_value")

        # 创建新的缓存实例，使用相同的目录
        new_cache = L2DiskCache(cache_dir=self.temp_dir, max_size_mb=100)
        result = new_cache.get("key1")

        assert result == "persistent_value"

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        result = self.cache.get("nonexistent")
        assert result is None

    def test_contains(self):
        """测试检查键是否存在"""
        self.cache.set("key1", "value1")

        assert self.cache.contains("key1") is True
        assert self.cache.contains("nonexistent") is False

    def test_remove(self):
        """测试删除缓存"""
        self.cache.set("key1", "value1")
        assert self.cache.get("key1") == "value1"

        self.cache.remove("key1")
        assert self.cache.get("key1") is None

    def test_clear(self):
        """测试清空缓存"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        self.cache.clear()

        assert self.cache.size() == 0
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None

    def test_ttl_expiration(self):
        """测试 TTL 过期"""
        # 设置一个很短的 TTL
        self.cache.set("key1", "value1", ttl=1)

        # 立即获取应该成功
        assert self.cache.get("key1") == "value1"

        # 等待过期
        time.sleep(1.1)

        # 获取应该返回 None
        assert self.cache.get("key1") is None

    def test_complex_value_serialization(self):
        """测试复杂值序列化"""
        value = {
            "name": "test",
            "data": [1, 2, 3, {"nested": "value"}],
            "nested": {"key": "value", "number": 42},
        }

        self.cache.set("complex_key", value)
        result = self.cache.get("complex_key")

        assert result == value

    def test_unicode_support(self):
        """测试 Unicode 支持"""
        value = {"message": "你好，世界！🌍", "emoji": "测试"}

        self.cache.set("unicode_key", value)
        result = self.cache.get("unicode_key")

        assert result == value

    def test_stats(self):
        """测试统计信息"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")

        stats = self.cache.stats()

        assert stats["entries"] == 3
        assert "size_mb" in stats
        assert stats["max_size_mb"] == 100
        assert "default_ttl" in stats

    def test_size_mb(self):
        """测试获取缓存大小"""
        # 添加一些数据
        large_value = {"data": "x" * 10000}
        self.cache.set("large_key", large_value)

        size_mb = self.cache.size_mb()
        assert size_mb > 0
        assert size_mb < 1  # 应该小于 1MB

    def test_cleanup_expired(self):
        """测试清理过期条目"""
        # 设置过期条目
        self.cache.set("expired1", "value1", ttl=1)
        self.cache.set("expired2", "value2", ttl=1)
        self.cache.set("valid", "value3", ttl=3600)

        # 等待过期
        time.sleep(1.1)

        # 清理
        deleted = self.cache.cleanup_expired()

        assert deleted == 2
        assert self.cache.contains("expired1") is False
        assert self.cache.contains("expired2") is False
        assert self.cache.contains("valid") is True

    def test_update_existing_key(self):
        """测试更新已存在的键"""
        self.cache.set("key1", "value1")
        self.cache.set("key1", "updated_value")

        result = self.cache.get("key1")
        assert result == "updated_value"

    def test_size_limit_enforcement(self):
        """测试大小限制执行"""
        # 创建很小的缓存用于测试
        cache = L2DiskCache(cache_dir=self.temp_dir, max_size_mb=1)

        # 添加大量数据
        large_value = "x" * (100 * 1024)  # 100KB
        for i in range(20):  # 总共 2MB
            cache.set(f"key_{i}", large_value)

        # 缓存大小应该受到限制
        assert cache.size_mb() <= 1.0

    def test_concurrent_access(self):
        """测试并发访问"""
        error_count = 0

        def access_cache():
            nonlocal error_count
            try:
                for i in range(50):
                    self.cache.set(f"key_{i}", f"value_{i}")
                    self.cache.get(f"key_{i}")
            except Exception:
                error_count += 1

        # 使用线程池并发访问
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(access_cache) for _ in range(5)]
            for future in futures:
                future.result()

        # 不应该有错误
        assert error_count == 0


class TestMultiLevelCache:
    """测试多级缓存"""

    def setup_method(self):
        """每个测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = MultiLevelCache(
            cache_dir=self.temp_dir,
            l1_max_size=100,
            l2_max_size_mb=100,
            l1_ttl=3600,
            l2_ttl=3600,
        )

    def teardown_method(self):
        """每个测试后的清理"""
        self.cache.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_set_get(self):
        """测试基本的设置和获取"""
        self.cache.set("key1", "value1")
        result = self.cache.get("key1")

        assert result == "value1"

    def test_two_level_lookup(self):
        """测试两级查找"""
        # 直接设置到 L2
        self.cache.l2_cache.set("l2_only_key", "l2_value")

        # 从多级缓存获取，应该能找到
        result = self.cache.get("l2_only_key")
        assert result == "l2_value"

    def test_l1_backfill(self):
        """测试 L1 回填"""
        # 直接设置到 L2
        self.cache.l2_cache.set("backfill_key", "backfill_value")

        # 获取，应该触发 L1 回填
        result = self.cache.get("backfill_key")
        assert result == "backfill_value"

        # 现在 L1 中应该有这个键
        assert self.cache.l1_cache.contains("backfill_key")

    def test_get_or_compute_cache_hit(self):
        """测试 get_or_compute - 缓存命中"""
        compute_count = 0

        def compute_fn():
            nonlocal compute_count
            compute_count += 1
            return "computed_value"

        # 第一次调用，应该计算
        result1 = self.cache.get_or_compute("compute_key", compute_fn)
        assert result1 == "computed_value"
        assert compute_count == 1

        # 第二次调用，应该从缓存获取
        result2 = self.cache.get_or_compute("compute_key", compute_fn)
        assert result2 == "computed_value"
        assert compute_count == 1  # 不应该再次计算

    def test_get_or_compute_cache_miss(self):
        """测试 get_or_compute - 缓存未命中"""
        compute_count = 0

        def compute_fn():
            nonlocal compute_count
            compute_count += 1
            return "new_value"

        result = self.cache.get_or_compute("new_key", compute_fn)
        assert result == "new_value"
        assert compute_count == 1

    def test_key_generation(self):
        """测试键生成"""
        # 相同的参数应该生成相同的键
        key1 = self.cache.generate_key("arg1", "arg2", kwarg1="value1")
        key2 = self.cache.generate_key("arg1", "arg2", kwarg1="value1")
        assert key1 == key2

        # 不同的参数应该生成不同的键
        key3 = self.cache.generate_key("arg1", "arg2", kwarg1="value2")
        assert key1 != key3

    def test_key_generation_with_complex_objects(self):
        """测试复杂对象的键生成"""
        obj1 = {"name": "test", "data": [1, 2, 3]}
        obj2 = {"name": "test", "data": [1, 2, 3]}

        # 相同的对象应该生成相同的键
        key1 = self.cache.generate_key(obj1)
        key2 = self.cache.generate_key(obj2)
        assert key1 == key2

    def test_set_with_different_ttls(self):
        """测试设置不同的 TTL"""
        self.cache.set("key1", "value1", ttl_l1=1, ttl_l2=3600)

        # L1 应该很快过期
        time.sleep(1.1)

        # L2 仍然应该有
        result = self.cache.get("key1")
        assert result == "value1"

    def test_remove_from_both_levels(self):
        """测试从两级删除"""
        self.cache.set("key1", "value1")

        # 确认在 L1 中
        assert self.cache.l1_cache.contains("key1")

        # 删除
        self.cache.remove("key1")

        # 两级都不应该有
        assert self.cache.l1_cache.contains("key1") is False
        assert self.cache.l2_cache.contains("key1") is False

    def test_clear_all(self):
        """测试清空所有"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        self.cache.clear()

        assert self.cache.l1_cache.size() == 0
        assert self.cache.l2_cache.size() == 0

    def test_stats(self):
        """测试统计信息"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        # 访问几次
        self.cache.get("key1")
        self.cache.get("key1")
        self.cache.get("key2")

        stats = self.cache.stats()

        assert "l1_cache" in stats
        assert "l2_cache" in stats
        assert "cache_dir" in stats

        assert stats["l1_cache"]["size"] >= 0
        assert "entries" in stats["l2_cache"]

    def test_contains_check(self):
        """测试包含检查"""
        self.cache.set("key1", "value1")

        assert self.cache.contains("key1") is True
        assert self.cache.contains("nonexistent") is False

    def test_performance(self):
        """测试性能"""
        # L1 访问应该很快
        self.cache.set("perf_key", "perf_value")

        start_time = time.time()
        for _ in range(1000):
            self.cache.get("perf_key")
        l1_time = time.time() - start_time

        # 1000 次 L1 访问应该在合理时间内完成
        assert l1_time < 1.0  # 小于 1 秒

    def test_global_cache_instance(self):
        """测试全局缓存实例"""
        # 获取全局实例
        global_cache = get_global_cache()
        assert global_cache is not None

        # 多次获取应该返回同一个实例
        global_cache2 = get_global_cache()
        assert global_cache is global_cache2

    def test_init_global_cache(self):
        """测试初始化全局缓存"""
        # 初始化新的全局缓存
        new_cache = init_global_cache(
            cache_dir=self.temp_dir, l1_max_size=50, l2_max_size_mb=50
        )

        assert new_cache is not None
        assert new_cache.l1_cache.max_size == 50
        assert new_cache.l2_cache.max_size_mb == 50

    def test_cleanup(self):
        """测试清理"""
        # 设置一些过期条目
        self.cache.set("expired", "value", ttl_l1=1, ttl_l2=1)

        time.sleep(1.1)

        # 清理
        stats = self.cache.cleanup(expired_only=True)

        assert "l2_expired_deleted" in stats

    def test_use_l2_flag(self):
        """测试 use_l2 标志"""
        # 只设置到 L2
        self.cache.l2_cache.set("l2_only", "l2_value")

        # 不使用 L2 查找
        result = self.cache.get("l2_only", use_l2=False)
        assert result is None

        # 使用 L2 查找
        result = self.cache.get("l2_only", use_l2=True)
        assert result == "l2_value"

    def test_check_l2_flag(self):
        """测试 check_l2 标志"""
        # 只设置到 L2
        self.cache.l2_cache.set("l2_only", "l2_value")

        # 不检查 L2
        result = self.cache.contains("l2_only", check_l2=False)
        assert result is False

        # 检查 L2
        result = self.cache.contains("l2_only", check_l2=True)
        assert result is True


class TestCacheIntegration:
    """测试缓存集成场景"""

    def setup_method(self):
        """每个测试前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = MultiLevelCache(
            cache_dir=self.temp_dir, l1_max_size=100, l2_max_size_mb=100
        )

    def teardown_method(self):
        """每个测试后的清理"""
        self.cache.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_analysis_result(self):
        """测试缓存分析结果"""
        # 模拟分析结果
        analysis_result = {
            "file": "test.java",
            "methods": ["method1", "method2"],
            "complexity": 5,
            "test_scenarios": [
                {"id": "1", "description": "Test normal case"},
                {"id": "2", "description": "Test edge case"},
            ],
        }

        # 生成缓存键
        cache_key = self.cache.generate_key("analysis", "test.java")

        # 缓存结果
        self.cache.set(cache_key, analysis_result, ttl_l1=1800, ttl_l2=43200)

        # 获取缓存
        cached_result = self.cache.get(cache_key)

        assert cached_result == analysis_result

    def test_cache_miss_triggers_computation(self):
        """测试缓存未命中触发计算"""
        computation_called = False

        def expensive_computation():
            nonlocal computation_called
            computation_called = True
            return {"result": "expensive_result"}

        # 第一次调用
        result = self.cache.get_or_compute(
            "computation_key", expensive_computation, ttl_l1=1800, ttl_l2=43200
        )

        assert result == {"result": "expensive_result"}
        assert computation_called

        # 第二次调用，不应该触发计算
        computation_called = False
        result = self.cache.get_or_compute(
            "computation_key", expensive_computation, ttl_l1=1800, ttl_l2=43200
        )

        assert result == {"result": "expensive_result"}
        assert not computation_called

    def test_cache_key_generation_consistency(self):
        """测试缓存键生成一致性"""
        # 使用不同的参数类型
        key1 = self.cache.generate_key(
            "file.java", {"method": "test", "line": 10}
        )
        key2 = self.cache.generate_key(
            "file.java", {"method": "test", "line": 10}
        )

        assert key1 == key2

    def test_large_value_caching(self):
        """测试大值缓存"""
        # 创建一个大值
        large_value = {
            "data": "x" * 100000,  # 100KB
            "metadata": {"size": "large"},
        }

        self.cache.set("large_key", large_value)
        result = self.cache.get("large_key")

        assert result == large_value
