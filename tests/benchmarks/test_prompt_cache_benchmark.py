"""性能基准测试：Prompt 缓存"""
import pytest
import asyncio
import time
from pyutagent.llm.prompt_cache import PromptCache


class SlowLLMClient:
    """模拟慢速 LLM 客户端"""
    
    def __init__(self, response: str = "Response", latency: float = 0.1):
        self.response = response
        self.latency = latency
        self.call_count = 0
    
    async def agenerate(self, prompt: str, system_prompt: str = None) -> str:
        self.call_count += 1
        await asyncio.sleep(self.latency)
        return self.response


class TestPromptCacheBenchmark:
    """Prompt 缓存性能基准测试"""
    
    @pytest.mark.asyncio
    async def test_cache_hit_vs_miss_performance(self):
        """测试缓存命中与未命中的性能对比"""
        cache = PromptCache()
        llm = SlowLLMClient(latency=0.05)
        
        # 第一次调用（缓存未命中）
        start = time.time()
        await cache.get_or_generate("prompt1", "system", llm)
        miss_time = time.time() - start
        
        # 第二次调用（缓存命中）
        start = time.time()
        await cache.get_or_generate("prompt1", "system", llm)
        hit_time = time.time() - start
        
        # 验证缓存命中显著更快
        assert hit_time < miss_time
        assert miss_time >= 0.05  # 至少包含 LLM 延迟
        assert hit_time < 0.01    # 命中应该非常快
        
        # 性能提升比例
        speedup = miss_time / hit_time if hit_time > 0 else float('inf')
        assert speedup > 5  # 至少 5 倍提升
    
    @pytest.mark.asyncio
    async def test_cache_size_impact_on_performance(self):
        """测试缓存大小对性能的影响"""
        small_cache = PromptCache(capacity=10)
        large_cache = PromptCache(capacity=10000)
        llm = SlowLLMClient(latency=0.01)
        
        # 填充小缓存
        for i in range(10):
            await small_cache.get_or_generate(f"prompt{i}", "system", llm)
        
        # 填充大缓存
        for i in range(10000):
            await large_cache.get_or_generate(f"prompt{i}", "system", llm)
        
        # 测试查找性能
        start = time.time()
        await small_cache.get_or_generate("prompt5", "system", llm)
        small_time = time.time() - start
        
        start = time.time()
        await large_cache.get_or_generate("prompt5000", "system", llm)
        large_time = time.time() - start
        
        # 验证 OrderedDict 的查找性能不受缓存大小影响
        assert abs(small_time - large_time) < 0.01  # 差异应该很小
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self):
        """测试并发访问缓存的性能"""
        cache = PromptCache(capacity=1000)
        llm = SlowLLMClient(latency=0.01)
        
        # 预热缓存
        for i in range(100):
            await cache.get_or_generate(f"prompt{i}", "system", llm)
        
        # 并发访问
        async def access_cache(prompt_id: int):
            return await cache.get_or_generate(
                f"prompt{prompt_id % 100}",
                "system",
                llm
            )
        
        start = time.time()
        tasks = [access_cache(i) for i in range(100)]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start
        
        # 验证并发性能
        assert concurrent_time < 1.0  # 应该很快（都是缓存命中）
        
        # 计算吞吐量
        throughput = 100 / concurrent_time
        assert throughput > 100  # 每秒至少 100 次请求
    
    @pytest.mark.asyncio
    async def test_cache_eviction_performance(self):
        """测试缓存淘汰策略的性能"""
        cache = PromptCache(capacity=100)
        llm = SlowLLMClient(latency=0.001)
        
        # 填充缓存并触发淘汰
        start = time.time()
        for i in range(200):  # 超过容量
            await cache.get_or_generate(f"prompt{i}", "system", llm)
        eviction_time = time.time() - start
        
        # 验证淘汰操作不会显著影响性能
        # 考虑到实际运行情况，允许更长的时间
        assert eviction_time < 5.0  # 应该在 5 秒内完成
        
        # 验证缓存大小保持在限制内
        assert len(cache._cache) <= 100
        
        # 计算平均每次操作的时间
        avg_time = eviction_time / 200
        assert avg_time < 0.025  # 平均每次操作小于 25ms
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_impact(self):
        """测试不同缓存命中率下的性能"""
        cache = PromptCache(capacity=100)
        llm = SlowLLMClient(latency=0.01)
        
        # 场景 1: 0% 命中率（所有都不同）
        start = time.time()
        for i in range(100):
            await cache.get_or_generate(f"unique{i}", "system", llm)
        zero_hit_time = time.time() - start
        
        # 场景 2: 100% 命中率（重复访问）
        start = time.time()
        for i in range(100):
            await cache.get_or_generate("same_prompt", "system", llm)
        hundred_hit_time = time.time() - start
        
        # 验证命中率的影响
        assert hundred_hit_time < zero_hit_time
        
        # 性能提升
        speedup = zero_hit_time / hundred_hit_time if hundred_hit_time > 0 else float('inf')
        assert speedup > 10  # 至少 10 倍提升
    
    @pytest.mark.asyncio
    async def test_cache_warmup_benefit(self):
        """测试缓存预热的好处"""
        cache = PromptCache(capacity=1000)
        llm = SlowLLMClient(latency=0.02)
        
        # 预热缓存
        common_prompts = [
            ("prompt1", "system1"),
            ("prompt2", "system2"),
            ("prompt3", "system3"),
        ]
        
        for prompt, system in common_prompts:
            await cache.get_or_generate(prompt, system, llm)
        
        # 模拟实际工作负载
        start = time.time()
        for _ in range(10):  # 10 轮
            for prompt, system in common_prompts:
                await cache.get_or_generate(prompt, system, llm)
        warmed_time = time.time() - start
        
        # 没有预热的情况
        cache2 = PromptCache(capacity=1000)
        llm2 = SlowLLMClient(latency=0.02)
        
        start = time.time()
        for _ in range(10):
            for prompt, system in common_prompts:
                await cache2.get_or_generate(prompt, system, llm2)
        unwarmed_time = time.time() - start
        
        # 验证预热的好处
        assert warmed_time < unwarmed_time
        speedup = unwarmed_time / warmed_time if warmed_time > 0 else float('inf')
        assert speedup > 2  # 至少 2 倍提升（考虑到实际运行波动）


class TestPromptCacheMetrics:
    """Prompt 缓存指标测试"""
    
    @pytest.mark.asyncio
    async def test_cache_statistics_accuracy(self):
        """测试缓存统计的准确性"""
        cache = PromptCache(capacity=100)
        llm = SlowLLMClient()
        
        # 执行一些操作
        await cache.get_or_generate("prompt1", "system", llm)  # miss
        await cache.get_or_generate("prompt1", "system", llm)  # hit
        await cache.get_or_generate("prompt2", "system", llm)  # miss
        await cache.get_or_generate("prompt1", "system", llm)  # hit
        
        # 获取统计
        stats = cache.get_stats()
        
        # 验证统计准确性
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["current_size"] == 2
    
    @pytest.mark.asyncio
    async def test_cache_monitoring(self):
        """测试缓存监控"""
        cache = PromptCache(capacity=1000)
        llm = SlowLLMClient(latency=0.001)
        
        # 模拟工作负载
        for i in range(100):
            await cache.get_or_generate(f"prompt{i % 10}", "system", llm)
        
        # 监控指标
        stats = cache.get_stats()
        
        # 验证监控数据
        assert stats["capacity"] == 1000
        assert stats["current_size"] == 10  # 只有 10 个不同的 prompt
        assert stats["hits"] == 90  # 90 次命中
        assert stats["misses"] == 10  # 10 次未命中
        assert stats["hit_rate"] == 0.9  # 90% 命中率


class TestPromptCacheStress:
    """Prompt 缓存压力测试"""
    
    @pytest.mark.asyncio
    async def test_high_load_scenario(self):
        """测试高负载场景"""
        cache = PromptCache(capacity=10000)
        llm = SlowLLMClient(latency=0.001)
        
        # 高负载：1000 次请求
        start = time.time()
        for i in range(1000):
            await cache.get_or_generate(f"prompt{i % 100}", "system", llm)
        total_time = time.time() - start
        
        # 验证性能
        assert total_time < 5.0  # 应该在 5 秒内完成
        
        # 计算 QPS
        qps = 1000 / total_time
        assert qps > 200  # 每秒至少 200 次请求
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """测试内存效率"""
        import sys
        
        cache = PromptCache(capacity=1000)
        llm = SlowLLMClient(response="x" * 1000)  # 1KB 响应
        
        # 填充缓存
        for i in range(1000):
            await cache.get_or_generate(f"prompt{i}", "system", llm)
        
        # 估算内存使用
        estimated_size = sys.getsizeof(cache._cache)
        for key, value in cache._cache.items():
            estimated_size += sys.getsizeof(key) + sys.getsizeof(value)
        
        # 验证内存使用合理（不超过 10MB）
        assert estimated_size < 10 * 1024 * 1024
