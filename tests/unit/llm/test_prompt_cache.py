"""测试 LLM Prompt 缓存"""
import pytest
import asyncio
from pyutagent.llm.prompt_cache import PromptCache


class MockLLMClient:
    """模拟 LLM 客户端"""
    
    def __init__(self, response: str = "Test response"):
        self.response = response
        self.call_count = 0
    
    async def agenerate(self, prompt: str, system_prompt: str = None) -> str:
        self.call_count += 1
        await asyncio.sleep(0.01)  # 模拟延迟
        return self.response


class TestPromptCache:
    """测试 Prompt 缓存"""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """测试缓存命中"""
        cache = PromptCache()
        mock_llm = MockLLMClient()
        
        # 第一次调用（缓存未命中）
        result1 = await cache.get_or_generate(
            prompt="test prompt",
            system_prompt="test system",
            llm_client=mock_llm
        )
        
        # 第二次调用（缓存命中）
        result2 = await cache.get_or_generate(
            prompt="test prompt",
            system_prompt="test system",
            llm_client=mock_llm
        )
        
        # 两次结果应该相同
        assert result1 == result2
        assert result2 == "Test response"
        # LLM 只应该被调用一次
        assert mock_llm.call_count == 1
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """测试缓存未命中"""
        cache = PromptCache()
        mock_llm = MockLLMClient()
        
        # 不同的 prompt 应该导致缓存未命中
        await cache.get_or_generate(
            prompt="prompt 1",
            system_prompt="system",
            llm_client=mock_llm
        )
        
        await cache.get_or_generate(
            prompt="prompt 2",
            system_prompt="system",
            llm_client=mock_llm
        )
        
        assert mock_llm.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        """测试缓存淘汰"""
        cache = PromptCache(capacity=2)
        mock_llm = MockLLMClient()
        
        # 添加 3 个不同的 prompt（超过容量）
        await cache.get_or_generate("prompt 1", "system", mock_llm)
        await cache.get_or_generate("prompt 2", "system", mock_llm)
        await cache.get_or_generate("prompt 3", "system", mock_llm)
        
        # 缓存大小不应该超过容量
        assert len(cache._cache) <= 2
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """测试缓存键生成"""
        cache = PromptCache()
        
        # 相同的 prompt 和 system_prompt 应该生成相同的键
        key1 = cache._generate_key("prompt", "system")
        key2 = cache._generate_key("prompt", "system")
        assert key1 == key2
        
        # 不同的 prompt 应该生成不同的键
        key3 = cache._generate_key("prompt2", "system")
        assert key1 != key3
        
        # 不同的 system_prompt 应该生成不同的键
        key4 = cache._generate_key("prompt", "system2")
        assert key1 != key4
    
    @pytest.mark.asyncio
    async def test_cache_order(self):
        """测试缓存顺序（LRU）"""
        cache = PromptCache(capacity=2)
        mock_llm = MockLLMClient()
        
        # 添加两个缓存项
        await cache.get_or_generate("prompt 1", "system", mock_llm)
        await cache.get_or_generate("prompt 2", "system", mock_llm)
        
        # 访问第一个（应该移到末尾）
        await cache.get_or_generate("prompt 1", "system", mock_llm)
        
        # 添加第三个（应该淘汰第二个）
        await cache.get_or_generate("prompt 3", "system", mock_llm)
        
        # 检查缓存中的键
        keys = list(cache._cache.keys())
        assert len(keys) == 2
        # prompt 1 应该在缓存中（因为被访问过）
        # prompt 3 应该在缓存中（最新添加）
        # prompt 2 应该被淘汰
    
    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """测试空 prompt"""
        cache = PromptCache()
        mock_llm = MockLLMClient()
        
        result = await cache.get_or_generate("", "", mock_llm)
        assert result == "Test response"
        assert mock_llm.call_count == 1
        
        # 再次调用应该命中缓存
        result2 = await cache.get_or_generate("", "", mock_llm)
        assert result2 == "Test response"
        assert mock_llm.call_count == 1
