"""Prompt 缓存 - 优化 LLM 调用性能"""
from typing import Dict, Any
import hashlib
from collections import OrderedDict
from pyutagent.llm.client import LLMClient
import logging

logger = logging.getLogger(__name__)


class PromptCache:
    """Prompt 结果缓存"""
    
    def __init__(self, capacity: int = 1000):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._capacity = capacity
        self._hits = 0
        self._misses = 0
    
    def _generate_key(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:
        """生成缓存键"""
        key_data = f"{system_prompt}|||{prompt}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get_or_generate(
        self,
        prompt: str,
        system_prompt: str,
        llm_client: LLMClient
    ) -> str:
        """获取或生成"""
        key = self._generate_key(prompt, system_prompt)
        
        # 检查缓存
        if key in self._cache:
            logger.debug(f"Prompt cache hit: {key[:16]}...")
            self._hits += 1
            # 移到末尾（LRU）
            self._cache.move_to_end(key)
            return self._cache[key]
        
        # 未命中
        logger.debug(f"Prompt cache miss: {key[:16]}...")
        self._misses += 1
        
        # 生成
        response = await llm_client.agenerate(prompt, system_prompt)
        
        # 缓存结果
        self._cache[key] = response
        self._cache.move_to_end(key)
        
        # 淘汰最旧的
        if len(self._cache) > self._capacity:
            oldest_key = next(iter(self._cache))
            self._cache.popitem(last=False)
            logger.debug(f"Evicted oldest cache entry: {oldest_key[:16]}...")
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "capacity": self._capacity,
            "current_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Prompt cache cleared")
