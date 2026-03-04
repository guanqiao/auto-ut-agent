"""Performance optimization utilities.

This module provides:
- LazyToolLoader: Lazy loading of tools
- ToolPreloader: Preload tools on demand
- OptimizedToolCache: Enhanced caching
"""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class LazyToolLoader:
    """Lazy loading of tools to improve startup time."""
    
    _tool_factories: Dict[str, Callable] = {}
    _loaded_tools: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, tool_name: str, factory: Callable):
        """Register a tool factory.
        
        Args:
            tool_name: Name of tool
            factory: Factory function to create tool
        """
        cls._tool_factories[tool_name] = factory
    
    @classmethod
    def get(cls, tool_name: str) -> Optional[Any]:
        """Get tool (lazy load if not loaded).
        
        Args:
            tool_name: Name of tool
        
        Returns:
            Tool instance or None
        """
        if tool_name in cls._loaded_tools:
            return cls._loaded_tools[tool_name]
        
        if tool_name in cls._tool_factories:
            tool = cls._tool_factories[tool_name]()
            cls._loaded_tools[tool_name] = tool
            logger.debug(f"[LazyToolLoader] Loaded: {tool_name}")
            return tool
        
        return None
    
    @classmethod
    def preload(cls, tool_names: List[str]):
        """Preload specific tools.
        
        Args:
            tool_names: List of tool names to preload
        """
        for name in tool_names:
            if name not in cls._loaded_tools:
                cls.get(name)
    
    @classmethod
    def clear(cls):
        """Clear loaded tools."""
        cls._loaded_tools.clear()
        logger.debug("[LazyToolLoader] Cleared")


class ToolPreloader:
    """Preload tools on demand."""
    
    def __init__(self, tool_service: Any):
        self.tool_service = tool_service
        self._preloaded = set()
    
    def preload_by_category(self, category: str):
        """Preload tools in category.
        
        Args:
            category: Tool category
        """
        tools = self.tool_service.list_available_tools(category)
        for tool_name in tools:
            if tool_name not in self._preloaded:
                self.tool_service.get_tool_info(tool_name)
                self._preloaded.add(tool_name)
        
        logger.info(f"[ToolPreloader] Preloaded {len(tools)} tools for {category}")
    
    def preload_common(self):
        """Preload commonly used tools."""
        common = ["read_file", "write_file", "grep", "bash", "git_status"]
        for tool_name in common:
            if tool_name not in self._preloaded:
                self.tool_service.get_tool_info(tool_name)
                self._preloaded.add(tool_name)
        
        logger.info(f"[ToolPreloader] Preloaded common tools")
    
    def get_preloaded(self) -> List[str]:
        """Get list of preloaded tools."""
        return list(self._preloaded)


class OptimizedCache:
    """Optimized caching with smart invalidation."""
    
    def __init__(self, max_size: int = 200, ttl: int = 1800):
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        import time
        
        if key not in self._cache:
            return None
        
        if time.time() - self._access_times.get(key, 0) > self._ttl:
            del self._cache[key]
            del self._access_times[key]
            return None
        
        self._access_times[key] = time.time()
        return self._cache[key]
    
    def set(self, key: str, value: Any):
        """Set cached value."""
        import time
        
        if len(self._cache) >= self._max_size:
            self._evict_lru()
        
        self._cache[key] = value
        self._access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used."""
        if not self._cache:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def invalidate(self, key: str):
        """Invalidate key."""
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
    
    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self._access_times.clear()


def create_lazy_loader() -> LazyToolLoader:
    """Create lazy loader."""
    return LazyToolLoader()


def create_preloader(tool_service: Any) -> ToolPreloader:
    """Create preloader."""
    return ToolPreloader(tool_service)


def create_optimized_cache(max_size: int = 200, ttl: int = 1800) -> OptimizedCache:
    """Create optimized cache."""
    return OptimizedCache(max_size, ttl)
