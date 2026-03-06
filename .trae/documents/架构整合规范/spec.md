# PyUTAgent 架构整合和优化规范

## 1. 概述

### 1.1 项目背景

PyUTAgent 是一个智能单元测试生成代理系统，经过长期迭代开发，存在多处功能重复和架构问题。本规范旨在指导架构整合和优化工作。

### 1.2 目标

1. **消除重复代码**：减少 20-30% 的冗余代码
2. **统一接口规范**：建立清晰的模块边界
3. **降低耦合度**：解决循环依赖问题
4. **提升可维护性**：单一职责、清晰职责划分

### 1.3 范围

本规范涵盖以下模块的整合：
- 缓存模块（6 个文件 → 1 个文件）
- 重试机制（3 个文件 → 1 个文件）
- 错误处理（5 个文件 → 1 个文件）
- 其他优化（消息总线、反馈循环、Agent 层次）

---

## 2. 架构设计

### 2.1 整体架构

```
pyutagent/
├── core/                      # 核心基础设施
│   ├── cache.py              # 统一缓存模块
│   ├── retry.py              # 统一重试模块
│   ├── error_handling.py     # 统一错误处理模块
│   ├── protocols.py          # 协议和接口定义
│   ├── exceptions.py         # 异常定义
│   ├── event_bus.py          # 事件总线
│   ├── message_bus.py        # 消息总线
│   ├── learning_feedback.py  # 学习型反馈（重命名）
│   └── termination.py        # 终止条件
│
├── agent/                     # Agent 实现
│   ├── base_agent.py         # 基础 Agent
│   ├── react_agent.py        # ReAct Agent
│   ├── enhanced_agent.py     # 增强 Agent
│   ├── components/           # Agent 组件
│   │   ├── core_agent.py
│   │   ├── feedback_loop.py
│   │   ├── recovery_manager.py
│   │   └── execution_steps.py
│   └── multi_agent/          # 多 Agent 系统
│       ├── agent_coordinator.py
│       ├── message_bus.py
│       └── ...
│
├── tools/                     # 工具模块
│   ├── robust_executor.py    # 健壮执行器
│   ├── safe_executor.py      # 安全执行器
│   └── ...
│
├── memory/                    # 内存模块
│   └── ...
│
└── llm/                       # LLM 模块
    ├── client.py
    └── ...
```

### 2.2 模块职责

#### 2.2.1 core/cache.py - 统一缓存模块

**职责：** 提供统一的缓存接口和实现

**类设计：**

```python
from typing import Protocol, Any, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
import json
import pickle
import hashlib


class CacheBackend(Protocol):
    """缓存后端协议"""
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        ...
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        ...
    
    def clear(self) -> bool:
        """清空缓存"""
        ...
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        ...


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class L1MemoryCache:
    """L1 内存缓存 - 基于 LRU 算法"""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None
    ):
        """
        初始化 L1 内存缓存
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒）
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._access_order: List[str] = []  # LRU 顺序
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                self._evict(key)
                return None
            # 更新 LRU 顺序
            self._access_order.remove(key)
            self._access_order.append(key)
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        with self._lock:
            # 如果键已存在，先删除
            if key in self._cache:
                self._evict(key)
            
            # 检查容量，必要时淘汰
            while len(self._cache) >= self._max_size:
                self._evict_lru()
            
            # 创建缓存条目
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = None
            if effective_ttl is not None:
                from datetime import timedelta
                expires_at = datetime.now() + timedelta(seconds=effective_ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at
            )
            self._cache[key] = entry
            self._access_order.append(key)
            return True
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            return self._evict(key)
    
    def clear(self) -> bool:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            return True
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._evict(key)
                return False
            return True
    
    def _evict(self, key: str) -> bool:
        """淘汰指定键"""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False
    
    def _evict_lru(self) -> bool:
        """淘汰最近最少使用的条目"""
        if not self._access_order:
            return False
        lru_key = self._access_order.pop(0)
        return self._evict(lru_key)


class L2DiskCache:
    """L2 磁盘缓存"""
    
    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 100,
        default_ttl: Optional[float] = None
    ):
        """
        初始化 L2 磁盘缓存
        
        Args:
            cache_dir: 缓存目录
            max_size_mb: 最大缓存大小（MB）
            default_ttl: 默认过期时间（秒）
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return None
        
        with self._lock:
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                if entry.is_expired():
                    cache_file.unlink()
                    return None
                return entry.value
            except Exception:
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        with self._lock:
            try:
                effective_ttl = ttl if ttl is not None else self._default_ttl
                expires_at = None
                if effective_ttl is not None:
                    from datetime import timedelta
                    expires_at = datetime.now() + timedelta(seconds=effective_ttl)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    expires_at=expires_at
                )
                
                cache_file = self._get_cache_path(key)
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry, f)
                return True
            except Exception:
                return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
            return True
        return False
    
    def clear(self) -> bool:
        """清空缓存"""
        with self._lock:
            for cache_file in self._cache_dir.glob("*.cache"):
                cache_file.unlink()
            return True
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return False
        
        with self._lock:
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                if entry.is_expired():
                    cache_file.unlink()
                    return False
                return True
            except Exception:
                return False
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"


class MultiLevelCache:
    """多级缓存协调器"""
    
    def __init__(self, backends: List[CacheBackend]):
        """
        初始化多级缓存
        
        Args:
            backends: 缓存后端列表（按优先级排序，L1 在前）
        """
        self._backends = backends
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值（从 L1 到 Ln 依次查找）"""
        for i, backend in enumerate(self._backends):
            value = backend.get(key)
            if value is not None:
                # 回填到更高级别的缓存
                for j in range(i):
                    self._backends[j].set(key, value)
                return value
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """设置缓存值（写入所有级别）"""
        success = True
        for backend in self._backends:
            if not backend.set(key, value, ttl):
                success = False
        return success
    
    def delete(self, key: str) -> bool:
        """删除缓存值（从所有级别删除）"""
        success = True
        for backend in self._backends:
            if not backend.delete(key):
                success = False
        return success
    
    def clear(self) -> bool:
        """清空所有缓存"""
        success = True
        for backend in self._backends:
            if not backend.clear():
                success = False
        return success
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        for backend in self._backends:
            if backend.exists(key):
                return True
        return False


class ToolResultCache(MultiLevelCache):
    """工具结果缓存 - 专用于工具执行结果"""
    
    def __init__(
        self,
        max_memory_entries: int = 500,
        cache_dir: Optional[Path] = None,
        default_ttl: float = 3600.0  # 1 小时
    ):
        """
        初始化工具结果缓存
        
        Args:
            max_memory_entries: 内存缓存最大条目数
            cache_dir: 磁盘缓存目录（可选）
            default_ttl: 默认过期时间（秒）
        """
        backends = [L1MemoryCache(max_memory_entries, default_ttl)]
        
        if cache_dir:
            backends.append(L2DiskCache(cache_dir, default_ttl=default_ttl))
        
        super().__init__(backends)
    
    @staticmethod
    def make_key(tool_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            "tool": tool_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()


class PromptCache(MultiLevelCache):
    """Prompt 缓存 - 专用于 LLM Prompt"""
    
    def __init__(
        self,
        max_memory_entries: int = 1000,
        cache_dir: Optional[Path] = None,
        default_ttl: float = 7200.0  # 2 小时
    ):
        """
        初始化 Prompt 缓存
        
        Args:
            max_memory_entries: 内存缓存最大条目数
            cache_dir: 磁盘缓存目录（可选）
            default_ttl: 默认过期时间（秒）
        """
        backends = [L1MemoryCache(max_memory_entries, default_ttl)]
        
        if cache_dir:
            backends.append(L2DiskCache(cache_dir, default_ttl=default_ttl))
        
        super().__init__(backends)
    
    @staticmethod
    def make_key(prompt: str, model: str, **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            "prompt": prompt,
            "model": model,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


# 全局缓存实例
_global_cache: Optional[MultiLevelCache] = None


def init_global_cache(
    max_memory_entries: int = 1000,
    cache_dir: Optional[Path] = None,
    default_ttl: float = 3600.0
) -> MultiLevelCache:
    """初始化全局缓存"""
    global _global_cache
    _global_cache = MultiLevelCache([
        L1MemoryCache(max_memory_entries, default_ttl),
        L2DiskCache(cache_dir, default_ttl=default_ttl) if cache_dir else None
    ])
    return _global_cache


def get_global_cache() -> MultiLevelCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = init_global_cache()
    return _global_cache
```

#### 2.2.2 core/retry.py - 统一重试模块

**职责：** 提供统一的重试策略和配置

**类设计：**

```python
from typing import Callable, Optional, Any, Tuple, Type, List
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio
import time
import random


class RetryStrategy(Enum):
    """重试策略"""
    IMMEDIATE = auto()              # 立即重试
    FIXED_DELAY = auto()            # 固定延迟
    LINEAR_BACKOFF = auto()         # 线性退避
    EXPONENTIAL_BACKOFF = auto()    # 指数退避
    EXPONENTIAL_JITTER = auto()     # 指数退避 + 抖动
    ADAPTIVE = auto()               # 自适应


@dataclass
class RetryConfig:
    """统一重试配置
    
    此配置用于所有重试机制，确保一致的行为和防止无限循环。
    """
    max_total_attempts: int = 50
    max_step_attempts: int = 2
    max_compilation_attempts: int = 2
    max_test_attempts: int = 2
    max_reset_count: int = 2
    
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    backoff_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    retryable_errors: Dict[str, bool] = field(default_factory=lambda: {
        "network": True,
        "timeout": True,
        "transient": True,
        "resource": True,
        "compilation": True,
        "test_failure": True,
        "llm_api": True,
    })
    
    enable_smart_retry: bool = True
    simple_retry_for_network: bool = True
    llm_analysis_for_code: bool = True
    max_llm_analysis_attempts: int = 3
    
    non_retryable_error_types: List[str] = field(default_factory=lambda: [
        "AuthenticationError",
        "PermissionError",
        "InvalidRequestError",
    ])
    
    def get_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        if self.backoff_strategy == RetryStrategy.IMMEDIATE:
            return 0
        elif self.backoff_strategy == RetryStrategy.FIXED_DELAY:
            return self.backoff_base
        elif self.backoff_strategy == RetryStrategy.LINEAR_BACKOFF:
            return min(self.backoff_base * attempt, self.backoff_max)
        elif self.backoff_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
        elif self.backoff_strategy == RetryStrategy.EXPONENTIAL_JITTER:
            delay = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_max)
            jitter = random.uniform(0, delay * 0.1)
            return delay + jitter
        else:
            return self.backoff_base
    
    def should_stop(self, attempt: int, step_name: Optional[str] = None) -> bool:
        """检查是否应该停止重试"""
        if attempt >= self.max_total_attempts:
            return True
        
        if step_name:
            step_limits = {
                "compilation": self.max_compilation_attempts,
                "test_execution": self.max_test_attempts,
            }
            step_limit = step_limits.get(step_name.lower(), self.max_step_attempts)
            if attempt >= step_limit:
                return True
        
        return False
    
    def get_max_attempts(self, step_name: Optional[str] = None) -> int:
        """获取最大重试次数"""
        if step_name:
            step_limits = {
                "compilation": self.max_compilation_attempts,
                "test_execution": self.max_test_attempts,
            }
            return step_limits.get(step_name.lower(), self.max_step_attempts)
        return self.max_step_attempts
    
    def can_reset(self, reset_count: int) -> bool:
        """检查是否允许重置"""
        return reset_count < self.max_reset_count
    
    def should_stop_reset(self, reset_count: int) -> bool:
        """检查是否应该停止重置"""
        return reset_count >= self.max_reset_count
    
    def is_retryable(self, error_category: str) -> bool:
        """检查错误是否可重试"""
        return self.retryable_errors.get(error_category.lower(), False)
    
    def with_overrides(self, **kwargs) -> 'RetryConfig':
        """创建带覆盖的新配置"""
        current_values = {
            'max_total_attempts': self.max_total_attempts,
            'max_step_attempts': self.max_step_attempts,
            'max_compilation_attempts': self.max_compilation_attempts,
            'max_test_attempts': self.max_test_attempts,
            'max_reset_count': self.max_reset_count,
            'backoff_base': self.backoff_base,
            'backoff_max': self.backoff_max,
            'backoff_strategy': self.backoff_strategy,
            'retryable_errors': self.retryable_errors.copy(),
            'enable_smart_retry': self.enable_smart_retry,
            'simple_retry_for_network': self.simple_retry_for_network,
            'llm_analysis_for_code': self.llm_analysis_for_code,
            'max_llm_analysis_attempts': self.max_llm_analysis_attempts,
        }
        current_values.update(kwargs)
        return RetryConfig(**current_values)


# 默认配置
DEFAULT_RETRY_CONFIG = RetryConfig()


def get_default_retry_config() -> RetryConfig:
    """获取默认重试配置"""
    return DEFAULT_RETRY_CONFIG


def create_retry_config(**kwargs) -> RetryConfig:
    """创建自定义重试配置"""
    return RetryConfig(**kwargs)


class RetryManager:
    """重试管理器"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or DEFAULT_RETRY_CONFIG
        self._attempt_counts: dict = {}
    
    def execute(
        self,
        operation: Callable[[], Any],
        operation_name: str = "operation"
    ) -> Any:
        """执行带重试的操作（同步）"""
        attempt = 0
        max_attempts = self.config.get_max_attempts(operation_name)
        
        while attempt < max_attempts:
            attempt += 1
            try:
                result = operation()
                return result
            except Exception as e:
                if attempt >= max_attempts:
                    raise
                
                delay = self.config.get_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
        
        raise RuntimeError(f"Max attempts ({max_attempts}) exceeded for {operation_name}")
    
    async def execute_async(
        self,
        operation: Callable[[], Any],
        operation_name: str = "operation"
    ) -> Any:
        """执行带重试的操作（异步）"""
        attempt = 0
        max_attempts = self.config.get_max_attempts(operation_name)
        
        while attempt < max_attempts:
            attempt += 1
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()
                return result
            except Exception as e:
                if attempt >= max_attempts:
                    raise
                
                delay = self.config.get_delay(attempt)
                if delay > 0:
                    await asyncio.sleep(delay)
        
        raise RuntimeError(f"Max attempts ({max_attempts}) exceeded for {operation_name}")


class InfiniteRetryManager(RetryManager):
    """无限重试管理器（用于 Agent 循环）"""
    
    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        max_iterations: int = 1000,
        check_interval: float = 1.0
    ):
        super().__init__(config)
        self.max_iterations = max_iterations
        self.check_interval = check_interval
        self._stop_requested = False
    
    def request_stop(self):
        """请求停止"""
        self._stop_requested = True
    
    def reset(self):
        """重置状态"""
        self._stop_requested = False
    
    async def run_with_retry(
        self,
        operation: Callable[[], Any],
        should_continue: Callable[[Any], bool]
    ) -> Any:
        """带重试的无限循环执行"""
        iteration = 0
        
        while not self._stop_requested and iteration < self.max_iterations:
            iteration += 1
            
            try:
                result = await self.execute_async(operation)
                if not should_continue(result):
                    return result
            except Exception as e:
                # 记录错误但继续
                pass
            
            await asyncio.sleep(self.check_interval)
        
        raise RuntimeError("Infinite retry loop terminated")
```

#### 2.2.3 core/error_handling.py - 统一错误处理模块

**职责：** 提供统一的错误分类、处理和恢复机制

**类设计：**

```python
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import traceback


class ErrorCategory(Enum):
    """错误分类"""
    TRANSIENT = auto()
    PERMANENT = auto()
    RESOURCE = auto()
    NETWORK = auto()
    TIMEOUT = auto()
    VALIDATION = auto()
    SYNTAX = auto()
    LOGIC = auto()
    COMPILATION_ERROR = auto()
    TEST_FAILURE = auto()
    TOOL_EXECUTION_ERROR = auto()
    PARSING_ERROR = auto()
    GENERATION_ERROR = auto()
    FILE_IO_ERROR = auto()
    LLM_API_ERROR = auto()
    DEPENDENCY_ERROR = auto()
    ENVIRONMENT_ERROR = auto()
    UNKNOWN = auto()


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = auto()
    RETRY_IMMEDIATE = auto()
    RETRY_WITH_BACKOFF = auto()
    BACKOFF = auto()
    FALLBACK = auto()
    RESET = auto()
    SKIP = auto()
    SKIP_AND_CONTINUE = auto()
    ABORT = auto()
    MANUAL = auto()
    ANALYZE_AND_FIX = auto()
    RESET_AND_REGENERATE = auto()
    FALLBACK_ALTERNATIVE = auto()
    ESCALATE_TO_USER = auto()
    INSTALL_DEPENDENCIES = auto()
    RESOLVE_DEPENDENCIES = auto()
    FIX_ENVIRONMENT = auto()


@dataclass
class ErrorContext:
    """错误上下文"""
    error: Exception
    error_type: Type[Exception]
    error_message: str
    stack_trace: str
    category: ErrorCategory
    timestamp: datetime
    operation: str
    attempt: int
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_exception(
        cls,
        error: Exception,
        operation: str = "unknown",
        attempt: int = 1,
        context_data: Optional[Dict[str, Any]] = None
    ) -> 'ErrorContext':
        """从异常创建错误上下文"""
        return cls(
            error=error,
            error_type=type(error),
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            category=ErrorCategory.UNKNOWN,
            timestamp=datetime.now(),
            operation=operation,
            attempt=attempt,
            context_data=context_data or {}
        )


class ErrorClassifier:
    """错误分类器"""
    
    def __init__(self):
        self._classification_rules: List[Callable[[Exception, Dict], Optional[ErrorCategory]]] = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        """注册默认分类规则"""
        self._classification_rules.extend([
            self._classify_network_error,
            self._classify_timeout_error,
            self._classify_compilation_error,
            self._classify_test_failure,
            self._classify_dependency_error,
        ])
    
    def classify(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorCategory:
        """分类错误"""
        context = context or {}
        
        for rule in self._classification_rules:
            category = rule(error, context)
            if category is not None:
                return category
        
        return ErrorCategory.UNKNOWN
    
    def _classify_network_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[ErrorCategory]:
        """分类网络错误"""
        error_name = type(error).__name__.lower()
        error_msg = str(error).lower()
        
        network_keywords = ['connection', 'network', 'socket', 'timeout', 'dns']
        if any(kw in error_name or kw in error_msg for kw in network_keywords):
            return ErrorCategory.NETWORK
        return None
    
    def _classify_timeout_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[ErrorCategory]:
        """分类超时错误"""
        if 'timeout' in type(error).__name__.lower():
            return ErrorCategory.TIMEOUT
        if 'timeout' in str(error).lower():
            return ErrorCategory.TIMEOUT
        return None
    
    def _classify_compilation_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[ErrorCategory]:
        """分类编译错误"""
        step_name = context.get('step', '').lower()
        if step_name in ('compilation', 'compile'):
            return ErrorCategory.COMPILATION_ERROR
        
        error_msg = str(error).lower()
        if 'compilation' in error_msg or 'compile error' in error_msg:
            return ErrorCategory.COMPILATION_ERROR
        return None
    
    def _classify_test_failure(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[ErrorCategory]:
        """分类测试失败"""
        step_name = context.get('step', '').lower()
        if step_name in ('test_execution', 'test', 'testing'):
            return ErrorCategory.TEST_FAILURE
        
        error_msg = str(error).lower()
        if 'test failed' in error_msg or 'assertion' in error_msg:
            return ErrorCategory.TEST_FAILURE
        return None
    
    def _classify_dependency_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[ErrorCategory]:
        """分类依赖错误"""
        error_msg = str(error).lower()
        
        dependency_keywords = [
            'cannot find symbol',
            'package does not exist',
            'classnotfound',
            'no such file',
            'dependency',
            'maven',
        ]
        
        if any(kw in error_msg for kw in dependency_keywords):
            return ErrorCategory.DEPENDENCY_ERROR
        return None


# 全局错误分类器实例
_error_classifier: Optional[ErrorClassifier] = None


def get_error_classifier() -> ErrorClassifier:
    """获取全局错误分类器"""
    global _error_classifier
    if _error_classifier is None:
        _error_classifier = ErrorClassifier()
    return _error_classifier


class RecoveryHandler(Protocol):
    """恢复处理器协议"""
    
    def can_handle(self, error: Exception, context: ErrorContext) -> bool:
        """检查是否能处理此错误"""
        ...
    
    def handle(
        self,
        error: Exception,
        context: ErrorContext
    ) -> Dict[str, Any]:
        """处理错误，返回恢复结果"""
        ...


@dataclass
class RecoveryResult:
    """恢复结果"""
    success: bool
    action: str
    message: str
    should_continue: bool = True
    data: Dict[str, Any] = field(default_factory=dict)


class RecoveryManager:
    """恢复管理器"""
    
    def __init__(self):
        self._handlers: List[RecoveryHandler] = []
        self._classifier = get_error_classifier()
    
    def register_handler(self, handler: RecoveryHandler):
        """注册恢复处理器"""
        self._handlers.append(handler)
    
    async def recover(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> RecoveryResult:
        """执行错误恢复"""
        context = context or {}
        
        # 分类错误
        category = self._classifier.classify(error, context)
        
        # 创建错误上下文
        error_context = ErrorContext.from_exception(
            error=error,
            operation=context.get('step', 'unknown'),
            attempt=context.get('attempt', 1),
            context_data=context
        )
        error_context.category = category
        
        # 查找合适的处理器
        for handler in self._handlers:
            if handler.can_handle(error, error_context):
                result = handler.handle(error, error_context)
                return RecoveryResult(
                    success=result.get('success', False),
                    action=result.get('action', 'unknown'),
                    message=result.get('message', ''),
                    should_continue=result.get('should_continue', True),
                    data=result.get('data', {})
                )
        
        # 默认处理
        return RecoveryResult(
            success=False,
            action='abort',
            message='No handler found for this error',
            should_continue=False
        )
```

---

## 3. 接口规范

### 3.1 缓存接口

所有缓存实现必须遵循 `CacheBackend` 协议：

```python
class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool: ...
    def delete(self, key: str) -> bool: ...
    def clear(self) -> bool: ...
    def exists(self, key: str) -> bool: ...
```

### 3.2 重试接口

所有重试操作必须使用 `RetryConfig` 进行配置：

```python
config = RetryConfig(
    max_total_attempts=10,
    backoff_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
)
manager = RetryManager(config)
result = await manager.execute_async(operation, "my_operation")
```

### 3.3 错误处理接口

所有错误恢复必须通过 `RecoveryManager` 进行：

```python
manager = RecoveryManager()
result = await manager.recover(error, {"step": "compilation"})
if result.should_continue:
    # 继续执行
    pass
```

---

## 4. 迁移指南

### 4.1 缓存迁移

**旧代码：**
```python
from pyutagent.tools.tool_cache import ToolResultCache
from pyutagent.llm.multi_level_cache import MultiLevelCache
from pyutagent.llm.prompt_cache import PromptCache
```

**新代码：**
```python
from pyutagent.core.cache import ToolResultCache, MultiLevelCache, PromptCache
```

### 4.2 重试迁移

**旧代码：**
```python
from pyutagent.core.retry_config import RetryConfig, DEFAULT_RETRY_CONFIG
from pyutagent.core.retry_manager import RetryManager
```

**新代码：**
```python
from pyutagent.core.retry import RetryConfig, DEFAULT_RETRY_CONFIG, RetryManager
```

### 4.3 错误处理迁移

**旧代码：**
```python
from pyutagent.core.error_classification import ErrorClassificationService
from pyutagent.core.error_recovery import ErrorRecoveryManager, RecoveryStrategy
```

**新代码：**
```python
from pyutagent.core.error_handling import ErrorClassifier, RecoveryManager, RecoveryStrategy
```

---

## 5. 约束条件

### 5.1 设计约束

1. **向后兼容**：保留原有公共接口，内部实现可变更
2. **单一职责**：每个模块只负责一个核心功能
3. **依赖方向**：依赖方向为 `agent → core → protocols`，禁止反向依赖
4. **无循环依赖**：模块间不得存在循环依赖

### 5.2 实现约束

1. **线程安全**：缓存实现必须是线程安全的
2. **异步支持**：重试和恢复机制必须支持异步操作
3. **可测试性**：所有公共接口必须可 mock
4. **日志记录**：关键操作必须记录日志

### 5.3 性能约束

1. **缓存命中率**：L1 缓存命中率应 > 80%
2. **重试延迟**：最大重试延迟不超过 60 秒
3. **错误分类**：错误分类耗时不超过 10ms

---

## 6. 质量属性

### 6.1 可维护性

- 代码重复率 < 5%
- 模块耦合度 < 0.3
- 平均圈复杂度 < 10

### 6.2 可测试性

- 单元测试覆盖率 > 80%
- 集成测试覆盖核心流程
- 性能测试覆盖关键路径

### 6.3 可扩展性

- 支持自定义缓存后端
- 支持自定义重试策略
- 支持自定义恢复处理器

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 导入错误导致运行失败 | 高 | 小步重构，每步运行测试 |
| 功能行为变化 | 高 | 保持接口兼容，添加适配层 |
| 测试覆盖不足 | 中 | 先补充测试，再重构 |
| 循环依赖 | 高 | 使用依赖注入，延迟导入 |

---

## 8. 参考资料

- [ARCHITECTURE.md](file:///d:/opensource/github/auto-ut-agent/ARCHITECTURE.md)
- [Python Protocol 文档](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [设计模式：策略模式](https://refactoring.guru/design-patterns/strategy)
