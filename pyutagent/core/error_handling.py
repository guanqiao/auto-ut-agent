"""统一错误处理机制"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重级别"""
    CRITICAL = 1    # 致命错误，系统无法继续
    HIGH = 2        # 严重错误，需要立即处理
    MEDIUM = 3      # 中等错误，可以稍后处理
    LOW = 4         # 轻微错误，警告级别


class ErrorCategory(Enum):
    """错误类别"""
    COMPILATION = "compilation"
    RUNTIME = "runtime"
    LOGIC = "logic"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class PyUTError:
    """统一错误定义"""
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    cause: Optional[Exception] = None
    recovery_suggestions: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"[{self.category.value}:{self.severity.name}] {self.error_type} - {self.message}"


class ErrorHandler:
    """错误处理器基类"""
    
    def can_handle(self, error: PyUTError) -> bool:
        """判断是否可以处理该错误"""
        return True
    
    def handle(self, error: PyUTError) -> bool:
        """处理错误，返回是否成功"""
        raise NotImplementedError


class LoggingErrorHandler(ErrorHandler):
    """日志记录错误处理器"""
    
    def handle(self, error: PyUTError) -> bool:
        logger.error(f"Error occurred: {error}")
        if error.context:
            logger.debug(f"Context: {error.context}")
        return True


class RecoveryStrategy:
    """恢复策略"""
    
    def __init__(self, name: str, handler: Callable[[PyUTError], bool]):
        self.name = name
        self.handler = handler
    
    def execute(self, error: PyUTError) -> bool:
        """执行恢复策略"""
        try:
            return self.handler(error)
        except Exception as e:
            logger.error(f"Recovery strategy {self.name} failed: {e}")
            return False


class ErrorPropagationChain:
    """错误传播链"""
    
    def __init__(self):
        self.handlers: List[ErrorHandler] = []
        self.strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {}
    
    def add_handler(self, handler: ErrorHandler):
        """添加错误处理器"""
        self.handlers.append(handler)
    
    def add_recovery_strategy(self, category: ErrorCategory, strategy: RecoveryStrategy):
        """添加恢复策略"""
        if category not in self.strategies:
            self.strategies[category] = []
        self.strategies[category].append(strategy)
    
    def handle_error(self, error: PyUTError) -> bool:
        """处理错误"""
        # 1. 记录错误
        for handler in self.handlers:
            if handler.can_handle(error):
                handler.handle(error)
        
        # 2. 尝试恢复
        if error.category in self.strategies:
            for strategy in self.strategies[error.category]:
                if strategy.execute(error):
                    logger.info(f"Successfully recovered using strategy: {strategy.name}")
                    return True
        
        return False


class ErrorTracker:
    """错误追踪器"""
    
    def __init__(self, max_history: int = 100):
        self.errors: List[PyUTError] = []
        self.max_history = max_history
        self.error_counts: Dict[str, int] = {}
    
    def track(self, error: PyUTError):
        """追踪错误"""
        self.errors.append(error)
        
        # 统计错误次数
        key = f"{error.category.value}:{error.error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # 限制历史记录
        if len(self.errors) > self.max_history:
            self.errors.pop(0)
    
    def get_error_frequency(self, category: ErrorCategory) -> int:
        """获取某类错误的发生频率"""
        return sum(
            count for key, count in self.error_counts.items()
            if key.startswith(category.value)
        )
    
    def get_recent_errors(self, limit: int = 10) -> List[PyUTError]:
        """获取最近的错误"""
        return self.errors[-limit:]
    
    def clear(self):
        """清空错误历史"""
        self.errors.clear()
        self.error_counts.clear()


# 预定义的恢复策略
def retry_strategy_factory(max_retries: int = 3):
    """重试策略工厂"""
    def retry_handler(error: PyUTError) -> bool:
        # 这里只是示例，实际需要结合具体场景
        logger.info(f"Retry strategy for error: {error.error_type}")
        return True  # 假设总是成功
    
    return RecoveryStrategy(f"retry_{max_retries}", retry_handler)


def fallback_strategy_factory(fallback_action: Callable):
    """降级策略工厂"""
    def fallback_handler(error: PyUTError) -> bool:
        logger.info(f"Fallback strategy for error: {error.error_type}")
        fallback_action(error)
        return True
    
    return RecoveryStrategy("fallback", fallback_handler)
