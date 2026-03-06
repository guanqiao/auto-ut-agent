"""统一错误处理机制

This module provides error handling infrastructure using the unified error types
from error_types.py.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
import logging

from .error_types import (
    ErrorCategory,
    RecoveryStrategy as RecoveryStrategyEnum,
    ErrorSeverity,
    ErrorContext,
    PyUTError as PyUTErrorBase,
)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重级别"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class PyUTError(PyUTErrorBase):
    """统一错误定义 - 继承自 error_types.PyUTError"""
    pass


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


class RecoveryStrategyHandler:
    """恢复策略处理器
    
    Note: This is a handler class, not to be confused with RecoveryStrategy enum
    in error_types.py which defines the strategy types.
    """
    
    def __init__(self, name: str, handler: Callable[[PyUTError], bool], strategy_type: Optional[RecoveryStrategyEnum] = None):
        self.name = name
        self.handler = handler
        self.strategy_type = strategy_type or RecoveryStrategyEnum.RETRY
    
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
        self.strategies: Dict[ErrorCategory, List[RecoveryStrategyHandler]] = {}
    
    def add_handler(self, handler: ErrorHandler):
        """添加错误处理器"""
        self.handlers.append(handler)
    
    def add_recovery_strategy(self, category: ErrorCategory, strategy: RecoveryStrategyHandler):
        """添加恢复策略"""
        if category not in self.strategies:
            self.strategies[category] = []
        self.strategies[category].append(strategy)
    
    def handle_error(self, error: PyUTError) -> bool:
        """处理错误"""
        for handler in self.handlers:
            if handler.can_handle(error):
                handler.handle(error)
        
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
        
        key = f"{error.category.name}:{error.error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        if len(self.errors) > self.max_history:
            self.errors.pop(0)
    
    def get_error_frequency(self, category: ErrorCategory) -> int:
        """获取某类错误的发生频率"""
        return sum(
            count for key, count in self.error_counts.items()
            if key.startswith(category.name)
        )
    
    def get_recent_errors(self, limit: int = 10) -> List[PyUTError]:
        """获取最近的错误"""
        return self.errors[-limit:]
    
    def clear(self):
        """清空错误历史"""
        self.errors.clear()
        self.error_counts.clear()


def retry_strategy_factory(max_retries: int = 3):
    """重试策略工厂"""
    def retry_handler(error: PyUTError) -> bool:
        logger.info(f"Retry strategy for error: {error.error_type}")
        return True
    
    return RecoveryStrategyHandler(f"retry_{max_retries}", retry_handler, RecoveryStrategyEnum.RETRY)


def fallback_strategy_factory(fallback_action: Callable):
    """降级策略工厂"""
    def fallback_handler(error: PyUTError) -> bool:
        logger.info(f"Fallback strategy for error: {error.error_type}")
        fallback_action(error)
        return True
    
    return RecoveryStrategyHandler("fallback", fallback_handler, RecoveryStrategyEnum.FALLBACK)
