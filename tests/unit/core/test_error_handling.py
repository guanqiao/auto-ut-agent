"""测试统一错误处理机制"""
import pytest
from pyutagent.core.error_handling import (
    ErrorSeverity,
    ErrorCategory,
    PyUTError,
    ErrorHandler,
    LoggingErrorHandler,
    RecoveryStrategy,
    ErrorPropagationChain,
    ErrorTracker,
    retry_strategy_factory
)
from pyutagent.core.error_types import (
    CompilationErrorType,
    RuntimeErrorType,
    NetworkErrorType
)


class TestErrorDefinitions:
    """测试错误定义"""
    
    def test_error_severity_levels(self):
        """测试错误严重级别"""
        assert ErrorSeverity.CRITICAL.value == 1
        assert ErrorSeverity.HIGH.value == 2
        assert ErrorSeverity.MEDIUM.value == 3
        assert ErrorSeverity.LOW.value == 4
    
    def test_error_categories(self):
        """测试错误类别"""
        assert ErrorCategory.COMPILATION.value == "compilation"
        assert ErrorCategory.RUNTIME.value == "runtime"
        assert ErrorCategory.NETWORK.value == "network"
    
    def test_pyut_error_creation(self):
        """测试创建 PyUTError"""
        error = PyUTError(
            error_type="test_error",
            message="Test error message",
            category=ErrorCategory.COMPILATION,
            severity=ErrorSeverity.HIGH
        )
        
        assert error.error_type == "test_error"
        assert error.message == "Test error message"
        assert error.category == ErrorCategory.COMPILATION
        assert error.severity == ErrorSeverity.HIGH
    
    def test_pyut_error_with_context(self):
        """测试带上下文的错误"""
        error = PyUTError(
            error_type="compilation_failed",
            message="Cannot compile test class",
            category=ErrorCategory.COMPILATION,
            severity=ErrorSeverity.HIGH,
            context={
                "file": "TestFile.java",
                "line": 42,
                "compiler_output": "error: cannot find symbol"
            },
            recovery_suggestions=[
                "Check import statements",
                "Verify classpath configuration"
            ]
        )
        
        assert error.context["file"] == "TestFile.java"
        assert error.context["line"] == 42
        assert len(error.recovery_suggestions) == 2
    
    def test_pyut_error_string_representation(self):
        """测试错误字符串表示"""
        error = PyUTError(
            error_type="test_error",
            message="Test message",
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.MEDIUM
        )
        
        error_str = str(error)
        assert "runtime" in error_str
        assert "MEDIUM" in error_str
        assert "test_error" in error_str


class TestErrorHandlers:
    """测试错误处理器"""
    
    def test_logging_error_handler(self):
        """测试日志错误处理器"""
        handler = LoggingErrorHandler()
        error = PyUTError(
            error_type="test",
            message="Test error",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.LOW
        )
        
        # 应该可以处理所有错误
        assert handler.can_handle(error) is True
        
        # 处理错误（不应该抛出异常）
        result = handler.handle(error)
        assert result is True


class TestRecoveryStrategies:
    """测试恢复策略"""
    
    def test_recovery_strategy_creation(self):
        """测试创建恢复策略"""
        def dummy_handler(error: PyUTError) -> bool:
            return True
        
        strategy = RecoveryStrategy("test_strategy", dummy_handler)
        
        assert strategy.name == "test_strategy"
        
        error = PyUTError(
            error_type="test",
            message="Test",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.LOW
        )
        
        result = strategy.execute(error)
        assert result is True
    
    def test_retry_strategy_factory(self):
        """测试重试策略工厂"""
        strategy = retry_strategy_factory(max_retries=3)
        
        assert strategy.name == "retry_3"
        
        error = PyUTError(
            error_type="network_error",
            message="Connection timeout",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH
        )
        
        # 重试策略应该执行成功
        result = strategy.execute(error)
        assert result is True


class TestErrorPropagationChain:
    """测试错误传播链"""
    
    def test_error_propagation_chain_creation(self):
        """测试创建错误传播链"""
        chain = ErrorPropagationChain()
        
        assert len(chain.handlers) == 0
        assert len(chain.strategies) == 0
    
    def test_add_handler(self):
        """测试添加错误处理器"""
        chain = ErrorPropagationChain()
        handler = LoggingErrorHandler()
        
        chain.add_handler(handler)
        
        assert len(chain.handlers) == 1
        assert chain.handlers[0] == handler
    
    def test_add_recovery_strategy(self):
        """测试添加恢复策略"""
        chain = ErrorPropagationChain()
        strategy = retry_strategy_factory()
        
        chain.add_recovery_strategy(ErrorCategory.NETWORK, strategy)
        
        assert ErrorCategory.NETWORK in chain.strategies
        assert len(chain.strategies[ErrorCategory.NETWORK]) == 1
    
    def test_handle_error_with_handlers_only(self):
        """测试只有处理器的错误处理"""
        chain = ErrorPropagationChain()
        chain.add_handler(LoggingErrorHandler())
        
        error = PyUTError(
            error_type="test",
            message="Test error",
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.LOW
        )
        
        # 没有恢复策略，应该返回 False
        result = chain.handle_error(error)
        assert result is False
    
    def test_handle_error_with_recovery(self):
        """测试带恢复的错误处理"""
        chain = ErrorPropagationChain()
        chain.add_handler(LoggingErrorHandler())
        
        def success_handler(error: PyUTError) -> bool:
            return True
        
        chain.add_recovery_strategy(
            ErrorCategory.COMPILATION,
            RecoveryStrategy("fix_compilation", success_handler)
        )
        
        error = PyUTError(
            error_type="compilation_failed",
            message="Cannot compile",
            category=ErrorCategory.COMPILATION,
            severity=ErrorSeverity.HIGH
        )
        
        # 有恢复策略且成功
        result = chain.handle_error(error)
        assert result is True


class TestErrorTracker:
    """测试错误追踪器"""
    
    def test_error_tracker_creation(self):
        """测试创建错误追踪器"""
        tracker = ErrorTracker()
        
        assert len(tracker.errors) == 0
        assert tracker.max_history == 100
    
    def test_track_error(self):
        """测试追踪错误"""
        tracker = ErrorTracker()
        
        error = PyUTError(
            error_type="test_error",
            message="Test",
            category=ErrorCategory.RUNTIME,
            severity=ErrorSeverity.MEDIUM
        )
        
        tracker.track(error)
        
        assert len(tracker.errors) == 1
        assert tracker.errors[0] == error
    
    def test_error_frequency_tracking(self):
        """测试错误频率追踪"""
        tracker = ErrorTracker()
        
        # 添加多个同类错误
        for i in range(3):
            error = PyUTError(
                error_type="network_timeout",
                message=f"Timeout {i}",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH
            )
            tracker.track(error)
        
        frequency = tracker.get_error_frequency(ErrorCategory.NETWORK)
        assert frequency == 3
    
    def test_get_recent_errors(self):
        """测试获取最近的错误"""
        tracker = ErrorTracker()
        
        # 添加 10 个错误
        for i in range(10):
            error = PyUTError(
                error_type=f"error_{i}",
                message=f"Error {i}",
                category=ErrorCategory.RUNTIME,
                severity=ErrorSeverity.LOW
            )
            tracker.track(error)
        
        recent = tracker.get_recent_errors(limit=5)
        assert len(recent) == 5
        assert recent[-1].error_type == "error_9"  # 最近的错误
    
    def test_error_history_limit(self):
        """测试错误历史记录限制"""
        tracker = ErrorTracker(max_history=5)
        
        # 添加超过限制的错误
        for i in range(10):
            error = PyUTError(
                error_type=f"error_{i}",
                message=f"Error {i}",
                category=ErrorCategory.RUNTIME,
                severity=ErrorSeverity.LOW
            )
            tracker.track(error)
        
        # 应该只保留最近的 5 个
        assert len(tracker.errors) == 5
        assert tracker.errors[0].error_type == "error_5"
    
    def test_clear_errors(self):
        """测试清空错误"""
        tracker = ErrorTracker()
        
        # 添加一些错误
        for i in range(3):
            error = PyUTError(
                error_type=f"error_{i}",
                message=f"Error {i}",
                category=ErrorCategory.RUNTIME,
                severity=ErrorSeverity.LOW
            )
            tracker.track(error)
        
        # 清空
        tracker.clear()
        
        assert len(tracker.errors) == 0
        assert len(tracker.error_counts) == 0


class TestErrorTypes:
    """测试错误类型枚举"""
    
    def test_compilation_error_types(self):
        """测试编译错误类型"""
        assert CompilationErrorType.SYNTAX_ERROR.value == "syntax_error"
        assert CompilationErrorType.IMPORT_ERROR.value == "import_error"
        assert CompilationErrorType.SYMBOL_NOT_FOUND.value == "symbol_not_found"
    
    def test_runtime_error_types(self):
        """测试运行时错误类型"""
        assert RuntimeErrorType.NULL_POINTER.value == "null_pointer"
        assert RuntimeErrorType.INDEX_OUT_OF_BOUNDS.value == "index_out_of_bounds"
    
    def test_network_error_types(self):
        """测试网络错误类型"""
        assert NetworkErrorType.CONNECTION_ERROR.value == "connection_error"
        assert NetworkErrorType.TIMEOUT.value == "timeout"
        assert NetworkErrorType.RATE_LIMIT.value == "rate_limit"
