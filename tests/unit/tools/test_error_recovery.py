"""Unit tests for error_recovery module.

This module provides comprehensive tests for error recovery functionality,
including error classification, recovery management, state preservation,
and graceful degradation.
"""

import pytest
import asyncio
import allure
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from pyutagent.tools.error_recovery import (
    ErrorClassifier, ErrorCategory, RecoveryManager, RecoveryStrategy,
    StatePreserver, GracefulDegradation, ErrorContext, RecoveryResult,
    safe_execution_context, create_recovery_manager, classify_error,
    is_retryable_error
)


# Custom decorator for display descriptions
def display_description(description: str):
    """Decorator to add display description for test cases."""
    def decorator(func):
        func.display_description = description
        return allure.description(description)(func)
    return decorator


@allure.feature("Error Recovery")
@allure.story("Error Classification")
class TestErrorClassifier:
    """Tests for ErrorClassifier class with display descriptions."""
    
    @display_description("验证网络错误分类 - ConnectionError 应被分类为 NETWORK")
    @allure.title("Test network error classification")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_classify_connection_error(self):
        """Test classifying connection errors."""
        error = ConnectionError("Connection refused")
        category = ErrorClassifier.classify(error)
        
        assert category == ErrorCategory.NETWORK
    
    @display_description("验证超时错误分类 - TimeoutError 应被分类为 TIMEOUT")
    @allure.title("Test timeout error classification")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_classify_timeout_error(self):
        """Test classifying timeout errors."""
        error = TimeoutError("Operation timed out")
        category = ErrorClassifier.classify(error)
        
        assert category == ErrorCategory.TIMEOUT
    
    @display_description("验证资源错误分类 - MemoryError 应被分类为 RESOURCE")
    @allure.title("Test resource error classification")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_classify_memory_error(self):
        """Test classifying memory errors."""
        error = MemoryError("Out of memory")
        category = ErrorClassifier.classify(error)
        
        assert category == ErrorCategory.RESOURCE
    
    @display_description("验证验证错误分类 - ValueError 应被分类为 VALIDATION")
    @allure.title("Test validation error classification")
    @allure.severity(allure.severity_level.NORMAL)
    def test_classify_validation_error(self):
        """Test classifying validation errors."""
        error = ValueError("Invalid value")
        category = ErrorClassifier.classify(error)
        
        assert category == ErrorCategory.VALIDATION
    
    @display_description("验证基于消息的错误分类 - 包含 'connection' 的错误应被分类为 NETWORK")
    @allure.title("Test message-based error classification")
    @allure.severity(allure.severity_level.NORMAL)
    def test_classify_by_message(self):
        """Test classifying errors based on message content."""
        error = Exception("Network connection failed")
        category = ErrorClassifier.classify(error)
        
        assert category == ErrorCategory.NETWORK
    
    @display_description("验证未知错误分类 - 无法识别的错误应被分类为 UNKNOWN")
    @allure.title("Test unknown error classification")
    @allure.severity(allure.severity_level.MINOR)
    def test_classify_unknown_error(self):
        """Test classifying unknown errors."""
        error = Exception("Some random error")
        category = ErrorClassifier.classify(error)
        
        assert category == ErrorCategory.UNKNOWN
    
    @display_description("验证可重试错误检测 - 网络错误应可重试")
    @allure.title("Test retryable error detection")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_is_retryable_network_error(self):
        """Test detecting retryable network errors."""
        error = ConnectionError("Connection failed")
        
        assert ErrorClassifier.is_retryable(error) is True
    
    @display_description("验证不可重试错误检测 - 验证错误不应重试")
    @allure.title("Test non-retryable error detection")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_is_not_retryable_validation_error(self):
        """Test detecting non-retryable validation errors."""
        error = ValueError("Invalid input")
        
        assert ErrorClassifier.is_retryable(error) is False


@allure.feature("Error Recovery")
@allure.story("Recovery Management")
class TestRecoveryManager:
    """Tests for RecoveryManager class with display descriptions."""
    
    @pytest.fixture
    def recovery_manager(self):
        """Create a RecoveryManager instance."""
        return RecoveryManager(max_retries=3, backoff_base=0.1)
    
    @display_description("验证成功操作执行 - 操作应成功执行无需重试")
    @allure.title("Test successful operation execution")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_execute_successful_operation(self, recovery_manager):
        """Test executing a successful operation."""
        operation = Mock(return_value="success")
        
        result = await recovery_manager.execute_with_recovery(
            operation, "test_operation"
        )
        
        assert result.success is True
        assert result.attempts_made == 1
        assert result.recovered_data == "success"
        operation.assert_called_once()
    
    @display_description("验证失败操作重试 - 失败操作应在重试后成功")
    @allure.title("Test failed operation with retry")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, recovery_manager):
        """Test executing operation that fails then succeeds."""
        operation = Mock(side_effect=[ConnectionError("Failed"), "success"])
        
        result = await recovery_manager.execute_with_recovery(
            operation, "test_operation"
        )
        
        assert result.success is True
        assert result.attempts_made == 2
        assert operation.call_count == 2
    
    @display_description("验证所有重试失败后返回失败")
    @allure.title("Test all retries exhausted")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_execute_all_retries_failed(self, recovery_manager):
        """Test executing operation that always fails."""
        operation = Mock(side_effect=ConnectionError("Always fails"))
        
        result = await recovery_manager.execute_with_recovery(
            operation, "test_operation"
        )
        
        assert result.success is False
        assert result.attempts_made == 3
        assert operation.call_count == 3
    
    @display_description("验证降级策略 - 主操作失败后应使用降级方案")
    @allure.title("Test fallback strategy")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_execute_with_fallback(self, recovery_manager):
        """Test executing with fallback."""
        operation = Mock(side_effect=Exception("Always fails"))
        fallback = Mock(return_value="fallback_success")
        
        result = await recovery_manager.execute_with_recovery(
            operation, "test_operation", fallback=fallback
        )
        
        assert result.success is True
        assert result.strategy_used == RecoveryStrategy.FALLBACK
        assert result.recovered_data == "fallback_success"
    
    @display_description("验证异步操作执行")
    @allure.title("Test async operation execution")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_execute_async_operation(self, recovery_manager):
        """Test executing async operation."""
        async def async_operation():
            return "async_success"
        
        result = await recovery_manager.execute_with_recovery(
            async_operation, "async_test"
        )
        
        assert result.success is True
        assert result.recovered_data == "async_success"
    
    @display_description("验证非重试错误直接失败")
    @allure.title("Test non-retryable error")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_non_retryable_error(self, recovery_manager):
        """Test that non-retryable errors fail immediately."""
        operation = Mock(side_effect=ValueError("Invalid value"))
        
        result = await recovery_manager.execute_with_recovery(
            operation, "test_operation"
        )
        
        assert result.success is False
        assert operation.call_count == 1  # Should not retry
    
    @display_description("验证恢复统计信息")
    @allure.title("Test recovery statistics")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_recovery_statistics(self, recovery_manager):
        """Test recovery statistics tracking."""
        # Execute some operations
        success_op = Mock(return_value="success")
        fail_op = Mock(side_effect=Exception("Fail"))
        
        await recovery_manager.execute_with_recovery(success_op, "op1")
        await recovery_manager.execute_with_recovery(fail_op, "op2")
        
        stats = recovery_manager.get_recovery_stats()
        
        assert "op1" in stats
        assert "op2" in stats
        assert stats["op1"]["success"] == 1
        assert stats["op2"]["failure"] == 1
    
    @display_description("验证错误历史记录")
    @allure.title("Test error history")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_error_history(self, recovery_manager):
        """Test error history tracking."""
        operation = Mock(side_effect=ConnectionError("Test error"))
        
        await recovery_manager.execute_with_recovery(operation, "test_op")
        
        history = recovery_manager.get_error_history()
        
        assert len(history) > 0
        assert history[0].operation == "test_op"
        assert "Test error" in history[0].error_message


@allure.feature("Error Recovery")
@allure.story("State Preservation")
class TestStatePreserver:
    """Tests for StatePreserver class with display descriptions."""
    
    @pytest.fixture
    def state_preserver(self):
        """Create a StatePreserver instance."""
        return StatePreserver()
    
    @display_description("验证状态保存 - 应成功保存状态并返回版本号")
    @allure.title("Test state saving")
    @allure.severity(allure.severity_level.BLOCKER)
    def test_save_state(self, state_preserver):
        """Test saving state."""
        state = {"code": "test code", "version": 1}
        
        version = state_preserver.save_state(state, label="test_save")
        
        assert version == 0
        assert len(state_preserver.state_stack) == 1
    
    @display_description("验证状态恢复 - 应能恢复到之前保存的状态")
    @allure.title("Test state restoration")
    @allure.severity(allure.severity_level.BLOCKER)
    def test_restore_state(self, state_preserver):
        """Test restoring state."""
        original_state = {"code": "original", "version": 1}
        state_preserver.save_state(original_state, label="original")
        
        restored = state_preserver.restore_state()
        
        assert restored is not None
        assert restored["code"] == "original"
    
    @display_description("验证按版本号恢复状态")
    @allure.title("Test restore by version")
    @allure.severity(allure.severity_level.NORMAL)
    def test_restore_by_version(self, state_preserver):
        """Test restoring specific version."""
        state1 = {"code": "version1"}
        state2 = {"code": "version2"}
        
        version1 = state_preserver.save_state(state1, label="v1")
        version2 = state_preserver.save_state(state2, label="v2")
        
        restored = state_preserver.restore_state(version1)
        
        assert restored["code"] == "version1"
    
    @display_description("验证恢复不存在版本返回 None")
    @allure.title("Test restore non-existent version")
    @allure.severity(allure.severity_level.NORMAL)
    def test_restore_nonexistent_version(self, state_preserver):
        """Test restoring non-existent version."""
        restored = state_preserver.restore_state(999)
        
        assert restored is None
    
    @display_description("验证空状态恢复返回 None")
    @allure.title("Test restore from empty stack")
    @allure.severity(allure.severity_level.MINOR)
    def test_restore_empty_stack(self, state_preserver):
        """Test restoring when stack is empty."""
        restored = state_preserver.restore_state()
        
        assert restored is None
    
    @display_description("验证状态历史清理")
    @allure.title("Test clear history")
    @allure.severity(allure.severity_level.NORMAL)
    def test_clear_history(self, state_preserver):
        """Test clearing state history."""
        state_preserver.save_state({"code": "test"}, label="test")
        
        state_preserver.clear_history()
        
        assert len(state_preserver.state_stack) == 0
    
    @display_description("验证状态栈大小限制")
    @allure.title("Test state stack size limit")
    @allure.severity(allure.severity_level.NORMAL)
    def test_state_stack_limit(self, state_preserver):
        """Test that state stack has size limit."""
        state_preserver.max_stack_size = 3
        
        for i in range(5):
            state_preserver.save_state({"code": f"v{i}"}, label=f"v{i}")
        
        assert len(state_preserver.state_stack) == 3


@allure.feature("Error Recovery")
@allure.story("Graceful Degradation")
class TestGracefulDegradation:
    """Tests for GracefulDegradation class with display descriptions."""
    
    @pytest.fixture
    def degradation(self):
        """Create a GracefulDegradation instance."""
        return GracefulDegradation()
    
    @display_description("验证降级链注册 - 应成功注册多个降级方法")
    @allure.title("Test degradation chain registration")
    @allure.severity(allure.severity_level.BLOCKER)
    def test_register_degradation_chain(self, degradation):
        """Test registering degradation chain."""
        methods = [
            Mock(return_value="high_quality"),
            Mock(return_value="medium_quality"),
            Mock(return_value="low_quality"),
        ]
        
        degradation.register_degradation_chain("test_op", methods)
        
        assert "test_op" in degradation.degradation_levels
        assert len(degradation.degradation_levels["test_op"]) == 3
    
    @display_description("验证最佳质量方法执行")
    @allure.title("Test best quality method execution")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_execute_best_quality(self, degradation):
        """Test executing with best quality method."""
        best_method = Mock(return_value="best_result")
        fallback_method = Mock(return_value="fallback_result")
        
        degradation.register_degradation_chain(
            "test_op", [best_method, fallback_method]
        )
        
        result = await degradation.execute_with_degradation("test_op")
        
        assert result == "best_result"
        best_method.assert_called_once()
        fallback_method.assert_not_called()
    
    @display_description("验证降级到次优方法")
    @allure.title("Test degradation to lower quality")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_degrade_to_lower_quality(self, degradation):
        """Test degrading to lower quality method."""
        best_method = Mock(side_effect=Exception("Best failed"))
        medium_method = Mock(return_value="medium_result")
        low_method = Mock(return_value="low_result")
        
        degradation.register_degradation_chain(
            "test_op", [best_method, medium_method, low_method]
        )
        
        result = await degradation.execute_with_degradation("test_op")
        
        assert result == "medium_result"
        assert degradation.current_level["test_op"] == 1
    
    @display_description("验证所有方法失败时抛出异常")
    @allure.title("Test all methods fail")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_all_methods_fail(self, degradation):
        """Test when all degradation methods fail."""
        methods = [
            Mock(side_effect=Exception("Fail 1")),
            Mock(side_effect=Exception("Fail 2")),
        ]
        
        degradation.register_degradation_chain("test_op", methods)
        
        with pytest.raises(Exception, match="All degradation levels failed"):
            await degradation.execute_with_degradation("test_op")
    
    @display_description("验证异步降级方法执行")
    @allure.title("Test async degradation methods")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_async_degradation_methods(self, degradation):
        """Test executing async degradation methods."""
        async def async_method():
            return "async_result"
        
        degradation.register_degradation_chain("test_op", [async_method])
        
        result = await degradation.execute_with_degradation("test_op")
        
        assert result == "async_result"
    
    @display_description("验证降级级别重置")
    @allure.title("Test reset degradation level")
    @allure.severity(allure.severity_level.NORMAL)
    def test_reset_level(self, degradation):
        """Test resetting degradation level."""
        degradation.register_degradation_chain("test_op", [Mock()])
        degradation.current_level["test_op"] = 2
        
        degradation.reset_level("test_op")
        
        assert degradation.current_level["test_op"] == 0
    
    @display_description("验证未注册操作抛出异常")
    @allure.title("Test unregistered operation")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_unregistered_operation(self, degradation):
        """Test executing unregistered operation."""
        with pytest.raises(ValueError, match="No degradation chain registered"):
            await degradation.execute_with_degradation("unregistered_op")


@allure.feature("Error Recovery")
@allure.story("Utility Functions")
class TestUtilityFunctions:
    """Tests for utility functions with display descriptions."""
    
    @display_description("验证便捷函数创建恢复管理器")
    @allure.title("Test create_recovery_manager utility")
    @allure.severity(allure.severity_level.NORMAL)
    def test_create_recovery_manager(self):
        """Test create_recovery_manager function."""
        manager = create_recovery_manager(max_retries=5, backoff_base=2.0)
        
        assert isinstance(manager, RecoveryManager)
        assert manager.max_retries == 5
        assert manager.backoff_base == 2.0
    
    @display_description("验证便捷错误分类函数")
    @allure.title("Test classify_error utility")
    @allure.severity(allure.severity_level.NORMAL)
    def test_classify_error(self):
        """Test classify_error function."""
        error = ConnectionError("Test")
        category = classify_error(error)
        
        assert category == ErrorCategory.NETWORK
    
    @display_description("验证便捷可重试检测函数")
    @allure.title("Test is_retryable_error utility")
    @allure.severity(allure.severity_level.NORMAL)
    def test_is_retryable_error(self):
        """Test is_retryable_error function."""
        assert is_retryable_error(ConnectionError("Test")) is True
        assert is_retryable_error(ValueError("Test")) is False


@allure.feature("Error Recovery")
@allure.story("Safe Execution Context")
class TestSafeExecutionContext:
    """Tests for safe_execution_context with display descriptions."""
    
    @display_description("验证安全执行上下文成功执行")
    @allure.title("Test safe context success")
    @allure.severity(allure.severity_level.NORMAL)
    def test_safe_context_success(self):
        """Test safe execution context with success."""
        with safe_execution_context("test_op"):
            result = "success"
        
        assert result == "success"
    
    @display_description("验证安全执行上下文错误处理")
    @allure.title("Test safe context error handling")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_safe_context_error(self):
        """Test safe execution context with error."""
        error_callback = Mock()
        
        with pytest.raises(ValueError):
            with safe_execution_context("test_op", on_error=error_callback):
                raise ValueError("Test error")
        
        error_callback.assert_called_once()
    
    @display_description("验证错误回调异常不中断主流程")
    @allure.title("Test error callback exception handling")
    @allure.severity(allure.severity_level.NORMAL)
    def test_error_callback_exception(self):
        """Test that error callback exception doesn't mask original error."""
        def failing_callback(e):
            raise RuntimeError("Callback failed")
        
        with pytest.raises(ValueError, match="Original error"):
            with safe_execution_context("test_op", on_error=failing_callback):
                raise ValueError("Original error")


@allure.feature("Error Recovery")
@allure.story("Error Context")
class TestErrorContext:
    """Tests for ErrorContext dataclass with display descriptions."""
    
    @display_description("验证从异常创建错误上下文")
    @allure.title("Test error context creation from exception")
    @allure.severity(allure.severity_level.NORMAL)
    def test_error_context_from_exception(self):
        """Test creating ErrorContext from exception."""
        try:
            raise ConnectionError("Test error")
        except Exception as e:
            context = ErrorContext.from_exception(e, "test_op", attempt=2)
        
        assert context.error_type == ConnectionError
        assert context.operation == "test_op"
        assert context.attempt == 2
        assert context.category == ErrorCategory.NETWORK
        assert "Test error" in context.error_message
        assert context.stack_trace is not None
    
    @display_description("验证错误上下文包含上下文数据")
    @allure.title("Test error context with context data")
    @allure.severity(allure.severity_level.NORMAL)
    def test_error_context_with_data(self):
        """Test ErrorContext with additional context data."""
        error = ValueError("Test")
        context_data = {"key": "value", "number": 42}
        
        context = ErrorContext.from_exception(
            error, "test_op", context_data=context_data
        )
        
        assert context.context_data == context_data


@allure.feature("Error Recovery")
@allure.story("Integration Tests")
class TestIntegration:
    """Integration tests with display descriptions."""
    
    @display_description("验证完整恢复流程 - 从错误到恢复")
    @allure.title("Test complete recovery workflow")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_complete_recovery_workflow(self):
        """Test complete error recovery workflow."""
        # Setup
        recovery_manager = create_recovery_manager(max_retries=3)
        state_preserver = StatePreserver()
        
        # Save initial state
        initial_state = {"code": "initial", "version": 1}
        state_version = state_preserver.save_state(initial_state, "initial")
        
        # Simulate operation that fails then succeeds
        call_count = 0
        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError(f"Attempt {call_count} failed")
            return "success"
        
        # Execute with recovery
        result = await recovery_manager.execute_with_recovery(
            operation, "test_workflow"
        )
        
        # Verify
        assert result.success is True
        assert result.attempts_made == 3
        assert call_count == 3
        
        # Verify state can be restored
        restored = state_preserver.restore_state(state_version)
        assert restored["code"] == "initial"
    
    @display_description("验证降级和恢复组合使用")
    @allure.title("Test degradation with recovery")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_degradation_with_recovery(self):
        """Test combining graceful degradation with recovery."""
        degradation = GracefulDegradation()
        recovery_manager = create_recovery_manager(max_retries=2)
        
        # Register degradation chain where first method always fails
        async def failing_method():
            raise ConnectionError("Network error")
        
        async def recovery_method():
            # This will be called through recovery manager
            return "recovered"
        
        degradation.register_degradation_chain(
            "test_op", [failing_method, recovery_method]
        )
        
        # Execute with degradation
        result = await degradation.execute_with_degradation("test_op")
        
        assert result == "recovered"
