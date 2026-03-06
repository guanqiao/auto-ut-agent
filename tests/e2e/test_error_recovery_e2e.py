"""E2E tests for error scenarios and recovery.

This module tests error handling and recovery mechanisms:
- Compilation error recovery
- Test failure recovery
- LLM error recovery
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any

import pytest

from pyutagent.core.error_recovery import ErrorRecoveryManager, ErrorClassifier
from pyutagent.core.error_types import ErrorCategory, RecoveryStrategy
from pyutagent.core.exceptions import JavaCompilationError, TestFailureError, LLMGenerationError


class TestCompilationErrorRecoveryE2E:
    """E2E tests for compilation error recovery."""
    
    @pytest.mark.asyncio
    async def test_missing_import_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from missing import errors."""
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.memory.working_memory import WorkingMemory
        
        working_memory = WorkingMemory(
            target_coverage=0.8,
            max_iterations=3,
            current_file="Calculator.java"
        )
        
        def generate_with_missing_import(*args, **kwargs):
            return '''
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        List<Integer> list = new ArrayList<>();
        assertEquals(5, calc.add(2, 3));
    }
}
'''
        
        mock_llm_client.agenerate = AsyncMock(side_effect=generate_with_missing_import)
        
        with patch('pyutagent.agent.react_agent.LLMClient') as mock_llm_class:
            mock_llm_class.from_config.return_value = mock_llm_client
            
            agent = ReActAgent(
                project_path=str(temp_maven_project),
                llm_client=mock_llm_client,
                working_memory=working_memory
            )
            
            recovery_manager = ErrorRecoveryManager()
            
            error = JavaCompilationError("cannot find symbol: class List")
            context = {
                "step": "compilation",
                "test_code": generate_with_missing_import(),
                "error_message": str(error)
            }
            
            result = await recovery_manager.recover(
                error=error,
                error_context=context
            )
            
            assert result is not None
            assert "action" in result
    
    @pytest.mark.asyncio
    async def test_missing_dependency_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from missing dependency errors."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = JavaCompilationError("package com.fasterxml.jackson.databind does not exist")
        context = {
            "step": "compilation",
            "test_code": "import com.fasterxml.jackson.databind.ObjectMapper;",
            "error_message": str(error),
            "project_path": str(temp_maven_project)
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result
    
    @pytest.mark.asyncio
    async def test_syntax_error_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from syntax errors."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = JavaCompilationError("';' expected")
        context = {
            "step": "compilation",
            "test_code": "public class Test { public void test() { int x = 5 } }",
            "error_message": str(error)
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result
    
    @pytest.mark.asyncio
    async def test_type_mismatch_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from type mismatch errors."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = JavaCompilationError("incompatible types: String cannot be converted to int")
        context = {
            "step": "compilation",
            "test_code": "int x = \"hello\";",
            "error_message": str(error)
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result


class TestTestFailureRecoveryE2E:
    """E2E tests for test failure recovery."""
    
    @pytest.mark.asyncio
    async def test_assertion_failure_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from assertion failures."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = TestFailureError("Assertion failed: expected:<5> but was:<4>")
        context = {
            "step": "test_execution",
            "test_code": "assertEquals(5, calc.add(2, 2));",
            "error_message": str(error),
            "test_method": "testAdd"
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result
    
    @pytest.mark.asyncio
    async def test_null_pointer_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from null pointer exceptions."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = TestFailureError("NullPointerException at CalculatorTest.testDivide")
        context = {
            "step": "test_execution",
            "test_code": "Calculator calc = null; calc.divide(1, 1);",
            "error_message": str(error),
            "test_method": "testDivide"
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result
    
    @pytest.mark.asyncio
    async def test_timeout_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from test timeouts."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = TestFailureError("test timed out after 10 seconds")
        context = {
            "step": "test_execution",
            "test_code": "while(true) { Thread.sleep(1000); }",
            "error_message": str(error),
            "test_method": "testInfiniteLoop"
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result


class TestLLMErrorRecoveryE2E:
    """E2E tests for LLM error recovery."""
    
    @pytest.mark.asyncio
    async def test_api_timeout_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from LLM API timeouts."""
        from pyutagent.core.retry_manager import RetryManager
        from pyutagent.core.retry_config import RetryConfig, RetryStrategy
        
        config = RetryConfig(
            max_total_attempts=3,
            backoff_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        retry_manager = RetryManager(config)
        
        call_count = 0
        
        async def operation_with_timeout():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("API request timed out")
            return "Success"
        
        result = await retry_manager.execute_async(
            operation_with_timeout,
            "llm_api_call"
        )
        
        assert result == "Success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from rate limit errors."""
        from pyutagent.core.retry_manager import RetryManager
        from pyutagent.core.retry_config import RetryConfig, RetryStrategy
        
        config = RetryConfig(
            max_total_attempts=3,
            backoff_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            backoff_base=1.0
        )
        
        retry_manager = RetryManager(config)
        
        call_count = 0
        
        async def operation_with_rate_limit():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Rate limit exceeded")
            return "Success"
        
        result = await retry_manager.execute_async(
            operation_with_rate_limit,
            "llm_api_call"
        )
        
        assert result == "Success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_invalid_response_recovery(self, temp_maven_project, mock_llm_client):
        """Test recovering from invalid LLM responses."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = LLMGenerationError("Invalid JSON response from LLM")
        context = {
            "step": "llm_generation",
            "response": "This is not valid JSON",
            "error_message": str(error)
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result


class TestErrorClassifierE2E:
    """E2E tests for error classification."""
    
    def test_classify_network_error(self):
        """Test classifying network errors."""
        classifier = ErrorClassifier()
        
        error = ConnectionError("Failed to connect to API endpoint")
        category = classifier.classify(error, {})
        
        assert category in [ErrorCategory.NETWORK, ErrorCategory.TRANSIENT]
    
    def test_classify_timeout_error(self):
        """Test classifying timeout errors."""
        classifier = ErrorClassifier()
        
        error = TimeoutError("Operation timed out")
        category = classifier.classify(error, {})
        
        assert category == ErrorCategory.TIMEOUT or category == ErrorCategory.TRANSIENT
    
    def test_classify_compilation_error(self):
        """Test classifying compilation errors."""
        classifier = ErrorClassifier()
        
        error = JavaCompilationError("cannot find symbol")
        category = classifier.classify(error, {"step": "compilation"})
        
        assert category == ErrorCategory.COMPILATION_ERROR
    
    def test_classify_test_failure(self):
        """Test classifying test failures."""
        classifier = ErrorClassifier()
        
        error = TestFailureError("Assertion failed")
        category = classifier.classify(error, {"step": "test_execution"})
        
        assert category == ErrorCategory.TEST_FAILURE


class TestRecoveryStrategyE2E:
    """E2E tests for recovery strategy selection."""
    
    @pytest.mark.asyncio
    async def test_retry_strategy_for_transient_errors(self):
        """Test retry strategy for transient errors."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = ConnectionError("Network unreachable")
        context = {
            "step": "llm_api_call",
            "attempt": 1
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert result.get("action") in ["retry", "retry_with_backoff", "retry_immediate"]
    
    @pytest.mark.asyncio
    async def test_analyze_and_fix_strategy_for_code_errors(self):
        """Test analyze-and-fix strategy for code errors."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        recovery_manager = ErrorRecoveryManager()
        
        error = CompilationError("cannot find symbol: class Calculator")
        context = {
            "step": "compilation",
            "test_code": "Calculator calc = new Calculator();",
            "attempt": 1
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert result.get("action") in ["analyze_and_fix", "regenerate", "retry"]


class TestErrorRecoveryIntegrationE2E:
    """E2E tests for complete error recovery integration."""
    
    @pytest.mark.asyncio
    async def test_full_recovery_workflow(self, temp_maven_project, mock_llm_client):
        """Test complete error recovery workflow."""
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.core.error_recovery import ErrorRecoveryManager
        
        working_memory = WorkingMemory(
            target_coverage=0.8,
            max_iterations=5,
            current_file="Calculator.java"
        )
        
        recovery_manager = ErrorRecoveryManager()
        
        call_count = 0
        
        def generate_with_errors(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return '''
package com.example;

import org.junit.jupiter.api.Test;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        UndefinedClass obj = new UndefinedClass();
    }
}
'''
            else:
                return '''
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(5, calc.add(2, 3));
    }
}
'''
        
        mock_llm_client.agenerate = AsyncMock(side_effect=generate_with_errors)
        
        error = CompilationError("cannot find symbol: class UndefinedClass")
        context = {
            "step": "compilation",
            "test_code": generate_with_errors(),
            "error_message": str(error),
            "attempt": 1
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
