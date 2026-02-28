"""Unit tests for Aider integration module with enhanced display descriptions.

This module provides comprehensive tests for Aider-style code editing functionality,
including compilation error fixing, test failure fixing, and coverage improvement.
"""

import pytest
import allure
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from pyutagent.tools.aider_integration import (
    AiderCodeFixer, AiderTestGenerator, AiderPromptBuilder,
    FixResult, FixStrategy, EditContext,
    fix_compilation_errors_with_aider,
    fix_test_failures_with_aider,
    apply_diff_edit
)
from pyutagent.tools.code_editor import EditResult, EditOperation
from pyutagent.tools.edit_validator import ValidationResult
from pyutagent.tools.error_analyzer import ErrorAnalysis, CompilationError, ErrorType
from pyutagent.tools.failure_analyzer import FailureAnalysis, TestFailure, FailureType


# Custom decorator for display descriptions
def display_description(description: str):
    """Decorator to add display description for test cases.
    
    Args:
        description: Human-readable description of the test case
    """
    def decorator(func):
        func.display_description = description
        # Also add allure annotation
        return allure.description(description)(func)
    return decorator


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = Mock()
    client.agenerate = AsyncMock(return_value="""
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
""")
    return client


@pytest.fixture
def aider_fixer(mock_llm_client):
    """Create an AiderCodeFixer instance with mock LLM."""
    return AiderCodeFixer(llm_client=mock_llm_client, max_attempts=2)


@pytest.fixture
def sample_error_analysis():
    """Create a sample error analysis for testing."""
    return ErrorAnalysis(
        errors=[
            CompilationError(
                error_type=ErrorType.IMPORT_ERROR,
                message="package org.junit.jupiter.api does not exist",
                file_path="Test.java",
                line_number=1,
                error_token="org.junit.jupiter.api",
                fix_hint="Add import statement: import org.junit.jupiter.api.Test;"
            )
        ],
        summary="Found 1 compilation error(s): 1 Import Error",
        fix_strategy="Fix import errors first",
        priority=1
    )


@pytest.fixture
def sample_failure_analysis():
    """Create a sample failure analysis for testing."""
    return FailureAnalysis(
        failures=[
            TestFailure(
                failure_type=FailureType.ASSERTION_FAILURE,
                test_class="MyTest",
                test_method="testAdd",
                message="expected: <3> but was: <4>",
                stack_trace="at MyTest.testAdd(Test.java:10)",
                line_number=10,
                fix_hint="Expected '3' but got '4'. Update assertion or fix code."
            )
        ],
        summary="Test Results: 0 passed, 1 failed, 0 errors, 1 total",
        fix_strategy="Review and fix assertion values",
        priority=3,
        total_tests=1,
        passed_tests=0,
        failed_tests=1,
        error_tests=0,
        skipped_tests=0
    )


@allure.feature("Aider Integration")
@allure.story("Prompt Building")
class TestAiderPromptBuilder:
    """Tests for AiderPromptBuilder class with display descriptions."""
    
    @pytest.fixture
    def builder(self):
        """Create an AiderPromptBuilder instance."""
        return AiderPromptBuilder()
    
    @display_description("验证编译错误修复提示词构建 - 应包含错误分析和代码上下文")
    @allure.title("Test compilation fix prompt building")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_build_compilation_fix_prompt(self, builder, sample_error_analysis):
        """Test building prompt for compilation error fixing."""
        context = EditContext(
            original_code="public class Test { }",
            error_analysis=sample_error_analysis
        )
        
        system_prompt, user_prompt = builder.build_compilation_fix_prompt(context)
        
        assert "Search/Replace" in system_prompt
        assert "Error Analysis" in user_prompt
        assert "Current Test Code" in user_prompt
        assert "public class Test" in user_prompt
        assert sample_error_analysis.summary in user_prompt
    
    @display_description("验证测试失败修复提示词构建 - 应包含失败详情和修复建议")
    @allure.title("Test failure fix prompt building")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_build_test_failure_fix_prompt(self, builder, sample_failure_analysis):
        """Test building prompt for test failure fixing."""
        context = EditContext(
            original_code="public class Test { @Test void test() { } }",
            failure_analysis=sample_failure_analysis
        )
        
        system_prompt, user_prompt = builder.build_test_failure_fix_prompt(context)
        
        assert "Search/Replace" in system_prompt
        assert "Failure Analysis" in user_prompt
        assert sample_failure_analysis.summary in user_prompt
        assert "testAdd" in user_prompt
    
    @display_description("验证覆盖率提升提示词构建 - 应包含未覆盖行号信息")
    @allure.title("Test coverage improvement prompt building")
    @allure.severity(allure.severity_level.NORMAL)
    def test_build_coverage_improvement_prompt(self, builder):
        """Test building prompt for coverage improvement."""
        context = EditContext(
            original_code="public class Test { @Test void test() { } }"
        )
        uncovered_lines = [10, 11, 12]
        
        system_prompt, user_prompt = builder.build_coverage_improvement_prompt(
            context, uncovered_lines
        )
        
        assert "Search/Replace" in system_prompt
        assert "Uncovered Lines" in user_prompt
        assert "10, 11, 12" in user_prompt or "[10, 11, 12]" in user_prompt


@allure.feature("Aider Integration")
@allure.story("Code Fixing")
class TestAiderCodeFixer:
    """Tests for AiderCodeFixer class with display descriptions."""
    
    @display_description("验证编译错误修复 - 应成功修复缺少导入的错误")
    @allure.title("Test fixing compilation errors - import error")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_fix_compilation_errors_success(
        self, aider_fixer, sample_error_analysis, mock_llm_client
    ):
        """Test successfully fixing compilation errors."""
        test_code = """public class MyTest {
    @Test
    void testMethod() {
    }
}"""
        
        mock_llm_client.agenerate.return_value = """
<<<<<<< SEARCH
public class MyTest {
=======
import org.junit.jupiter.api.Test;

public class MyTest {
>>>>>>> REPLACE
"""
        
        result = await aider_fixer.fix_compilation_errors(
            test_code=test_code,
            error_analysis=sample_error_analysis
        )
        
        assert result.success
        assert "import org.junit.jupiter.api.Test" in result.fixed_code
        assert result.strategy_used == FixStrategy.SINGLE_EDIT
        assert result.attempts >= 1
    
    @display_description("验证测试失败修复 - 应成功修复断言错误")
    @allure.title("Test fixing test failures - assertion error")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_fix_test_failures_success(
        self, aider_fixer, sample_failure_analysis, mock_llm_client
    ):
        """Test successfully fixing test failures."""
        test_code = """import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MyTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(4, calc.add(1, 2));
    }
}"""
        
        mock_llm_client.agenerate.return_value = """
<<<<<<< SEARCH
        assertEquals(4, calc.add(1, 2));
=======
        assertEquals(3, calc.add(1, 2));
>>>>>>> REPLACE
"""
        
        result = await aider_fixer.fix_test_failures(
            test_code=test_code,
            failure_analysis=sample_failure_analysis
        )
        
        assert result.success
        assert "assertEquals(3, calc.add(1, 2))" in result.fixed_code
    
    @display_description("验证修复失败时的重试机制 - 应在多次尝试后返回失败")
    @allure.title("Test retry mechanism on fix failure")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_fix_with_retry_failure(self, aider_fixer, sample_error_analysis, mock_llm_client):
        """Test retry mechanism when fix fails."""
        # Return invalid diff that won't apply
        mock_llm_client.agenerate.return_value = """
<<<<<<< SEARCH
nonexistent code
=======
new code
>>>>>>> REPLACE
"""
        
        test_code = "public class Test { }"
        
        result = await aider_fixer.fix_compilation_errors(
            test_code=test_code,
            error_analysis=sample_error_analysis
        )
        
        assert not result.success
        assert result.attempts == aider_fixer.max_attempts
        assert result.error_message != ""
    
    @display_description("验证覆盖率提升功能 - 应为未覆盖行添加测试")
    @allure.title("Test coverage improvement")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_improve_coverage(self, aider_fixer, mock_llm_client):
        """Test improving test coverage."""
        test_code = """import org.junit.jupiter.api.Test;

public class MyTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(3, calc.add(1, 2));
    }
}"""
        
        mock_llm_client.agenerate.return_value = """
<<<<<<< SEARCH
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(3, calc.add(1, 2));
    }
=======
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(3, calc.add(1, 2));
    }
    
    @Test
    void testSubtract() {
        Calculator calc = new Calculator();
        assertEquals(1, calc.subtract(3, 2));
    }
>>>>>>> REPLACE
"""
        
        result = await aider_fixer.improve_coverage(
            test_code=test_code,
            uncovered_lines=[15, 16, 17]
        )
        
        assert result.success
        assert "testSubtract" in result.fixed_code
    
    @display_description("验证直接应用 diff 编辑 - 应无需 LLM 直接应用编辑")
    @allure.title("Test direct diff application")
    @allure.severity(allure.severity_level.NORMAL)
    def test_apply_direct_edit(self, aider_fixer):
        """Test applying direct edit without LLM."""
        test_code = """public class Test {
    public void method() {
        int x = 1;
    }
}"""
        
        diff_text = """
<<<<<<< SEARCH
        int x = 1;
=======
        int x = 2;
>>>>>>> REPLACE
"""
        
        result = aider_fixer.apply_direct_edit(test_code, diff_text)
        
        assert result.success
        assert "int x = 2" in result.fixed_code


@allure.feature("Aider Integration")
@allure.story("Test Generation")
class TestAiderTestGenerator:
    """Tests for AiderTestGenerator class with display descriptions."""
    
    @display_description("验证初始测试代码生成 - 应生成完整的 JUnit 5 测试类")
    @allure.title("Test initial test generation")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_generate_initial_test(self, mock_llm_client):
        """Test generating initial test code."""
        generator = AiderTestGenerator(llm_client=mock_llm_client)
        
        class_info = {
            'name': 'Calculator',
            'package': 'com.example',
            'methods': [
                {
                    'name': 'add',
                    'return_type': 'int',
                    'parameters': [('int', 'a'), ('int', 'b')]
                }
            ]
        }
        
        mock_llm_client.agenerate.return_value = """
```java
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(3, calc.add(1, 2));
    }
}
```
"""
        
        result = await generator.generate_initial_test(class_info)
        
        assert "CalculatorTest" in result
        assert "@Test" in result
        assert "testAdd" in result
    
    @display_description("验证带反馈的迭代生成 - 应在多次迭代后成功")
    @allure.title("Test iterative generation with feedback")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_iterate_with_feedback_success(self, mock_llm_client):
        """Test iterative generation with feedback."""
        generator = AiderTestGenerator(llm_client=mock_llm_client)
        
        test_code = "public class Test { }"
        
        # Mock feedback callback - succeeds on second attempt
        call_count = 0
        def feedback_callback(code):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False, ErrorAnalysis(
                    errors=[CompilationError(
                        error_type=ErrorType.IMPORT_ERROR,
                        message="missing import",
                        fix_hint="Add import"
                    )],
                    summary="1 error",
                    fix_strategy="Fix imports",
                    priority=1
                )
            return True, None
        
        mock_llm_client.agenerate.return_value = """
<<<<<<< SEARCH
public class Test { }
=======
import org.junit.jupiter.api.Test;

public class Test { }
>>>>>>> REPLACE
"""
        
        result = await generator.iterate_with_feedback(
            test_code=test_code,
            feedback_callback=feedback_callback,
            max_iterations=3
        )
        
        assert result['success']
        assert result['iterations'] == 2


@allure.feature("Aider Integration")
@allure.story("Utility Functions")
class TestUtilityFunctions:
    """Tests for utility functions with display descriptions."""
    
    @display_description("验证 fix_compilation_errors_with_aider 便捷函数")
    @allure.title("Test fix_compilation_errors_with_aider utility")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_fix_compilation_errors_with_aider_utility(self, mock_llm_client):
        """Test fix_compilation_errors_with_aider utility function."""
        mock_llm_client.agenerate.return_value = """
<<<<<<< SEARCH
public class Test { }
=======
import org.junit.jupiter.api.Test;

public class Test { }
>>>>>>> REPLACE
"""
        
        error_analysis = ErrorAnalysis(
            errors=[],
            summary="No errors",
            fix_strategy="None",
            priority=5
        )
        
        result = await fix_compilation_errors_with_aider(
            test_code="public class Test { }",
            error_analysis=error_analysis,
            llm_client=mock_llm_client
        )
        
        assert isinstance(result, FixResult)
    
    @display_description("验证 apply_diff_edit 便捷函数")
    @allure.title("Test apply_diff_edit utility")
    @allure.severity(allure.severity_level.NORMAL)
    def test_apply_diff_edit_utility(self):
        """Test apply_diff_edit utility function."""
        test_code = "public class Test { int x = 1; }"
        diff_text = """
<<<<<<< SEARCH
int x = 1;
=======
int x = 2;
>>>>>>> REPLACE
"""
        
        result = apply_diff_edit(test_code, diff_text)
        
        assert isinstance(result, FixResult)
        assert "int x = 2" in result.fixed_code


@allure.feature("Aider Integration")
@allure.story("Error Handling")
class TestErrorHandling:
    """Tests for error handling with display descriptions."""
    
    @display_description("验证 LLM 调用异常时的错误处理")
    @allure.title("Test LLM exception handling")
    @allure.severity(allure.severity_level.CRITICAL)
    @pytest.mark.asyncio
    async def test_llm_exception_handling(self, mock_llm_client):
        """Test handling of LLM exceptions."""
        mock_llm_client.agenerate.side_effect = Exception("LLM API error")
        
        fixer = AiderCodeFixer(llm_client=mock_llm_client, max_attempts=1)
        
        error_analysis = ErrorAnalysis(
            errors=[CompilationError(
                error_type=ErrorType.SYNTAX_ERROR,
                message="syntax error",
                fix_hint="Fix syntax"
            )],
            summary="1 error",
            fix_strategy="Fix syntax",
            priority=1
        )
        
        result = await fixer.fix_compilation_errors(
            test_code="public class Test { }",
            error_analysis=error_analysis
        )
        
        assert not result.success
        assert "LLM API error" in result.error_message or result.attempts == 1
    
    @display_description("验证无效 diff 格式的错误处理")
    @allure.title("Test invalid diff format handling")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.asyncio
    async def test_invalid_diff_format(self, aider_fixer, mock_llm_client):
        """Test handling of invalid diff format."""
        mock_llm_client.agenerate.return_value = "Invalid response without diff format"
        
        error_analysis = ErrorAnalysis(
            errors=[],
            summary="No errors",
            fix_strategy="None",
            priority=5
        )
        
        result = await aider_fixer.fix_compilation_errors(
            test_code="public class Test { }",
            error_analysis=error_analysis
        )
        
        assert not result.success


@allure.feature("Aider Integration")
@allure.story("Edge Cases")
class TestEdgeCases:
    """Tests for edge cases with display descriptions."""
    
    @display_description("验证空代码的处理")
    @allure.title("Test empty code handling")
    @allure.severity(allure.severity_level.MINOR)
    def test_empty_code_handling(self):
        """Test handling of empty code."""
        result = apply_diff_edit("", "")
        
        assert not result.success
    
    @display_description("验证大文件编辑的性能")
    @allure.title("Test large file editing performance")
    @allure.severity(allure.severity_level.NORMAL)
    def test_large_file_editing(self, aider_fixer):
        """Test editing large files."""
        # Generate large test code
        large_code = "public class Test {\n"
        for i in range(100):
            large_code += f"    public void method{i}() {{ }}\n"
        large_code += "}"
        
        diff_text = """
<<<<<<< SEARCH
    public void method0() { }
=======
    public void method0() { int x = 1; }
>>>>>>> REPLACE
"""
        
        result = aider_fixer.apply_direct_edit(large_code, diff_text)
        
        assert result.success
        assert "int x = 1" in result.fixed_code
    
    @display_description("验证多文件 diff 的处理")
    @allure.title("Test multi-file diff handling")
    @allure.severity(allure.severity_level.NORMAL)
    def test_multi_file_diff(self, aider_fixer):
        """Test handling diffs for multiple files."""
        test_code = "public class Test { }"
        
        diff_text = """
### Test.java
<<<<<<< SEARCH
public class Test { }
=======
import org.junit.jupiter.api.Test;

public class Test { }
>>>>>>> REPLACE
"""
        
        result = aider_fixer.apply_direct_edit(test_code, diff_text)
        
        assert result.success
        assert "import org.junit.jupiter.api.Test" in result.fixed_code


# Integration tests
@allure.feature("Aider Integration")
@allure.story("Integration Tests")
class TestIntegration:
    """Integration tests with display descriptions."""
    
    @display_description("验证完整的编译错误修复流程")
    @allure.title("Test complete compilation fix workflow")
    @allure.severity(allure.severity_level.BLOCKER)
    @pytest.mark.asyncio
    async def test_complete_compilation_fix_workflow(self, mock_llm_client):
        """Test complete workflow for fixing compilation errors."""
        # Original code with missing imports
        original_code = """public class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(3, calc.add(1, 2));
    }
}"""
        
        # Expected fixed code
        fixed_code = """import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class CalculatorTest {
    @Test
    void testAdd() {
        Calculator calc = new Calculator();
        assertEquals(3, calc.add(1, 2));
    }
}"""
        
        mock_llm_client.agenerate.return_value = f"""
<<<<<<< SEARCH
public class CalculatorTest {{
=======
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class CalculatorTest {{
>>>>>>> REPLACE
"""
        
        error_analysis = ErrorAnalysis(
            errors=[
                CompilationError(
                    error_type=ErrorType.IMPORT_ERROR,
                    message="cannot find symbol: class Test",
                    fix_hint="Add import"
                )
            ],
            summary="1 import error",
            fix_strategy="Fix imports",
            priority=1
        )
        
        result = await fix_compilation_errors_with_aider(
            test_code=original_code,
            error_analysis=error_analysis,
            llm_client=mock_llm_client
        )
        
        assert result.success
        assert "import org.junit.jupiter.api.Test" in result.fixed_code
        assert "import static org.junit.jupiter.api.Assertions.assertEquals" in result.fixed_code
