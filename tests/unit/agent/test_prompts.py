"""Tests for agent prompts module."""

import pytest
from unittest.mock import Mock

from pyutagent.agent.prompts import PromptBuilder


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a PromptBuilder instance."""
        return PromptBuilder()

    def test_initialization(self, builder):
        """Test PromptBuilder initialization."""
        assert builder.system_prompt is not None
        assert "JUnit 5" in builder.system_prompt
        assert "@DisplayName" in builder.system_prompt

    def test_build_system_prompt(self, builder):
        """Test building system prompt."""
        prompt = builder._build_system_prompt()
        assert "JUnit 5" in prompt
        assert "Mockito" in prompt
        assert "@Test" in prompt
        assert "@DisplayName" in prompt

    def test_build_initial_test_prompt(self, builder):
        """Test building initial test prompt."""
        class_info = {
            "name": "Calculator",
            "package": "com.example",
            "methods": [
                {"name": "add", "parameters": [{"name": "a"}, {"name": "b"}], "return_type": "int"},
                {"name": "subtract", "parameters": [{"name": "a"}, {"name": "b"}], "return_type": "int"}
            ],
            "fields": [{"name": "value", "type": "int"}],
            "imports": ["java.util.List"]
        }
        source_code = "public class Calculator { public int add(int a, int b) { return a + b; } }"

        prompt = builder.build_initial_test_prompt(class_info, source_code)

        assert "Calculator" in prompt
        assert "com.example" in prompt
        assert "add" in prompt
        assert "subtract" in prompt
        assert "value" in prompt
        assert "CalculatorTest" in prompt
        assert source_code in prompt

    def test_build_initial_test_prompt_empty_methods(self, builder):
        """Test building prompt with empty methods list."""
        class_info = {
            "name": "EmptyClass",
            "package": "",
            "methods": [],
            "fields": [],
            "imports": []
        }
        source_code = "public class EmptyClass {}"

        prompt = builder.build_initial_test_prompt(class_info, source_code)

        assert "EmptyClass" in prompt
        assert "EmptyClassTest" in prompt

    def test_build_fix_compilation_prompt(self, builder):
        """Test building fix compilation prompt."""
        test_code = "public class Test { invalid syntax }"
        compilation_errors = "Test.java:1: error: cannot find symbol"

        prompt = builder.build_fix_compilation_prompt(test_code, compilation_errors)

        assert test_code in prompt
        assert compilation_errors in prompt
        assert "Fix the compilation errors" in prompt

    def test_build_fix_test_failure_prompt(self, builder):
        """Test building fix test failure prompt."""
        test_code = "@Test void test1() { assertEquals(1, 2); }"
        failures = [
            {"test_name": "test1", "error": "expected:<1> but was:<2>"},
            {"test_name": "test2", "error": "NullPointerException"}
        ]

        prompt = builder.build_fix_test_failure_prompt(test_code, failures)

        assert test_code in prompt
        assert "test1" in prompt
        assert "test2" in prompt
        assert "expected:<1> but was:<2>" in prompt

    def test_build_additional_tests_prompt(self, builder):
        """Test building additional tests prompt."""
        class_info = {"name": "Calculator"}
        existing_tests = "@Test void testAdd() { }"
        uncovered_info = {
            "lines": [10, 20, 30, 40, 50],
            "methods": ["divide", "multiply"]
        }
        current_coverage = 0.5

        prompt = builder.build_additional_tests_prompt(
            class_info, existing_tests, uncovered_info, current_coverage
        )

        assert "50.0%" in prompt or "50%" in prompt
        assert "80%" in prompt
        assert existing_tests in prompt
        assert "divide" in prompt
        assert "multiply" in prompt
        assert "additional test methods" in prompt.lower()

    def test_build_additional_tests_prompt_many_uncovered_lines(self, builder):
        """Test building prompt with many uncovered lines."""
        class_info = {"name": "Calculator"}
        existing_tests = "@Test void testAdd() { }"
        uncovered_info = {
            "lines": list(range(1, 100)),  # 99 lines
            "methods": ["method1", "method2", "method3"]
        }
        current_coverage = 0.3

        prompt = builder.build_additional_tests_prompt(
            class_info, existing_tests, uncovered_info, current_coverage
        )

        assert "and 79 more" in prompt or "79 more" in prompt

    def test_build_coverage_analysis_prompt(self, builder):
        """Test building coverage analysis prompt."""
        class_info = {"name": "Calculator"}
        coverage_report = {
            "line_coverage": 0.75,
            "branch_coverage": 0.60,
            "method_coverage": 0.80
        }

        prompt = builder.build_coverage_analysis_prompt(class_info, coverage_report)

        assert "Calculator" in prompt
        assert "75.0%" in prompt or "75%" in prompt
        assert "60.0%" in prompt or "60%" in prompt
        assert "80.0%" in prompt or "80%" in prompt

    def test_format_uncovered_areas(self, builder):
        """Test formatting uncovered areas."""
        coverage_report = {"lines": [1, 2, 3]}
        result = builder._format_uncovered_areas(coverage_report)
        assert "detailed coverage report" in result

    def test_build_method_test_prompt(self, builder):
        """Test building method-specific test prompt."""
        class_info = {"name": "Calculator"}
        method_info = {
            "name": "add",
            "parameters": [
                {"type": "int", "name": "a"},
                {"type": "int", "name": "b"}
            ],
            "return_type": "int"
        }
        source_code = "public int add(int a, int b) { return a + b; }"

        prompt = builder.build_method_test_prompt(class_info, method_info, source_code)

        assert "Calculator" in prompt
        assert "add" in prompt
        assert "int a" in prompt
        assert "int b" in prompt
        assert source_code in prompt
        assert "Edge cases" in prompt

    def test_build_method_test_prompt_no_parameters(self, builder):
        """Test building prompt for method without parameters."""
        class_info = {"name": "Calculator"}
        method_info = {
            "name": "clear",
            "parameters": [],
            "return_type": "void"
        }
        source_code = "public void clear() { }"

        prompt = builder.build_method_test_prompt(class_info, method_info, source_code)

        assert "clear()" in prompt
        assert "void" in prompt

    def test_build_incremental_test_prompt(self, builder):
        """Test building incremental test prompt."""
        class_info = {"name": "Calculator"}
        existing_tests = "@Test void testAdd() { }"
        changed_methods = ["add", "subtract", "multiply"]

        prompt = builder.build_incremental_test_prompt(
            class_info, existing_tests, changed_methods
        )

        # Note: Calculator class name is not included in this prompt template
        assert existing_tests in prompt
        assert "add" in prompt
        assert "subtract" in prompt
        assert "multiply" in prompt
        assert "Changed Methods" in prompt

    def test_build_error_analysis_prompt(self, builder):
        """Test building error analysis prompt."""
        error_category = "COMPILATION_ERROR"
        error_message = "Cannot find symbol"
        error_details = {"line": 10, "symbol": "MyClass"}
        local_analysis = {
            "local_insights": {"missing_import": True},
            "suggested_fixes": [{"type": "import", "hint": "Add import"}]
        }
        attempt_history = [
            {"attempt": 1, "strategy": "RETRY", "success": False},
            {"attempt": 2, "strategy": "FIX", "success": False}
        ]
        current_test_code = "public class Test { }"
        target_class_info = {"name": "TargetClass"}

        prompt = builder.build_error_analysis_prompt(
            error_category, error_message, error_details,
            local_analysis, attempt_history, current_test_code, target_class_info
        )

        assert error_category in prompt
        assert error_message in prompt
        assert "TargetClass" in prompt
        assert "Attempt 1" in prompt
        assert "Attempt 2" in prompt
        assert "RETRY" in prompt
        assert "Strategy" in prompt
        assert "Confidence" in prompt

    def test_build_error_analysis_prompt_empty_history(self, builder):
        """Test building error analysis prompt with empty history."""
        prompt = builder.build_error_analysis_prompt(
            error_category="ERROR",
            error_message="Something failed",
            error_details={},
            local_analysis={"local_insights": {}, "suggested_fixes": []},
            attempt_history=[],
            current_test_code=None,
            target_class_info=None
        )

        assert "No previous attempts" in prompt
        assert "Unknown" in prompt

    def test_build_error_analysis_prompt_long_history(self, builder):
        """Test that only last 5 attempts are shown."""
        attempt_history = [
            {"attempt": i, "strategy": f"STRATEGY_{i}", "success": False}
            for i in range(1, 10)  # 9 attempts
        ]

        prompt = builder.build_error_analysis_prompt(
            error_category="ERROR",
            error_message="Failed",
            error_details={},
            local_analysis={"local_insights": {}, "suggested_fixes": []},
            attempt_history=attempt_history,
            current_test_code=None,
            target_class_info=None
        )

        # Should only show attempts 5-9
        assert "Attempt 5" in prompt
        assert "Attempt 9" in prompt
        assert "Attempt 1" not in prompt

    def test_build_comprehensive_fix_prompt(self, builder):
        """Test building comprehensive fix prompt."""
        error_category = "COMPILATION_ERROR"
        error_message = "Cannot find symbol"
        error_details = {"line": 10}
        local_analysis = {"insight": "missing import"}
        llm_insights = "The class needs to be imported"
        specific_fixes = ["Add import statement", "Fix class name"]
        current_test_code = "public class Test { MyClass obj; }"
        target_class_info = {
            "name": "TargetClass",
            "package": "com.example",
            "methods": [{"name": "method1"}, {"name": "method2"}]
        }
        attempt_history = [{"attempt": 1, "success": False, "message": "Failed"}]

        prompt = builder.build_comprehensive_fix_prompt(
            error_category, error_message, error_details,
            local_analysis, llm_insights, specific_fixes,
            current_test_code, target_class_info, attempt_history
        )

        assert error_category in prompt
        assert error_message in prompt
        assert "missing import" in prompt
        assert "Add import statement" in prompt
        assert current_test_code in prompt
        assert "TargetClass" in prompt
        assert "method1" in prompt
        assert "COMPLETE fixed test code" in prompt

    def test_build_comprehensive_fix_prompt_no_fixes(self, builder):
        """Test building fix prompt with no specific fixes."""
        prompt = builder.build_comprehensive_fix_prompt(
            error_category="ERROR",
            error_message="Failed",
            error_details={},
            local_analysis={},
            llm_insights="",
            specific_fixes=[],
            current_test_code="",
            target_class_info=None,
            attempt_history=[]
        )

        assert "None" in prompt or "No previous attempts" in prompt
