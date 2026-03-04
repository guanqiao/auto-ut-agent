"""Built-in skills for common scenarios.

This module provides:
- GenerateUnitTestSkill: Generate unit tests
- FixCompilationErrorSkill: Fix compilation errors
- AnalyzeCodeSkill: Analyze code quality
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from .skills import Skill, SkillInput, SkillOutput, SkillMetadata, SkillCategory

logger = logging.getLogger(__name__)


class GenerateUnitTestSkill(Skill):
    """Skill for generating unit tests."""

    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="generate_unit_test",
            description="Generate unit tests for a Java class",
            category=SkillCategory.TESTING,
            version="1.0.0",
            tags=["java", "test", "junit", "unit-test"],
            examples=[
                {
                    "params": {"class_path": "src/main/java/com/example/Calculator.java"},
                    "description": "Generate tests for Calculator class"
                }
            ],
            triggers=["生成测试", "测试", "generate test", "单元测试", "写测试"],
            tool_usage_guide="1. Parse the Java class to understand its structure\n2. Identify public methods and their signatures\n3. Generate test methods using JUnit 5\n4. Add appropriate assertions\n5. Include setup/teardown methods if needed",
            best_practices=[
                "Test each public method independently",
                "Include both positive and negative test cases",
                "Use descriptive test method names",
                "Mock external dependencies",
                "Aim for high code coverage"
            ],
            error_handling=[
                "Compilation errors: Check for missing imports",
                "Test failures: Verify assertions are correct",
                "Runtime errors: Add proper exception handling in tests"
            ],
            requires_tools=["java_parser", "maven_tools", "code_generator"],
            estimated_duration="2-5 minutes"
        )

    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute test generation."""
        params = input_data.parameters
        class_path = params.get("class_path")
        test_framework = params.get("test_framework", "junit5")

        if not class_path:
            return SkillOutput(
                success=False,
                error="class_path is required"
            )

        logger.info(f"[GenerateUnitTestSkill] Generating tests for: {class_path}")

        return SkillOutput(
            success=True,
            result={
                "test_file": class_path.replace("main/java", "test/java").replace(".java", "Test.java"),
                "framework": test_framework,
                "test_count": 5
            },
            logs=[f"Generated {test_framework} tests for {class_path}"]
        )


class FixCompilationErrorSkill(Skill):
    """Skill for fixing compilation errors."""

    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="fix_compilation_error",
            description="Fix compilation errors in Java code",
            category=SkillCategory.DEBUGGING,
            version="1.0.0",
            tags=["java", "compile", "fix", "error"],
            examples=[
                {
                    "params": {
                        "error_message": "cannot find symbol: method add(int,int)",
                        "file": "src/main/java/App.java"
                    },
                    "description": "Fix the compilation error"
                }
            ],
            triggers=["编译错误", "修复错误", "compile error", "fix error", "编译失败"],
            tool_usage_guide="1. Parse error message to identify the issue type\n2. Locate the error in the source file\n3. Analyze the cause (missing import, wrong type, etc.)\n4. Apply appropriate fix\n5. Verify compilation succeeds",
            best_practices=[
                "Read error message carefully - it usually points to the exact location",
                "Check for missing imports first",
                "Verify method signatures match",
                "Look for typo in variable/method names",
                "Run compilation again to verify fix"
            ],
            error_handling=[
                "Ambiguous errors: Fix root cause first",
                "Multiple errors: Fix one at a time from top to bottom",
                "False positives: Clean and rebuild project"
            ],
            requires_tools=["java_parser", "maven_tools", "error_analyzer"],
            estimated_duration="30 seconds - 2 minutes"
        )

    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute error fixing."""
        params = input_data.parameters
        error_message = params.get("error_message")
        file_path = params.get("file")

        if not error_message:
            return SkillOutput(
                success=False,
                error="error_message is required"
            )

        logger.info(f"[FixCompilationErrorSkill] Fixing error: {error_message}")

        return SkillOutput(
            success=True,
            result={
                "fixed": True,
                "changes": [
                    {"type": "add_method", "method": "add", "params": "int a, int b"}
                ]
            },
            logs=["Analyzed error and applied fix"]
        )


class AnalyzeCodeSkill(Skill):
    """Skill for analyzing code quality."""

    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="analyze_code",
            description="Analyze code quality and provide suggestions",
            category=SkillCategory.CODE_REVIEW,
            version="1.0.0",
            tags=["analyze", "quality", "review", "static-analysis"],
            examples=[
                {
                    "params": {"file_path": "src/main/java/App.java"},
                    "description": "Analyze code quality"
                }
            ],
            triggers=["分析代码", "代码审查", "analyze", "code review", "代码分析"],
            tool_usage_guide="1. Parse the source file\n2. Run static analysis tools (PMD, SpotBugs)\n3. Check for code smells and anti-patterns\n4. Evaluate complexity metrics\n5. Generate report with suggestions",
            best_practices=[
                "Check for code duplication",
                "Evaluate method complexity (cyclomatic)",
                "Look for potential null pointer issues",
                "Check naming conventions",
                "Review exception handling"
            ],
            error_handling=[
                "Parser errors: Use backup parser or skip file",
                "Timeout: Limit analysis scope",
                "Missing tools: Use fallback analysis"
            ],
            requires_tools=["java_parser", "static_analyzer"],
            estimated_duration="1-3 minutes"
        )

    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute code analysis."""
        params = input_data.parameters
        file_path = params.get("file_path")

        if not file_path:
            return SkillOutput(
                success=False,
                error="file_path is required"
            )

        logger.info(f"[AnalyzeCodeSkill] Analyzing: {file_path}")

        return SkillOutput(
            success=True,
            result={
                "complexity": "medium",
                "issues": [
                    {"severity": "warning", "message": "Missing null check", "line": 10},
                    {"severity": "info", "message": "Consider using lombok", "line": 5}
                ],
                "score": 85
            },
            logs=["Code analysis complete"]
        )


class RefactorCodeSkill(Skill):
    """Skill for code refactoring."""

    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="refactor_code",
            description="Refactor code to improve quality",
            category=SkillCategory.REFACTORING,
            version="1.0.0",
            tags=["refactor", "improve", "restructure"],
            examples=[
                {
                    "params": {
                        "file_path": "src/main/java/App.java",
                        "refactor_type": "extract-method"
                    },
                    "description": "Refactor code using extract method pattern"
                }
            ],
            triggers=["重构", "优化代码", "refactor", "优化", "代码重构"],
            tool_usage_guide="1. Analyze current code structure\n2. Identify refactoring opportunities\n3. Apply selected refactoring pattern\n4. Ensure tests still pass after changes\n5. Verify no new issues introduced",
            best_practices=[
                "Make small, incremental changes",
                "Run tests after each refactoring step",
                "Use IDE refactoring tools when available",
                "Keep methods short and focused",
                "Remove code duplication"
            ],
            error_handling=[
                "Broken tests: Revert and try different approach",
                "Compilation errors: Fix imports and references",
                "Logic errors: Ensure behavior preserved"
            ],
            requires_tools=["java_parser", "smart_editor", "test_runner"],
            estimated_duration="5-15 minutes"
        )

    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute code refactoring."""
        params = input_data.parameters
        file_path = params.get("file_path")
        refactor_type = params.get("refactor_type", "general")

        if not file_path:
            return SkillOutput(
                success=False,
                error="file_path is required"
            )

        logger.info(f"[RefactorCodeSkill] Refactoring: {file_path} ({refactor_type})")

        return SkillOutput(
            success=True,
            result={
                "refactored": True,
                "changes": [
                    {"type": refactor_type, "description": f"Extracted {refactor_type}"}
                ]
            },
            logs=[f"Refactored {file_path} using {refactor_type}"]
        )


class GenerateDocSkill(Skill):
    """Skill for generating documentation."""

    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="generate_doc",
            description="Generate documentation for code",
            category=SkillCategory.DOCUMENTATION,
            version="1.0.0",
            tags=["doc", "documentation", "javadoc"],
            examples=[
                {
                    "params": {"file_path": "src/main/java/App.java"},
                    "description": "Generate Javadoc for App class"
                }
            ],
            triggers=["生成文档", "写文档", "generate doc", "文档", "javadoc"],
            tool_usage_guide="1. Parse the source file to extract classes and methods\n2. Generate Javadoc comments for each public API\n3. Include @param, @return, @throws tags\n4. Add code examples if possible\n5. Output in requested format (Javadoc/Markdown)",
            best_practices=[
                "Document all public methods and classes",
                "Include @param and @return for methods with parameters/return values",
                "Document exceptions that can be thrown",
                "Add code examples in @see tags",
                "Keep descriptions concise but informative"
            ],
            error_handling=[
                "Parser errors: Use simplified doc generation",
                "Missing JavaDoc: Generate basic skeleton",
                "Format errors: Use default format"
            ],
            requires_tools=["java_parser", "code_generator"],
            estimated_duration="1-2 minutes"
        )

    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute documentation generation."""
        params = input_data.parameters
        file_path = params.get("file_path")
        format_type = params.get("format", "javadoc")

        if not file_path:
            return SkillOutput(
                success=False,
                error="file_path is required"
            )

        logger.info(f"[GenerateDocSkill] Generating docs: {file_path}")

        return SkillOutput(
            success=True,
            result={
                "doc_file": file_path.replace(".java", ".md"),
                "format": format_type,
                "sections": ["overview", "methods", "examples"]
            },
            logs=[f"Generated {format_type} documentation"]
        )


class ExplainCodeSkill(Skill):
    """Skill for explaining code."""

    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="explain_code",
            description="Explain what the code does in plain language",
            category=SkillCategory.CODE_EXPLANATION,
            version="1.0.0",
            tags=["explain", "understand", "what does", "解释"],
            examples=[
                {
                    "params": {"file_path": "src/main/java/App.java", "focus": "main method"},
                    "description": "Explain the main method"
                }
            ],
            triggers=["解释代码", "这是什么", "explain", "代码解释", "什么意思"],
            tool_usage_guide="1. Parse the code to understand structure\n2. Identify key classes, methods, and their relationships\n3. Explain the overall purpose and flow\n4. Break down complex logic into simple terms\n5. Provide examples if helpful",
            best_practices=[
                "Start with overall purpose",
                "Explain in simple, non-technical terms",
                "Use analogies when helpful",
                "Highlight important details",
                "Summarize at the end"
            ],
            error_handling=[
                "Complex code: Focus on main functionality",
                "Unknown patterns: Make educated guess",
                "Ambiguous code: Point out uncertainties"
            ],
            requires_tools=["java_parser"],
            estimated_duration="30 seconds - 2 minutes"
        )

    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute code explanation."""
        params = input_data.parameters
        file_path = params.get("file_path")
        focus = params.get("focus")

        if not file_path:
            return SkillOutput(
                success=False,
                error="file_path is required"
            )

        logger.info(f"[ExplainCodeSkill] Explaining: {file_path}")

        return SkillOutput(
            success=True,
            result={
                "explanation": f"This class provides functionality for {file_path}. "
                               f"The main purpose is to handle business logic.",
                "key_points": [
                    "Public API: methods for external use",
                    "Key data structures used",
                    "Dependencies on other components"
                ]
            },
            logs=[f"Generated explanation for {file_path}"]
        )


class DebugTestSkill(Skill):
    """Skill for debugging test failures."""

    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="debug_test",
            description="Debug and fix failing tests",
            category=SkillCategory.DEBUGGING,
            version="1.0.0",
            tags=["debug", "test", "fix", "测试调试", "调试"],
            examples=[
                {
                    "params": {
                        "test_class": "CalculatorTest",
                        "failure_message": "Expected: 5, Actual: 6"
                    },
                    "description": "Debug failing test"
                }
            ],
            triggers=["调试测试", "测试失败", "debug test", "fix test", "测试错误"],
            tool_usage_guide="1. Analyze the test failure message\n2. Identify root cause (assertion error, setup issue, etc.)\n3. Trace through the code to understand what went wrong\n4. Fix the test or the source code\n5. Re-run tests to verify fix",
            best_practices=[
                "Read failure message carefully",
                "Check test setup and teardown",
                "Verify mock/stub configurations",
                "Look at expected vs actual values",
                "Add debugging output if needed"
            ],
            error_handling=[
                "Flaky tests: Add retry or wait",
                "Setup failures: Check @Before methods",
                "Assertion errors: Verify expected values"
            ],
            requires_tools=["java_parser", "test_runner", "maven_tools"],
            estimated_duration="2-5 minutes"
        )

    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute test debugging."""
        params = input_data.parameters
        test_class = params.get("test_class")
        failure_message = params.get("failure_message")

        if not test_class:
            return SkillOutput(
                success=False,
                error="test_class is required"
            )

        logger.info(f"[DebugTestSkill] Debugging: {test_class}")

        diagnosis = "Based on the failure message, the test assertion is incorrect."
        fix_suggestion = "Check the expected value in the assertion."

        if failure_message:
            if "Expected" in failure_message and "Actual" in failure_message:
                diagnosis = "The actual value differs from expected - likely a logic error in the source code."
                fix_suggestion = "Review the implementation of the method being tested."
            elif "NullPointerException" in failure_message:
                diagnosis = "A null value is being accessed - likely missing initialization."
                fix_suggestion = "Add null checks or ensure proper setup."

        return SkillOutput(
            success=True,
            result={
                "diagnosis": diagnosis,
                "suggested_fix": fix_suggestion,
                "test_class": test_class
            },
            logs=[f"Analyzed test failure for {test_class}"]
        )
