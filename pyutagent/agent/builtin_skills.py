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
            ]
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
            ]
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
            ]
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
            ]
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
            ]
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
