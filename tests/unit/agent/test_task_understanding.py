"""Tests for task understanding module."""

import pytest
from pathlib import Path

from pyutagent.agent.task_understanding import (
    TaskType,
    TaskPriority,
    TaskComplexity,
    TargetScope,
    Constraint,
    SuccessCriterion,
    TaskUnderstanding,
    TaskClassifier,
)


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types_exist(self):
        assert TaskType.UT_GENERATION.value == "ut_generation"
        assert TaskType.CODE_REFACTORING.value == "code_refactoring"
        assert TaskType.BUG_FIX.value == "bug_fix"
        assert TaskType.FEATURE_ADD.value == "feature_add"
        assert TaskType.CODE_REVIEW.value == "code_review"
        assert TaskType.UNKNOWN.value == "unknown"

    def test_all_task_types_have_value(self):
        for task_type in TaskType:
            assert isinstance(task_type.value, str)
            assert len(task_type.value) > 0


class TestTargetScope:
    """Tests for TargetScope."""

    def test_empty_scope(self):
        scope = TargetScope()
        assert scope.is_empty()
        assert scope.files == []
        assert scope.directories == []

    def test_scope_with_files(self):
        scope = TargetScope(files=["src/main/java/Example.java"])
        assert not scope.is_empty()
        assert len(scope.files) == 1

    def test_to_dict(self):
        scope = TargetScope(
            files=["file1.java"],
            directories=["src/"],
            packages=["com.example"],
            classes=["Example"],
            methods=["method1"],
        )
        result = scope.to_dict()
        
        assert result["files"] == ["file1.java"]
        assert result["directories"] == ["src/"]
        assert result["packages"] == ["com.example"]
        assert result["classes"] == ["Example"]
        assert result["methods"] == ["method1"]


class TestConstraint:
    """Tests for Constraint."""

    def test_constraint_creation(self):
        constraint = Constraint(
            name="max_lines",
            value=100,
            description="Maximum lines of code"
        )
        assert constraint.name == "max_lines"
        assert constraint.value == 100

    def test_to_dict(self):
        constraint = Constraint(name="timeout", value=30)
        result = constraint.to_dict()
        assert result["name"] == "timeout"
        assert result["value"] == 30


class TestSuccessCriterion:
    """Tests for SuccessCriterion."""

    def test_measurable_criterion(self):
        criterion = SuccessCriterion(
            description="Coverage above 80%",
            measurable=True,
            threshold=0.8
        )
        assert criterion.measurable
        assert criterion.threshold == 0.8

    def test_non_measurable_criterion(self):
        criterion = SuccessCriterion(
            description="Code follows style guide",
            measurable=False
        )
        assert not criterion.measurable


class TestTaskUnderstanding:
    """Tests for TaskUnderstanding."""

    def test_basic_understanding(self):
        understanding = TaskUnderstanding(
            task_type=TaskType.UT_GENERATION,
            original_request="Generate tests for Example.java",
            requirements="Generate unit tests",
        )
        
        assert understanding.task_type == TaskType.UT_GENERATION
        assert understanding.priority == TaskPriority.MEDIUM
        assert understanding.complexity == TaskComplexity.MODERATE

    def test_to_dict(self):
        understanding = TaskUnderstanding(
            task_type=TaskType.BUG_FIX,
            original_request="Fix the bug",
            requirements="Fix null pointer exception",
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.SIMPLE,
            confidence=0.9,
        )
        
        result = understanding.to_dict()
        
        assert result["task_type"] == "bug_fix"
        assert result["priority"] == "high"
        assert result["complexity"] == "simple"
        assert result["confidence"] == 0.9


class TestTaskClassifier:
    """Tests for TaskClassifier."""

    @pytest.fixture
    def classifier(self):
        return TaskClassifier()

    def test_classify_ut_generation(self, classifier):
        result = classifier.classify_by_keywords(
            "Generate unit tests for UserService.java"
        )
        assert result == TaskType.UT_GENERATION

    def test_classify_bug_fix(self, classifier):
        result = classifier.classify_by_keywords(
            "Fix the null pointer exception in the login method"
        )
        assert result == TaskType.BUG_FIX

    def test_classify_refactoring(self, classifier):
        result = classifier.classify_by_keywords(
            "Refactor the UserService class to follow SOLID principles"
        )
        assert result == TaskType.CODE_REFACTORING

    def test_classify_feature_add(self, classifier):
        result = classifier.classify_by_keywords(
            "Extend the system with new functionality for user authentication"
        )
        assert result == TaskType.FEATURE_ADD

    def test_classify_code_review(self, classifier):
        result = classifier.classify_by_keywords(
            "Review the code for best practices"
        )
        assert result == TaskType.CODE_REVIEW

    def test_classify_unknown(self, classifier):
        result = classifier.classify_by_keywords(
            "Hello world"
        )
        assert result == TaskType.UNKNOWN

    def test_determine_priority_high(self, classifier):
        result = classifier.determine_priority(
            "This is urgent, fix it ASAP"
        )
        assert result == TaskPriority.CRITICAL

    def test_determine_priority_normal(self, classifier):
        result = classifier.determine_priority(
            "Generate tests for this file"
        )
        assert result == TaskPriority.MEDIUM

    def test_determine_complexity_simple(self, classifier):
        result = classifier.determine_complexity(
            "Simple fix in a single file"
        )
        assert result == TaskComplexity.SIMPLE

    def test_determine_complexity_complex(self, classifier):
        result = classifier.determine_complexity(
            "Complex refactoring across multiple files"
        )
        assert result == TaskComplexity.COMPLEX

    def test_extract_target_files_java(self, classifier):
        result = classifier.extract_target_files(
            "Generate tests for src/main/java/UserService.java"
        )
        assert "src/main/java/UserService.java" in result

    def test_extract_target_files_quoted(self, classifier):
        result = classifier.extract_target_files(
            'Fix the bug in "UserService.java"'
        )
        assert "UserService.java" in result

    def test_create_basic_understanding(self, classifier):
        understanding = classifier.create_basic_understanding(
            "Generate unit tests for UserService.java"
        )
        
        assert understanding.task_type == TaskType.UT_GENERATION
        assert "UserService.java" in understanding.original_request
        assert understanding.confidence > 0

    def test_chinese_keywords(self, classifier):
        result = classifier.classify_by_keywords(
            "为UserService生成单元测试"
        )
        assert result == TaskType.UT_GENERATION

        result = classifier.classify_by_keywords(
            "修复登录功能的bug"
        )
        assert result == TaskType.BUG_FIX
