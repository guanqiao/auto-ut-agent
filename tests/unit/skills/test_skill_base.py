"""Unit tests for SkillBase and related classes."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from pyutagent.skills.skill_base import (
    SkillBase,
    SkillCategory,
    SkillMeta,
    SkillResult,
    SkillExample,
    skill,
)


class TestSkillExample:
    """Tests for SkillExample dataclass."""

    def test_create_skill_example(self):
        """Test creating a skill example."""
        example = SkillExample(
            task="Build the project",
            description="Compile all source code",
            expected_result="BUILD SUCCESS",
            code_example="mvn compile",
        )
        
        assert example.task == "Build the project"
        assert example.description == "Compile all source code"
        assert example.expected_result == "BUILD SUCCESS"
        assert example.code_example == "mvn compile"

    def test_skill_example_to_dict(self):
        """Test converting skill example to dictionary."""
        example = SkillExample(
            task="Run tests",
            description="Execute all tests",
        )
        
        result = example.to_dict()
        
        assert result["task"] == "Run tests"
        assert result["description"] == "Execute all tests"
        assert result["expected_result"] == ""
        assert result["code_example"] == ""

    def test_skill_example_defaults(self):
        """Test skill example default values."""
        example = SkillExample(
            task="Test",
            description="Description",
        )
        
        assert example.expected_result == ""
        assert example.code_example == ""


class TestSkillMeta:
    """Tests for SkillMeta dataclass."""

    def test_create_skill_meta(self):
        """Test creating skill metadata."""
        meta = SkillMeta(
            name="test_skill",
            description="A test skill",
            category=SkillCategory.TEST,
        )
        
        assert meta.name == "test_skill"
        assert meta.description == "A test skill"
        assert meta.category == SkillCategory.TEST
        assert meta.required_tools == []
        assert meta.version == "1.0.0"

    def test_skill_meta_with_all_fields(self):
        """Test skill metadata with all fields."""
        examples = [SkillExample(task="t1", description="d1")]
        meta = SkillMeta(
            name="full_skill",
            description="Full skill",
            required_tools=["tool1", "tool2"],
            category=SkillCategory.BUILD,
            instructions="Do something",
            examples=examples,
            best_practices=["bp1"],
            common_mistakes=["cm1"],
            prerequisites=["pre1"],
            version="2.0.0",
        )
        
        assert meta.name == "full_skill"
        assert meta.required_tools == ["tool1", "tool2"]
        assert meta.category == SkillCategory.BUILD
        assert meta.instructions == "Do something"
        assert len(meta.examples) == 1
        assert meta.best_practices == ["bp1"]
        assert meta.common_mistakes == ["cm1"]
        assert meta.prerequisites == ["pre1"]
        assert meta.version == "2.0.0"

    def test_skill_meta_to_dict(self):
        """Test converting skill metadata to dictionary."""
        meta = SkillMeta(
            name="dict_skill",
            description="Dictionary skill",
            category=SkillCategory.UTILITY,
            required_tools=["tool1"],
        )
        
        result = meta.to_dict()
        
        assert result["name"] == "dict_skill"
        assert result["description"] == "Dictionary skill"
        assert result["category"] == "utility"
        assert result["required_tools"] == ["tool1"]

    def test_skill_meta_get_prompt_context(self):
        """Test generating prompt context."""
        meta = SkillMeta(
            name="prompt_skill",
            description="Prompt skill description",
            prerequisites=["Java 11+"],
            required_tools=["maven"],
            instructions="Run mvn compile",
            examples=[
                SkillExample(
                    task="Build",
                    description="Build project",
                    code_example="mvn compile",
                )
            ],
            best_practices=["Always clean first"],
            common_mistakes=["Forgetting to compile"],
        )
        
        context = meta.get_prompt_context()
        
        assert "# Skill: prompt_skill" in context
        assert "Prompt skill description" in context
        assert "## Prerequisites" in context
        assert "- Java 11+" in context
        assert "## Required Tools" in context
        assert "- maven" in context
        assert "## Instructions" in context
        assert "Run mvn compile" in context
        assert "## Examples" in context
        assert "## Best Practices" in context
        assert "- Always clean first" in context
        assert "## Common Mistakes to Avoid" in context
        assert "- Forgetting to compile" in context


class TestSkillResult:
    """Tests for SkillResult dataclass."""

    def test_create_success_result(self):
        """Test creating a successful result."""
        result = SkillResult.ok(
            message="Operation successful",
            data={"key": "value"},
            artifacts=["file1.java"],
        )
        
        assert result.success is True
        assert result.message == "Operation successful"
        assert result.data == {"key": "value"}
        assert result.artifacts == ["file1.java"]

    def test_create_failure_result(self):
        """Test creating a failed result."""
        result = SkillResult.fail(
            message="Operation failed",
            data={"error": "ERROR_CODE"},
        )
        
        assert result.success is False
        assert result.message == "Operation failed"
        assert result.data == {"error": "ERROR_CODE"}
        assert result.artifacts == []

    def test_skill_result_defaults(self):
        """Test skill result default values."""
        result = SkillResult(success=True, message="OK")
        
        assert result.data == {}
        assert result.artifacts == []
        assert result.duration_ms == 0

    def test_skill_result_to_dict(self):
        """Test converting skill result to dictionary."""
        result = SkillResult(
            success=True,
            message="Done",
            data={"count": 5},
            artifacts=["test.java"],
            duration_ms=100,
        )
        
        d = result.to_dict()
        
        assert d["success"] is True
        assert d["message"] == "Done"
        assert d["data"] == {"count": 5}
        assert d["artifacts"] == ["test.java"]
        assert d["duration_ms"] == 100


class ConcreteSkill(SkillBase):
    """Concrete skill for testing."""

    name = "concrete_skill"
    description = "A concrete skill for testing"
    category = SkillCategory.UTILITY
    required_tools = ["tool1"]

    def get_instructions(self) -> str:
        return "Test instructions"

    def get_examples(self):
        return [
            SkillExample(task="ex1", description="Example 1"),
        ]

    def get_best_practices(self):
        return ["Best practice 1"]

    def get_common_mistakes(self):
        return ["Mistake 1"]

    def get_prerequisites(self):
        return ["Prerequisite 1"]

    async def execute(self, task, context, tools):
        return SkillResult.ok(message=f"Executed: {task}")


class TestSkillBase:
    """Tests for SkillBase abstract class."""

    def test_create_skill_instance(self):
        """Test creating a skill instance."""
        skill_instance = ConcreteSkill()
        
        assert skill_instance.name == "concrete_skill"
        assert skill_instance.description == "A concrete skill for testing"
        assert skill_instance.category == SkillCategory.UTILITY
        assert skill_instance.required_tools == ["tool1"]

    def test_skill_metadata(self):
        """Test skill metadata generation."""
        skill_instance = ConcreteSkill()
        meta = skill_instance.metadata
        
        assert meta.name == "concrete_skill"
        assert meta.description == "A concrete skill for testing"
        assert meta.category == SkillCategory.UTILITY
        assert meta.required_tools == ["tool1"]
        assert meta.instructions == "Test instructions"
        assert len(meta.examples) == 1
        assert meta.best_practices == ["Best practice 1"]
        assert meta.common_mistakes == ["Mistake 1"]
        assert meta.prerequisites == ["Prerequisite 1"]

    def test_skill_metadata_caching(self):
        """Test that metadata is cached."""
        skill_instance = ConcreteSkill()
        meta1 = skill_instance.metadata
        meta2 = skill_instance.metadata
        
        assert meta1 is meta2

    def test_skill_repr(self):
        """Test skill string representation."""
        skill_instance = ConcreteSkill()
        
        assert repr(skill_instance) == "Skill(concrete_skill, category=utility)"

    def test_skill_get_prompt_context(self):
        """Test getting prompt context from skill."""
        skill_instance = ConcreteSkill()
        context = skill_instance.get_prompt_context()
        
        assert "# Skill: concrete_skill" in context
        assert "Test instructions" in context

    @pytest.mark.asyncio
    async def test_skill_execute(self):
        """Test skill execution."""
        skill_instance = ConcreteSkill()
        result = await skill_instance.execute(
            task="test task",
            context={},
            tools={},
        )
        
        assert result.success is True
        assert result.message == "Executed: test task"

    @pytest.mark.asyncio
    async def test_skill_run_with_timing(self):
        """Test skill run method with timing."""
        skill_instance = ConcreteSkill()
        result = await skill_instance.run(
            task="timed task",
            context={},
            tools={},
        )
        
        assert result.success is True
        assert result.message == "Executed: timed task"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_skill_run_handles_exception(self):
        """Test that run method handles exceptions."""
        class FailingSkill(SkillBase):
            name = "failing"
            description = "Failing skill"
            
            async def execute(self, task, context, tools):
                raise ValueError("Test error")
        
        skill_instance = FailingSkill()
        result = await skill_instance.run(
            task="fail task",
            context={},
            tools={},
        )
        
        assert result.success is False
        assert "Test error" in result.message
        assert result.data["exception_type"] == "ValueError"


class TestSkillDecorator:
    """Tests for @skill decorator."""

    def test_skill_decorator_async(self):
        """Test @skill decorator with async function."""
        @skill("async_skill", "An async skill", SkillCategory.TEST)
        async def async_handler(task, context, tools):
            return SkillResult.ok(message=f"Async: {task}")
        
        assert async_handler.name == "async_skill"
        assert async_handler.description == "An async skill"
        assert async_handler.category == SkillCategory.TEST

    @pytest.mark.asyncio
    async def test_skill_decorator_async_execution(self):
        """Test executing async skill from decorator."""
        @skill("async_exec", "Async execution skill")
        async def async_exec(task, context, tools):
            return SkillResult.ok(message=f"Executed async: {task}")
        
        instance = async_exec()
        result = await instance.run("test", {}, {})
        
        assert result.success is True
        assert result.message == "Executed async: test"

    def test_skill_decorator_sync(self):
        """Test @skill decorator with sync function."""
        @skill("sync_skill", "A sync skill")
        def sync_handler(task, context, tools):
            return SkillResult.ok(message=f"Sync: {task}")
        
        assert sync_handler.name == "sync_skill"
        assert sync_handler.description == "A sync skill"

    @pytest.mark.asyncio
    async def test_skill_decorator_sync_execution(self):
        """Test executing sync skill from decorator."""
        @skill("sync_exec", "Sync execution skill")
        def sync_exec(task, context, tools):
            return SkillResult.ok(message=f"Executed sync: {task}")
        
        instance = sync_exec()
        result = await instance.run("test", {}, {})
        
        assert result.success is True
        assert result.message == "Executed sync: test"

    def test_skill_decorator_with_required_tools(self):
        """Test @skill decorator with required tools."""
        @skill(
            "tool_skill",
            "Skill with tools",
            required_tools=["tool1", "tool2"],
        )
        def tool_handler(task, context, tools):
            return SkillResult.ok()
        
        assert tool_handler.required_tools == ["tool1", "tool2"]


class TestSkillCategory:
    """Tests for SkillCategory enum."""

    def test_skill_categories(self):
        """Test all skill categories exist."""
        categories = [
            SkillCategory.BUILD,
            SkillCategory.TEST,
            SkillCategory.CODE,
            SkillCategory.GIT,
            SkillCategory.SEARCH,
            SkillCategory.ANALYSIS,
            SkillCategory.UTILITY,
        ]
        
        for cat in categories:
            assert isinstance(cat.value, str)

    def test_skill_category_values(self):
        """Test skill category values."""
        assert SkillCategory.BUILD.value == "build"
        assert SkillCategory.TEST.value == "test"
        assert SkillCategory.CODE.value == "code"
        assert SkillCategory.GIT.value == "git"
        assert SkillCategory.SEARCH.value == "search"
        assert SkillCategory.ANALYSIS.value == "analysis"
        assert SkillCategory.UTILITY.value == "utility"
