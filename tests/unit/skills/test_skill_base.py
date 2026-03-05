"""Unit tests for skill_base module."""

import pytest
from pathlib import Path
from typing import Dict, Any

from pyutagent.skills.skill_base import (
    SkillBase,
    SkillCategory,
    SkillLevel,
    SkillMetadata,
    SkillContext,
    SkillInput,
    SkillOutput,
    SkillParameter,
    SkillExample,
    SkillResult,
    SkillVersion,
    skill,
)


class TestSkillVersion:
    """Tests for SkillVersion."""

    def test_default_version(self):
        """Test default version creation."""
        version = SkillVersion()
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert str(version) == "1.0.0"

    def test_version_from_string(self):
        """Test parsing version from string."""
        version = SkillVersion.from_string("2.5.3")
        assert version.major == 2
        assert version.minor == 5
        assert version.patch == 3

    def test_version_from_short_string(self):
        """Test parsing version from short string."""
        version = SkillVersion.from_string("3.2")
        assert version.major == 3
        assert version.minor == 2
        assert version.patch == 0

    def test_version_compatibility(self):
        """Test version compatibility checking."""
        v1 = SkillVersion(1, 0, 0)
        v2 = SkillVersion(1, 5, 0)
        v3 = SkillVersion(2, 0, 0)

        assert v1.is_compatible_with(v2)
        assert v2.is_compatible_with(v1)
        assert not v1.is_compatible_with(v3)


class TestSkillParameter:
    """Tests for SkillParameter."""

    def test_to_dict(self):
        """Test parameter to dict conversion."""
        param = SkillParameter(
            name="test_param",
            type="string",
            description="A test parameter",
            required=True,
            default="default_value",
            enum=["a", "b", "c"],
            example="example_value",
        )

        result = param.to_dict()
        assert result["name"] == "test_param"
        assert result["type"] == "string"
        assert result["description"] == "A test parameter"
        assert result["required"] is True
        assert result["default"] == "default_value"
        assert result["enum"] == ["a", "b", "c"]
        assert result["example"] == "example_value"


class TestSkillInput:
    """Tests for SkillInput."""

    def test_validate_required(self):
        """Test validation of required parameters."""
        input_spec = SkillInput(
            parameters=[
                SkillParameter(name="required_param", type="string", description="Required", required=True),
                SkillParameter(name="optional_param", type="string", description="Optional", required=False),
            ]
        )

        # Missing required parameter
        errors = input_spec.validate({"optional_param": "value"})
        assert len(errors) == 1
        assert "required_param" in errors[0]

        # All parameters provided
        errors = input_spec.validate({"required_param": "value", "optional_param": "value"})
        assert len(errors) == 0

    def test_validate_enum(self):
        """Test validation of enum parameters."""
        input_spec = SkillInput(
            parameters=[
                SkillParameter(
                    name="enum_param",
                    type="string",
                    description="Enum param",
                    enum=["a", "b", "c"],
                ),
            ]
        )

        # Valid enum value
        errors = input_spec.validate({"enum_param": "a"})
        assert len(errors) == 0

        # Invalid enum value
        errors = input_spec.validate({"enum_param": "d"})
        assert len(errors) == 1
        assert "Invalid value" in errors[0]


class TestSkillMetadata:
    """Tests for SkillMetadata."""

    def test_to_dict(self):
        """Test metadata to dict conversion."""
        metadata = SkillMetadata(
            name="test_skill",
            description="A test skill",
            version=SkillVersion(1, 2, 3),
            category=SkillCategory.TEST,
            level=SkillLevel.INTERMEDIATE,
            tags=["test", "example"],
            author="Test Author",
        )

        result = metadata.to_dict()
        assert result["name"] == "test_skill"
        assert result["description"] == "A test skill"
        assert result["version"] == "1.2.3"
        assert result["category"] == "test"
        assert result["level"] == "intermediate"
        assert result["tags"] == ["test", "example"]
        assert result["author"] == "Test Author"

    def test_get_prompt_context_brief(self):
        """Test brief prompt context."""
        metadata = SkillMetadata(
            name="test_skill",
            description="A test skill",
        )

        context = metadata.get_prompt_context("brief")
        assert "test_skill" in context
        assert "A test skill" in context

    def test_get_prompt_context_full(self):
        """Test full prompt context."""
        metadata = SkillMetadata(
            name="test_skill",
            description="A test skill",
            instructions="Do something useful",
            best_practices=["Practice 1", "Practice 2"],
            common_mistakes=["Mistake 1"],
        )

        context = metadata.get_prompt_context("full")
        assert "test_skill" in context
        assert "Do something useful" in context
        assert "Practice 1" in context
        assert "Mistake 1" in context


class TestSkillContext:
    """Tests for SkillContext."""

    def test_to_dict(self):
        """Test context to dict conversion."""
        context = SkillContext(
            project_path=Path("/project"),
            working_dir=Path("/work"),
            config={"key": "value"},
            session_id="session_123",
        )

        result = context.to_dict()
        # Use Path to handle Windows/Unix path differences
        assert Path(result["project_path"]).name == "project"
        assert Path(result["working_dir"]).name == "work"
        assert result["config"] == {"key": "value"}
        assert result["session_id"] == "session_123"

    def test_get_tool(self):
        """Test getting tools from context."""
        mock_tool = {"name": "mock_tool"}
        context = SkillContext(
            tools={"mock_tool": mock_tool}
        )

        assert context.get_tool("mock_tool") == mock_tool
        assert context.get_tool("nonexistent") is None

    def test_has_tool(self):
        """Test checking tool availability."""
        context = SkillContext(
            tools={"tool1": {}, "tool2": {}}
        )

        assert context.has_tool("tool1") is True
        assert context.has_tool("nonexistent") is False


class TestSkillResult:
    """Tests for SkillResult."""

    def test_ok_result(self):
        """Test successful result creation."""
        result = SkillResult.ok(
            message="Success",
            data={"key": "value"},
            artifacts=["file1.txt"],
        )

        assert result.success is True
        assert result.message == "Success"
        assert result.data == {"key": "value"}
        assert result.artifacts == ["file1.txt"]

    def test_fail_result(self):
        """Test failed result creation."""
        result = SkillResult.fail(
            message="Failed",
            error_code="ERROR_001",
            data={"detail": "error detail"},
        )

        assert result.success is False
        assert result.message == "Failed"
        assert result.metadata["error_code"] == "ERROR_001"
        assert result.data == {"detail": "error detail"}

    def test_partial_result(self):
        """Test partial result creation."""
        result = SkillResult.partial(
            message="Partial success",
            data={"progress": 50},
            completed_steps=2,
            total_steps=4,
        )

        assert result.success is True
        assert result.is_partial() is True
        assert result.metadata["completed_steps"] == 2
        assert result.metadata["total_steps"] == 4

    def test_to_dict(self):
        """Test result to dict conversion."""
        result = SkillResult.ok(message="Test")
        result.duration_ms = 100

        data = result.to_dict()
        assert data["success"] is True
        assert data["message"] == "Test"
        assert data["duration_ms"] == 100


class TestSkillBase:
    """Tests for SkillBase."""

    def test_skill_metadata(self):
        """Test skill metadata generation."""

        class TestSkill(SkillBase):
            name = "test_skill"
            description = "A test skill"
            category = SkillCategory.TEST
            version = "1.5.0"
            author = "Test Author"
            tags = ["test", "example"]

            def get_instructions(self):
                return "Test instructions"

            async def execute(self, task, context, inputs):
                return SkillResult.ok()

        skill = TestSkill()
        metadata = skill.metadata

        assert metadata.name == "test_skill"
        assert metadata.description == "A test skill"
        assert metadata.category == SkillCategory.TEST
        assert str(metadata.version) == "1.5.0"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "example"]
        assert metadata.instructions == "Test instructions"

    def test_validate_inputs(self):
        """Test input validation."""

        class TestSkill(SkillBase):
            name = "test_skill"
            description = "A test skill"

            def get_input_spec(self):
                return SkillInput(
                    parameters=[
                        SkillParameter(name="required", type="string", description="Required", required=True),
                    ]
                )

            async def execute(self, task, context, inputs):
                return SkillResult.ok()

        skill = TestSkill()

        # Missing required input
        errors = skill.validate_inputs({})
        assert len(errors) == 1

        # Valid input
        errors = skill.validate_inputs({"required": "value"})
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_run_with_validation_error(self):
        """Test run with validation error."""

        class TestSkill(SkillBase):
            name = "test_skill"
            description = "A test skill"

            def get_input_spec(self):
                return SkillInput(
                    parameters=[
                        SkillParameter(name="required", type="string", description="Required", required=True),
                    ]
                )

            async def execute(self, task, context, inputs):
                return SkillResult.ok()

        skill = TestSkill()
        context = SkillContext()

        result = await skill.run("task", context, {})

        assert result.success is False
        assert "VALIDATION_ERROR" in result.metadata.get("error_code", "")

    @pytest.mark.asyncio
    async def test_run_success(self):
        """Test successful run."""

        class TestSkill(SkillBase):
            name = "test_skill"
            description = "A test skill"

            async def execute(self, task, context, inputs):
                return SkillResult.ok(message="Success")

        skill = TestSkill()
        context = SkillContext()

        result = await skill.run("task", context, {})

        assert result.success is True
        assert result.message == "Success"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_with_exception(self):
        """Test run with execution exception."""

        class TestSkill(SkillBase):
            name = "test_skill"
            description = "A test skill"

            async def execute(self, task, context, inputs):
                raise ValueError("Test error")

        skill = TestSkill()
        context = SkillContext()

        result = await skill.run("task", context, {})

        assert result.success is False
        assert "EXECUTION_ERROR" in result.metadata.get("error_code", "")
        assert "Test error" in result.message


class TestSkillDecorator:
    """Tests for @skill decorator."""

    @pytest.mark.asyncio
    async def test_skill_decorator(self):
        """Test skill decorator creates valid skill class."""

        @skill("echo", "Echo the input", category=SkillCategory.UTILITY)
        async def echo_skill(task, context, inputs):
            return SkillResult.ok(message=inputs.get("message", ""))

        # Check the created skill class
        assert echo_skill.name == "echo"
        assert echo_skill.description == "Echo the input"
        assert echo_skill.category == SkillCategory.UTILITY

        # Test execution
        instance = echo_skill()
        context = SkillContext()
        result = await instance.run("task", context, {"message": "hello"})

        assert result.success is True
        assert result.message == "hello"


class TestSkillExample:
    """Tests for SkillExample."""

    def test_to_dict(self):
        """Test example to dict conversion."""
        example = SkillExample(
            task="Test task",
            description="Test description",
            expected_result="Test result",
            code_example="code",
            inputs={"key": "value"},
            outputs={"result": "value"},
        )

        result = example.to_dict()
        assert result["task"] == "Test task"
        assert result["description"] == "Test description"
        assert result["expected_result"] == "Test result"
        assert result["code_example"] == "code"
        assert result["inputs"] == {"key": "value"}
        assert result["outputs"] == {"result": "value"}
