"""Tests for agent actions module."""

import pytest
from unittest.mock import Mock, AsyncMock

from pyutagent.agent.actions import (
    ActionType,
    ActionResult,
    Action,
    ActionRegistry,
    ParseCodeAction,
    GenerateTestsAction,
    CompileAction,
    RunTestsAction,
    AnalyzeCoverageAction,
    FixErrorsAction,
)


class TestActionType:
    """Tests for ActionType enum."""

    def test_action_type_values(self):
        """Test that all action types are defined."""
        assert ActionType.PARSE_CODE is not None
        assert ActionType.GENERATE_TESTS is not None
        assert ActionType.COMPILE is not None
        assert ActionType.RUN_TESTS is not None
        assert ActionType.ANALYZE_COVERAGE is not None
        assert ActionType.FIX_ERRORS is not None
        assert ActionType.STORE_MEMORY is not None
        assert ActionType.RETRIEVE_MEMORY is not None


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_action_result_creation(self):
        """Test creating an ActionResult."""
        result = ActionResult(
            success=True,
            message="Test message",
            data={"key": "value"},
            action_type=ActionType.PARSE_CODE
        )
        assert result.success is True
        assert result.message == "Test message"
        assert result.data == {"key": "value"}
        assert result.action_type == ActionType.PARSE_CODE

    def test_action_result_failure(self):
        """Test creating a failed ActionResult."""
        result = ActionResult(
            success=False,
            message="Error occurred",
            data={"error": "details"},
            action_type=ActionType.COMPILE
        )
        assert result.success is False
        assert result.message == "Error occurred"
        assert result.data == {"error": "details"}


class TestActionRegistry:
    """Tests for ActionRegistry."""

    def test_registry_initialization(self):
        """Test registry initialization."""
        registry = ActionRegistry()
        assert registry.list_actions() == []
        assert registry.get_schemas() == []

    def test_register_action(self):
        """Test registering an action."""
        registry = ActionRegistry()
        mock_action = Mock()
        mock_action.action_type = ActionType.PARSE_CODE
        mock_action.name = "parse_code"

        registry.register(mock_action)

        assert registry.get(ActionType.PARSE_CODE) == mock_action
        assert registry.get_by_name("parse_code") == mock_action
        assert len(registry.list_actions()) == 1

    def test_get_nonexistent_action(self):
        """Test getting an action that doesn't exist."""
        registry = ActionRegistry()
        assert registry.get(ActionType.COMPILE) is None
        assert registry.get_by_name("nonexistent") is None

    def test_get_schemas(self):
        """Test getting action schemas."""
        registry = ActionRegistry()
        mock_action = Mock()
        mock_action.action_type = ActionType.PARSE_CODE
        mock_action.name = "parse_code"
        mock_action.get_schema.return_value = {"name": "parse_code"}

        registry.register(mock_action)
        schemas = registry.get_schemas()

        assert len(schemas) == 1
        assert schemas[0] == {"name": "parse_code"}

    @pytest.mark.asyncio
    async def test_execute_action(self):
        """Test executing an action through registry."""
        registry = ActionRegistry()
        mock_action = Mock()
        mock_action.action_type = ActionType.PARSE_CODE
        mock_action.name = "parse_code"
        expected_result = ActionResult(
            success=True,
            message="Success",
            data={},
            action_type=ActionType.PARSE_CODE
        )
        mock_action.execute = AsyncMock(return_value=expected_result)

        registry.register(mock_action)
        result = await registry.execute(ActionType.PARSE_CODE)

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_execute_nonexistent_action(self):
        """Test executing an action that doesn't exist."""
        registry = ActionRegistry()
        result = await registry.execute(ActionType.COMPILE)

        assert result.success is False
        assert "not found" in result.message
        assert result.action_type == ActionType.COMPILE


class TestParseCodeAction:
    """Tests for ParseCodeAction."""

    @pytest.fixture
    def mock_java_parser(self):
        """Create a mock Java parser."""
        return Mock()

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_java_parser):
        """Test successful code parsing."""
        mock_java_parser.parse_class.return_value = {
            "name": "TestClass",
            "methods": []
        }

        action = ParseCodeAction(mock_java_parser)
        result = await action.execute(
            source_code="public class TestClass {}",
            file_path="/path/Test.java"
        )

        assert result.success is True
        assert "TestClass" in result.message
        assert result.data["class_info"]["name"] == "TestClass"
        assert result.data["file_path"] == "/path/Test.java"

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_java_parser):
        """Test failed code parsing."""
        mock_java_parser.parse_class.side_effect = Exception("Parse error")

        action = ParseCodeAction(mock_java_parser)
        result = await action.execute(
            source_code="invalid code",
            file_path="/path/Test.java"
        )

        assert result.success is False
        assert "Failed to parse" in result.message
        assert "Parse error" in result.data["error"]

    def test_get_schema(self, mock_java_parser):
        """Test getting action schema."""
        action = ParseCodeAction(mock_java_parser)
        schema = action.get_schema()

        assert schema["name"] == "parse_code"
        assert "parameters" in schema
        assert "source_code" in schema["parameters"]["properties"]

    def test_get_parameters(self, mock_java_parser):
        """Test getting parameter schema."""
        action = ParseCodeAction(mock_java_parser)
        params = action._get_parameters()

        assert params["type"] == "object"
        assert "source_code" in params["properties"]
        assert "file_path" in params["properties"]
        assert "source_code" in params["required"]


class TestGenerateTestsAction:
    """Tests for GenerateTestsAction."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return Mock()

    @pytest.fixture
    def mock_prompt_builder(self):
        """Create a mock prompt builder."""
        return Mock()

    @pytest.mark.asyncio
    async def test_execute_initial_tests(self, mock_llm_client, mock_prompt_builder):
        """Test generating initial tests."""
        mock_prompt_builder.build_initial_test_prompt.return_value = "initial prompt"
        mock_llm_client.generate = AsyncMock(return_value="generated test code")

        action = GenerateTestsAction(mock_llm_client, mock_prompt_builder)
        result = await action.execute(
            class_info={"name": "TestClass"},
            source_code="public class TestClass {}"
        )

        assert result.success is True
        assert result.data["test_code"] == "generated test code"
        mock_prompt_builder.build_initial_test_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_additional_tests(self, mock_llm_client, mock_prompt_builder):
        """Test generating additional tests for coverage."""
        mock_prompt_builder.build_additional_tests_prompt.return_value = "additional prompt"
        mock_llm_client.generate = AsyncMock(return_value="additional test code")

        action = GenerateTestsAction(mock_llm_client, mock_prompt_builder)
        result = await action.execute(
            class_info={"name": "TestClass"},
            source_code="public class TestClass {}",
            existing_tests="existing tests",
            uncovered_info={"lines": [10, 20]},
            current_coverage=0.5
        )

        assert result.success is True
        mock_prompt_builder.build_additional_tests_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_llm_client, mock_prompt_builder):
        """Test generation failure."""
        mock_prompt_builder.build_initial_test_prompt.return_value = "prompt"
        mock_llm_client.generate = AsyncMock(side_effect=Exception("LLM error"))

        action = GenerateTestsAction(mock_llm_client, mock_prompt_builder)
        result = await action.execute(
            class_info={"name": "TestClass"},
            source_code="public class TestClass {}"
        )

        assert result.success is False
        assert "Failed to generate" in result.message


class TestCompileAction:
    """Tests for CompileAction."""

    @pytest.fixture
    def mock_maven_runner(self):
        """Create a mock Maven runner."""
        return Mock()

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_maven_runner):
        """Test successful compilation."""
        mock_maven_runner.compile_project.return_value = True

        action = CompileAction(mock_maven_runner)
        result = await action.execute()

        assert result.success is True
        assert "successful" in result.message

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_maven_runner):
        """Test failed compilation."""
        mock_maven_runner.compile_project.return_value = False

        action = CompileAction(mock_maven_runner)
        result = await action.execute()

        assert result.success is False
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_exception(self, mock_maven_runner):
        """Test compilation with exception."""
        mock_maven_runner.compile_project.side_effect = Exception("Maven error")

        action = CompileAction(mock_maven_runner)
        result = await action.execute()

        assert result.success is False
        assert "error" in result.message.lower()


class TestRunTestsAction:
    """Tests for RunTestsAction."""

    @pytest.fixture
    def mock_maven_runner(self):
        """Create a mock Maven runner."""
        return Mock()

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_maven_runner):
        """Test successful test run."""
        mock_maven_runner.run_tests.return_value = True

        action = RunTestsAction(mock_maven_runner)
        result = await action.execute()

        assert result.success is True
        assert "passed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_maven_runner):
        """Test failed test run."""
        mock_maven_runner.run_tests.return_value = False

        action = RunTestsAction(mock_maven_runner)
        result = await action.execute()

        assert result.success is False
        assert "failed" in result.message.lower()


class TestAnalyzeCoverageAction:
    """Tests for AnalyzeCoverageAction."""

    @pytest.fixture
    def mock_coverage_analyzer(self):
        """Create a mock coverage analyzer."""
        return Mock()

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_coverage_analyzer):
        """Test successful coverage analysis."""
        mock_report = Mock()
        mock_report.line_coverage = 0.85
        mock_report.branch_coverage = 0.75
        mock_report.method_coverage = 0.90
        mock_coverage_analyzer.parse_report.return_value = mock_report

        action = AnalyzeCoverageAction(mock_coverage_analyzer)
        result = await action.execute()

        assert result.success is True
        assert result.data["line_coverage"] == 0.85
        assert result.data["branch_coverage"] == 0.75
        assert result.data["method_coverage"] == 0.90

    @pytest.mark.asyncio
    async def test_execute_no_report(self, mock_coverage_analyzer):
        """Test when no coverage report is found."""
        mock_coverage_analyzer.parse_report.return_value = None

        action = AnalyzeCoverageAction(mock_coverage_analyzer)
        result = await action.execute()

        assert result.success is False
        assert "Failed to parse" in result.message


class TestFixErrorsAction:
    """Tests for FixErrorsAction."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        return Mock()

    @pytest.fixture
    def mock_prompt_builder(self):
        """Create a mock prompt builder."""
        return Mock()

    @pytest.mark.asyncio
    async def test_execute_compilation_fix(self, mock_llm_client, mock_prompt_builder):
        """Test fixing compilation errors."""
        mock_prompt_builder.build_fix_compilation_prompt.return_value = "fix prompt"
        mock_llm_client.generate = AsyncMock(return_value="fixed code")

        action = FixErrorsAction(mock_llm_client, mock_prompt_builder)
        result = await action.execute(
            error_type="compilation",
            test_code="broken code",
            errors=["error1", "error2"],
            class_info={"name": "TestClass"}
        )

        assert result.success is True
        assert result.data["fixed_code"] == "fixed code"
        mock_prompt_builder.build_fix_compilation_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_test_failure_fix(self, mock_llm_client, mock_prompt_builder):
        """Test fixing test failures."""
        mock_prompt_builder.build_fix_test_failure_prompt.return_value = "fix prompt"
        mock_llm_client.generate = AsyncMock(return_value="fixed code")

        action = FixErrorsAction(mock_llm_client, mock_prompt_builder)
        result = await action.execute(
            error_type="test",
            test_code="failing test",
            errors=[{"test_name": "test1", "error": "assertion failed"}],
            class_info={"name": "TestClass"}
        )

        assert result.success is True
        mock_prompt_builder.build_fix_test_failure_prompt.assert_called_once()

    def test_get_parameters(self, mock_llm_client, mock_prompt_builder):
        """Test getting parameter schema."""
        action = FixErrorsAction(mock_llm_client, mock_prompt_builder)
        params = action._get_parameters()

        assert params["type"] == "object"
        assert "error_type" in params["properties"]
        assert "test_code" in params["properties"]
        assert "errors" in params["properties"]
        assert params["required"] == ["error_type", "test_code", "errors"]
