"""Tests for unified test base classes."""
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from tests import (
    BaseTestCase,
    AsyncTestCase,
    ComponentTestCase,
    AgentTestCase,
    ToolTestCase,
    IntegrationTestCase,
    require_fixture,
    slow_test,
    flaky_test,
)
from pyutagent.core.interfaces import ExecutionResult


class TestBaseTestCase(BaseTestCase):
    """Test cases for BaseTestCase."""
    
    def test_assert_success_with_success_result(self):
        """Test assert_success with successful result."""
        result = ExecutionResult(success=True)
        self.assert_success(result)  # Should not raise
    
    def test_assert_success_with_failure_result(self):
        """Test assert_success with failure result raises assertion error."""
        result = ExecutionResult(success=False)
        with pytest.raises(AssertionError):
            self.assert_success(result)
    
    def test_assert_success_with_error(self):
        """Test assert_success with error raises assertion error."""
        result = ExecutionResult(success=True, error=Exception("test error"))
        with pytest.raises(AssertionError):
            self.assert_success(result)
    
    def test_assert_failure_with_failure_result(self):
        """Test assert_failure with failure result."""
        result = ExecutionResult(success=False)
        self.assert_failure(result)  # Should not raise
    
    def test_assert_failure_with_success_result(self):
        """Test assert_failure with success result raises assertion error."""
        result = ExecutionResult(success=True)
        with pytest.raises(AssertionError):
            self.assert_failure(result)
    
    def test_assert_contains_with_list(self):
        """Test assert_contains with list."""
        container = [1, 2, 3]
        self.assert_contains(container, 2)  # Should not raise
    
    def test_assert_contains_with_string(self):
        """Test assert_contains with string."""
        container = "hello world"
        self.assert_contains(container, "world")  # Should not raise
    
    def test_assert_contains_not_found(self):
        """Test assert_contains when item not found."""
        container = [1, 2, 3]
        with pytest.raises(AssertionError):
            self.assert_contains(container, 4)
    
    def test_assert_not_empty_with_non_empty_list(self):
        """Test assert_not_empty with non-empty list."""
        container = [1, 2, 3]
        self.assert_not_empty(container)  # Should not raise
    
    def test_assert_not_empty_with_empty_list(self):
        """Test assert_not_empty with empty list raises assertion error."""
        container = []
        with pytest.raises(AssertionError):
            self.assert_not_empty(container)
    
    def test_assert_is_instance_with_correct_type(self):
        """Test assert_is_instance with correct type."""
        obj = "test"
        self.assert_is_instance(obj, str)  # Should not raise
    
    def test_assert_is_instance_with_wrong_type(self):
        """Test assert_is_instance with wrong type raises assertion error."""
        obj = "test"
        with pytest.raises(AssertionError):
            self.assert_is_instance(obj, int)
    
    def test_create_mock(self):
        """Test create_mock creates a Mock object."""
        mock = self.create_mock()
        assert isinstance(mock, Mock)
    
    def test_create_mock_with_spec(self):
        """Test create_mock with spec."""
        class TestClass:
            def method(self): pass
        
        mock = self.create_mock(spec=TestClass)
        assert isinstance(mock, Mock)
    
    def test_create_async_mock(self):
        """Test create_async_mock creates an AsyncMock object."""
        mock = self.create_async_mock()
        assert isinstance(mock, AsyncMock)
    
    def test_create_async_mock_with_return_value(self):
        """Test create_async_mock with return value."""
        mock = self.create_async_mock(return_value="test")
        assert mock.return_value == "test"
    
    def test_create_temp_file(self):
        """Test create_temp_file creates a file."""
        path = self.create_temp_file(content="test content")
        assert path.exists()
        assert path.read_text() == "test content"
        path.unlink()  # Cleanup
    
    def test_create_temp_dir(self):
        """Test create_temp_dir creates a directory."""
        path = self.create_temp_dir()
        assert path.exists()
        assert path.is_dir()
        import shutil
        shutil.rmtree(path)  # Cleanup
    
    def test_read_file(self):
        """Test read_file reads file content."""
        path = self.create_temp_file(content="test content")
        content = self.read_file(path)
        assert content == "test content"
        path.unlink()  # Cleanup
    
    def test_write_file(self):
        """Test write_file writes content to file."""
        path = Path(self.create_temp_dir()) / "test.txt"
        self.write_file(path, "test content")
        assert path.read_text() == "test content"
        import shutil
        shutil.rmtree(path.parent)  # Cleanup


class TestAsyncTestCase(AsyncTestCase):
    """Test cases for AsyncTestCase."""
    
    @pytest.mark.asyncio
    async def test_run_async(self):
        """Test run_async runs coroutine."""
        async def coro():
            return "result"
        
        result = await self.run_async(coro())
        assert result == "result"


class TestComponentTestCase(ComponentTestCase):
    """Test cases for ComponentTestCase."""
    
    def test_register_component(self):
        """Test register_component registers component."""
        component = Mock()
        result = self.register_component(component)
        assert result is component


class TestAgentTestCase(AgentTestCase):
    """Test cases for AgentTestCase."""
    
    def test_create_test_task(self):
        """Test create_test_task creates task dictionary."""
        task = self.create_test_task("my_task")
        assert task["name"] == "my_task"
        assert task["description"] == "Test task: my_task"
        assert task["inputs"] == {}
        assert task["expected_outputs"] == []
    
    def test_create_test_task_with_kwargs(self):
        """Test create_test_task with additional kwargs."""
        task = self.create_test_task(
            "my_task",
            description="Custom description",
            inputs={"key": "value"},
            expected_outputs=["output1"]
        )
        assert task["name"] == "my_task"
        assert task["description"] == "Custom description"
        assert task["inputs"] == {"key": "value"}
        assert task["expected_outputs"] == ["output1"]


class TestToolTestCase(ToolTestCase):
    """Test cases for ToolTestCase."""
    
    def test_assert_tool_success_with_success_status(self):
        """Test assert_tool_success with success status."""
        from pyutagent.tools.core.tool_result import ToolResult, ResultStatus
        
        result = ToolResult(status=ResultStatus.SUCCESS)
        self.assert_tool_success(result)  # Should not raise
    
    def test_assert_tool_success_with_failure_status(self):
        """Test assert_tool_success with failure status raises error."""
        from pyutagent.tools.core.tool_result import ToolResult, ResultStatus
        
        result = ToolResult(status=ResultStatus.FAILURE)
        with pytest.raises(AssertionError):
            self.assert_tool_success(result)
    
    def test_assert_tool_failure_with_failure_status(self):
        """Test assert_tool_failure with failure status."""
        from pyutagent.tools.core.tool_result import ToolResult, ResultStatus
        
        result = ToolResult(status=ResultStatus.FAILURE)
        self.assert_tool_failure(result)  # Should not raise
    
    def test_assert_tool_failure_with_success_status(self):
        """Test assert_tool_failure with success status raises error."""
        from pyutagent.tools.core.tool_result import ToolResult, ResultStatus
        
        result = ToolResult(status=ResultStatus.SUCCESS)
        with pytest.raises(AssertionError):
            self.assert_tool_failure(result)


class TestIntegrationTestCase(IntegrationTestCase):
    """Test cases for IntegrationTestCase."""
    
    def test_create_maven_project(self, tmp_path):
        """Test create_maven_project creates Maven project."""
        project_path = self.create_maven_project(tmp_path)
        assert project_path.exists()
        assert (project_path / "pom.xml").exists()
        assert (project_path / "src" / "main" / "java").exists()
        assert (project_path / "src" / "test" / "java").exists()


class TestDecorators:
    """Test cases for test decorators."""
    
    def test_require_fixture_decorator(self):
        """Test require_fixture decorator."""
        decorated = require_fixture("mock_llm_client")
        assert decorated is not None
        # The decorator should be a pytest marker
        assert hasattr(decorated, 'mark')
    
    def test_slow_test_decorator(self):
        """Test slow_test decorator."""
        decorated = slow_test("This test is slow")
        assert decorated is not None
        assert hasattr(decorated, 'mark')
    
    def test_flaky_test_decorator(self):
        """Test flaky_test decorator."""
        decorated = flaky_test(max_runs=5)
        assert decorated is not None
        assert hasattr(decorated, 'mark')


class TestTestExports:
    """Test that all test classes are properly exported."""
    
    def test_base_test_case_exported(self):
        """Test BaseTestCase is exported from tests package."""
        from tests import BaseTestCase
        assert BaseTestCase is not None
    
    def test_async_test_case_exported(self):
        """Test AsyncTestCase is exported from tests package."""
        from tests import AsyncTestCase
        assert AsyncTestCase is not None
    
    def test_component_test_case_exported(self):
        """Test ComponentTestCase is exported from tests package."""
        from tests import ComponentTestCase
        assert ComponentTestCase is not None
    
    def test_agent_test_case_exported(self):
        """Test AgentTestCase is exported from tests package."""
        from tests import AgentTestCase
        assert AgentTestCase is not None
    
    def test_tool_test_case_exported(self):
        """Test ToolTestCase is exported from tests package."""
        from tests import ToolTestCase
        assert ToolTestCase is not None
    
    def test_integration_test_case_exported(self):
        """Test IntegrationTestCase is exported from tests package."""
        from tests import IntegrationTestCase
        assert IntegrationTestCase is not None
    
    def test_gui_test_case_exported(self):
        """Test GUITestCase is exported from tests package."""
        from tests import GUITestCase
        assert GUITestCase is not None
    
    def test_require_fixture_exported(self):
        """Test require_fixture is exported from tests package."""
        from tests import require_fixture
        assert require_fixture is not None
    
    def test_slow_test_exported(self):
        """Test slow_test is exported from tests package."""
        from tests import slow_test
        assert slow_test is not None
    
    def test_flaky_test_exported(self):
        """Test flaky_test is exported from tests package."""
        from tests import flaky_test
        assert flaky_test is not None
