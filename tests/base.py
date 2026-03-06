"""Unified Test Base Classes - 统一测试基类

This module provides base classes for all tests in the PyUT Agent project.
Use these base classes to ensure consistent test behavior and reduce boilerplate.

Usage:
    >>> from tests.base import BaseTestCase, AsyncTestCase
    >>> 
    >>> class TestMyFeature(BaseTestCase):
    ...     def test_something(self):
    ...         result = self.do_something()
    ...         self.assert_success(result)
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from _pytest.fixtures import FixtureRequest

T = TypeVar('T')


class BaseTestCase:
    """Base class for all test cases.
    
    Provides common utilities and setup/teardown behavior.
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_case(self, request: FixtureRequest):
        """Setup method called before each test.
        
        Override _setup() in subclasses to add custom setup logic.
        """
        self._test_name = request.node.name
        self._setup()
        yield
        self._teardown()
    
    def _setup(self) -> None:
        """Override this method to add custom setup logic."""
        pass
    
    def _teardown(self) -> None:
        """Override this method to add custom teardown logic."""
        pass
    
    # =========================================================================
    # Assertion Helpers
    # =========================================================================
    
    def assert_success(self, result: Any, message: str = "") -> None:
        """Assert that a result indicates success.
        
        Checks for:
        - result.success == True
        - result.error is None
        
        Args:
            result: Result object to check
            message: Optional message to display on failure
        """
        assert hasattr(result, 'success'), f"Result has no 'success' attribute: {result}"
        assert result.success is True, message or f"Expected success, got: {result}"
        
        if hasattr(result, 'error'):
            assert result.error is None, f"Expected no error, got: {result.error}"
    
    def assert_failure(self, result: Any, expected_error: Optional[str] = None) -> None:
        """Assert that a result indicates failure.
        
        Args:
            result: Result object to check
            expected_error: Optional error message to check for
        """
        assert hasattr(result, 'success'), f"Result has no 'success' attribute: {result}"
        assert result.success is False, f"Expected failure, got success: {result}"
        
        if expected_error and hasattr(result, 'error'):
            error_str = str(result.error)
            assert expected_error in error_str, f"Expected error containing '{expected_error}', got: {error_str}"
    
    def assert_contains(self, container: Any, item: Any, message: str = "") -> None:
        """Assert that container contains item.
        
        Args:
            container: Container to check (list, dict, str, etc.)
            item: Item to look for
            message: Optional message to display on failure
        """
        assert item in container, message or f"Expected {container} to contain {item}"
    
    def assert_not_empty(self, container: Any, message: str = "") -> None:
        """Assert that container is not empty.
        
        Args:
            container: Container to check
            message: Optional message to display on failure
        """
        assert len(container) > 0, message or f"Expected non-empty container, got: {container}"
    
    def assert_is_instance(self, obj: Any, class_type: type, message: str = "") -> None:
        """Assert that object is instance of class_type.
        
        Args:
            obj: Object to check
            class_type: Expected type
            message: Optional message to display on failure
        """
        assert isinstance(obj, class_type), message or f"Expected {class_type}, got {type(obj)}"
    
    # =========================================================================
    # Mock Helpers
    # =========================================================================
    
    def create_mock(self, spec: Optional[type] = None, **kwargs) -> Mock:
        """Create a mock object.
        
        Args:
            spec: Class to mock
            **kwargs: Attributes to set on the mock
            
        Returns:
            Mock object
        """
        mock = Mock(spec=spec, **kwargs)
        return mock
    
    def create_async_mock(self, return_value: Any = None, **kwargs) -> AsyncMock:
        """Create an async mock object.
        
        Args:
            return_value: Value to return from async methods
            **kwargs: Attributes to set on the mock
            
        Returns:
            AsyncMock object
        """
        mock = AsyncMock(**kwargs)
        if return_value is not None:
            mock.return_value = return_value
        return mock
    
    def patch_object(self, target: str, **kwargs) -> Any:
        """Patch an object and return the mock.
        
        Args:
            target: Object path to patch (e.g., 'module.ClassName.method')
            **kwargs: Attributes to set on the mock
            
        Returns:
            Mock object
        """
        patcher = patch(target, **kwargs)
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock
    
    def addCleanup(self, func, *args, **kwargs) -> None:
        """Add a cleanup function to be called after the test.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        # Store cleanup functions
        if not hasattr(self, '_cleanups'):
            self._cleanups = []
        self._cleanups.append((func, args, kwargs))
    
    # =========================================================================
    # File System Helpers
    # =========================================================================
    
    def create_temp_file(self, content: str = "", suffix: str = ".txt") -> Path:
        """Create a temporary file with content.
        
        Args:
            content: File content
            suffix: File suffix
            
        Returns:
            Path to temporary file
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return Path(f.name)
    
    def create_temp_dir(self) -> Path:
        """Create a temporary directory.
        
        Returns:
            Path to temporary directory
        """
        return Path(tempfile.mkdtemp())
    
    def read_file(self, path: Path) -> str:
        """Read file content.
        
        Args:
            path: File path
            
        Returns:
            File content
        """
        return path.read_text(encoding='utf-8')
    
    def write_file(self, path: Path, content: str) -> None:
        """Write content to file.
        
        Args:
            path: File path
            content: Content to write
        """
        path.write_text(content, encoding='utf-8')


class AsyncTestCase(BaseTestCase):
    """Base class for async test cases.
    
    Provides utilities for testing async code.
    """
    
    @pytest.fixture(autouse=True)
    async def setup_async_test_case(self, request: FixtureRequest):
        """Setup method called before each async test."""
        self._test_name = request.node.name
        self._setup()
        await self._async_setup()
        yield
        await self._async_teardown()
        self._teardown()
    
    async def _async_setup(self) -> None:
        """Override this method to add custom async setup logic."""
        pass
    
    async def _async_teardown(self) -> None:
        """Override this method to add custom async teardown logic."""
        pass
    
    async def run_async(self, coro) -> Any:
        """Run an async coroutine.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
        """
        return await coro


class ComponentTestCase(BaseTestCase):
    """Base class for testing components.
    
    Provides utilities for testing PyUT Agent components.
    """
    
    def _setup(self) -> None:
        """Setup method called before each test method."""
        super()._setup()
        self._component_instances: List[Any] = []
    
    def register_component(self, component: Any) -> Any:
        """Register a component for cleanup.
        
        Args:
            component: Component to register
            
        Returns:
            The component
        """
        self._component_instances.append(component)
        return component
    
    def _teardown(self) -> None:
        """Teardown method called after each test method."""
        # Cleanup all registered components
        for component in reversed(self._component_instances):
            if hasattr(component, 'cleanup'):
                component.cleanup()
            elif hasattr(component, 'shutdown'):
                if asyncio.iscoroutinefunction(component.shutdown):
                    asyncio.run(component.shutdown())
                else:
                    component.shutdown()
        
        self._component_instances.clear()
        super()._teardown()


class AgentTestCase(AsyncTestCase):
    """Base class for testing agents.
    
    Provides utilities for testing agent implementations.
    """
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        from tests.fixtures import create_mock_llm_client
        return create_mock_llm_client()
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock agent context."""
        mock = MagicMock()
        mock.get = MagicMock(return_value=None)
        mock.set = MagicMock()
        return mock
    
    def create_test_task(self, name: str = "test_task", **kwargs) -> Dict[str, Any]:
        """Create a test task.
        
        Args:
            name: Task name
            **kwargs: Additional task attributes
            
        Returns:
            Task dictionary
        """
        task = {
            "name": name,
            "description": kwargs.get("description", f"Test task: {name}"),
            "inputs": kwargs.get("inputs", {}),
            "expected_outputs": kwargs.get("expected_outputs", []),
        }
        task.update(kwargs)
        return task


class ToolTestCase(BaseTestCase):
    """Base class for testing tools.
    
    Provides utilities for testing tool implementations.
    """
    
    @pytest.fixture
    def tool_context(self, tmp_path):
        """Create a tool context."""
        from pyutagent.tools.core.tool_context import ToolContext
        return ToolContext(project_path=tmp_path)
    
    def assert_tool_success(self, result: Any) -> None:
        """Assert that a tool execution was successful.
        
        Args:
            result: Tool result to check
        """
        from pyutagent.tools.core.tool_result import ResultStatus
        
        if hasattr(result, 'status'):
            assert result.status == ResultStatus.SUCCESS, f"Tool failed: {result}"
        else:
            self.assert_success(result)
    
    def assert_tool_failure(self, result: Any, expected_error_code: Optional[str] = None) -> None:
        """Assert that a tool execution failed.
        
        Args:
            result: Tool result to check
            expected_error_code: Optional error code to check for
        """
        from pyutagent.tools.core.tool_result import ResultStatus
        
        if hasattr(result, 'status'):
            assert result.status == ResultStatus.FAILURE, f"Expected failure, got: {result}"
            
            if expected_error_code and hasattr(result, 'error'):
                assert result.error.code == expected_error_code, \
                    f"Expected error code {expected_error_code}, got: {result.error.code}"
        else:
            self.assert_failure(result)


class IntegrationTestCase(AsyncTestCase):
    """Base class for integration tests.
    
    Provides utilities for integration testing.
    """
    
    @pytest.fixture(scope="class")
    def temp_project_dir(self, tmp_path_factory):
        """Create a temporary project directory for integration tests."""
        return tmp_path_factory.mktemp("integration_test")
    
    def create_maven_project(self, base_path: Path) -> Path:
        """Create a minimal Maven project structure.
        
        Args:
            base_path: Base path for the project
            
        Returns:
            Path to project directory
        """
        from tests.fixtures.maven_projects import create_minimal_maven_project
        return create_minimal_maven_project(base_path)


class GUITestCase(BaseTestCase):
    """Base class for GUI tests.
    
    Provides utilities for testing GUI components.
    """
    
    @pytest.fixture(autouse=True)
    def setup_qt(self, qtbot):
        """Setup QtBot for GUI testing."""
        self.qtbot = qtbot
        yield
    
    def wait_for_signal(self, signal, timeout: int = 1000):
        """Wait for a Qt signal.
        
        Args:
            signal: Signal to wait for
            timeout: Timeout in milliseconds
        """
        from pytestqt.qt_compat import qt_api
        
        spy = self.qtbot.waitSignal(signal, timeout=timeout)
        return spy


# =========================================================================
# Decorators
# =========================================================================

def require_fixture(fixture_name: str):
    """Decorator to mark a test as requiring a specific fixture.
    
    Args:
        fixture_name: Name of the required fixture
        
    Usage:
        @require_fixture("mock_llm_client")
        def test_something(self, mock_llm_client):
            pass
    """
    return pytest.mark.usefixtures(fixture_name)


def slow_test(reason: str = ""):
    """Decorator to mark a test as slow.
    
    Args:
        reason: Reason why the test is slow
    """
    return pytest.mark.slow(reason)


def flaky_test(max_runs: int = 3):
    """Decorator to mark a test as flaky.
    
    Args:
        max_runs: Maximum number of runs
    """
    return pytest.mark.flaky(max_runs=max_runs)


# =========================================================================
# Exports
# =========================================================================

__all__ = [
    # Base Classes
    'BaseTestCase',
    'AsyncTestCase',
    'ComponentTestCase',
    'AgentTestCase',
    'ToolTestCase',
    'IntegrationTestCase',
    'GUITestCase',
    # Decorators
    'require_fixture',
    'slow_test',
    'flaky_test',
]
