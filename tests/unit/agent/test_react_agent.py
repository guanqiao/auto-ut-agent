"""Tests for ReActAgent."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from pyutagent.agent.react_agent import ReActAgent
from pyutagent.agent.base_agent import AgentState, AgentResult, StepResult
from pyutagent.llm.client import LLMClient
from pyutagent.memory.working_memory import WorkingMemory


class TestReActAgentInitialization:
    """Test ReActAgent initialization."""

    def test_init_with_dependencies(self):
        """Test initialization with all dependencies."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        project_path = "/test/project"

        agent = ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=project_path
        )

        assert agent.llm_client == llm_client
        assert agent.working_memory == working_memory
        # Path is normalized by Path constructor
        assert agent.project_path is not None
        assert agent._stop_requested is False
        assert agent.current_test_file is None

    def test_init_with_container(self):
        """Test initialization with dependency injection container."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        project_path = "/test/project"

        container = Mock()
        container.resolve = Mock(side_effect=KeyError("Not found"))

        agent = ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=project_path,
            container=container
        )

        assert agent._container == container

    def test_init_creates_default_dependencies(self):
        """Test that default dependencies are created when not in container."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        project_path = "/test/project"

        agent = ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=project_path
        )

        assert agent.prompt_builder is not None
        assert agent.action_registry is not None
        assert agent.java_parser is not None
        assert agent.maven_runner is not None
        assert agent.coverage_analyzer is not None
        assert agent.project_scanner is not None


class TestReActAgentStopAndReset:
    """Test stop and reset functionality."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        return ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path="/test/project"
        )

    def test_stop(self, agent):
        """Test stop method."""
        agent.stop()

        assert agent._stop_requested is True
        assert agent.retry_manager._stop_requested is True

    def test_reset(self, agent):
        """Test reset method."""
        agent._stop_requested = True

        agent.reset()

        assert agent._stop_requested is False
        assert agent.retry_manager._stop_requested is False


class TestReActAgentTryResolve:
    """Test _try_resolve method."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        return ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path="/test/project"
        )

    def test_try_resolve_success(self, agent):
        """Test successful resolution."""
        mock_component = Mock()
        agent._container.resolve = Mock(return_value=mock_component)

        result = agent._try_resolve(Mock)

        assert result == mock_component

    def test_try_resolve_key_error(self, agent):
        """Test resolution with KeyError."""
        agent._container.resolve = Mock(side_effect=KeyError("Not found"))

        result = agent._try_resolve(Mock)

        assert result is None

    def test_try_resolve_other_exception(self, agent):
        """Test resolution with other exception."""
        agent._container.resolve = Mock(side_effect=RuntimeError("Error"))

        result = agent._try_resolve(Mock)

        assert result is None


class TestReActAgentParseTargetFile:
    """Test parse target file functionality."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create a test agent with temporary project path."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        return ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=str(tmp_path)
        )

    @pytest.mark.asyncio
    async def test_parse_target_file_not_found(self, agent, tmp_path):
        """Test file parsing when file not found."""
        result = await agent._parse_target_file("NonExistent.java")

        assert result.success is False
        assert result.state == AgentState.FAILED
        assert "File not found" in result.message


class TestReActAgentGenerateInitialTests:
    """Test generate initial tests functionality."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create a test agent."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        agent = ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=str(tmp_path)
        )
        agent.target_class_info = {
            "class_name": "TestClass",
            "methods": [{"name": "testMethod"}],
            "source": "public class TestClass {}"
        }
        return agent

    @pytest.mark.asyncio
    async def test_generate_initial_tests_success(self, agent):
        """Test successful initial test generation."""
        test_code = "@Test public void test() {}"

        agent.llm_client.generate = AsyncMock(return_value=test_code)
        agent.prompt_builder.build_initial_test_prompt = Mock(return_value="prompt")
        agent._extract_java_code = Mock(return_value=test_code)
        agent._write_test_file = Mock(return_value=True)

        result = await agent._generate_initial_tests()

        assert isinstance(result, StepResult)
        agent.llm_client.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_initial_tests_empty_response(self, agent):
        """Test generation with empty response - should still succeed if file is written."""
        agent.llm_client.generate = AsyncMock(return_value="")
        agent.prompt_builder.build_initial_test_prompt = Mock(return_value="prompt")
        agent._extract_java_code = Mock(return_value="")
        agent._write_test_file = Mock(return_value=True)

        result = await agent._generate_initial_tests()

        assert isinstance(result, StepResult)
        # Empty response but file written successfully returns success
        assert result.success is True


class TestReActAgentStateManagement:
    """Test state management functionality."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        return ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path="/test/project"
        )

    def test_update_state(self, agent):
        """Test state update."""
        callback = Mock()
        agent.progress_callback = callback

        agent._update_state(AgentState.GENERATING, "Generating tests")

        assert agent.state == AgentState.GENERATING
        callback.assert_called_once()

    def test_update_state_no_callback(self, agent):
        """Test state update without callback."""
        agent.progress_callback = None

        agent._update_state(AgentState.COMPILING, "Compiling")

        assert agent.state == AgentState.COMPILING


class TestReActAgentProgressCallback:
    """Test progress callback functionality."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        return ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path="/test/project"
        )

    def test_on_recovery_progress(self, agent):
        """Test recovery progress callback."""
        callback = Mock()
        agent.progress_callback = callback

        agent._on_recovery_progress("ANALYZING", "Analyzing error")

        assert agent.state == AgentState.FIXING
        callback.assert_called_once()


class TestReActAgentAiderFixer:
    """Test Aider fixer initialization."""

    def test_init_aider_fixer_success(self):
        """Test successful Aider fixer initialization."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)

        with patch('pyutagent.agent.react_agent.AiderCodeFixer') as mock_fixer:
            mock_fixer.return_value = Mock()
            agent = ReActAgent(
                llm_client=llm_client,
                working_memory=working_memory,
                project_path="/test/project"
            )

            assert agent.aider_fixer is not None

    def test_init_aider_fixer_import_error(self):
        """Test Aider fixer initialization with import error."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)

        with patch('pyutagent.agent.react_agent.AiderCodeFixer', side_effect=ImportError()):
            agent = ReActAgent(
                llm_client=llm_client,
                working_memory=working_memory,
                project_path="/test/project"
            )

            assert agent.aider_fixer is None

    def test_init_aider_fixer_from_container(self):
        """Test Aider fixer resolved from container."""
        llm_client = Mock(spec=LLMClient)
        working_memory = Mock(spec=WorkingMemory)
        mock_fixer = Mock()

        container = Mock()
        container.resolve = Mock(return_value=mock_fixer)

        agent = ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path="/test/project",
            container=container
        )

        assert agent.aider_fixer == mock_fixer
