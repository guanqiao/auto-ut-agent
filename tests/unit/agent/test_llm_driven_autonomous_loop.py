"""Tests for LLM-Driven Autonomous Loop."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from pyutagent.agent.llm_driven_autonomous_loop import (
    LLMActionDecider,
    DynamicToolSelector,
    LLMDrivenAutonomousLoop,
    DecisionStrategy,
    DecisionContext,
    LLMDecision,
    create_llm_driven_loop
)
from pyutagent.tools.tool import Tool, ToolDefinition, ToolResult, ToolCategory


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool"):
        super().__init__()
        self._name = name
        self._definition = ToolDefinition(
            name=name,
            description="Mock tool for testing",
            category=ToolCategory.COMMAND,
            parameters=[]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"Executed {self._name}")


class TestDecisionContext:
    """Test DecisionContext dataclass."""

    def test_decision_context_creation(self):
        """Test creating DecisionContext."""
        context = DecisionContext(
            task_goal="Fix the bug",
            current_state="initial",
            iteration=1,
            max_iterations=10
        )

        assert context.task_goal == "Fix the bug"
        assert context.current_state == "initial"
        assert context.iteration == 1
        assert context.max_iterations == 10

    def test_decision_context_defaults(self):
        """Test DecisionContext default values."""
        context = DecisionContext(
            task_goal="Test",
            current_state="start"
        )

        assert context.recent_actions == []
        assert context.available_tools == []
        assert context.last_result is None
        assert context.iteration == 0


class TestLLMDecision:
    """Test LLMDecision dataclass."""

    def test_llm_decision_creation(self):
        """Test creating LLMDecision."""
        decision = LLMDecision(
            reasoning="Need to read the file first",
            action="read_file",
            parameters={"path": "test.py"},
            expected_outcome="Get file content",
            confidence=0.8,
            alternatives=[]
        )

        assert decision.reasoning == "Need to read the file first"
        assert decision.action == "read_file"
        assert decision.confidence == 0.8

    def test_llm_decision_with_alternatives(self):
        """Test LLMDecision with alternatives."""
        decision = LLMDecision(
            reasoning="Test reasoning",
            action="bash",
            parameters={},
            expected_outcome="Test",
            confidence=0.5,
            alternatives=[
                {"action": "grep", "reasoning": "Alternative approach"}
            ]
        )

        assert len(decision.alternatives) == 1
        assert decision.alternatives[0]["action"] == "grep"


class TestLLMActionDecider:
    """Test LLMActionDecider class."""

    @pytest.mark.asyncio
    async def test_action_decider_initialization(self):
        """Test initializing ActionDecider."""
        llm_client = Mock()
        tool_registry = Mock()
        tool_registry.list_tools = Mock(return_value=[])

        decider = LLMActionDecider(
            llm_client=llm_client,
            tool_registry=tool_registry,
            strategy=DecisionStrategy.GOAL_ORIENTED
        )

        assert decider.llm_client == llm_client
        assert decider.tool_registry == tool_registry
        assert decider.strategy == DecisionStrategy.GOAL_ORIENTED

    @pytest.mark.asyncio
    async def test_fallback_decision(self):
        """Test fallback decision when LLM fails."""
        llm_client = Mock()
        tool_registry = Mock()

        decider = LLMActionDecider(llm_client, tool_registry)

        context = DecisionContext(
            task_goal="Test",
            current_state="start",
            recent_actions=[]
        )

        decision = decider._fallback_decision(context)

        assert decision.action == "git_status"
        assert decision.confidence == 0.3

    @pytest.mark.asyncio
    async def test_get_available_tools(self):
        """Test getting available tools."""
        llm_client = Mock()
        tool_registry = Mock()

        mock_tool = MockTool("read_file")
        tool_registry.list_tools = Mock(return_value=[mock_tool])

        decider = LLMActionDecider(llm_client, tool_registry)

        tools = await decider._get_available_tools()

        assert "read_file" in tools


class TestDynamicToolSelector:
    """Test DynamicToolSelector class."""

    @pytest.mark.asyncio
    async def test_tool_selector_initialization(self):
        """Test initializing ToolSelector."""
        tool_registry = Mock()
        selector = DynamicToolSelector(tool_registry)

        assert selector.tool_registry == tool_registry

    @pytest.mark.asyncio
    async def test_select_ut_tools(self):
        """Test selecting UT generation tools."""
        tool_registry = Mock()
        tool_map = {
            "read_file": MockTool("read_file"),
            "grep": MockTool("grep"),
            "bash": MockTool("bash"),
            "git_status": MockTool("git_status")
        }

        selector = DynamicToolSelector(tool_registry)
        tools = selector._select_ut_tools(tool_map)

        assert "read_file" in tools
        assert "grep" in tools

    @pytest.mark.asyncio
    async def test_select_refactor_tools(self):
        """Test selecting refactoring tools."""
        tool_registry = Mock()
        tool_map = {
            "read_file": MockTool("read_file"),
            "edit_tool": MockTool("edit_tool"),
            "git_diff": MockTool("git_diff")
        }

        selector = DynamicToolSelector(tool_registry)
        tools = selector._select_refactor_tools(tool_map)

        assert "read_file" in tools
        assert "edit_tool" in tools


class TestLLMDrivenAutonomousLoop:
    """Test LLMDrivenAutonomousLoop class."""

    @pytest.mark.asyncio
    async def test_loop_initialization(self):
        """Test initializing autonomous loop."""
        llm_client = Mock()
        tool_registry = Mock()

        loop = LLMDrivenAutonomousLoop(
            llm_client=llm_client,
            tool_registry=tool_registry,
            max_iterations=10
        )

        assert loop.llm_client == llm_client
        assert loop.tool_registry == tool_registry
        assert loop.max_iterations == 10

    @pytest.mark.asyncio
    async def test_analyze_state(self):
        """Test state analysis."""
        llm_client = Mock()
        tool_registry = Mock()

        loop = LLMDrivenAutonomousLoop(llm_client, tool_registry)

        result_success = ToolResult(success=True, output="OK")
        state = loop._analyze_state(result_success)
        assert state == "action_succeeded"

        result_failure = ToolResult(success=False, error="Error")
        state = loop._analyze_state(result_failure)
        assert state == "action_failed"

    @pytest.mark.asyncio
    async def test_is_task_complete(self):
        """Test task completion check."""
        llm_client = Mock()
        tool_registry = Mock()

        loop = LLMDrivenAutonomousLoop(llm_client, tool_registry)

        result_success = ToolResult(success=True, output="OK")
        assert loop._is_task_complete("test", result_success) is True

        result_failure = ToolResult(success=False, error="Error")
        assert loop._is_task_complete("test", result_failure) is False


class TestCreateLLMDrivenLoop:
    """Test factory function."""

    @pytest.mark.asyncio
    async def test_create_llm_driven_loop(self):
        """Test creating LLM-driven loop."""
        llm_client = Mock()
        tool_registry = Mock()

        loop = create_llm_driven_loop(
            llm_client=llm_client,
            tool_registry=tool_registry,
            strategy=DecisionStrategy.EXPLORATION
        )

        assert isinstance(loop, LLMDrivenAutonomousLoop)
        assert loop.strategy == DecisionStrategy.EXPLORATION
