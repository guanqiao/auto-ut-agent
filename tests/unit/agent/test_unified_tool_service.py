"""Tests for UnifiedToolService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from pyutagent.agent.unified_tool_service import (
    UnifiedToolService,
    ToolServiceConfig,
    ToolCall,
    ToolState,
    ExecutionPlan,
    PlanState,
    OrchestrationResult,
    DependencyGraph,
    create_unified_tool_service,
)
from pyutagent.tools.tool import Tool, ToolDefinition, ToolCategory, ToolResult, ToolParameter


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name: str, output: str = "test output"):
        self._name = name
        self._output = output
        self.call_count = 0
        self._definition = ToolDefinition(
            name=name,
            description=f"Mock tool {name}",
            category=ToolCategory.FILE,
            parameters=[]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        self.call_count += 1
        return ToolResult(success=True, output=self._output)


class TestToolServiceConfig:
    """Test ToolServiceConfig."""

    def test_default_config(self):
        config = ToolServiceConfig()
        assert config.timeout_seconds == 60
        assert config.max_retries == 3
        assert config.enable_cache is True

    def test_custom_config(self):
        config = ToolServiceConfig(
            timeout_seconds=120,
            max_retries=5,
            enable_cache=False
        )
        assert config.timeout_seconds == 120
        assert config.max_retries == 5
        assert config.enable_cache is False


class TestToolCall:
    """Test ToolCall dataclass."""

    def test_default_values(self):
        call = ToolCall(tool_name="test_tool")
        assert call.tool_name == "test_tool"
        assert call.parameters == {}
        assert call.state == ToolState.PENDING
        assert call.result is None
        assert call.attempt == 0

    def test_duration_ms(self):
        from datetime import datetime, timedelta
        call = ToolCall(
            tool_name="test",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1)
        )
        assert call.duration_ms == 1000


class TestExecutionPlan:
    """Test ExecutionPlan."""

    def test_default_values(self):
        plan = ExecutionPlan(id="test-plan", goal="test goal", steps=[])
        assert plan.id == "test-plan"
        assert plan.goal == "test goal"
        assert plan.state == PlanState.CREATED
        assert plan.progress == 0.0

    def test_progress_calculation(self):
        steps = [
            ToolCall(tool_name="tool1", state=ToolState.COMPLETED),
            ToolCall(tool_name="tool2", state=ToolState.PENDING),
            ToolCall(tool_name="tool3", state=ToolState.FAILED),
        ]
        plan = ExecutionPlan(id="test", goal="test", steps=steps)
        assert plan.progress == pytest.approx(2/3)
        assert plan.completed_steps == 1
        assert plan.failed_steps == 1


class TestDependencyGraph:
    """Test DependencyGraph."""

    def test_add_tool(self):
        graph = DependencyGraph()
        graph.add_tool("tool1", dependencies=["tool2"])
        assert "tool1" in graph.nodes
        assert "tool2" in graph.nodes["tool1"]

    def test_get_execution_order(self):
        graph = DependencyGraph()
        graph.add_tool("tool3", dependencies=["tool2"])
        graph.add_tool("tool2", dependencies=["tool1"])
        graph.add_tool("tool1")

        order = graph.get_execution_order(["tool1", "tool2", "tool3"])
        assert order.index("tool1") < order.index("tool2")
        assert order.index("tool2") < order.index("tool3")

    def test_has_cycle(self):
        graph = DependencyGraph()
        graph.add_tool("tool1", dependencies=["tool2"])
        graph.add_tool("tool2", dependencies=["tool1"])
        assert graph.has_cycle() is True

    def test_no_cycle(self):
        graph = DependencyGraph()
        graph.add_tool("tool1")
        graph.add_tool("tool2", dependencies=["tool1"])
        assert graph.has_cycle() is False


class TestUnifiedToolService:
    """Test UnifiedToolService."""

    def test_init(self):
        service = UnifiedToolService()
        assert service.registry is not None
        assert service.config is not None

    def test_init_with_config(self):
        config = ToolServiceConfig(timeout_seconds=120)
        service = UnifiedToolService(config=config)
        assert service.config.timeout_seconds == 120

    def test_register_tool(self):
        service = UnifiedToolService()
        tool = MockTool("test_tool")
        service.register_tool(tool)

        assert service.get_tool("test_tool") is not None
        assert "test_tool" in service.list_tools()

    def test_unregister_tool(self):
        service = UnifiedToolService()
        tool = MockTool("test_tool")
        service.register_tool(tool)

        result = service.unregister_tool("test_tool")
        assert result is True
        assert service.get_tool("test_tool") is None

    def test_register_function(self):
        service = UnifiedToolService()

        def test_func(arg1: str) -> str:
            return f"result: {arg1}"

        service.register_function("test_func", test_func)
        assert "test_func" in service._tool_functions

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        service = UnifiedToolService()
        tool = MockTool("test_tool", "hello world")
        service.register_tool(tool)

        result = await service.execute_tool("test_tool", {})
        assert result.success is True
        assert result.output == "hello world"
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        service = UnifiedToolService()

        result = await service.execute_tool("nonexistent", {})
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_with_cache(self):
        service = UnifiedToolService(config=ToolServiceConfig(enable_cache=True))
        tool = MockTool("cached_tool", "cached output")
        service.register_tool(tool)

        result1 = await service.execute_tool("cached_tool", {})
        result2 = await service.execute_tool("cached_tool", {})

        assert result1.success is True
        assert result2.success is True
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_tools_parallel(self):
        service = UnifiedToolService()
        tool1 = MockTool("tool1", "output1")
        tool2 = MockTool("tool2", "output2")
        service.register_tool(tool1)
        service.register_tool(tool2)

        calls = [
            {"tool_name": "tool1", "parameters": {}},
            {"tool_name": "tool2", "parameters": {}}
        ]

        results = await service.execute_tools_parallel(calls)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_tools_sequence(self):
        service = UnifiedToolService()
        tool1 = MockTool("tool1", "output1")
        tool2 = MockTool("tool2", "output2")
        service.register_tool(tool1)
        service.register_tool(tool2)

        calls = [
            {"tool_name": "tool1", "parameters": {}},
            {"tool_name": "tool2", "parameters": {}}
        ]

        results = await service.execute_tools_sequence(calls)
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_create_plan(self):
        service = UnifiedToolService()

        tool_sequence = [
            {"tool_name": "tool1", "parameters": {"arg": "value"}, "reasoning": "test"}
        ]

        plan = service.create_plan("test goal", tool_sequence)
        assert plan.goal == "test goal"
        assert len(plan.steps) == 1
        assert plan.state == PlanState.CREATED

    @pytest.mark.asyncio
    async def test_execute_plan(self):
        service = UnifiedToolService()
        tool = MockTool("test_tool", "result")
        service.register_tool(tool)

        tool_sequence = [{"tool_name": "test_tool", "parameters": {}}]
        plan = service.create_plan("test goal", tool_sequence)

        result = await service.execute_plan(plan)
        assert result.success is True
        assert plan.state == PlanState.COMPLETED

    def test_cancel_plan(self):
        service = UnifiedToolService()
        plan = service.create_plan("test", [{"tool_name": "tool1", "parameters": {}}])

        result = service.cancel_plan(plan.id)
        assert result is True
        assert plan.state == PlanState.CANCELLED

    def test_get_stats(self):
        service = UnifiedToolService()
        tool = MockTool("test_tool")
        service.register_tool(tool)

        stats = service.get_stats()
        assert "total_tools" in stats
        assert stats["total_tools"] == 1

    def test_clear_cache(self):
        service = UnifiedToolService()
        service._cache["test_key"] = ("value", 0)

        service.clear_cache()
        assert len(service._cache) == 0


class TestCreateUnifiedToolService:
    """Test factory function."""

    def test_create_default(self):
        service = create_unified_tool_service()
        assert isinstance(service, UnifiedToolService)

    def test_create_with_options(self):
        service = create_unified_tool_service(
            timeout_seconds=120,
            max_retries=5,
            enable_cache=False
        )
        assert service.config.timeout_seconds == 120
        assert service.config.max_retries == 5
        assert service.config.enable_cache is False
