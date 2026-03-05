"""Tests for enhanced ToolOrchestrator.

This module tests:
- plan_from_goal() method
- Dynamic tool chain planning
- Tool execution result reasoning
- Tool memory integration
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from pyutagent.agent.tool_orchestrator import (
    ToolOrchestrator,
    ToolDefinition,
    ToolCall,
    ExecutionPlan,
    ToolChainStep,
    ToolChainPlan,
    ToolState,
    PlanState,
    DependencyGraph,
)
from pyutagent.memory.tool_memory import ToolMemory, ToolRecommendation


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    response = Mock()
    response.content = '''[
        {"tool_name": "parse_code", "parameters": {"file_path": "test.java"}, "reasoning": "Parse the code first"},
        {"tool_name": "generate_tests", "parameters": {"class_info": "$context.class_info"}, "reasoning": "Generate tests"}
    ]'''
    client.chat = AsyncMock(return_value=response)
    return client


@pytest.fixture
def sample_tool_definitions():
    """Create sample tool definitions."""
    return {
        "parse_code": ToolDefinition(
            name="parse_code",
            description="Parse source code to extract structure",
            category="analysis",
            dependencies=[],
            can_run_parallel=True,
            provided_outputs=["class_info", "source_code"]
        ),
        "generate_tests": ToolDefinition(
            name="generate_tests",
            description="Generate unit tests",
            category="generation",
            dependencies=["parse_code"],
            can_run_parallel=False,
            required_inputs=["class_info"],
            provided_outputs=["test_code", "test_file"]
        ),
        "compile_tests": ToolDefinition(
            name="compile_tests",
            description="Compile generated tests",
            category="build",
            dependencies=["generate_tests"],
            can_run_parallel=False,
            required_inputs=["test_file"],
            provided_outputs=["compilation_result"]
        ),
        "run_tests": ToolDefinition(
            name="run_tests",
            description="Run compiled tests",
            category="execution",
            dependencies=["compile_tests"],
            can_run_parallel=False,
            required_inputs=["test_file"],
            provided_outputs=["test_results", "failures"]
        ),
    }


@pytest.fixture
def orchestrator(sample_tool_definitions):
    """Create a ToolOrchestrator instance."""
    return ToolOrchestrator(tool_definitions=sample_tool_definitions)


class TestPlanFromGoal:
    """Tests for plan_from_goal method."""
    
    def test_plan_from_goal_with_rules(self, orchestrator):
        """Test rule-based planning."""
        goal = "Generate tests for my code"
        context = {"source_file": "Test.java"}
        
        plan = orchestrator.plan_from_goal(goal, context, use_llm=False)
        
        assert plan is not None
        assert plan.goal == goal
        assert len(plan.steps) > 0
        assert plan.state == PlanState.CREATED
        assert plan.metadata["planning_method"] == "rule_based"
    
    def test_plan_from_goal_with_test_goal(self, orchestrator):
        """Test planning for test generation goal."""
        goal = "generate unit tests for Calculator class"
        
        plan = orchestrator.plan_from_goal(goal, use_llm=False)
        
        tool_names = [step.tool_name for step in plan.steps]
        assert "parse_code" in tool_names
        assert "generate_tests" in tool_names
    
    def test_plan_from_goal_with_llm(self, orchestrator, mock_llm_client):
        """Test LLM-based planning."""
        orchestrator.set_llm_client(mock_llm_client)
        
        goal = "Analyze and test my code"
        plan = orchestrator.plan_from_goal(goal, use_llm=True)
        
        assert plan is not None
        assert plan.metadata["planning_method"] == "llm"
        mock_llm_client.chat.assert_called_once()
    
    def test_plan_from_goal_fallback_to_rules(self, orchestrator, mock_llm_client):
        """Test fallback to rule-based when LLM fails."""
        mock_llm_client.chat.side_effect = Exception("LLM error")
        orchestrator.set_llm_client(mock_llm_client)
        
        goal = "Generate tests"
        plan = orchestrator.plan_from_goal(goal, use_llm=True)
        
        assert plan is not None
        assert len(plan.steps) > 0


class TestDynamicToolChain:
    """Tests for dynamic tool chain planning."""
    
    def test_create_tool_chain_for_tests(self, orchestrator):
        """Test creating tool chain for test generation."""
        goal = "Generate comprehensive tests"
        
        chain = orchestrator.create_tool_chain(goal)
        
        assert isinstance(chain, ToolChainPlan)
        assert chain.goal == goal
        assert len(chain.steps) > 0
        
        step_names = [step.tool_name for step in chain.steps]
        assert "parse_code" in step_names
        assert "generate_tests" in step_names
    
    def test_create_tool_chain_with_fallback(self, orchestrator):
        """Test tool chain includes fallback steps."""
        goal = "Generate tests"
        
        chain = orchestrator.create_tool_chain(goal)
        
        assert len(chain.fallback_steps) > 0
        fallback_names = [step.tool_name for step in chain.fallback_steps]
        assert "fix_compilation" in fallback_names or "fix_tests" in fallback_names
    
    def test_create_tool_chain_for_analysis(self, orchestrator):
        """Test tool chain for code analysis."""
        goal = "Analyze code structure"
        
        chain = orchestrator.create_tool_chain(goal)
        
        step_names = [step.tool_name for step in chain.steps]
        assert "parse_code" in step_names
    
    def test_identify_parallel_groups(self, orchestrator):
        """Test identifying parallel tool groups."""
        steps = [
            ToolChainStep(tool_name="parse_code", parameters={}, reasoning="Parse"),
            ToolChainStep(tool_name="generate_tests", parameters={}, reasoning="Generate"),
        ]
        
        groups = orchestrator._identify_parallel_groups(steps)
        
        assert isinstance(groups, list)
        # parse_code can run parallel, generate_tests cannot
        assert len(groups) >= 1


class TestToolExecutionReasoning:
    """Tests for tool execution result reasoning."""
    
    @pytest.mark.asyncio
    async def test_reason_about_parse_code_success(self, orchestrator):
        """Test reasoning after successful parse_code."""
        plan = ExecutionPlan(
            id="test-plan",
            goal="Generate tests",
            steps=[
                ToolCall(tool_name="parse_code", state=ToolState.COMPLETED),
                ToolCall(tool_name="generate_tests", state=ToolState.PENDING)
            ]
        )
        result = {"class_info": {"name": "Calculator"}, "success": True}
        
        reasoning = await orchestrator.reason_about_result(
            "parse_code", result, "Generate tests", plan
        )
        
        assert reasoning["goal_achieved"] is False
        assert "generate_tests" in reasoning["next_tools"]
        assert reasoning["confidence"] > 0
    
    @pytest.mark.asyncio
    async def test_reason_about_run_tests_success(self, orchestrator):
        """Test reasoning when all tests pass."""
        plan = ExecutionPlan(
            id="test-plan",
            goal="Generate and run tests",
            steps=[
                ToolCall(tool_name="run_tests", state=ToolState.COMPLETED)
            ]
        )
        result = {"all_passed": True, "passed": 10, "failed": 0}
        
        reasoning = await orchestrator.reason_about_result(
            "run_tests", result, "Generate and run tests", plan
        )
        
        assert reasoning["goal_achieved"] is True
        assert reasoning["confidence"] > 0.9
    
    @pytest.mark.asyncio
    async def test_reason_about_compilation_failure(self, orchestrator):
        """Test reasoning when compilation fails."""
        plan = ExecutionPlan(
            id="test-plan",
            goal="Generate tests",
            steps=[
                ToolCall(tool_name="compile_tests", state=ToolState.FAILED)
            ]
        )
        result = {"success": False, "errors": ["Syntax error"]}
        
        reasoning = await orchestrator.reason_about_result(
            "compile_tests", result, "Generate tests", plan
        )
        
        assert reasoning["should_adapt"] is True
        assert "fix_compilation" in reasoning["next_tools"]
    
    @pytest.mark.asyncio
    async def test_reason_with_llm(self, orchestrator, mock_llm_client):
        """Test LLM-based reasoning."""
        orchestrator.set_llm_client(mock_llm_client)
        mock_llm_client.chat.return_value = Mock(content='''{
            "goal_achieved": true,
            "next_tools": [],
            "should_adapt": false,
            "reasoning": "Goal achieved successfully",
            "confidence": 0.95
        }''')
        
        plan = ExecutionPlan(id="test", goal="Test", steps=[])
        
        reasoning = await orchestrator.reason_about_result(
            "parse_code", {"success": True}, "Test", plan
        )
        
        assert reasoning["goal_achieved"] is True
        assert reasoning["confidence"] == 0.95


class TestToolMemoryIntegration:
    """Tests for tool memory integration."""
    
    @pytest.mark.asyncio
    async def test_record_success_in_memory(self, orchestrator):
        """Test recording successful tool execution."""
        mock_memory = AsyncMock()
        orchestrator.set_tool_memory(mock_memory)
        
        plan = ExecutionPlan(id="test", goal="Test", steps=[])
        result = {"success": True}
        
        await orchestrator.reason_about_result("parse_code", result, "Test", plan)
        
        mock_memory.record_success.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_recommended_tools(self, orchestrator):
        """Test getting tool recommendations."""
        mock_memory = AsyncMock()
        mock_memory.get_recommended_tools.return_value = [
            ToolRecommendation(
                tool_name="parse_code",
                reason="Commonly used",
                success_rate=0.95,
                usage_count=10,
                avg_duration=1.0
            )
        ]
        orchestrator.set_tool_memory(mock_memory)
        
        recommendations = await orchestrator.get_recommended_tools_for_goal(
            "Generate tests", limit=5
        )
        
        assert len(recommendations) > 0
        assert recommendations[0]["tool_name"] == "parse_code"
    
    def test_classify_task_type(self, orchestrator):
        """Test task type classification."""
        assert orchestrator._classify_task_type("Generate tests for class") == "test_generation"
        assert orchestrator._classify_task_type("Fix failing tests") == "test_fixing"
        assert orchestrator._classify_task_type("Analyze code coverage") == "coverage_analysis"
        assert orchestrator._classify_task_type("Parse the code") == "code_analysis"
        assert orchestrator._classify_task_type("Do something else") == "general"


class TestParameterResolution:
    """Tests for parameter resolution."""
    
    def test_resolve_simple_parameters(self, orchestrator):
        """Test resolving simple parameters."""
        context = {"file_path": "/path/to/file.java"}
        parameters = {"path": "/path/to/file.java"}
        
        resolved = orchestrator._resolve_parameters(parameters, context)
        
        assert resolved["path"] == "/path/to/file.java"
    
    def test_resolve_context_reference(self, orchestrator):
        """Test resolving context references."""
        context = {"source_file": "Test.java", "class_info": {"name": "Test"}}
        parameters = {"file": "$context.source_file"}
        
        resolved = orchestrator._resolve_parameters(parameters, context)
        
        assert resolved["file"] == "Test.java"
    
    def test_resolve_unknown_reference(self, orchestrator):
        """Test handling unknown references."""
        context = {}
        parameters = {"file": "$unknown.variable"}
        
        resolved = orchestrator._resolve_parameters(parameters, context)
        
        assert resolved["file"] == "unknown.variable"


class TestToolOrchestratorIntegration:
    """Integration tests for ToolOrchestrator."""
    
    def test_full_workflow_without_llm(self, orchestrator):
        """Test complete workflow using rule-based planning."""
        goal = "Generate tests for Calculator class"
        context = {"source_file": "Calculator.java"}
        
        # Create plan
        plan = orchestrator.plan_from_goal(goal, context, use_llm=False)
        assert plan is not None
        assert len(plan.steps) > 0
        
        # Create tool chain
        chain = orchestrator.create_tool_chain(goal)
        assert len(chain.steps) > 0
        
        # Verify plan is tracked
        retrieved_plan = orchestrator.get_plan(plan.id)
        assert retrieved_plan == plan
    
    @pytest.mark.asyncio
    async def test_reasoning_workflow(self, orchestrator):
        """Test reasoning workflow."""
        plan = orchestrator.plan_from_goal("Generate tests", use_llm=False)
        
        # Simulate execution and reasoning
        result = {"class_info": {"name": "Test"}}
        reasoning = await orchestrator.reason_about_result(
            "parse_code", result, "Generate tests", plan
        )
        
        assert "next_tools" in reasoning
        assert "goal_achieved" in reasoning


class TestDependencyGraph:
    """Tests for DependencyGraph."""
    
    def test_add_tool(self):
        """Test adding tools to graph."""
        graph = DependencyGraph()
        tool = ToolDefinition(name="test", description="Test", category="test")
        
        graph.add_tool(tool)
        
        assert "test" in graph.nodes
    
    def test_get_execution_order(self, sample_tool_definitions):
        """Test getting execution order."""
        graph = DependencyGraph()
        for tool in sample_tool_definitions.values():
            graph.add_tool(tool)
        graph.build_dependencies()
        
        levels = graph.get_execution_order()
        
        assert len(levels) > 0
        # parse_code has no dependencies, should be first
        assert "parse_code" in levels[0]
    
    def test_detect_cycle(self):
        """Test cycle detection."""
        graph = DependencyGraph()
        
        tool_a = ToolDefinition(name="a", description="A", category="test", dependencies=["b"])
        tool_b = ToolDefinition(name="b", description="B", category="test", dependencies=["a"])
        
        graph.add_tool(tool_a)
        graph.add_tool(tool_b)
        graph.build_dependencies()
        
        assert graph.has_cycle() is True
    
    def test_no_cycle(self, sample_tool_definitions):
        """Test no cycle in valid dependencies."""
        graph = DependencyGraph()
        for tool in sample_tool_definitions.values():
            graph.add_tool(tool)
        graph.build_dependencies()
        
        assert graph.has_cycle() is False


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_goal(self, orchestrator):
        """Test handling empty goal."""
        plan = orchestrator.plan_from_goal("", use_llm=False)
        assert plan is not None
    
    def test_unknown_goal_type(self, orchestrator):
        """Test handling unknown goal type."""
        plan = orchestrator.plan_from_goal("Do something completely different", use_llm=False)
        assert plan is not None
        assert len(plan.steps) >= 0
    
    def test_cancel_nonexistent_plan(self, orchestrator):
        """Test cancelling non-existent plan."""
        result = orchestrator.cancel_plan("nonexistent-id")
        assert result is False
    
    def test_get_stats_empty_history(self, orchestrator):
        """Test getting stats with no history."""
        stats = orchestrator.get_tool_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0
