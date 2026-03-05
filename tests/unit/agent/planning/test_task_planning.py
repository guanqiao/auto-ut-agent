"""Unit tests for Task Planning module."""

import pytest
from unittest.mock import MagicMock

from pyutagent.agent.planning.decomposer import (
    TaskDecomposer,
    SimpleTaskDecomposer,
    LLMTaskDecomposer,
    TemplateTaskDecomposer,
    DecompositionStrategy,
    DecompositionContext,
    get_task_decomposer,
)
from pyutagent.agent.planning.dependency_analyzer import (
    DependencyAnalyzer,
    AdvancedDependencyAnalyzer,
    DependencyGraph,
    DependencyNode,
)
from pyutagent.agent.planning.parallel_executor import (
    ParallelExecutionEngine,
    ParallelExecutionConfig,
    SubTaskResult,
    ResourceType,
    ResourcePool,
)
from pyutagent.agent.execution.execution_plan import Step, StepStatus, StepType


class TestDecompositionStrategy:
    """Tests for DecompositionStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert DecompositionStrategy.ATOMIC.value == "atomic"
        assert DecompositionStrategy.SEQUENTIAL.value == "sequential"
        assert DecompositionStrategy.COMPOSITE.value == "composite"
        assert DecompositionStrategy.PARALLEL.value == "parallel"
        assert DecompositionStrategy.CONDITIONAL.value == "conditional"
        assert DecompositionStrategy.TEMPLATE_BASED.value == "template"


class TestDecompositionContext:
    """Tests for DecompositionContext."""

    def test_create_context(self):
        """Test creating decomposition context."""
        context = DecompositionContext(
            task_description="Test Task",
            complexity=5,
            available_tools=["tool1"],
            available_skills=["skill1"],
        )
        
        assert context.task_description == "Test Task"
        assert context.available_tools == ["tool1"]
        assert context.available_skills == ["skill1"]


class TestSimpleTaskDecomposer:
    """Tests for SimpleTaskDecomposer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decomposer = SimpleTaskDecomposer()

    def test_get_strategy_atomic(self):
        """Test strategy for atomic task."""
        strategy = self.decomposer.get_strategy(complexity=1)
        assert strategy == DecompositionStrategy.ATOMIC

    def test_get_strategy_sequential(self):
        """Test strategy for sequential task."""
        strategy = self.decomposer.get_strategy(complexity=3)
        assert strategy == DecompositionStrategy.SEQUENTIAL

    def test_get_strategy_composite(self):
        """Test strategy for composite task."""
        strategy = self.decomposer.get_strategy(complexity=6)
        assert strategy == DecompositionStrategy.COMPOSITE

    def test_decompose_atomic(self):
        """Test decomposing atomic task."""
        context = DecompositionContext(
            task_description="Simple task",
            complexity=1,
            available_tools=[],
            available_skills=[],
        )
        
        steps = self.decomposer.decompose(context)
        
        assert len(steps) == 1

    def test_decompose_sequential(self):
        """Test decomposing sequential task."""
        context = DecompositionContext(
            task_description="Step 1\nStep 2\nStep 3",
            complexity=3,
            available_tools=[],
            available_skills=[],
        )
        
        steps = self.decomposer.decompose(context)
        
        assert len(steps) == 3


class TestTemplateTaskDecomposer:
    """Tests for TemplateTaskDecomposer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decomposer = TemplateTaskDecomposer()

    def test_load_templates(self):
        """Test templates are loaded."""
        templates = self.decomposer._templates
        
        assert "test_generation" in templates
        assert "code_refactoring" in templates

    def test_decompose_test_generation(self):
        """Test decomposing test generation task."""
        context = DecompositionContext(
            task_description="Generate unit tests",
            complexity=5,
            task_type="test_generation",
            available_tools=["java_parser", "llm_client", "maven_tools"],
            available_skills=[],
        )
        
        steps = self.decomposer.decompose(context)
        
        assert len(steps) > 0
        assert steps[0].name == "Analyze target class"


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    def setup_method(self):
        """Set up test fixtures."""
        self.graph = DependencyGraph()

    def test_add_step(self):
        """Test adding a step."""
        step = Step(
            id="step_1",
            name="Test",
            description="Test",
            step_type=StepType.ANALYZE,
        )
        
        self.graph.add_step(step)
        
        assert "step_1" in self.graph._nodes

    def test_add_dependency(self):
        """Test adding a dependency."""
        st1 = Step(id="step_1", name="A", description="", step_type=StepType.ANALYZE)
        st2 = Step(id="step_2", name="B", description="", step_type=StepType.GENERATE)
        
        self.graph.add_step(st1)
        self.graph.add_step(st2)
        self.graph.add_dependency("step_1", "step_2")
        
        assert "step_1" in self.graph._nodes["step_2"].dependencies
        assert "step_2" in self.graph._nodes["step_1"].dependents

    def test_get_dependencies(self):
        """Test getting dependencies."""
        st1 = Step(id="step_1", name="A", description="", step_type=StepType.ANALYZE)
        st2 = Step(id="step_2", name="B", description="", step_type=StepType.GENERATE)
        
        self.graph.add_step(st1)
        self.graph.add_step(st2)
        self.graph.add_dependency("step_1", "step_2")
        
        deps = self.graph.get_dependencies("step_2")
        
        assert "step_1" in deps

    def test_get_dependents(self):
        """Test getting dependents."""
        st1 = Step(id="step_1", name="A", description="", step_type=StepType.ANALYZE)
        st2 = Step(id="step_2", name="B", description="", step_type=StepType.GENERATE)
        
        self.graph.add_step(st1)
        self.graph.add_step(st2)
        self.graph.add_dependency("step_1", "step_2")
        
        dependents = self.graph.get_dependents("step_1")
        
        assert "step_2" in dependents

    def test_topological_sort(self):
        """Test topological sort."""
        st1 = Step(id="step_1", name="A", description="", step_type=StepType.ANALYZE)
        st2 = Step(id="step_2", name="B", description="", step_type=StepType.GENERATE)
        st3 = Step(id="step_3", name="C", description="", step_type=StepType.TEST)
        
        self.graph.add_step(st1)
        self.graph.add_step(st2)
        self.graph.add_step(st3)
        self.graph.add_dependency("step_1", "step_2")
        self.graph.add_dependency("step_2", "step_3")
        
        order = self.graph.topological_sort()
        
        assert order == ["step_1", "step_2", "step_3"]

    def test_get_execution_groups(self):
        """Test getting execution groups."""
        st1 = Step(id="step_1", name="A", description="", step_type=StepType.ANALYZE)
        st2 = Step(id="step_2", name="B", description="", step_type=StepType.GENERATE)
        st3 = Step(id="step_3", name="C", description="", step_type=StepType.TEST)
        st4 = Step(id="step_4", name="D", description="", step_type=StepType.COMPILE)
        
        self.graph.add_step(st1)
        self.graph.add_step(st2)
        self.graph.add_step(st3)
        self.graph.add_step(st4)
        self.graph.add_dependency("step_1", "step_3")
        
        groups = self.graph.get_execution_groups()
        
        assert len(groups) >= 1

    def test_detect_cycles(self):
        """Test cycle detection."""
        st1 = Step(id="step_1", name="A", description="", step_type=StepType.ANALYZE)
        st2 = Step(id="step_2", name="B", description="", step_type=StepType.GENERATE)
        
        self.graph.add_step(st1)
        self.graph.add_step(st2)
        self.graph.add_dependency("step_1", "step_2")
        self.graph.add_dependency("step_2", "step_1")
        
        cycles = self.graph.detect_cycles()
        
        assert len(cycles) > 0


class TestAdvancedDependencyAnalyzer:
    """Tests for AdvancedDependencyAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = AdvancedDependencyAnalyzer()

    def test_analyze(self):
        """Test analyzing steps."""
        st1 = Step(id="step_1", name="A", description="", step_type=StepType.ANALYZE)
        st2 = Step(id="step_2", name="B", description="", step_type=StepType.GENERATE, dependencies=["step_1"])
        
        graph = self.analyzer.analyze([st1, st2])
        
        assert "step_1" in graph._nodes
        assert "step_2" in graph._nodes

    def test_detect_resource_conflicts(self):
        """Test detecting resource conflicts."""
        st1 = Step(id="step_1", name="A", description="", step_type=StepType.ANALYZE)
        st2 = Step(id="step_2", name="B", description="", step_type=StepType.GENERATE)
        
        graph = self.analyzer.analyze([st1, st2])
        conflicts = self.analyzer.detect_resource_conflicts(graph)
        
        assert isinstance(conflicts, list)


class TestResourcePool:
    """Tests for ResourcePool."""

    def test_create_pool(self):
        """Test creating a resource pool."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            max_concurrent=4,
        )
        
        assert pool.resource_type == ResourceType.CPU
        assert pool.max_concurrent == 4
        assert pool.current_usage == 0

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test acquiring and releasing resources."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            max_concurrent=2,
        )
        
        acquired = await pool.acquire(1)
        assert acquired is True
        
        await pool.release(1)
        assert pool.current_usage == 0

    @pytest.mark.asyncio
    async def test_acquire_exceeds_limit(self):
        """Test acquiring exceeds limit."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            max_concurrent=1,
        )
        
        acquired = await pool.acquire(1)
        assert acquired is True
        
        acquired = await pool.acquire(1)
        assert acquired is False


class TestParallelExecutionEngine:
    """Tests for ParallelExecutionEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ParallelExecutionConfig(max_concurrent_tasks=2)
        self.engine = ParallelExecutionEngine(self.config)

    @pytest.mark.asyncio
    async def test_execute_empty(self):
        """Test executing empty task list."""
        results = await self.engine.execute([], lambda x: x)
        
        assert results == {}

    @pytest.mark.asyncio
    async def test_execute_single_task(self):
        """Test executing a single task."""
        st1 = Step(
            id="step_1",
            name="Test",
            description="Test",
            step_type=StepType.ANALYZE,
        )
        
        async def executor_func(step):
            return {"result": "ok"}
        
        results = await self.engine.execute([st1], executor_func)
        
        assert "step_1" in results


class TestGetTaskDecomposer:
    """Tests for get_task_decomposer function."""

    def test_get_simple_decomposer(self):
        """Test getting SimpleTaskDecomposer."""
        decomposer = get_task_decomposer(use_templates=False, llm_client=None)
        
        assert isinstance(decomposer, SimpleTaskDecomposer)

    def test_get_template_decomposer(self):
        """Test getting TemplateTaskDecomposer."""
        decomposer = get_task_decomposer(use_templates=True, llm_client=None)
        
        assert isinstance(decomposer, TemplateTaskDecomposer)

    def test_get_llm_decomposer(self):
        """Test getting LLMTaskDecomposer."""
        mock_client = MagicMock()
        decomposer = get_task_decomposer(llm_client=mock_client)
        
        assert isinstance(decomposer, LLMTaskDecomposer)
