"""Integration tests for SubAgent coordination.

Tests end-to-end scenarios:
- Task delegation flow
- Multi-SubAgent parallel execution
- Conflict detection and resolution
- Exception recovery
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from pyutagent.agent.subagents import (
    SubAgentConfig,
    Task,
    TaskPriority,
    AgentCapability,
    AgentStatus,
    SubAgentManager
)
from pyutagent.agent.delegating_subagent import (
    DelegatingSubAgent,
    DelegationContext,
    DelegationMode,
    create_delegating_subagent
)
from pyutagent.agent.subagent_factory import (
    SubAgentFactory,
    AgentType,
    create_subagent_factory
)
from pyutagent.agent.hierarchical_planner import (
    HierarchicalTaskPlanner,
    Subtask,
    SubtaskType,
    SubtaskStatus,
    create_hierarchical_planner
)
from pyutagent.agent.task_router import (
    IntelligentTaskRouter,
    RoutingStrategy,
    create_task_router
)
from pyutagent.agent.conflict_resolver import (
    ConflictResolver,
    ConflictType,
    ResolutionStrategy,
    create_conflict_resolver
)
from pyutagent.agent.delegation_mixin import (
    AgentDelegationMixin,
    DelegationOptions,
    DelegationMode as MixinDelegationMode
)
from pyutagent.agent.shared_context import (
    SharedContextManager,
    create_shared_context_manager
)
from pyutagent.agent.result_aggregator import (
    ResultAggregator,
    AggregationStrategy,
    create_result_aggregator
)
from pyutagent.agent.subagent_orchestrator import (
    SubAgentOrchestrator,
    OrchestrationMode,
    create_subagent_orchestrator
)


class TestEndToEndDelegation:
    """End-to-end delegation tests."""

    @pytest.mark.asyncio
    async def test_simple_delegation_flow(self):
        """Test simple task delegation flow."""
        factory = create_subagent_factory()

        agent = factory.create_agent("generic")

        task = Task(
            id=str(uuid4()),
            name="simple_task",
            description="A simple task",
            input_data={"param": "value"}
        )

        with patch.object(agent, '_execute_with_strategy', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"success": True, "result": "completed"}

            result = await agent.delegate(task)

            assert result.success is True
            assert result.task_id == task.id

    @pytest.mark.asyncio
    async def test_delegation_with_skill(self):
        """Test delegation with skill binding."""
        factory = create_subagent_factory()

        skill = MagicMock()
        skill.name = "test_skill"
        skill.metadata = MagicMock()
        skill.metadata.triggers = ["test"]
        skill.metadata.tags = ["testing"]
        skill.metadata.requires = []

        agent = factory.create_from_skill(skill)

        task = Task(
            id=str(uuid4()),
            name="test_task",
            description="Test task",
            input_data={"skill_name": "test_skill"}
        )

        with patch.object(agent, 'execute_with_skill', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"success": True, "result": "done"}

            result = await agent.delegate(task)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_delegation_with_context(self):
        """Test delegation with shared context."""
        context_manager = create_shared_context_manager()
        factory = create_subagent_factory()

        parent_context = context_manager.create_context("main", "parent_agent")

        agent = factory.create_agent("generic")
        child_context = context_manager.create_context("parent_agent", agent.id)

        context_manager.update_context("parent_agent", {"shared_data": "value"})

        assert context_manager.get_value(agent.id, "shared_data") == "value"


class TestMultiAgentCoordination:
    """Tests for multi-agent coordination."""

    @pytest.mark.asyncio
    async def test_parallel_delegation(self):
        """Test parallel task delegation."""
        factory = create_subagent_factory()
        router = create_task_router()

        agents = [factory.create_agent("generic") for _ in range(3)]

        for agent in agents:
            router.register_agent(agent)

        tasks = [
            Task(
                id=f"task_{i}",
                name=f"parallel_task_{i}",
                description=f"Parallel task {i}",
                input_data={"index": i}
            )
            for i in range(3)
        ]

        for agent in agents:
            with patch.object(agent, '_execute_with_strategy', new_callable=AsyncMock) as mock:
                mock.return_value = {"success": True, "result": f"result_{i}"}

        results = await asyncio.gather(
            *[agent.delegate(task) for agent, task in zip(agents, tasks)]
        )

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_task_routing(self):
        """Test intelligent task routing."""
        factory = create_subagent_factory()
        router = create_task_router()

        test_agent = factory.create_agent("test_generator")
        code_agent = factory.create_agent("code_reviewer")

        router.register_agent(test_agent)
        router.register_agent(code_agent)

        test_task = Task(
            id=str(uuid4()),
            name="Generate unit tests",
            description="Generate unit tests for the code",
            input_data={}
        )

        with patch.object(test_agent, 'is_available', True):
            with patch.object(code_agent, 'is_available', True):
                selected = router.route(test_task, [test_agent, code_agent])

                assert selected is not None

    @pytest.mark.asyncio
    async def test_conflict_detection_and_resolution(self):
        """Test conflict detection and resolution."""
        resolver = create_conflict_resolver()

        tasks = [
            {
                "id": "task1",
                "name": "Write to file",
                "resources": [{"resource_id": "output.py", "access_mode": "write"}]
            },
            {
                "id": "task2",
                "name": "Write to same file",
                "resources": [{"resource_id": "output.py", "access_mode": "write"}]
            }
        ]

        conflicts = resolver.detect_conflicts(tasks)

        assert len(conflicts) > 0

        resolution = await resolver.resolve(conflicts[0])

        assert resolution.success is True


class TestHierarchicalTaskExecution:
    """Tests for hierarchical task execution."""

    @pytest.mark.asyncio
    async def test_task_decomposition(self):
        """Test task decomposition."""
        planner = create_hierarchical_planner()

        tree = await planner.decompose("Generate unit tests for the authentication module")

        assert tree is not None
        assert len(tree.subtasks) > 0

        for subtask in tree.get_all_subtasks():
            assert subtask.name is not None
            assert subtask.task_type is not None

    @pytest.mark.asyncio
    async def test_dependency_analysis(self):
        """Test dependency analysis."""
        planner = create_hierarchical_planner()

        subtasks = [
            Subtask.create(
                name="analyze",
                description="Analyze code",
                task_type=SubtaskType.ANALYSIS
            ),
            Subtask.create(
                name="generate",
                description="Generate tests",
                task_type=SubtaskType.GENERATION,
                dependencies=["analyze"]
            ),
            Subtask.create(
                name="run",
                description="Run tests",
                task_type=SubtaskType.TESTING,
                dependencies=["generate"]
            )
        ]

        graph = planner.analyze_dependencies(subtasks)

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    @pytest.mark.asyncio
    async def test_execution_plan_creation(self):
        """Test execution plan creation."""
        planner = create_hierarchical_planner()

        tree = await planner.decompose("Fix the bug in the login function")

        plan = planner.optimize_execution_order(tree)

        assert plan is not None
        assert len(plan.execution_order) > 0
        assert len(plan.parallel_groups) > 0


class TestOrchestration:
    """Tests for SubAgent orchestration."""

    @pytest.mark.asyncio
    async def test_orchestrate_sequential(self):
        """Test sequential orchestration."""
        orchestrator = create_subagent_orchestrator(max_parallel_tasks=1)

        planner = create_hierarchical_planner()
        tree = await planner.decompose("Simple task")
        plan = planner.optimize_execution_order(tree)

        with patch.object(orchestrator, '_execute_single_task', new_callable=AsyncMock) as mock:
            mock.return_value = {"success": True, "output": "done"}

            result = await orchestrator.orchestrate(plan, OrchestrationMode.SEQUENTIAL)

            assert result is not None

    @pytest.mark.asyncio
    async def test_orchestrate_parallel(self):
        """Test parallel orchestration."""
        orchestrator = create_subagent_orchestrator(max_parallel_tasks=5)

        planner = create_hierarchical_planner()
        tree = await planner.decompose("Generate tests and documentation")
        plan = planner.optimize_execution_order(tree)

        with patch.object(orchestrator, '_execute_single_task', new_callable=AsyncMock) as mock:
            mock.return_value = {"success": True, "output": "done"}

            result = await orchestrator.orchestrate(plan, OrchestrationMode.PARALLEL)

            assert result is not None

    @pytest.mark.asyncio
    async def test_orchestration_stop(self):
        """Test stopping orchestration."""
        orchestrator = create_subagent_orchestrator()

        planner = create_hierarchical_planner()
        tree = await planner.decompose("Task")
        plan = planner.optimize_execution_order(tree)

        orchestrator.stop()

        assert orchestrator._stop_requested is True


class TestResultAggregation:
    """Tests for result aggregation."""

    @pytest.mark.asyncio
    async def test_aggregate_successful_results(self):
        """Test aggregating successful results."""
        aggregator = create_result_aggregator()

        results = [
            {"success": True, "result": {"files": ["a.py"]}},
            {"success": True, "result": {"files": ["b.py"]}},
            {"success": True, "result": {"files": ["c.py"]}}
        ]

        aggregated = await aggregator.aggregate(results)

        assert aggregated.success is True
        assert aggregated.total_tasks == 3
        assert aggregated.successful_tasks == 3

    @pytest.mark.asyncio
    async def test_aggregate_with_failures(self):
        """Test aggregating with some failures."""
        aggregator = create_result_aggregator()

        results = [
            {"success": True, "result": "ok"},
            {"success": False, "error": "failed"},
            {"success": True, "result": "ok"}
        ]

        aggregated = await aggregator.aggregate(results)

        assert aggregated.success is False
        assert aggregated.failed_tasks == 1

    @pytest.mark.asyncio
    async def test_inconsistency_detection(self):
        """Test inconsistency detection in results."""
        aggregator = create_result_aggregator()

        results = [
            {"success": True, "result": {"count": 5}},
            {"success": True, "result": {"count": 10}},
            {"success": True, "result": {"count": 5}}
        ]

        aggregated = await aggregator.aggregate(results)

        assert len(aggregated.inconsistencies) > 0


class TestContextSharing:
    """Tests for context sharing between agents."""

    def test_context_inheritance(self):
        """Test context inheritance from parent."""
        manager = create_shared_context_manager()

        manager.create_context("main", "parent")
        manager.update_context("parent", {"config": {"timeout": 30}})

        manager.create_context("parent", "child")

        assert manager.get_value("child", "config") == {"timeout": 30}

    def test_context_isolation(self):
        """Test context isolation."""
        manager = create_shared_context_manager()

        manager.create_context("main", "agent1")
        manager.create_context("main", "agent2")

        manager.update_context("agent1", {"private": "data"})

        assert manager.get_value("agent2", "private") is None

    def test_context_snapshot_and_restore(self):
        """Test context snapshot and restore."""
        manager = create_shared_context_manager()

        manager.create_context("main", "agent")
        manager.update_context("agent", {"state": "initial"})

        snapshot = manager.create_snapshot("agent")

        manager.update_context("agent", {"state": "modified"})
        assert manager.get_value("agent", "state") == "modified"

        manager.restore_snapshot(snapshot.snapshot_id)
        assert manager.get_value("agent", "state") == "initial"


class TestDelegationMixin:
    """Tests for delegation mixin."""

    @pytest.mark.asyncio
    async def test_delegate_subtask(self):
        """Test delegating a subtask via mixin."""
        mixin = AgentDelegationMixin()
        mixin.init_delegation()

        subtask = Subtask.create(
            name="test_subtask",
            description="Test subtask",
            task_type=SubtaskType.ANALYSIS,
            input_data={"param": "value"}
        )

        with patch.object(mixin._subagent_manager, 'delegate_task', new_callable=AsyncMock) as mock:
            mock_task = MagicMock()
            mock_task.error = None
            mock_task.result = {"success": True}
            mock.return_value = mock_task

            result = await mixin.delegate_subtask(subtask)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_parallel_delegation_via_mixin(self):
        """Test parallel delegation via mixin."""
        mixin = AgentDelegationMixin()
        mixin.init_delegation()

        subtasks = [
            Subtask.create(
                name=f"subtask_{i}",
                description=f"Subtask {i}",
                task_type=SubtaskType.ANALYSIS
            )
            for i in range(3)
        ]

        with patch.object(mixin, 'delegate_subtask', new_callable=AsyncMock) as mock:
            from pyutagent.agent.delegating_subagent import DelegationResult
            mock.return_value = DelegationResult(
                success=True,
                task_id="test",
                agent_id="agent"
            )

            results = await mixin.delegate_parallel(subtasks)

            assert len(results) == 3


class TestFullWorkflow:
    """Full workflow integration tests."""

    @pytest.mark.asyncio
    async def test_complete_delegation_workflow(self):
        """Test complete delegation workflow from task to result."""
        factory = create_subagent_factory()
        planner = create_hierarchical_planner()
        router = create_task_router()
        resolver = create_conflict_resolver()
        aggregator = create_result_aggregator()

        tree = await planner.decompose("Generate unit tests for the user service")

        assert len(tree.subtasks) > 0

        plan = planner.optimize_execution_order(tree)

        assert len(plan.execution_order) > 0

        agents = [factory.create_agent("generic") for _ in range(min(3, len(tree.subtasks)))]

        for agent in agents:
            router.register_agent(agent)

        results = []
        for subtask in tree.get_all_subtasks()[:3]:
            task = Task(
                id=subtask.id,
                name=subtask.name,
                description=subtask.description,
                input_data=subtask.input_data
            )

            with patch.object(agents[0], '_execute_with_strategy', new_callable=AsyncMock) as mock:
                mock.return_value = {"success": True, "result": "completed"}
                result = await agents[0].delegate(task)
                results.append({
                    "success": result.success,
                    "result": result.output
                })

        aggregated = await aggregator.aggregate(results)

        assert aggregated is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
