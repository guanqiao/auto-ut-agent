"""Unit tests for SubAgent enhancements.

Tests for:
- DelegatingSubAgent
- SubAgentFactory
- HierarchicalTaskPlanner
- IntelligentTaskRouter
- ConflictResolver
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from pyutagent.agent.delegating_subagent import (
    DelegatingSubAgent,
    DelegationContext,
    DelegationMode,
    DelegationResult,
    ProgressUpdate,
    create_delegating_subagent
)
from pyutagent.agent.subagent_factory import (
    SubAgentFactory,
    AgentType,
    AgentTemplate,
    AgentPoolConfig,
    create_subagent_factory
)
from pyutagent.agent.hierarchical_planner import (
    HierarchicalTaskPlanner,
    Subtask,
    SubtaskType,
    SubtaskStatus,
    TaskTree,
    DependencyGraph,
    ExecutionPlan,
    create_hierarchical_planner
)
from pyutagent.agent.task_router import (
    IntelligentTaskRouter,
    RoutingStrategy,
    RoutingDecision,
    create_task_router
)
from pyutagent.agent.conflict_resolver import (
    ConflictResolver,
    ConflictType,
    ConflictSeverity,
    ConflictStatus,
    ResolutionStrategy,
    Conflict,
    create_conflict_resolver
)
from pyutagent.agent.subagents import (
    SubAgentConfig,
    Task,
    TaskPriority,
    AgentCapability,
    AgentStatus
)
from pyutagent.agent.skills import Skill, SkillMetadata, SkillInput, SkillOutput


class TestDelegatingSubAgent:
    """Tests for DelegatingSubAgent."""

    def test_create_delegating_subagent(self):
        """Test creating a DelegatingSubAgent."""
        agent = create_delegating_subagent(
            name="test_agent",
            agent_type="test",
            description="Test agent"
        )

        assert agent is not None
        assert agent.config.name == "test_agent"
        assert agent.config.agent_type == "test"

    def test_bind_skill(self):
        """Test binding a skill to agent."""
        agent = create_delegating_subagent(
            name="test_agent",
            agent_type="test",
            description="Test agent"
        )

        skill = MagicMock(spec=Skill)
        skill.name = "test_skill"

        agent.bind_skill(skill)

        assert "test_skill" in agent.get_bound_skills()
        assert agent.unbind_skill("test_skill") is True
        assert "test_skill" not in agent.get_bound_skills()

    def test_set_callbacks(self):
        """Test setting progress and result callbacks."""
        agent = create_delegating_subagent(
            name="test_agent",
            agent_type="test",
            description="Test agent"
        )

        progress_callback = MagicMock()
        result_callback = MagicMock()

        agent.set_progress_callback(progress_callback)
        agent.set_result_callback(result_callback)

        assert agent._progress_callback == progress_callback
        assert agent._result_callback == result_callback

    def test_delegation_context(self):
        """Test delegation context."""
        context = DelegationContext(
            parent_agent_id="parent_1",
            delegation_mode=DelegationMode.HYBRID,
            timeout=600
        )

        assert context.parent_agent_id == "parent_1"
        assert context.delegation_mode == DelegationMode.HYBRID
        assert context.timeout == 600

    @pytest.mark.asyncio
    async def test_delegate_task(self):
        """Test delegating a task."""
        config = SubAgentConfig(
            name="test_agent",
            agent_type="test",
            description="Test agent"
        )

        agent = DelegatingSubAgent(config=config)
        await agent.initialize()

        task = Task(
            id=str(uuid4()),
            name="test_task",
            description="Test task",
            input_data={"param": "value"}
        )

        with patch.object(agent, '_execute_with_strategy', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"success": True, "result": "done"}

            result = await agent.delegate(task)

            assert result.success is True
            assert result.task_id == task.id


class TestSubAgentFactory:
    """Tests for SubAgentFactory."""

    def test_create_factory(self):
        """Test creating a factory."""
        factory = create_subagent_factory()
        assert factory is not None

    def test_create_agent(self):
        """Test creating an agent."""
        factory = SubAgentFactory()

        agent = factory.create_agent("generic")

        assert agent is not None
        assert agent.config.agent_type == "generic"

    def test_create_agent_from_skill(self):
        """Test creating agent from skill."""
        factory = SubAgentFactory()

        skill = MagicMock(spec=Skill)
        skill.name = "test_skill"
        skill.metadata = SkillMetadata(
            name="test_skill",
            description="Test skill"
        )

        agent = factory.create_from_skill(skill)

        assert agent is not None
        assert "test_skill" in agent.get_bound_skills()

    def test_agent_pool(self):
        """Test agent pooling."""
        factory = SubAgentFactory(pool_config=AgentPoolConfig(max_size=5))

        agent1 = factory.create_agent("generic")
        factory.return_to_pool(agent1)

        agent2 = factory.get_or_create_agent("generic")

        assert agent2.id == agent1.id

    def test_destroy_agent(self):
        """Test destroying an agent."""
        factory = SubAgentFactory()

        agent = factory.create_agent("generic")
        agent_id = agent.id

        result = factory.destroy_agent(agent_id)

        assert result is True
        assert factory.get_agent(agent_id) is None

    def test_get_stats(self):
        """Test getting factory stats."""
        factory = SubAgentFactory()

        factory.create_agent("generic")
        factory.create_agent("test_generator")

        stats = factory.get_stats()

        assert stats["total_agents"] == 2
        assert stats["agents_created"] == 2


class TestHierarchicalTaskPlanner:
    """Tests for HierarchicalTaskPlanner."""

    def test_create_planner(self):
        """Test creating a planner."""
        planner = create_hierarchical_planner()
        assert planner is not None

    def test_create_subtask(self):
        """Test creating a subtask."""
        subtask = Subtask.create(
            name="test_subtask",
            description="Test subtask",
            task_type=SubtaskType.ANALYSIS
        )

        assert subtask.name == "test_subtask"
        assert subtask.task_type == SubtaskType.ANALYSIS
        assert subtask.status == SubtaskStatus.PENDING

    def test_task_tree(self):
        """Test task tree operations."""
        tree = TaskTree(
            root_id="root",
            task_description="Main task"
        )

        subtask1 = Subtask.create(
            name="subtask1",
            description="First subtask",
            task_type=SubtaskType.ANALYSIS
        )

        subtask2 = Subtask.create(
            name="subtask2",
            description="Second subtask",
            task_type=SubtaskType.GENERATION,
            dependencies=[subtask1.id]
        )

        tree.add_subtask(subtask1)
        tree.add_subtask(subtask2)

        assert len(tree.subtasks) == 2
        assert len(tree.get_root_subtasks()) == 2

    def test_dependency_graph(self):
        """Test dependency graph."""
        graph = DependencyGraph()

        graph.add_node("task1")
        graph.add_node("task2")
        graph.add_edge("task1", "task2")

        assert "task1" in graph.nodes
        assert "task2" in graph.nodes
        assert graph.get_dependencies("task2") == ["task1"]
        assert graph.get_dependents("task1") == ["task2"]

    def test_topological_sort(self):
        """Test topological sort."""
        graph = DependencyGraph()

        graph.add_edge("task1", "task2")
        graph.add_edge("task2", "task3")

        order = graph.topological_sort()

        assert order.index("task1") < order.index("task2")
        assert order.index("task2") < order.index("task3")

    def test_identify_parallel_tasks(self):
        """Test identifying parallel tasks."""
        graph = DependencyGraph()

        graph.add_node("task1")
        graph.add_node("task2")
        graph.add_node("task3")
        graph.add_edge("task1", "task3")
        graph.add_edge("task2", "task3")

        planner = HierarchicalTaskPlanner()
        parallel_groups = planner.identify_parallel_tasks(graph)

        assert len(parallel_groups) >= 1
        assert "task1" in parallel_groups[0] or "task2" in parallel_groups[0]

    @pytest.mark.asyncio
    async def test_decompose_task(self):
        """Test task decomposition."""
        planner = HierarchicalTaskPlanner()

        tree = await planner.decompose("Generate unit tests for the code")

        assert tree is not None
        assert len(tree.subtasks) > 0


class TestIntelligentTaskRouter:
    """Tests for IntelligentTaskRouter."""

    def test_create_router(self):
        """Test creating a router."""
        router = create_task_router()
        assert router is not None

    def test_register_agent(self):
        """Test registering an agent."""
        router = IntelligentTaskRouter()

        agent = MagicMock(spec=SubAgent)
        agent.id = "agent_1"
        agent.config = SubAgentConfig(
            name="test",
            agent_type="test",
            description="Test"
        )
        agent.is_available = True

        router.register_agent(agent)

        stats = router.get_routing_stats()
        assert stats["registered_agents"] == 1

    def test_route_task(self):
        """Test routing a task."""
        router = IntelligentTaskRouter()

        agent = MagicMock(spec=SubAgent)
        agent.id = "agent_1"
        agent.config = SubAgentConfig(
            name="test",
            agent_type="test",
            description="Test",
            capabilities=[AgentCapability(name="test_generation", description="Generate tests")]
        )
        agent.is_available = True

        router.register_agent(agent)

        task = Task(
            id=str(uuid4()),
            name="Generate tests",
            description="Generate unit tests",
            input_data={}
        )

        selected = router.route(task, [agent])

        assert selected is not None

    def test_calculate_affinity(self):
        """Test affinity calculation."""
        router = IntelligentTaskRouter()

        agent = MagicMock(spec=SubAgent)
        agent.id = "agent_1"
        agent.config = SubAgentConfig(
            name="test",
            agent_type="test",
            description="Test"
        )

        task = Task(
            id=str(uuid4()),
            name="Test task",
            description="Test description",
            input_data={}
        )

        affinity = router.calculate_affinity(task, agent)

        assert 0.0 <= affinity <= 1.0

    def test_record_routing_result(self):
        """Test recording routing result."""
        router = IntelligentTaskRouter()

        agent = MagicMock(spec=SubAgent)
        agent.id = "agent_1"
        agent.config = SubAgentConfig(
            name="test",
            agent_type="test",
            description="Test"
        )
        agent.is_available = True

        router.register_agent(agent)

        task = Task(
            id="task_1",
            name="Test task",
            description="Test",
            input_data={}
        )

        router.route(task, [agent])
        router.record_routing_result("task_1", "agent_1", True, 100)

        stats = router.get_agent_stats("agent_1")
        assert stats["total_tasks"] == 1
        assert stats["successful_tasks"] == 1


class TestConflictResolver:
    """Tests for ConflictResolver."""

    def test_create_resolver(self):
        """Test creating a resolver."""
        resolver = create_conflict_resolver()
        assert resolver is not None

    def test_detect_resource_conflicts(self):
        """Test detecting resource conflicts."""
        resolver = ConflictResolver()

        tasks = [
            {
                "id": "task1",
                "resources": [{"resource_id": "file.py", "access_mode": "write"}]
            },
            {
                "id": "task2",
                "resources": [{"resource_id": "file.py", "access_mode": "write"}]
            }
        ]

        conflicts = resolver.detect_conflicts(tasks)

        assert len(conflicts) > 0
        assert conflicts[0].conflict_type == ConflictType.RESOURCE

    def test_detect_dependency_conflicts(self):
        """Test detecting dependency cycles."""
        resolver = ConflictResolver()

        tasks = [
            {"id": "task1", "dependencies": ["task2"]},
            {"id": "task2", "dependencies": ["task1"]}
        ]

        conflicts = resolver.detect_conflicts(tasks)

        dependency_conflicts = [c for c in conflicts if c.conflict_type == ConflictType.DEPENDENCY]
        assert len(dependency_conflicts) > 0

    @pytest.mark.asyncio
    async def test_resolve_conflict(self):
        """Test resolving a conflict."""
        resolver = ConflictResolver()

        conflict = Conflict(
            conflict_id=str(uuid4()),
            conflict_type=ConflictType.RESOURCE,
            severity=ConflictSeverity.HIGH,
            description="Test conflict",
            parties=[
                MagicMock(party_id="party1", priority=5),
                MagicMock(party_id="party2", priority=3)
            ]
        )

        resolution = await resolver.resolve(conflict, ResolutionStrategy.PRIORITY_BASED)

        assert resolution is not None
        assert resolution.success is True

    def test_get_stats(self):
        """Test getting resolver stats."""
        resolver = ConflictResolver()

        stats = resolver.get_stats()

        assert "conflicts_detected" in stats
        assert "conflicts_resolved" in stats


class TestResultAggregator:
    """Tests for ResultAggregator."""

    def test_create_aggregator(self):
        """Test creating an aggregator."""
        from pyutagent.agent.result_aggregator import create_result_aggregator
        aggregator = create_result_aggregator()
        assert aggregator is not None

    @pytest.mark.asyncio
    async def test_aggregate_results(self):
        """Test aggregating results."""
        from pyutagent.agent.result_aggregator import (
            ResultAggregator,
            AggregationStrategy
        )

        aggregator = ResultAggregator()

        results = [
            {"success": True, "result": {"value": 1}},
            {"success": True, "result": {"value": 2}}
        ]

        aggregated = await aggregator.aggregate(results)

        assert aggregated.success is True
        assert aggregated.total_tasks == 2

    def test_detect_inconsistencies(self):
        """Test detecting inconsistencies."""
        from pyutagent.agent.result_aggregator import ResultAggregator

        aggregator = ResultAggregator()

        results = [
            {"success": True, "result": {"key": "value1"}},
            {"success": True, "result": {"key": "value2"}}
        ]

        inconsistencies = aggregator.detect_inconsistencies(results)

        assert len(inconsistencies) > 0
        assert inconsistencies[0].inconsistency_type.value == "value_mismatch"

    def test_validate_results(self):
        """Test validating results."""
        from pyutagent.agent.result_aggregator import ResultAggregator

        aggregator = ResultAggregator()

        results = [
            {"success": True, "result": "value1"},
            {"success": False, "result": None, "error": "Failed"}
        ]

        validation = aggregator.validate_results(results)

        assert validation.total_results == 2
        assert validation.valid_results == 1


class TestSharedContextManager:
    """Tests for SharedContextManager."""

    def test_create_context_manager(self):
        """Test creating context manager."""
        from pyutagent.agent.shared_context import create_shared_context_manager
        manager = create_shared_context_manager()
        assert manager is not None

    def test_create_context(self):
        """Test creating agent context."""
        from pyutagent.agent.shared_context import SharedContextManager

        manager = SharedContextManager()

        context = manager.create_context("parent_1", "child_1")

        assert context is not None
        assert context.agent_id == "child_1"
        assert context.parent_id == "parent_1"

    def test_update_context(self):
        """Test updating context."""
        from pyutagent.agent.shared_context import SharedContextManager, ContextScope

        manager = SharedContextManager()
        manager.create_context("parent_1", "child_1")

        result = manager.update_context("child_1", {"key": "value"})

        assert result is True
        assert manager.get_value("child_1", "key") == "value"

    def test_create_snapshot(self):
        """Test creating snapshot."""
        from pyutagent.agent.shared_context import SharedContextManager

        manager = SharedContextManager()
        manager.create_context("parent_1", "child_1")
        manager.update_context("child_1", {"key": "value"})

        snapshot = manager.create_snapshot("child_1", label="test_snapshot")

        assert snapshot is not None
        assert snapshot.label == "test_snapshot"

    def test_restore_snapshot(self):
        """Test restoring snapshot."""
        from pyutagent.agent.shared_context import SharedContextManager

        manager = SharedContextManager()
        manager.create_context("parent_1", "child_1")
        manager.update_context("child_1", {"key": "value1"})

        snapshot = manager.create_snapshot("child_1")

        manager.update_context("child_1", {"key": "value2"})
        assert manager.get_value("child_1", "key") == "value2"

        manager.restore_snapshot(snapshot.snapshot_id)
        assert manager.get_value("child_1", "key") == "value1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
