"""Tests for Universal Task Planner - 通用任务规划器测试"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from pyutagent.agent.universal_planner import (
    TaskType,
    TaskUnderstanding,
    Subtask,
    ExecutionPlan,
    SubtaskResult,
    ExecutionResult,
    TaskHandler,
    UniversalTaskPlanner
)


class MockTaskHandler(TaskHandler):
    """Mock task handler for testing"""
    
    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.handle_calls = []
    
    async def handle(self, subtask: Subtask, context: Dict[str, Any]) -> SubtaskResult:
        self.handle_calls.append((subtask, context))
        return SubtaskResult(
            subtask_id=subtask.id,
            success=self.should_succeed,
            data={'mock': 'data'}
        )
    
    def can_handle(self, task_type: TaskType) -> bool:
        return True


class TestTaskType:
    """Test TaskType enum"""
    
    def test_task_type_values(self):
        """Test task type enum values"""
        assert TaskType.TEST_GENERATION.value == "test_generation"
        assert TaskType.CODE_REFACTORING.value == "code_refactoring"
        assert TaskType.BUG_FIX.value == "bug_fix"
        assert TaskType.FEATURE_ADD.value == "feature_add"


class TestTaskUnderstanding:
    """Test TaskUnderstanding dataclass"""
    
    def test_task_understanding_creation(self):
        """Test creating TaskUnderstanding"""
        understanding = TaskUnderstanding(
            task_type=TaskType.TEST_GENERATION,
            description="Generate tests for UserService",
            target_files=["UserService.java"],
            constraints=["Use JUnit 5"],
            success_criteria=["Tests compile and pass"],
            estimated_complexity=3
        )
        
        assert understanding.task_type == TaskType.TEST_GENERATION
        assert understanding.description == "Generate tests for UserService"
        assert understanding.target_files == ["UserService.java"]
        assert understanding.estimated_complexity == 3
    
    def test_task_understanding_defaults(self):
        """Test TaskUnderstanding default values"""
        understanding = TaskUnderstanding(
            task_type=TaskType.QUERY,
            description="Simple query"
        )
        
        assert understanding.target_files == []
        assert understanding.constraints == []
        assert understanding.success_criteria == []
        assert understanding.estimated_complexity == 3


class TestSubtask:
    """Test Subtask dataclass"""
    
    def test_subtask_creation(self):
        """Test creating Subtask"""
        subtask = Subtask(
            id="analyze_target",
            description="Analyze target class",
            task_type=TaskType.QUERY,
            dependencies=[],
            tools_needed=["file_read"],
            estimated_complexity=2,
            success_criteria="Get class structure"
        )
        
        assert subtask.id == "analyze_target"
        assert subtask.description == "Analyze target class"
        assert subtask.max_retries == 3
        assert subtask.timeout_seconds == 300


class TestExecutionPlan:
    """Test ExecutionPlan dataclass"""
    
    def test_execution_plan_creation(self):
        """Test creating ExecutionPlan"""
        understanding = TaskUnderstanding(
            task_type=TaskType.TEST_GENERATION,
            description="Generate tests"
        )
        
        subtasks = [
            Subtask(id="st1", description="Step 1", task_type=TaskType.QUERY),
            Subtask(id="st2", description="Step 2", task_type=TaskType.TEST_GENERATION)
        ]
        
        plan = ExecutionPlan(
            task_id="task_123",
            original_request="Generate tests for UserService",
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["st1", "st2"]
        )
        
        assert plan.task_id == "task_123"
        assert len(plan.subtasks) == 2
        assert plan.execution_order == ["st1", "st2"]


class TestUniversalTaskPlanner:
    """Test UniversalTaskPlanner"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client"""
        llm = Mock()
        llm.generate = AsyncMock(return_value='''{
            "task_type": "test_generation",
            "description": "Generate unit tests",
            "target_files": ["UserService.java"],
            "constraints": [],
            "success_criteria": ["Tests pass"],
            "estimated_complexity": 3,
            "context_requirements": []
        }''')
        return llm
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create mock project analyzer"""
        return Mock()
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry"""
        return Mock()
    
    @pytest.fixture
    def planner(self, mock_llm, mock_analyzer, mock_tool_registry):
        """Create UniversalTaskPlanner instance"""
        return UniversalTaskPlanner(mock_llm, mock_analyzer, mock_tool_registry)
    
    @pytest.mark.asyncio
    async def test_understand_task(self, planner, mock_llm):
        """Test task understanding"""
        user_request = "Generate tests for UserService"
        project_context = {
            'language': 'java',
            'build_tool': 'maven',
            'structure': {'src': 'src/main/java'}
        }
        
        understanding = await planner.understand_task(user_request, project_context)
        
        assert understanding.task_type == TaskType.TEST_GENERATION
        assert understanding.description == "Generate unit tests"
        assert understanding.target_files == ["UserService.java"]
        assert understanding.estimated_complexity == 3
        
        # Verify LLM was called
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args[0][0]
        assert "Generate tests for UserService" in call_args
        assert "java" in call_args
    
    @pytest.mark.asyncio
    async def test_understand_task_fallback(self, planner, mock_llm):
        """Test task understanding fallback on error"""
        mock_llm.generate.side_effect = Exception("LLM error")
        
        user_request = "Some request"
        project_context = {}
        
        understanding = await planner.understand_task(user_request, project_context)
        
        # Should fallback to QUERY type
        assert understanding.task_type == TaskType.QUERY
        assert understanding.description == "Some request"
    
    @pytest.mark.asyncio
    async def test_decompose_test_generation_task(self, planner):
        """Test decomposing test generation task"""
        understanding = TaskUnderstanding(
            task_type=TaskType.TEST_GENERATION,
            description="Generate tests for UserService",
            target_files=["UserService.java"]
        )
        project_context = {}
        
        plan = await planner.decompose_task(understanding, project_context)
        
        assert plan.task_id.startswith("task_")
        assert plan.original_request == "Generate tests for UserService"
        assert len(plan.subtasks) == 6
        
        # Check subtask IDs
        subtask_ids = [s.id for s in plan.subtasks]
        assert "analyze_target" in subtask_ids
        assert "generate_test" in subtask_ids
        assert "compile_test" in subtask_ids
        assert "run_test" in subtask_ids
        assert "fix_issues" in subtask_ids
    
    @pytest.mark.asyncio
    async def test_decompose_refactoring_task(self, planner):
        """Test decomposing refactoring task"""
        understanding = TaskUnderstanding(
            task_type=TaskType.CODE_REFACTORING,
            description="Refactor UserService"
        )
        project_context = {}
        
        plan = await planner.decompose_task(understanding, project_context)
        
        assert len(plan.subtasks) == 5
        subtask_ids = [s.id for s in plan.subtasks]
        assert "analyze_impact" in subtask_ids
        assert "backup_code" in subtask_ids
        assert "execute_refactoring" in subtask_ids
    
    @pytest.mark.asyncio
    async def test_decompose_bug_fix_task(self, planner):
        """Test decomposing bug fix task"""
        understanding = TaskUnderstanding(
            task_type=TaskType.BUG_FIX,
            description="Fix NPE in OrderProcessor"
        )
        project_context = {}
        
        plan = await planner.decompose_task(understanding, project_context)
        
        assert len(plan.subtasks) == 5
        subtask_ids = [s.id for s in plan.subtasks]
        assert "reproduce_bug" in subtask_ids
        assert "analyze_root_cause" in subtask_ids
        assert "implement_fix" in subtask_ids
    
    @pytest.mark.asyncio
    async def test_execute_with_feedback_success(self, planner):
        """Test successful execution with feedback"""
        # Create a simple plan
        understanding = TaskUnderstanding(
            task_type=TaskType.QUERY,
            description="Simple task"
        )
        
        subtasks = [
            Subtask(id="st1", description="Step 1", task_type=TaskType.QUERY),
            Subtask(id="st2", description="Step 2", task_type=TaskType.QUERY)
        ]
        
        plan = ExecutionPlan(
            task_id="task_123",
            original_request="Simple task",
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["st1", "st2"]
        )
        
        # Register mock handler
        mock_handler = MockTaskHandler(should_succeed=True)
        planner.register_task_handler(TaskType.QUERY, mock_handler)
        
        # Execute
        result = await planner.execute_with_feedback(plan, {})
        
        assert result.success is True
        assert len(result.subtask_results) == 2
        assert len(result.completed_subtasks) == 2
        assert len(result.failed_subtasks) == 0
        assert result.execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_execute_with_feedback_failure(self, planner):
        """Test execution with failure"""
        understanding = TaskUnderstanding(
            task_type=TaskType.QUERY,
            description="Failing task"
        )
        
        subtasks = [
            Subtask(id="st1", description="Step 1", task_type=TaskType.QUERY)
        ]
        
        plan = ExecutionPlan(
            task_id="task_123",
            original_request="Failing task",
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["st1"]
        )
        
        # Register failing handler
        mock_handler = MockTaskHandler(should_succeed=False)
        planner.register_task_handler(TaskType.QUERY, mock_handler)
        
        # Execute
        result = await planner.execute_with_feedback(plan, {})
        
        assert result.success is False
        assert len(result.failed_subtasks) == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_progress_callback(self, planner):
        """Test execution with progress callback"""
        understanding = TaskUnderstanding(
            task_type=TaskType.QUERY,
            description="Task with callback"
        )
        
        subtasks = [
            Subtask(id="st1", description="Step 1", task_type=TaskType.QUERY)
        ]
        
        plan = ExecutionPlan(
            task_id="task_123",
            original_request="Task with callback",
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["st1"]
        )
        
        # Register handler
        mock_handler = MockTaskHandler(should_succeed=True)
        planner.register_task_handler(TaskType.QUERY, mock_handler)
        
        # Track callbacks
        callbacks = []
        async def progress_callback(subtask, result):
            callbacks.append((subtask.id, result.success))
        
        # Execute
        await planner.execute_with_feedback(plan, {}, progress_callback)
        
        assert len(callbacks) == 1
        assert callbacks[0] == ("st1", True)
    
    def test_register_task_handler(self, planner):
        """Test registering task handler"""
        handler = MockTaskHandler()
        
        planner.register_task_handler(TaskType.TEST_GENERATION, handler)
        
        assert TaskType.TEST_GENERATION in planner._task_handlers
        assert planner._task_handlers[TaskType.TEST_GENERATION] == handler
    
    def test_get_statistics_empty(self, planner):
        """Test getting statistics with no executions"""
        stats = planner.get_statistics()
        
        assert stats['total_executions'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_execution_time'] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_executions(self, planner):
        """Test getting statistics after executions"""
        # Create and execute a plan
        understanding = TaskUnderstanding(
            task_type=TaskType.QUERY,
            description="Test task"
        )
        
        subtasks = [Subtask(id="st1", description="Step 1", task_type=TaskType.QUERY)]
        
        plan = ExecutionPlan(
            task_id="task_123",
            original_request="Test",
            understanding=understanding,
            subtasks=subtasks,
            execution_order=["st1"]
        )
        
        mock_handler = MockTaskHandler(should_succeed=True)
        planner.register_task_handler(TaskType.QUERY, mock_handler)
        
        await planner.execute_with_feedback(plan, {})
        
        stats = planner.get_statistics()
        
        assert stats['total_executions'] == 1
        assert stats['successful_executions'] == 1
        assert stats['success_rate'] == 1.0
        assert stats['average_execution_time'] >= 0
    
    def test_get_execution_history(self, planner):
        """Test getting execution history"""
        history = planner.get_execution_history()
        assert history == []


class TestSubtaskDependencies:
    """Test subtask dependency handling"""
    
    @pytest.fixture
    def planner(self):
        """Create planner with mocked dependencies"""
        llm = Mock()
        analyzer = Mock()
        tools = Mock()
        return UniversalTaskPlanner(llm, analyzer, tools)
    
    @pytest.mark.asyncio
    async def test_dependency_order(self, planner):
        """Test that dependencies are respected in execution order"""
        understanding = TaskUnderstanding(
            task_type=TaskType.TEST_GENERATION,
            description="Test dependencies"
        )
        
        project_context = {}
        plan = await planner.decompose_task(understanding, project_context)
        
        # Check that all dependencies are executed before dependent tasks
        executed = set()
        for subtask_id in plan.execution_order:
            subtask = next((s for s in plan.subtasks if s.id == subtask_id), None)
            if subtask:
                # All dependencies should already be executed
                for dep in subtask.dependencies:
                    assert dep in executed, f"Dependency {dep} not executed before {subtask_id}"
                executed.add(subtask_id)
    
    @pytest.mark.asyncio
    async def test_test_generation_dependencies(self, planner):
        """Test test generation task dependencies"""
        understanding = TaskUnderstanding(
            task_type=TaskType.TEST_GENERATION,
            description="Generate tests",
            target_files=["UserService.java"]
        )
        
        plan = await planner.decompose_task(understanding, {})
        
        # Find subtasks
        analyze_target = next(s for s in plan.subtasks if s.id == "analyze_target")
        analyze_deps = next(s for s in plan.subtasks if s.id == "analyze_dependencies")
        generate_test = next(s for s in plan.subtasks if s.id == "generate_test")
        
        # Check dependencies
        assert analyze_target.dependencies == []
        assert "analyze_target" in analyze_deps.dependencies
        assert "analyze_target" in generate_test.dependencies
        assert "analyze_dependencies" in generate_test.dependencies


class TestTaskDecompositionStrategies:
    """Test different task decomposition strategies"""
    
    @pytest.fixture
    def planner(self):
        """Create planner"""
        llm = Mock()
        analyzer = Mock()
        tools = Mock()
        return UniversalTaskPlanner(llm, analyzer, tools)
    
    @pytest.mark.asyncio
    async def test_feature_add_decomposition(self, planner):
        """Test feature add task decomposition"""
        understanding = TaskUnderstanding(
            task_type=TaskType.FEATURE_ADD,
            description="Add new payment feature"
        )
        
        plan = await planner.decompose_task(understanding, {})
        
        subtask_ids = [s.id for s in plan.subtasks]
        assert "analyze_requirements" in subtask_ids
        assert "design_solution" in subtask_ids
        assert "implement_feature" in subtask_ids
        assert "write_tests" in subtask_ids
        assert "verify_feature" in subtask_ids
    
    @pytest.mark.asyncio
    async def test_code_review_decomposition(self, planner):
        """Test code review task decomposition"""
        understanding = TaskUnderstanding(
            task_type=TaskType.CODE_REVIEW,
            description="Review UserService"
        )
        
        plan = await planner.decompose_task(understanding, {})
        
        subtask_ids = [s.id for s in plan.subtasks]
        assert "read_code" in subtask_ids
        assert "analyze_quality" in subtask_ids
        assert "generate_report" in subtask_ids
    
    @pytest.mark.asyncio
    async def test_exploration_decomposition(self, planner):
        """Test exploration task decomposition"""
        understanding = TaskUnderstanding(
            task_type=TaskType.EXPLORATION,
            description="Explore codebase structure"
        )
        
        plan = await planner.decompose_task(understanding, {})
        
        subtask_ids = [s.id for s in plan.subtasks]
        assert "explore_structure" in subtask_ids
        assert "analyze_dependencies" in subtask_ids
        assert "summarize_findings" in subtask_ids
    
    @pytest.mark.asyncio
    async def test_planning_decomposition(self, planner):
        """Test planning task decomposition"""
        understanding = TaskUnderstanding(
            task_type=TaskType.PLANNING,
            description="Design new architecture"
        )
        
        plan = await planner.decompose_task(understanding, {})
        
        subtask_ids = [s.id for s in plan.subtasks]
        assert "gather_requirements" in subtask_ids
        assert "research_solutions" in subtask_ids
        assert "create_design_doc" in subtask_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
