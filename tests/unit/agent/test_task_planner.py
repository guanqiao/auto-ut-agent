"""Tests for task planner module."""

import pytest
from datetime import datetime

from pyutagent.agent.task_understanding import (
    TaskUnderstanding,
    TaskType,
    TaskPriority,
    TaskComplexity,
    TargetScope,
)
from pyutagent.agent.task_planner import (
    SubTaskStatus,
    SubTaskType,
    SubTask,
    ExecutionPlan,
    TaskPlanner,
    PlanExecutor,
)


class TestSubTask:
    """Tests for SubTask."""

    def test_subtask_creation(self):
        subtask = SubTask(
            id="st_1",
            description="Read target file",
            task_type=SubTaskType.READ,
        )
        
        assert subtask.id == "st_1"
        assert subtask.status == SubTaskStatus.PENDING
        assert subtask.retry_count == 0

    def test_mark_started(self):
        subtask = SubTask(
            id="st_1",
            description="Test",
            task_type=SubTaskType.ANALYZE,
        )
        
        subtask.mark_started()
        
        assert subtask.status == SubTaskStatus.IN_PROGRESS
        assert subtask.started_at is not None

    def test_mark_completed(self):
        subtask = SubTask(
            id="st_1",
            description="Test",
            task_type=SubTaskType.ANALYZE,
        )
        
        subtask.mark_started()
        subtask.mark_completed({"result": "success"})
        
        assert subtask.status == SubTaskStatus.COMPLETED
        assert subtask.result == {"result": "success"}
        assert subtask.completed_at is not None

    def test_mark_failed(self):
        subtask = SubTask(
            id="st_1",
            description="Test",
            task_type=SubTaskType.ANALYZE,
        )
        
        subtask.mark_started()
        subtask.mark_failed("Something went wrong")
        
        assert subtask.status == SubTaskStatus.FAILED
        assert subtask.error == "Something went wrong"

    def test_can_retry(self):
        subtask = SubTask(
            id="st_1",
            description="Test",
            task_type=SubTaskType.ANALYZE,
            max_retries=3,
        )
        
        assert subtask.can_retry()
        
        subtask.retry_count = 3
        assert not subtask.can_retry()

    def test_increment_retry(self):
        subtask = SubTask(
            id="st_1",
            description="Test",
            task_type=SubTaskType.ANALYZE,
        )
        
        subtask.status = SubTaskStatus.FAILED
        subtask.error = "Error"
        
        subtask.increment_retry()
        
        assert subtask.retry_count == 1
        assert subtask.status == SubTaskStatus.PENDING
        assert subtask.error is None

    def test_to_dict(self):
        subtask = SubTask(
            id="st_1",
            description="Read file",
            task_type=SubTaskType.READ,
            dependencies=["st_0"],
            priority=8,
        )
        
        result = subtask.to_dict()
        
        assert result["id"] == "st_1"
        assert result["task_type"] == "read"
        assert result["dependencies"] == ["st_0"]
        assert result["priority"] == 8


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    @pytest.fixture
    def sample_understanding(self):
        return TaskUnderstanding(
            task_type=TaskType.UT_GENERATION,
            original_request="Generate tests",
            requirements="Generate unit tests",
        )

    @pytest.fixture
    def sample_plan(self, sample_understanding):
        subtasks = [
            SubTask(id="st_1", description="Step 1", task_type=SubTaskType.ANALYZE),
            SubTask(id="st_2", description="Step 2", task_type=SubTaskType.READ, dependencies=["st_1"]),
            SubTask(id="st_3", description="Step 3", task_type=SubTaskType.GENERATE, dependencies=["st_2"]),
        ]
        
        return ExecutionPlan(
            id="plan_1",
            task_understanding=sample_understanding,
            subtasks=subtasks,
        )

    def test_plan_creation(self, sample_plan):
        assert sample_plan.id == "plan_1"
        assert len(sample_plan.subtasks) == 3

    def test_get_subtask(self, sample_plan):
        subtask = sample_plan.get_subtask("st_2")
        assert subtask is not None
        assert subtask.description == "Step 2"

    def test_get_subtask_not_found(self, sample_plan):
        subtask = sample_plan.get_subtask("st_999")
        assert subtask is None

    def test_get_pending_subtasks(self, sample_plan):
        pending = sample_plan.get_pending_subtasks()
        assert len(pending) == 3

    def test_get_ready_subtasks(self, sample_plan):
        ready = sample_plan.get_ready_subtasks()
        assert len(ready) == 1
        assert ready[0].id == "st_1"

    def test_get_ready_subtasks_with_completed_deps(self, sample_plan):
        sample_plan.subtasks[0].status = SubTaskStatus.COMPLETED
        
        ready = sample_plan.get_ready_subtasks()
        assert len(ready) == 1
        assert ready[0].id == "st_2"

    def test_get_progress(self, sample_plan):
        progress = sample_plan.get_progress()
        
        assert progress["total"] == 3
        assert progress["completed"] == 0
        assert progress["pending"] == 3

    def test_get_progress_partial(self, sample_plan):
        sample_plan.subtasks[0].status = SubTaskStatus.COMPLETED
        sample_plan.subtasks[1].status = SubTaskStatus.IN_PROGRESS
        
        progress = sample_plan.get_progress()
        
        assert progress["completed"] == 1
        assert progress["in_progress"] == 1
        assert progress["pending"] == 1

    def test_is_complete(self, sample_plan):
        assert not sample_plan.is_complete()
        
        for st in sample_plan.subtasks:
            st.status = SubTaskStatus.COMPLETED
        
        assert sample_plan.is_complete()

    def test_is_successful(self, sample_plan):
        for st in sample_plan.subtasks:
            st.status = SubTaskStatus.COMPLETED
        
        assert sample_plan.is_successful()

    def test_is_successful_with_failure(self, sample_plan):
        sample_plan.subtasks[0].status = SubTaskStatus.COMPLETED
        sample_plan.subtasks[1].status = SubTaskStatus.FAILED
        sample_plan.subtasks[2].status = SubTaskStatus.COMPLETED
        
        assert sample_plan.is_complete()
        assert not sample_plan.is_successful()

    def test_to_dict(self, sample_plan):
        result = sample_plan.to_dict()
        
        assert result["id"] == "plan_1"
        assert "subtasks" in result
        assert "task_understanding" in result


class TestTaskPlanner:
    """Tests for TaskPlanner."""

    @pytest.fixture
    def planner(self):
        return TaskPlanner()

    @pytest.fixture
    def sample_understanding(self):
        return TaskUnderstanding(
            task_type=TaskType.UT_GENERATION,
            original_request="Generate tests for UserService.java",
            requirements="Generate comprehensive unit tests",
            target_scope=TargetScope(files=["UserService.java"]),
        )

    def test_create_plan_from_template(self, planner, sample_understanding):
        plan = planner.create_plan(sample_understanding, use_llm=False)
        
        assert plan is not None
        assert len(plan.subtasks) > 0
        assert plan.task_understanding.task_type == TaskType.UT_GENERATION

    def test_plan_has_correct_subtask_types(self, planner, sample_understanding):
        plan = planner.create_plan(sample_understanding, use_llm=False)
        
        task_types = [st.task_type for st in plan.subtasks]
        
        assert SubTaskType.ANALYZE in task_types
        assert SubTaskType.GENERATE in task_types

    def test_plan_dependencies_chain(self, planner, sample_understanding):
        plan = planner.create_plan(sample_understanding, use_llm=False)
        
        for i, subtask in enumerate(plan.subtasks):
            if i > 0:
                assert len(subtask.dependencies) > 0

    def test_plan_for_bug_fix(self, planner):
        understanding = TaskUnderstanding(
            task_type=TaskType.BUG_FIX,
            original_request="Fix null pointer exception",
            requirements="Fix the NPE in login method",
        )
        
        plan = planner.create_plan(understanding, use_llm=False)
        
        task_types = [st.task_type for st in plan.subtasks]
        assert SubTaskType.ANALYZE in task_types
        assert SubTaskType.EDIT in task_types or SubTaskType.SEARCH in task_types

    def test_plan_for_code_review(self, planner):
        understanding = TaskUnderstanding(
            task_type=TaskType.CODE_REVIEW,
            original_request="Review this code",
            requirements="Review code for quality",
        )
        
        plan = planner.create_plan(understanding, use_llm=False)
        
        task_types = [st.task_type for st in plan.subtasks]
        assert SubTaskType.READ in task_types
        assert SubTaskType.REVIEW in task_types or SubTaskType.ANALYZE in task_types

    def test_refine_plan_with_failed_subtask(self, planner, sample_understanding):
        plan = planner.create_plan(sample_understanding, use_llm=False)
        
        plan.subtasks[0].status = SubTaskStatus.COMPLETED
        plan.subtasks[1].status = SubTaskStatus.FAILED
        plan.subtasks[1].error = "Compilation error"
        
        feedback = {
            "error_type": "compilation_error",
        }
        
        refined_plan = planner.refine_plan(plan, feedback)
        
        assert refined_plan is not None


class TestPlanExecutor:
    """Tests for PlanExecutor."""

    @pytest.fixture
    def executor(self):
        return PlanExecutor()

    @pytest.fixture
    def sample_plan(self):
        understanding = TaskUnderstanding(
            task_type=TaskType.UT_GENERATION,
            original_request="Test",
            requirements="Test",
        )
        
        subtasks = [
            SubTask(id="st_1", description="Analyze", task_type=SubTaskType.ANALYZE),
        ]
        
        return ExecutionPlan(
            id="plan_1",
            task_understanding=understanding,
            subtasks=subtasks,
        )

    def test_executor_creation(self, executor):
        assert executor is not None
        assert executor.tool_registry is not None

    def test_executor_with_tools(self):
        tools = {
            "analyze": lambda **kwargs: {"result": "analyzed"},
        }
        
        executor = PlanExecutor(tool_registry=tools)
        
        assert "analyze" in executor.tool_registry
