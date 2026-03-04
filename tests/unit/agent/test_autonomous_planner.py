"""Tests for Autonomous Planner."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from pyutagent.agent.autonomous_planner import (
    TaskType,
    TaskPriority,
    TaskUnderstanding,
    Subtask,
    ExecutionPlan,
    ExecutionFeedback,
    AutonomousPlanner,
    create_autonomous_planner
)


class TestTaskType:
    """Test TaskType enum."""
    
    def test_task_type_values(self):
        """Test task type enum values."""
        assert TaskType.UNIT_TEST_GENERATION.name == "UNIT_TEST_GENERATION"
        assert TaskType.CODE_REFACTORING.name == "CODE_REFACTORING"
        assert TaskType.FEATURE_IMPLEMENTATION.name == "FEATURE_IMPLEMENTATION"
        assert TaskType.BUG_FIXING.name == "BUG_FIXING"


class TestTaskUnderstanding:
    """Test TaskUnderstanding dataclass."""
    
    def test_task_understanding_creation(self):
        """Test creating TaskUnderstanding."""
        understanding = TaskUnderstanding(
            original_request="Generate tests for UserService",
            task_type=TaskType.UNIT_TEST_GENERATION,
            intent="Generate unit tests",
            target_files=["UserService.java"],
            confidence=0.9
        )
        
        assert understanding.original_request == "Generate tests for UserService"
        assert understanding.task_type == TaskType.UNIT_TEST_GENERATION
        assert understanding.confidence == 0.9
        assert understanding.target_files == ["UserService.java"]
    
    def test_task_understanding_defaults(self):
        """Test TaskUnderstanding default values."""
        understanding = TaskUnderstanding(
            original_request="Test",
            task_type=TaskType.UNKNOWN,
            intent="Test intent"
        )
        
        assert understanding.target_files == []
        assert understanding.constraints == []
        assert understanding.confidence == 0.0
        assert isinstance(understanding.timestamp, datetime)


class TestSubtask:
    """Test Subtask dataclass."""
    
    def test_subtask_creation(self):
        """Test creating Subtask."""
        subtask = Subtask(
            id="test_001",
            name="Parse file",
            description="Parse the target file",
            task_type=TaskType.UNIT_TEST_GENERATION,
            priority=TaskPriority.HIGH,
            required_tools=["read_file"]
        )
        
        assert subtask.id == "test_001"
        assert subtask.name == "Parse file"
        assert subtask.completed is False
        assert subtask.result is None


class TestAutonomousPlanner:
    """Test AutonomousPlanner class."""
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        client.chat = AsyncMock()
        return client
    
    @pytest.fixture
    def planner(self, mock_llm_client):
        """Create AutonomousPlanner instance."""
        return AutonomousPlanner(llm_client=mock_llm_client)
    
    def test_planner_initialization(self, planner):
        """Test planner initialization."""
        assert planner.llm_client is not None
        assert planner.max_subtasks == 10
        assert planner.refinement_threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_understand_task_with_pattern(self, planner):
        """Test task understanding with pattern matching."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '''
        {
            "task_type": "UNIT_TEST_GENERATION",
            "intent": "Generate unit tests for UserService",
            "target_files": ["UserService.java"],
            "constraints": [],
            "requirements": ["High coverage"],
            "confidence": 0.95
        }
        '''
        planner.llm_client.chat.return_value = mock_response
        
        understanding = await planner.understand_task(
            "Generate tests for UserService"
        )
        
        assert understanding.task_type == TaskType.UNIT_TEST_GENERATION
        assert "UserService" in understanding.intent
        assert understanding.confidence > 0
    
    @pytest.mark.asyncio
    async def test_understand_task_fallback(self, planner):
        """Test task understanding fallback."""
        # Make LLM fail
        planner.llm_client.chat.side_effect = Exception("LLM error")
        
        understanding = await planner.understand_task(
            "Generate tests for UserService"
        )
        
        # Should fallback to pattern matching
        assert understanding.task_type == TaskType.UNIT_TEST_GENERATION
        assert understanding.confidence == 0.3
    
    def test_classify_by_pattern_test_generation(self, planner):
        """Test pattern classification for test generation."""
        result = planner._classify_by_pattern("Generate unit tests")
        assert result == TaskType.UNIT_TEST_GENERATION
    
    def test_classify_by_pattern_refactoring(self, planner):
        """Test pattern classification for refactoring."""
        result = planner._classify_by_pattern("Refactor this code")
        assert result == TaskType.CODE_REFACTORING
    
    def test_classify_by_pattern_bug_fix(self, planner):
        """Test pattern classification for bug fixing."""
        result = planner._classify_by_pattern("Fix the error in login")
        assert result == TaskType.BUG_FIXING
    
    def test_classify_by_pattern_unknown(self, planner):
        """Test pattern classification for unknown task."""
        result = planner._classify_by_pattern("Do something random")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_decompose_test_generation(self, planner):
        """Test test generation task decomposition."""
        understanding = TaskUnderstanding(
            original_request="Generate tests",
            task_type=TaskType.UNIT_TEST_GENERATION,
            intent="Generate tests for UserService",
            target_files=["UserService.java"]
        )
        
        subtasks = await planner.decompose_task(understanding)
        
        assert len(subtasks) > 0
        assert all(st.task_type == TaskType.UNIT_TEST_GENERATION for st in subtasks)
        
        # Check for expected subtask types
        task_names = [st.name for st in subtasks]
        assert any("Parse" in name for name in task_names)
        assert any("Generate" in name for name in task_names)
    
    @pytest.mark.asyncio
    async def test_decompose_refactoring(self, planner):
        """Test refactoring task decomposition."""
        understanding = TaskUnderstanding(
            original_request="Refactor code",
            task_type=TaskType.CODE_REFACTORING,
            intent="Refactor UserService"
        )
        
        subtasks = await planner.decompose_task(understanding)
        
        assert len(subtasks) > 0
        task_names = [st.name for st in subtasks]
        assert any("Analyze" in name for name in task_names)
        assert any("Apply" in name for name in task_names)
    
    @pytest.mark.asyncio
    async def test_decompose_bug_fixing(self, planner):
        """Test bug fixing task decomposition."""
        understanding = TaskUnderstanding(
            original_request="Fix bug",
            task_type=TaskType.BUG_FIXING,
            intent="Fix login bug"
        )
        
        subtasks = await planner.decompose_task(understanding)
        
        assert len(subtasks) >= 4  # reproduce, analyze, fix, verify
        task_names = [st.name for st in subtasks]
        assert any("Reproduce" in name for name in task_names)
        assert any("Analyze" in name for name in task_names)
        assert any("fix" in name.lower() for name in task_names)
    
    @pytest.mark.asyncio
    async def test_create_plan(self, planner):
        """Test complete plan creation."""
        mock_response = Mock()
        mock_response.content = '''
        {
            "task_type": "UNIT_TEST_GENERATION",
            "intent": "Generate tests",
            "target_files": ["Test.java"],
            "confidence": 0.9
        }
        '''
        planner.llm_client.chat.return_value = mock_response
        
        plan = await planner.create_plan("Generate tests for Test.java")
        
        assert plan.task_id is not None
        assert plan.understanding is not None
        assert len(plan.subtasks) > 0
        assert plan.understanding.task_type == TaskType.UNIT_TEST_GENERATION
    
    @pytest.mark.asyncio
    async def test_refine_plan_success(self, planner):
        """Test plan refinement with successful feedback."""
        # Create initial plan
        subtask = Subtask(
            id="test_001",
            name="Test subtask",
            description="Test",
            task_type=TaskType.UNIT_TEST_GENERATION,
            priority=TaskPriority.HIGH
        )
        
        plan = ExecutionPlan(
            task_id="plan_001",
            understanding=TaskUnderstanding(
                original_request="Test",
                task_type=TaskType.UNIT_TEST_GENERATION,
                intent="Test"
            ),
            subtasks=[subtask]
        )
        
        feedback = ExecutionFeedback(
            subtask_id="test_001",
            success=True,
            message="Completed"
        )
        
        refined = await planner.refine_plan(plan, feedback)
        
        assert refined.subtasks[0].completed is True
    
    @pytest.mark.asyncio
    async def test_refine_plan_failure(self, planner):
        """Test plan refinement with failed feedback."""
        subtask = Subtask(
            id="test_001",
            name="Test subtask",
            description="Test",
            task_type=TaskType.UNIT_TEST_GENERATION,
            priority=TaskPriority.HIGH
        )
        
        plan = ExecutionPlan(
            task_id="plan_001",
            understanding=TaskUnderstanding(
                original_request="Test",
                task_type=TaskType.UNIT_TEST_GENERATION,
                intent="Test"
            ),
            subtasks=[subtask]
        )
        
        feedback = ExecutionFeedback(
            subtask_id="test_001",
            success=False,
            message="Compilation error"
        )
        
        refined = await planner.refine_plan(plan, feedback)
        
        # Should add recovery subtasks
        assert len(refined.subtasks) > 1
        assert refined.subtasks[0].completed is False
    
    def test_validate_and_optimize(self, planner):
        """Test subtask validation and optimization."""
        subtasks = [
            Subtask(
                id=f"test_{i}",
                name=f"Task {i}",
                description="Test",
                task_type=TaskType.UNKNOWN,
                priority=TaskPriority.MEDIUM
            )
            for i in range(15)
        ]
        
        optimized = planner._validate_and_optimize(subtasks)
        
        # Should be limited to max_subtasks
        assert len(optimized) <= planner.max_subtasks
    
    def test_extract_json_valid(self, planner):
        """Test JSON extraction from valid content."""
        content = '''Some text
        {
            "key": "value",
            "number": 123
        }
        More text'''
        
        result = planner._extract_json(content)
        
        assert result is not None
        assert result["key"] == "value"
        assert result["number"] == 123
    
    def test_extract_json_invalid(self, planner):
        """Test JSON extraction from invalid content."""
        content = "No JSON here"
        
        result = planner._extract_json(content)
        
        assert result is None
    
    def test_parse_task_type_valid(self, planner):
        """Test parsing valid task type."""
        result = planner._parse_task_type("BUG_FIXING")
        assert result == TaskType.BUG_FIXING
    
    def test_parse_task_type_invalid(self, planner):
        """Test parsing invalid task type."""
        result = planner._parse_task_type("INVALID_TYPE")
        assert result == TaskType.UNKNOWN


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_autonomous_planner(self):
        """Test factory function."""
        mock_llm = Mock()
        planner = create_autonomous_planner(mock_llm, max_subtasks=5)
        
        assert isinstance(planner, AutonomousPlanner)
        assert planner.max_subtasks == 5
        assert planner.llm_client == mock_llm
