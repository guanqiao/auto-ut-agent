"""Tests for Enhanced Autonomous Loop."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from pyutagent.agent.enhanced_autonomous_loop import (
    DecisionStrategy,
    ExecutionContext,
    LLMThought,
    LearningEntry,
    EnhancedAutonomousLoop,
    create_enhanced_autonomous_loop
)
from pyutagent.agent.autonomous_loop import (
    LoopState,
    Observation,
    Thought
)


class TestDecisionStrategy:
    """Test DecisionStrategy enum."""
    
    def test_strategy_values(self):
        """Test strategy enum values."""
        assert DecisionStrategy.RULE_BASED.name == "RULE_BASED"
        assert DecisionStrategy.LLM_BASED.name == "LLM_BASED"
        assert DecisionStrategy.HYBRID.name == "HYBRID"
        assert DecisionStrategy.ADAPTIVE.name == "ADAPTIVE"


class TestExecutionContext:
    """Test ExecutionContext dataclass."""
    
    def test_context_creation(self):
        """Test creating execution context."""
        context = ExecutionContext(
            task="Test task",
            iteration=1,
            max_iterations=10,
            observations=[],
            thoughts=[],
            actions_taken=[],
            available_tools=["tool1", "tool2"],
            tool_descriptions={"tool1": "Description 1"}
        )
        
        assert context.task == "Test task"
        assert context.iteration == 1
        assert context.available_tools == ["tool1", "tool2"]


class TestEnhancedAutonomousLoop:
    """Test EnhancedAutonomousLoop class."""
    
    @pytest.fixture
    def mock_tool_service(self):
        """Create mock tool service."""
        service = Mock()
        service.list_available_tools = Mock(return_value=[
            "read_file", "write_file", "git_status", "bash"
        ])
        service.get_tool = Mock(return_value=Mock(
            definition=Mock(description="Test tool description")
        ))
        service.execute_tool = AsyncMock(return_value=Mock(
            success=True,
            output="Test output"
        ))
        return service
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock()
        response = Mock()
        response.content = '''
        {
            "reasoning": "Test reasoning",
            "decision": "Use read_file tool",
            "confidence": 0.85,
            "plan": [
                {
                    "tool_name": "read_file",
                    "parameters": {"file_path": "test.py"},
                    "expected_outcome": "Read the file"
                }
            ],
            "risk_assessment": "Low risk",
            "alternative_approaches": ["Use glob"]
        }
        '''
        client.chat = AsyncMock(return_value=response)
        return client
    
    @pytest.fixture
    def loop(self, mock_tool_service, mock_llm_client):
        """Create EnhancedAutonomousLoop instance."""
        return EnhancedAutonomousLoop(
            tool_service=mock_tool_service,
            llm_client=mock_llm_client,
            max_iterations=5,
            decision_strategy=DecisionStrategy.HYBRID
        )
    
    def test_initialization(self, loop):
        """Test loop initialization."""
        assert loop.llm_client is not None
        assert loop.decision_strategy == DecisionStrategy.HYBRID
        assert loop.enable_self_correction is True
        assert loop.max_iterations == 5
    
    @pytest.mark.asyncio
    async def test_think_llm_based(self, loop, mock_llm_client):
        """Test LLM-based thinking."""
        observation = Observation(
            timestamp=datetime.now(),
            state_summary="Test state",
            relevant_data={}
        )
        
        thought = await loop._think_llm_based(
            ExecutionContext(
                task="Test task",
                iteration=1,
                max_iterations=5,
                observations=[observation],
                thoughts=[],
                actions_taken=[],
                available_tools=["read_file"],
                tool_descriptions={"read_file": "Read files"}
            )
        )
        
        assert thought.confidence > 0
        assert len(thought.plan) > 0
        assert thought.plan[0]["tool_name"] == "read_file"
    
    @pytest.mark.asyncio
    async def test_think_rule_based_test_task(self, loop):
        """Test rule-based thinking for test task."""
        context = ExecutionContext(
            task="Generate tests for the code",
            iteration=1,
            max_iterations=5,
            observations=[],
            thoughts=[],
            actions_taken=[],
            available_tools=["glob", "git_status"],
            tool_descriptions={}
        )
        
        thought = await loop._think_rule_based(context)
        
        assert "test" in thought.reasoning.lower()
        assert len(thought.plan) > 0
    
    @pytest.mark.asyncio
    async def test_think_rule_based_debug_task(self, loop):
        """Test rule-based thinking for debug task."""
        context = ExecutionContext(
            task="Fix the bug in login",
            iteration=1,
            max_iterations=5,
            observations=[],
            thoughts=[],
            actions_taken=[],
            available_tools=["git_status", "git_diff"],
            tool_descriptions={}
        )
        
        thought = await loop._think_rule_based(context)
        
        assert "debug" in thought.reasoning.lower() or "bug" in thought.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_think_hybrid(self, loop, mock_llm_client):
        """Test hybrid thinking."""
        observation = Observation(
            timestamp=datetime.now(),
            state_summary="Test state",
            relevant_data={}
        )
        
        thought = await loop._think_hybrid(
            ExecutionContext(
                task="Test task",
                iteration=1,
                max_iterations=5,
                observations=[observation],
                thoughts=[],
                actions_taken=[],
                available_tools=["read_file"],
                tool_descriptions={"read_file": "Read files"}
            )
        )
        
        assert thought.confidence > 0
        mock_llm_client.chat.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_think_adaptive_prefers_llm(self, loop):
        """Test adaptive thinking prefers LLM when it has higher success rate."""
        # Set up success rates
        loop._tool_success_rates["strategy_llm"] = {
            "successes": 8,
            "failures": 2,
            "total": 10
        }
        loop._tool_success_rates["strategy_rule"] = {
            "successes": 5,
            "failures": 5,
            "total": 10
        }
        
        observation = Observation(
            timestamp=datetime.now(),
            state_summary="Test state",
            relevant_data={}
        )
        
        thought = await loop._think_adaptive(
            ExecutionContext(
                task="Test task",
                iteration=1,
                max_iterations=5,
                observations=[observation],
                thoughts=[],
                actions_taken=[],
                available_tools=["read_file"],
                tool_descriptions={}
            )
        )
        
        assert "Adaptive" in thought.decision
    
    def test_build_thinking_prompt(self, loop):
        """Test building thinking prompt."""
        context = ExecutionContext(
            task="Test task",
            iteration=1,
            max_iterations=5,
            observations=[],
            thoughts=[],
            actions_taken=[],
            available_tools=["read_file"],
            tool_descriptions={"read_file": "Read files"},
            previous_errors=["Error 1"]
        )
        
        prompt = loop._build_thinking_prompt(context)
        
        assert "Test task" in prompt
        assert "read_file" in prompt
        assert "Error 1" in prompt
        assert "JSON" in prompt
    
    def test_parse_llm_thought_valid(self, loop):
        """Test parsing valid LLM response."""
        content = '''
        {
            "reasoning": "Test reasoning",
            "decision": "Test decision",
            "confidence": 0.9,
            "plan": [{"tool_name": "test"}],
            "risk_assessment": "Low",
            "alternative_approaches": ["Alt 1"]
        }
        '''
        
        llm_thought = loop._parse_llm_thought(content)
        
        assert llm_thought.reasoning == "Test reasoning"
        assert llm_thought.confidence == 0.9
        assert len(llm_thought.plan) == 1
    
    def test_parse_llm_thought_invalid(self, loop):
        """Test parsing invalid LLM response."""
        content = "Not valid JSON"
        
        llm_thought = loop._parse_llm_thought(content)
        
        assert llm_thought.confidence == 0.3
        assert "Failed to parse" in llm_thought.reasoning
    
    def test_validate_plan(self, loop):
        """Test plan validation."""
        plan = [
            {"tool_name": "read_file", "parameters": {}, "expected_outcome": ""},
            {"tool_name": "invalid_tool", "parameters": {}, "expected_outcome": ""},
            {"tool_name": "write_file", "parameters": "invalid", "expected_outcome": ""}
        ]
        
        validated = loop._validate_plan(plan, ["read_file", "write_file"])
        
        assert len(validated) == 2  # invalid_tool removed
        assert validated[0]["tool_name"] == "read_file"
        assert validated[1]["parameters"] == {}  # Invalid parameters sanitized
    
    def test_adjust_confidence(self, loop):
        """Test confidence adjustment."""
        # No errors, early iteration
        assert loop._adjust_confidence(0.8, 1, 0) == 0.8
        
        # With errors
        adjusted = loop._adjust_confidence(0.8, 1, 2)
        assert adjusted < 0.8
        
        # Late iteration
        adjusted = loop._adjust_confidence(0.8, 4, 0)
        assert adjusted < 0.8
    
    def test_determine_recovery_strategy_file_not_found(self, loop):
        """Test recovery strategy for file not found error."""
        strategy = loop._determine_recovery_strategy(
            {"tool_name": "read_file", "parameters": {"file_path": "test.py"}},
            "File not found: test.py"
        )
        
        assert strategy is not None
        assert strategy["tool_name"] == "glob"
    
    def test_determine_recovery_strategy_permission(self, loop):
        """Test recovery strategy for permission error."""
        strategy = loop._determine_recovery_strategy(
            {"tool_name": "write_file", "parameters": {}},
            "Permission denied"
        )
        
        assert strategy is not None
        assert strategy["tool_name"] == "bash"
    
    def test_determine_recovery_strategy_no_match(self, loop):
        """Test recovery strategy when no match."""
        strategy = loop._determine_recovery_strategy(
            {"tool_name": "test", "parameters": {}},
            "Unknown error"
        )
        
        assert strategy is None
    
    def test_track_tool_result(self, loop):
        """Test tracking tool results."""
        result = Mock(success=True)
        loop._track_tool_result("test_tool", result)
        
        assert "test_tool" in loop._tool_success_rates
        assert loop._tool_success_rates["test_tool"]["successes"] == 1
        
        # Track failure
        result2 = Mock(success=False)
        loop._track_tool_result("test_tool", result2)
        
        assert loop._tool_success_rates["test_tool"]["failures"] == 1
    
    def test_track_error(self, loop):
        """Test tracking errors."""
        loop._track_error("FileNotFoundError: file not found")
        
        assert "FileNotFoundError" in loop._error_patterns
        assert loop._error_patterns["FileNotFoundError"] == 1
    
    def test_extract_lesson_success(self, loop):
        """Test extracting lesson from success."""
        result = Mock()
        lesson = loop._extract_lesson("test_tool", True, result)
        
        assert "effective" in lesson
        assert "test_tool" in lesson
    
    def test_extract_lesson_failure(self, loop):
        """Test extracting lesson from failure."""
        result = Mock(error="Something went wrong")
        lesson = loop._extract_lesson("test_tool", False, result)
        
        assert "failed" in lesson
        assert "Something went wrong" in lesson
    
    def test_calculate_strategy_success_rate(self, loop):
        """Test calculating strategy success rate."""
        # No data
        assert loop._calculate_strategy_success_rate("llm") == 0.5
        
        # With data
        loop._tool_success_rates["strategy_llm"] = {
            "successes": 8,
            "failures": 2,
            "total": 10
        }
        
        assert loop._calculate_strategy_success_rate("llm") == 0.8
    
    def test_get_learning_summary(self, loop):
        """Test getting learning summary."""
        # Add some learning history
        loop._learning_history.append(LearningEntry(
            timestamp=datetime.now(),
            situation="Test situation",
            action_taken="Used tool",
            outcome="Success",
            lesson="Tool works well",
            success=True
        ))
        
        loop._tool_success_rates["test_tool"] = {
            "successes": 5,
            "failures": 1,
            "total": 6
        }
        
        summary = loop.get_learning_summary()
        
        assert summary["total_learnings"] == 1
        assert summary["success_rate"] == 1.0
        assert "test_tool" in summary["tool_success_rates"]
        assert len(summary["recent_lessons"]) == 1
    
    def test_merge_plans(self, loop):
        """Test merging plans."""
        llm_plan = [
            {"tool_name": "tool1", "parameters": {}},
            {"tool_name": "tool2", "parameters": {}}
        ]
        rule_plan = [
            {"tool_name": "tool2", "parameters": {}},  # Duplicate
            {"tool_name": "tool3", "parameters": {}}   # New
        ]
        
        merged = loop._merge_plans(llm_plan, rule_plan)
        
        assert len(merged) == 3
        tool_names = [a["tool_name"] for a in merged]
        assert "tool1" in tool_names
        assert "tool2" in tool_names
        assert "tool3" in tool_names
    
    def test_build_test_plan(self, loop):
        """Test building test plan."""
        context = ExecutionContext(
            task="Test",
            iteration=1,
            max_iterations=5,
            observations=[],
            thoughts=[],
            actions_taken=[],
            available_tools=[],
            tool_descriptions={}
        )
        
        plan = loop._build_test_plan(context)
        
        assert len(plan) > 0
        assert any("glob" in p["tool_name"] for p in plan)
    
    def test_build_debug_plan(self, loop):
        """Test building debug plan."""
        context = ExecutionContext(
            task="Debug",
            iteration=1,
            max_iterations=5,
            observations=[],
            thoughts=[],
            actions_taken=[],
            available_tools=[],
            tool_descriptions={}
        )
        
        plan = loop._build_debug_plan(context)
        
        assert len(plan) > 0
        assert any("git" in p["tool_name"] for p in plan)


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_enhanced_autonomous_loop(self):
        """Test factory function."""
        mock_tool_service = Mock()
        mock_llm_client = Mock()
        
        loop = create_enhanced_autonomous_loop(
            tool_service=mock_tool_service,
            llm_client=mock_llm_client,
            max_iterations=10,
            decision_strategy=DecisionStrategy.LLM_BASED
        )
        
        assert isinstance(loop, EnhancedAutonomousLoop)
        assert loop.max_iterations == 10
        assert loop.decision_strategy == DecisionStrategy.LLM_BASED
