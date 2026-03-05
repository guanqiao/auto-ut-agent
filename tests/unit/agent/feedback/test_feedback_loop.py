"""Tests for Feedback Loop."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from pyutagent.agent.feedback import (
    FeedbackLoop,
    FeedbackLoopConfig,
    LoopResult,
    LoopPhase,
)
from pyutagent.agent.core.agent_state import AgentState, StateManager
from pyutagent.agent.core.agent_context import AgentContext
from pyutagent.agent.execution.executor import StepExecutor, ExecutionResult


class TestFeedbackLoopConfig:
    """Tests for FeedbackLoopConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = FeedbackLoopConfig()
        
        assert config.max_iterations == 10
        assert config.target_coverage == 0.8
        assert config.max_compile_attempts == 3
        assert config.max_test_attempts == 3
        assert config.enable_incremental is True
    
    def test_to_dict(self):
        """Test configuration serialization."""
        config = FeedbackLoopConfig(
            max_iterations=5,
            target_coverage=0.9,
        )
        
        result = config.to_dict()
        
        assert result["max_iterations"] == 5
        assert result["target_coverage"] == 0.9


class TestLoopResult:
    """Tests for LoopResult."""
    
    def test_coverage_reached(self):
        """Test coverage reached check."""
        result = LoopResult(
            success=True,
            message="Done",
            coverage=0.85,
            target_coverage=0.8,
        )
        
        assert result.coverage_reached is True
    
    def test_coverage_not_reached(self):
        """Test coverage not reached check."""
        result = LoopResult(
            success=True,
            message="Done",
            coverage=0.7,
            target_coverage=0.8,
        )
        
        assert result.coverage_reached is False
    
    def test_to_dict(self):
        """Test result serialization."""
        result = LoopResult(
            success=True,
            message="Test message",
            phase=LoopPhase.COMPLETE,
            iteration=3,
            coverage=0.85,
            target_coverage=0.8,
        )
        
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["phase"] == "COMPLETE"
        assert data["iteration"] == 3
        assert data["coverage"] == 0.85
        assert data["coverage_reached"] is True


class TestFeedbackLoop:
    """Tests for FeedbackLoop."""
    
    @pytest.fixture
    def components(self):
        """Create test components."""
        state_manager = StateManager()
        context = AgentContext()
        
        executor = MagicMock(spec=StepExecutor)
        executor.execute_step_with_retry = AsyncMock()
        
        return state_manager, context, executor
    
    @pytest.fixture
    def feedback_loop(self, components):
        """Create a feedback loop instance."""
        state_manager, context, executor = components
        config = FeedbackLoopConfig(max_iterations=3)
        
        return FeedbackLoop(
            state_manager=state_manager,
            context=context,
            executor=executor,
            config=config,
        )
    
    @pytest.mark.asyncio
    async def test_properties(self, feedback_loop):
        """Test basic properties."""
        assert feedback_loop.current_iteration == 0
        assert feedback_loop.current_phase == LoopPhase.INITIALIZE
    
    def test_request_stop(self, feedback_loop):
        """Test stop request."""
        assert feedback_loop._stop_requested is False
        
        feedback_loop.request_stop()
        
        assert feedback_loop._stop_requested is True
    
    def test_pause_resume(self, feedback_loop):
        """Test pause and resume."""
        feedback_loop.pause()
        assert not feedback_loop._pause_event.is_set()
        
        feedback_loop.resume()
        assert feedback_loop._pause_event.is_set()
    
    @pytest.mark.asyncio
    async def test_run_successful(self, feedback_loop, components):
        """Test successful feedback loop run."""
        state_manager, context, executor = components
        
        executor.execute_step_with_retry.side_effect = [
            ExecutionResult(
                success=True,
                step_id="parse",
                step_name="Parse",
                data={"result": {"name": "TestClass", "methods": []}},
            ),
            ExecutionResult(
                success=True,
                step_id="generate",
                step_name="Generate",
                data={"result": {"test_file": "Test.java"}},
            ),
            ExecutionResult(
                success=True,
                step_id="compile_1",
                step_name="Compile",
            ),
            ExecutionResult(
                success=True,
                step_id="test_1",
                step_name="Test",
            ),
            ExecutionResult(
                success=True,
                step_id="analyze_1",
                step_name="Analyze",
                data={"result": {"line_coverage": 0.85}},
            ),
        ]
        
        result = await feedback_loop.run("TestClass.java")
        
        assert result.success is True
        assert result.coverage_reached is True
        assert result.phase == LoopPhase.COMPLETE
    
    @pytest.mark.asyncio
    async def test_run_parse_failure(self, feedback_loop, components):
        """Test handling parse failure."""
        state_manager, context, executor = components
        
        executor.execute_step_with_retry.return_value = ExecutionResult(
            success=False,
            step_id="parse",
            step_name="Parse",
            error="File not found",
        )
        
        result = await feedback_loop.run("Missing.java")
        
        assert result.success is False
        assert result.phase == LoopPhase.PARSE
        assert "Failed to parse" in result.message
    
    @pytest.mark.asyncio
    async def test_run_compile_failure(self, feedback_loop, components):
        """Test handling compile failure."""
        state_manager, context, executor = components
        
        executor.execute_step_with_retry.side_effect = [
            ExecutionResult(
                success=True,
                step_id="parse",
                step_name="Parse",
                data={"result": {"name": "TestClass"}},
            ),
            ExecutionResult(
                success=True,
                step_id="generate",
                step_name="Generate",
                data={"result": {"test_file": "Test.java"}},
            ),
            ExecutionResult(
                success=False,
                step_id="compile_1",
                step_name="Compile",
                error="Compilation failed",
                data={"errors": ["Syntax error"]},
            ),
        ]
        
        result = await feedback_loop.run("TestClass.java")
        
        assert result.success is False
        assert result.phase == LoopPhase.COMPILE
        assert len(result.compilation_errors) == 1
    
    @pytest.mark.asyncio
    async def test_run_max_iterations(self, feedback_loop, components):
        """Test reaching max iterations."""
        state_manager, context, executor = components
        
        executor.execute_step_with_retry.side_effect = [
            ExecutionResult(success=True, step_id="parse", step_name="Parse", data={"result": {}}),
            ExecutionResult(success=True, step_id="generate", step_name="Generate", data={"result": {}}),
            ExecutionResult(success=True, step_id="compile_1", step_name="Compile"),
            ExecutionResult(success=True, step_id="test_1", step_name="Test"),
            ExecutionResult(success=True, step_id="analyze_1", step_name="Analyze", data={"result": {"line_coverage": 0.5}}),
            ExecutionResult(success=True, step_id="optimize_1", step_name="Optimize"),
            ExecutionResult(success=True, step_id="compile_2", step_name="Compile"),
            ExecutionResult(success=True, step_id="test_2", step_name="Test"),
            ExecutionResult(success=True, step_id="analyze_2", step_name="Analyze", data={"result": {"line_coverage": 0.6}}),
            ExecutionResult(success=True, step_id="optimize_2", step_name="Optimize"),
            ExecutionResult(success=True, step_id="compile_3", step_name="Compile"),
            ExecutionResult(success=True, step_id="test_3", step_name="Test"),
            ExecutionResult(success=True, step_id="analyze_3", step_name="Analyze", data={"result": {"line_coverage": 0.7}}),
        ]
        
        result = await feedback_loop.run("TestClass.java")
        
        assert result.iteration == 3
        assert "Max iterations" in result.message
    
    @pytest.mark.asyncio
    async def test_run_stopped_by_user(self, feedback_loop, components):
        """Test stopping by user."""
        state_manager, context, executor = components
        
        executor.execute_step_with_retry.side_effect = [
            ExecutionResult(success=True, step_id="parse", step_name="Parse", data={"result": {}}),
            ExecutionResult(success=True, step_id="generate", step_name="Generate", data={"result": {}}),
            ExecutionResult(success=True, step_id="compile_1", step_name="Compile"),
            ExecutionResult(success=True, step_id="test_1", step_name="Test"),
            ExecutionResult(success=True, step_id="analyze_1", step_name="Analyze", data={"result": {"line_coverage": 0.5}}),
        ]
        
        async def run_with_stop():
            task = asyncio.create_task(feedback_loop.run("TestClass.java"))
            await asyncio.sleep(0.1)
            feedback_loop.request_stop()
            return await task
        
        result = await run_with_stop()
        
        assert result.success is False
        assert "Stopped by user" in result.message
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, components):
        """Test progress callback is called."""
        state_manager, context, executor = components
        progress_events = []
        
        def progress_callback(event):
            progress_events.append(event)
        
        loop = FeedbackLoop(
            state_manager=state_manager,
            context=context,
            executor=executor,
            progress_callback=progress_callback,
        )
        
        executor.execute_step_with_retry.side_effect = [
            ExecutionResult(success=True, step_id="parse", step_name="Parse", data={"result": {}}),
            ExecutionResult(success=True, step_id="generate", step_name="Generate", data={"result": {}}),
            ExecutionResult(success=True, step_id="compile_1", step_name="Compile"),
            ExecutionResult(success=True, step_id="test_1", step_name="Test"),
            ExecutionResult(success=True, step_id="analyze_1", step_name="Analyze", data={"result": {"line_coverage": 0.85}}),
        ]
        
        await loop.run("TestClass.java")
        
        assert len(progress_events) > 0
        assert any(e.get("phase") == "PARSE" for e in progress_events)
    
    def test_get_stats(self, feedback_loop):
        """Test getting statistics."""
        stats = feedback_loop.get_stats()
        
        assert "current_iteration" in stats
        assert "current_phase" in stats
        assert "config" in stats
        assert stats["config"]["max_iterations"] == 3
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, feedback_loop, components):
        """Test state manager transitions during execution."""
        state_manager, context, executor = components
        
        executor.execute_step_with_retry.side_effect = [
            ExecutionResult(success=True, step_id="parse", step_name="Parse", data={"result": {}}),
            ExecutionResult(success=True, step_id="generate", step_name="Generate", data={"result": {}}),
            ExecutionResult(success=True, step_id="compile_1", step_name="Compile"),
            ExecutionResult(success=True, step_id="test_1", step_name="Test"),
            ExecutionResult(success=True, step_id="analyze_1", step_name="Analyze", data={"result": {"line_coverage": 0.85}}),
        ]
        
        await feedback_loop.run("TestClass.java")
        
        assert state_manager.state == AgentState.COMPLETED
        assert len(state_manager.history) > 0
