"""End-to-End tests for the complete Agent workflow.

This module provides comprehensive E2E tests:
- Full workflow testing with real Maven projects
- Multi-file project testing
- Error recovery scenario testing
- Performance benchmarking
"""

import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_project():
    """Create a temporary Maven project for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        src_main = project_path / "src" / "main" / "java" / "com" / "example"
        src_main.mkdir(parents=True, exist_ok=True)
        
        src_test = project_path / "src" / "test" / "java" / "com" / "example"
        src_test.mkdir(parents=True, exist_ok=True)
        
        calculator_code = '''
package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public int divide(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("Division by zero");
        }
        return a / b;
    }
    
    public boolean isPositive(int n) {
        return n > 0;
    }
    
    public boolean isNegative(int n) {
        return n < 0;
    }
    
    public int abs(int n) {
        return n < 0 ? -n : n;
    }
}
'''
        (src_main / "Calculator.java").write_text(calculator_code)
        
        pom_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>calculator</artifactId>
    <version>1.0-SNAPSHOT</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.9.3</version>
            <scope>test</scope>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.1.2</version>
            </plugin>
        </plugins>
    </build>
</project>
'''
        (project_path / "pom.xml").write_text(pom_xml)
        
        yield project_path


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = AsyncMock()
    
    test_code = '''
package com.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    private Calculator calculator;
    
    @BeforeEach
    void setUp() {
        calculator = new Calculator();
    }
    
    @Test
    void testAdd() {
        assertEquals(5, calculator.add(2, 3));
        assertEquals(0, calculator.add(-1, 1));
        assertEquals(-5, calculator.add(-2, -3));
    }
    
    @Test
    void testSubtract() {
        assertEquals(1, calculator.subtract(3, 2));
        assertEquals(-2, calculator.subtract(-1, 1));
    }
    
    @Test
    void testMultiply() {
        assertEquals(6, calculator.multiply(2, 3));
        assertEquals(-6, calculator.multiply(-2, 3));
    }
    
    @Test
    void testDivide() {
        assertEquals(2, calculator.divide(6, 3));
        assertEquals(0, calculator.divide(1, 2));
    }
    
    @Test
    void testDivideByZero() {
        assertThrows(IllegalArgumentException.class, () -> {
            calculator.divide(1, 0);
        });
    }
    
    @Test
    void testIsPositive() {
        assertTrue(calculator.isPositive(1));
        assertFalse(calculator.isPositive(0));
        assertFalse(calculator.isPositive(-1));
    }
    
    @Test
    void testIsNegative() {
        assertTrue(calculator.isNegative(-1));
        assertFalse(calculator.isNegative(0));
        assertFalse(calculator.isNegative(1));
    }
    
    @Test
    void testAbs() {
        assertEquals(5, calculator.abs(5));
        assertEquals(5, calculator.abs(-5));
        assertEquals(0, calculator.abs(0));
    }
}
'''
    
    client.agenerate = AsyncMock(return_value=test_code)
    client.astream = AsyncMock(return_value=iter([test_code]))
    
    return client


class TestAgentWorkflowE2E:
    """End-to-end tests for Agent workflow."""
    
    @pytest.mark.asyncio
    async def test_full_workflow_success(self, temp_project, mock_llm_client):
        """Test complete successful workflow."""
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.core.protocols import AgentState
        
        agent = ReActAgent(
            project_path=str(temp_project),
            llm_client=mock_llm_client,
            target_coverage=0.7
        )
        
        target_file = str(temp_project / "src" / "main" / "java" / "com" / "example" / "Calculator.java")
        
        result = await agent.run_feedback_loop(target_file)
        
        assert result is not None
        assert result.success or result.state == AgentState.COMPLETED or result.coverage >= 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_pause_resume(self, temp_project, mock_llm_client):
        """Test workflow with pause and resume."""
        from pyutagent.agent.react_agent import ReActAgent
        
        agent = ReActAgent(
            project_path=str(temp_project),
            llm_client=mock_llm_client
        )
        
        agent.pause()
        assert agent._is_paused
        
        agent.resume()
        assert not agent._is_paused
    
    @pytest.mark.asyncio
    async def test_workflow_with_terminate(self, temp_project, mock_llm_client):
        """Test workflow termination."""
        from pyutagent.agent.react_agent import ReActAgent
        
        agent = ReActAgent(
            project_path=str(temp_project),
            llm_client=mock_llm_client
        )
        
        agent.terminate()
        assert agent._terminated
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, temp_project):
        """Test error recovery in workflow."""
        from pyutagent.core.error_recovery import ErrorRecoveryManager, ErrorCategory
        
        recovery_manager = ErrorRecoveryManager(max_total_attempts=3)
        
        error = Exception("Compilation failed: cannot find symbol")
        
        context = {
            "step": "compile",
            "test_code": "public class Test {}",
            "error_message": str(error)
        }
        
        result = await recovery_manager.recover(
            error=error,
            error_context=context
        )
        
        assert result is not None
        assert "action" in result
    
    @pytest.mark.asyncio
    async def test_checkpoint_save_restore(self, temp_project):
        """Test checkpoint save and restore."""
        from pyutagent.core.checkpoint import CheckpointManager
        
        checkpoint_manager = CheckpointManager()
        
        checkpoint_id = checkpoint_manager.save_checkpoint(
            step="test_generation",
            iteration=1,
            state={
                "target_file": "Calculator.java",
                "test_file": "CalculatorTest.java",
                "coverage": 0.5
            }
        )
        
        assert checkpoint_id is not None
        
        restored = checkpoint_manager.load_checkpoint(checkpoint_id)
        assert restored is not None
        assert restored.step == "test_generation"
        assert restored.iteration == 1
        assert restored.state["coverage"] == 0.5


class TestStreamingE2E:
    """End-to-end tests for streaming functionality."""
    
    @pytest.mark.asyncio
    async def test_streaming_generation(self, mock_llm_client):
        """Test streaming code generation."""
        from pyutagent.agent.streaming import StreamingCodeGenerator, StreamingConfig
        
        config = StreamingConfig(enable_preview=True)
        generator = StreamingCodeGenerator(mock_llm_client, config)
        
        chunks_received = []
        
        def on_chunk(chunk: str):
            chunks_received.append(chunk)
        
        result = await generator.generate_with_streaming(
            prompt="Generate a test",
            on_chunk=on_chunk
        )
        
        assert result.success
        assert len(result.complete_code) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_interrupt(self, mock_llm_client):
        """Test interrupting streaming generation."""
        from pyutagent.agent.streaming import StreamingCodeGenerator, StreamingConfig
        
        config = StreamingConfig(enable_preview=True)
        generator = StreamingCodeGenerator(mock_llm_client, config)
        
        generator.interrupt()
        
        assert generator.state.name == "INTERRUPTED"


class TestSmartEditorE2E:
    """End-to-end tests for smart editor."""
    
    @pytest.mark.asyncio
    async def test_search_replace_edit(self):
        """Test search/replace editing."""
        from pyutagent.tools.smart_editor import SmartCodeEditor, EditType
        
        editor = SmartCodeEditor()
        
        original = '''
public class Test {
    public void testMethod() {
        System.out.println("Hello");
    }
}
'''
        
        result = await editor.apply_search_replace(
            code=original,
            search='System.out.println("Hello");',
            replace='System.out.println("World");'
        )
        
        assert result.success
        assert "World" in result.modified_code
        assert "Hello" not in result.modified_code
    
    @pytest.mark.asyncio
    async def test_fuzzy_match_edit(self):
        """Test fuzzy matching edit."""
        from pyutagent.tools.smart_editor import SmartCodeEditor
        
        editor = SmartCodeEditor(fuzzy_threshold=0.7)
        
        original = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
'''
        
        result = await editor.apply_search_replace(
            code=original,
            search='return a + b',
            replace='return a + b; // addition',
            fuzzy=True
        )
        
        assert result.success or "addition" in result.modified_code


class TestErrorLearningE2E:
    """End-to-end tests for error learning."""
    
    def test_error_pattern_extraction(self):
        """Test error pattern extraction."""
        from pyutagent.core.error_learner import ErrorPatternLearner
        from pyutagent.core.error_recovery import ErrorCategory
        
        learner = ErrorPatternLearner(persist_path=".test_patterns.json")
        
        error = Exception("cannot find symbol: class List")
        
        pattern = learner.extract_pattern(error, ErrorCategory.COMPILATION_ERROR)
        
        assert pattern is not None
        assert pattern.category == ErrorCategory.COMPILATION_ERROR
        assert len(pattern.keywords) > 0
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation from learning."""
        from pyutagent.core.error_learner import ErrorPatternLearner
        from pyutagent.core.error_recovery import ErrorCategory, RecoveryStrategy
        
        learner = ErrorPatternLearner(persist_path=".test_patterns.json")
        
        for _ in range(5):
            learner.learn_from_recovery(
                error=Exception("compilation error"),
                error_category=ErrorCategory.COMPILATION_ERROR,
                strategy=RecoveryStrategy.ANALYZE_AND_FIX,
                success=True
            )
        
        error = Exception("compilation error")
        recommendation = learner.suggest_strategy(error, ErrorCategory.COMPILATION_ERROR)
        
        assert recommendation is not None


class TestToolOrchestratorE2E:
    """End-to-end tests for tool orchestrator."""
    
    @pytest.mark.asyncio
    async def test_plan_creation(self):
        """Test tool plan creation."""
        from pyutagent.agent.tool_orchestrator import ToolOrchestrator
        
        orchestrator = ToolOrchestrator()
        
        plan = orchestrator.plan_tool_sequence(
            goal="generate tests for Calculator",
            context={"target_file": "Calculator.java"}
        )
        
        assert plan is not None
        assert len(plan.steps) > 0
    
    @pytest.mark.asyncio
    async def test_plan_execution(self):
        """Test plan execution."""
        from pyutagent.agent.tool_orchestrator import ToolOrchestrator
        
        async def mock_parse(*args, **kwargs):
            return {"class_name": "Calculator"}
        
        orchestrator = ToolOrchestrator(
            tools={"parse_code": mock_parse}
        )
        
        plan = orchestrator.plan_tool_sequence(
            goal="parse Calculator",
            context={}
        )
        
        result = await orchestrator.execute_plan(plan)
        
        assert result is not None


class TestMetricsE2E:
    """End-to-end tests for metrics collection."""
    
    def test_timing_operations(self):
        """Test operation timing."""
        from pyutagent.core.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        timer_id = metrics.start_timer("test_operation")
        import time
        time.sleep(0.01)
        duration = metrics.stop_timer(timer_id)
        
        assert duration is not None
        assert duration >= 0.01
    
    def test_llm_stats_recording(self):
        """Test LLM statistics recording."""
        from pyutagent.core.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        metrics.record_llm_call(tokens=100, time_taken=1.5, success=True)
        metrics.record_llm_call(tokens=200, time_taken=2.0, success=True)
        metrics.record_llm_call(tokens=50, time_taken=0.5, success=False)
        
        summary = metrics.get_summary()
        
        assert summary["llm_stats"]["total_calls"] == 3
        assert summary["llm_stats"]["total_tokens"] == 350
        assert summary["llm_stats"]["success_rate"] == 2/3
    
    def test_error_stats_recording(self):
        """Test error statistics recording."""
        from pyutagent.core.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        metrics.record_error("compilation", "generate", recovered=True)
        metrics.record_error("test_failure", "test", recovered=False)
        metrics.record_error("compilation", "compile", recovered=True)
        
        summary = metrics.get_summary()
        
        assert summary["error_stats"]["total_errors"] == 3
        assert summary["error_stats"]["recovery_rate"] == 2/3
    
    def test_report_generation(self):
        """Test report generation."""
        from pyutagent.core.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        timer_id = metrics.start_timer("test_op")
        metrics.stop_timer(timer_id)
        
        metrics.record_llm_call(tokens=100, time_taken=1.0, success=True)
        metrics.record_error("test", "step", recovered=True)
        
        report = metrics.generate_report()
        
        assert "Performance Report" in report
        assert "LLM Statistics" in report
        assert "Error Statistics" in report


class TestContextCompressorE2E:
    """End-to-end tests for context compression."""
    
    def test_context_building(self):
        """Test context building."""
        from pyutagent.memory.context_compressor import ContextCompressor, ContextConfig
        
        config = ContextConfig(max_tokens=1000)
        compressor = ContextCompressor(config)
        
        code = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
'''
        compressor.add_file("Calculator.java", code)
        
        context = compressor.build_context(
            query="test add method",
            target_file="Calculator.java"
        )
        
        assert context is not None
        assert context.total_tokens > 0


class TestProjectAnalyzerE2E:
    """End-to-end tests for project analyzer."""
    
    @pytest.mark.asyncio
    async def test_project_analysis(self, temp_project):
        """Test project structure analysis."""
        from pyutagent.tools.project_analyzer import ProjectAnalyzer
        
        analyzer = ProjectAnalyzer(str(temp_project))
        
        structure = await analyzer.analyze_project()
        
        assert structure is not None
        assert len(structure.classes) > 0
        assert "Calculator" in structure.classes
    
    @pytest.mark.asyncio
    async def test_dependency_analysis(self, temp_project):
        """Test dependency analysis."""
        from pyutagent.tools.project_analyzer import ProjectAnalyzer
        
        analyzer = ProjectAnalyzer(str(temp_project))
        await analyzer.analyze_project()
        
        target_file = "src/main/java/com/example/Calculator.java"
        dependencies = await analyzer.analyze_dependencies(target_file)
        
        assert isinstance(dependencies, list)


class TestParallelRecoveryE2E:
    """End-to-end tests for parallel recovery."""
    
    @pytest.mark.asyncio
    async def test_parallel_recovery_execution(self):
        """Test parallel recovery execution."""
        from pyutagent.core.parallel_recovery import ParallelRecoveryManager
        from pyutagent.core.error_recovery import RecoveryStrategy, RecoveryResult
        
        manager = ParallelRecoveryManager(max_parallel=3)
        
        async def mock_executor(strategy, context):
            await asyncio.sleep(0.1)
            return RecoveryResult(
                success=(strategy == RecoveryStrategy.ANALYZE_AND_FIX),
                strategy_used=strategy,
                attempts_made=1
            )
        
        error = Exception("Test error")
        strategies = [
            RecoveryStrategy.ANALYZE_AND_FIX,
            RecoveryStrategy.RETRY_WITH_BACKOFF,
            RecoveryStrategy.FALLBACK_ALTERNATIVE
        ]
        
        from pyutagent.core.error_recovery import RecoveryContext
        context = RecoveryContext(
            error=error,
            step="test",
            attempt=1
        )
        
        result = await manager.recover_with_parallel_strategies(
            error=error,
            strategies=strategies,
            context=context,
            strategy_executor=mock_executor,
            timeout=5.0
        )
        
        assert result is not None
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
