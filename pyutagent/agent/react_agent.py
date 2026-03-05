"""ReAct Agent for UT generation with self-feedback loop and infinite retry.

This module provides the main ReActAgent class, which now uses a modular architecture
with delegated functionality to specialized components.

Architecture:
- AgentCore: Core state management and lifecycle
- AgentInitializer: Component initialization and dependency injection
- FeedbackLoopExecutor: Main feedback loop execution
- StepExecutor: Individual step execution (parse, generate, compile, test, analyze)
- AgentRecoveryManager: Error recovery with learning
- AgentHelpers: Utility and helper methods
- AgentExtensions: Advanced features (quality analysis, refactoring, static analysis)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from .base_agent import BaseAgent
from ..core.protocols import AgentState, AgentResult
from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient
from ..core.container import Container, get_container
from ..core.config import get_settings

from .components import (
    AgentCore,
    AgentInitializer,
    FeedbackLoopExecutor,
    StepExecutor,
    AgentRecoveryManager,
    AgentHelpers,
    AgentExtensions,
)

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """ReAct agent for iterative UT generation with feedback loop.
    
    This is now a Facade that delegates to specialized components:
    - AgentCore: Core state and lifecycle
    - AgentInitializer: Component initialization
    - FeedbackLoopExecutor: Feedback loop execution
    - StepExecutor: Step execution (parse, generate, compile, test, analyze)
    - AgentRecoveryManager: Error recovery
    - AgentHelpers: Utility methods
    - AgentExtensions: Advanced features
    
    Key features:
    - Infinite retry until success or user stops
    - AI-powered error recovery for all error types
    - Local + LLM double-layer error analysis
    - Automatic strategy adjustment based on failure history
    - Dependency injection for better testability
    - Context window management for large files
    - Pre-compilation code quality evaluation
    - Partial success handling for incremental repair
    - Incremental mode for preserving existing passing tests
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        container: Optional[Container] = None,
        model_name: Optional[str] = None,
        ab_test_id: Optional[str] = None,
        incremental_mode: bool = False,
        preserve_passing_tests: bool = True,
        skip_test_analysis: bool = False
    ):
        """Initialize ReAct agent.
        
        Args:
            llm_client: LLM client for generation
            working_memory: Working memory for context
            project_path: Path to the project
            progress_callback: Optional callback for progress updates
            container: Optional dependency injection container
            model_name: LLM model name for prompt optimization
            ab_test_id: Optional A/B test ID for prompt variant testing
            incremental_mode: Enable incremental test generation mode
            preserve_passing_tests: Whether to preserve passing tests in incremental mode
            skip_test_analysis: Skip running existing tests, just analyze file content
        """
        super().__init__(llm_client, working_memory, project_path, progress_callback)
        
        self._container = container or get_container()
        self.model_name = model_name or "gpt-4"
        self.ab_test_id = ab_test_id
        self.incremental_mode = incremental_mode
        self.preserve_passing_tests = preserve_passing_tests
        self.skip_test_analysis = skip_test_analysis
        
        logger.info(f"[ReActAgent] Initializing agent - Project: {project_path}, Model: {self.model_name}, IncrementalMode: {incremental_mode}")
        
        self._init_components()
        
        logger.info("[ReActAgent] Initialization complete")
    
    def _init_components(self):
        """Initialize all components."""
        initializer = AgentInitializer(
            llm_client=self.llm_client,
            working_memory=self.working_memory,
            project_path=self.project_path,
            container=self._container,
            model_name=self.model_name,
            ab_test_id=self.ab_test_id
        )
        
        components = initializer.initialize_all_components(self)
        
        self._core = AgentCore(
            llm_client=self.llm_client,
            working_memory=self.working_memory,
            project_path=self.project_path,
            progress_callback=self.progress_callback,
        )
        
        for key, value in components.items():
            setattr(self._core, key, value)
            setattr(self, key, value)
        
        self._step_executor = StepExecutor(self._core, components)
        self._feedback_loop = FeedbackLoopExecutor(self._core, self._step_executor)
        self._recovery_manager = AgentRecoveryManager(components, self._core)
        self._helpers = AgentHelpers(self._core, components)
        self._extensions = AgentExtensions(self._core, components)
        
        if self.incremental_mode:
            from pyutagent.agent.incremental_manager import create_incremental_manager
            self.incremental_manager = create_incremental_manager(
                project_path=str(self.project_path),
                incremental_mode=True,
                maven_runner=components.get("maven_runner"),
                coverage_analyzer=components.get("coverage_analyzer"),
                preserve_passing_tests=self.preserve_passing_tests,
                skip_analysis=self.skip_test_analysis,
            )
            self._core.incremental_manager = self.incremental_manager
            self._core.incremental_mode = True
            self._core.skip_test_analysis = self.skip_test_analysis
            logger.info("[ReActAgent] Incremental manager initialized")
        
        settings = get_settings()
        self.max_compilation_attempts = settings.coverage.max_compilation_attempts
        self.max_test_attempts = settings.coverage.max_test_attempts
        logger.info(f"[ReActAgent] Attempt limits set - Compilation: {self.max_compilation_attempts}, Test: {self.max_test_attempts}")
    
    def stop(self):
        """Stop agent execution gracefully (legacy, use pause instead)."""
        self._core.stop()
        if hasattr(self, 'retry_manager'):
            self.retry_manager.stop()
        if hasattr(self.llm_client, 'cancel'):
            self.llm_client.cancel()
    
    def pause(self):
        """Pause agent execution."""
        self._core.pause()
        if hasattr(self, 'retry_manager'):
            self.retry_manager.stop()
        if hasattr(self.llm_client, 'cancel'):
            self.llm_client.cancel()
    
    def resume(self):
        """Resume agent execution."""
        self._core.resume()
        if hasattr(self, 'retry_manager'):
            self.retry_manager.reset()
    
    def terminate(self):
        """Terminate agent execution immediately."""
        logger.info("[ReActAgent] Terminating agent execution")
        self._core.terminate()
        
        # Cancel LLM client operations
        if hasattr(self.llm_client, 'cancel'):
            logger.info("[ReActAgent] Cancelling LLM client")
            self.llm_client.cancel()
        
        # Cancel LLM client current task if available
        if hasattr(self.llm_client, 'cancel_current_task'):
            logger.info("[ReActAgent] Force cancelling current LLM task")
            self.llm_client.cancel_current_task()
        
        # Stop retry manager
        if hasattr(self, 'retry_manager'):
            self.retry_manager.stop()
        
        # Clear error recovery history
        if hasattr(self, 'error_recovery'):
            self.error_recovery.clear_history()
        
        logger.info("[ReActAgent] Agent termination complete")
    
    def reset(self):
        """Reset agent state."""
        self._core.reset()
        if hasattr(self, 'retry_manager'):
            self.retry_manager.reset()
        if hasattr(self, 'error_recovery'):
            self.error_recovery.clear_history()
        if hasattr(self, 'checkpoint_manager'):
            self.checkpoint_manager.clear()
    
    async def generate_tests(self, target_file: str) -> AgentResult:
        """Generate tests for a target file with feedback loop.
        
        Args:
            target_file: Path to target Java file
            
        Returns:
            AgentResult with generation results
        """
        logger.info(f"[ReActAgent] Starting test generation - Target: {target_file}")
        return await self.run_feedback_loop(target_file)
    
    async def resume_from_checkpoint(
        self,
        checkpoint_id: Optional[str] = None
    ) -> AgentResult:
        """Resume execution from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to resume from, or None for latest
            
        Returns:
            AgentResult after resuming
        """
        logger.info(f"[ReActAgent] Resuming from checkpoint: {checkpoint_id or 'latest'}")
        
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if not checkpoint:
            logger.warning("[ReActAgent] No checkpoint found to resume from")
            return AgentResult(
                success=False,
                message="No checkpoint found to resume from",
                state=AgentState.FAILED
            )
        
        state = checkpoint.state
        target_file = state.get("target_file")
        self.current_test_file = state.get("test_file")
        self.target_class_info = state.get("class_info")
        
        logger.info(f"[ReActAgent] Restored state from checkpoint - "
                    f"Target: {target_file}, TestFile: {self.current_test_file}")
        
        return await self.run_feedback_loop(target_file)
    
    async def run_feedback_loop(self, target_file: str) -> AgentResult:
        """Run the complete feedback loop for UT generation.
        
        Args:
            target_file: Path to target Java file
            
        Returns:
            AgentResult with final results
        """
        return await self._feedback_loop.run_feedback_loop(target_file)
    
    @property
    def current_test_file(self) -> Optional[str]:
        """Get current test file path."""
        return self._core.current_test_file
    
    @current_test_file.setter
    def current_test_file(self, value: Optional[str]):
        """Set current test file path."""
        self._core.current_test_file = value
    
    @property
    def target_class_info(self) -> Optional[Dict[str, Any]]:
        """Get target class information."""
        return self._core.target_class_info
    
    @target_class_info.setter
    def target_class_info(self, value: Optional[Dict[str, Any]]):
        """Set target class information."""
        self._core.target_class_info = value
    
    @property
    def project_path(self) -> Path:
        """Get project path."""
        if hasattr(self, '_project_path'):
            return self._project_path
        # Fallback to base_agent's project_path if available
        return Path('.')
    
    @project_path.setter
    def project_path(self, value: str | Path):
        """Set project path."""
        self._project_path = Path(value) if isinstance(value, str) else value
    
    @property
    def max_iterations(self) -> int:
        """Get maximum iterations."""
        if hasattr(self, '_max_iterations'):
            return self._max_iterations
        if hasattr(self, 'working_memory') and self.working_memory:
            return self.working_memory.max_iterations
        return 10
    
    @max_iterations.setter
    def max_iterations(self, value: int):
        """Set maximum iterations."""
        self._max_iterations = value
    
    @property
    def target_coverage(self) -> float:
        """Get target coverage."""
        if hasattr(self, '_target_coverage'):
            return self._target_coverage
        if hasattr(self, 'working_memory') and self.working_memory:
            return self.working_memory.target_coverage
        return 0.8
    
    @target_coverage.setter
    def target_coverage(self, value: float):
        """Set target coverage."""
        self._target_coverage = value
    
    @property
    def current_iteration(self) -> int:
        """Get current iteration."""
        return self.working_memory.current_iteration
    
    @current_iteration.setter
    def current_iteration(self, value: int):
        """Set current iteration."""
        self.working_memory.current_iteration = value
    
    def _update_state(self, state: AgentState, message: str):
        """Update agent state and notify progress."""
        self._core._update_state(state, message)
    
    async def _check_pause(self) -> None:
        """Check if agent should pause and wait."""
        await self._core._check_pause()
    
    def _create_terminated_result(self, context: str) -> AgentResult:
        """Create result when generation is terminated."""
        return self._core._create_terminated_result(context)
    
    def _create_success_result(self, coverage: float, source: str = "jacoco", confidence: float = 1.0) -> AgentResult:
        """Create success result when target coverage is reached."""
        return self._core._create_success_result(coverage, source, confidence)
    
    def _extract_java_code(self, response: str) -> str:
        """Extract Java code from LLM response."""
        return self._core._extract_java_code(response)
    
    # Delegate methods to StepExecutor
    async def _parse_target_file(self, target_file: str):
        """Parse target file."""
        return await self._step_executor.parse_target_file(target_file)
    
    async def _generate_initial_tests(self, use_streaming: bool = True):
        """Generate initial tests."""
        return await self._step_executor.generate_initial_tests(use_streaming)
    
    async def _compile_tests(self):
        """Compile tests."""
        return await self._step_executor.compile_tests()
    
    async def _compile_with_recovery(self) -> bool:
        """Compile with recovery."""
        return await self._step_executor.compile_with_recovery()
    
    async def _run_tests(self):
        """Run tests."""
        return await self._step_executor.run_tests()
    
    async def _run_tests_with_recovery(self) -> bool:
        """Run tests with recovery."""
        return await self._step_executor.run_tests_with_recovery()
    
    async def _analyze_coverage(self):
        """Analyze coverage."""
        return await self._step_executor.analyze_coverage()
    
    async def _generate_additional_tests(self, coverage_data: Dict[str, Any]):
        """Generate additional tests."""
        return await self._step_executor.generate_additional_tests(coverage_data)
    
    async def _write_test_file(self, code: str):
        """Write test file."""
        await self._step_executor._write_test_file(code)
    
    async def _execute_with_recovery(self, operation, *args, step_name: str = "operation", **kwargs):
        """Execute with recovery."""
        return await self._step_executor.execute_with_recovery(operation, *args, step_name=step_name, **kwargs)
    
    async def _try_recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to recover from error."""
        return await self._recovery_manager.recover_from_error(
            error, context, step_name=context.get("step", "unknown"), attempt=context.get("attempt", 1)
        )
    
    # Delegate methods to AgentHelpers
    def get_build_tool_info(self) -> Dict[str, Any]:
        """Get build tool info."""
        return self._helpers.get_build_tool_info()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._helpers.get_performance_metrics()
    
    def get_adaptive_strategy_recommendation(self, error_category: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get adaptive strategy recommendation."""
        return self._helpers.get_adaptive_strategy_recommendation(error_category, context)
    
    def record_strategy_outcome(self, strategy_name: str, success: bool, execution_time_ms: float, error_category: Optional[str] = None):
        """Record strategy outcome."""
        self._helpers.record_strategy_outcome(strategy_name, success, execution_time_ms, error_category)
    
    async def semantic_search_code(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Semantic search code."""
        return await self._helpers.semantic_search_code(query, limit)
    
    async def index_code_for_search(self, code: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Index code for search."""
        return await self._helpers.index_code_for_search(code, metadata)
    
    async def run_tests_with_build_tool(self, test_class: Optional[str] = None, test_method: Optional[str] = None) -> Dict[str, Any]:
        """Run tests with build tool."""
        return await self._helpers.run_tests_with_build_tool(test_class, test_method)
    
    # Delegate methods to AgentExtensions
    async def analyze_test_quality(self, test_code: Optional[str] = None) -> Dict[str, Any]:
        """Analyze test quality."""
        return await self._extensions.analyze_test_quality(test_code)
    
    async def suggest_refactorings(self, test_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Suggest refactorings."""
        return await self._extensions.suggest_refactorings(test_code)
    
    async def apply_refactoring(self, refactoring_type: str, test_code: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Apply refactoring."""
        return await self._extensions.apply_refactoring(refactoring_type, test_code, **kwargs)
    
    async def auto_refactor_tests(self, test_code: Optional[str] = None, max_refactorings: int = 5, min_confidence: float = 0.8) -> Dict[str, Any]:
        """Auto refactor tests."""
        return await self._extensions.auto_refactor_tests(test_code, max_refactorings, min_confidence)
    
    async def execute_test_in_interpreter(self, test_code: str, test_method_name: Optional[str] = None) -> Dict[str, Any]:
        """Execute test in interpreter."""
        return await self._extensions.execute_test_in_interpreter(test_code, test_method_name)
    
    async def validate_test_with_interpreter(self, test_code: Optional[str] = None) -> Dict[str, Any]:
        """Validate test with interpreter."""
        return await self._extensions.validate_test_with_interpreter(test_code)
    
    def get_quality_trend(self, last_n: int = 10) -> Dict[str, Any]:
        """Get quality trend."""
        return self._extensions.get_quality_trend(last_n)
    
    async def run_static_analysis(self, source_path: Optional[str] = None, tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run static analysis."""
        return await self._extensions.run_static_analysis(source_path, tools)
    
    async def query_error_knowledge(self, error_message: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Query error knowledge."""
        return await self._extensions.query_error_knowledge(error_message, limit)
    
    async def record_error_solution(self, error_message: str, error_category: str, solution_description: str, success: bool) -> bool:
        """Record error solution."""
        return await self._extensions.record_error_solution(error_message, error_category, solution_description, success)
    
    def get_ab_test_analysis(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """Get A/B test analysis results.
        
        Args:
            test_id: Test ID to analyze, or None for current test
            
        Returns:
            Analysis results
        """
        test_id = test_id or self.ab_test_id
        
        if not test_id or not hasattr(self, 'prompt_optimizer'):
            return {"error": "No A/B test configured"}
        
        try:
            return self.prompt_optimizer.analyze_ab_test(test_id)
        except Exception as e:
            logger.error(f"[ReActAgent] Failed to analyze A/B test: {e}")
            return {"error": str(e)}
