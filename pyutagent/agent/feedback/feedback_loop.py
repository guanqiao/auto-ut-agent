"""Feedback Loop for Agent Test Generation.

This module provides the main feedback loop that:
1. Parses target Java file
2. Generates initial tests
3. Compiles tests
4. Runs tests
5. Analyzes coverage
6. Generates additional tests until target coverage is reached
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from ..core.agent_state import AgentState, StateManager
from ..core.agent_context import AgentContext, ContextKey
from ..execution.execution_plan import ExecutionPlan, Step, StepType, StepStatus
from ..execution.executor import StepExecutor, ExecutionResult

logger = logging.getLogger(__name__)


class LoopPhase(Enum):
    """Phases of the feedback loop."""
    INITIALIZE = auto()
    PARSE = auto()
    GENERATE = auto()
    COMPILE = auto()
    TEST = auto()
    ANALYZE = auto()
    OPTIMIZE = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class FeedbackLoopConfig:
    """Configuration for feedback loop."""
    
    max_iterations: int = 10
    target_coverage: float = 0.8
    
    max_compile_attempts: int = 3
    max_test_attempts: int = 3
    max_generate_attempts: int = 2
    
    compile_timeout: float = 120.0
    test_timeout: float = 300.0
    generate_timeout: float = 60.0
    
    enable_incremental: bool = True
    enable_coverage_optimization: bool = True
    
    stop_on_target_coverage: bool = True
    stop_on_all_passing: bool = False
    
    checkpoint_interval: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "target_coverage": self.target_coverage,
            "max_compile_attempts": self.max_compile_attempts,
            "max_test_attempts": self.max_test_attempts,
            "max_generate_attempts": self.max_generate_attempts,
            "enable_incremental": self.enable_incremental,
            "enable_coverage_optimization": self.enable_coverage_optimization,
        }


@dataclass
class LoopResult:
    """Result from feedback loop execution."""
    
    success: bool
    message: str
    
    phase: LoopPhase = LoopPhase.COMPLETE
    iteration: int = 0
    
    test_file: Optional[str] = None
    coverage: float = 0.0
    target_coverage: float = 0.8
    
    compilation_errors: List[str] = field(default_factory=list)
    test_failures: List[Dict[str, Any]] = field(default_factory=list)
    
    duration_ms: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def coverage_reached(self) -> bool:
        """Check if target coverage was reached."""
        return self.coverage >= self.target_coverage
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "phase": self.phase.name,
            "iteration": self.iteration,
            "test_file": self.test_file,
            "coverage": self.coverage,
            "target_coverage": self.target_coverage,
            "coverage_reached": self.coverage_reached,
            "compilation_errors": self.compilation_errors,
            "test_failures": self.test_failures,
            "duration_ms": self.duration_ms,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
        }


class FeedbackLoop:
    """Main feedback loop for test generation.
    
    Orchestrates the test generation process:
    1. Parse target file
    2. Generate initial tests
    3. Compile tests (with retry)
    4. Run tests (with retry)
    5. Analyze coverage
    6. Generate additional tests if needed
    7. Repeat until target coverage or max iterations
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        context: AgentContext,
        executor: StepExecutor,
        config: Optional[FeedbackLoopConfig] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize feedback loop.
        
        Args:
            state_manager: State manager for transitions
            context: Agent context for data sharing
            executor: Step executor for running steps
            config: Feedback loop configuration
            progress_callback: Optional callback for progress updates
        """
        self.state_manager = state_manager
        self.context = context
        self.executor = executor
        self.config = config or FeedbackLoopConfig()
        self.progress_callback = progress_callback
        
        self._current_iteration = 0
        self._current_phase = LoopPhase.INITIALIZE
        self._start_time: Optional[datetime] = None
        
        self._stop_requested = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        
        self._results: List[ExecutionResult] = []
        self._coverage_history: List[float] = []
    
    @property
    def current_iteration(self) -> int:
        """Get current iteration number."""
        return self._current_iteration
    
    @property
    def current_phase(self) -> LoopPhase:
        """Get current phase."""
        return self._current_phase
    
    def request_stop(self) -> None:
        """Request the loop to stop."""
        self._stop_requested = True
        logger.info("[FeedbackLoop] Stop requested")
    
    def pause(self) -> None:
        """Pause the loop."""
        self._pause_event.clear()
        logger.info("[FeedbackLoop] Paused")
    
    def resume(self) -> None:
        """Resume the loop."""
        self._pause_event.set()
        logger.info("[FeedbackLoop] Resumed")
    
    async def _check_pause(self) -> bool:
        """Check if paused and wait for resume.
        
        Returns:
            True if should continue, False if stopped
        """
        if self._stop_requested:
            return False
        
        await self._pause_event.wait()
        return not self._stop_requested
    
    def _update_phase(self, phase: LoopPhase, message: str = "") -> None:
        """Update current phase and notify progress."""
        self._current_phase = phase
        
        if self.progress_callback:
            self.progress_callback({
                "phase": phase.name,
                "iteration": self._current_iteration,
                "message": message,
                "coverage": self.context.get(ContextKey.CURRENT_COVERAGE, 0.0),
            })
        
        logger.info(f"[FeedbackLoop] Phase: {phase.name} - {message}")
    
    def _report_progress(self, event: str, data: Dict[str, Any] = None) -> None:
        """Report progress to callback."""
        if self.progress_callback:
            self.progress_callback({
                "event": event,
                "iteration": self._current_iteration,
                "phase": self._current_phase.name,
                **(data or {}),
            })
    
    async def run(self, target_file: str) -> LoopResult:
        """Run the feedback loop.
        
        Args:
            target_file: Path to target Java file
            
        Returns:
            LoopResult with final status
        """
        self._start_time = datetime.now()
        self._stop_requested = False
        self._results.clear()
        self._coverage_history.clear()
        
        self.context.set(ContextKey.TARGET_FILE, target_file)
        self.context.set(ContextKey.TARGET_COVERAGE, self.config.target_coverage)
        self.context.set(ContextKey.MAX_ITERATIONS, self.config.max_iterations)
        
        self._update_phase(LoopPhase.INITIALIZE, f"Starting feedback loop for {Path(target_file).name}")
        self.state_manager.transition(AgentState.INITIALIZING, "Initializing feedback loop")
        
        try:
            if not await self._check_pause():
                return self._create_stopped_result()
            
            parse_result = await self._phase_parse(target_file)
            if not parse_result:
                return parse_result
            
            if not await self._check_pause():
                return self._create_stopped_result()
            
            generate_result = await self._phase_generate()
            if not generate_result:
                return generate_result
            
            return await self._run_feedback_iterations()
            
        except Exception as e:
            logger.exception(f"[FeedbackLoop] Error: {e}")
            return self._create_error_result(str(e))
    
    async def _phase_parse(self, target_file: str) -> Optional[LoopResult]:
        """Phase: Parse target file.
        
        Args:
            target_file: Target file path
            
        Returns:
            LoopResult if failed, None if successful
        """
        self._update_phase(LoopPhase.PARSE, "Parsing target file")
        self.state_manager.transition(AgentState.PARSING, "Parsing target Java file")
        
        step = Step(
            id="parse_target",
            name="Parse Target",
            step_type=StepType.PARSE,
            params={"target_file": target_file},
        )
        
        result = await self.executor.execute_step_with_retry(step, self.config.generate_timeout)
        self._results.append(result)
        
        if not result.success:
            return LoopResult(
                success=False,
                message=f"Failed to parse target file: {result.error}",
                phase=LoopPhase.PARSE,
            )
        
        class_info = result.data.get("result")
        if class_info:
            self.context.set(ContextKey.CLASS_INFO, class_info)
            self._report_progress("parse_complete", {
                "class_name": class_info.get("name"),
                "method_count": len(class_info.get("methods", [])),
            })
        
        return None
    
    async def _phase_generate(self) -> Optional[LoopResult]:
        """Phase: Generate initial tests.
        
        Returns:
            LoopResult if failed, None if successful
        """
        self._update_phase(LoopPhase.GENERATE, "Generating initial tests")
        self.state_manager.transition(AgentState.GENERATING, "Generating initial test cases")
        
        step = Step(
            id="generate_initial",
            name="Generate Initial Tests",
            step_type=StepType.GENERATE,
            params={"incremental": self.config.enable_incremental},
        )
        
        result = await self.executor.execute_step_with_retry(step, self.config.generate_timeout)
        self._results.append(result)
        
        if not result.success:
            return LoopResult(
                success=False,
                message=f"Failed to generate tests: {result.error}",
                phase=LoopPhase.GENERATE,
            )
        
        test_file = result.data.get("result", {}).get("test_file")
        if test_file:
            self.context.set(ContextKey.TEST_FILE, test_file)
        
        return None
    
    async def _run_feedback_iterations(self) -> LoopResult:
        """Run the main feedback loop iterations.
        
        Returns:
            Final LoopResult
        """
        while self._current_iteration < self.config.max_iterations:
            if not await self._check_pause():
                return self._create_stopped_result()
            
            self._current_iteration += 1
            self.context.set(ContextKey.CURRENT_ITERATION, self._current_iteration)
            
            logger.info(f"[FeedbackLoop] === Iteration {self._current_iteration}/{self.config.max_iterations} ===")
            
            iteration_result = await self._run_single_iteration()
            
            if iteration_result:
                return iteration_result
            
            current_coverage = self.context.get(ContextKey.CURRENT_COVERAGE, 0.0)
            self._coverage_history.append(current_coverage)
            
            if self.config.stop_on_target_coverage and current_coverage >= self.config.target_coverage:
                return self._create_success_result()
        
        return self._create_max_iterations_result()
    
    async def _run_single_iteration(self) -> Optional[LoopResult]:
        """Run a single iteration of the feedback loop.
        
        Returns:
            LoopResult if should stop, None to continue
        """
        compile_result = await self._iteration_compile()
        if compile_result:
            return compile_result
        
        if not await self._check_pause():
            return self._create_stopped_result()
        
        test_result = await self._iteration_test()
        if test_result:
            return test_result
        
        if not await self._check_pause():
            return self._create_stopped_result()
        
        coverage_result = await self._iteration_analyze()
        if coverage_result:
            return coverage_result
        
        if not await self._check_pause():
            return self._create_stopped_result()
        
        return await self._iteration_optimize()
    
    async def _iteration_compile(self) -> Optional[LoopResult]:
        """Iteration: Compile tests.
        
        Returns:
            LoopResult if should stop, None to continue
        """
        self._update_phase(LoopPhase.COMPILE, f"Compiling tests (iteration {self._current_iteration})")
        self.state_manager.transition(AgentState.COMPILING, "Compiling generated tests")
        
        step = Step(
            id=f"compile_{self._current_iteration}",
            name="Compile Tests",
            step_type=StepType.COMPILE,
            max_retries=self.config.max_compile_attempts,
        )
        
        result = await self.executor.execute_step_with_retry(step, self.config.compile_timeout)
        self._results.append(result)
        
        if not result.success:
            errors = result.data.get("errors", [])
            self.context.set(ContextKey.COMPILATION_ERRORS, errors)
            
            return LoopResult(
                success=False,
                message=f"Compilation failed after {self.config.max_compile_attempts} attempts",
                phase=LoopPhase.COMPILE,
                iteration=self._current_iteration,
                compilation_errors=errors,
            )
        
        return None
    
    async def _iteration_test(self) -> Optional[LoopResult]:
        """Iteration: Run tests.
        
        Returns:
            LoopResult if should stop, None to continue
        """
        self._update_phase(LoopPhase.TEST, f"Running tests (iteration {self._current_iteration})")
        self.state_manager.transition(AgentState.TESTING, "Running test cases")
        
        step = Step(
            id=f"test_{self._current_iteration}",
            name="Run Tests",
            step_type=StepType.TEST,
            max_retries=self.config.max_test_attempts,
        )
        
        result = await self.executor.execute_step_with_retry(step, self.config.test_timeout)
        self._results.append(result)
        
        if not result.success:
            failures = result.data.get("failures", [])
            self.context.set(ContextKey.TEST_FAILURES, failures)
            
            return LoopResult(
                success=False,
                message=f"Tests failed: {len(failures)} failures",
                phase=LoopPhase.TEST,
                iteration=self._current_iteration,
                test_failures=failures,
            )
        
        return None
    
    async def _iteration_analyze(self) -> Optional[LoopResult]:
        """Iteration: Analyze coverage.
        
        Returns:
            LoopResult if should stop, None to continue
        """
        self._update_phase(LoopPhase.ANALYZE, f"Analyzing coverage (iteration {self._current_iteration})")
        self.state_manager.transition(AgentState.ANALYZING, "Analyzing test coverage")
        
        step = Step(
            id=f"analyze_{self._current_iteration}",
            name="Analyze Coverage",
            step_type=StepType.ANALYZE,
        )
        
        result = await self.executor.execute_step_with_retry(step, timeout=60.0)
        self._results.append(result)
        
        if result.success:
            coverage = result.data.get("result", {}).get("line_coverage", 0.0)
            self.context.set(ContextKey.CURRENT_COVERAGE, coverage)
            
            self._report_progress("coverage_update", {
                "coverage": coverage,
                "target": self.config.target_coverage,
            })
            
            if coverage >= self.config.target_coverage:
                return self._create_success_result()
        
        return None
    
    async def _iteration_optimize(self) -> Optional[LoopResult]:
        """Iteration: Generate additional tests for coverage.
        
        Returns:
            LoopResult if should stop, None to continue
        """
        if not self.config.enable_coverage_optimization:
            return None
        
        current_coverage = self.context.get(ContextKey.CURRENT_COVERAGE, 0.0)
        
        if current_coverage >= self.config.target_coverage:
            return None
        
        self._update_phase(LoopPhase.OPTIMIZE, f"Optimizing coverage (iteration {self._current_iteration})")
        self.state_manager.transition(AgentState.OPTIMIZING, "Generating additional tests for coverage")
        
        step = Step(
            id=f"optimize_{self._current_iteration}",
            name="Optimize Coverage",
            step_type=StepType.OPTIMIZE,
            params={"current_coverage": current_coverage},
        )
        
        result = await self.executor.execute_step_with_retry(step, self.config.generate_timeout)
        self._results.append(result)
        
        if not result.success:
            logger.warning(f"[FeedbackLoop] Optimization failed: {result.error}")
        
        return None
    
    def _create_success_result(self) -> LoopResult:
        """Create a success result."""
        duration_ms = int((datetime.now() - self._start_time).total_seconds() * 1000) if self._start_time else 0
        coverage = self.context.get(ContextKey.CURRENT_COVERAGE, 0.0)
        
        self.state_manager.transition(AgentState.COMPLETED, f"Target coverage reached: {coverage:.1%}")
        
        return LoopResult(
            success=True,
            message=f"Target coverage reached: {coverage:.1%}",
            phase=LoopPhase.COMPLETE,
            iteration=self._current_iteration,
            test_file=self.context.get(ContextKey.TEST_FILE),
            coverage=coverage,
            target_coverage=self.config.target_coverage,
            duration_ms=duration_ms,
            steps_completed=sum(1 for r in self._results if r.success),
            steps_failed=sum(1 for r in self._results if not r.success),
        )
    
    def _create_max_iterations_result(self) -> LoopResult:
        """Create a result for max iterations reached."""
        duration_ms = int((datetime.now() - self._start_time).total_seconds() * 1000) if self._start_time else 0
        coverage = self.context.get(ContextKey.CURRENT_COVERAGE, 0.0)
        
        self.state_manager.transition(AgentState.COMPLETED, f"Max iterations reached: {coverage:.1%}")
        
        return LoopResult(
            success=coverage > 0,
            message=f"Max iterations ({self.config.max_iterations}) reached. Coverage: {coverage:.1%}",
            phase=LoopPhase.COMPLETE,
            iteration=self._current_iteration,
            test_file=self.context.get(ContextKey.TEST_FILE),
            coverage=coverage,
            target_coverage=self.config.target_coverage,
            duration_ms=duration_ms,
            steps_completed=sum(1 for r in self._results if r.success),
            steps_failed=sum(1 for r in self._results if not r.success),
        )
    
    def _create_stopped_result(self) -> LoopResult:
        """Create a result for user stop."""
        duration_ms = int((datetime.now() - self._start_time).total_seconds() * 1000) if self._start_time else 0
        coverage = self.context.get(ContextKey.CURRENT_COVERAGE, 0.0)
        
        self.state_manager.transition(AgentState.PAUSED, "Stopped by user")
        
        return LoopResult(
            success=False,
            message="Stopped by user",
            phase=self._current_phase,
            iteration=self._current_iteration,
            test_file=self.context.get(ContextKey.TEST_FILE),
            coverage=coverage,
            target_coverage=self.config.target_coverage,
            duration_ms=duration_ms,
            steps_completed=sum(1 for r in self._results if r.success),
            steps_failed=sum(1 for r in self._results if not r.success),
        )
    
    def _create_error_result(self, error: str) -> LoopResult:
        """Create an error result."""
        duration_ms = int((datetime.now() - self._start_time).total_seconds() * 1000) if self._start_time else 0
        
        self.state_manager.transition(AgentState.FAILED, error)
        
        return LoopResult(
            success=False,
            message=f"Error: {error}",
            phase=LoopPhase.ERROR,
            iteration=self._current_iteration,
            duration_ms=duration_ms,
            steps_completed=sum(1 for r in self._results if r.success),
            steps_failed=sum(1 for r in self._results if not r.success),
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get feedback loop statistics."""
        return {
            "current_iteration": self._current_iteration,
            "current_phase": self._current_phase.name,
            "total_steps": len(self._results),
            "successful_steps": sum(1 for r in self._results if r.success),
            "failed_steps": sum(1 for r in self._results if not r.success),
            "coverage_history": self._coverage_history,
            "config": self.config.to_dict(),
        }
