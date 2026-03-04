"""Feedback Loop Executor - Main control flow for ReAct agent."""

import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from pyutagent.core.protocols import AgentState, AgentResult

logger = logging.getLogger(__name__)


class FeedbackLoopExecutor:
    """Executes the main feedback loop for test generation.
    
    The feedback loop follows this pattern:
    1. Parse target Java file (with retry)
    2. Generate initial tests (with retry)
    3. Compile tests -> if fails, AI analyzes & fixes -> retry
    4. Run tests -> if fails, AI analyzes & fixes -> retry
    5. Check coverage -> if < target, generate additional tests -> back to 3
    6. Repeat until success or user stops
    """
    
    def __init__(self, agent_core: Any, step_executor: Any):
        """Initialize feedback loop executor.
        
        Args:
            agent_core: AgentCore instance
            step_executor: StepExecutor instance
        """
        self.agent_core = agent_core
        self.step_executor = step_executor
        
        logger.debug("[FeedbackLoopExecutor] Initialized")
    
    async def run_feedback_loop(self, target_file: str) -> AgentResult:
        """Run the complete feedback loop for UT generation.
        
        Args:
            target_file: Path to target Java file
            
        Returns:
            AgentResult with final status
        """
        logger.info(f"[FeedbackLoopExecutor] 🎯 Starting test generation for: {Path(target_file).name}")
        logger.info(f"[FeedbackLoopExecutor] 📊 Configuration - MaxIterations: {self.agent_core.max_iterations}, TargetCoverage: {self.agent_core.target_coverage:.1%}")
        
        await self._check_pause()
        if self.agent_core._terminated:
            return self.agent_core._create_terminated_result("before starting")
        
        parse_result = await self._phase_parse_target(target_file)
        if not parse_result.success:
            return parse_result
        
        generate_result = await self._phase_generate_initial_tests()
        if not generate_result.success:
            return generate_result
        
        self._save_initial_checkpoint(target_file)
        
        return await self._phase_feedback_loop()
    
    async def _phase_parse_target(self, target_file: str) -> AgentResult:
        """Phase 1: Parse target Java file.
        
        Args:
            target_file: Path to the target Java file
            
        Returns:
            AgentResult with parsing results
        """
        self.agent_core._update_state(AgentState.PARSING, "📖 Step 1/6: Parsing target Java file...")
        logger.info("[FeedbackLoopExecutor] 📖 Step 1: Parsing target file")
        
        parse_result = await self.step_executor.execute_with_recovery(
            self.step_executor.parse_target_file,
            target_file,
            step_name="parsing"
        )
        
        if not parse_result.success or self.agent_core._stop_requested:
            logger.error(f"[FeedbackLoopExecutor] Failed to parse target file - {parse_result.message}")
            return AgentResult(
                success=False,
                message=f"Failed to parse target file after all recovery attempts: {parse_result.message}",
                errors=[parse_result.message]
            )
        
        self.agent_core.target_class_info = parse_result.data.get("class_info")
        self.agent_core.working_memory.current_file = target_file
        class_name = self.agent_core.target_class_info.get('name', 'unknown')
        method_count = len(self.agent_core.target_class_info.get('methods', []))
        logger.info(f"[FeedbackLoopExecutor] ✅ Parsing complete - Class: {class_name}, Methods: {method_count}")
        
        return AgentResult(success=True, state=AgentState.PARSING)
    
    async def _phase_generate_initial_tests(self) -> AgentResult:
        """Phase 2: Generate initial tests.
        
        Returns:
            AgentResult with generation results
        """
        class_name = self.agent_core.target_class_info.get('name', 'unknown')
        method_count = len(self.agent_core.target_class_info.get('methods', []))
        
        self.agent_core._update_state(AgentState.GENERATING, f"✨ Step 2/6: Generating initial tests for {class_name}...")
        logger.info("[FeedbackLoopExecutor] ✨ Step 2: Generating initial tests")
        logger.info(f"[FeedbackLoopExecutor] 🤖 Calling LLM to generate tests for {class_name} with {method_count} methods...")
        
        generate_result = await self.step_executor.execute_with_recovery(
            self.step_executor.generate_initial_tests,
            step_name="generating initial tests"
        )
        
        if not generate_result.success or self.agent_core._stop_requested:
            logger.error(f"[FeedbackLoopExecutor] Failed to generate initial tests - {generate_result.message}")
            return AgentResult(
                success=False,
                message=f"Failed to generate tests after all recovery attempts: {generate_result.message}",
                errors=[generate_result.message]
            )
        
        self.agent_core.current_test_file = generate_result.data.get("test_file")
        logger.info(f"[FeedbackLoopExecutor] ✅ Initial test generation complete - TestFile: {self.agent_core.current_test_file}")
        
        return AgentResult(success=True, state=AgentState.GENERATING)
    
    def _save_initial_checkpoint(self, target_file: str):
        """Save checkpoint after initial test generation.
        
        Args:
            target_file: Path to the target file
        """
        self.agent_core.checkpoint_manager.save_checkpoint(
            step="initial_generation",
            iteration=0,
            state={
                "target_file": target_file,
                "test_file": self.agent_core.current_test_file,
                "class_info": self.agent_core.target_class_info
            }
        )
    
    async def _phase_feedback_loop(self) -> AgentResult:
        """Phase 3-6: Compile-Test-Analyze-Optimize loop.
        
        Returns:
            AgentResult with final results
        """
        loop_start_time = asyncio.get_event_loop().time()
        
        self.agent_core._update_state(AgentState.COMPILING, "🔨 Step 3/6: Compiling generated tests...")
        logger.info("[FeedbackLoopExecutor] 🔨 Step 3: Starting compile-test loop")
        
        while not self.agent_core._stop_requested and not self.agent_core._terminated:
            if await self._check_should_stop("during pause"):
                break
            
            self.agent_core.current_iteration += 1
            self.agent_core.working_memory.increment_iteration()
            
            logger.info(f"[FeedbackLoopExecutor] 🔄 ===== Iteration {self.agent_core.current_iteration}/{self.agent_core.max_iterations} started =====")
            
            if self._should_stop_iteration():
                break
            
            iteration_result = await self._execute_iteration()
            if iteration_result:
                return iteration_result
        
        return self._create_final_result(loop_start_time)
    
    def _should_stop_iteration(self) -> bool:
        """Check if iteration should stop based on termination conditions.
        
        Returns:
            True if should stop, False otherwise
        """
        if self.agent_core.current_iteration > self.agent_core.max_iterations:
            logger.warning(f"[FeedbackLoopExecutor] Max iterations reached - Max: {self.agent_core.max_iterations}, FinalCoverage: {self.agent_core.working_memory.current_coverage:.1%}")
            self.agent_core._update_state(
                AgentState.COMPLETED,
                f"Max iterations ({self.agent_core.max_iterations}) reached. Final coverage: {self.agent_core.working_memory.current_coverage:.1%}"
            )
            return True
        
        if self.agent_core.working_memory.current_coverage >= self.agent_core.target_coverage:
            logger.info(f"[FeedbackLoopExecutor] Target coverage reached - Current: {self.agent_core.working_memory.current_coverage:.1%}, Target: {self.agent_core.target_coverage:.1%}")
            self.agent_core._update_state(
                AgentState.COMPLETED,
                f"Target coverage reached: {self.agent_core.working_memory.current_coverage:.1%}"
            )
            return True
        
        return False
    
    async def _check_should_stop(self, context: str) -> bool:
        """Check if execution should stop (pause/terminate).
        
        Args:
            context: Context for logging
            
        Returns:
            True if should stop, False otherwise
        """
        await self._check_pause()
        if self.agent_core._terminated:
            logger.info(f"[FeedbackLoopExecutor] ⏹️ Terminated {context}")
            return True
        return False
    
    async def _check_pause(self) -> None:
        """Check if agent should pause and wait."""
        await self.agent_core._check_pause()
    
    async def _execute_iteration(self) -> Optional[AgentResult]:
        """Execute one complete iteration of the feedback loop.
        
        Returns:
            AgentResult if iteration completes or fails, None to continue
        """
        if await self._check_should_stop("before compilation"):
            return None
        
        if not await self._iteration_compile():
            return None
        
        if await self._check_should_stop("before testing"):
            return None
        
        if not await self._iteration_test():
            return None
        
        if await self._check_should_stop("before coverage analysis"):
            return None
        
        coverage_result = await self._iteration_coverage()
        if not coverage_result:
            return None
        
        current_coverage = coverage_result.data.get("line_coverage", 0.0)
        
        if current_coverage >= self.agent_core.target_coverage:
            return self.agent_core._create_success_result(current_coverage)
        
        if await self._check_should_stop("before generating additional tests"):
            return None
        
        if not await self._iteration_generate_additional(current_coverage):
            return None
        
        logger.info(f"[FeedbackLoopExecutor] ✅ ===== Iteration {self.agent_core.current_iteration} complete =====")
        return None
    
    async def _iteration_compile(self) -> bool:
        """Execute compilation step.
        
        Returns:
            True if successful, False to retry
        """
        logger.info(f"[FeedbackLoopExecutor] 🔨 Step 3: Compiling tests (Iteration {self.agent_core.current_iteration})")
        compile_success = await self.step_executor.compile_with_recovery()
        
        if not compile_success or self.agent_core._stop_requested or self.agent_core._terminated:
            if self.agent_core._stop_requested or self.agent_core._terminated:
                logger.info("[FeedbackLoopExecutor] ⏹️ User stopped - Compilation phase")
                return False
            logger.warning("[FeedbackLoopExecutor] ⚠️ Compilation failed, preparing to retry...")
            return False
        
        return True
    
    async def _iteration_test(self) -> bool:
        """Execute test running step.
        
        Returns:
            True if successful, False to retry
        """
        self.agent_core._update_state(AgentState.TESTING, f"🧪 Step 4/6: Running tests (Iteration {self.agent_core.current_iteration})...")
        logger.info(f"[FeedbackLoopExecutor] 🧪 Step 4: Running tests (Iteration {self.agent_core.current_iteration})")
        
        test_success = await self.step_executor.run_tests_with_recovery()
        
        if not test_success or self.agent_core._stop_requested or self.agent_core._terminated:
            if self.agent_core._stop_requested or self.agent_core._terminated:
                logger.info("[FeedbackLoopExecutor] ⏹️ User stopped - Test phase")
                return False
            logger.warning("[FeedbackLoopExecutor] ⚠️ Tests failed, preparing to retry...")
            return False
        
        return True
    
    async def _iteration_coverage(self) -> Optional[Any]:
        """Execute coverage analysis step.
        
        Returns:
            StepResult if successful, None to retry
        """
        self.agent_core._update_state(AgentState.ANALYZING, f"📊 Step 5/6: Analyzing coverage (Iteration {self.agent_core.current_iteration})...")
        logger.info(f"[FeedbackLoopExecutor] 📊 Step 5: Analyzing coverage (Iteration {self.agent_core.current_iteration})")
        
        coverage_result = await self.step_executor.execute_with_recovery(
            self.step_executor.analyze_coverage,
            step_name="analyzing coverage"
        )
        
        if not coverage_result.success:
            logger.warning("[FeedbackLoopExecutor] Coverage analysis failed, preparing to retry")
            self.agent_core._update_state(AgentState.FIXING, "Coverage analysis failed, retrying...")
            return None
        
        if coverage_result.data.get("skipped", False):
            logger.info(f"[FeedbackLoopExecutor] ⏭️ Coverage analysis skipped: {coverage_result.data.get('reason', 'unknown reason')}")
            self.agent_core._update_state(
                AgentState.ANALYZING,
                f"⏭️ Coverage analysis skipped - {coverage_result.message}"
            )
            return coverage_result
        
        current_coverage = coverage_result.data.get("line_coverage", 0.0)
        self.agent_core.working_memory.update_coverage(current_coverage)
        logger.info(f"[FeedbackLoopExecutor] 📈 Current coverage: {current_coverage:.1%} (Target: {self.agent_core.target_coverage:.1%})")
        
        if current_coverage >= self.agent_core.target_coverage:
            logger.info(f"[FeedbackLoopExecutor] 🎉 Target coverage reached at {current_coverage:.1%}, skipping additional test generation")
            return coverage_result
        
        report = coverage_result.data.get("report")
        if report:
            uncovered_lines = self.agent_core.coverage_analyzer.get_uncovered_lines(
                Path(self.agent_core.target_class_info.get("file", "")).name if self.agent_core.target_class_info else ""
            )
            if not uncovered_lines or len(uncovered_lines) == 0:
                logger.info(f"[FeedbackLoopExecutor] 🎉 No uncovered lines found, optimization complete")
                return coverage_result
        
        return coverage_result
    
    async def _iteration_generate_additional(self, current_coverage: float) -> bool:
        """Execute additional test generation step.
        
        Args:
            current_coverage: Current coverage percentage
            
        Returns:
            True if successful, False to retry
        """
        if current_coverage >= self.agent_core.target_coverage:
            logger.info(f"[FeedbackLoopExecutor] ✅ Coverage target already reached ({current_coverage:.1%} >= {self.agent_core.target_coverage:.1%}), skipping additional tests")
            return True
        
        report = None
        try:
            target_file_name = Path(self.agent_core.target_class_info.get("file", "")).name if self.agent_core.target_class_info else ""
            uncovered_lines = self.agent_core.coverage_analyzer.get_uncovered_lines(target_file_name)
            
            if not uncovered_lines or len(uncovered_lines) == 0:
                logger.info(f"[FeedbackLoopExecutor] ✅ No uncovered lines found, all code is covered ({current_coverage:.1%})")
                return True
            
            logger.info(f"[FeedbackLoopExecutor] 📊 Found {len(uncovered_lines)} uncovered lines to target")
        except Exception as e:
            logger.warning(f"[FeedbackLoopExecutor] Failed to get uncovered lines: {e}, will attempt generation anyway")
        
        logger.info(f"[FeedbackLoopExecutor] 🚀 Step 6: Generating additional tests - Coverage: {current_coverage:.1%} < Target: {self.agent_core.target_coverage:.1%}")
        self.agent_core._update_state(
            AgentState.OPTIMIZING,
            f"🚀 Coverage {current_coverage:.1%} < target {self.agent_core.target_coverage:.1%}, calling LLM to generate additional tests..."
        )
        
        additional_result = await self.step_executor.execute_with_recovery(
            self.step_executor.generate_additional_tests,
            {"line_coverage": current_coverage},
            step_name="generating additional tests"
        )
        
        if not additional_result.success:
            logger.warning("[FeedbackLoopExecutor] ⚠️ Additional test generation failed, preparing to retry...")
            self.agent_core._update_state(AgentState.FIXING, "⚠️ Additional test generation failed, retrying...")
            return False
        
        return True
    
    def _create_final_result(self, loop_start_time: float) -> AgentResult:
        """Create final result after loop completion.
        
        Args:
            loop_start_time: Start time of the loop
            
        Returns:
            AgentResult with final status
        """
        final_coverage = self.agent_core.working_memory.current_coverage
        elapsed = asyncio.get_event_loop().time() - loop_start_time
        
        if self.agent_core._terminated:
            logger.info(f"[FeedbackLoopExecutor] ⏹️ User terminated - Iterations: {self.agent_core.current_iteration}, Coverage: {final_coverage:.1%}, Time: {elapsed:.1f}s")
            return AgentResult(
                success=False,
                message="Generation terminated by user",
                test_file=self.agent_core.current_test_file,
                coverage=final_coverage,
                iterations=self.agent_core.current_iteration,
                state=AgentState.FAILED
            )
        
        if self.agent_core._stop_requested:
            logger.info(f"[FeedbackLoopExecutor] ⏹️ User stopped - Iterations: {self.agent_core.current_iteration}, Coverage: {final_coverage:.1%}, Time: {elapsed:.1f}s")
            return AgentResult(
                success=False,
                message="Generation stopped by user",
                test_file=self.agent_core.current_test_file,
                coverage=final_coverage,
                iterations=self.agent_core.current_iteration,
                state=AgentState.PAUSED
            )
        
        logger.info(f"[FeedbackLoopExecutor] ✨ Feedback loop completed - Iterations: {self.agent_core.current_iteration}, Coverage: {final_coverage:.1%}, Time: {elapsed:.1f}s")
        return AgentResult(
            success=final_coverage > 0,
            message=f"Completed after {self.agent_core.current_iteration} iterations with {final_coverage:.1%} coverage",
            test_file=self.agent_core.current_test_file,
            coverage=final_coverage,
            iterations=self.agent_core.current_iteration
        )
