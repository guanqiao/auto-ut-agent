"""ReAct Agent for UT generation with self-feedback loop and infinite retry."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio

from .base_agent import BaseAgent, AgentState, AgentResult, StepResult
from .prompts import PromptBuilder
from .actions import ActionRegistry
from .error_recovery import ErrorRecoveryManager, ErrorCategory
from .retry_manager import InfiniteRetryManager, RetryConfig, RetryStrategy
from ..tools.java_parser import JavaCodeParser
from ..tools.maven_tools import MavenRunner, CoverageAnalyzer, ProjectScanner
from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient


class ReActAgent(BaseAgent):
    """ReAct agent for iterative UT generation with feedback loop.
    
    Key features:
    - Infinite retry until success or user stops
    - AI-powered error recovery for all error types
    - Local + LLM double-layer error analysis
    - Automatic strategy adjustment based on failure history
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize ReAct agent."""
        super().__init__(llm_client, working_memory, project_path, progress_callback)
        
        self.prompt_builder = PromptBuilder()
        self.action_registry = ActionRegistry()
        self.java_parser = JavaCodeParser()
        self.maven_runner = MavenRunner(project_path)
        self.coverage_analyzer = CoverageAnalyzer(project_path)
        self.project_scanner = ProjectScanner(project_path)
        
        # Error recovery and retry managers
        self.error_recovery = ErrorRecoveryManager(
            llm_client=llm_client,
            project_path=project_path,
            prompt_builder=self.prompt_builder,
            progress_callback=self._on_recovery_progress
        )
        
        retry_config = RetryConfig(
            strategy=RetryStrategy.ADAPTIVE,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=1.5
        )
        self.retry_manager = InfiniteRetryManager(retry_config)
        
        self.current_test_file: Optional[str] = None
        self.target_class_info: Optional[Dict[str, Any]] = None
        self._stop_requested = False
        
    def _on_recovery_progress(self, state: str, message: str):
        """Handle recovery progress updates."""
        self._update_state(AgentState.FIXING, f"[{state}] {message}")
    
    def stop(self):
        """Stop agent execution."""
        self._stop_requested = True
        self.retry_manager.stop()
        self.error_recovery.clear_history()
        super().pause()
    
    def reset(self):
        """Reset agent state."""
        self._stop_requested = False
        self.retry_manager.reset()
        self.error_recovery.clear_history()
    
    async def generate_tests(self, target_file: str) -> AgentResult:
        """Generate tests for a target file with feedback loop."""
        return await self.run_feedback_loop(target_file)
    
    async def run_feedback_loop(self, target_file: str) -> AgentResult:
        """Run the complete feedback loop for UT generation with infinite retry.
        
        The loop follows this pattern:
        1. Parse target Java file (with retry)
        2. Generate initial tests (with retry)
        3. Compile tests → if fails, AI analyzes & fixes → retry
        4. Run tests → if fails, AI analyzes & fixes → retry
        5. Check coverage → if < target, generate additional tests → back to 3
        6. Repeat until success or user stops
        """
        self._stop_requested = False
        
        # Step 1: Parse target file (with infinite retry)
        parse_result = await self._execute_with_recovery(
            self._parse_target_file,
            target_file,
            step_name="parsing"
        )
        
        if not parse_result.success or self._stop_requested:
            return AgentResult(
                success=False,
                message=f"Failed to parse target file after all recovery attempts: {parse_result.message}",
                errors=[parse_result.message]
            )
        
        self.target_class_info = parse_result.data.get("class_info")
        self.working_memory.current_file = target_file
        
        # Step 2: Generate initial tests (with infinite retry)
        generate_result = await self._execute_with_recovery(
            self._generate_initial_tests,
            step_name="generating initial tests"
        )
        
        if not generate_result.success or self._stop_requested:
            return AgentResult(
                success=False,
                message=f"Failed to generate tests after all recovery attempts: {generate_result.message}",
                errors=[generate_result.message]
            )
        
        self.current_test_file = generate_result.data.get("test_file")
        
        # Main feedback loop with infinite retry
        while not self._stop_requested:
            self.current_iteration += 1
            self.working_memory.increment_iteration()
            
            # Check if we've reached max iterations
            if self.current_iteration > self.max_iterations:
                self._update_state(
                    AgentState.COMPLETED,
                    f"Max iterations ({self.max_iterations}) reached. Final coverage: {self.working_memory.current_coverage:.1%}"
                )
                break
            
            # Check if target coverage reached
            if self.working_memory.current_coverage >= self.target_coverage:
                self._update_state(
                    AgentState.COMPLETED,
                    f"Target coverage reached: {self.working_memory.current_coverage:.1%}"
                )
                break
            
            # Step 3: Compile tests (with infinite retry)
            compile_success = await self._compile_with_recovery()
            if not compile_success or self._stop_requested:
                if self._stop_requested:
                    break
                continue  # Retry from beginning of loop
            
            # Step 4: Run tests (with infinite retry)
            test_success = await self._run_tests_with_recovery()
            if not test_success or self._stop_requested:
                if self._stop_requested:
                    break
                continue  # Retry from beginning of loop
            
            # Step 5: Analyze coverage
            coverage_result = await self._execute_with_recovery(
                self._analyze_coverage,
                step_name="analyzing coverage"
            )
            
            if not coverage_result.success:
                self._update_state(AgentState.FIXING, "Coverage analysis failed, retrying...")
                continue
            
            current_coverage = coverage_result.data.get("line_coverage", 0.0)
            self.working_memory.update_coverage(current_coverage)
            
            # Check if target reached
            if current_coverage >= self.target_coverage:
                self._update_state(
                    AgentState.COMPLETED,
                    f"🎉 Target coverage reached: {current_coverage:.1%}"
                )
                return AgentResult(
                    success=True,
                    message=f"Successfully generated tests with {current_coverage:.1%} coverage",
                    test_file=self.current_test_file,
                    coverage=current_coverage,
                    iterations=self.current_iteration
                )
            
            # Step 6: Generate additional tests for uncovered code
            self._update_state(
                AgentState.OPTIMIZING,
                f"Coverage {current_coverage:.1%} < target {self.target_coverage:.1%}, generating additional tests"
            )
            
            additional_result = await self._execute_with_recovery(
                self._generate_additional_tests,
                coverage_result.data,
                step_name="generating additional tests"
            )
            
            if not additional_result.success:
                self._update_state(AgentState.FIXING, "Additional test generation failed, retrying...")
                continue
        
        # Loop ended (user stopped or max iterations)
        final_coverage = self.working_memory.current_coverage
        
        if self._stop_requested:
            return AgentResult(
                success=False,
                message="Generation stopped by user",
                test_file=self.current_test_file,
                coverage=final_coverage,
                iterations=self.current_iteration
            )
        
        return AgentResult(
            success=final_coverage > 0,
            message=f"Completed after {self.current_iteration} iterations with {final_coverage:.1%} coverage",
            test_file=self.current_test_file,
            coverage=final_coverage,
            iterations=self.current_iteration
        )
    
    async def _execute_with_recovery(
        self,
        operation,
        *args,
        step_name: str = "operation",
        **kwargs
    ) -> StepResult:
        """Execute an operation with automatic error recovery.
        
        Args:
            operation: The operation to execute
            *args: Positional arguments
            step_name: Name of the step for logging
            **kwargs: Keyword arguments
            
        Returns:
            StepResult
        """
        attempt = 0
        
        while not self._stop_requested:
            attempt += 1
            
            try:
                # Try to execute
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                if result.success:
                    return result
                else:
                    # Operation returned failure, try to recover
                    error = Exception(result.message)
                    recovery_result = await self._try_recover(
                        error,
                        {"step": step_name, "attempt": attempt, "result": result}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        return StepResult(
                            success=False,
                            state=AgentState.FAILED,
                            message=f"Recovery failed for {step_name}"
                        )
                    
                    # Apply recovery action
                    action = recovery_result.get("action", "retry")
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                    elif action == "reset":
                        # Reset and regenerate
                        return await self._execute_with_recovery(
                            self._generate_initial_tests,
                            step_name="regenerating tests"
                        )
                    
                    # Continue to next attempt
                    continue
                    
            except Exception as e:
                # Exception occurred, try to recover
                recovery_result = await self._try_recover(
                    e,
                    {"step": step_name, "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    return StepResult(
                        success=False,
                        state=AgentState.FAILED,
                        message=f"Recovery failed for {step_name}: {str(e)}"
                    )
                
                # Apply recovery action
                action = recovery_result.get("action", "retry")
                if action == "fix":
                    fixed_code = recovery_result.get("fixed_code")
                    if fixed_code:
                        await self._write_test_file(fixed_code)
                elif action == "skip":
                    return StepResult(
                        success=True,
                        state=AgentState.COMPLETED,
                        message=f"Skipped {step_name}",
                        data={}
                    )
                
                # Continue to next attempt
                continue
        
        # Stop requested
        return StepResult(
            success=False,
            state=AgentState.PAUSED,
            message="Operation stopped by user"
        )
    
    async def _try_recover(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to recover from an error.
        
        Args:
            error: The error that occurred
            context: Error context
            
        Returns:
            Recovery result
        """
        # Read current test code if available
        current_test_code = None
        if self.current_test_file:
            try:
                test_file_path = Path(self.project_path) / self.current_test_file
                if test_file_path.exists():
                    current_test_code = test_file_path.read_text(encoding='utf-8')
            except Exception:
                pass
        
        # Use error recovery manager
        recovery_result = await self.error_recovery.recover(
            error,
            error_context=context,
            current_test_code=current_test_code,
            target_class_info=self.target_class_info
        )
        
        return recovery_result
    
    async def _compile_with_recovery(self) -> bool:
        """Compile tests with automatic error recovery.
        
        Returns:
            True if compilation successful
        """
        attempt = 0
        
        while not self._stop_requested:
            attempt += 1
            self._update_state(AgentState.COMPILING, f"Attempt {attempt}: Compiling tests...")
            
            try:
                result = await self._compile_tests()
                
                if result.success:
                    self._update_state(AgentState.COMPILING, "✅ Compilation successful")
                    return True
                else:
                    # Compilation failed, try to fix
                    errors = result.data.get("errors", [])
                    self._update_state(
                        AgentState.FIXING,
                        f"❌ Compilation failed with {len(errors)} error(s). Analyzing..."
                    )
                    
                    # Try to recover
                    error = Exception("Compilation failed: " + "\n".join(errors[:3]))
                    recovery_result = await self._try_recover(
                        error,
                        {"step": "compilation", "attempt": attempt, "compiler_output": "\n".join(errors)}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        self._update_state(AgentState.FAILED, "Recovery failed, cannot fix compilation errors")
                        return False
                    
                    # Apply fix
                    action = recovery_result.get("action", "retry")
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                            self._update_state(AgentState.FIXING, "Applied fix, retrying compilation...")
                    elif action == "reset":
                        self._update_state(AgentState.FIXING, "Resetting and regenerating...")
                        reset_result = await self._execute_with_recovery(
                            self._generate_initial_tests,
                            step_name="regenerating after compilation failure"
                        )
                        if not reset_result.success:
                            return False
                    elif action == "fallback":
                        self._update_state(AgentState.FIXING, "Trying alternative approach...")
                        # Could try simpler test generation here
                    
                    # Continue to next attempt
                    continue
                    
            except Exception as e:
                self._update_state(AgentState.FIXING, f"Compilation error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "compilation", "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    return False
                
                continue
        
        return False
    
    async def _run_tests_with_recovery(self) -> bool:
        """Run tests with automatic error recovery.
        
        Returns:
            True if tests pass
        """
        attempt = 0
        
        while not self._stop_requested:
            attempt += 1
            self._update_state(AgentState.TESTING, f"Attempt {attempt}: Running tests...")
            
            try:
                result = await self._run_tests()
                
                if result.success:
                    self._update_state(AgentState.TESTING, "✅ All tests passed")
                    return True
                else:
                    # Tests failed, try to fix
                    failures = result.data.get("failures", [])
                    self._update_state(
                        AgentState.FIXING,
                        f"❌ {len(failures)} test(s) failed. Analyzing..."
                    )
                    
                    # Try to recover
                    error = Exception(f"Test failures: {len(failures)} tests failed")
                    recovery_result = await self._try_recover(
                        error,
                        {"step": "test_execution", "attempt": attempt, "failures": failures}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        self._update_state(AgentState.FAILED, "Recovery failed, cannot fix test failures")
                        return False
                    
                    # Apply fix
                    action = recovery_result.get("action", "retry")
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                            self._update_state(AgentState.FIXING, "Applied fix, retrying tests...")
                    elif action == "reset":
                        self._update_state(AgentState.FIXING, "Resetting and regenerating...")
                        reset_result = await self._execute_with_recovery(
                            self._generate_initial_tests,
                            step_name="regenerating after test failure"
                        )
                        if not reset_result.success:
                            return False
                    
                    # Continue to next attempt
                    continue
                    
            except Exception as e:
                self._update_state(AgentState.FIXING, f"Test execution error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "test_execution", "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    return False
                
                continue
        
        return False
    
    async def _write_test_file(self, code: str):
        """Write test code to file.
        
        Args:
            code: Test code to write
        """
        if not self.current_test_file:
            return
        
        try:
            test_file_path = Path(self.project_path) / self.current_test_file
            test_file_path.write_text(code, encoding='utf-8')
        except Exception as e:
            self._update_state(AgentState.FAILED, f"Failed to write test file: {e}")
    
    async def _parse_target_file(self, target_file: str) -> StepResult:
        """Parse the target Java file."""
        try:
            file_path = Path(self.project_path) / target_file
            if not file_path.exists():
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"File not found: {target_file}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            class_info = self.java_parser.parse_class(source_code)
            
            return StepResult(
                success=True,
                state=AgentState.PARSING,
                message=f"Successfully parsed {class_info.get('name', 'unknown')}",
                data={"class_info": class_info, "source_code": source_code}
            )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error parsing file: {str(e)}"
            )
    
    async def _generate_initial_tests(self) -> StepResult:
        """Generate initial test cases."""
        try:
            prompt = self.prompt_builder.build_initial_test_prompt(
                class_info=self.target_class_info,
                source_code=self.target_class_info.get("source", "")
            )
            
            response = await self.llm_client.generate(prompt)
            test_code = self._extract_java_code(response)
            
            # Determine test file path
            class_name = self.target_class_info.get("name", "Unknown")
            test_file_name = f"{class_name}Test.java"
            
            # Find appropriate test directory
            test_dir = Path(self.project_path) / "src" / "test" / "java"
            package_path = self.target_class_info.get("package", "").replace(".", "/")
            if package_path:
                test_dir = test_dir / package_path
            
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file_path = test_dir / test_file_name
            
            # Write test file
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            self.current_test_file = str(test_file_path.relative_to(self.project_path))
            self.working_memory.add_generated_test(
                file=self.current_test_file,
                method="initial",
                code=test_code
            )
            
            return StepResult(
                success=True,
                state=AgentState.GENERATING,
                message=f"Generated initial tests: {self.current_test_file}",
                data={"test_file": self.current_test_file, "test_code": test_code}
            )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating tests: {str(e)}"
            )
    
    async def _compile_tests(self) -> StepResult:
        """Compile the generated tests."""
        try:
            import subprocess
            
            # Get classpath
            classpath_result = subprocess.run(
                ["mvn", "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q"],
                cwd=self.project_path,
                capture_output=True,
                text=True
            )
            
            classpath = ""
            cp_file = Path(self.project_path) / "cp.txt"
            if cp_file.exists():
                classpath = cp_file.read_text().strip()
            
            # Add target/classes and target/test-classes
            classpath = f"{self.project_path}/target/classes;{self.project_path}/target/test-classes;{classpath}"
            
            # Compile test file
            test_file_path = Path(self.project_path) / self.current_test_file
            compile_cmd = [
                "javac", "-cp", classpath,
                "-d", str(Path(self.project_path) / "target" / "test-classes"),
                str(test_file_path)
            ]
            
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return StepResult(
                    success=True,
                    state=AgentState.COMPILING,
                    message="Tests compiled successfully"
                )
            else:
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Compilation failed",
                    data={"errors": [result.stderr], "stdout": result.stdout}
                )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error compiling tests: {str(e)}"
            )
    
    async def _run_tests(self) -> StepResult:
        """Run the generated tests."""
        try:
            success = self.maven_runner.run_tests()
            
            if success:
                return StepResult(
                    success=True,
                    state=AgentState.TESTING,
                    message="All tests passed"
                )
            else:
                # Parse test failures
                failures = self._parse_test_failures()
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Some tests failed",
                    data={"failures": failures}
                )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error running tests: {str(e)}"
            )
    
    async def _analyze_coverage(self) -> StepResult:
        """Analyze test coverage."""
        try:
            # Generate coverage report
            self.maven_runner.generate_coverage()
            
            # Parse report
            report = self.coverage_analyzer.parse_report()
            
            if report:
                return StepResult(
                    success=True,
                    state=AgentState.ANALYZING,
                    message=f"Coverage: {report.line_coverage:.1%}",
                    data={
                        "line_coverage": report.line_coverage,
                        "branch_coverage": report.branch_coverage,
                        "method_coverage": report.method_coverage,
                        "report": report
                    }
                )
            else:
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message="Failed to parse coverage report"
                )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error analyzing coverage: {str(e)}"
            )
    
    async def _generate_additional_tests(self, coverage_data: Dict[str, Any]) -> StepResult:
        """Generate additional tests for uncovered code."""
        try:
            # Get uncovered lines/methods
            report = coverage_data.get("report")
            uncovered_info = self._get_uncovered_info(report)
            
            # Read current test code
            test_file_path = Path(self.project_path) / self.current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            prompt = self.prompt_builder.build_additional_tests_prompt(
                class_info=self.target_class_info,
                existing_tests=current_test_code,
                uncovered_info=uncovered_info,
                current_coverage=coverage_data.get("line_coverage", 0.0)
            )
            
            response = await self.llm_client.generate(prompt)
            additional_tests = self._extract_java_code(response)
            
            # Append to existing test file
            self._append_tests_to_file(test_file_path, additional_tests)
            
            return StepResult(
                success=True,
                state=AgentState.OPTIMIZING,
                message="Generated additional tests for uncovered code",
                data={"additional_tests": additional_tests}
            )
        except Exception as e:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating additional tests: {str(e)}"
            )
    
    def _extract_java_code(self, response: str) -> str:
        """Extract Java code from LLM response."""
        # Try to extract code from markdown code blocks
        code_block_pattern = r'```(?:java)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return the whole response
        return response.strip()
    
    def _parse_test_failures(self) -> List[Dict[str, Any]]:
        """Parse test failures from Maven output."""
        failures = []
        surefire_dir = Path(self.project_path) / "target" / "surefire-reports"
        
        if surefire_dir.exists():
            for report_file in surefire_dir.glob("*.txt"):
                content = report_file.read_text()
                if "FAILURE" in content or "ERROR" in content:
                    failures.append({
                        "test_name": report_file.stem,
                        "error": content[:500]
                    })
        
        return failures
    
    def _get_uncovered_info(self, report) -> Dict[str, Any]:
        """Get information about uncovered code."""
        uncovered_info = {
            "methods": [],
            "lines": [],
            "branches": []
        }
        
        if report and report.files:
            for file_coverage in report.files:
                for line_num, is_covered in file_coverage.lines:
                    if not is_covered:
                        uncovered_info["lines"].append(line_num)
        
        return uncovered_info
    
    def _append_tests_to_file(self, test_file_path: Path, additional_tests: str):
        """Append additional tests to existing test file."""
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the last closing brace
        last_brace = content.rfind('}')
        if last_brace > 0:
            new_content = content[:last_brace] + "\n" + additional_tests + "\n" + content[last_brace:]
        else:
            new_content = content + "\n" + additional_tests
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)