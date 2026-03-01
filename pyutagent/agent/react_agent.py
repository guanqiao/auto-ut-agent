"""ReAct Agent for UT generation with self-feedback loop and infinite retry."""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import asyncio

from .base_agent import BaseAgent, AgentState, AgentResult, StepResult

logger = logging.getLogger(__name__)
from .prompts import PromptBuilder
from .actions import ActionRegistry
from ..core.error_recovery import ErrorRecoveryManager, ErrorCategory
from ..core.retry_manager import InfiniteRetryManager, RetryConfig, RetryStrategy
from ..core.container import Container, get_container
from ..tools.java_parser import JavaCodeParser
from ..tools.maven_tools import MavenRunner, CoverageAnalyzer, ProjectScanner
from ..tools.aider_integration import AiderCodeFixer, AiderConfig
from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient
from ..core.config import get_settings


class ReActAgent(BaseAgent):
    """ReAct agent for iterative UT generation with feedback loop.
    
    Key features:
    - Infinite retry until success or user stops
    - AI-powered error recovery for all error types
    - Local + LLM double-layer error analysis
    - Automatic strategy adjustment based on failure history
    - Dependency injection for better testability
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        container: Optional[Container] = None
    ):
        """Initialize ReAct agent.
        
        Args:
            llm_client: LLM client for generation
            working_memory: Working memory for context
            project_path: Path to the project
            progress_callback: Optional callback for progress updates
            container: Optional dependency injection container
        """
        super().__init__(llm_client, working_memory, project_path, progress_callback)
        
        self._container = container or get_container()
        
        logger.info(f"[ReActAgent] Initializing agent - Project: {project_path}")
        
        self._init_dependencies(project_path)
        
        logger.info("[ReActAgent] Initialization complete")
    
    def _init_dependencies(self, project_path: str):
        """Initialize dependencies from container or create defaults.
        
        Args:
            project_path: Path to the project
        """
        self.prompt_builder = self._try_resolve(PromptBuilder)
        if not self.prompt_builder:
            self.prompt_builder = PromptBuilder()
            logger.debug("[ReActAgent] Created default PromptBuilder")
        
        self.action_registry = self._try_resolve(ActionRegistry)
        if not self.action_registry:
            self.action_registry = ActionRegistry()
            logger.debug("[ReActAgent] Created default ActionRegistry")
        
        self.java_parser = self._try_resolve(JavaCodeParser)
        if not self.java_parser:
            self.java_parser = JavaCodeParser()
            logger.debug("[ReActAgent] Created default JavaCodeParser")
        
        self.maven_runner = self._try_resolve(MavenRunner)
        if not self.maven_runner:
            self.maven_runner = MavenRunner(project_path)
            logger.debug("[ReActAgent] Created default MavenRunner")
        
        self.coverage_analyzer = self._try_resolve(CoverageAnalyzer)
        if not self.coverage_analyzer:
            self.coverage_analyzer = CoverageAnalyzer(project_path)
            logger.debug("[ReActAgent] Created default CoverageAnalyzer")
        
        self.project_scanner = self._try_resolve(ProjectScanner)
        if not self.project_scanner:
            self.project_scanner = ProjectScanner(project_path)
            logger.debug("[ReActAgent] Created default ProjectScanner")
        
        logger.debug("[ReActAgent] Dependencies initialized from container")
        
        self.error_recovery = ErrorRecoveryManager(
            llm_client=self.llm_client,
            project_path=self.project_path,
            prompt_builder=self.prompt_builder,
            progress_callback=self._on_recovery_progress
        )
        
        self._init_aider_fixer()
        
        retry_config = RetryConfig(
            strategy=RetryStrategy.ADAPTIVE,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=1.5
        )
        self.retry_manager = InfiniteRetryManager(retry_config)
        
        logger.debug(f"[ReActAgent] Retry manager initialized - Strategy: {retry_config.strategy}, BaseDelay: {retry_config.base_delay}s")
        
        self.current_test_file: Optional[str] = None
        self.target_class_info: Optional[Dict[str, Any]] = None
        self._stop_requested = False
    
    def _init_aider_fixer(self):
        """Initialize AiderCodeFixer for enhanced error fixing."""
        self.aider_fixer = self._try_resolve(AiderCodeFixer)
        if not self.aider_fixer:
            try:
                aider_config = self._try_resolve(AiderConfig)
                if not aider_config:
                    aider_config = AiderConfig()
                self.aider_fixer = AiderCodeFixer(
                    llm_client=self.llm_client,
                    config=aider_config
                )
                logger.debug("[ReActAgent] Created default AiderCodeFixer")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"[ReActAgent] Aider dependencies not available: {e}")
                self.aider_fixer = None
            except ValueError as e:
                logger.warning(f"[ReActAgent] Invalid Aider configuration: {e}")
                self.aider_fixer = None
            except Exception as e:
                logger.warning(f"[ReActAgent] Failed to create AiderCodeFixer: {e}")
                self.aider_fixer = None
        else:
            logger.debug("[ReActAgent] Resolved AiderCodeFixer from container")
    
    def _try_resolve(self, component_type):
        """Try to resolve a component from the container.

        Args:
            component_type: The type to resolve

        Returns:
            The resolved instance or None
        """
        try:
            return self._container.resolve(component_type)
        except KeyError:
            return None
        except Exception as e:
            logger.debug(f"[ReActAgent] Failed to resolve {component_type}: {e}")
            return None
    
    def _on_recovery_progress(self, state: str, message: str):
        """Handle recovery progress updates."""
        logger.info(f"[ReActAgent] Recovery progress - State: {state}, Message: {message}")
        self._update_state(AgentState.FIXING, f"[{state}] {message}")
    
    def stop(self):
        """Stop agent execution."""
        logger.info("[ReActAgent] Stopping agent execution")
        self._stop_requested = True
        self.retry_manager.stop()
        self.error_recovery.clear_history()
        super().pause()
    
    def reset(self):
        """Reset agent state."""
        logger.info("[ReActAgent] Resetting agent state")
        self._stop_requested = False
        self.retry_manager.reset()
        self.error_recovery.clear_history()
    
    async def generate_tests(self, target_file: str) -> AgentResult:
        """Generate tests for a target file with feedback loop."""
        logger.info(f"[ReActAgent] Starting test generation - Target: {target_file}")
        return await self.run_feedback_loop(target_file)
    
    async def run_feedback_loop(self, target_file: str) -> AgentResult:
        """Run the complete feedback loop for UT generation with infinite retry.
        
        The loop follows this pattern:
        1. Parse target Java file (with retry)
        2. Generate initial tests (with retry)
        3. Compile tests -> if fails, AI analyzes & fixes -> retry
        4. Run tests -> if fails, AI analyzes & fixes -> retry
        5. Check coverage -> if < target, generate additional tests -> back to 3
        6. Repeat until success or user stops
        """
        self._stop_requested = False
        logger.info(f"[ReActAgent] Starting feedback loop - Target: {target_file}, MaxIterations: {self.max_iterations}, TargetCoverage: {self.target_coverage:.1%}")
        
        logger.info("[ReActAgent] Step 1: Parsing target file")
        parse_result = await self._execute_with_recovery(
            self._parse_target_file,
            target_file,
            step_name="parsing"
        )
        
        if not parse_result.success or self._stop_requested:
            logger.error(f"[ReActAgent] Failed to parse target file - {parse_result.message}")
            return AgentResult(
                success=False,
                message=f"Failed to parse target file after all recovery attempts: {parse_result.message}",
                errors=[parse_result.message]
            )
        
        self.target_class_info = parse_result.data.get("class_info")
        self.working_memory.current_file = target_file
        logger.info(f"[ReActAgent] Parsing complete - Class: {self.target_class_info.get('name', 'unknown')}")
        
        logger.info("[ReActAgent] Step 2: Generating initial tests")
        generate_result = await self._execute_with_recovery(
            self._generate_initial_tests,
            step_name="generating initial tests"
        )
        
        if not generate_result.success or self._stop_requested:
            logger.error(f"[ReActAgent] Failed to generate initial tests - {generate_result.message}")
            return AgentResult(
                success=False,
                message=f"Failed to generate tests after all recovery attempts: {generate_result.message}",
                errors=[generate_result.message]
            )
        
        self.current_test_file = generate_result.data.get("test_file")
        logger.info(f"[ReActAgent] Initial test generation complete - TestFile: {self.current_test_file}")
        
        loop_start_time = asyncio.get_event_loop().time()
        
        while not self._stop_requested:
            self.current_iteration += 1
            self.working_memory.increment_iteration()
            
            logger.info(f"[ReActAgent] ===== Iteration {self.current_iteration}/{self.max_iterations} started =====")
            
            if self.current_iteration > self.max_iterations:
                logger.warning(f"[ReActAgent] Max iterations reached - Max: {self.max_iterations}, FinalCoverage: {self.working_memory.current_coverage:.1%}")
                self._update_state(
                    AgentState.COMPLETED,
                    f"Max iterations ({self.max_iterations}) reached. Final coverage: {self.working_memory.current_coverage:.1%}"
                )
                break
            
            if self.working_memory.current_coverage >= self.target_coverage:
                logger.info(f"[ReActAgent] Target coverage reached - Current: {self.working_memory.current_coverage:.1%}, Target: {self.target_coverage:.1%}")
                self._update_state(
                    AgentState.COMPLETED,
                    f"Target coverage reached: {self.working_memory.current_coverage:.1%}"
                )
                break
            
            logger.info("[ReActAgent] Step 3: Compiling tests")
            compile_success = await self._compile_with_recovery()
            if not compile_success or self._stop_requested:
                if self._stop_requested:
                    logger.info("[ReActAgent] User stopped - Compilation phase")
                    break
                logger.warning("[ReActAgent] Compilation failed, preparing to retry")
                continue
            
            logger.info("[ReActAgent] Step 4: Running tests")
            test_success = await self._run_tests_with_recovery()
            if not test_success or self._stop_requested:
                if self._stop_requested:
                    logger.info("[ReActAgent] User stopped - Test phase")
                    break
                logger.warning("[ReActAgent] Tests failed, preparing to retry")
                continue
            
            logger.info("[ReActAgent] Step 5: Analyzing coverage")
            coverage_result = await self._execute_with_recovery(
                self._analyze_coverage,
                step_name="analyzing coverage"
            )
            
            if not coverage_result.success:
                logger.warning("[ReActAgent] Coverage analysis failed, preparing to retry")
                self._update_state(AgentState.FIXING, "Coverage analysis failed, retrying...")
                continue
            
            current_coverage = coverage_result.data.get("line_coverage", 0.0)
            self.working_memory.update_coverage(current_coverage)
            logger.info(f"[ReActAgent] Current coverage - {current_coverage:.1%}")
            
            if current_coverage >= self.target_coverage:
                logger.info(f"[ReActAgent] Target coverage reached - {current_coverage:.1%}")
                self._update_state(
                    AgentState.COMPLETED,
                    f"Target coverage reached: {current_coverage:.1%}"
                )
                return AgentResult(
                    success=True,
                    message=f"Successfully generated tests with {current_coverage:.1%} coverage",
                    test_file=self.current_test_file,
                    coverage=current_coverage,
                    iterations=self.current_iteration
                )
            
            logger.info(f"[ReActAgent] Step 6: Generating additional tests - Coverage: {current_coverage:.1%} < Target: {self.target_coverage:.1%}")
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
                logger.warning("[ReActAgent] Additional test generation failed, preparing to retry")
                self._update_state(AgentState.FIXING, "Additional test generation failed, retrying...")
                continue
            
            logger.info(f"[ReActAgent] ===== Iteration {self.current_iteration} complete =====")
        
        final_coverage = self.working_memory.current_coverage
        elapsed = asyncio.get_event_loop().time() - loop_start_time
        
        if self._stop_requested:
            logger.info(f"[ReActAgent] User stopped - Iterations: {self.current_iteration}, Coverage: {final_coverage:.1%}, Time: {elapsed:.1f}s")
            return AgentResult(
                success=False,
                message="Generation stopped by user",
                test_file=self.current_test_file,
                coverage=final_coverage,
                iterations=self.current_iteration
            )
        
        logger.info(f"[ReActAgent] Feedback loop ended - Iterations: {self.current_iteration}, Coverage: {final_coverage:.1%}, Time: {elapsed:.1f}s")
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
        
        logger.info(f"[ReActAgent] Starting step execution - Step: {step_name}")
        
        while not self._stop_requested:
            attempt += 1
            
            logger.debug(f"[ReActAgent] Step attempt - Step: {step_name}, Attempt: {attempt}")
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                if result.success:
                    logger.info(f"[ReActAgent] Step executed successfully - Step: {step_name}, Attempt: {attempt}")
                    return result
                else:
                    logger.warning(f"[ReActAgent] Step returned failure - Step: {step_name}, Attempt: {attempt}, Message: {result.message}")
                    error = Exception(result.message)
                    recovery_result = await self._try_recover(
                        error,
                        {"step": step_name, "attempt": attempt, "result": result}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error(f"[ReActAgent] Recovery failed, step terminated - Step: {step_name}")
                        return StepResult(
                            success=False,
                            state=AgentState.FAILED,
                            message=f"Recovery failed for {step_name}"
                        )
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[ReActAgent] Applying recovery action - Action: {action}")
                    
                    if action == "fix":
                        fixed_code = recovery_result.get("fixed_code")
                        if fixed_code:
                            await self._write_test_file(fixed_code)
                    elif action == "reset":
                        logger.info("[ReActAgent] Resetting and regenerating")
                        return await self._execute_with_recovery(
                            self._generate_initial_tests,
                            step_name="regenerating tests"
                        )
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[ReActAgent] Step execution exception - Step: {step_name}, Attempt: {attempt}, Error: {e}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": step_name, "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    logger.error(f"[ReActAgent] Recovery failed, step terminated - Step: {step_name}")
                    return StepResult(
                        success=False,
                        state=AgentState.FAILED,
                        message=f"Recovery failed for {step_name}: {str(e)}"
                    )
                
                action = recovery_result.get("action", "retry")
                if action == "fix":
                    fixed_code = recovery_result.get("fixed_code")
                    if fixed_code:
                        await self._write_test_file(fixed_code)
                elif action == "skip":
                    logger.info(f"[ReActAgent] Skipping step - Step: {step_name}")
                    return StepResult(
                        success=True,
                        state=AgentState.COMPLETED,
                        message=f"Skipped {step_name}",
                        data={}
                    )
                
                continue
        
        logger.info(f"[ReActAgent] User stopped step - Step: {step_name}")
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
        logger.info(f"[ReActAgent] Attempting recovery - Error: {error}, Context: {context}")
        
        current_test_code = None
        if self.current_test_file:
            try:
                test_file_path = Path(self.project_path) / self.current_test_file
                if test_file_path.exists():
                    current_test_code = test_file_path.read_text(encoding='utf-8')
                    logger.debug(f"[ReActAgent] Read current test code - Length: {len(current_test_code)}")
            except Exception as e:
                logger.warning(f"[ReActAgent] Failed to read test code: {e}")
        
        recovery_result = await self.error_recovery.recover(
            error,
            error_context=context,
            current_test_code=current_test_code,
            target_class_info=self.target_class_info
        )
        
        logger.info(f"[ReActAgent] Recovery result - Action: {recovery_result.get('action')}, ShouldContinue: {recovery_result.get('should_continue')}")
        
        return recovery_result
    
    async def _compile_with_recovery(self) -> bool:
        """Compile tests with automatic error recovery.
        
        Returns:
            True if compilation successful
        """
        attempt = 0
        
        logger.info("[ReActAgent] Starting test compilation (with recovery)")
        
        while not self._stop_requested:
            attempt += 1
            self._update_state(AgentState.COMPILING, f"Attempt {attempt}: Compiling tests...")
            
            logger.debug(f"[ReActAgent] Compilation attempt {attempt}")
            
            try:
                result = await self._compile_tests()
                
                if result.success:
                    logger.info(f"[ReActAgent] Compilation successful - Attempt: {attempt}")
                    self._update_state(AgentState.COMPILING, "Compilation successful")
                    return True
                else:
                    errors = result.data.get("errors", [])
                    self._update_state(
                        AgentState.FIXING,
                        f"Compilation failed with {len(errors)} error(s). Analyzing..."
                    )
                    
                    logger.warning(f"[ReActAgent] Compilation failed - Errors: {len(errors)}")
                    
                    error = Exception("Compilation failed: " + "\n".join(errors[:3]))
                    recovery_result = await self._try_recover(
                        error,
                        {"step": "compilation", "attempt": attempt, "compiler_output": "\n".join(errors)}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error("[ReActAgent] Compilation error recovery failed")
                        self._update_state(AgentState.FAILED, "Recovery failed, cannot fix compilation errors")
                        return False
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[ReActAgent] Compilation recovery action - Action: {action}")
                    
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
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[ReActAgent] Compilation exception: {e}")
                self._update_state(AgentState.FIXING, f"Compilation error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "compilation", "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    return False
                
                continue
        
        logger.info("[ReActAgent] Compilation stopped (user request)")
        return False
    
    async def _run_tests_with_recovery(self) -> bool:
        """Run tests with automatic error recovery.
        
        Returns:
            True if tests pass
        """
        attempt = 0
        
        logger.info("[ReActAgent] Starting test execution (with recovery)")
        
        while not self._stop_requested:
            attempt += 1
            self._update_state(AgentState.TESTING, f"Attempt {attempt}: Running tests...")
            
            logger.debug(f"[ReActAgent] Test run attempt {attempt}")
            
            try:
                result = await self._run_tests()
                
                if result.success:
                    logger.info(f"[ReActAgent] All tests passed - Attempt: {attempt}")
                    self._update_state(AgentState.TESTING, "All tests passed")
                    return True
                else:
                    failures = result.data.get("failures", [])
                    self._update_state(
                        AgentState.FIXING,
                        f"{len(failures)} test(s) failed. Analyzing..."
                    )
                    
                    logger.warning(f"[ReActAgent] Tests failed - Failures: {len(failures)}")
                    
                    error = Exception(f"Test failures: {len(failures)} tests failed")
                    recovery_result = await self._try_recover(
                        error,
                        {"step": "test_execution", "attempt": attempt, "failures": failures}
                    )
                    
                    if not recovery_result.get("should_continue", True):
                        logger.error("[ReActAgent] Test failure recovery failed")
                        self._update_state(AgentState.FAILED, "Recovery failed, cannot fix test failures")
                        return False
                    
                    action = recovery_result.get("action", "retry")
                    logger.info(f"[ReActAgent] Test recovery action - Action: {action}")
                    
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
                    
                    continue
                    
            except Exception as e:
                logger.exception(f"[ReActAgent] Test execution exception: {e}")
                self._update_state(AgentState.FIXING, f"Test execution error: {str(e)}")
                
                recovery_result = await self._try_recover(
                    e,
                    {"step": "test_execution", "attempt": attempt}
                )
                
                if not recovery_result.get("should_continue", True):
                    return False
                
                continue
        
        logger.info("[ReActAgent] Test execution stopped (user request)")
        return False
    
    async def _write_test_file(self, code: str):
        """Write test code to file.
        
        Args:
            code: Test code to write
        """
        if not self.current_test_file:
            logger.warning("[ReActAgent] Cannot write test file - current_test_file is empty")
            return
        
        try:
            test_file_path = Path(self.project_path) / self.current_test_file
            test_file_path.write_text(code, encoding='utf-8')
            logger.info(f"[ReActAgent] Wrote test file - Path: {test_file_path}, Length: {len(code)}")
        except PermissionError as e:
            logger.error(f"[ReActAgent] Permission denied writing test file: {e}")
            self._update_state(AgentState.FAILED, f"Permission denied: {e}")
        except OSError as e:
            logger.error(f"[ReActAgent] OS error writing test file: {e}")
            self._update_state(AgentState.FAILED, f"File system error: {e}")
        except Exception as e:
            logger.exception(f"[ReActAgent] Failed to write test file: {e}")
            self._update_state(AgentState.FAILED, f"Failed to write test file: {e}")
    
    async def _parse_target_file(self, target_file: str) -> StepResult:
        """Parse the target Java file."""
        logger.info(f"[ReActAgent] Parsing target file - File: {target_file}")
        
        try:
            file_path = Path(self.project_path) / target_file
            if not file_path.exists():
                logger.error(f"[ReActAgent] Target file not found - Path: {file_path}")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message=f"File not found: {target_file}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            logger.debug(f"[ReActAgent] Read file content - Length: {len(source_code)}")
            
            parsed_class = self.java_parser.parse(source_code.encode('utf-8'))
            
            class_info = {
                'name': parsed_class.name,
                'package': parsed_class.package,
                'methods': [
                    {
                        'name': m.name,
                        'return_type': m.return_type,
                        'parameters': m.parameters,
                        'modifiers': m.modifiers,
                        'annotations': m.annotations,
                    }
                    for m in parsed_class.methods
                ],
                'fields': parsed_class.fields,
                'imports': parsed_class.imports,
                'annotations': parsed_class.annotations,
                'source': source_code,
            }
            
            logger.info(f"[ReActAgent] Parsing complete - Class: {class_info.get('name', 'unknown')}, Methods: {len(class_info.get('methods', []))}")
            
            return StepResult(
                success=True,
                state=AgentState.PARSING,
                message=f"Successfully parsed {class_info.get('name', 'unknown')}",
                data={"class_info": class_info, "source_code": source_code}
            )
        except Exception as e:
            logger.exception(f"[ReActAgent] Failed to parse file: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error parsing file: {str(e)}"
            )
    
    async def _generate_initial_tests(self) -> StepResult:
        """Generate initial test cases."""
        logger.info("[ReActAgent] Generating initial tests")
        
        try:
            prompt = self.prompt_builder.build_initial_test_prompt(
                class_info=self.target_class_info,
                source_code=self.target_class_info.get("source", "")
            )
            
            logger.debug(f"[ReActAgent] Initial test prompt - Length: {len(prompt)}")
            
            response = await self.llm_client.generate(prompt)
            test_code = self._extract_java_code(response)
            
            logger.debug(f"[ReActAgent] Extracted test code - Length: {len(test_code)}")
            
            class_name = self.target_class_info.get("name", "Unknown")
            test_file_name = f"{class_name}Test.java"
            
            settings = get_settings()
            test_dir = Path(self.project_path) / settings.project_paths.src_test_java
            package_path = self.target_class_info.get("package", "").replace(".", "/")
            if package_path:
                test_dir = test_dir / package_path
            
            test_dir.mkdir(parents=True, exist_ok=True)
            test_file_path = test_dir / test_file_name
            
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            self.current_test_file = str(test_file_path.relative_to(self.project_path))
            self.working_memory.add_generated_test(
                file=self.current_test_file,
                method="initial",
                code=test_code
            )
            
            logger.info(f"[ReActAgent] Initial test generation complete - TestFile: {self.current_test_file}")
            
            return StepResult(
                success=True,
                state=AgentState.GENERATING,
                message=f"Generated initial tests: {self.current_test_file}",
                data={"test_file": self.current_test_file, "test_code": test_code}
            )
        except Exception as e:
            logger.exception(f"[ReActAgent] Failed to generate initial tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating tests: {str(e)}"
            )
    
    async def _compile_tests(self) -> StepResult:
        """Compile the generated tests asynchronously."""
        logger.info("[ReActAgent] Compiling tests")

        try:
            logger.debug("[ReActAgent] Getting Maven dependency classpath")

            # Use async subprocess for Maven command
            maven_process = await asyncio.create_subprocess_exec(
                "mvn", "dependency:build-classpath", "-Dmdep.outputFile=cp.txt", "-q",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await maven_process.communicate()

            classpath = ""
            cp_file = Path(self.project_path) / "cp.txt"
            if cp_file.exists():
                classpath = cp_file.read_text(encoding='utf-8').strip()
                logger.debug(f"[ReActAgent] Classpath length: {len(classpath)}")

            settings = get_settings()
            classpath = f"{self.project_path}/{settings.project_paths.target_classes};{self.project_path}/{settings.project_paths.target_test_classes};{classpath}"

            test_file_path = Path(self.project_path) / self.current_test_file

            # Use async subprocess for javac command
            compile_process = await asyncio.create_subprocess_exec(
                "javac", "-cp", classpath,
                "-d", str(Path(self.project_path) / "target" / "test-classes"),
                str(test_file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await compile_process.communicate()

            if compile_process.returncode == 0:
                logger.info("[ReActAgent] Compilation successful")
                return StepResult(
                    success=True,
                    state=AgentState.COMPILING,
                    message="Tests compiled successfully"
                )
            else:
                errors = [stderr.decode('utf-8', errors='replace')] if stderr else ["Unknown compilation error"]
                logger.warning(f"[ReActAgent] Compilation failed - Errors: {len(errors)}")
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Compilation failed",
                    data={"errors": errors, "stdout": stdout.decode('utf-8', errors='replace') if stdout else ""}
                )
        except Exception as e:
            logger.exception(f"[ReActAgent] Compilation exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error compiling tests: {str(e)}"
            )
    
    async def _run_tests(self) -> StepResult:
        """Run the generated tests."""
        logger.info("[ReActAgent] Running tests")
        
        try:
            success = self.maven_runner.run_tests()
            
            if success:
                logger.info("[ReActAgent] All tests passed")
                return StepResult(
                    success=True,
                    state=AgentState.TESTING,
                    message="All tests passed"
                )
            else:
                failures = self._parse_test_failures()
                logger.warning(f"[ReActAgent] Tests failed - Failures: {len(failures)}")
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Some tests failed",
                    data={"failures": failures}
                )
        except Exception as e:
            logger.exception(f"[ReActAgent] Test execution exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error running tests: {str(e)}"
            )
    
    async def _analyze_coverage(self) -> StepResult:
        """Analyze test coverage."""
        logger.info("[ReActAgent] Analyzing coverage")
        
        try:
            logger.debug("[ReActAgent] Generating coverage report")
            self.maven_runner.generate_coverage()
            
            report = self.coverage_analyzer.parse_report()
            
            if report:
                logger.info(f"[ReActAgent] Coverage analysis complete - Line: {report.line_coverage:.1%}, Branch: {report.branch_coverage:.1%}, Method: {report.method_coverage:.1%}")
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
                logger.warning("[ReActAgent] Failed to parse coverage report")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message="Failed to parse coverage report"
                )
        except Exception as e:
            logger.exception(f"[ReActAgent] Coverage analysis exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error analyzing coverage: {str(e)}"
            )
    
    async def _generate_additional_tests(self, coverage_data: Dict[str, Any]) -> StepResult:
        """Generate additional tests for uncovered code."""
        logger.info("[ReActAgent] Generating additional tests")
        
        try:
            report = coverage_data.get("report")
            uncovered_info = self._get_uncovered_info(report)
            
            logger.debug(f"[ReActAgent] Uncovered info - Lines: {len(uncovered_info.get('lines', []))}")
            
            test_file_path = Path(self.project_path) / self.current_test_file
            with open(test_file_path, 'r', encoding='utf-8') as f:
                current_test_code = f.read()
            
            prompt = self.prompt_builder.build_additional_tests_prompt(
                class_info=self.target_class_info,
                existing_tests=current_test_code,
                uncovered_info=uncovered_info,
                current_coverage=coverage_data.get("line_coverage", 0.0)
            )
            
            logger.debug(f"[ReActAgent] Additional tests prompt - Length: {len(prompt)}")
            
            response = await self.llm_client.generate(prompt)
            additional_tests = self._extract_java_code(response)
            
            logger.debug(f"[ReActAgent] Extracted additional test code - Length: {len(additional_tests)}")
            
            self._append_tests_to_file(test_file_path, additional_tests)
            
            logger.info("[ReActAgent] Additional test generation complete")
            
            return StepResult(
                success=True,
                state=AgentState.OPTIMIZING,
                message="Generated additional tests for uncovered code",
                data={"additional_tests": additional_tests}
            )
        except Exception as e:
            logger.exception(f"[ReActAgent] Failed to generate additional tests: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error generating additional tests: {str(e)}"
            )
    
    def _extract_java_code(self, response: str) -> str:
        """Extract Java code from LLM response."""
        code_block_pattern = r'```(?:java)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return response.strip()
    
    def _parse_test_failures(self) -> List[Dict[str, Any]]:
        """Parse test failures from Maven output."""
        failures = []
        settings = get_settings()
        surefire_dir = Path(self.project_path) / settings.project_paths.target_surefire_reports
        
        if surefire_dir.exists():
            for report_file in surefire_dir.glob("*.txt"):
                content = report_file.read_text()
                if "FAILURE" in content or "ERROR" in content:
                    failures.append({
                        "test_name": report_file.stem,
                        "error": content[:500]
                    })
        
        logger.debug(f"[ReActAgent] Parsed test failures - Failures: {len(failures)}")
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
        
        logger.debug(f"[ReActAgent] Uncovered info - Lines: {len(uncovered_info['lines'])}")
        return uncovered_info
    
    def _append_tests_to_file(self, test_file_path: Path, additional_tests: str):
        """Append additional tests to existing test file."""
        with open(test_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        last_brace = content.rfind('}')
        if last_brace > 0:
            new_content = content[:last_brace] + "\n" + additional_tests + "\n" + content[last_brace:]
        else:
            new_content = content + "\n" + additional_tests
        
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.debug(f"[ReActAgent] Appended tests to file - Path: {test_file_path}, AddedLength: {len(additional_tests)}")
