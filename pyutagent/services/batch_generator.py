"""Batch generator for generating tests for multiple files in parallel."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch generation.
    
    Attributes:
        parallel_workers: Number of parallel workers (0 = unlimited)
        timeout_per_file: Timeout in seconds for each file
        continue_on_error: Whether to continue if a file fails
        coverage_target: Target coverage percentage
        max_iterations: Maximum iterations per file
        defer_compilation: Whether to defer compilation until all files are generated
        compile_only_at_end: If True, skip compilation/verification during generation
        incremental_mode: Enable incremental test generation (preserve existing passing tests)
        preserve_passing_tests: Whether to preserve passing tests in incremental mode
        skip_test_analysis: Skip running existing tests, just analyze file content
        use_enhanced_agent: Use EnhancedAgent instead of basic ReActAgent
        enable_error_prediction: Enable error prediction in EnhancedAgent
        enable_self_reflection: Enable self-reflection in EnhancedAgent
        enable_pattern_library: Enable pattern library in EnhancedAgent
        enable_multi_agent: Whether to use multi-agent collaboration
        multi_agent_workers: Number of multi-agent workers
    """
    parallel_workers: int = 1
    timeout_per_file: int = 300
    continue_on_error: bool = True
    coverage_target: int = 80
    max_iterations: int = 10
    defer_compilation: bool = False
    compile_only_at_end: bool = False
    incremental_mode: bool = False
    preserve_passing_tests: bool = True
    skip_test_analysis: bool = False
    use_enhanced_agent: bool = True
    enable_error_prediction: bool = True
    enable_self_reflection: bool = True
    enable_pattern_library: bool = True
    enable_multi_agent: bool = True
    multi_agent_workers: int = 3


@dataclass
class FileResult:
    """Result of test generation for a single file.
    
    Attributes:
        file_path: Path to the source file
        success: Whether generation was successful
        coverage: Achieved coverage percentage
        coverage_source: Source of coverage data ("jacoco" or "llm_estimated")
        coverage_confidence: Confidence level for LLM estimation
        iterations: Number of iterations used
        test_file: Path to generated test file
        error: Error message if failed
        duration: Time taken in seconds
        incremental_mode: Whether incremental mode was used
        preserved_tests: Number of tests preserved in incremental mode
        new_tests: Number of new tests added
        fixed_tests: Number of tests fixed
    """
    file_path: str
    success: bool
    coverage: float = 0.0
    coverage_source: str = "jacoco"
    coverage_confidence: float = 1.0
    iterations: int = 0
    test_file: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0
    incremental_mode: bool = False
    preserved_tests: int = 0
    new_tests: int = 0
    fixed_tests: int = 0


@dataclass
class BatchProgress:
    """Progress information for batch generation.
    
    Attributes:
        total_files: Total number of files to process
        completed_files: Number of successfully completed files
        failed_files: Number of failed files
        current_file: Currently processing file
        current_status: Current status message
    """
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    current_file: str = ""
    current_status: str = ""
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_files == 0:
            return 0.0
        return ((self.completed_files + self.failed_files) / self.total_files) * 100


@dataclass
class BatchCompilationResult:
    """Result of batch compilation."""
    success: bool
    compiled_files: int = 0
    failed_files: int = 0
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0


@dataclass
class BatchResult:
    """Result of batch test generation.
    
    Attributes:
        total_files: Total number of files processed
        success_count: Number of successful generations
        failed_count: Number of failed generations
        skipped_count: Number of skipped files
        results: List of individual file results
        total_duration: Total time taken in seconds
        compilation_result: Optional compilation result if defer_compilation is True
        incremental_mode: Whether incremental mode was enabled for batch
        total_preserved_tests: Total tests preserved across all files
        total_new_tests: Total new tests added across all files
        total_fixed_tests: Total tests fixed across all files
    """
    total_files: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    results: List[FileResult] = field(default_factory=list)
    total_duration: float = 0.0
    compilation_result: Optional[BatchCompilationResult] = None
    incremental_mode: bool = False
    total_preserved_tests: int = 0
    total_new_tests: int = 0
    total_fixed_tests: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.success_count / self.total_files) * 100


class BatchGenerator:
    """Generator for batch test generation with parallel execution support.
    
    This class handles generating tests for multiple Java files with:
    - Configurable parallel execution
    - Error isolation (one file failure doesn't stop others)
    - Progress tracking and callbacks
    - Timeout handling per file
    """
    
    def __init__(
        self,
        llm_client,
        project_path: str,
        config: Optional[BatchConfig] = None,
        progress_callback: Optional[Callable[[BatchProgress], None]] = None
    ):
        """Initialize batch generator.
        
        Args:
            llm_client: LLM client for test generation
            project_path: Path to the Maven project
            config: Batch configuration
            progress_callback: Callback for progress updates
        """
        self.llm_client = llm_client
        self.project_path = project_path
        self.config = config or BatchConfig()
        self.progress_callback = progress_callback
        
        self._progress = BatchProgress()
        self._stop_requested = False
        self._results: List[FileResult] = []
        
        logger.info(
            f"[BatchGenerator] Initialized - Project: {project_path}, "
            f"ParallelWorkers: {self.config.parallel_workers}, "
            f"Timeout: {self.config.timeout_per_file}s, "
            f"DeferCompilation: {self.config.defer_compilation}, "
            f"IncrementalMode: {self.config.incremental_mode}, "
            f"MultiAgent: {self.config.enable_multi_agent}"
        )
        
        # Initialize multi-agent components if enabled
        self._multi_agent_initialized = False
        self.agent_coordinator = None
        self.code_analysis_agent = None
        self.test_generation_agent = None
        self.test_fix_agent = None
        
        if self.config.enable_multi_agent:
            self._init_multi_agent_components()
        
        # Initialize build tool manager for compilation
        from ..tools.build_tool_manager import BuildToolManager
        self.build_tool_manager = BuildToolManager(project_path)
        self.build_runner = self.build_tool_manager.get_runner()
    
    def _init_multi_agent_components(self):
        """Initialize multi-agent collaboration components."""
        try:
            from ..agent.multi_agent import (
                AgentCoordinator, MessageBus, SharedKnowledgeBase, ExperienceReplay,
                CodeAnalysisAgent, TestGenerationAgent, TestFixAgent, AgentRole
            )
            
            # Create shared components
            message_bus = MessageBus()
            shared_knowledge = SharedKnowledgeBase()
            experience_replay = ExperienceReplay()
            
            # Create coordinator
            self.agent_coordinator = AgentCoordinator(
                message_bus=message_bus,
                knowledge_base=shared_knowledge,
                experience_replay=experience_replay
            )
            
            # Create specialized agents
            self.code_analysis_agent = CodeAnalysisAgent(
                agent_id="batch_code_analyzer",
                message_bus=message_bus,
                knowledge_base=shared_knowledge,
                experience_replay=experience_replay
            )
            
            self.test_generation_agent = TestGenerationAgent(
                agent_id="batch_test_generator",
                message_bus=message_bus,
                knowledge_base=shared_knowledge,
                experience_replay=experience_replay,
                llm_client=self.llm_client
            )
            
            self.test_fix_agent = TestFixAgent(
                agent_id="batch_test_fixer",
                message_bus=message_bus,
                knowledge_base=shared_knowledge,
                experience_replay=experience_replay,
                llm_client=self.llm_client
            )
            
            # Register agents with coordinator
            self.agent_coordinator.register_agent(
                agent_id=self.code_analysis_agent.agent_id,
                capabilities=self.code_analysis_agent.capabilities,
                role=AgentRole.ANALYZER
            )
            
            self.agent_coordinator.register_agent(
                agent_id=self.test_generation_agent.agent_id,
                capabilities=self.test_generation_agent.capabilities,
                role=AgentRole.IMPLEMENTER
            )
            
            self.agent_coordinator.register_agent(
                agent_id=self.test_fix_agent.agent_id,
                capabilities=self.test_fix_agent.capabilities,
                role=AgentRole.FIXER
            )
            
            self._multi_agent_initialized = True
            logger.info(f"[BatchGenerator] Multi-agent system initialized with {len(self.agent_coordinator.agents)} agents")
            
        except Exception as e:
            logger.warning(f"[BatchGenerator] Failed to initialize multi-agent components: {e}")
            self._multi_agent_initialized = False
    
    def stop(self):
        """Stop the batch generation."""
        logger.info("[BatchGenerator] Stop requested")
        self._stop_requested = True
    
    def _update_progress(self, current_file: str = "", status: str = ""):
        """Update progress and notify callback."""
        self._progress.current_file = current_file
        self._progress.current_status = status
        self._progress.completed_files = sum(1 for r in self._results if r.success)
        self._progress.failed_files = sum(1 for r in self._results if not r.success)
        
        if self.progress_callback:
            try:
                self.progress_callback(self._progress)
            except Exception as e:
                logger.warning(f"[BatchGenerator] Progress callback error: {e}")
    
    async def generate_all(self, files: List[str]) -> BatchResult:
        """Generate tests for all files with parallel execution.
        
        Args:
            files: List of file paths to generate tests for
            
        Returns:
            BatchResult with all results
        """
        start_time = time.time()
        self._stop_requested = False
        self._results = []
        
        self._progress = BatchProgress(total_files=len(files))
        
        logger.info(
            f"[BatchGenerator] Starting batch generation - "
            f"Files: {len(files)}, Parallel: {self.config.parallel_workers}"
        )
        
        semaphore = asyncio.Semaphore(
            self.config.parallel_workers if self.config.parallel_workers > 0 
            else len(files)
        )
        
        async def generate_with_semaphore(file_path: str) -> FileResult:
            async with semaphore:
                if self._stop_requested:
                    return FileResult(
                        file_path=file_path,
                        success=False,
                        error="Generation stopped by user"
                    )
                return await self._generate_single(file_path)
        
        tasks = [generate_with_semaphore(f) for f in files]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"[BatchGenerator] Task exception for {files[i]}: {result}")
                    self._results.append(FileResult(
                        file_path=files[i],
                        success=False,
                        error=str(result)
                    ))
                elif isinstance(result, FileResult):
                    self._results.append(result)
                else:
                    self._results.append(FileResult(
                        file_path=files[i],
                        success=False,
                        error=f"Unexpected result type: {type(result)}"
                    ))
        except Exception as e:
            logger.exception(f"[BatchGenerator] Batch generation error: {e}")
        
        total_duration = time.time() - start_time
        
        success_count = sum(1 for r in self._results if r.success)
        failed_count = sum(1 for r in self._results if not r.success)
        
        logger.info(
            f"[BatchGenerator] Batch generation complete - "
            f"Success: {success_count}, Failed: {failed_count}, "
            f"Duration: {total_duration:.1f}s"
        )
        
        # Phase 2: Batch compilation if defer_compilation is enabled
        compilation_result = None
        if self.config.defer_compilation and success_count > 0:
            logger.info("[BatchGenerator] Starting Phase 2: Batch compilation of all generated tests")
            compilation_result = await self._compile_all_tests()
            
            # Update results based on compilation
            if not compilation_result.success:
                logger.warning(
                    f"[BatchGenerator] Batch compilation failed - "
                    f"Compiled: {compilation_result.compiled_files}, "
                    f"Failed: {compilation_result.failed_files}"
                )
        
        logger.info(
            f"[BatchGenerator] Batch complete - "
            f"Success: {success_count}, Failed: {failed_count}, "
            f"Duration: {total_duration:.1f}s"
        )
        
        # Calculate incremental mode statistics
        total_preserved = sum(r.preserved_tests for r in self._results if r.success)
        total_new = sum(r.new_tests for r in self._results if r.success)
        total_fixed = sum(r.fixed_tests for r in self._results if r.success)
        
        return BatchResult(
            total_files=len(files),
            success_count=success_count,
            failed_count=failed_count,
            skipped_count=0,
            results=self._results,
            total_duration=total_duration,
            compilation_result=compilation_result,
            incremental_mode=self.config.incremental_mode,
            total_preserved_tests=total_preserved,
            total_new_tests=total_new,
            total_fixed_tests=total_fixed
        )
    
    def generate_all_sync(self, files: List[str]) -> BatchResult:
        """Synchronous wrapper for generate_all.
        
        Args:
            files: List of file paths
            
        Returns:
            BatchResult with all results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.generate_all(files)
                )
                return future.result()
        else:
            return loop.run_until_complete(self.generate_all(files))
    
    async def _generate_single(self, file_path: str) -> FileResult:
        """Generate tests for a single file.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            FileResult with generation result
        """
        # Use multi-agent approach if enabled and initialized
        if self.config.enable_multi_agent and self._multi_agent_initialized:
            return await self._generate_single_multi_agent(file_path)
        else:
            return await self._generate_single_standard(file_path)
    
    async def _generate_single_multi_agent(self, file_path: str) -> FileResult:
        """Generate tests using multi-agent collaboration.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            FileResult with generation result
        """
        start_time = time.time()
        file_name = Path(file_path).name
        
        logger.info(f"[BatchGenerator] Starting multi-agent generation for: {file_name}")
        self._update_progress(file_name, "Multi-agent: Analyzing code...")
        
        try:
            # Step 1: Code Analysis
            analysis_task_id = await self.agent_coordinator.submit_task(
                task_type="analyze_code",
                payload={"file_path": file_path},
                priority=1
            )
            
            # Wait for analysis
            analysis_success = await self.agent_coordinator.wait_for_task(
                analysis_task_id, 
                timeout=30
            )
            
            if not analysis_success:
                logger.warning(f"[BatchGenerator] Code analysis failed for {file_name}, falling back to standard")
                return await self._generate_single_standard(file_path)
            
            # Get analysis result
            analysis_task = self.agent_coordinator.tasks.get(analysis_task_id)
            analysis_result = analysis_task.result.output if analysis_task and analysis_task.result else {}
            
            self._update_progress(file_name, "Multi-agent: Generating tests...")
            
            # Step 2: Test Generation
            methods = analysis_result.get("methods", [])
            generation_task_id = await self.agent_coordinator.submit_task(
                task_type="generate_tests",
                payload={
                    "file_path": file_path,
                    "class_info": analysis_result,
                    "methods": methods,
                    "options": {
                        "framework": "JUnit5",
                        "mock_framework": "Mockito",
                        "include_edge_cases": True
                    }
                },
                priority=1,
                dependencies=[analysis_task_id]
            )
            
            # Wait for generation
            generation_success = await self.agent_coordinator.wait_for_task(
                generation_task_id,
                timeout=self.config.timeout_per_file - 30
            )
            
            duration = time.time() - start_time
            
            if generation_success:
                generation_task = self.agent_coordinator.tasks.get(generation_task_id)
                generation_result = generation_task.result.output if generation_task and generation_task.result else {}
                
                # Write generated test to file
                test_code = generation_result.get("test_code", "")
                test_class_name = generation_result.get("test_class_name", "")
                
                if test_code and test_class_name:
                    test_file_path = await self._write_test_file(file_path, test_class_name, test_code)
                    
                    logger.info(
                        f"[BatchGenerator] Multi-agent success for {file_name} - "
                        f"Duration: {duration:.1f}s"
                    )
                    
                    return FileResult(
                        file_path=file_path,
                        success=True,
                        coverage=0.0,  # Will be determined later
                        iterations=1,
                        test_file=test_file_path,
                        duration=duration
                    )
            
            # If multi-agent failed, fall back to standard
            logger.warning(f"[BatchGenerator] Multi-agent generation failed for {file_name}, falling back")
            return await self._generate_single_standard(file_path)
            
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"[BatchGenerator] Multi-agent exception for {file_name}: {e}")
            # Fall back to standard generation
            return await self._generate_single_standard(file_path)
    
    async def _write_test_file(self, source_file_path: str, test_class_name: str, test_code: str) -> str:
        """Write test code to file.
        
        Args:
            source_file_path: Path to source file
            test_class_name: Name of test class
            test_code: Test code to write
            
        Returns:
            Path to test file
        """
        source_path = Path(source_file_path)
        
        # Determine test directory
        test_dir = source_path.parent.parent / "test" / "java" / source_path.parent.name
        if not test_dir.exists():
            # Try standard Maven structure
            project_path = Path(self.project_path)
            relative_path = source_path.relative_to(project_path / "src" / "main" / "java")
            test_dir = project_path / "src" / "test" / "java" / relative_path.parent
        
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_file_path = test_dir / f"{test_class_name}.java"
        test_file_path.write_text(test_code, encoding='utf-8')
        
        return str(test_file_path)
    
    async def _generate_single_standard(self, file_path: str) -> FileResult:
        """Generate tests using standard ReActAgent.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            FileResult with generation result
        """
        start_time = time.time()
        file_name = Path(file_path).name
        
        logger.info(f"[BatchGenerator] Starting standard generation for: {file_name}")
        self._update_progress(file_name, "Starting...")
        
        try:
            from ..agent.react_agent import ReActAgent
            from ..agent.enhanced_agent import EnhancedAgent, EnhancedAgentConfig
            from ..memory.working_memory import WorkingMemory
            
            working_memory = WorkingMemory(
                target_coverage=self.config.coverage_target / 100.0,
                max_iterations=self.config.max_iterations,
                current_file=file_path
            )
            
            if self.config.use_enhanced_agent:
                agent_config = EnhancedAgentConfig(
                    model_name=self.llm_client.model if hasattr(self.llm_client, 'model') else 'gpt-4',
                    enable_error_prediction=self.config.enable_error_prediction,
                    enable_strategy_optimization=True,
                    enable_self_reflection=self.config.enable_self_reflection,
                    enable_knowledge_graph=False,
                    enable_pattern_library=self.config.enable_pattern_library,
                    enable_chain_of_thought=True,
                    enable_metrics=True,
                )
                agent = EnhancedAgent(
                    llm_client=self.llm_client,
                    working_memory=working_memory,
                    project_path=self.project_path,
                    progress_callback=lambda p: self._on_agent_progress(file_name, p),
                    config=agent_config,
                )
                logger.info(f"[BatchGenerator] Using EnhancedAgent for {file_name}")
            else:
                agent = ReActAgent(
                    llm_client=self.llm_client,
                    working_memory=working_memory,
                    project_path=self.project_path,
                    progress_callback=lambda p: self._on_agent_progress(file_name, p),
                    incremental_mode=self.config.incremental_mode,
                    preserve_passing_tests=self.config.preserve_passing_tests,
                    skip_test_analysis=self.config.skip_test_analysis,
                )
                logger.info(f"[BatchGenerator] Using ReActAgent for {file_name}")
            
            if self.config.incremental_mode:
                self._update_progress(file_name, "Incremental mode: analyzing existing tests...")
                logger.info(f"[BatchGenerator] Incremental mode enabled for {file_name}")
            else:
                self._update_progress(file_name, "Generating tests...")
            
            if self.config.defer_compilation:
                logger.info(f"[BatchGenerator] Defer compilation mode - generating code only for {file_name}")
                working_memory.skip_verification = True
            
            try:
                result = await asyncio.wait_for(
                    agent.generate_tests(file_path),
                    timeout=self.config.timeout_per_file
                )
            except asyncio.TimeoutError:
                logger.warning(f"[BatchGenerator] Timeout for {file_name}")
                return FileResult(
                    file_path=file_path,
                    success=False,
                    error=f"Timeout after {self.config.timeout_per_file}s",
                    duration=time.time() - start_time
                )
            
            duration = time.time() - start_time
            
            if result.success:
                logger.info(
                    f"[BatchGenerator] Success for {file_name} - "
                    f"Coverage: {result.coverage:.1%}, Duration: {duration:.1f}s"
                )
                
                preserved_tests = 0
                new_tests = 0
                fixed_tests = 0
                if hasattr(result, 'metadata') and result.metadata:
                    preserved_tests = result.metadata.get("preserved_tests", 0)
                    new_tests = result.metadata.get("new_tests", 0)
                    fixed_tests = result.metadata.get("fixed_tests", 0)
                
                coverage_source = getattr(result, 'coverage_source', 'jacoco')
                coverage_confidence = getattr(result, 'coverage_confidence', 1.0)
                
                return FileResult(
                    file_path=file_path,
                    success=True,
                    coverage=result.coverage,
                    coverage_source=coverage_source,
                    coverage_confidence=coverage_confidence,
                    iterations=result.iterations,
                    test_file=result.test_file,
                    duration=duration,
                    incremental_mode=self.config.incremental_mode,
                    preserved_tests=preserved_tests,
                    new_tests=new_tests,
                    fixed_tests=fixed_tests
                )
            else:
                logger.warning(
                    f"[BatchGenerator] Failed for {file_name} - "
                    f"Error: {result.message}"
                )
                return FileResult(
                    file_path=file_path,
                    success=False,
                    error=result.message,
                    iterations=result.iterations,
                    duration=duration
                )
                
        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"[BatchGenerator] Exception for {file_name}: {e}")
            
            if not self.config.continue_on_error:
                self._stop_requested = True
            
            return FileResult(
                file_path=file_path,
                success=False,
                error=str(e),
                duration=duration
            )
    
    async def _compile_all_tests(self) -> BatchCompilationResult:
        """Compile all generated test files in batch.
        
        Returns:
            BatchCompilationResult with compilation results
        """
        start_time = time.time()
        logger.info(f"[BatchGenerator] Compiling all generated tests...")
        
        try:
            # Use Maven to compile all test classes at once
            if self.build_runner:
                compile_success = await self.build_runner.compile_tests()
                
                elapsed = time.time() - start_time
                
                if compile_success:
                    logger.info(f"[BatchGenerator] ✅ Batch compilation successful - Duration: {elapsed:.1f}s")
                    return BatchCompilationResult(
                        success=True,
                        compiled_files=self._progress.completed_files,
                        failed_files=0,
                        errors=[],
                        duration=elapsed
                    )
                else:
                    logger.error(f"[BatchGenerator] ❌ Batch compilation failed - Duration: {elapsed:.1f}s")
                    return BatchCompilationResult(
                        success=False,
                        compiled_files=0,
                        failed_files=self._progress.completed_files,
                        errors=["Batch compilation failed"],
                        duration=elapsed
                    )
            else:
                logger.error("[BatchGenerator] No build runner available")
                return BatchCompilationResult(
                    success=False,
                    compiled_files=0,
                    failed_files=0,
                    errors=["No build runner available"],
                    duration=time.time() - start_time
                )
                
        except Exception as e:
            logger.exception(f"[BatchGenerator] Batch compilation exception: {e}")
            return BatchCompilationResult(
                success=False,
                compiled_files=0,
                failed_files=0,
                errors=[str(e)],
                duration=time.time() - start_time
            )
    
    def _on_agent_progress(self, file_name: str, progress_info: Dict[str, Any]):
        """Handle agent progress updates.
        
        Args:
            file_name: Name of the file being processed
            progress_info: Progress information from agent
        """
        state = progress_info.get("state", "")
        message = progress_info.get("message", "")
        
        # Skip compilation/verification steps if defer_compilation is enabled
        if self.config.defer_compilation and state in ["COMPILING", "TESTING", "ANALYZING"]:
            return
        
        status_map = {
            "PARSING": "Parsing...",
            "GENERATING": "Generating tests...",
            "COMPILING": "Compiling...",
            "TESTING": "Running tests...",
            "ANALYZING": "Analyzing coverage...",
            "FIXING": "Fixing issues...",
            "OPTIMIZING": "Optimizing...",
        }
        
        status = status_map.get(state, message)
        self._update_progress(file_name, f"{status} {message}")
