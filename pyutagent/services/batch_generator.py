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
    """
    parallel_workers: int = 1
    timeout_per_file: int = 300
    continue_on_error: bool = True
    coverage_target: int = 80
    max_iterations: int = 10


@dataclass
class FileResult:
    """Result of test generation for a single file.
    
    Attributes:
        file_path: Path to the source file
        success: Whether generation was successful
        coverage: Achieved coverage percentage
        iterations: Number of iterations used
        test_file: Path to generated test file
        error: Error message if failed
        duration: Time taken in seconds
    """
    file_path: str
    success: bool
    coverage: float = 0.0
    iterations: int = 0
    test_file: Optional[str] = None
    error: Optional[str] = None
    duration: float = 0.0


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
class BatchResult:
    """Result of batch test generation.
    
    Attributes:
        total_files: Total number of files processed
        success_count: Number of successful generations
        failed_count: Number of failed generations
        skipped_count: Number of skipped files
        results: List of individual file results
        total_duration: Total time taken in seconds
    """
    total_files: int = 0
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    results: List[FileResult] = field(default_factory=list)
    total_duration: float = 0.0
    
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
            f"Timeout: {self.config.timeout_per_file}s"
        )
    
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
            f"[BatchGenerator] Batch complete - "
            f"Success: {success_count}, Failed: {failed_count}, "
            f"Duration: {total_duration:.1f}s"
        )
        
        return BatchResult(
            total_files=len(files),
            success_count=success_count,
            failed_count=failed_count,
            skipped_count=0,
            results=self._results,
            total_duration=total_duration
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
        start_time = time.time()
        file_name = Path(file_path).name
        
        logger.info(f"[BatchGenerator] Starting generation for: {file_name}")
        self._update_progress(file_name, "Starting...")
        
        try:
            from ..agent.react_agent import ReActAgent
            from ..memory.working_memory import WorkingMemory
            
            working_memory = WorkingMemory(
                target_coverage=self.config.coverage_target / 100.0,
                max_iterations=self.config.max_iterations,
                current_file=file_path
            )
            
            agent = ReActAgent(
                llm_client=self.llm_client,
                working_memory=working_memory,
                project_path=self.project_path,
                progress_callback=lambda p: self._on_agent_progress(file_name, p)
            )
            
            self._update_progress(file_name, "Generating tests...")
            
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
                return FileResult(
                    file_path=file_path,
                    success=True,
                    coverage=result.coverage,
                    iterations=result.iterations,
                    test_file=result.test_file,
                    duration=duration
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
    
    def _on_agent_progress(self, file_name: str, progress_info: Dict[str, Any]):
        """Handle agent progress updates.
        
        Args:
            file_name: Name of the file being processed
            progress_info: Progress information from agent
        """
        state = progress_info.get("state", "")
        message = progress_info.get("message", "")
        
        status_map = {
            "PARSING": "Parsing...",
            "GENERATING": "Generating tests...",
            "COMPILING": "Compiling...",
            "TESTING": "Running tests...",
            "ANALYZING": "Analyzing coverage...",
            "FIXING": "Fixing issues...",
            "OPTIMIZING": "Optimizing...",
        }
        
        status = status_map.get(state, state)
        self._update_progress(file_name, f"{status} {message}")
