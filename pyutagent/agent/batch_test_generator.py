"""Batch test generator for processing multiple files concurrently."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json

from .test_generator import TestGeneratorAgent
from ..core.project_config import ProjectContext, TestFramework, MockFramework
from ..tools.maven_tools import CoverageAnalyzer, MavenRunner
from ..cache.test_cache import TestCache

logger = logging.getLogger(__name__)


@dataclass
class BatchOptions:
    """Batch generation options."""
    
    concurrency: int = 3  # Number of concurrent files to process
    target_coverage: float = 0.8
    max_iterations: int = 3
    skip_existing: bool = True  # Skip if test file already exists
    generate_for_tests: bool = False  # Generate for test classes
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    timeout_per_file: int = 300  # Timeout in seconds per file


@dataclass
class FileResult:
    """Result for a single file."""
    
    file_path: str
    success: bool
    test_file_path: Optional[str]
    error_message: Optional[str]
    generation_time: float
    coverage_contribution: float
    skipped: bool = False
    skipped_reason: Optional[str] = None


@dataclass
class BatchResult:
    """Result of batch generation."""
    
    total_files: int
    successful: int
    failed: int
    skipped: int
    total_time: float
    average_time: float
    coverage_before: Optional[float]
    coverage_after: Optional[float]
    results: List[FileResult]
    start_time: datetime
    end_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_files': self.total_files,
            'successful': self.successful,
            'failed': self.failed,
            'skipped': self.skipped,
            'total_time': self.total_time,
            'average_time': self.average_time,
            'coverage_before': self.coverage_before,
            'coverage_after': self.coverage_after,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'results': [
                {
                    'file_path': r.file_path,
                    'success': r.success,
                    'test_file_path': r.test_file_path,
                    'error_message': r.error_message,
                    'generation_time': r.generation_time,
                    'coverage_contribution': r.coverage_contribution,
                    'skipped': r.skipped,
                    'skipped_reason': r.skipped_reason
                }
                for r in self.results
            ]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class BatchTestGenerator:
    """Batch test generator for processing multiple files.
    
    Features:
    - Concurrent processing with configurable concurrency
    - Progress tracking and callbacks
    - Result caching
    - Coverage-driven generation
    - Timeout handling
    """
    
    def __init__(
        self,
        project_path: str,
        options: Optional[BatchOptions] = None
    ):
        """Initialize batch test generator.
        
        Args:
            project_path: Path to the project
            options: Batch generation options
        """
        self.project_path = Path(project_path)
        self.options = options or BatchOptions()
        
        # Project configuration
        self.project_config = ProjectContext(str(project_path))
        
        # Tools
        self.maven_runner = MavenRunner(str(project_path))
        self.coverage_analyzer = CoverageAnalyzer(str(project_path))
        
        # Test cache
        cache_dir = Path(project_path) / ".pyutagent" / "cache"
        self.test_cache = TestCache(str(cache_dir))
        
        # Progress tracking
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None
        self._result_callback: Optional[Callable[[FileResult], None]] = None
        
        # Result cache
        self._result_cache: Dict[str, FileResult] = {}
        
        # Cancellation
        self._cancelled = False
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set progress callback.
        
        Args:
            callback: Function to call with (processed, total, message)
        """
        self._progress_callback = callback
    
    def set_result_callback(self, callback: Callable[[FileResult], None]):
        """Set result callback.
        
        Args:
            callback: Function to call with each file result
        """
        self._result_callback = callback
    
    def cancel(self):
        """Cancel batch generation."""
        self._cancelled = True
        logger.info("Batch generation cancelled")
    
    def is_cancelled(self) -> bool:
        """Check if cancelled."""
        return self._cancelled
    
    def scan_target_files(self) -> List[str]:
        """Scan for target Java files.
        
        Returns:
            List of Java file paths relative to src/main/java
        """
        source_dir = self.project_path / "src" / "main" / "java"
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return []
        
        java_files = []
        for java_file in source_dir.rglob("*.java"):
            rel_path = str(java_file.relative_to(source_dir))
            
            # Apply include/exclude patterns
            if self._should_include(rel_path):
                java_files.append(rel_path)
        
        logger.info(f"Found {len(java_files)} Java files to process")
        return java_files
    
    def _should_include(self, file_path: str) -> bool:
        """Check if file should be included.
        
        Args:
            file_path: Relative file path
            
        Returns:
            True if should include
        """
        # Check exclude patterns
        for pattern in self.options.exclude_patterns:
            if pattern in file_path:
                return False
        
        # Check include patterns
        if self.options.include_patterns:
            for pattern in self.options.include_patterns:
                if pattern in file_path:
                    return True
            return False
        
        return True
    
    async def generate_batch(
        self,
        files: Optional[List[str]] = None
    ) -> BatchResult:
        """Generate tests for multiple files.
        
        Args:
            files: Optional list of files to process. If None, scans all files.
            
        Returns:
            BatchResult with statistics and individual results
        """
        start_time = datetime.now()
        logger.info(f"Starting batch generation at {start_time}")
        
        # Get files to process
        if files is None:
            files = self.scan_target_files()
        
        if not files:
            logger.warning("No files to process")
            return BatchResult(
                total_files=0,
                successful=0,
                failed=0,
                skipped=0,
                total_time=0,
                average_time=0,
                coverage_before=None,
                coverage_after=None,
                results=[],
                start_time=start_time,
                end_time=datetime.now()
            )
        
        # Get coverage before
        coverage_before = self._get_current_coverage()
        logger.info(f"Current coverage: {coverage_before}")
        
        # Process files in batches
        all_results: List[FileResult] = []
        successful = 0
        failed = 0
        skipped = 0
        
        # Chunk files for concurrent processing
        batches = self._chunk_array(files, self.options.concurrency)
        logger.info(f"Processing {len(files)} files in {len(batches)} batches")
        
        for i, batch in enumerate(batches):
            if self.is_cancelled():
                logger.info("Batch generation cancelled by user")
                break
            
            self._update_progress(
                processed=len(all_results),
                total=len(files),
                message=f"Processing batch {i+1}/{len(batches)}"
            )
            
            # Process batch concurrently
            batch_results = await self._process_batch(batch)
            
            # Update counters
            for result in batch_results:
                if result.skipped:
                    skipped += 1
                elif result.success:
                    successful += 1
                else:
                    failed += 1
                
                all_results.append(result)
                
                # Call result callback
                if self._result_callback:
                    self._result_callback(result)
        
        # Get coverage after
        coverage_after = self._get_current_coverage()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        result = BatchResult(
            total_files=len(files),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_time=total_time,
            average_time=total_time / len(files) if files else 0,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            results=all_results,
            start_time=start_time,
            end_time=end_time
        )
        
        logger.info(
            f"Batch generation completed: "
            f"{successful} successful, {failed} failed, {skipped} skipped"
        )
        
        return result
    
    async def _process_batch(self, files: List[str]) -> List[FileResult]:
        """Process a batch of files concurrently.
        
        Args:
            files: List of files to process
            
        Returns:
            List of file results
        """
        # Create tasks for concurrent processing
        tasks = [
            asyncio.create_task(self._generate_single_file(file))
            for file in files
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        file_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Handle exception
                file_result = FileResult(
                    file_path=files[i],
                    success=False,
                    test_file_path=None,
                    error_message=str(result),
                    generation_time=0,
                    coverage_contribution=0
                )
                logger.error(f"Failed to process {files[i]}: {result}")
            else:
                file_result = result
            
            file_results.append(file_result)
        
        return file_results
    
    async def _generate_single_file(self, file_path: str) -> FileResult:
        """Generate test for a single file.
        
        Args:
            file_path: Relative file path
            
        Returns:
            File result
        """
        start_time = datetime.now()
        
        try:
            # Check result cache
            if file_path in self._result_cache:
                cached = self._result_cache[file_path]
                if cached.skipped:
                    return cached
            
            # Check test code cache
            source_path = self.project_path / "src" / "main" / "java" / file_path
            test_framework = self.project_config.test_preferences.test_framework.value
            mock_framework = self.project_config.test_preferences.mock_framework.value
            target_coverage = self.options.target_coverage
            
            cached_test = self.test_cache.get(
                file_path,
                source_path,
                test_framework,
                mock_framework,
                target_coverage
            )
            
            if cached_test:
                logger.info(f"Using cached test for {file_path}")
                # Save cached test to file
                test_path = self._get_test_file_path(file_path)
                if test_path:
                    test_path.parent.mkdir(parents=True, exist_ok=True)
                    test_path.write_text(cached_test, encoding='utf-8')
                    
                    result = FileResult(
                        file_path=file_path,
                        success=True,
                        test_file_path=str(test_path),
                        error_message=None,
                        generation_time=0,
                        coverage_contribution=0,
                        skipped=True,
                        skipped_reason="Used cached test"
                    )
                    self._result_cache[file_path] = result
                    return result
            
            # Check if test already exists
            if self.options.skip_existing:
                test_path = self._get_test_file_path(file_path)
                if test_path and test_path.exists():
                    result = FileResult(
                        file_path=file_path,
                        success=False,
                        test_file_path=str(test_path),
                        error_message=None,
                        generation_time=0,
                        coverage_contribution=0,
                        skipped=True,
                        skipped_reason="Test file already exists"
                    )
                    self._result_cache[file_path] = result
                    return result
            
            # Create test generator agent
            from ..memory.working_memory import WorkingMemory
            from ..agent.conversation import ConversationManager
            from ..core.config import LLMConfig
            
            # Use default LLM config
            llm_config = LLMConfig()
            conversation = ConversationManager()
            working_memory = WorkingMemory()
            
            agent = TestGeneratorAgent(
                project_path=str(self.project_path),
                llm_config=llm_config,
                conversation=conversation,
                working_memory=working_memory
            )
            
            # Set progress callback
            def progress_cb(value: int, status: str):
                elapsed = (datetime.now() - start_time).total_seconds()
                self._update_progress(
                    processed=0,
                    total=1,
                    message=f"{file_path}: {status} ({elapsed:.1f}s)"
                )
            
            agent.on_progress = progress_cb
            agent.on_log = lambda msg: logger.debug(f"[{file_path}] {msg}")
            
            # Generate tests with timeout
            result_dict = await asyncio.wait_for(
                agent.generate_tests(
                    target_file=file_path,
                    target_coverage=self.options.target_coverage,
                    max_iterations=self.options.max_iterations
                ),
                timeout=self.options.timeout_per_file
            )
            
            # Calculate generation time
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            # Calculate coverage contribution
            coverage_contribution = self._calculate_coverage_contribution(file_path)
            
            # Create result
            if result_dict.get('success', False):
                result = FileResult(
                    file_path=file_path,
                    success=True,
                    test_file_path=result_dict.get('test_file'),
                    error_message=None,
                    generation_time=generation_time,
                    coverage_contribution=coverage_contribution
                )
                
                # Cache the generated test
                test_file_path = result_dict.get('test_file')
                if test_file_path:
                    try:
                        test_code = Path(test_file_path).read_text(encoding='utf-8')
                        source_path = self.project_path / "src" / "main" / "java" / file_path
                        test_framework = self.project_config.test_preferences.test_framework.value
                        mock_framework = self.project_config.test_preferences.mock_framework.value
                        
                        self.test_cache.set(
                            file_path,
                            source_path,
                            test_framework,
                            mock_framework,
                            self.options.target_coverage,
                            test_code
                        )
                        logger.debug(f"Cached generated test for {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to cache test for {file_path}: {e}")
            else:
                result = FileResult(
                    file_path=file_path,
                    success=False,
                    test_file_path=None,
                    error_message=result_dict.get('error', 'Unknown error'),
                    generation_time=generation_time,
                    coverage_contribution=0
                )
            
            self._result_cache[file_path] = result
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing {file_path}")
            return FileResult(
                file_path=file_path,
                success=False,
                test_file_path=None,
                error_message=f"Timeout after {self.options.timeout_per_file}s",
                generation_time=self.options.timeout_per_file,
                coverage_contribution=0
            )
        except Exception as e:
            logger.exception(f"Failed to generate test for {file_path}: {e}")
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            return FileResult(
                file_path=file_path,
                success=False,
                test_file_path=None,
                error_message=str(e),
                generation_time=generation_time,
                coverage_contribution=0
            )
    
    def _get_current_coverage(self) -> Optional[float]:
        """Get current coverage."""
        report = self.coverage_analyzer.parse_report()
        return report.line_coverage if report else None
    
    def _get_test_file_path(self, source_file: str) -> Optional[Path]:
        """Get test file path for source file.
        
        Args:
            source_file: Relative source file path
            
        Returns:
            Test file path or None
        """
        test_dir = self.project_path / "src" / "test" / "java"
        return test_dir / source_file.replace(".java", "Test.java")
    
    def _calculate_coverage_contribution(self, source_file: str) -> float:
        """Calculate coverage contribution for a file.
        
        Args:
            source_file: Relative source file path
            
        Returns:
            Coverage ratio
        """
        file_cov = self.coverage_analyzer.get_file_coverage(source_file)
        return file_cov.line_coverage if file_cov else 0.0
    
    def _chunk_array(self, array: List[Any], size: int) -> List[List[Any]]:
        """Chunk array into batches.
        
        Args:
            array: Array to chunk
            size: Batch size
            
        Returns:
            List of batches
        """
        return [array[i:i + size] for i in range(0, len(array), size)]
    
    def _update_progress(self, processed: int, total: int, message: str):
        """Update progress.
        
        Args:
            processed: Number of processed files
            total: Total number of files
            message: Progress message
        """
        if self._progress_callback:
            self._progress_callback(processed, total, message)
        
        # Also log progress
        logger.info(f"Progress: {processed}/{total} - {message}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        total = len(self._result_cache)
        successful = sum(1 for r in self._result_cache.values() if r.success)
        skipped = sum(1 for r in self._result_cache.values() if r.skipped)
        
        return {
            'total_cached': total,
            'successful': successful,
            'skipped': skipped,
            'failed': total - successful - skipped
        }
