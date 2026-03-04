"""Incremental Test Manager - Manages incremental test generation mode.

This module handles:
- Detecting existing test files
- Analyzing existing test status (pass/fail)
- Building incremental generation context
- Merging preserved tests with new tests
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pyutagent.agent.partial_success_handler import (
    PartialSuccessHandler,
    TestCodeParser,
    TestMethodInfo,
    TestMethodResult,
    TestStatus,
    PartialTestResult,
)
from pyutagent.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class IncrementalConfig:
    """Configuration for incremental test generation.
    
    Attributes:
        enabled: Whether incremental mode is enabled
        preserve_passing_tests: Whether to preserve passing tests
        analyze_existing_coverage: Whether to analyze coverage before generating
        max_preserved_tests: Maximum number of tests to preserve
        min_tests_to_preserve: Minimum tests to preserve to enable incremental
        force_regenerate_failed: Whether to force regenerate failed tests
        skip_analysis: Skip running tests, just analyze file content
    """
    enabled: bool = False
    preserve_passing_tests: bool = True
    analyze_existing_coverage: bool = True
    max_preserved_tests: int = 50
    min_tests_to_preserve: int = 1
    force_regenerate_failed: bool = True
    skip_analysis: bool = False


@dataclass
class ExistingTestAnalysis:
    """Analysis result of existing test file.
    
    Attributes:
        test_file_path: Path to the test file
        exists: Whether the test file exists
        test_methods: List of test method information
        passing_tests: List of passing test method names
        failing_tests: List of failing test method names
        error_tests: List of tests with errors
        skipped_tests: List of skipped tests
        current_coverage: Current line coverage percentage
        uncovered_lines: List of uncovered line numbers
        uncovered_branches: List of uncovered branch info
        last_run_time: When tests were last run
        test_code: Full test code content
        has_compilation_errors: Whether there are compilation errors
        compilation_errors: List of compilation errors
    """
    test_file_path: str
    exists: bool = False
    test_methods: List[TestMethodInfo] = field(default_factory=list)
    passing_tests: List[str] = field(default_factory=list)
    failing_tests: List[str] = field(default_factory=list)
    error_tests: List[str] = field(default_factory=list)
    skipped_tests: List[str] = field(default_factory=list)
    current_coverage: float = 0.0
    uncovered_lines: List[int] = field(default_factory=list)
    uncovered_branches: List[int] = field(default_factory=list)
    last_run_time: Optional[datetime] = None
    test_code: str = ""
    has_compilation_errors: bool = False
    compilation_errors: List[str] = field(default_factory=list)
    
    @property
    def total_tests(self) -> int:
        """Get total number of tests."""
        return len(self.test_methods)
    
    @property
    def has_tests(self) -> bool:
        """Check if there are any tests."""
        return len(self.test_methods) > 0
    
    @property
    def has_passing_tests(self) -> bool:
        """Check if there are passing tests."""
        return len(self.passing_tests) > 0
    
    @property
    def has_failing_tests(self) -> bool:
        """Check if there are failing tests."""
        return len(self.failing_tests) > 0 or len(self.error_tests) > 0
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return len(self.passing_tests) / self.total_tests
    
    @property
    def should_use_incremental(self) -> bool:
        """Determine if incremental mode should be used."""
        return self.has_passing_tests and not self.has_compilation_errors


@dataclass
class IncrementalContext:
    """Context for incremental test generation.
    
    Attributes:
        existing_tests_code: Full existing test code
        preserved_test_names: Names of tests to preserve
        preserved_test_code: Code of tests to preserve
        tests_to_fix: List of failing tests that need fixing
        new_code_to_cover: New/modified code that needs tests
        uncovered_code: Code that is not covered
        target_coverage_gap: Gap between current and target coverage
        source_code: Target class source code
        class_info: Target class information
    """
    existing_tests_code: str = ""
    preserved_test_names: List[str] = field(default_factory=list)
    preserved_test_code: str = ""
    tests_to_fix: List[TestMethodResult] = field(default_factory=list)
    new_code_to_cover: List[Dict[str, Any]] = field(default_factory=list)
    uncovered_code: List[Dict[str, Any]] = field(default_factory=list)
    target_coverage_gap: float = 0.0
    source_code: str = ""
    class_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_preserved_tests(self) -> bool:
        """Check if there are tests to preserve."""
        return len(self.preserved_test_names) > 0
    
    @property
    def has_tests_to_fix(self) -> bool:
        """Check if there are tests to fix."""
        return len(self.tests_to_fix) > 0
    
    @property
    def needs_additional_coverage(self) -> bool:
        """Check if additional coverage is needed."""
        return self.target_coverage_gap > 0


class IncrementalTestManager:
    """Manages incremental test generation workflow.
    
    This class orchestrates the incremental generation process:
    1. Detect if existing test file exists
    2. Analyze existing tests (run and collect results)
    3. Build context for incremental generation
    4. Merge preserved and new tests
    """
    
    def __init__(
        self,
        project_path: str,
        config: Optional[IncrementalConfig] = None,
        maven_runner=None,
        coverage_analyzer=None,
    ):
        """Initialize incremental test manager.
        
        Args:
            project_path: Path to the project
            config: Incremental configuration
            maven_runner: Maven runner for test execution
            coverage_analyzer: Coverage analyzer
        """
        self.project_path = Path(project_path)
        self.config = config or IncrementalConfig()
        self.maven_runner = maven_runner
        self.coverage_analyzer = coverage_analyzer
        self.parser = TestCodeParser()
        self.partial_handler = PartialSuccessHandler()
        
        logger.info(
            f"[IncrementalTestManager] Initialized - "
            f"Project: {project_path}, Enabled: {self.config.enabled}"
        )
    
    def detect_existing_test(
        self,
        target_file: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Detect if a test file already exists for the target.
        
        Args:
            target_file: Path to the target Java file
            class_info: Optional class information (for package info)
            
        Returns:
            Path to existing test file, or None if not found
        """
        settings = get_settings()
        src_test_java = settings.project_paths.src_test_java
        
        target_path = Path(target_file)
        class_name = target_path.stem
        
        if class_info:
            package = class_info.get("package", "")
            if package:
                package_path = package.replace(".", "/")
                test_dir = self.project_path / src_test_java / package_path
            else:
                test_dir = self.project_path / src_test_java
        else:
            relative_dir = target_path.parent
            if "src/main/java" in str(target_file):
                relative_dir = Path(str(target_file).split("src/main/java")[1]).parent
            test_dir = self.project_path / src_test_java / relative_dir
        
        test_file_name = f"{class_name}Test.java"
        test_file_path = test_dir / test_file_name
        
        if test_file_path.exists():
            logger.info(f"[IncrementalTestManager] Found existing test file: {test_file_path}")
            return str(test_file_path.relative_to(self.project_path))
        
        logger.debug(f"[IncrementalTestManager] No existing test file found for {target_file}")
        return None
    
    async def analyze_existing_tests(
        self,
        test_file_path: str,
        run_tests: bool = True
    ) -> ExistingTestAnalysis:
        """Analyze existing test file.
        
        Args:
            test_file_path: Path to the test file
            run_tests: Whether to run tests to get status
            
        Returns:
            ExistingTestAnalysis with analysis results
        """
        full_path = self.project_path / test_file_path
        
        if not full_path.exists():
            logger.info(f"[IncrementalTestManager] Test file does not exist: {test_file_path}")
            return ExistingTestAnalysis(test_file_path=test_file_path, exists=False)
        
        logger.info(f"[IncrementalTestManager] Analyzing existing tests: {test_file_path}")
        
        test_code = full_path.read_text(encoding="utf-8")
        test_methods = self.parser.parse_test_methods(test_code)
        
        analysis = ExistingTestAnalysis(
            test_file_path=test_file_path,
            exists=True,
            test_methods=test_methods,
            test_code=test_code,
        )
        
        if not run_tests or self.config.skip_analysis:
            logger.info("[IncrementalTestManager] Skipping test execution analysis")
            analysis.passing_tests = [m.method_name for m in test_methods]
            return analysis
        
        if self.maven_runner:
            try:
                test_result = await self._run_existing_tests(test_file_path)
                
                if test_result:
                    analysis.passing_tests = [t.method_name for t in test_result.get_passed_tests()]
                    analysis.failing_tests = [
                        t.method_name for t in test_result.test_results
                        if t.status == TestStatus.FAILED
                    ]
                    analysis.error_tests = [
                        t.method_name for t in test_result.test_results
                        if t.status == TestStatus.ERROR
                    ]
                    analysis.skipped_tests = [
                        t.method_name for t in test_result.test_results
                        if t.status == TestStatus.SKIPPED
                    ]
                    analysis.last_run_time = datetime.now()
                    
                    logger.info(
                        f"[IncrementalTestManager] Test execution results - "
                        f"Passed: {len(analysis.passing_tests)}, "
                        f"Failed: {len(analysis.failing_tests)}, "
                        f"Errors: {len(analysis.error_tests)}"
                    )
            except Exception as e:
                logger.warning(f"[IncrementalTestManager] Failed to run tests: {e}")
                analysis.passing_tests = [m.method_name for m in test_methods]
        
        if self.config.analyze_existing_coverage and self.coverage_analyzer:
            try:
                coverage_info = self._analyze_coverage()
                if coverage_info:
                    analysis.current_coverage = coverage_info.get("line_coverage", 0.0)
                    analysis.uncovered_lines = coverage_info.get("uncovered_lines", [])
                    analysis.uncovered_branches = coverage_info.get("uncovered_branches", [])
                    
                    logger.info(
                        f"[IncrementalTestManager] Coverage analysis - "
                        f"Line: {analysis.current_coverage:.1%}, "
                        f"Uncovered lines: {len(analysis.uncovered_lines)}"
                    )
            except Exception as e:
                logger.warning(f"[IncrementalTestManager] Failed to analyze coverage: {e}")
        
        return analysis
    
    async def _run_existing_tests(self, test_file_path: str) -> Optional[PartialTestResult]:
        """Run existing tests and collect results.
        
        Args:
            test_file_path: Path to the test file
            
        Returns:
            PartialTestResult if successful
        """
        if not self.maven_runner:
            return None
        
        logger.info(f"[IncrementalTestManager] Running existing tests: {test_file_path}")
        
        test_class_name = Path(test_file_path).stem
        
        try:
            success = self.maven_runner.run_tests(test_class=test_class_name)
            
            settings = get_settings()
            surefire_dir = self.project_path / settings.project_paths.target_surefire_reports
            
            test_output = ""
            if hasattr(self.maven_runner, 'last_output'):
                test_output = self.maven_runner.last_output
            
            result = self.partial_handler.analyze_test_results(
                test_output=test_output,
                surefire_reports_dir=surefire_dir if surefire_dir.exists() else None
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[IncrementalTestManager] Error running tests: {e}")
            return None
    
    def _analyze_coverage(self) -> Optional[Dict[str, Any]]:
        """Analyze current test coverage.
        
        Returns:
            Coverage information dictionary
        """
        if not self.coverage_analyzer:
            return None
        
        try:
            report = self.coverage_analyzer.parse_report()
            
            if report:
                return {
                    "line_coverage": report.line_coverage,
                    "branch_coverage": report.branch_coverage,
                    "method_coverage": report.method_coverage,
                    "uncovered_lines": [],
                    "uncovered_branches": [],
                }
        except Exception as e:
            logger.warning(f"[IncrementalTestManager] Coverage analysis error: {e}")
        
        return None
    
    def build_incremental_context(
        self,
        analysis: ExistingTestAnalysis,
        class_info: Dict[str, Any],
        source_code: str,
        target_coverage: float = 0.8
    ) -> IncrementalContext:
        """Build context for incremental test generation.
        
        Args:
            analysis: Existing test analysis
            class_info: Target class information
            source_code: Target class source code
            target_coverage: Target coverage percentage
            
        Returns:
            IncrementalContext for generation
        """
        logger.info("[IncrementalTestManager] Building incremental context")
        
        context = IncrementalContext(
            existing_tests_code=analysis.test_code,
            source_code=source_code,
            class_info=class_info,
        )
        
        if self.config.preserve_passing_tests and analysis.has_passing_tests:
            preserved_count = min(len(analysis.passing_tests), self.config.max_preserved_tests)
            context.preserved_test_names = analysis.passing_tests[:preserved_count]
            
            context.preserved_test_code = self._extract_preserved_tests(
                analysis.test_code,
                context.preserved_test_names
            )
            
            logger.info(
                f"[IncrementalTestManager] Preserving {len(context.preserved_test_names)} passing tests"
            )
        
        if analysis.has_failing_tests and self.config.force_regenerate_failed:
            for test_name in analysis.failing_tests + analysis.error_tests:
                for method in analysis.test_methods:
                    if method.method_name == test_name:
                        context.tests_to_fix.append(TestMethodResult(
                            method_name=test_name,
                            status=TestStatus.FAILED,
                        ))
                        break
            
            logger.info(
                f"[IncrementalTestManager] Identified {len(context.tests_to_fix)} tests to fix"
            )
        
        context.target_coverage_gap = max(0, target_coverage - analysis.current_coverage)
        
        if analysis.uncovered_lines:
            context.uncovered_code = [
                {"line": line, "type": "line"}
                for line in analysis.uncovered_lines[:50]
            ]
        
        return context
    
    def _extract_preserved_tests(
        self,
        test_code: str,
        test_names: List[str]
    ) -> str:
        """Extract code for tests to preserve.
        
        Args:
            test_code: Full test code
            test_names: Names of tests to preserve
            
        Returns:
            Code containing only preserved tests
        """
        if not test_names:
            return ""
        
        all_methods = self.parser.parse_test_methods(test_code)
        preserved_methods = [m for m in all_methods if m.method_name in test_names]
        
        if not preserved_methods:
            return ""
        
        skeleton = self.parser.extract_class_skeleton(test_code)
        
        method_codes = [m.content for m in preserved_methods]
        
        last_brace = skeleton.rfind('}')
        if last_brace > 0:
            combined = skeleton[:last_brace] + "\n\n" + "\n\n".join(method_codes) + "\n}"
        else:
            combined = skeleton + "\n\n" + "\n\n".join(method_codes)
        
        return combined
    
    def merge_tests(
        self,
        preserved_test_code: str,
        new_test_code: str,
        class_name: str
    ) -> str:
        """Merge preserved tests with new tests.
        
        Args:
            preserved_test_code: Code of preserved tests
            new_test_code: Code of new tests
            class_name: Name of the test class
            
        Returns:
            Merged test code
        """
        logger.info("[IncrementalTestManager] Merging preserved and new tests")
        
        new_methods = self.parser.parse_test_methods(new_test_code)
        
        if not new_methods:
            logger.warning("[IncrementalTestManager] No new test methods found")
            return preserved_test_code
        
        new_method_names = {m.method_name for m in new_methods}
        
        preserved_methods = self.parser.parse_test_methods(preserved_test_code)
        
        filtered_preserved = [
            m for m in preserved_methods
            if m.method_name not in new_method_names
        ]
        
        skeleton = self.parser.extract_class_skeleton(preserved_test_code)
        
        all_methods = filtered_preserved + new_methods
        method_codes = [m.content for m in all_methods]
        
        last_brace = skeleton.rfind('}')
        if last_brace > 0:
            merged = skeleton[:last_brace] + "\n\n" + "\n\n".join(method_codes) + "\n}"
        else:
            merged = skeleton + "\n\n" + "\n\n".join(method_codes)
        
        logger.info(
            f"[IncrementalTestManager] Merged {len(filtered_preserved)} preserved + "
            f"{len(new_methods)} new = {len(all_methods)} total tests"
        )
        
        return merged
    
    def should_use_incremental_mode(
        self,
        analysis: ExistingTestAnalysis
    ) -> bool:
        """Determine if incremental mode should be used.
        
        Args:
            analysis: Existing test analysis
            
        Returns:
            True if incremental mode should be used
        """
        if not self.config.enabled:
            logger.info("[IncrementalTestManager] Incremental mode disabled by config")
            return False
        
        if not analysis.exists:
            logger.info("[IncrementalTestManager] No existing test file")
            return False
        
        if analysis.has_compilation_errors:
            logger.info("[IncrementalTestManager] Existing tests have compilation errors")
            return False
        
        if len(analysis.passing_tests) < self.config.min_tests_to_preserve:
            logger.info(
                f"[IncrementalTestManager] Not enough passing tests "
                f"({len(analysis.passing_tests)} < {self.config.min_tests_to_preserve})"
            )
            return False
        
        logger.info("[IncrementalTestManager] Incremental mode recommended")
        return True


def create_incremental_manager(
    project_path: str,
    incremental_mode: bool = False,
    maven_runner=None,
    coverage_analyzer=None,
    **kwargs
) -> IncrementalTestManager:
    """Factory function to create incremental test manager.
    
    Args:
        project_path: Path to the project
        incremental_mode: Whether incremental mode is enabled
        maven_runner: Maven runner instance
        coverage_analyzer: Coverage analyzer instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured IncrementalTestManager
    """
    config = IncrementalConfig(
        enabled=incremental_mode,
        preserve_passing_tests=kwargs.get("preserve_passing_tests", True),
        analyze_existing_coverage=kwargs.get("analyze_existing_coverage", True),
        max_preserved_tests=kwargs.get("max_preserved_tests", 50),
        min_tests_to_preserve=kwargs.get("min_tests_to_preserve", 1),
        force_regenerate_failed=kwargs.get("force_regenerate_failed", True),
        skip_analysis=kwargs.get("skip_analysis", False),
    )
    
    return IncrementalTestManager(
        project_path=project_path,
        config=config,
        maven_runner=maven_runner,
        coverage_analyzer=coverage_analyzer,
    )
