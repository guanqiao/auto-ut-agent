"""Test execution service for UT generation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from ...core.protocols import AgentState
from ..base_agent import StepResult
from ...core.config import get_settings
from ...tools.maven_tools import MavenRunner

logger = logging.getLogger(__name__)


class TestExecutionService:
    """Service for executing tests and analyzing failures.
    
    Responsibilities:
    - Run tests via Maven
    - Parse test failures
    - Report test results
    """
    
    def __init__(
        self,
        project_path: str,
        maven_runner: Optional[MavenRunner] = None,
        progress_callback: Optional[Callable[[AgentState, str], None]] = None
    ):
        """Initialize test execution service.
        
        Args:
            project_path: Path to the project
            maven_runner: Optional MavenRunner instance
            progress_callback: Optional callback for progress updates
        """
        self.project_path = Path(project_path)
        self._maven_runner = maven_runner or MavenRunner(project_path)
        self.progress_callback = progress_callback
        self._stop_requested = False
    
    def stop(self):
        """Stop execution."""
        self._stop_requested = True
    
    def reset(self):
        """Reset service state."""
        self._stop_requested = False
    
    def _update_state(self, state: AgentState, message: str):
        """Update state via callback."""
        if self.progress_callback:
            self.progress_callback(state, message)
    
    async def run_tests(self) -> StepResult:
        """Run the generated tests.
        
        Returns:
            StepResult with test execution results
        """
        if self._stop_requested:
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="Test execution stopped by user"
            )
        
        logger.info("[TestExecutionService] Running tests")
        self._update_state(AgentState.TESTING, "Running tests...")
        
        try:
            success = self._maven_runner.run_tests()
            
            if success:
                logger.info("[TestExecutionService] All tests passed")
                self._update_state(AgentState.TESTING, "All tests passed")
                return StepResult(
                    success=True,
                    state=AgentState.TESTING,
                    message="All tests passed"
                )
            else:
                failures = self._parse_test_failures()
                logger.warning(f"[TestExecutionService] Tests failed - Failures: {len(failures)}")
                self._update_state(AgentState.FIXING, f"{len(failures)} test(s) failed")
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message="Some tests failed",
                    data={"failures": failures}
                )
        except Exception as e:
            logger.exception(f"[TestExecutionService] Test execution exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error running tests: {str(e)}"
            )
    
    def _parse_test_failures(self) -> List[Dict[str, Any]]:
        """Parse test failures from Maven output.
        
        Returns:
            List of failure information
        """
        failures = []
        settings = get_settings()
        surefire_dir = self.project_path / settings.project_paths.target_surefire_reports
        
        if surefire_dir.exists():
            for report_file in surefire_dir.glob("*.txt"):
                content = report_file.read_text()
                if "FAILURE" in content or "ERROR" in content:
                    failures.append({
                        "test_name": report_file.stem,
                        "error": content[:500]
                    })
        
        logger.debug(f"[TestExecutionService] Parsed test failures - Failures: {len(failures)}")
        return failures
    
    def get_test_results_summary(self) -> Dict[str, Any]:
        """Get summary of test results.
        
        Returns:
            Dictionary with test results summary
        """
        settings = get_settings()
        surefire_dir = self.project_path / settings.project_paths.target_surefire_reports
        
        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0
        }
        
        if surefire_dir.exists():
            for report_file in surefire_dir.glob("*.txt"):
                content = report_file.read_text()
                summary["total_tests"] += content.count("Tests run:")
                if "FAILURE" in content:
                    summary["failed"] += 1
                if "ERROR" in content:
                    summary["errors"] += 1
        
        return summary
