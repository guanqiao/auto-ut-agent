"""Handler for test execution operations."""

import logging
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any

from ..base_agent import AgentState, StepResult
from ...tools.maven_tools import MavenRunner
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class TestExecutionHandler:
    """Handles test execution and failure analysis."""
    
    def __init__(
        self,
        project_path: str,
        maven_runner: Optional[MavenRunner] = None,
        progress_callback: Optional[Callable[[AgentState, str], None]] = None
    ):
        """Initialize test execution handler.
        
        Args:
            project_path: Path to the project
            maven_runner: Optional MavenRunner instance
            progress_callback: Optional callback for progress updates
        """
        self.project_path = Path(project_path)
        self.maven_runner = maven_runner or MavenRunner(project_path)
        self.progress_callback = progress_callback
        self._stop_requested = False
    
    def stop(self):
        """Stop test execution."""
        self._stop_requested = True
    
    def reset(self):
        """Reset handler state."""
        self._stop_requested = False
    
    def _update_state(self, state: AgentState, message: str):
        """Update state via callback."""
        if self.progress_callback:
            self.progress_callback(state, message)
    
    async def run_tests(
        self,
        test_file: Optional[str] = None,
        attempt: int = 1
    ) -> StepResult:
        """Run the generated tests.
        
        Args:
            test_file: Optional specific test file to run
            attempt: Current attempt number
            
        Returns:
            StepResult with test execution outcome
        """
        if self._stop_requested:
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="Test execution stopped by user"
            )
        
        self._update_state(AgentState.TESTING, f"Attempt {attempt}: Running tests...")
        logger.info(f"[TestExecutionHandler] Starting test execution - Attempt: {attempt}")
        
        try:
            success = self.maven_runner.run_tests(test_file)
            
            if success:
                logger.info("[TestExecutionHandler] All tests passed")
                self._update_state(AgentState.TESTING, "All tests passed")
                return StepResult(
                    success=True,
                    state=AgentState.TESTING,
                    message="All tests passed"
                )
            else:
                failures = self._parse_test_failures()
                logger.warning(f"[TestExecutionHandler] Tests failed - Failures: {len(failures)}")
                
                self._update_state(
                    AgentState.FIXING,
                    f"{len(failures)} test(s) failed. Analyzing..."
                )
                
                return StepResult(
                    success=False,
                    state=AgentState.FIXING,
                    message=f"{len(failures)} tests failed",
                    data={"failures": failures}
                )
        except Exception as e:
            logger.exception(f"[TestExecutionHandler] Test execution exception: {e}")
            self._update_state(AgentState.FIXING, f"Test execution error: {str(e)}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error running tests: {str(e)}"
            )
    
    def _parse_test_failures(self) -> List[Dict[str, Any]]:
        """Parse test failures from Maven surefire reports.
        
        Returns:
            List of failure dictionaries
        """
        failures = []
        settings = get_settings()
        surefire_dir = self.project_path / settings.project_paths.target_surefire_reports
        
        if not surefire_dir.exists():
            logger.warning(f"[TestExecutionHandler] Surefire reports directory not found: {surefire_dir}")
            return failures
        
        try:
            for report_file in surefire_dir.glob("*.txt"):
                try:
                    content = report_file.read_text(encoding='utf-8')
                    if "FAILURE" in content or "ERROR" in content:
                        failure_info = self._extract_failure_info(content, report_file.stem)
                        failures.append(failure_info)
                except Exception as e:
                    logger.warning(f"[TestExecutionHandler] Failed to read report {report_file}: {e}")
        except Exception as e:
            logger.exception(f"[TestExecutionHandler] Error parsing test failures: {e}")
        
        logger.debug(f"[TestExecutionHandler] Parsed {len(failures)} test failures")
        return failures
    
    def _extract_failure_info(
        self,
        content: str,
        test_name: str
    ) -> Dict[str, Any]:
        """Extract detailed failure information.
        
        Args:
            content: Report file content
            test_name: Name of the test
            
        Returns:
            Failure information dictionary
        """
        max_error_length = 1000
        error_text = content[:max_error_length]
        
        if len(content) > max_error_length:
            error_text += "\n... (truncated)"
        
        return {
            "test_name": test_name,
            "error": error_text,
            "has_failure": "FAILURE" in content,
            "has_error": "ERROR" in content
        }
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of test execution.
        
        Returns:
            Summary dictionary with test counts
        """
        settings = get_settings()
        surefire_dir = self.project_path / settings.project_paths.target_surefire_reports
        
        summary = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0
        }
        
        if not surefire_dir.exists():
            return summary
        
        try:
            for report_file in surefire_dir.glob("*.txt"):
                summary["total"] += 1
                try:
                    content = report_file.read_text(encoding='utf-8')
                    if "FAILURE" in content:
                        summary["failed"] += 1
                    elif "ERROR" in content:
                        summary["errors"] += 1
                    else:
                        summary["passed"] += 1
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[TestExecutionHandler] Error getting test summary: {e}")
        
        return summary
