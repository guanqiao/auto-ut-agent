"""Coverage analysis service for UT generation."""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, Callable

from ...core.protocols import AgentState
from ..base_agent import StepResult
from ...tools.maven_tools import MavenRunner, CoverageAnalyzer

logger = logging.getLogger(__name__)


class CoverageAnalysisService:
    """Service for analyzing test coverage.
    
    Responsibilities:
    - Generate coverage reports via Maven/JaCoCo
    - Parse coverage reports
    - Provide coverage metrics
    """
    
    def __init__(
        self,
        project_path: str,
        maven_runner: Optional[MavenRunner] = None,
        coverage_analyzer: Optional[CoverageAnalyzer] = None,
        progress_callback: Optional[Callable[[AgentState, str], None]] = None
    ):
        """Initialize coverage analysis service.
        
        Args:
            project_path: Path to the project
            maven_runner: Optional MavenRunner instance
            coverage_analyzer: Optional CoverageAnalyzer instance
            progress_callback: Optional callback for progress updates
        """
        self.project_path = Path(project_path)
        self._maven_runner = maven_runner or MavenRunner(project_path)
        self._coverage_analyzer = coverage_analyzer or CoverageAnalyzer(project_path)
        self.progress_callback = progress_callback
        self._stop_requested = False
    
    def stop(self):
        """Stop analysis."""
        self._stop_requested = True
    
    def reset(self):
        """Reset service state."""
        self._stop_requested = False
    
    def _update_state(self, state: AgentState, message: str):
        """Update state via callback."""
        if self.progress_callback:
            self.progress_callback(state, message)
    
    async def analyze_coverage(self) -> StepResult:
        """Analyze test coverage.
        
        Returns:
            StepResult with coverage data
        """
        if self._stop_requested:
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="Coverage analysis stopped by user"
            )
        
        logger.info("[CoverageAnalysisService] Analyzing coverage")
        self._update_state(AgentState.ANALYZING, "Analyzing coverage...")
        
        try:
            logger.debug("[CoverageAnalysisService] Generating coverage report")
            self._maven_runner.generate_coverage()
            
            report = self._coverage_analyzer.parse_report()
            
            if report:
                logger.info(f"[CoverageAnalysisService] Coverage analysis complete - Line: {report.line_coverage:.1%}, Branch: {report.branch_coverage:.1%}, Method: {report.method_coverage:.1%}")
                self._update_state(AgentState.ANALYZING, f"Coverage: {report.line_coverage:.1%}")
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
                logger.warning("[CoverageAnalysisService] Failed to parse coverage report")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message="Failed to parse coverage report"
                )
        except Exception as e:
            logger.exception(f"[CoverageAnalysisService] Coverage analysis exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error analyzing coverage: {str(e)}"
            )
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get coverage summary.
        
        Returns:
            Dictionary with coverage summary
        """
        try:
            report = self._coverage_analyzer.parse_report()
            if report:
                return {
                    "line_coverage": report.line_coverage,
                    "branch_coverage": report.branch_coverage,
                    "method_coverage": report.method_coverage,
                    "class_coverage": report.class_coverage,
                    "covered_lines": report.covered_lines,
                    "total_lines": report.total_lines
                }
        except Exception as e:
            logger.warning(f"[CoverageAnalysisService] Failed to get coverage summary: {e}")
        
        return {
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
            "method_coverage": 0.0,
            "class_coverage": 0.0,
            "covered_lines": 0,
            "total_lines": 0
        }
    
    def get_uncovered_lines(self, source_file: Optional[str] = None) -> list:
        """Get list of uncovered lines.
        
        Args:
            source_file: Optional specific source file to check
            
        Returns:
            List of uncovered line numbers
        """
        try:
            report = self._coverage_analyzer.parse_report()
            if report and hasattr(report, 'files'):
                uncovered = []
                for file_coverage in report.files:
                    if source_file and source_file not in str(file_coverage.path):
                        continue
                    for line_num, is_covered in file_coverage.lines:
                        if not is_covered:
                            uncovered.append(line_num)
                return uncovered
        except Exception as e:
            logger.warning(f"[CoverageAnalysisService] Failed to get uncovered lines: {e}")
        
        return []
