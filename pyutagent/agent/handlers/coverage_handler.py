"""Handler for coverage analysis operations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set

from ...core.protocols import AgentState
from ..base_agent import StepResult
from ...tools.maven_tools import MavenRunner, CoverageAnalyzer as MavenCoverageAnalyzer

logger = logging.getLogger(__name__)


class CoverageHandler:
    """Handles test coverage analysis and reporting."""
    
    def __init__(
        self,
        project_path: str,
        maven_runner: Optional[MavenRunner] = None,
        coverage_analyzer: Optional[MavenCoverageAnalyzer] = None,
        progress_callback: Optional[Callable[[AgentState, str], None]] = None
    ):
        """Initialize coverage handler.
        
        Args:
            project_path: Path to the project
            maven_runner: Optional MavenRunner instance
            coverage_analyzer: Optional CoverageAnalyzer instance
            progress_callback: Optional callback for progress updates
        """
        self.project_path = Path(project_path)
        self.maven_runner = maven_runner or MavenRunner(project_path)
        self.coverage_analyzer = coverage_analyzer or MavenCoverageAnalyzer(project_path)
        self.progress_callback = progress_callback
        self._stop_requested = False
    
    def stop(self):
        """Stop coverage analysis."""
        self._stop_requested = True
    
    def reset(self):
        """Reset handler state."""
        self._stop_requested = False
    
    def _update_state(self, state: AgentState, message: str):
        """Update state via callback."""
        if self.progress_callback:
            self.progress_callback(state, message)
    
    async def analyze_coverage(self) -> StepResult:
        """Analyze test coverage.
        
        Returns:
            StepResult with coverage analysis outcome
        """
        if self._stop_requested:
            return StepResult(
                success=False,
                state=AgentState.PAUSED,
                message="Coverage analysis stopped by user"
            )
        
        logger.info("[CoverageHandler] Starting coverage analysis")
        
        try:
            logger.debug("[CoverageHandler] Generating coverage report")
            self.maven_runner.generate_coverage()
            
            report = self.coverage_analyzer.parse_report()
            
            if report:
                logger.info(
                    f"[CoverageHandler] Coverage analysis complete - "
                    f"Line: {report.line_coverage:.1%}, "
                    f"Branch: {report.branch_coverage:.1%}, "
                    f"Method: {report.method_coverage:.1%}"
                )
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
                logger.warning("[CoverageHandler] Failed to parse coverage report")
                return StepResult(
                    success=False,
                    state=AgentState.FAILED,
                    message="Failed to parse coverage report"
                )
        except Exception as e:
            logger.exception(f"[CoverageHandler] Coverage analysis exception: {e}")
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message=f"Error analyzing coverage: {str(e)}"
            )
    
    def get_uncovered_info(self, report: Any) -> Dict[str, Any]:
        """Get information about uncovered code.
        
        Args:
            report: Coverage report object
            
        Returns:
            Dictionary with uncovered methods, lines, and branches
        """
        uncovered_info = {
            "methods": [],
            "lines": [],
            "branches": [],
            "files": []
        }
        
        if not report or not report.files:
            logger.debug("[CoverageHandler] No coverage report or files available")
            return uncovered_info
        
        try:
            for file_coverage in report.files:
                file_info = {
                    "name": getattr(file_coverage, 'name', 'unknown'),
                    "uncovered_lines": []
                }
                
                if hasattr(file_coverage, 'lines') and file_coverage.lines:
                    for line_info in file_coverage.lines:
                        if isinstance(line_info, tuple) and len(line_info) >= 2:
                            line_num, is_covered = line_info[0], line_info[1]
                            if not is_covered:
                                uncovered_info["lines"].append(line_num)
                                file_info["uncovered_lines"].append(line_num)
                        elif isinstance(line_info, dict):
                            line_num = line_info.get('number')
                            is_covered = line_info.get('covered', False)
                            if line_num and not is_covered:
                                uncovered_info["lines"].append(line_num)
                                file_info["uncovered_lines"].append(line_num)
                
                if file_info["uncovered_lines"]:
                    uncovered_info["files"].append(file_info)
            
            logger.debug(
                f"[CoverageHandler] Uncovered info - "
                f"Lines: {len(uncovered_info['lines'])}, "
                f"Files: {len(uncovered_info['files'])}"
            )
        except Exception as e:
            logger.warning(f"[CoverageHandler] Error extracting uncovered info: {e}")
        
        return uncovered_info
    
    def get_coverage_summary(self, report: Any) -> Dict[str, Any]:
        """Get human-readable coverage summary.
        
        Args:
            report: Coverage report object
            
        Returns:
            Summary dictionary
        """
        if not report:
            return {
                "line_coverage": 0.0,
                "branch_coverage": 0.0,
                "method_coverage": 0.0,
                "status": "no_report"
            }
        
        summary = {
            "line_coverage": getattr(report, 'line_coverage', 0.0),
            "branch_coverage": getattr(report, 'branch_coverage', 0.0),
            "method_coverage": getattr(report, 'method_coverage', 0.0),
            "status": "ok"
        }
        
        if summary["line_coverage"] >= 0.8:
            summary["status"] = "excellent"
        elif summary["line_coverage"] >= 0.6:
            summary["status"] = "good"
        elif summary["line_coverage"] >= 0.4:
            summary["status"] = "fair"
        else:
            summary["status"] = "poor"
        
        return summary
    
    def identify_uncovered_methods(
        self,
        class_info: Dict[str, Any],
        uncovered_lines: List[int]
    ) -> List[str]:
        """Identify which methods are not fully covered.
        
        Args:
            class_info: Class information dictionary
            uncovered_lines: List of uncovered line numbers
            
        Returns:
            List of uncovered method names
        """
        uncovered_methods = []
        uncovered_line_set = set(uncovered_lines)
        
        methods = class_info.get("methods", [])
        for method in methods:
            method_name = method.get("name", "unknown")
            line_range = method.get("line_range")
            
            if line_range:
                start_line, end_line = line_range
                method_lines = set(range(start_line, end_line + 1))
                
                if method_lines & uncovered_line_set:
                    uncovered_methods.append(method_name)
        
        logger.debug(
            f"[CoverageHandler] Identified {len(uncovered_methods)} "
            f"uncovered methods out of {len(methods)}"
        )
        return uncovered_methods
