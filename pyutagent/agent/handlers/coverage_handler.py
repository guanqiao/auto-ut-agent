"""Handler for coverage analysis operations."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set

from ...core.protocols import AgentState
from ..base_agent import StepResult
from ...tools.maven_tools import MavenRunner, CoverageAnalyzer as MavenCoverageAnalyzer
from ..llm_coverage_evaluator import LLMCoverageEvaluator, LLMCoverageReport, CoverageSource

logger = logging.getLogger(__name__)


class CoverageHandler:
    """Handles test coverage analysis and reporting.
    
    Supports both JaCoCo (precise) and LLM-based (estimated) coverage analysis.
    Falls back to LLM estimation when JaCoCo is not available.
    """
    
    def __init__(
        self,
        project_path: str,
        maven_runner: Optional[MavenRunner] = None,
        coverage_analyzer: Optional[MavenCoverageAnalyzer] = None,
        progress_callback: Optional[Callable[[AgentState, str], None]] = None,
        llm_client: Optional[Any] = None,
        source_code: Optional[str] = None,
        test_code: Optional[str] = None,
        class_info: Optional[Dict[str, Any]] = None
    ):
        """Initialize coverage handler.
        
        Args:
            project_path: Path to the project
            maven_runner: Optional MavenRunner instance
            coverage_analyzer: Optional CoverageAnalyzer instance
            progress_callback: Optional callback for progress updates
            llm_client: Optional LLM client for fallback coverage estimation
            source_code: Optional source code for LLM estimation
            test_code: Optional test code for LLM estimation
            class_info: Optional class info for LLM estimation
        """
        self.project_path = Path(project_path)
        self.maven_runner = maven_runner or MavenRunner(project_path)
        self.coverage_analyzer = coverage_analyzer or MavenCoverageAnalyzer(project_path)
        self.progress_callback = progress_callback
        self._stop_requested = False
        
        self.llm_client = llm_client
        self.source_code = source_code
        self.test_code = test_code
        self.class_info = class_info
        
        self._llm_evaluator: Optional[LLMCoverageEvaluator] = None
        if llm_client:
            self._llm_evaluator = LLMCoverageEvaluator(llm_client)
    
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
        
        First attempts to use JaCoCo for precise coverage.
        Falls back to LLM estimation if JaCoCo is not available.
        
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
            logger.debug("[CoverageHandler] Attempting JaCoCo coverage report")
            self.maven_runner.generate_coverage()
            
            report = self.coverage_analyzer.parse_report()
            
            if report:
                logger.info(
                    f"[CoverageHandler] JaCoCo coverage analysis complete - "
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
                        "report": report,
                        "source": CoverageSource.JACOCO.value
                    }
                )
            else:
                logger.warning("[CoverageHandler] JaCoCo report not available, trying LLM estimation")
                return await self._fallback_to_llm_estimation()
                
        except Exception as e:
            logger.warning(f"[CoverageHandler] JaCoCo analysis failed: {e}, trying LLM estimation")
            return await self._fallback_to_llm_estimation()
    
    async def _fallback_to_llm_estimation(self) -> StepResult:
        """Fall back to LLM-based coverage estimation.
        
        Returns:
            StepResult with estimated coverage
        """
        if not self._llm_evaluator:
            if self.llm_client:
                self._llm_evaluator = LLMCoverageEvaluator(self.llm_client)
        
        if not self._llm_evaluator or not self.source_code or not self.test_code:
            logger.warning("[CoverageHandler] LLM estimation not available, using quick heuristic")
            return self._quick_estimate_fallback()
        
        try:
            logger.info("[CoverageHandler] Using LLM for coverage estimation")
            self._update_state(AgentState.ANALYZING, "📊 使用 LLM 估算覆盖率...")
            
            llm_report = await self._llm_evaluator.evaluate_coverage(
                self.source_code,
                self.test_code,
                self.class_info
            )
            
            logger.info(
                f"[CoverageHandler] LLM coverage estimation complete - "
                f"Line: {llm_report.line_coverage:.1%}, "
                f"Branch: {llm_report.branch_coverage:.1%}, "
                f"Method: {llm_report.method_coverage:.1%}, "
                f"Confidence: {llm_report.confidence:.1%}"
            )
            
            return StepResult(
                success=True,
                state=AgentState.ANALYZING,
                message=f"Coverage (LLM estimated): {llm_report.line_coverage:.1%}",
                data={
                    "line_coverage": llm_report.line_coverage,
                    "branch_coverage": llm_report.branch_coverage,
                    "method_coverage": llm_report.method_coverage,
                    "report": llm_report,
                    "source": CoverageSource.LLM_ESTIMATED.value,
                    "confidence": llm_report.confidence,
                    "uncovered_methods": llm_report.uncovered_methods,
                    "recommendations": llm_report.recommendations
                }
            )
        except Exception as e:
            logger.exception(f"[CoverageHandler] LLM estimation failed: {e}")
            return self._quick_estimate_fallback()
    
    def _quick_estimate_fallback(self) -> StepResult:
        """Quick heuristic-based fallback when LLM is not available.
        
        Returns:
            StepResult with heuristic coverage estimate
        """
        logger.info("[CoverageHandler] Using quick heuristic for coverage estimation")
        
        if not self.source_code or not self.test_code:
            return StepResult(
                success=False,
                state=AgentState.FAILED,
                message="Coverage analysis failed: No JaCoCo report and insufficient data for estimation"
            )
        
        if self._llm_evaluator:
            llm_report = self._llm_evaluator.quick_estimate(
                self.source_code,
                self.test_code,
                self.class_info
            )
        else:
            evaluator = LLMCoverageEvaluator(None)
            llm_report = evaluator.quick_estimate(
                self.source_code,
                self.test_code,
                self.class_info
            )
        
        return StepResult(
            success=True,
            state=AgentState.ANALYZING,
            message=f"Coverage (estimated): {llm_report.line_coverage:.1%}",
            data={
                "line_coverage": llm_report.line_coverage,
                "branch_coverage": llm_report.branch_coverage,
                "method_coverage": llm_report.method_coverage,
                "report": llm_report,
                "source": CoverageSource.LLM_ESTIMATED.value,
                "confidence": llm_report.confidence
            }
        )
    
    def set_context(
        self,
        source_code: Optional[str] = None,
        test_code: Optional[str] = None,
        class_info: Optional[Dict[str, Any]] = None
    ):
        """Set context for LLM-based coverage estimation.
        
        Args:
            source_code: Source code being tested
            test_code: Test code
            class_info: Class information from parsing
        """
        if source_code:
            self.source_code = source_code
        if test_code:
            self.test_code = test_code
        if class_info:
            self.class_info = class_info
    
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
