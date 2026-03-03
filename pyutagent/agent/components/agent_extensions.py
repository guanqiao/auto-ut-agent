"""Agent Extensions - Advanced features and extensions."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyutagent.core.config import get_settings

logger = logging.getLogger(__name__)


class AgentExtensions:
    """Extension features for ReActAgent.
    
    Provides advanced functionality:
    - Test quality analysis
    - Refactoring suggestions and application
    - Static analysis integration
    - Error knowledge base queries
    - Code interpreter execution
    """
    
    def __init__(self, agent_core: Any, components: Dict[str, Any]):
        """Initialize extensions.
        
        Args:
            agent_core: AgentCore instance
            components: Dictionary of all components
        """
        self.agent_core = agent_core
        self.components = components
        
        logger.debug("[AgentExtensions] Initialized")
    
    async def analyze_test_quality(self, test_code: Optional[str] = None) -> Dict[str, Any]:
        """Analyze test code quality using the quality analyzer.
        
        Args:
            test_code: Test code to analyze, or None to use current test file
            
        Returns:
            Quality analysis report
        """
        if test_code is None:
            if not self.agent_core.current_test_file:
                return {"error": "No test file available"}
            
            try:
                test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[AgentExtensions] Failed to read test file: {e}")
                return {"error": str(e)}
        
        try:
            quality_analyzer = self.components["test_quality_analyzer"]
            report = quality_analyzer.analyze(test_code)
            
            logger.info(f"[AgentExtensions] Quality analysis - Score: {report.overall_score:.1f}, "
                       f"Issues: {report.total_issues}, Critical: {report.critical_issues}")
            
            return {
                "overall_score": report.overall_score,
                "total_issues": report.total_issues,
                "critical_issues": report.critical_issues,
                "test_methods_analyzed": report.test_methods_analyzed,
                "dimension_scores": {
                    dim: {"score": score.score, "grade": score.grade}
                    for dim, score in report.dimension_scores.items()
                },
                "improvement_suggestions": report.improvement_suggestions,
                "report_markdown": quality_analyzer.generate_report_markdown(report)
            }
        except Exception as e:
            logger.error(f"[AgentExtensions] Quality analysis failed: {e}")
            return {"error": str(e)}
    
    async def suggest_refactorings(self, test_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """Suggest refactorings for test code.
        
        Args:
            test_code: Test code to analyze, or None to use current test file
            
        Returns:
            List of refactoring suggestions
        """
        if test_code is None:
            if not self.agent_core.current_test_file:
                return []
            
            try:
                test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[AgentExtensions] Failed to read test file: {e}")
                return []
        
        try:
            refactoring_engine = self.components["refactoring_engine"]
            suggestions = refactoring_engine.analyze(test_code)
            
            logger.info(f"[AgentExtensions] Refactoring analysis - Suggestions: {len(suggestions)}")
            
            return [
                {
                    "type": s.refactoring_type.value,
                    "description": s.description,
                    "location": s.location,
                    "priority": s.priority,
                    "confidence": s.confidence,
                    "impact": s.impact,
                    "rationale": s.rationale,
                    "suggested_code": s.suggested_code,
                    "original_code": s.original_code
                }
                for s in suggestions
            ]
        except Exception as e:
            logger.error(f"[AgentExtensions] Refactoring analysis failed: {e}")
            return []
    
    async def apply_refactoring(
        self,
        refactoring_type: str,
        test_code: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply a specific refactoring to test code.
        
        Args:
            refactoring_type: Type of refactoring to apply
            test_code: Test code to refactor, or None to use current test file
            **kwargs: Additional arguments for the refactoring
            
        Returns:
            Refactoring result
        """
        if test_code is None:
            if not self.agent_core.current_test_file:
                return {"success": False, "error": "No test file available"}
            
            try:
                test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[AgentExtensions] Failed to read test file: {e}")
                return {"success": False, "error": str(e)}
        
        try:
            refactoring_engine = self.components["refactoring_engine"]
            suggestions = refactoring_engine.analyze(test_code)
            
            matching_suggestions = [
                s for s in suggestions
                if s.refactoring_type.value == refactoring_type
            ]
            
            if not matching_suggestions:
                logger.warning(f"[AgentExtensions] No matching refactoring found: {refactoring_type}")
                return {"success": False, "error": f"No matching refactoring found: {refactoring_type}"}
            
            best_suggestion = max(matching_suggestions, key=lambda s: s.confidence)
            
            result = refactoring_engine.apply_refactoring(test_code, best_suggestion)
            
            if result.success:
                logger.info(f"[AgentExtensions] Refactoring applied - Type: {refactoring_type}, "
                           f"Changes: {len(result.changes_made)}")
                
                if self.agent_core.current_test_file:
                    await self._write_test_file(result.refactored_code)
            else:
                logger.warning(f"[AgentExtensions] Refactoring failed: {result.errors}")
            
            return {
                "success": result.success,
                "refactoring_type": result.refactoring_type.value,
                "changes_made": result.changes_made,
                "errors": result.errors
            }
        except Exception as e:
            logger.error(f"[AgentExtensions] Refactoring failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def auto_refactor_tests(
        self,
        test_code: Optional[str] = None,
        max_refactorings: int = 5,
        min_confidence: float = 0.8
    ) -> Dict[str, Any]:
        """Automatically apply high-confidence refactorings.
        
        Args:
            test_code: Test code to refactor, or None to use current test file
            max_refactorings: Maximum number of refactorings to apply
            min_confidence: Minimum confidence threshold
            
        Returns:
            Auto-refactoring results
        """
        if test_code is None:
            if not self.agent_core.current_test_file:
                return {"success": False, "error": "No test file available"}
            
            try:
                test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[AgentExtensions] Failed to read test file: {e}")
                return {"success": False, "error": str(e)}
        
        try:
            refactoring_engine = self.components["refactoring_engine"]
            refactored_code, results = refactoring_engine.auto_refactor(
                test_code,
                max_refactorings=max_refactorings,
                min_confidence=min_confidence
            )
            
            logger.info(f"[AgentExtensions] Auto-refactoring complete - "
                       f"Applied: {len(results)}, Original length: {len(test_code)}, "
                       f"New length: {len(refactored_code)}")
            
            if results and self.agent_core.current_test_file:
                await self._write_test_file(refactored_code)
            
            return {
                "success": len(results) > 0,
                "refactorings_applied": len(results),
                "results": [
                    {
                        "type": r.refactoring_type.value,
                        "success": r.success,
                        "changes": r.changes_made
                    }
                    for r in results
                ],
                "summary": refactoring_engine.get_refactoring_summary()
            }
        except Exception as e:
            logger.error(f"[AgentExtensions] Auto-refactoring failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_test_in_interpreter(
        self,
        test_code: str,
        test_method_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute test code in the code interpreter.
        
        Args:
            test_code: Test code to execute
            test_method_name: Specific test method to run, or None for all
            
        Returns:
            Execution result
        """
        try:
            code_interpreter = self.components["code_interpreter"]
            result = code_interpreter.execute_test(
                test_code=test_code,
                test_method_name=test_method_name
            )
            
            logger.info(f"[AgentExtensions] Code interpreter execution - "
                       f"Success: {result.success}, "
                       f"TestsRun: {result.tests_run}, "
                       f"Failures: {result.failures}, "
                       f"Errors: {result.errors}")
            
            return {
                "success": result.success,
                "output": result.output,
                "error_output": result.error_output,
                "tests_run": result.tests_run,
                "tests_passed": result.tests_passed,
                "failures": result.failures,
                "errors": result.errors,
                "execution_time": result.execution_time,
                "assertion_failures": result.assertion_failures,
                "runtime_errors": result.runtime_errors
            }
        except Exception as e:
            logger.error(f"[AgentExtensions] Code interpreter execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def validate_test_with_interpreter(
        self,
        test_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate test code using the code interpreter before compilation.
        
        Args:
            test_code: Test code to validate, or None to use current test file
            
        Returns:
            Validation result with potential issues
        """
        if test_code is None:
            if not self.agent_core.current_test_file:
                return {"valid": False, "error": "No test file available"}
            
            try:
                test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
                test_code = test_file_path.read_text(encoding='utf-8')
            except Exception as e:
                logger.error(f"[AgentExtensions] Failed to read test file: {e}")
                return {"valid": False, "error": str(e)}
        
        try:
            code_interpreter = self.components["code_interpreter"]
            validation_result = code_interpreter.validate_test_code(test_code)
            
            logger.info(f"[AgentExtensions] Test validation - Valid: {validation_result['valid']}, "
                       f"Issues: {len(validation_result.get('issues', []))}")
            
            return validation_result
        except Exception as e:
            logger.error(f"[AgentExtensions] Test validation failed: {e}")
            return {"valid": False, "error": str(e)}
    
    def get_quality_trend(self, last_n: int = 10) -> Dict[str, Any]:
        """Get quality trend from recent analyses.
        
        Args:
            last_n: Number of recent analyses to include
            
        Returns:
            Quality trend data
        """
        try:
            quality_analyzer = self.components["test_quality_analyzer"]
            trend = quality_analyzer.get_quality_trend(last_n)
            logger.debug(f"[AgentExtensions] Quality trend: {trend.get('trend', 'unknown')}")
            return trend
        except Exception as e:
            logger.error(f"[AgentExtensions] Failed to get quality trend: {e}")
            return {"error": str(e)}
    
    async def run_static_analysis(
        self,
        source_path: Optional[str] = None,
        tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run static analysis on source code.
        
        Args:
            source_path: Path to source file/directory, or None for project
            tools: List of tools to use (spotbugs, pmd, checkstyle)
            
        Returns:
            Static analysis results
        """
        try:
            static_analysis_manager = await self._get_static_analysis_manager()
            
            target_files = None
            if source_path:
                path = Path(source_path)
                if path.is_file():
                    target_files = [str(path)]
            
            results = await static_analysis_manager.run_all_analysis(
                target_files=target_files,
                include_tests=False
            )
            
            formatted_results = {}
            total_issues = 0
            
            for tool_type, analysis_result in results.items():
                tool_name = tool_type.name.lower()
                formatted_results[tool_name] = {
                    "success": analysis_result.success,
                    "bug_count": analysis_result.bug_count,
                    "bugs": [
                        {
                            "type": bug.bug_type,
                            "severity": bug.severity.value,
                            "message": bug.message,
                            "class_name": bug.class_name,
                            "method_name": bug.method_name,
                            "line_number": bug.line_number,
                            "suggestion": bug.suggestion
                        }
                        for bug in analysis_result.bugs
                    ]
                }
                total_issues += analysis_result.bug_count
            
            logger.info(f"[AgentExtensions] Static analysis complete - "
                       f"Tools: {list(formatted_results.keys())}, "
                       f"Total issues: {total_issues}")
            
            return {
                "success": True,
                "results": formatted_results,
                "total_issues": total_issues
            }
        except Exception as e:
            logger.error(f"[AgentExtensions] Static analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def query_error_knowledge(
        self,
        error_message: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Query error knowledge base for similar errors and solutions.
        
        Args:
            error_message: Error message to search for
            limit: Maximum number of results
            
        Returns:
            List of similar errors and their solutions
        """
        try:
            from pyutagent.core.error_knowledge_base import ErrorContext, ErrorCategory
            
            error_knowledge_base = await self._get_error_knowledge_base()
            error_context = ErrorContext(
                error_message=error_message,
                category=ErrorCategory.UNKNOWN
            )
            
            similar_errors = error_knowledge_base.find_similar_errors(
                error_context=error_context,
                min_similarity=0.6,
                max_results=limit
            )
            
            logger.info(f"[AgentExtensions] Error knowledge query - "
                       f"Found: {len(similar_errors)} similar errors")
            
            return [
                {
                    "solution_id": result.solution.solution_id,
                    "error_pattern": result.solution.error_pattern,
                    "category": result.solution.category.value,
                    "fix_description": result.solution.fix_description,
                    "fix_code": result.solution.fix_code,
                    "similarity_score": result.similarity_score,
                    "success_rate": result.solution.success_rate,
                    "status": result.solution.status.value
                }
                for result in similar_errors
            ]
        except Exception as e:
            logger.error(f"[AgentExtensions] Error knowledge query failed: {e}")
            return []
    
    async def record_error_solution(
        self,
        error_message: str,
        error_category: str,
        solution_description: str,
        success: bool
    ) -> bool:
        """Record an error and its solution to the knowledge base.
        
        Args:
            error_message: The error message
            error_category: Category of the error
            solution_description: Description of the solution
            success: Whether the solution worked
            
        Returns:
            True if recorded successfully
        """
        try:
            from pyutagent.core.error_knowledge_base import ErrorContext, ErrorCategory
            
            error_knowledge_base = await self._get_error_knowledge_base()
            category = ErrorCategory.UNKNOWN
            try:
                category = ErrorCategory(error_category.lower())
            except ValueError:
                pass
            
            error_context = ErrorContext(
                error_message=error_message,
                category=category
            )
            
            solution_id = error_knowledge_base.record_solution(
                error_context=error_context,
                fix_description=solution_description
            )
            
            error_knowledge_base.record_outcome(
                error_context=error_context,
                solution_id=solution_id,
                success=success
            )
            
            logger.info(f"[AgentExtensions] Recorded error solution - "
                       f"Category: {error_category}, Success: {success}")
            
            return True
        except Exception as e:
            logger.error(f"[AgentExtensions] Failed to record error solution: {e}")
            return False
    
    async def _get_static_analysis_manager(self) -> Any:
        """Get or create static analysis manager with lazy initialization."""
        if self.components["_static_analysis_manager"] is None:
            try:
                from pyutagent.tools.static_analysis_manager import StaticAnalysisManager
                self.components["_static_analysis_manager"] = StaticAnalysisManager(self.agent_core.project_path)
                logger.debug("[AgentExtensions] StaticAnalysisManager lazy-initialized")
            except Exception as e:
                logger.warning(f"[AgentExtensions] StaticAnalysisManager initialization failed: {e}, using no-op fallback")
                self.components["_static_analysis_manager"] = self._create_noop_static_analysis_manager()
        
        return self.components["_static_analysis_manager"]
    
    def _create_noop_static_analysis_manager(self) -> Any:
        """Create a no-op static analysis manager as fallback."""
        class NoOpStaticAnalysisManager:
            async def run_all_analysis(self, target_files=None, include_tests=False):
                from dataclasses import dataclass, field
                from typing import List
                from enum import Enum
                
                class AnalysisToolType(Enum):
                    SPOTBUGS = "spotbugs"
                    PMD = "pmd"
                    CHECKSTYLE = "checkstyle"
                
                @dataclass
                class BugInfo:
                    bug_type: str = ""
                    severity: str = "LOW"
                    message: str = ""
                    class_name: str = ""
                    method_name: str = ""
                    line_number: int = 0
                    suggestion: str = ""
                
                @dataclass
                class AnalysisResult:
                    tool_type: AnalysisToolType = AnalysisToolType.SPOTBUGS
                    success: bool = False
                    bug_count: int = 0
                    bugs: List[BugInfo] = field(default_factory=list)
                
                return {AnalysisToolType.SPOTBUGS: AnalysisResult()}
        
        return NoOpStaticAnalysisManager()
    
    async def _get_error_knowledge_base(self) -> Any:
        """Get or create error knowledge base with lazy initialization."""
        if self.components["_error_knowledge_base"] is None:
            try:
                from pyutagent.core.error_knowledge_base import ErrorKnowledgeBase
                db_path = str(Path(self.agent_core.project_path) / ".utagent" / "error_knowledge.db")
                self.components["_error_knowledge_base"] = ErrorKnowledgeBase(db_path=db_path)
                logger.debug("[AgentExtensions] ErrorKnowledgeBase lazy-initialized")
            except Exception as e:
                logger.warning(f"[AgentExtensions] ErrorKnowledgeBase initialization failed: {e}, using in-memory fallback")
                self.components["_error_knowledge_base"] = self._create_in_memory_error_knowledge_base()
        
        return self.components["_error_knowledge_base"]
    
    def _create_in_memory_error_knowledge_base(self) -> Any:
        """Create an in-memory error knowledge base as fallback."""
        from dataclasses import dataclass, field
        from typing import List, Optional
        from enum import Enum
        
        class ErrorCategory(Enum):
            COMPILATION = "compilation"
            RUNTIME = "runtime"
            ASSERTION = "assertion"
            TIMEOUT = "timeout"
            UNKNOWN = "unknown"
        
        @dataclass
        class ErrorContext:
            error_message: str
            category: ErrorCategory = ErrorCategory.UNKNOWN
            stack_trace: Optional[str] = None
            test_class: Optional[str] = None
            test_method: Optional[str] = None
        
        @dataclass
        class SearchResult:
            solution: Any
            similarity_score: float = 0.0
        
        class InMemoryErrorKnowledgeBase:
            def __init__(self):
                self._solutions: List[Any] = []
            
            def find_similar_errors(self, error_context: ErrorContext, min_similarity: float = 0.6, max_results: int = 5) -> List[SearchResult]:
                return []
            
            def record_solution(self, error_context: ErrorContext, fix_description: str, fix_code: Optional[str] = None, metadata: Optional[dict] = None) -> str:
                return "in-memory-solution-id"
            
            def record_outcome(self, error_context: ErrorContext, solution_id: str, success: bool, metadata: Optional[dict] = None) -> None:
                pass
        
        return InMemoryErrorKnowledgeBase()
    
    async def _get_adaptive_strategy_manager(self) -> Any:
        """Get or create adaptive strategy manager with lazy initialization."""
        if self.components["_adaptive_strategy_manager"] is None:
            try:
                from pyutagent.core.adaptive_strategy import AdaptiveStrategyManager
                db_path = str(Path(self.agent_core.project_path) / ".utagent" / "adaptive_strategy.db")
                self.components["_adaptive_strategy_manager"] = AdaptiveStrategyManager(db_path=db_path)
                logger.debug("[AgentExtensions] AdaptiveStrategyManager lazy-initialized")
            except Exception as e:
                logger.warning(f"[AgentExtensions] AdaptiveStrategyManager initialization failed: {e}, using default fallback")
                self.components["_adaptive_strategy_manager"] = self._create_default_adaptive_strategy_manager()
        
        return self.components["_adaptive_strategy_manager"]
    
    def _create_default_adaptive_strategy_manager(self) -> Any:
        """Create a default adaptive strategy manager as fallback."""
        class DefaultAdaptiveStrategyManager:
            def select_strategy(self, error_category: str, available_strategies: List, context: dict, allow_exploration: bool = True):
                if available_strategies:
                    return available_strategies[0]
                from pyutagent.core.parallel_recovery import RecoveryStrategy
                return RecoveryStrategy.DEFAULT
            
            def record_attempt(self, strategy, error_category: str, success: bool, execution_time_ms: float, context: dict):
                pass
        
        return DefaultAdaptiveStrategyManager()
    
    async def _get_vector_store(self) -> Optional[Any]:
        """Get or create vector store with lazy initialization (optional)."""
        if self.components["_vector_store"] is None:
            try:
                from pyutagent.memory.vector_store import SQLiteVecStore
                db_path = str(Path(self.agent_core.project_path) / ".utagent" / "vectors.db")
                self.components["_vector_store"] = SQLiteVecStore(db_path=db_path, dimension=384)
                logger.debug("[AgentExtensions] VectorStore lazy-initialized")
            except Exception as e:
                logger.warning(f"[AgentExtensions] VectorStore initialization failed: {e}")
                self.components["_vector_store"] = None
        
        return self.components["_vector_store"]
    
    async def _write_test_file(self, code: str):
        """Write test code to file.
        
        Args:
            code: Test code to write
        """
        if not self.agent_core.current_test_file:
            logger.warning("[AgentExtensions] Cannot write test file - current_test_file is empty")
            return
        
        try:
            test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
            test_file_path.write_text(code, encoding='utf-8')
            logger.info(f"[AgentExtensions] Wrote test file - Path: {test_file_path}, Length: {len(code)}")
        except Exception as e:
            logger.error(f"[AgentExtensions] Failed to write test file: {e}")
