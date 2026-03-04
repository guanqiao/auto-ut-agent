"""Self-reflection mechanism for critiquing and improving generated code.

This module provides the ability for the agent to critique and improve its own output,
achieving higher quality test code and reducing repeated errors.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any

from ..core.component_registry import SimpleComponent, component


logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Dimensions for code quality evaluation."""
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"
    COMPLETENESS = "completeness"
    CORRECTNESS = "correctness"
    BEST_PRACTICES = "best_practices"


class IssueSeverity(Enum):
    """Severity levels for identified issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SUGGESTION = "suggestion"


@dataclass
class QualityMetric:
    """Quality metric for a dimension."""
    dimension: QualityDimension
    score: float
    details: str = ""


@dataclass
class IdentifiedIssue:
    """Identified issue in generated code."""
    issue_type: str
    severity: IssueSeverity
    description: str
    line_number: Optional[int] = None
    suggestion: str = ""
    confidence: float = 0.0


@dataclass
class CoverageEstimate:
    """Estimated test coverage."""
    estimated_line_coverage: float
    estimated_branch_coverage: float
    estimated_method_coverage: Dict[str, float]
    uncovered_scenarios: List[str] = field(default_factory=list)


@dataclass
class CritiqueResult:
    """Result of critiquing generated test code."""
    overall_quality_score: float
    quality_metrics: List[QualityMetric]
    identified_issues: List[IdentifiedIssue]
    coverage_estimate: CoverageEstimate
    improvement_suggestions: List[str]
    should_regenerate: bool
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_issues_by_severity(self, severity: IssueSeverity) -> List[IdentifiedIssue]:
        """Get issues filtered by severity."""
        return [i for i in self.identified_issues if i.severity == severity]
    
    def get_critical_issues(self) -> List[IdentifiedIssue]:
        """Get all critical issues."""
        return self.get_issues_by_severity(IssueSeverity.CRITICAL)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_quality_score": self.overall_quality_score,
            "quality_metrics": [
                {
                    "dimension": m.dimension.name,
                    "score": m.score,
                    "details": m.details
                }
                for m in self.quality_metrics
            ],
            "identified_issues": [
                {
                    "issue_type": i.issue_type,
                    "severity": i.severity.name,
                    "description": i.description,
                    "line_number": i.line_number,
                    "suggestion": i.suggestion,
                    "confidence": i.confidence
                }
                for i in self.identified_issues
            ],
            "coverage_estimate": {
                "line_coverage": self.coverage_estimate.estimated_line_coverage,
                "branch_coverage": self.coverage_estimate.estimated_branch_coverage,
                "method_coverage": self.coverage_estimate.estimated_method_coverage,
                "uncovered_scenarios": self.coverage_estimate.uncovered_scenarios,
            },
            "improvement_suggestions": self.improvement_suggestions,
            "should_regenerate": self.should_regenerate,
            "confidence": self.confidence,
        }


@dataclass
class ImprovementResult:
    """Result of self-improvement."""
    original_code: str
    improved_code: str
    improvements_made: List[str]
    quality_before: float
    quality_after: float
    iterations: int
    success: bool


@component(
    component_id="self_reflection",
    dependencies=[],
    description="Self-reflection mechanism for code quality evaluation"
)
class SelfReflection(SimpleComponent):
    """Self-reflection mechanism for critiquing and improving generated code.
    
    Features:
    - Multi-dimensional quality evaluation
    - Test coverage estimation
    - Issue identification with suggestions
    - Self-improvement loop
    - Learning from feedback
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        quality_threshold: float = 0.7,
        max_improvement_iterations: int = 3
    ):
        """Initialize self-reflection mechanism.
        
        Args:
            llm_client: LLM client for analysis
            quality_threshold: Minimum quality threshold
            max_improvement_iterations: Maximum improvement iterations
        """
        super().__init__()
        self.llm_client = llm_client
        self.quality_threshold = quality_threshold
        self.max_improvement_iterations = max_improvement_iterations
        
        self._critique_history: List[CritiqueResult] = []
        self._improvement_patterns: Dict[str, List[str]] = {}
        
        logger.info(f"[SelfReflection] Initialized - Quality threshold: {quality_threshold}")
    
    async def critique_generated_test(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        class_info: Optional[Dict[str, Any]] = None
    ) -> CritiqueResult:
        """Critique generated test code.
        
        Args:
            test_code: Generated test code
            source_code: Original source code being tested
            class_info: Information about the class being tested
            
        Returns:
            CritiqueResult with detailed analysis
        """
        logger.info("[SelfReflection] Starting critique of generated test code")
        
        quality_metrics = await self._evaluate_quality(test_code, source_code)
        coverage_estimate = await self._estimate_coverage(test_code, source_code, class_info)
        issues = await self._identify_issues(test_code, source_code)
        
        overall_score = self._calculate_overall_score(quality_metrics, issues)
        suggestions = self._generate_improvement_suggestions(issues, quality_metrics)
        
        should_regenerate = (
            overall_score < self.quality_threshold or
            any(i.severity == IssueSeverity.CRITICAL for i in issues)
        )
        
        result = CritiqueResult(
            overall_quality_score=overall_score,
            quality_metrics=quality_metrics,
            identified_issues=issues,
            coverage_estimate=coverage_estimate,
            improvement_suggestions=suggestions,
            should_regenerate=should_regenerate,
            confidence=self._calculate_confidence(quality_metrics, issues)
        )
        
        self._critique_history.append(result)
        
        logger.info(f"[SelfReflection] Critique complete - Score: {overall_score:.2f}, Issues: {len(issues)}")
        
        return result
    
    async def _evaluate_quality(
        self,
        test_code: str,
        source_code: Optional[str]
    ) -> List[QualityMetric]:
        """Evaluate code quality across multiple dimensions."""
        metrics = []
        
        metrics.append(await self._evaluate_readability(test_code))
        metrics.append(await self._evaluate_maintainability(test_code))
        metrics.append(await self._evaluate_testability(test_code))
        metrics.append(await self._evaluate_completeness(test_code, source_code))
        metrics.append(await self._evaluate_correctness(test_code))
        metrics.append(await self._evaluate_best_practices(test_code))
        
        return metrics
    
    async def _evaluate_readability(self, code: str) -> QualityMetric:
        """Evaluate code readability."""
        score = 1.0
        details = []
        
        lines = code.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        if avg_line_length > 100:
            score -= 0.1
            details.append(f"Long average line length ({avg_line_length:.1f})")
        
        if avg_line_length < 20:
            score += 0.2
            details.append(f"Good average line length ({avg_line_length:.1f})")
        
        if len(lines) > 500:
            score -= 0.2
            details.append(f"Very long file ({len(lines)} lines)")
        
        return QualityMetric(
            dimension=QualityDimension.READABILITY,
            score=max(0.0, min(1.0, score)),
            details="; ".join(details) if details else "Good readability"
        )
    
    async def _evaluate_maintainability(self, code: str) -> QualityMetric:
        """Evaluate code maintainability."""
        score = 1.0
        details = []
        
        if 'TODO' in code or 'FIXME' in code:
            score -= 0.1
            details.append("TODO/FIXME comments found")
        
        if '@Mock' in code or 'Mockito' in code:
            score += 0.1
            details.append("Good use of mocking")
        
        if '@BeforeEach' not in code:
            score -= 0.1
            details.append("Missing @BeforeEach setup")
        
        if '@Test' not in code:
            score -= 0.2
            details.append("Missing @Test annotations")
        
        if '@Disabled' in code:
            score -= 0.1
            details.append("@Disabled test method detected")
        
        if 'System.out' in code:
            score -= 0.1
            details.append("System.out.println detected")
        
        return QualityMetric(
            dimension=QualityDimension.MAINTAINABILITY,
            score=max(0.0, min(1.0, score)),
            details="; ".join(details) if details else "Good maintainability"
        )
    
    async def _evaluate_testability(self, code: str) -> QualityMetric:
        """Evaluate code testability."""
        score = 1.0
        details = []
        
        assertion_count = len(re.findall(r'assert\w+\s*\(', code))
        if assertion_count == 0:
            score -= 0.3
            details.append("No assertions found")
        elif assertion_count < 3:
            score -= 0.1
            details.append(f"Few assertions ({assertion_count})")
        else:
            details.append(f"Good assertion count ({assertion_count})")
        
        test_method_count = len(re.findall(r'@Test', code))
        if test_method_count == 0:
            score -= 0.3
            details.append("No test methods found")
        else:
            details.append(f"Found {test_method_count} test methods")
        
        return QualityMetric(
            dimension=QualityDimension.TESTABILITY,
            score=max(0.0, min(1.0, score)),
            details="; ".join(details)
        )
    
    async def _evaluate_completeness(
        self,
        test_code: str,
        source_code: Optional[str]
    ) -> QualityMetric:
        """Evaluate test completeness."""
        score = 1.0
        details = []
        
        if not source_code:
            return QualityMetric(
                dimension=QualityDimension.COMPLETENESS,
                score=0.7,
                details="Cannot evaluate completeness without source code"
            )
        
        source_methods = set(re.findall(r'public\s+\w+\s+(\w+)\s*\(', source_code))
        test_methods = set(re.findall(r'void\s+(test\w+|should\w+|when\w+)\s*\(', test_code, re.IGNORECASE))
        
        if source_methods:
            coverage_ratio = min(len(test_methods) / max(len(source_methods), 1), 1.0)
            score = coverage_ratio
            details.append(f"Method coverage ratio: {coverage_ratio:.2f}")
        
        if 'assertThrows' not in test_code:
            score -= 0.1
            details.append("Missing exception tests")
        
        return QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=max(0.0, min(1.0, score)),
            details="; ".join(details) if details else "Good completeness"
        )
    
    async def _evaluate_correctness(self, code: str) -> QualityMetric:
        """Evaluate code correctness."""
        score = 1.0
        details = []
        
        if 'assert' not in code:
            score -= 0.3
            details.append("No assertions found")
        
        if re.search(r'assertEquals\s*\(\s*expected\s*,\s*actual\s*\)', code):
            details.append("Good assertion pattern")
        
        if 'assertNull' in code or 'assertNotNull' in code:
            details.append("Null checks present")
        
        return QualityMetric(
            dimension=QualityDimension.CORRECTNESS,
            score=max(0.0, min(1.0, score)),
            details="; ".join(details) if details else "Good correctness"
        )
    
    async def _evaluate_best_practices(self, code: str) -> QualityMetric:
        """Evaluate best practices adherence."""
        score = 1.0
        details = []
        
        if '@ParameterizedTest' in code:
            score += 0.1
            details.append("Uses parameterized tests")
        
        if '@Nested' in code:
            score += 0.1
            details.append("Uses nested test classes")
        
        if '@DisplayName' in code:
            score += 0.1
            details.append("Uses display names")
        
        if 'assertThrows' not in code:
            score -= 0.1
            details.append("Consider using assertThrows for exception tests")
        
        if '@Timeout' not in code and 'assertTimeout' not in code:
            details.append("Consider adding timeout for long-running tests")
        
        return QualityMetric(
            dimension=QualityDimension.BEST_PRACTICES,
            score=max(0.0, min(1.0, score)),
            details="; ".join(details) if details else "Follows best practices"
        )
    
    async def _estimate_coverage(
        self,
        test_code: str,
        source_code: Optional[str] = None,
        class_info: Optional[Dict[str, Any]] = None
    ) -> CoverageEstimate:
        """Estimate test coverage."""
        line_coverage = 0.0
        branch_coverage = 0.0
        method_coverage = {}
        uncovered_scenarios = []
        
        test_methods = re.findall(r'void\s+(test\w+|should\w+|when\w+)\s*\(', test_code, re.IGNORECASE)
        
        if class_info and 'methods' in class_info:
            methods = class_info['methods']
            for method in methods:
                method_name = method.get('name', '') if isinstance(method, dict) else str(method)
                has_test = any(method_name.lower() in tm.lower() for tm in test_methods)
                method_coverage[method_name] = 1.0 if has_test else 0.0
                if not has_test:
                    uncovered_scenarios.append(f"Method '{method_name}' may not be tested")
        
        if source_code:
            source_lines = source_code.split('\n')
            code_lines = [l for l in source_lines if l.strip() and not l.strip().startswith(('//', '/*', '*'))]
            if code_lines:
                line_coverage = min(len(test_methods) * 10 / len(code_lines), 1.0)
            
            branch_count = len(re.findall(r'\bif\b|\bswitch\b|\bfor\b|\bwhile\b', source_code))
            if branch_count > 0:
                branch_coverage = min(len(test_methods) / branch_count, 1.0)
        
        return CoverageEstimate(
            estimated_line_coverage=line_coverage,
            estimated_branch_coverage=branch_coverage,
            estimated_method_coverage=method_coverage,
            uncovered_scenarios=uncovered_scenarios
        )
    
    async def _identify_issues(
        self,
        test_code: str,
        source_code: Optional[str]
    ) -> List[IdentifiedIssue]:
        """Identify issues in generated code."""
        issues = []
        
        issues.extend(self._check_mock_issues(test_code, source_code))
        issues.extend(self._check_null_pointer_issues(test_code, source_code))
        issues.extend(self._check_setup_issues(test_code, source_code))
        issues.extend(self._check_naming_issues(test_code, source_code))
        issues.extend(self._check_boundary_issues(test_code, source_code))
        issues.extend(self._check_assertion_issues(test_code, source_code))
        issues.extend(self._check_timeout_issues(test_code, source_code))
        issues.extend(self._check_duplicate_code(test_code, source_code))
        issues.extend(self._check_hardcoded_values(test_code, source_code))
        
        return issues
    
    def _check_mock_issues(
        self, test_code: str, source_code: Optional[str]
    ) -> List[IdentifiedIssue]:
        """Check for mock-related issues."""
        issues = []
        
        if '@Mock' in test_code or 'Mockito' in test_code:
            if 'when(' not in test_code and 'doReturn' not in test_code:
                issues.append(IdentifiedIssue(
                    issue_type="UNSTUBBED_MOCK",
                    severity=IssueSeverity.HIGH,
                    description="Mocks declared but not stubbed",
                    suggestion="Add when().thenReturn() for mock behavior"
                ))
        
        mock_count = len(re.findall(r'@Mock', test_code))
        if mock_count > 3 and '@InjectMocks' not in test_code:
            issues.append(IdentifiedIssue(
                issue_type="MANUAL_MOCK_INJECTION",
                severity=IssueSeverity.LOW,
                description="Multiple mocks without @InjectMocks",
                suggestion="Consider using @InjectMocks for cleaner code"
            ))
        
        return issues
    
    def _check_null_pointer_issues(
        self, test_code: str, source_code: Optional[str]
    ) -> List[IdentifiedIssue]:
        """Check for potential null pointer issues."""
        issues = []
        
        if source_code and 'null' in source_code.lower():
            if 'null' not in test_code.lower() and 'assertNull' not in test_code:
                issues.append(IdentifiedIssue(
                    issue_type="MISSING_NULL_TEST",
                    severity=IssueSeverity.MEDIUM,
                    description="Source handles null but test doesn't cover it",
                    suggestion="Add test for null input handling"
                ))
        
        return issues
    
    def _check_setup_issues(self, test_code: str, source_code: Optional[str]) -> List[IdentifiedIssue]:
        """Check for setup issues."""
        issues = []
        
        setup_in_tests = re.findall(r'@Test.*?\n.*?new\s+\w+', test_code, re.DOTALL)
        if len(setup_in_tests) > 2:
            issues.append(IdentifiedIssue(
                issue_type="DUPLICATE_SETUP",
                severity=IssueSeverity.LOW,
                description="Similar setup code in multiple tests",
                suggestion="Consider extracting to @BeforeEach method"
            ))
        
        return issues
    
    def _check_naming_issues(self, test_code: str, source_code: Optional[str]) -> List[IdentifiedIssue]:
        """Check for naming issues."""
        issues = []
        
        test_methods = re.findall(r'void\s+(\w+)\s*\(', test_code)
        for method_name in test_methods:
            if not method_name.startswith(('test', 'should', 'when', 'given')):
                issues.append(IdentifiedIssue(
                    issue_type="NAMING_CONVENTION",
                    severity=IssueSeverity.LOW,
                    description=f"Test method '{method_name}' may not follow naming convention",
                    suggestion="Consider using should_xxx_when_xxx pattern"
                ))
        
        return issues
    
    def _check_boundary_issues(self, test_code: str, source_code: Optional[str]) -> List[IdentifiedIssue]:
        """Check for boundary value tests."""
        issues = []
        
        if not source_code:
            return issues
        
        if re.search(r'\bmax\b|\bmin\b|\blimit\b|\bboundary\b', source_code, re.IGNORECASE):
            if 'max' not in test_code.lower() and 'min' not in test_code.lower():
                issues.append(IdentifiedIssue(
                    issue_type="MISSING_BOUNDARY_TEST",
                    severity=IssueSeverity.MEDIUM,
                    description="Source has boundary conditions but tests don't cover them",
                    suggestion="Add tests for boundary values (max, min, edge cases)"
                ))
        
        return issues
    
    def _check_assertion_issues(self, test_code: str, source_code: Optional[str]) -> List[IdentifiedIssue]:
        """Check for assertion issues."""
        issues = []
        
        assertions = re.findall(r'assert(Equals|True|False|NotNull|Null|Throws)\s*\(', test_code)
        if not assertions:
            issues.append(IdentifiedIssue(
                issue_type="MISSING_ASSERTIONS",
                severity=IssueSeverity.HIGH,
                description="No assertions found in test",
                suggestion="Add assertions to verify expected behavior"
            ))
        
        return issues
    
    def _check_timeout_issues(self, test_code: str, source_code: Optional[str]) -> List[IdentifiedIssue]:
        """Check for timeout issues."""
        issues = []
        
        timeout_match = re.search(r'assertTimeout\s*\(\s*Duration\.ofMillis\s*\(\s*(\d+)', test_code)
        if timeout_match:
            try:
                timeout_value = int(timeout_match.group(1))
                if timeout_value > 5000:
                    issues.append(IdentifiedIssue(
                        issue_type="LONG_TIMEOUT",
                        severity=IssueSeverity.LOW,
                        description=f"Timeout of {timeout_value}ms may be too long",
                        suggestion="Consider reducing timeout for faster test feedback"
                    ))
            except ValueError:
                pass
        
        return issues
    
    def _check_duplicate_code(self, test_code: str, source_code: Optional[str]) -> List[IdentifiedIssue]:
        """Check for duplicate code."""
        issues = []
        
        lines = test_code.split('\n')
        seen_blocks = {}
        
        for i in range(len(lines) - 5):
            block = '\n'.join(lines[i:i+5])
            if block in seen_blocks:
                seen_blocks[block].append(i)
            else:
                seen_blocks[block] = [i]
        
        for block, positions in seen_blocks.items():
            if len(positions) > 1 and len(block.strip()) > 30:
                issues.append(IdentifiedIssue(
                    issue_type="DUPLICATE_CODE",
                    severity=IssueSeverity.LOW,
                    description=f"Duplicate code block detected at lines {positions}",
                    suggestion="Consider extracting common code to a helper method"
                ))
                break
        
        return issues
    
    def _check_hardcoded_values(self, test_code: str, source_code: Optional[str]) -> List[IdentifiedIssue]:
        """Check for hardcoded values."""
        issues = []
        
        hardcoded_strings = re.findall(r'"([^"]{3,})"', test_code)
        for value in hardcoded_strings:
            if value not in ['expected', 'actual', 'result', 'test']:
                issues.append(IdentifiedIssue(
                    issue_type="HARDCODED_VALUE",
                    severity=IssueSeverity.SUGGESTION,
                    description=f"Hardcoded value detected: '{value[:20]}...'",
                    suggestion="Consider using constants or test parameters"
                ))
                break
        
        return issues
    
    def _generate_improvement_suggestions(
        self,
        issues: List[IdentifiedIssue],
        metrics: List[QualityMetric]
    ) -> List[str]:
        """Generate improvement suggestions based on issues and metrics."""
        suggestions = []
        
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        if critical_issues:
            suggestions.append("CRITICAL: Fix the following issues before proceeding:")
            for issue in critical_issues:
                suggestions.append(f"  - {issue.description}: {issue.suggestion}")
        
        high_issues = [i for i in issues if i.severity == IssueSeverity.HIGH]
        if high_issues:
            suggestions.append("HIGH PRIORITY: Address these issues:")
            for issue in high_issues:
                suggestions.append(f"  - {issue.description}: {issue.suggestion}")
        
        low_score_metrics = [m for m in metrics if m.score < 0.5]
        if low_score_metrics:
            suggestions.append("QUALITY: Improve the following dimensions:")
            for metric in low_score_metrics:
                suggestions.append(f"  - {metric.dimension.name}: {metric.details}")
        
        return suggestions
    
    def _calculate_overall_score(
        self,
        metrics: List[QualityMetric],
        issues: List[IdentifiedIssue]
    ) -> float:
        """Calculate overall quality score."""
        if not metrics:
            return 0.0
        
        metric_score = sum(m.score for m in metrics) / len(metrics)
        
        issue_penalty = sum(
            0.1 if i.severity == IssueSeverity.LOW else
            0.2 if i.severity == IssueSeverity.MEDIUM else
            0.4 if i.severity == IssueSeverity.HIGH else
            0.6 if i.severity == IssueSeverity.CRITICAL else
            0.05
            for i in issues
        )
        
        return max(0.0, min(1.0, metric_score - issue_penalty))
    
    def _calculate_confidence(
        self,
        metrics: List[QualityMetric],
        issues: List[IdentifiedIssue]
    ) -> float:
        """Calculate confidence in the critique."""
        if not metrics:
            return 0.5
        
        metric_confidence = sum(m.score for m in metrics) / len(metrics)
        
        if issues:
            issue_confidence = sum(i.confidence if i.confidence > 0 else 0.5 for i in issues) / len(issues)
        else:
            issue_confidence = 0.8
        
        return (metric_confidence + issue_confidence) / 2
    
    def get_critique_stats(self) -> Dict[str, Any]:
        """Get statistics about critiques performed."""
        if not self._critique_history:
            return {"total": 0}
        
        return {
            "total": len(self._critique_history),
            "average_score": sum(c.overall_quality_score for c in self._critique_history) / len(self._critique_history),
            "average_issues": sum(len(c.identified_issues) for c in self._critique_history) / len(self._critique_history),
            "regeneration_rate": sum(1 for c in self._critique_history if c.should_regenerate) / len(self._critique_history),
        }
