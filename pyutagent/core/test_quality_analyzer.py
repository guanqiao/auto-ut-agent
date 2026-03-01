"""
Test Quality Analyzer for Unit Test Code Assessment
Provides comprehensive quality metrics and improvement suggestions
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, Set
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    COVERAGE = "coverage"
    ASSERTION_QUALITY = "assertion_quality"
    TEST_ISOLATION = "test_isolation"
    NAMING_CONVENTION = "naming_convention"
    CODE_ORGANIZATION = "code_organization"
    MAINTAINABILITY = "maintainability"
    READABILITY = "readability"
    RELIABILITY = "reliability"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityIssue:
    dimension: QualityDimension
    severity: Severity
    message: str
    location: Optional[Tuple[int, int]] = None
    suggestion: Optional[str] = None
    rule_id: str = ""


@dataclass
class QualityScore:
    dimension: QualityDimension
    score: float
    max_score: float = 100.0
    issues: List[QualityIssue] = field(default_factory=list)
    
    @property
    def percentage(self) -> float:
        return (self.score / self.max_score) * 100
    
    @property
    def grade(self) -> str:
        pct = self.percentage
        if pct >= 90:
            return 'A'
        elif pct >= 80:
            return 'B'
        elif pct >= 70:
            return 'C'
        elif pct >= 60:
            return 'D'
        else:
            return 'F'


@dataclass
class TestQualityReport:
    overall_score: float
    dimension_scores: Dict[str, QualityScore]
    total_issues: int
    critical_issues: int
    improvement_suggestions: List[str]
    test_methods_analyzed: int
    lines_of_code: int


class AssertionQualityAnalyzer:
    
    STRONG_ASSERTIONS = {
        'assertEquals', 'assertArrayEquals', 'assertSame', 'assertNotSame',
        'assertThrows', 'assertDoesNotThrow', 'assertIterableEquals',
        'assertLinesMatch', 'assertTimeout', 'assertTimeoutPreemptively'
    }
    
    WEAK_ASSERTIONS = {
        'assertTrue', 'assertFalse', 'assertNull', 'assertNotNull'
    }
    
    ANTI_PATTERNS = [
        (r'assertTrue\s*\(\s*(?:true|false)\s*\)', "Tautology assertion always passes or fails"),
        (r'assertFalse\s*\(\s*(?:true|false)\s*\)', "Tautology assertion always passes or fails"),
        (r'assertNull\s*\(\s*null\s*\)', "Tautology assertion always passes"),
        (r'assertNotNull\s*\(\s*new\s+\w+', "Asserting new object is not null is redundant"),
        (r'assertTrue\s*\([^)]*\.toString\(\)', "String comparison should use assertEquals"),
        (r'assertTrue\s*\([^)]*==\s*[^)]*\)', "Use assertEquals instead of assertTrue with =="),
        (r'assertFalse\s*\([^)]*==\s*[^)]*\)', "Use assertNotEquals instead of assertFalse with =="),
        (r'assertTrue\s*\([^)]*!=\s*[^)]*\)', "Use assertNotEquals instead of assertTrue with !="),
        (r'print(?:ln)?\s*\([^)]*\)', "Print statement in test - use assertions instead"),
        (r'fail\s*\(\s*\)', "Empty fail() - provide a message"),
    ]
    
    def analyze(self, test_methods: Dict[str, Dict[str, Any]]) -> QualityScore:
        issues = []
        total_assertions = 0
        strong_assertions = 0
        weak_assertions = 0
        
        for method_name, method_info in test_methods.items():
            if method_info.get('annotation') != 'Test':
                continue
            
            body = method_info.get('body', '')
            
            for pattern, message in self.ANTI_PATTERNS:
                matches = re.findall(pattern, body)
                for match in matches:
                    issues.append(QualityIssue(
                        dimension=QualityDimension.ASSERTION_QUALITY,
                        severity=Severity.HIGH,
                        message=f"Anti-pattern detected: {message}",
                        location=(method_info.get('start_line', 0), method_info.get('end_line', 0)),
                        suggestion=self._suggest_fix(pattern),
                        rule_id="ASSERT_ANTI_PATTERN"
                    ))
            
            for assertion in self.STRONG_ASSERTIONS:
                count = len(re.findall(rf'\b{assertion}\s*\(', body))
                total_assertions += count
                strong_assertions += count
            
            for assertion in self.WEAK_ASSERTIONS:
                count = len(re.findall(rf'\b{assertion}\s*\(', body))
                total_assertions += count
                weak_assertions += count
            
            if total_assertions == 0:
                issues.append(QualityIssue(
                    dimension=QualityDimension.ASSERTION_QUALITY,
                    severity=Severity.CRITICAL,
                    message=f"Test method '{method_name}' has no assertions",
                    location=(method_info.get('start_line', 0), method_info.get('end_line', 0)),
                    suggestion="Add meaningful assertions to verify the expected behavior",
                    rule_id="NO_ASSERTIONS"
                ))
        
        if total_assertions == 0:
            return QualityScore(
                dimension=QualityDimension.ASSERTION_QUALITY,
                score=0,
                issues=issues
            )
        
        quality_ratio = strong_assertions / total_assertions if total_assertions > 0 else 0
        
        penalty = len([i for i in issues if i.severity in [Severity.CRITICAL, Severity.HIGH]]) * 10
        score = max(0, quality_ratio * 100 - penalty)
        
        return QualityScore(
            dimension=QualityDimension.ASSERTION_QUALITY,
            score=score,
            issues=issues
        )
    
    def _suggest_fix(self, pattern: str) -> str:
        suggestions = {
            r'assertTrue\s*\(\s*(?:true|false)\s*\)': "Remove or fix the assertion",
            r'assertFalse\s*\(\s*(?:true|false)\s*\)': "Remove or fix the assertion",
            r'assertNull\s*\(\s*null\s*\)': "Remove this assertion",
            r'assertTrue\s*\([^)]*==\s*[^)]*\)': "Use assertEquals(expected, actual)",
            r'assertFalse\s*\([^)]*==\s*[^)]*\)': "Use assertNotEquals(expected, actual)",
            r'print(?:ln)?\s*\([^)]*\)': "Replace with assertions",
        }
        return suggestions.get(pattern, "Review and fix the assertion")


class TestIsolationAnalyzer:
    
    SHARED_STATE_PATTERNS = [
        (r'static\s+\w+\s+\w+\s*=', "Static mutable field detected - may cause test interdependency"),
        (r'@BeforeAll', "@BeforeAll suggests shared state - ensure tests are independent"),
        (r'@AfterAll', "@AfterAll suggests shared state - ensure tests are independent"),
    ]
    
    EXTERNAL_DEPENDENCY_PATTERNS = [
        (r'new\s+Socket\s*\(', "Direct socket creation - use mocks for network dependencies"),
        (r'new\s+ServerSocket\s*\(', "Direct server socket - use mocks for network dependencies"),
        (r'new\s+File\s*\(', "Direct file I/O - use in-memory filesystem or mocks"),
        (r'FileOutputStream|FileInputStream', "Direct file I/O - use in-memory alternatives"),
        (r'new\s+URL\s*\(', "Direct URL creation - use mocks for HTTP calls"),
        (r'Runtime\.getRuntime\(\)', "Direct runtime access - may affect other tests"),
        (r'System\.(?:in|out|err)', "Direct system stream access - use system rules or mocks"),
        (r'Thread\.sleep\s*\(', "Thread.sleep makes tests slow and flaky"),
        (r'Class\.forName\s*\(', "Dynamic class loading - may have side effects"),
    ]
    
    def analyze(self, source_code: str, test_methods: Dict[str, Dict[str, Any]]) -> QualityScore:
        issues = []
        isolation_score = 100.0
        
        for pattern, message in self.SHARED_STATE_PATTERNS:
            matches = re.findall(pattern, source_code)
            if matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.TEST_ISOLATION,
                    severity=Severity.MEDIUM,
                    message=message,
                    suggestion="Use instance fields and @BeforeEach for test isolation",
                    rule_id="SHARED_STATE"
                ))
                isolation_score -= 5
        
        for pattern, message in self.EXTERNAL_DEPENDENCY_PATTERNS:
            matches = re.findall(pattern, source_code)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.TEST_ISOLATION,
                    severity=Severity.HIGH,
                    message=message,
                    suggestion="Use dependency injection and mocks for external dependencies",
                    rule_id="EXTERNAL_DEPENDENCY"
                ))
                isolation_score -= 10
        
        for method_name, method_info in test_methods.items():
            if method_info.get('annotation') != 'Test':
                continue
            
            body = method_info.get('body', '')
            
            if re.search(r'\bthis\.\w+\s*=', body) and not re.search(r'@BeforeEach|@Before', source_code):
                pass
        
        return QualityScore(
            dimension=QualityDimension.TEST_ISOLATION,
            score=max(0, isolation_score),
            issues=issues
        )


class NamingConventionAnalyzer:
    
    TEST_METHOD_PATTERN = r'@Test\s*(?:\n\s*)?(?:public\s+)?(?:void\s+)?(\w+)\s*\('
    
    GOOD_TEST_NAME_PATTERNS = [
        r'^should[A-Z][a-zA-Z0-9]*$',
        r'^when[A-Z][a-zA-Z0-9]*Then[A-Z][a-zA-Z0-9]*$',
        r'^given[A-Z][a-zA-Z0-9]*When[A-Z][a-zA-Z0-9]*Then[A-Z][a-zA-Z0-9]*$',
        r'^test[A-Z][a-zA-Z0-9]*$',
    ]
    
    BAD_TEST_NAME_PATTERNS = [
        r'^test[a-z]$',  # test1, test2, etc.
        r'^testMethod\d*$',
        r'^test\d+$',
        r'^method\d*$',
        r'^test[A-Z]{1,2}$',  # testA, testAB
    ]
    
    def analyze(self, source_code: str, test_methods: Dict[str, Dict[str, Any]]) -> QualityScore:
        issues = []
        good_names = 0
        total_test_methods = 0
        
        for method_name, method_info in test_methods.items():
            if method_info.get('annotation') != 'Test':
                continue
            
            total_test_methods += 1
            
            is_good = any(re.match(p, method_name) for p in self.GOOD_TEST_NAME_PATTERNS)
            is_bad = any(re.match(p, method_name) for p in self.BAD_TEST_NAME_PATTERNS)
            
            if is_good:
                good_names += 1
            elif is_bad:
                issues.append(QualityIssue(
                    dimension=QualityDimension.NAMING_CONVENTION,
                    severity=Severity.HIGH,
                    message=f"Poor test method name: '{method_name}'",
                    location=(method_info.get('start_line', 0), method_info.get('end_line', 0)),
                    suggestion=self._suggest_better_name(method_name, method_info.get('body', '')),
                    rule_id="BAD_TEST_NAME"
                ))
            else:
                if not method_name.startswith(('should', 'when', 'given', 'test', 'verify', 'check')):
                    issues.append(QualityIssue(
                        dimension=QualityDimension.NAMING_CONVENTION,
                        severity=Severity.MEDIUM,
                        message=f"Non-standard test method name: '{method_name}'",
                        location=(method_info.get('start_line', 0), method_info.get('end_line', 0)),
                        suggestion="Consider using 'should[Behavior]' or 'when[Condition]Then[Outcome]' naming",
                        rule_id="NON_STANDARD_NAME"
                    ))
        
        if total_test_methods == 0:
            return QualityScore(
                dimension=QualityDimension.NAMING_CONVENTION,
                score=100,
                issues=issues
            )
        
        score = (good_names / total_test_methods) * 100
        
        return QualityScore(
            dimension=QualityDimension.NAMING_CONVENTION,
            score=score,
            issues=issues
        )
    
    def _suggest_better_name(self, current_name: str, body: str) -> str:
        assertions = re.findall(r'assert(?:Equals|True|False)\s*\([^,]+,\s*(\w+)', body)
        conditions = re.findall(r'when\s*\(\s*(\w+)', body, re.IGNORECASE)
        
        if conditions and assertions:
            return f"should{self._capitalize(assertions[0])}When{self._capitalize(conditions[0])}"
        elif assertions:
            return f"shouldReturn{self._capitalize(assertions[0])}"
        elif conditions:
            return f"shouldHandle{self._capitalize(conditions[0])}"
        
        return f"should{self._capitalize(current_name.replace('test', '').replace('method', ''))}"
    
    def _capitalize(self, s: str) -> str:
        return s[0].upper() + s[1:] if s else ""


class MaintainabilityAnalyzer:
    
    COMPLEXITY_PATTERNS = [
        (r'\bif\s*\(', 1),
        (r'\belse\s+if\s*\(', 1),
        (r'\bfor\s*\(', 2),
        (r'\bwhile\s*\(', 2),
        (r'\bswitch\s*\(', 2),
        (r'\bcase\s+', 1),
        (r'\bcatch\s*\(', 1),
        (r'\?\s*:', 1),
        (r'&&', 1),
        (r'\|\|', 1),
    ]
    
    MAGIC_NUMBER_PATTERN = r'(?<!["\w])(?:\d{2,}|\d+\.\d+)(?!["\w])'
    
    HARDCODED_VALUES = [
        (r'".{50,}"', "Long hardcoded string - consider using a constant or test data builder"),
        (r'\b\d{4,}\b', "Large magic number - consider using a named constant"),
    ]
    
    def analyze(self, source_code: str, test_methods: Dict[str, Dict[str, Any]]) -> QualityScore:
        issues = []
        total_complexity = 0
        method_count = 0
        
        for method_name, method_info in test_methods.items():
            if method_info.get('annotation') != 'Test':
                continue
            
            method_count += 1
            body = method_info.get('body', '')
            
            complexity = 1
            for pattern, weight in self.COMPLEXITY_PATTERNS:
                complexity += len(re.findall(pattern, body)) * weight
            
            total_complexity += complexity
            
            if complexity > 10:
                issues.append(QualityIssue(
                    dimension=QualityDimension.MAINTAINABILITY,
                    severity=Severity.HIGH,
                    message=f"High complexity ({complexity}) in test '{method_name}'",
                    location=(method_info.get('start_line', 0), method_info.get('end_line', 0)),
                    suggestion="Consider extracting helper methods to reduce complexity",
                    rule_id="HIGH_COMPLEXITY"
                ))
            elif complexity > 5:
                issues.append(QualityIssue(
                    dimension=QualityDimension.MAINTAINABILITY,
                    severity=Severity.MEDIUM,
                    message=f"Moderate complexity ({complexity}) in test '{method_name}'",
                    location=(method_info.get('start_line', 0), method_info.get('end_line', 0)),
                    suggestion="Consider simplifying the test logic",
                    rule_id="MODERATE_COMPLEXITY"
                ))
            
            lines = body.split('\n')
            if len(lines) > 30:
                issues.append(QualityIssue(
                    dimension=QualityDimension.MAINTAINABILITY,
                    severity=Severity.MEDIUM,
                    message=f"Test method '{method_name}' is too long ({len(lines)} lines)",
                    location=(method_info.get('start_line', 0), method_info.get('end_line', 0)),
                    suggestion="Consider splitting into multiple focused tests",
                    rule_id="LONG_METHOD"
                ))
        
        for pattern, message in self.HARDCODED_VALUES:
            matches = re.findall(pattern, source_code)
            if len(matches) > 3:
                issues.append(QualityIssue(
                    dimension=QualityDimension.MAINTAINABILITY,
                    severity=Severity.LOW,
                    message=message,
                    suggestion="Extract values to constants or use test data builders",
                    rule_id="HARDCODED_VALUES"
                ))
        
        avg_complexity = total_complexity / method_count if method_count > 0 else 1
        complexity_score = max(0, 100 - (avg_complexity - 1) * 10)
        
        return QualityScore(
            dimension=QualityDimension.MAINTAINABILITY,
            score=complexity_score,
            issues=issues
        )


class ReadabilityAnalyzer:
    
    COMMENT_PATTERNS = [
        r'//\s*TODO',
        r'//\s*FIXME',
        r'//\s*HACK',
        r'//\s*XXX',
    ]
    
    def analyze(self, source_code: str, test_methods: Dict[str, Dict[str, Any]]) -> QualityScore:
        issues = []
        readability_score = 100.0
        
        lines = source_code.split('\n')
        total_lines = len(lines)
        
        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                comment_lines += 1
        
        comment_ratio = comment_lines / total_lines if total_lines > 0 else 0
        
        if comment_ratio < 0.05:
            issues.append(QualityIssue(
                dimension=QualityDimension.READABILITY,
                severity=Severity.LOW,
                message="Low comment ratio - consider adding documentation",
                suggestion="Add comments explaining test intent and setup",
                rule_id="LOW_COMMENTS"
            ))
            readability_score -= 5
        
        for pattern in self.COMMENT_PATTERNS:
            matches = re.findall(pattern, source_code, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.READABILITY,
                    severity=Severity.INFO,
                    message=f"Found TODO/FIXME comment: {match}",
                    suggestion="Address or remove the TODO/FIXME comment",
                    rule_id="TODO_COMMENT"
                ))
        
        magic_strings = re.findall(r'assert(?:Equals|True|False)\s*\(\s*"[^"]{30,}"', source_code)
        if magic_strings:
            issues.append(QualityIssue(
                dimension=QualityDimension.READABILITY,
                severity=Severity.LOW,
                message="Long string literals in assertions",
                suggestion="Consider extracting to constants for better readability",
                rule_id="LONG_STRING_LITERAL"
            ))
        
        return QualityScore(
            dimension=QualityDimension.READABILITY,
            score=max(0, readability_score),
            issues=issues
        )


class ReliabilityAnalyzer:
    
    FLAKY_PATTERNS = [
        (r'Thread\.sleep\s*\(', "Thread.sleep can cause flaky tests", Severity.HIGH),
        (r'System\.currentTimeMillis\(\)', "Time-dependent code can cause flaky tests", Severity.MEDIUM),
        (r'new\s+Random\s*\(', "Random values can cause non-deterministic tests", Severity.MEDIUM),
        (r'Math\.random\(\)', "Random values can cause non-deterministic tests", Severity.MEDIUM),
        (r'new\s+Date\s*\(\)', "Current date/time can cause flaky tests", Severity.MEDIUM),
        (r'LocalDateTime\.now\(\)', "Current time can cause flaky tests", Severity.MEDIUM),
        (r'InetAddress\.getLocalHost\(\)', "Network-dependent code can cause flaky tests", Severity.HIGH),
        (r'new\s+ServerSocket\s*\(\d+\)', "Fixed port can cause port conflicts", Severity.HIGH),
    ]
    
    ERROR_PRONE_PATTERNS = [
        (r'catch\s*\(\s*Exception\s+\w+\s*\)\s*\{?\s*\}', "Empty catch block swallows exceptions", Severity.HIGH),
        (r'catch\s*\(\s*Throwable', "Catching Throwable is too broad", Severity.MEDIUM),
        (r'\.printStackTrace\(\)', "printStackTrace is not proper error handling", Severity.MEDIUM),
        (r'suppress\s*=\s*true', "Suppressed warnings may hide issues", Severity.LOW),
        (r'@Ignore\s*(?:\n|$)', "Ignored test should have explanation", Severity.LOW),
        (r'@Disabled\s*(?:\n|$)', "Disabled test should have explanation", Severity.LOW),
    ]
    
    def analyze(self, source_code: str, test_methods: Dict[str, Dict[str, Any]]) -> QualityScore:
        issues = []
        reliability_score = 100.0
        
        for pattern, message, severity in self.FLAKY_PATTERNS:
            matches = re.findall(pattern, source_code)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.RELIABILITY,
                    severity=severity,
                    message=message,
                    suggestion="Use deterministic alternatives or proper test doubles",
                    rule_id="FLAKY_PATTERN"
                ))
                reliability_score -= severity.value * 3
        
        for pattern, message, severity in self.ERROR_PRONE_PATTERNS:
            matches = re.findall(pattern, source_code)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.RELIABILITY,
                    severity=severity,
                    message=message,
                    suggestion="Handle errors properly or document why they're ignored",
                    rule_id="ERROR_PRONE"
                ))
                reliability_score -= severity.value * 2
        
        return QualityScore(
            dimension=QualityDimension.RELIABILITY,
            score=max(0, reliability_score),
            issues=issues
        )


class TestQualityAnalyzer:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.assertion_analyzer = AssertionQualityAnalyzer()
        self.isolation_analyzer = TestIsolationAnalyzer()
        self.naming_analyzer = NamingConventionAnalyzer()
        self.maintainability_analyzer = MaintainabilityAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.reliability_analyzer = ReliabilityAnalyzer()
        
        self._analysis_history: List[TestQualityReport] = []
    
    def analyze(self, source_code: str) -> TestQualityReport:
        test_methods = self._extract_test_methods(source_code)
        
        assertion_score = self.assertion_analyzer.analyze(test_methods)
        isolation_score = self.isolation_analyzer.analyze(source_code, test_methods)
        naming_score = self.naming_analyzer.analyze(source_code, test_methods)
        maintainability_score = self.maintainability_analyzer.analyze(source_code, test_methods)
        readability_score = self.readability_analyzer.analyze(source_code, test_methods)
        reliability_score = self.reliability_analyzer.analyze(source_code, test_methods)
        
        dimension_scores = {
            QualityDimension.ASSERTION_QUALITY.value: assertion_score,
            QualityDimension.TEST_ISOLATION.value: isolation_score,
            QualityDimension.NAMING_CONVENTION.value: naming_score,
            QualityDimension.MAINTAINABILITY.value: maintainability_score,
            QualityDimension.READABILITY.value: readability_score,
            QualityDimension.RELIABILITY.value: reliability_score,
        }
        
        weights = {
            QualityDimension.ASSERTION_QUALITY.value: 0.25,
            QualityDimension.TEST_ISOLATION.value: 0.20,
            QualityDimension.NAMING_CONVENTION.value: 0.10,
            QualityDimension.MAINTAINABILITY.value: 0.20,
            QualityDimension.READABILITY.value: 0.10,
            QualityDimension.RELIABILITY.value: 0.15,
        }
        
        overall_score = sum(
            dimension_scores[dim].score * weights[dim]
            for dim in weights
        )
        
        all_issues = []
        for score in dimension_scores.values():
            all_issues.extend(score.issues)
        
        critical_issues = len([i for i in all_issues if i.severity == Severity.CRITICAL])
        
        improvement_suggestions = self._generate_improvement_suggestions(dimension_scores)
        
        report = TestQualityReport(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            total_issues=len(all_issues),
            critical_issues=critical_issues,
            improvement_suggestions=improvement_suggestions,
            test_methods_analyzed=len([m for m in test_methods.values() if m.get('annotation') == 'Test']),
            lines_of_code=len(source_code.split('\n'))
        )
        
        self._analysis_history.append(report)
        
        return report
    
    def _extract_test_methods(self, source_code: str) -> Dict[str, Dict[str, Any]]:
        methods = {}
        
        method_pattern = r'@(Test|Before|After|BeforeEach|AfterEach|BeforeClass|AfterClass)\s*(?:\n\s*)?(?:public\s+)?(?:static\s+)?(?:void\s+)?(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
        
        for match in re.finditer(method_pattern, source_code):
            annotation = match.group(1)
            method_name = match.group(2)
            start_pos = match.start()
            
            brace_count = 0
            method_start = match.end()
            method_end = method_start
            
            for i in range(method_start, len(source_code)):
                if source_code[i] == '{':
                    brace_count += 1
                elif source_code[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        method_end = i + 1
                        break
            
            method_body = source_code[method_start:method_end]
            
            methods[method_name] = {
                'annotation': annotation,
                'body': method_body,
                'start': start_pos,
                'end': method_end,
                'start_line': source_code[:start_pos].count('\n') + 1,
                'end_line': source_code[:method_end].count('\n') + 1
            }
        
        return methods
    
    def _generate_improvement_suggestions(
        self,
        dimension_scores: Dict[str, QualityScore]
    ) -> List[str]:
        suggestions = []
        
        sorted_dimensions = sorted(
            dimension_scores.items(),
            key=lambda x: x[1].score
        )
        
        for dim_name, score in sorted_dimensions:
            if score.score < 60:
                critical_issues = [
                    i for i in score.issues
                    if i.severity in [Severity.CRITICAL, Severity.HIGH]
                ]
                for issue in critical_issues[:3]:
                    if issue.suggestion:
                        suggestions.append(f"[{dim_name}] {issue.suggestion}")
        
        if not suggestions:
            for dim_name, score in sorted_dimensions[:2]:
                if score.score < 80:
                    suggestions.append(
                        f"Improve {dim_name} score (current: {score.score:.1f}%)"
                    )
        
        return suggestions[:10]
    
    def compare_reports(
        self,
        before: TestQualityReport,
        after: TestQualityReport
    ) -> Dict[str, Any]:
        comparison = {
            'overall_score_change': after.overall_score - before.overall_score,
            'dimension_changes': {},
            'issue_changes': {
                'total': after.total_issues - before.total_issues,
                'critical': after.critical_issues - before.critical_issues
            },
            'improved': [],
            'regressed': []
        }
        
        for dim in before.dimension_scores:
            before_score = before.dimension_scores[dim].score
            after_score = after.dimension_scores[dim].score
            change = after_score - before_score
            
            comparison['dimension_changes'][dim] = {
                'before': before_score,
                'after': after_score,
                'change': change
            }
            
            if change > 5:
                comparison['improved'].append(dim)
            elif change < -5:
                comparison['regressed'].append(dim)
        
        return comparison
    
    def get_quality_trend(self, last_n: int = 10) -> Dict[str, Any]:
        if len(self._analysis_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent_reports = self._analysis_history[-last_n:]
        
        scores = [r.overall_score for r in recent_reports]
        
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            earlier_avg = sum(scores[:3]) / 3 if len(scores) > 3 else scores[0]
            
            if recent_avg > earlier_avg + 5:
                trend = "improving"
            elif recent_avg < earlier_avg - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        dimension_trends = {}
        for dim in recent_reports[0].dimension_scores:
            dim_scores = [r.dimension_scores[dim].score for r in recent_reports]
            dimension_trends[dim] = {
                'current': dim_scores[-1],
                'average': sum(dim_scores) / len(dim_scores),
                'min': min(dim_scores),
                'max': max(dim_scores)
            }
        
        return {
            "trend": trend,
            "overall_scores": scores,
            "dimension_trends": dimension_trends,
            "total_analyses": len(self._analysis_history)
        }
    
    def generate_report_markdown(self, report: TestQualityReport) -> str:
        lines = [
            "# Test Quality Report",
            "",
            f"**Overall Score:** {report.overall_score:.1f}/100 ({self._get_grade(report.overall_score)})",
            f"**Test Methods Analyzed:** {report.test_methods_analyzed}",
            f"**Lines of Code:** {report.lines_of_code}",
            f"**Total Issues:** {report.total_issues}",
            f"**Critical Issues:** {report.critical_issues}",
            "",
            "## Dimension Scores",
            "",
        ]
        
        for dim_name, score in sorted(
            report.dimension_scores.items(),
            key=lambda x: x[1].score
        ):
            lines.append(f"### {dim_name.replace('_', ' ').title()}")
            lines.append(f"- Score: {score.score:.1f}/100 ({score.grade})")
            lines.append(f"- Issues: {len(score.issues)}")
            
            if score.issues:
                lines.append("")
                lines.append("Issues:")
                for issue in score.issues[:5]:
                    lines.append(f"  - [{issue.severity.value}] {issue.message}")
                    if issue.suggestion:
                        lines.append(f"    - Suggestion: {issue.suggestion}")
            lines.append("")
        
        if report.improvement_suggestions:
            lines.extend([
                "## Top Improvement Suggestions",
                "",
            ])
            for i, suggestion in enumerate(report.improvement_suggestions, 1):
                lines.append(f"{i}. {suggestion}")
        
        return '\n'.join(lines)
    
    def _get_grade(self, score: float) -> str:
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
