"""Code generation quality evaluation for pre-validation of generated tests.

This module provides comprehensive evaluation of generated test code
before compilation to identify and fix issues early in the pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Dimensions of code quality evaluation."""
    SYNTAX = auto()         # Syntax correctness
    COMPLETENESS = auto()   # Code completeness
    STYLE = auto()          # Code style conformance
    COVERAGE_POTENTIAL = auto()  # Potential to cover target code
    MOCK_USAGE = auto()     # Proper mock usage
    ASSERTION_QUALITY = auto()  # Quality of assertions


class IssueSeverity(Enum):
    """Severity levels for detected issues."""
    CRITICAL = "critical"    # Will definitely cause failure
    HIGH = "high"           # Likely to cause failure
    MEDIUM = "medium"       # May cause issues
    LOW = "low"             # Minor issue
    INFO = "info"           # Informational


@dataclass
class QualityIssue:
    """Represents a quality issue in generated code."""
    dimension: QualityDimension
    severity: IssueSeverity
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class CoverageEstimate:
    """Estimated coverage potential."""
    line_coverage_potential: float  # 0.0 - 1.0
    branch_coverage_potential: float
    method_coverage_potential: float
    uncovered_methods: List[str] = field(default_factory=list)
    uncovered_branches: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class EvaluationResult:
    """Result of code generation evaluation."""
    overall_score: float  # 0.0 - 1.0
    is_acceptable: bool
    issues: List[QualityIssue] = field(default_factory=list)
    coverage_estimate: Optional[CoverageEstimate] = None
    dimension_scores: Dict[QualityDimension, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def get_critical_issues(self) -> List[QualityIssue]:
        """Get critical issues that must be fixed."""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]
    
    def get_issues_by_dimension(self, dimension: QualityDimension) -> List[QualityIssue]:
        """Get issues for a specific dimension."""
        return [i for i in self.issues if i.dimension == dimension]


class GenerationEvaluator:
    """Evaluates quality of generated test code.
    
    Provides pre-compilation validation to catch issues early
    and reduce iteration cycles.
    """
    
    # Quality thresholds
    MIN_ACCEPTABLE_SCORE = 0.6
    GOOD_SCORE = 0.8
    EXCELLENT_SCORE = 0.9
    
    def __init__(self):
        """Initialize the evaluator."""
        self.issues: List[QualityIssue] = []
        logger.info("[GenerationEvaluator] Initialized")
    
    def evaluate(
        self,
        test_code: str,
        target_class_info: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Comprehensive evaluation of generated test code.
        
        Args:
            test_code: Generated test code
            target_class_info: Information about target class being tested
            
        Returns:
            Evaluation result with scores and issues
        """
        self.issues = []
        dimension_scores = {}
        
        logger.info("[GenerationEvaluator] Starting evaluation")
        
        # Evaluate each dimension
        dimension_scores[QualityDimension.SYNTAX] = self._evaluate_syntax(test_code)
        dimension_scores[QualityDimension.COMPLETENESS] = self._evaluate_completeness(test_code)
        dimension_scores[QualityDimension.STYLE] = self._evaluate_style(test_code)
        dimension_scores[QualityDimension.MOCK_USAGE] = self._evaluate_mock_usage(test_code)
        dimension_scores[QualityDimension.ASSERTION_QUALITY] = self._evaluate_assertions(test_code)
        
        # Estimate coverage potential
        coverage_estimate = None
        if target_class_info:
            coverage_estimate = self._estimate_coverage_potential(
                test_code, target_class_info
            )
            dimension_scores[QualityDimension.COVERAGE_POTENTIAL] = (
                coverage_estimate.line_coverage_potential
            )
        
        # Calculate overall score (weighted average)
        weights = {
            QualityDimension.SYNTAX: 0.25,
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.STYLE: 0.10,
            QualityDimension.COVERAGE_POTENTIAL: 0.20,
            QualityDimension.MOCK_USAGE: 0.15,
            QualityDimension.ASSERTION_QUALITY: 0.10
        }
        
        overall_score = sum(
            dimension_scores.get(dim, 0.0) * weight
            for dim, weight in weights.items()
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(dimension_scores)
        
        # Determine if acceptable
        is_acceptable = (
            overall_score >= self.MIN_ACCEPTABLE_SCORE and
            dimension_scores.get(QualityDimension.SYNTAX, 0) >= 0.5 and
            not any(i.severity == IssueSeverity.CRITICAL for i in self.issues)
        )
        
        result = EvaluationResult(
            overall_score=overall_score,
            is_acceptable=is_acceptable,
            issues=self.issues,
            coverage_estimate=coverage_estimate,
            dimension_scores=dimension_scores,
            recommendations=recommendations
        )
        
        logger.info(f"[GenerationEvaluator] Evaluation complete - Score: {overall_score:.2f}, "
                   f"Acceptable: {is_acceptable}, Issues: {len(self.issues)}")
        
        return result
    
    def quick_check(self, test_code: str) -> Tuple[bool, List[str]]:
        """Quick syntax and completeness check.
        
        Args:
            test_code: Generated test code
            
        Returns:
            Tuple of (is_valid, list of critical issues)
        """
        self.issues = []
        
        self._evaluate_syntax(test_code)
        self._evaluate_completeness(test_code)
        
        critical_issues = [
            i.message for i in self.issues 
            if i.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)
        ]
        
        is_valid = len(critical_issues) == 0
        
        return is_valid, critical_issues
    
    def _evaluate_syntax(self, test_code: str) -> float:
        """Evaluate syntax correctness.
        
        Args:
            test_code: Test code to evaluate
            
        Returns:
            Score from 0.0 to 1.0
        """
        score = 1.0
        
        # Check for balanced braces
        open_braces = test_code.count('{')
        close_braces = test_code.count('}')
        if open_braces != close_braces:
            diff = abs(open_braces - close_braces)
            self.issues.append(QualityIssue(
                dimension=QualityDimension.SYNTAX,
                severity=IssueSeverity.CRITICAL,
                message=f"Unbalanced braces: {open_braces} open, {close_braces} close",
                suggestion="Ensure all opening braces have matching closing braces",
                auto_fixable=False
            ))
            score -= min(0.3, diff * 0.1)
        
        # Check for balanced parentheses
        open_parens = test_code.count('(')
        close_parens = test_code.count(')')
        if open_parens != close_parens:
            diff = abs(open_parens - close_parens)
            self.issues.append(QualityIssue(
                dimension=QualityDimension.SYNTAX,
                severity=IssueSeverity.CRITICAL,
                message=f"Unbalanced parentheses: {open_parens} open, {close_parens} close",
                suggestion="Ensure all opening parentheses have matching closing ones",
                auto_fixable=False
            ))
            score -= min(0.3, diff * 0.1)
        
        # Check for basic Java structure
        if 'class' not in test_code:
            self.issues.append(QualityIssue(
                dimension=QualityDimension.SYNTAX,
                severity=IssueSeverity.CRITICAL,
                message="Missing class declaration",
                suggestion="Add a class declaration for the test class"
            ))
            score -= 0.5
        
        # Check for semicolons in appropriate places
        lines = test_code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip lines that don't need semicolons
            if any(stripped.startswith(x) for x in ['//', '/*', '*', 'package', 'import', '@']):
                continue
            if stripped.endswith('{') or stripped.endswith('}') or stripped.endswith(')'):
                continue
            if not stripped or stripped.startswith('//'):
                continue
            
            # Check for missing semicolons on statements
            if (re.match(r'^\s*(\w+\s+)?\w+\s*=', stripped) or  # Assignment
                re.match(r'^\s*\w+\.', stripped) or  # Method call
                re.match(r'^\s*(return|throw)\s+', stripped)):  # Return/throw
                if not stripped.rstrip().endswith(';'):
                    self.issues.append(QualityIssue(
                        dimension=QualityDimension.SYNTAX,
                        severity=IssueSeverity.HIGH,
                        message=f"Possible missing semicolon at line {i}",
                        line_number=i,
                        suggestion="Add semicolon at end of statement"
                    ))
                    score -= 0.05
        
        # Check for common syntax errors
        if 'void void' in test_code or 'public public' in test_code:
            self.issues.append(QualityIssue(
                dimension=QualityDimension.SYNTAX,
                severity=IssueSeverity.CRITICAL,
                message="Duplicate keywords detected",
                suggestion="Remove duplicate keywords"
            ))
            score -= 0.2
        
        return max(0.0, score)
    
    def _evaluate_completeness(self, test_code: str) -> float:
        """Evaluate code completeness.
        
        Args:
            test_code: Test code to evaluate
            
        Returns:
            Score from 0.0 to 1.0
        """
        score = 1.0
        
        # Check for required JUnit annotations
        if '@Test' not in test_code:
            self.issues.append(QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=IssueSeverity.HIGH,
                message="No @Test annotations found",
                suggestion="Add @Test annotation to test methods"
            ))
            score -= 0.3
        
        # Check for class declaration completeness
        class_pattern = re.compile(r'public\s+class\s+(\w+)')
        match = class_pattern.search(test_code)
        if match:
            class_name = match.group(1)
            # Check if class name follows convention (ends with Test)
            if not class_name.endswith('Test'):
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=IssueSeverity.LOW,
                    message=f"Class name '{class_name}' doesn't follow Test suffix convention",
                    suggestion=f"Consider renaming to {class_name}Test"
                ))
                score -= 0.05
        
        # Check for proper imports
        required_imports = ['org.junit.jupiter.api.Test']
        for imp in required_imports:
            if imp not in test_code:
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=IssueSeverity.HIGH,
                    message=f"Missing import: {imp}",
                    suggestion=f"Add import {imp};",
                    auto_fixable=True
                ))
                score -= 0.1
        
        # Check for test methods
        test_methods = re.findall(r'@Test[\s\S]*?void\s+(\w+)\s*\(', test_code)
        if not test_methods:
            self.issues.append(QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=IssueSeverity.CRITICAL,
                message="No test methods found",
                suggestion="Add test methods with @Test annotation"
            ))
            score -= 0.5
        
        # Check for assertions
        assertion_patterns = [
            r'assertEquals\s*\(',
            r'assertTrue\s*\(',
            r'assertFalse\s*\(',
            r'assertNull\s*\(',
            r'assertNotNull\s*\(',
            r'assertThrows\s*\(',
            r'Assertions\.assert',
        ]
        has_assertion = any(re.search(p, test_code) for p in assertion_patterns)
        if not has_assertion:
            self.issues.append(QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=IssueSeverity.HIGH,
                message="No assertions found in test code",
                suggestion="Add assertions to verify test outcomes"
            ))
            score -= 0.2
        
        return max(0.0, score)
    
    def _evaluate_style(self, test_code: str) -> float:
        """Evaluate code style conformance.
        
        Args:
            test_code: Test code to evaluate
            
        Returns:
            Score from 0.0 to 1.0
        """
        score = 1.0
        
        # Check method naming convention (camelCase)
        method_pattern = re.compile(r'void\s+(\w+)\s*\(')
        for match in method_pattern.finditer(test_code):
            method_name = match.group(1)
            if method_name == 'main':
                continue
            # Skip if it's a test method with descriptive name
            if '_' in method_name and not method_name.startswith('test'):
                # snake_case might be intentional for readability
                continue
            if not method_name[0].islower():
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.STYLE,
                    severity=IssueSeverity.LOW,
                    message=f"Method '{method_name}' doesn't follow camelCase convention",
                    suggestion=f"Consider renaming to {method_name[0].lower() + method_name[1:]}"
                ))
                score -= 0.02
        
        # Check indentation consistency
        lines = test_code.split('\n')
        indent_sizes = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indent_sizes.append(indent)
        
        if indent_sizes:
            # Check if indentation is consistent (multiples of 2 or 4)
            common_sizes = set()
            for size in indent_sizes:
                if size % 4 == 0:
                    common_sizes.add(4)
                elif size % 2 == 0:
                    common_sizes.add(2)
            
            if len(common_sizes) > 1:
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.STYLE,
                    severity=IssueSeverity.LOW,
                    message="Inconsistent indentation detected",
                    suggestion="Use consistent indentation (2 or 4 spaces)"
                ))
                score -= 0.05
        
        # Check for proper spacing around operators
        # This is a simplified check
        if re.search(r'\w=\w', test_code):  # Missing spaces around =
            self.issues.append(QualityIssue(
                dimension=QualityDimension.STYLE,
                severity=IssueSeverity.INFO,
                message="Consider adding spaces around assignment operators",
                suggestion="Use 'a = b' instead of 'a=b'"
            ))
            score -= 0.02
        
        return max(0.0, score)
    
    def _evaluate_mock_usage(self, test_code: str) -> float:
        """Evaluate proper mock usage.
        
        Args:
            test_code: Test code to evaluate
            
        Returns:
            Score from 0.0 to 1.0
        """
        score = 1.0
        
        # Check for Mockito imports if mocks are used
        if 'mock(' in test_code or '@Mock' in test_code or 'Mockito.' in test_code:
            if 'org.mockito' not in test_code:
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.MOCK_USAGE,
                    severity=IssueSeverity.HIGH,
                    message="Mockito usage detected but no Mockito import found",
                    suggestion="Add import org.mockito.Mockito;",
                    auto_fixable=True
                ))
                score -= 0.2
            
            # Check for proper mock setup
            if '@Mock' in test_code and '@ExtendWith' not in test_code:
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.MOCK_USAGE,
                    severity=IssueSeverity.MEDIUM,
                    message="@Mock annotations require @ExtendWith(MockitoExtension.class)",
                    suggestion="Add @ExtendWith(MockitoExtension.class) to the test class"
                ))
                score -= 0.1
            
            # Check for verify() usage without when()
            if 'verify(' in test_code and 'when(' not in test_code:
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.MOCK_USAGE,
                    severity=IssueSeverity.INFO,
                    message="Mock verification found but no stubbing detected",
                    suggestion="Consider if you need to stub mock behavior with when()"
                ))
                score -= 0.02
        
        return max(0.0, score)
    
    def _evaluate_assertions(self, test_code: str) -> float:
        """Evaluate assertion quality.
        
        Args:
            test_code: Test code to evaluate
            
        Returns:
            Score from 0.0 to 1.0
        """
        score = 1.0
        
        # Count assertions per test method
        test_method_pattern = re.compile(
            r'@Test[\s\S]*?void\s+(\w+)\s*\([^)]*\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}',
            re.MULTILINE
        )
        
        for match in test_method_pattern.finditer(test_code):
            method_name = match.group(1)
            method_body = match.group(2)
            
            # Count assertions in this method
            assertion_count = len(re.findall(r'assert\w+\s*\(', method_body))
            
            if assertion_count == 0:
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.ASSERTION_QUALITY,
                    severity=IssueSeverity.HIGH,
                    message=f"Test method '{method_name}' has no assertions",
                    suggestion="Add at least one assertion to verify the test outcome"
                ))
                score -= 0.1
            elif assertion_count > 5:
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.ASSERTION_QUALITY,
                    severity=IssueSeverity.LOW,
                    message=f"Test method '{method_name}' has many assertions ({assertion_count})",
                    suggestion="Consider splitting into multiple focused test methods"
                ))
                score -= 0.02
        
        # Check for meaningful assertions (not just assertTrue(true))
        trivial_patterns = [
            r'assertTrue\s*\(\s*true\s*\)',
            r'assertEquals\s*\(\s*\w+\s*,\s*\w+\s*\)'  # Same variable compared
        ]
        for pattern in trivial_patterns:
            if re.search(pattern, test_code):
                self.issues.append(QualityIssue(
                    dimension=QualityDimension.ASSERTION_QUALITY,
                    severity=IssueSeverity.MEDIUM,
                    message="Trivial assertion detected",
                    suggestion="Replace with meaningful assertions that verify actual behavior"
                ))
                score -= 0.05
        
        return max(0.0, score)
    
    def _estimate_coverage_potential(
        self,
        test_code: str,
        target_class_info: Dict[str, Any]
    ) -> CoverageEstimate:
        """Estimate potential coverage of generated tests.
        
        Args:
            test_code: Generated test code
            target_class_info: Information about target class
            
        Returns:
            Coverage estimate
        """
        target_methods = target_class_info.get('methods', [])
        target_method_names = {m.get('name') for m in target_methods}
        
        # Find which methods are tested
        tested_methods = set()
        test_method_pattern = re.compile(r'void\s+(\w+)\s*\(')
        
        for match in test_method_pattern.finditer(test_code):
            test_method_name = match.group(1)
            # Check if test method name references a target method
            for target_method in target_method_names:
                if target_method and target_method.lower() in test_method_name.lower():
                    tested_methods.add(target_method)
                    break
        
        # Calculate method coverage potential
        if target_method_names:
            method_coverage = len(tested_methods) / len(target_method_names)
        else:
            method_coverage = 0.0
        
        # Estimate line coverage (rough heuristic)
        # More test methods generally mean better line coverage
        test_count = len(re.findall(r'@Test', test_code))
        target_method_count = len(target_methods)
        
        if target_method_count > 0:
            line_coverage = min(1.0, (test_count / target_method_count) * 0.8)
        else:
            line_coverage = 0.0
        
        # Estimate branch coverage (lower than line coverage typically)
        branch_coverage = line_coverage * 0.7
        
        # Find uncovered methods
        uncovered_methods = list(target_method_names - tested_methods)
        
        # Generate reasoning
        reasoning_parts = [
            f"Testing {len(tested_methods)} out of {len(target_method_names)} methods",
            f"Generated {test_count} test methods"
        ]
        
        if uncovered_methods:
            reasoning_parts.append(f"Uncovered methods: {', '.join(uncovered_methods[:5])}")
        
        return CoverageEstimate(
            line_coverage_potential=line_coverage,
            branch_coverage_potential=branch_coverage,
            method_coverage_potential=method_coverage,
            uncovered_methods=uncovered_methods[:10],
            reasoning="; ".join(reasoning_parts)
        )
    
    def _generate_recommendations(
        self,
        dimension_scores: Dict[QualityDimension, float]
    ) -> List[str]:
        """Generate improvement recommendations.
        
        Args:
            dimension_scores: Scores for each dimension
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Identify low-scoring dimensions
        for dimension, score in dimension_scores.items():
            if score < 0.5:
                if dimension == QualityDimension.SYNTAX:
                    recommendations.append("Fix syntax errors before proceeding")
                elif dimension == QualityDimension.COMPLETENESS:
                    recommendations.append("Add missing required elements (imports, annotations, assertions)")
                elif dimension == QualityDimension.COVERAGE_POTENTIAL:
                    recommendations.append("Add more test methods to cover untested target methods")
                elif dimension == QualityDimension.MOCK_USAGE:
                    recommendations.append("Review mock setup and add missing Mockito configuration")
                elif dimension == QualityDimension.ASSERTION_QUALITY:
                    recommendations.append("Improve assertions to better verify test outcomes")
        
        # Add specific issue-based recommendations
        critical_count = sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)
        if critical_count > 0:
            recommendations.append(f"Address {critical_count} critical issues immediately")
        
        if not recommendations:
            recommendations.append("Code quality is good - proceed with compilation")
        
        return recommendations


# Convenience functions

def evaluate_test_code(
    test_code: str,
    target_class_info: Optional[Dict[str, Any]] = None
) -> EvaluationResult:
    """Quick evaluation of test code.
    
    Args:
        test_code: Test code to evaluate
        target_class_info: Target class information
        
    Returns:
        Evaluation result
    """
    evaluator = GenerationEvaluator()
    return evaluator.evaluate(test_code, target_class_info)


def is_code_compilable(test_code: str) -> Tuple[bool, List[str]]:
    """Quick check if code is likely compilable.
    
    Args:
        test_code: Test code to check
        
        Returns:
        Tuple of (is_likely_compilable, list of issues)
    """
    evaluator = GenerationEvaluator()
    is_valid, issues = evaluator.quick_check(test_code)
    return is_valid, issues
