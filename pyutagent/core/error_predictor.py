"""Error predictor for pre-compilation error detection.

This module provides error prediction capabilities:
- Static code analysis for error patterns
- Dynamic prediction based on historical data
- Test failure prediction
- Confidence scoring
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path

from .error_knowledge_base import ErrorKnowledgeBase

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can be predicted."""
    SYNTAX_ERROR = auto()
    TYPE_MISMATCH = auto()
    NULL_POINTER = auto()
    IMPORT_MISSING = auto()
    METHOD_NOT_FOUND = auto()
    VARIABLE_NOT_FOUND = auto()
    ACCESS_VIOLATION = auto()
    RESOURCE_LEAK = auto()
    CONCURRENCY_ISSUE = auto()
    TEST_ASSERTION_FAILURE = auto()
    TEST_SETUP_FAILURE = auto()


class Severity(Enum):
    """Severity levels for predicted errors."""
    CRITICAL = auto()  # Will definitely cause failure
    HIGH = auto()      # Very likely to cause failure
    MEDIUM = auto()    # May cause failure
    LOW = auto()       # Minor issue


@dataclass
class PredictedError:
    """A predicted error with metadata."""
    error_type: ErrorType
    severity: Severity
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    confidence: float = 0.0  # 0.0 - 1.0
    suggestion: str = ""
    code_snippet: str = ""
    related_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type.name,
            "severity": self.severity.name,
            "message": self.message,
            "line_number": self.line_number,
            "column": self.column,
            "confidence": self.confidence,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet,
            "related_patterns": self.related_patterns
        }


@dataclass
class PredictionResult:
    """Result of error prediction."""
    predicted_errors: List[PredictedError]
    overall_risk_score: float  # 0.0 - 1.0
    has_critical_errors: bool
    
    @property
    def has_errors(self) -> bool:
        return len(self.predicted_errors) > 0
    
    @property
    def error_count(self) -> int:
        return len(self.predicted_errors)
    
    def get_errors_by_severity(self, severity: Severity) -> List[PredictedError]:
        """Get errors filtered by severity."""
        return [e for e in self.predicted_errors if e.severity == severity]
    
    def get_errors_by_type(self, error_type: ErrorType) -> List[PredictedError]:
        """Get errors filtered by type."""
        return [e for e in self.predicted_errors if e.error_type == error_type]


class ErrorPredictor:
    """Error predictor for pre-compilation error detection.
    
    Features:
    - Static code analysis for common error patterns
    - Dynamic prediction based on historical error data
    - Test failure prediction
    - Confidence scoring
    """
    
    def __init__(self, knowledge_base: Optional[ErrorKnowledgeBase] = None):
        """Initialize error predictor.
        
        Args:
            knowledge_base: Optional error knowledge base for learning
        """
        self.knowledge_base = knowledge_base
        
        # Compile regex patterns for efficiency
        self._init_patterns()
        
        logger.info("[ErrorPredictor] Initialized")
    
    def _init_patterns(self):
        """Initialize error detection patterns."""
        # Syntax error patterns
        self.syntax_patterns = [
            (r'\)\s*\{', "Missing semicolon before brace", Severity.MEDIUM),
            (r'\b(if|while|for)\s*\([^)]+\)\s*[^;{]', "Missing braces for control statement", Severity.LOW),
            (r'\bclass\s+\w+\s*\{[^}]*\bclass\s+', "Nested class without proper structure", Severity.HIGH),
            (r'\breturn\s+[^;]+[^;{}]\s*$', "Missing semicolon after return", Severity.CRITICAL),
        ]
        
        # Type mismatch patterns
        self.type_patterns = [
            (r'\bString\s+\w+\s*=\s*\d+\b', "Assigning number to String", Severity.HIGH),
            (r'\bint\s+\w+\s*=\s*"[^"]*"', "Assigning string to int", Severity.CRITICAL),
            (r'\.equals\s*\(\s*\d+\s*\)', "Using equals() with primitive", Severity.HIGH),
            (r'==\s*"[^"]*"', "Using == for String comparison", Severity.MEDIUM),
        ]
        
        # Null pointer patterns
        self.null_patterns = [
            (r'\b\w+\s*\.\s*\w+\s*\([^)]*\)', "Method call on potentially null object", Severity.MEDIUM),
            (r'\.\s*length\b', "Accessing length without null check", Severity.MEDIUM),
            (r'\.\s*get\s*\(', "Calling get() without null check", Severity.MEDIUM),
            (r'@Nullable.*\n.*\.', "Using @Nullable without check", Severity.HIGH),
        ]
        
        # Import missing patterns
        self.import_patterns = [
            (r'\bList\s*<', "Missing import for List", Severity.MEDIUM),
            (r'\bMap\s*<', "Missing import for Map", Severity.MEDIUM),
            (r'\bSet\s*<', "Missing import for Set", Severity.MEDIUM),
            (r'\bArrayList\s*<', "Missing import for ArrayList", Severity.MEDIUM),
            (r'\bHashMap\s*<', "Missing import for HashMap", Severity.MEDIUM),
            (r'\b@Test\b', "Missing JUnit import", Severity.MEDIUM),
            (r'\b@BeforeEach\b', "Missing JUnit import", Severity.MEDIUM),
            (r'\bassertEquals\s*\(', "Missing Assertions import", Severity.MEDIUM),
            (r'\bMockito\.', "Missing Mockito import", Severity.MEDIUM),
            (r'\bwhen\s*\(', "Missing Mockito import", Severity.MEDIUM),
        ]
        
        # Method not found patterns
        self.method_patterns = [
            (r'\.\s*size\s*\(\s*\)', "Using size() on array (should be length)", Severity.HIGH),
            (r'\.\s*length\s*\(\s*\)', "Using length() on String (should be length())", Severity.MEDIUM),
            (r'\.\s*add\s*\(\s*\d+\s*,', "Using add(index, element) without checking List type", Severity.LOW),
        ]
        
        # Variable not found patterns
        self.variable_patterns = [
            (r'\b\w+\s*=\\w+\s*;', "Using undefined variable", Severity.CRITICAL),
        ]
        
        # Resource leak patterns
        self.resource_patterns = [
            (r'\bnew\s+FileInputStream\s*\(', "FileInputStream not closed", Severity.MEDIUM),
            (r'\bnew\s+FileOutputStream\s*\(', "FileOutputStream not closed", Severity.MEDIUM),
            (r'\bnew\s+Scanner\s*\(', "Scanner not closed", Severity.LOW),
            (r'\.create\(\)', "Resource created but not closed", Severity.LOW),
        ]
        
        # Concurrency patterns
        self.concurrency_patterns = [
            (r'\bArrayList\s*<[^>]*>\s+\w+\s*=\s*new\s+ArrayList', "Using non-thread-safe ArrayList in concurrent context", Severity.MEDIUM),
            (r'\bHashMap\s*<[^>]*>\s+\w+\s*=\s*new\s+HashMap', "Using non-thread-safe HashMap in concurrent context", Severity.MEDIUM),
        ]
        
        # Test assertion patterns
        self.test_assertion_patterns = [
            (r'assertEquals\s*\(\s*expected\s*,\s*actual\s*\)', "assertEquals with wrong argument order", Severity.HIGH),
            (r'assertTrue\s*\(\s*false\s*\)', "assertTrue(false) - always fails", Severity.CRITICAL),
            (r'assertFalse\s*\(\s*true\s*\)', "assertFalse(true) - always fails", Severity.CRITICAL),
            (r'assertNull\s*\(\s*new\s+', "assertNull with new object - always fails", Severity.CRITICAL),
            (r'assertNotNull\s*\(\s*null\s*\)', "assertNotNull(null) - always fails", Severity.CRITICAL),
        ]
        
        # Test setup patterns
        self.test_setup_patterns = [
            (r'@Test\s+void\s+\w+\s*\(\s*\)\s*\{\s*\}', "Empty test method", Severity.LOW),
            (r'@Test\s+void\s+\w+\s*\(\s*\)\s*\{[^}]*//\s*TODO', "Test with TODO comment", Severity.LOW),
        ]
    
    def predict_compilation_errors(self, code: str, file_path: Optional[str] = None) -> PredictionResult:
        """Predict compilation errors before compilation.
        
        Args:
            code: Java source code
            file_path: Optional file path for context
            
        Returns:
            PredictionResult with predicted errors
        """
        predicted_errors = []
        
        # Check syntax patterns
        predicted_errors.extend(self._check_syntax_patterns(code))
        
        # Check type patterns
        predicted_errors.extend(self._check_type_patterns(code))
        
        # Check null pointer patterns
        predicted_errors.extend(self._check_null_patterns(code))
        
        # Check import patterns
        predicted_errors.extend(self._check_import_patterns(code))
        
        # Check method patterns
        predicted_errors.extend(self._check_method_patterns(code))
        
        # Check from knowledge base if available
        if self.knowledge_base:
            predicted_errors.extend(self._check_historical_patterns(code))
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(predicted_errors)
        has_critical = any(e.severity == Severity.CRITICAL for e in predicted_errors)
        
        result = PredictionResult(
            predicted_errors=predicted_errors,
            overall_risk_score=risk_score,
            has_critical_errors=has_critical
        )
        
        logger.info(f"[ErrorPredictor] Predicted {len(predicted_errors)} errors, "
                   f"risk score: {risk_score:.2f}")
        
        return result
    
    def predict_test_failures(self, test_code: str, source_code: Optional[str] = None) -> PredictionResult:
        """Predict test failures before execution.
        
        Args:
            test_code: Test source code
            source_code: Optional source code being tested
            
        Returns:
            PredictionResult with predicted failures
        """
        predicted_errors = []
        
        # Check assertion patterns
        predicted_errors.extend(self._check_assertion_patterns(test_code))
        
        # Check test setup patterns
        predicted_errors.extend(self._check_test_setup_patterns(test_code))
        
        # Check resource patterns
        predicted_errors.extend(self._check_resource_patterns(test_code))
        
        # Check concurrency patterns
        predicted_errors.extend(self._check_concurrency_patterns(test_code))
        
        # If source code provided, check test coverage
        if source_code:
            predicted_errors.extend(self._check_test_coverage(test_code, source_code))
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(predicted_errors)
        has_critical = any(e.severity == Severity.CRITICAL for e in predicted_errors)
        
        result = PredictionResult(
            predicted_errors=predicted_errors,
            overall_risk_score=risk_score,
            has_critical_errors=has_critical
        )
        
        logger.info(f"[ErrorPredictor] Predicted {len(predicted_errors)} test failures, "
                   f"risk score: {risk_score:.2f}")
        
        return result
    
    def _check_syntax_patterns(self, code: str) -> List[PredictedError]:
        """Check for syntax error patterns."""
        errors = []
        lines = code.split('\n')
        
        for pattern, message, severity in self.syntax_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    errors.append(PredictedError(
                        error_type=ErrorType.SYNTAX_ERROR,
                        severity=severity,
                        message=message,
                        line_number=i,
                        confidence=0.8,
                        suggestion="Check syntax and add missing elements",
                        code_snippet=line.strip()
                    ))
        
        return errors
    
    def _check_type_patterns(self, code: str) -> List[PredictedError]:
        """Check for type mismatch patterns."""
        errors = []
        lines = code.split('\n')
        
        for pattern, message, severity in self.type_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    errors.append(PredictedError(
                        error_type=ErrorType.TYPE_MISMATCH,
                        severity=severity,
                        message=message,
                        line_number=i,
                        confidence=0.85,
                        suggestion="Ensure type compatibility in assignments and comparisons",
                        code_snippet=line.strip()
                    ))
        
        return errors
    
    def _check_null_patterns(self, code: str) -> List[PredictedError]:
        """Check for null pointer patterns."""
        errors = []
        lines = code.split('\n')
        
        for pattern, message, severity in self.null_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line) and 'null' not in line.lower():
                    errors.append(PredictedError(
                        error_type=ErrorType.NULL_POINTER,
                        severity=severity,
                        message=message,
                        line_number=i,
                        confidence=0.6,
                        suggestion="Add null check before accessing object members",
                        code_snippet=line.strip()
                    ))
        
        return errors
    
    def _check_import_patterns(self, code: str) -> List[PredictedError]:
        """Check for missing import patterns."""
        errors = []
        
        # Check if imports exist
        has_import_section = 'import ' in code
        
        if has_import_section:
            for pattern, message, severity in self.import_patterns:
                if re.search(pattern, code):
                    # Check if corresponding import exists
                    import_map = {
                        'List': 'java.util.List',
                        'Map': 'java.util.Map',
                        'Set': 'java.util.Set',
                        'ArrayList': 'java.util.ArrayList',
                        'HashMap': 'java.util.HashMap',
                        '@Test': 'org.junit.jupiter.api.Test',
                        '@BeforeEach': 'org.junit.jupiter.api.BeforeEach',
                        'assertEquals': 'org.junit.jupiter.api.Assertions',
                        'Mockito': 'org.mockito.Mockito',
                        'when': 'org.mockito.Mockito',
                    }
                    
                    for key, import_stmt in import_map.items():
                        if key in message and import_stmt not in code:
                            errors.append(PredictedError(
                                error_type=ErrorType.IMPORT_MISSING,
                                severity=severity,
                                message=message,
                                confidence=0.75,
                                suggestion=f"Add import: {import_stmt}",
                                code_snippet=f"import {import_stmt};"
                            ))
        
        return errors
    
    def _check_method_patterns(self, code: str) -> List[PredictedError]:
        """Check for method-related patterns."""
        errors = []
        lines = code.split('\n')
        
        for pattern, message, severity in self.method_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    errors.append(PredictedError(
                        error_type=ErrorType.METHOD_NOT_FOUND,
                        severity=severity,
                        message=message,
                        line_number=i,
                        confidence=0.7,
                        suggestion="Check method name and signature",
                        code_snippet=line.strip()
                    ))
        
        return errors
    
    def _check_assertion_patterns(self, code: str) -> List[PredictedError]:
        """Check for assertion error patterns."""
        errors = []
        lines = code.split('\n')
        
        for pattern, message, severity in self.test_assertion_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    errors.append(PredictedError(
                        error_type=ErrorType.TEST_ASSERTION_FAILURE,
                        severity=severity,
                        message=message,
                        line_number=i,
                        confidence=0.9,
                        suggestion="Fix assertion logic",
                        code_snippet=line.strip()
                    ))
        
        return errors
    
    def _check_test_setup_patterns(self, code: str) -> List[PredictedError]:
        """Check for test setup patterns."""
        errors = []
        lines = code.split('\n')
        
        for pattern, message, severity in self.test_setup_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    errors.append(PredictedError(
                        error_type=ErrorType.TEST_SETUP_FAILURE,
                        severity=severity,
                        message=message,
                        line_number=i,
                        confidence=0.6,
                        suggestion="Complete test implementation",
                        code_snippet=line.strip()
                    ))
        
        return errors
    
    def _check_resource_patterns(self, code: str) -> List[PredictedError]:
        """Check for resource leak patterns."""
        errors = []
        lines = code.split('\n')
        
        for pattern, message, severity in self.resource_patterns:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    # Check if resource is closed
                    resource_var = re.search(r'(\w+)\s*=\s*new', line)
                    if resource_var:
                        var_name = resource_var.group(1)
                        if not re.search(rf'{var_name}\.close\(\)', code):
                            errors.append(PredictedError(
                                error_type=ErrorType.RESOURCE_LEAK,
                                severity=severity,
                                message=message,
                                line_number=i,
                                confidence=0.65,
                                suggestion=f"Close resource: {var_name}.close() or use try-with-resources",
                                code_snippet=line.strip()
                            ))
        
        return errors
    
    def _check_concurrency_patterns(self, code: str) -> List[PredictedError]:
        """Check for concurrency issue patterns."""
        errors = []
        lines = code.split('\n')
        
        # Check for synchronized or concurrent annotations
        is_concurrent = any(annotation in code for annotation in 
                          ['@ThreadSafe', '@Synchronized', 'synchronized', 'Executor', 'Thread'])
        
        if is_concurrent:
            for pattern, message, severity in self.concurrency_patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        errors.append(PredictedError(
                            error_type=ErrorType.CONCURRENCY_ISSUE,
                            severity=severity,
                            message=message,
                            line_number=i,
                            confidence=0.7,
                            suggestion="Use thread-safe alternatives (Vector, ConcurrentHashMap, CopyOnWriteArrayList)",
                            code_snippet=line.strip()
                        ))
        
        return errors
    
    def _check_historical_patterns(self, code: str) -> List[PredictedError]:
        """Check for patterns from historical errors."""
        errors = []
        
        if not self.knowledge_base:
            return errors
        
        # Query knowledge base for similar errors
        similar_errors = self.knowledge_base.find_similar_errors(code, top_k=5)
        
        for error_record in similar_errors:
            if error_record.get('similarity', 0) > 0.7:
                errors.append(PredictedError(
                    error_type=ErrorType.SYNTAX_ERROR,  # Default type
                    severity=Severity.MEDIUM,
                    message=f"Similar to historical error: {error_record.get('message', '')}",
                    confidence=error_record.get('similarity', 0.5),
                    suggestion=error_record.get('solution', 'Review and fix'),
                    related_patterns=[error_record.get('pattern', '')]
                ))
        
        return errors
    
    def _check_test_coverage(self, test_code: str, source_code: str) -> List[PredictedError]:
        """Check for potential test coverage issues."""
        errors = []
        
        # Extract methods from source code
        source_methods = re.findall(r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(', source_code)
        
        # Extract test method names
        test_methods = re.findall(r'@Test\s+(?:public\s+)?void\s+(\w+)\s*\(', test_code)
        
        # Check for untested methods
        for method in source_methods:
            method_tested = any(method.lower() in test.lower() for test in test_methods)
            if not method_tested and not method.startswith('get') and not method.startswith('set'):
                errors.append(PredictedError(
                    error_type=ErrorType.TEST_SETUP_FAILURE,
                    severity=Severity.LOW,
                    message=f"Method '{method}' may not have corresponding test",
                    confidence=0.5,
                    suggestion=f"Consider adding test for {method}",
                    related_patterns=[method]
                ))
        
        return errors
    
    def _calculate_risk_score(self, errors: List[PredictedError]) -> float:
        """Calculate overall risk score from predicted errors.
        
        Args:
            errors: List of predicted errors
            
        Returns:
            Risk score from 0.0 to 1.0
        """
        if not errors:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            Severity.CRITICAL: 1.0,
            Severity.HIGH: 0.7,
            Severity.MEDIUM: 0.4,
            Severity.LOW: 0.1
        }
        
        total_weight = sum(
            severity_weights.get(e.severity, 0.1) * e.confidence
            for e in errors
        )
        
        # Normalize to 0-1 range
        max_possible = len(errors) * 1.0
        risk_score = min(total_weight / max_possible if max_possible > 0 else 0, 1.0)
        
        return risk_score
    
    def get_prediction_summary(self, result: PredictionResult) -> str:
        """Get human-readable prediction summary.
        
        Args:
            result: Prediction result
            
        Returns:
            Summary string
        """
        lines = [
            f"Error Prediction Summary:",
            f"  Total predicted errors: {result.error_count}",
            f"  Overall risk score: {result.overall_risk_score:.1%}",
            f"  Critical errors: {len(result.get_errors_by_severity(Severity.CRITICAL))}",
            f"  High severity: {len(result.get_errors_by_severity(Severity.HIGH))}",
            f"  Medium severity: {len(result.get_errors_by_severity(Severity.MEDIUM))}",
            f"  Low severity: {len(result.get_errors_by_severity(Severity.LOW))}",
            "",
            "Top issues:"
        ]
        
        # Show top 5 errors by severity and confidence
        sorted_errors = sorted(
            result.predicted_errors,
            key=lambda e: (e.severity.value, e.confidence),
            reverse=True
        )
        
        for error in sorted_errors[:5]:
            lines.append(f"  [{error.severity.name}] {error.message}")
            if error.line_number:
                lines.append(f"    Line {error.line_number}: {error.suggestion}")
            else:
                lines.append(f"    {error.suggestion}")
        
        return "\n".join(lines)


def create_error_predictor(
    knowledge_base: Optional[ErrorKnowledgeBase] = None
) -> ErrorPredictor:
    """Create an error predictor instance.
    
    Args:
        knowledge_base: Optional error knowledge base
        
    Returns:
        Configured ErrorPredictor
    """
    return ErrorPredictor(knowledge_base=knowledge_base)
