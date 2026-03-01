"""Error prediction for proactive error prevention.

This module provides error prediction capabilities:
- Static analysis for potential errors
- Pattern-based prediction
- Risk assessment
- Preventive recommendations
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    VERY_HIGH = auto()


class ErrorType(Enum):
    """Types of predictable errors."""
    MISSING_IMPORT = auto()
    UNDEFINED_VARIABLE = auto()
    TYPE_MISMATCH = auto()
    NULL_POINTER = auto()
    SYNTAX_ERROR = auto()
    COMPILATION_ERROR = auto()
    TEST_FAILURE = auto()
    ASSERTION_FAILURE = auto()
    MOCK_CONFIGURATION = auto()
    RESOURCE_LEAK = auto()
    CONCURRENCY_ISSUE = auto()


@dataclass
class PredictedError:
    """A predicted error."""
    error_type: ErrorType
    message: str
    location: Optional[Tuple[int, int]]
    confidence: PredictionConfidence
    probability: float
    prevention_suggestion: str
    related_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of error prediction."""
    predictions: List[PredictedError]
    overall_risk_score: float
    analyzed_lines: int
    analysis_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StaticAnalyzer:
    """Static analysis for error prediction."""
    
    def __init__(self):
        self._patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[ErrorType, List[re.Pattern]]:
        """Load error detection patterns."""
        return {
            ErrorType.MISSING_IMPORT: [
                re.compile(r'\b(List|Map|Set|ArrayList|HashMap|HashSet)\b'),
                re.compile(r'\b(assertEquals|assertTrue|assertFalse|assertNull)\b'),
                re.compile(r'\b(Mock|InjectMocks|Mockito)\b'),
                re.compile(r'\b(Test|BeforeEach|AfterEach|BeforeAll|AfterAll)\b'),
            ],
            ErrorType.UNDEFINED_VARIABLE: [
                re.compile(r'\b\w+\s*\.\s*\w+\s*\('),
            ],
            ErrorType.NULL_POINTER: [
                re.compile(r'\.\w+\(\)(?!\s*;|\s*\))'),
                re.compile(r'return\s+null\s*;'),
            ],
            ErrorType.TYPE_MISMATCH: [
                re.compile(r'=\s*null\s*;(?![^;]*String|[^;]*Integer|[^;]*Object)'),
            ],
            ErrorType.ASSERTION_FAILURE: [
                re.compile(r'assertEquals\s*\(\s*null\s*,'),
                re.compile(r'assertTrue\s*\(\s*false\s*\)'),
                re.compile(r'assertFalse\s*\(\s*true\s*\)'),
            ],
            ErrorType.MOCK_CONFIGURATION: [
                re.compile(r'@Mock(?![\s\S]*MockitoAnnotations\.openMocks)'),
                re.compile(r'@InjectMocks(?![\s\S]*MockitoAnnotations\.openMocks)'),
            ],
        }
    
    def analyze(self, code: str) -> List[PredictedError]:
        """Analyze code for potential errors.
        
        Args:
            code: Code to analyze
            
        Returns:
            List of predicted errors
        """
        predictions = []
        lines = code.split('\n')
        
        imports = self._extract_imports(code)
        
        for error_type, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(code):
                    line_no = code[:match.start()].count('\n') + 1
                    
                    if error_type == ErrorType.MISSING_IMPORT:
                        symbol = match.group(1) if match.groups() else match.group(0)
                        if not self._is_imported(symbol, imports):
                            predictions.append(PredictedError(
                                error_type=error_type,
                                message=f"Potential missing import for: {symbol}",
                                location=(line_no, match.start() - code[:match.start()].rfind('\n')),
                                confidence=PredictionConfidence.HIGH,
                                probability=0.85,
                                prevention_suggestion=f"Add import statement for {symbol}",
                                related_code=match.group(0)
                            ))
                    
                    elif error_type == ErrorType.ASSERTION_FAILURE:
                        predictions.append(PredictedError(
                            error_type=error_type,
                            message="Assertion that always fails",
                            location=(line_no, match.start() - code[:match.start()].rfind('\n')),
                            confidence=PredictionConfidence.VERY_HIGH,
                            probability=0.95,
                            prevention_suggestion="Review assertion logic",
                            related_code=match.group(0)
                        ))
                    
                    elif error_type == ErrorType.MOCK_CONFIGURATION:
                        if '@BeforeEach' not in code and 'MockitoAnnotations.openMocks' not in code:
                            predictions.append(PredictedError(
                                error_type=error_type,
                                message="Mock annotations without initialization",
                                location=(line_no, match.start() - code[:match.start()].rfind('\n')),
                                confidence=PredictionConfidence.HIGH,
                                probability=0.80,
                                prevention_suggestion="Add MockitoAnnotations.openMocks(this) in @BeforeEach",
                                related_code=match.group(0)
                            ))
        
        return predictions
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements."""
        imports = []
        for match in re.finditer(r'import\s+(?:static\s+)?([\w.]+)', code):
            imports.append(match.group(1))
        return imports
    
    def _is_imported(self, symbol: str, imports: List[str]) -> bool:
        """Check if a symbol is imported."""
        for imp in imports:
            if imp.endswith(f'.{symbol}') or imp.endswith('.*'):
                return True
            if imp.split('.')[-1] == symbol:
                return True
        return False


class PatternBasedPredictor:
    """Predicts errors based on historical patterns."""
    
    def __init__(self):
        self._error_patterns: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_pattern(
        self,
        code_pattern: str,
        predicted_error: PredictedError
    ):
        """Register an error pattern."""
        if code_pattern not in self._error_patterns:
            self._error_patterns[code_pattern] = []
        self._error_patterns[code_pattern].append({
            "error": predicted_error,
            "occurrences": 0,
            "confirmed": 0
        })
    
    def predict(self, code: str) -> List[PredictedError]:
        """Predict errors based on patterns.
        
        Args:
            code: Code to analyze
            
        Returns:
            List of predicted errors
        """
        predictions = []
        
        for pattern, errors in self._error_patterns.items():
            if re.search(pattern, code, re.DOTALL):
                for error_data in errors:
                    error = error_data["error"]
                    occurrences = error_data["occurrences"]
                    confirmed = error_data["confirmed"]
                    
                    probability = confirmed / occurrences if occurrences > 0 else 0.5
                    
                    predictions.append(PredictedError(
                        error_type=error.error_type,
                        message=error.message,
                        location=error.location,
                        confidence=self._get_confidence(probability),
                        probability=probability,
                        prevention_suggestion=error.prevention_suggestion,
                        metadata={"pattern": pattern}
                    ))
        
        return predictions
    
    def _get_confidence(self, probability: float) -> PredictionConfidence:
        """Get confidence level from probability."""
        if probability >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif probability >= 0.6:
            return PredictionConfidence.HIGH
        elif probability >= 0.4:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW
    
    def record_occurrence(self, pattern: str, confirmed: bool):
        """Record a pattern occurrence."""
        if pattern in self._error_patterns:
            for error_data in self._error_patterns[pattern]:
                error_data["occurrences"] += 1
                if confirmed:
                    error_data["confirmed"] += 1


class RiskAssessor:
    """Assesses overall risk of code."""
    
    def __init__(self):
        self._risk_weights = {
            ErrorType.COMPILATION_ERROR: 0.9,
            ErrorType.NULL_POINTER: 0.8,
            ErrorType.MISSING_IMPORT: 0.7,
            ErrorType.TYPE_MISMATCH: 0.7,
            ErrorType.TEST_FAILURE: 0.6,
            ErrorType.ASSERTION_FAILURE: 0.5,
            ErrorType.MOCK_CONFIGURATION: 0.5,
            ErrorType.UNDEFINED_VARIABLE: 0.8,
            ErrorType.SYNTAX_ERROR: 0.9,
            ErrorType.RESOURCE_LEAK: 0.6,
            ErrorType.CONCURRENCY_ISSUE: 0.7,
        }
    
    def assess_risk(
        self,
        predictions: List[PredictedError]
    ) -> float:
        """Assess overall risk score.
        
        Args:
            predictions: List of predicted errors
            
        Returns:
            Risk score (0.0-1.0)
        """
        if not predictions:
            return 0.0
        
        total_risk = 0.0
        
        for pred in predictions:
            weight = self._risk_weights.get(pred.error_type, 0.5)
            confidence_mult = {
                PredictionConfidence.VERY_HIGH: 1.0,
                PredictionConfidence.HIGH: 0.8,
                PredictionConfidence.MEDIUM: 0.6,
                PredictionConfidence.LOW: 0.4,
            }.get(pred.confidence, 0.5)
            
            risk = weight * pred.probability * confidence_mult
            total_risk += risk
        
        max_possible_risk = len(predictions)
        normalized_risk = min(1.0, total_risk / max_possible_risk) if max_possible_risk > 0 else 0.0
        
        return normalized_risk


class ErrorPredictor:
    """Main error predictor combining all strategies.
    
    Features:
    - Static analysis
    - Pattern-based prediction
    - Risk assessment
    - Prevention recommendations
    """
    
    def __init__(self):
        self.static_analyzer = StaticAnalyzer()
        self.pattern_predictor = PatternBasedPredictor()
        self.risk_assessor = RiskAssessor()
        
        self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load default error patterns."""
        self.pattern_predictor.register_pattern(
            r'@Test\s+public\s+void\s+\w+\(\)\s*\{[^}]*\}',
            PredictedError(
                error_type=ErrorType.TEST_FAILURE,
                message="Test method without assertions",
                location=None,
                confidence=PredictionConfidence.MEDIUM,
                probability=0.6,
                prevention_suggestion="Add assertions to verify expected behavior"
            )
        )
        
        self.pattern_predictor.register_pattern(
            r'new\s+\w+\(\)\s*;\s*(?!\s*assert)',
            PredictedError(
                error_type=ErrorType.TEST_FAILURE,
                message="Object created but not used in assertions",
                location=None,
                confidence=PredictionConfidence.LOW,
                probability=0.4,
                prevention_suggestion="Use the created object in assertions"
            )
        )
    
    def predict_compilation_errors(
        self,
        code: str
    ) -> List[PredictedError]:
        """Predict compilation errors.
        
        Args:
            code: Code to analyze
            
        Returns:
            List of predicted compilation errors
        """
        import time
        start_time = time.time()
        
        static_predictions = self.static_analyzer.analyze(code)
        
        pattern_predictions = self.pattern_predictor.predict(code)
        
        all_predictions = static_predictions + pattern_predictions
        
        compilation_types = {
            ErrorType.MISSING_IMPORT,
            ErrorType.SYNTAX_ERROR,
            ErrorType.TYPE_MISMATCH,
            ErrorType.UNDEFINED_VARIABLE,
        }
        
        compilation_predictions = [
            p for p in all_predictions
            if p.error_type in compilation_types
        ]
        
        logger.info(
            f"[ErrorPredictor] Found {len(compilation_predictions)} potential "
            f"compilation errors in {time.time() - start_time:.2f}s"
        )
        
        return compilation_predictions
    
    def predict_test_failures(
        self,
        test_code: str
    ) -> List[PredictedError]:
        """Predict test failures.
        
        Args:
            test_code: Test code to analyze
            
        Returns:
            List of predicted test failures
        """
        import time
        start_time = time.time()
        
        static_predictions = self.static_analyzer.analyze(test_code)
        
        pattern_predictions = self.pattern_predictor.predict(test_code)
        
        all_predictions = static_predictions + pattern_predictions
        
        test_types = {
            ErrorType.TEST_FAILURE,
            ErrorType.ASSERTION_FAILURE,
            ErrorType.MOCK_CONFIGURATION,
            ErrorType.NULL_POINTER,
        }
        
        test_predictions = [
            p for p in all_predictions
            if p.error_type in test_types
        ]
        
        logger.info(
            f"[ErrorPredictor] Found {len(test_predictions)} potential "
            f"test failures in {time.time() - start_time:.2f}s"
        )
        
        return test_predictions
    
    def analyze(
        self,
        code: str,
        code_type: str = "source"
    ) -> PredictionResult:
        """Perform full error prediction analysis.
        
        Args:
            code: Code to analyze
            code_type: Type of code ("source" or "test")
            
        Returns:
            PredictionResult with all predictions
        """
        import time
        start_time = time.time()
        
        static_predictions = self.static_analyzer.analyze(code)
        pattern_predictions = self.pattern_predictor.predict(code)
        
        all_predictions = static_predictions + pattern_predictions
        
        all_predictions.sort(key=lambda p: p.probability, reverse=True)
        
        overall_risk = self.risk_assessor.assess_risk(all_predictions)
        
        lines = code.split('\n')
        analysis_time = time.time() - start_time
        
        logger.info(
            f"[ErrorPredictor] Analysis complete - "
            f"Predictions: {len(all_predictions)}, "
            f"Risk: {overall_risk:.2f}, "
            f"Time: {analysis_time:.2f}s"
        )
        
        return PredictionResult(
            predictions=all_predictions,
            overall_risk_score=overall_risk,
            analyzed_lines=len(lines),
            analysis_time=analysis_time,
            metadata={"code_type": code_type}
        )
    
    def get_prevention_recommendations(
        self,
        predictions: List[PredictedError]
    ) -> List[str]:
        """Get prevention recommendations.
        
        Args:
            predictions: List of predicted errors
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for pred in sorted(predictions, key=lambda p: p.probability, reverse=True):
            if pred.prevention_suggestion not in recommendations:
                recommendations.append(pred.prevention_suggestion)
        
        return recommendations[:10]


def create_error_predictor() -> ErrorPredictor:
    """Create an ErrorPredictor instance."""
    return ErrorPredictor()
