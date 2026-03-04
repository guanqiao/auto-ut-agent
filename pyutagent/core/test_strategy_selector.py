"""Intelligent Test Strategy Selector for choosing optimal test generation strategies.

This module analyzes code characteristics and selects the most appropriate
test generation strategy for different scenarios.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class TestStrategy(Enum):
    """Available test generation strategies."""
    UNIT_BASIC = "unit_basic"
    UNIT_MOCK = "unit_mock"
    UNIT_PARAMETERIZED = "unit_parameterized"
    INTEGRATION = "integration"
    BOUNDARY = "boundary"
    EXCEPTION = "exception"
    PROPERTY_BASED = "property_based"
    BEHAVIOR_DRIVEN = "behavior_driven"
    DATA_DRIVEN = "data_driven"
    PERFORMANCE = "performance"


class CodeCharacteristic(Enum):
    """Characteristics of code that influence strategy selection."""
    HAS_DEPENDENCIES = "has_dependencies"
    HAS_EXCEPTIONS = "has_exceptions"
    HAS_CONDITIONALS = "has_conditionals"
    HAS_LOOPS = "has_loops"
    HAS_COLLECTIONS = "has_collections"
    HAS_EXTERNAL_CALLS = "has_external_calls"
    IS_STATELESS = "is_stateless"
    IS_PURE_FUNCTION = "is_pure_function"
    HAS_COMPLEX_LOGIC = "has_complex_logic"
    HAS_VALIDATION = "has_validation"
    HAS_BOUNDARY_CONDITIONS = "has_boundary_conditions"
    IS_SERVICE_CLASS = "is_service_class"
    IS_UTILITY_CLASS = "is_utility_class"
    IS_DATA_CLASS = "is_data_class"


@dataclass
class StrategyScore:
    """Score for a test strategy."""
    strategy: TestStrategy
    score: float
    reasons: List[str]
    applicable_patterns: List[str]
    estimated_effort: str


@dataclass
class CodeAnalysis:
    """Result of code analysis for strategy selection."""
    characteristics: List[CodeCharacteristic]
    complexity_score: float
    dependency_count: int
    method_count: int
    branch_count: int
    exception_types: List[str]
    parameter_types: List[str]
    return_type: str
    annotations: List[str]
    suggested_strategies: List[TestStrategy]


@dataclass
class StrategyRecommendation:
    """Final strategy recommendation."""
    primary_strategy: TestStrategy
    secondary_strategies: List[TestStrategy]
    analysis: CodeAnalysis
    scores: List[StrategyScore]
    confidence: float
    reasoning: str


class TestStrategySelector:
    """Intelligent test strategy selector.
    
    Analyzes code characteristics and recommends optimal test strategies.
    
    Features:
    - Code characteristic detection
    - Strategy scoring and ranking
    - Context-aware recommendations
    - Effort estimation
    - Pattern matching
    """
    
    def __init__(self):
        """Initialize the test strategy selector."""
        self._strategy_weights = self._initialize_strategy_weights()
        self._characteristic_detectors = self._initialize_detectors()
        
        logger.info("[TestStrategySelector] Initialized")
    
    def _initialize_strategy_weights(self) -> Dict[TestStrategy, Dict[CodeCharacteristic, float]]:
        """Initialize weights for strategy-characteristic combinations."""
        return {
            TestStrategy.UNIT_BASIC: {
                CodeCharacteristic.IS_STATELESS: 0.9,
                CodeCharacteristic.IS_PURE_FUNCTION: 0.9,
                CodeCharacteristic.IS_UTILITY_CLASS: 0.8,
                CodeCharacteristic.HAS_DEPENDENCIES: -0.3,
            },
            TestStrategy.UNIT_MOCK: {
                CodeCharacteristic.HAS_DEPENDENCIES: 0.9,
                CodeCharacteristic.IS_SERVICE_CLASS: 0.8,
                CodeCharacteristic.HAS_EXTERNAL_CALLS: 0.8,
                CodeCharacteristic.IS_STATELESS: -0.2,
            },
            TestStrategy.UNIT_PARAMETERIZED: {
                CodeCharacteristic.HAS_VALIDATION: 0.8,
                CodeCharacteristic.HAS_BOUNDARY_CONDITIONS: 0.8,
                CodeCharacteristic.HAS_COLLECTIONS: 0.7,
                CodeCharacteristic.HAS_LOOPS: 0.6,
            },
            TestStrategy.INTEGRATION: {
                CodeCharacteristic.HAS_EXTERNAL_CALLS: 0.9,
                CodeCharacteristic.HAS_DEPENDENCIES: 0.7,
                CodeCharacteristic.IS_SERVICE_CLASS: 0.6,
            },
            TestStrategy.BOUNDARY: {
                CodeCharacteristic.HAS_BOUNDARY_CONDITIONS: 0.95,
                CodeCharacteristic.HAS_VALIDATION: 0.8,
                CodeCharacteristic.HAS_CONDITIONALS: 0.6,
            },
            TestStrategy.EXCEPTION: {
                CodeCharacteristic.HAS_EXCEPTIONS: 0.95,
                CodeCharacteristic.HAS_VALIDATION: 0.7,
            },
            TestStrategy.PROPERTY_BASED: {
                CodeCharacteristic.IS_PURE_FUNCTION: 0.9,
                CodeCharacteristic.HAS_COLLECTIONS: 0.7,
                CodeCharacteristic.HAS_COMPLEX_LOGIC: 0.6,
            },
            TestStrategy.BEHAVIOR_DRIVEN: {
                CodeCharacteristic.IS_SERVICE_CLASS: 0.8,
                CodeCharacteristic.HAS_DEPENDENCIES: 0.7,
                CodeCharacteristic.HAS_COMPLEX_LOGIC: 0.6,
            },
            TestStrategy.DATA_DRIVEN: {
                CodeCharacteristic.HAS_COLLECTIONS: 0.8,
                CodeCharacteristic.HAS_VALIDATION: 0.7,
                CodeCharacteristic.IS_DATA_CLASS: 0.6,
            },
            TestStrategy.PERFORMANCE: {
                CodeCharacteristic.HAS_LOOPS: 0.7,
                CodeCharacteristic.HAS_COLLECTIONS: 0.6,
                CodeCharacteristic.HAS_COMPLEX_LOGIC: 0.5,
            },
        }
    
    def _initialize_detectors(self) -> Dict[CodeCharacteristic, callable]:
        """Initialize characteristic detection functions."""
        return {
            CodeCharacteristic.HAS_DEPENDENCIES: self._detect_dependencies,
            CodeCharacteristic.HAS_EXCEPTIONS: self._detect_exceptions,
            CodeCharacteristic.HAS_CONDITIONALS: self._detect_conditionals,
            CodeCharacteristic.HAS_LOOPS: self._detect_loops,
            CodeCharacteristic.HAS_COLLECTIONS: self._detect_collections,
            CodeCharacteristic.HAS_EXTERNAL_CALLS: self._detect_external_calls,
            CodeCharacteristic.IS_STATELESS: self._detect_stateless,
            CodeCharacteristic.IS_PURE_FUNCTION: self._detect_pure_function,
            CodeCharacteristic.HAS_COMPLEX_LOGIC: self._detect_complex_logic,
            CodeCharacteristic.HAS_VALIDATION: self._detect_validation,
            CodeCharacteristic.HAS_BOUNDARY_CONDITIONS: self._detect_boundaries,
            CodeCharacteristic.IS_SERVICE_CLASS: self._detect_service_class,
            CodeCharacteristic.IS_UTILITY_CLASS: self._detect_utility_class,
            CodeCharacteristic.IS_DATA_CLASS: self._detect_data_class,
        }
    
    def analyze_code(
        self,
        source_code: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> CodeAnalysis:
        """Analyze code to extract characteristics.
        
        Args:
            source_code: Java source code
            class_info: Optional class metadata
            
        Returns:
            CodeAnalysis with detected characteristics
        """
        characteristics = []
        
        for char_type, detector in self._characteristic_detectors.items():
            if detector(source_code, class_info):
                characteristics.append(char_type)
        
        complexity_score = self._calculate_complexity(source_code)
        
        dependency_count = len(re.findall(r'@Autowired|@Inject|private\s+\w+\s+\w+\s*;', source_code))
        method_count = len(re.findall(r'(?:public|private|protected)\s+\w+\s+\w+\s*\(', source_code))
        branch_count = len(re.findall(r'\bif\b|\bswitch\b|\bcase\b', source_code))
        
        exception_types = re.findall(r'throws\s+([\w,\s]+)', source_code)
        exception_types = [e.strip() for ex in exception_types for e in ex.split(',')]
        
        param_match = re.search(r'public\s+\w+\s+\w+\s*\(([^)]*)\)', source_code)
        parameter_types = []
        if param_match:
            params = param_match.group(1)
            parameter_types = [p.strip().split()[0] for p in params.split(',') if p.strip()]
        
        return_match = re.search(r'public\s+(\w+(?:<[\w<>,\s]+>)?)\s+\w+\s*\(', source_code)
        return_type = return_match.group(1) if return_match else "void"
        
        annotations = re.findall(r'@(\w+)', source_code)
        
        suggested = self._suggest_strategies(characteristics)
        
        return CodeAnalysis(
            characteristics=characteristics,
            complexity_score=complexity_score,
            dependency_count=dependency_count,
            method_count=method_count,
            branch_count=branch_count,
            exception_types=exception_types,
            parameter_types=parameter_types,
            return_type=return_type,
            annotations=annotations,
            suggested_strategies=suggested
        )
    
    def select_strategy(
        self,
        source_code: str,
        class_info: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> StrategyRecommendation:
        """Select the optimal test strategy.
        
        Args:
            source_code: Java source code
            class_info: Optional class metadata
            preferences: Optional user preferences
            
        Returns:
            StrategyRecommendation with selected strategy
        """
        analysis = self.analyze_code(source_code, class_info)
        
        scores = self._score_strategies(analysis, preferences)
        
        if not scores:
            scores = [StrategyScore(
                strategy=TestStrategy.UNIT_BASIC,
                score=0.5,
                reasons=["Default strategy for unknown code patterns"],
                applicable_patterns=["basic_test"],
                estimated_effort="low"
            )]
        
        scores.sort(key=lambda s: s.score, reverse=True)
        
        primary = scores[0].strategy
        secondary = [s.strategy for s in scores[1:4] if s.score > 0.3]
        
        confidence = self._calculate_confidence(scores)
        reasoning = self._generate_reasoning(analysis, scores[0])
        
        return StrategyRecommendation(
            primary_strategy=primary,
            secondary_strategies=secondary,
            analysis=analysis,
            scores=scores,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _score_strategies(
        self,
        analysis: CodeAnalysis,
        preferences: Optional[Dict[str, Any]] = None
    ) -> List[StrategyScore]:
        """Score all strategies based on analysis."""
        scores = []
        
        for strategy in TestStrategy:
            score = 0.5
            reasons = []
            applicable_patterns = []
            
            weights = self._strategy_weights.get(strategy, {})
            
            for char in analysis.characteristics:
                if char in weights:
                    weight = weights[char]
                    score += weight
                    if weight > 0:
                        reasons.append(f"{char.value}: +{weight:.2f}")
            
            if analysis.complexity_score > 0.7:
                if strategy in [TestStrategy.UNIT_PARAMETERIZED, TestStrategy.PROPERTY_BASED]:
                    score += 0.1
                    reasons.append("High complexity favors comprehensive testing")
            
            if analysis.dependency_count > 3:
                if strategy == TestStrategy.UNIT_MOCK:
                    score += 0.2
                    reasons.append("Multiple dependencies favor mock testing")
            
            if analysis.branch_count > 5:
                if strategy in [TestStrategy.BOUNDARY, TestStrategy.UNIT_PARAMETERIZED]:
                    score += 0.15
                    reasons.append("Many branches favor boundary/parameterized tests")
            
            if preferences:
                preferred = preferences.get("preferred_strategies", [])
                if strategy.value in preferred:
                    score += 0.2
                    reasons.append("User preferred strategy")
                
                avoided = preferences.get("avoided_strategies", [])
                if strategy.value in avoided:
                    score -= 0.3
                    reasons.append("User avoided strategy")
            
            applicable_patterns = self._get_applicable_patterns(strategy)
            
            effort = self._estimate_effort(strategy, analysis)
            
            score = max(0.0, min(1.0, score))
            
            if score > 0.2:
                scores.append(StrategyScore(
                    strategy=strategy,
                    score=score,
                    reasons=reasons,
                    applicable_patterns=applicable_patterns,
                    estimated_effort=effort
                ))
        
        return scores
    
    def _suggest_strategies(self, characteristics: List[CodeCharacteristic]) -> List[TestStrategy]:
        """Suggest strategies based on characteristics."""
        suggestions = []
        
        if CodeCharacteristic.HAS_EXCEPTIONS in characteristics:
            suggestions.append(TestStrategy.EXCEPTION)
        
        if CodeCharacteristic.HAS_DEPENDENCIES in characteristics:
            suggestions.append(TestStrategy.UNIT_MOCK)
        
        if CodeCharacteristic.HAS_BOUNDARY_CONDITIONS in characteristics:
            suggestions.append(TestStrategy.BOUNDARY)
        
        if CodeCharacteristic.HAS_VALIDATION in characteristics:
            suggestions.append(TestStrategy.UNIT_PARAMETERIZED)
        
        if CodeCharacteristic.IS_PURE_FUNCTION in characteristics:
            suggestions.append(TestStrategy.UNIT_BASIC)
            suggestions.append(TestStrategy.PROPERTY_BASED)
        
        if not suggestions:
            suggestions.append(TestStrategy.UNIT_BASIC)
        
        return list(set(suggestions))
    
    def _calculate_complexity(self, source_code: str) -> float:
        """Calculate code complexity score."""
        score = 0.0
        
        lines = source_code.split('\n')
        score += min(len(lines) / 100, 0.2)
        
        score += min(len(re.findall(r'\bif\b', source_code)) * 0.05, 0.2)
        score += min(len(re.findall(r'\bfor\b|\bwhile\b', source_code)) * 0.05, 0.15)
        score += min(len(re.findall(r'\bswitch\b', source_code)) * 0.1, 0.15)
        
        nesting = self._calculate_nesting(source_code)
        score += min(nesting * 0.05, 0.15)
        
        score += min(len(re.findall(r'&&|\|\|', source_code)) * 0.02, 0.15)
        
        return min(score, 1.0)
    
    def _calculate_nesting(self, source_code: str) -> int:
        """Calculate maximum nesting depth."""
        max_nesting = 0
        current_nesting = 0
        
        for char in source_code:
            if char == '{':
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == '}':
                current_nesting -= 1
        
        return max_nesting
    
    def _calculate_confidence(self, scores: List[StrategyScore]) -> float:
        """Calculate confidence in the recommendation."""
        if len(scores) < 2:
            return 0.5
        
        top_score = scores[0].score
        second_score = scores[1].score
        
        gap = top_score - second_score
        
        confidence = 0.5 + gap * 0.5
        
        if top_score > 0.8:
            confidence += 0.1
        elif top_score < 0.5:
            confidence -= 0.1
        
        return max(0.1, min(0.95, confidence))
    
    def _generate_reasoning(self, analysis: CodeAnalysis, top_score: StrategyScore) -> str:
        """Generate human-readable reasoning."""
        parts = []
        
        parts.append(f"Selected {top_score.strategy.value} strategy (score: {top_score.score:.2f})")
        
        if analysis.characteristics:
            chars = [c.value for c in analysis.characteristics[:3]]
            parts.append(f"Key characteristics: {', '.join(chars)}")
        
        if top_score.reasons:
            parts.append("Reasons: " + "; ".join(top_score.reasons[:3]))
        
        parts.append(f"Estimated effort: {top_score.estimated_effort}")
        
        return ". ".join(parts)
    
    def _get_applicable_patterns(self, strategy: TestStrategy) -> List[str]:
        """Get applicable test patterns for a strategy."""
        pattern_map = {
            TestStrategy.UNIT_BASIC: ["basic_test", "aaa_pattern"],
            TestStrategy.UNIT_MOCK: ["mock_test", "mockito_pattern"],
            TestStrategy.UNIT_PARAMETERIZED: ["parameterized_test", "method_source"],
            TestStrategy.INTEGRATION: ["integration_test", "spring_boot_test"],
            TestStrategy.BOUNDARY: ["boundary_test", "edge_case_test"],
            TestStrategy.EXCEPTION: ["exception_test", "assert_throws"],
            TestStrategy.PROPERTY_BASED: ["property_test", "jqwik_pattern"],
            TestStrategy.BEHAVIOR_DRIVEN: ["bdd_test", "given_when_then"],
            TestStrategy.DATA_DRIVEN: ["csv_source_test", "value_source_test"],
            TestStrategy.PERFORMANCE: ["timeout_test", "benchmark_test"],
        }
        return pattern_map.get(strategy, [])
    
    def _estimate_effort(self, strategy: TestStrategy, analysis: CodeAnalysis) -> str:
        """Estimate effort for implementing a strategy."""
        base_effort = {
            TestStrategy.UNIT_BASIC: 1,
            TestStrategy.UNIT_MOCK: 2,
            TestStrategy.UNIT_PARAMETERIZED: 3,
            TestStrategy.INTEGRATION: 4,
            TestStrategy.BOUNDARY: 2,
            TestStrategy.EXCEPTION: 2,
            TestStrategy.PROPERTY_BASED: 4,
            TestStrategy.BEHAVIOR_DRIVEN: 3,
            TestStrategy.DATA_DRIVEN: 3,
            TestStrategy.PERFORMANCE: 4,
        }
        
        effort = base_effort.get(strategy, 2)
        
        if analysis.complexity_score > 0.7:
            effort += 1
        if analysis.dependency_count > 3:
            effort += 1
        if analysis.branch_count > 5:
            effort += 1
        
        if effort <= 1:
            return "low"
        elif effort <= 2:
            return "medium"
        elif effort <= 3:
            return "high"
        else:
            return "very_high"
    
    def _detect_dependencies(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code has dependencies."""
        patterns = [
            r'@Autowired',
            r'@Inject',
            r'private\s+\w+\s+\w+\s*;',
            r'new\s+\w+\s*\(',
        ]
        return any(re.search(p, code) for p in patterns)
    
    def _detect_exceptions(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code throws exceptions."""
        return bool(re.search(r'throws\s+\w+|throw\s+new|try\s*\{', code))
    
    def _detect_conditionals(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code has conditional logic."""
        return bool(re.search(r'\bif\b|\bswitch\b|\bcase\b', code))
    
    def _detect_loops(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code has loops."""
        return bool(re.search(r'\bfor\b|\bwhile\b|\bdo\s*\{', code))
    
    def _detect_collections(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code uses collections."""
        patterns = [
            r'List<', r'Set<', r'Map<', r'Collection<',
            r'ArrayList', r'HashSet', r'HashMap',
            r'\.stream\(\)', r'\.collect\(',
        ]
        return any(re.search(p, code) for p in patterns)
    
    def _detect_external_calls(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code makes external calls."""
        patterns = [
            r'RestTemplate', r'WebClient', r'HttpClient',
            r'@FeignClient', r'JdbcTemplate',
            r'Repository', r'@Repository',
        ]
        return any(re.search(p, code) for p in patterns)
    
    def _detect_stateless(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code is stateless."""
        if re.search(r'private\s+\w+\s+\w+\s*;(?!\s*final)', code):
            return False
        if re.search(r'@Service|@Component|@Repository', code):
            return True
        return bool(re.search(r'static', code))
    
    def _detect_pure_function(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code is a pure function."""
        if re.search(r'void\s+\w+\s*\(', code):
            return False
        if re.search(r'new\s+\w+|\.set\(|\.add\(', code):
            return False
        return bool(re.search(r'static\s+\w+\s+\w+\s*\([^)]*\)\s*\{[^}]*return', code, re.DOTALL))
    
    def _detect_complex_logic(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code has complex logic."""
        conditionals = len(re.findall(r'\bif\b|\bswitch\b', code))
        loops = len(re.findall(r'\bfor\b|\bwhile\b', code))
        operators = len(re.findall(r'&&|\|\|', code))
        
        return (conditionals + loops + operators) > 5
    
    def _detect_validation(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code has validation logic."""
        patterns = [
            r'@Valid', r'@NotNull', r'@NotEmpty', r'@NotBlank',
            r'@Size', r'@Pattern', r'@Min', r'@Max',
            r'validate\(|isValid|checkValid',
            r'if\s*\([^)]*==\s*null\)',
            r'if\s*\([^)]*!=\s*null\)',
        ]
        return any(re.search(p, code) for p in patterns)
    
    def _detect_boundaries(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code has boundary conditions."""
        patterns = [
            r'\bmax\b|\bmin\b|\blimit\b',
            r'>\s*\d+|<\s*\d+|>=\s*\d+|<=\s*\d+',
            r'==\s*0|!=\s*0',
            r'\.length\(\)\s*[<>=]',
            r'\.size\(\)\s*[<>=]',
            r'Integer\.(MAX|MIN)_VALUE',
        ]
        return any(re.search(p, code, re.IGNORECASE) for p in patterns)
    
    def _detect_service_class(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code is a service class."""
        return bool(re.search(r'@Service|class\s+\w*Service', code))
    
    def _detect_utility_class(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code is a utility class."""
        patterns = [
            r'class\s+\w*Utils?\b',
            r'class\s+\w*Helper\b',
            r'private\s+\w*\s*\(\s*\)\s*\{\s*\}',
            r'static\s+\w+\s+\w+\s*\(',
        ]
        return sum(1 for p in patterns if re.search(p, code)) >= 2
    
    def _detect_data_class(self, code: str, class_info: Optional[Dict]) -> bool:
        """Detect if code is a data class."""
        patterns = [
            r'@Data',
            r'@Getter.*@Setter',
            r'class\s+\w*DTO\b',
            r'class\s+\w*Entity\b',
            r'class\s+\w*Model\b',
        ]
        if any(re.search(p, code) for p in patterns):
            return True
        
        getters = len(re.findall(r'get\w+\s*\(', code))
        setters = len(re.findall(r'set\w+\s*\(', code))
        return getters > 2 and setters > 2
