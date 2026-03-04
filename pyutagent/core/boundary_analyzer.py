"""Boundary Value Analyzer for intelligent boundary test case generation.

This module analyzes code to identify boundary conditions and generates
appropriate test cases for edge cases and boundary values.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict

from .component_registry import SimpleComponent, component


logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """Types of boundary conditions."""
    NULL = "null"
    EMPTY = "empty"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    ZERO = "zero"
    NEGATIVE = "negative"
    POSITIVE = "positive"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    LENGTH = "length"
    INDEX = "index"
    SIZE = "size"
    DATE = "date"
    TIME = "time"
    FORMAT = "format"
    RANGE = "range"
    COLLECTION = "collection"


class ParameterType(Enum):
    """Types of parameters."""
    INTEGER = "integer"
    LONG = "long"
    DOUBLE = "double"
    STRING = "string"
    BOOLEAN = "boolean"
    COLLECTION = "collection"
    ARRAY = "array"
    DATE = "date"
    OBJECT = "object"
    ENUM = "enum"


@dataclass
class BoundaryValue:
    """A boundary value for testing."""
    value: Any
    boundary_type: BoundaryType
    description: str
    expected_behavior: str
    is_valid: bool = True
    test_priority: int = 1


@dataclass
class ParameterBoundary:
    """Boundary analysis for a parameter."""
    parameter_name: str
    parameter_type: ParameterType
    boundaries: List[BoundaryValue]
    constraints: List[str]
    suggested_tests: List[str]


@dataclass
class BoundaryAnalysisResult:
    """Result of boundary analysis."""
    method_name: str
    parameters: List[ParameterBoundary]
    total_test_cases: int
    coverage_score: float
    recommendations: List[str]


@component(
    component_id="boundary_analyzer",
    dependencies=[],
    description="Boundary value analyzer for edge case test generation"
)
class BoundaryAnalyzer(SimpleComponent):
    """Analyzer for boundary value testing.
    
    Features:
    - Parameter type detection
    - Boundary value generation
    - Constraint extraction
    - Test case suggestion
    - Coverage analysis
    """
    
    def __init__(self):
        """Initialize the boundary analyzer."""
        super().__init__()
        self._type_handlers = self._initialize_type_handlers()
        self._constraint_patterns = self._initialize_constraint_patterns()
        
        logger.info("[BoundaryAnalyzer] Initialized")
    
    def _initialize_type_handlers(self) -> Dict[ParameterType, callable]:
        """Initialize handlers for different parameter types."""
        return {
            ParameterType.INTEGER: self._generate_integer_boundaries,
            ParameterType.LONG: self._generate_long_boundaries,
            ParameterType.DOUBLE: self._generate_double_boundaries,
            ParameterType.STRING: self._generate_string_boundaries,
            ParameterType.BOOLEAN: self._generate_boolean_boundaries,
            ParameterType.COLLECTION: self._generate_collection_boundaries,
            ParameterType.ARRAY: self._generate_array_boundaries,
            ParameterType.DATE: self._generate_date_boundaries,
            ParameterType.OBJECT: self._generate_object_boundaries,
        }
    
    def _initialize_constraint_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize patterns for constraint detection."""
        return {
            "min": re.compile(r'@Min\s*\(\s*(\d+)\s*\)|min\s*=\s*(\d+)', re.IGNORECASE),
            "max": re.compile(r'@Max\s*\(\s*(\d+)\s*\)|max\s*=\s*(\d+)', re.IGNORECASE),
            "size": re.compile(r'@Size\s*\([^)]*\)|size\s*=\s*(\d+)', re.IGNORECASE),
            "length": re.compile(r'@Length\s*\([^)]*\)|@MaxLength\s*\(\s*(\d+)\s*\)', re.IGNORECASE),
            "pattern": re.compile(r'@Pattern\s*\(\s*regexp\s*=\s*"([^"]+)"', re.IGNORECASE),
            "range": re.compile(r'@Range\s*\([^)]*\)|range\s*=\s*\[?(\d+)\s*,\s*(\d+)\]?', re.IGNORECASE),
            "positive": re.compile(r'@Positive|@PositiveOrZero', re.IGNORECASE),
            "negative": re.compile(r'@Negative|@NegativeOrZero', re.IGNORECASE),
            "not_null": re.compile(r'@NotNull|@NonNull', re.IGNORECASE),
            "not_empty": re.compile(r'@NotEmpty', re.IGNORECASE),
            "not_blank": re.compile(r'@NotBlank', re.IGNORECASE),
        }
    
    def analyze_method(
        self,
        method_signature: str,
        method_body: Optional[str] = None,
        annotations: Optional[List[str]] = None
    ) -> BoundaryAnalysisResult:
        """Analyze a method for boundary conditions.
        
        Args:
            method_signature: Method signature string
            method_body: Optional method body for deeper analysis
            annotations: Optional list of annotations
            
        Returns:
            BoundaryAnalysisResult with analysis details
        """
        parameters = self._extract_parameters(method_signature, annotations or [])
        
        for param in parameters:
            constraints = self._extract_constraints(method_body or "", annotations or [], param.parameter_name)
            param.constraints = constraints
            
            additional_boundaries = self._constraints_to_boundaries(constraints, param.parameter_type)
            param.boundaries.extend(additional_boundaries)
            
            param.suggested_tests = self._generate_test_suggestions(param)
        
        total_cases = sum(len(p.suggested_tests) for p in parameters)
        coverage_score = self._calculate_coverage_score(parameters)
        recommendations = self._generate_recommendations(parameters)
        
        method_name = self._extract_method_name(method_signature)
        
        return BoundaryAnalysisResult(
            method_name=method_name,
            parameters=parameters,
            total_test_cases=total_cases,
            coverage_score=coverage_score,
            recommendations=recommendations
        )
    
    def analyze_class(
        self,
        source_code: str
    ) -> Dict[str, BoundaryAnalysisResult]:
        """Analyze all methods in a class.
        
        Args:
            source_code: Java source code
            
        Returns:
            Dictionary mapping method names to analysis results
        """
        results = {}
        
        method_pattern = re.compile(
            r'(?:@[\w]+\s*\([^)]*\)\s*)*'  
            r'(?:public|private|protected)?\s*'
            r'(?:static)?\s*'
            r'(\w+(?:<[\w<>,\s]+>)?)\s+'
            r'(\w+)\s*\(([^)]*)\)',
            re.MULTILINE
        )
        
        for match in method_pattern.finditer(source_code):
            return_type = match.group(1)
            method_name = match.group(2)
            params = match.group(3)
            
            if method_name in ('equals', 'hashCode', 'toString', 'clone'):
                continue
            
            start = match.start()
            annotations = self._extract_annotations_before(source_code, start)
            
            method_body = self._extract_method_body(source_code, match.end())
            
            signature = f"{return_type} {method_name}({params})"
            
            result = self.analyze_method(signature, method_body, annotations)
            results[method_name] = result
        
        return results
    
    def generate_boundary_tests(
        self,
        analysis: BoundaryAnalysisResult,
        test_class_name: str = "BoundaryTest"
    ) -> str:
        """Generate boundary test cases from analysis.
        
        Args:
            analysis: Boundary analysis result
            test_class_name: Name for the test class
            
        Returns:
            Generated test code
        """
        lines = [
            f"class {test_class_name} {{",
            "",
        ]
        
        for param in analysis.parameters:
            for i, boundary in enumerate(param.boundaries[:5]):
                test_name = f"should_handle_{boundary.boundary_type.value}_for_{param.parameter_name}"
                
                lines.append(f"    @Test")
                lines.append(f"    @DisplayName(\"should handle {boundary.description}\")")
                lines.append(f"    void {test_name}_{i}() {{")
                lines.append(f"        // Given: {boundary.description}")
                lines.append(f"        // {param.parameter_name} = {self._format_value(boundary.value)}")
                lines.append(f"        // Expected: {boundary.expected_behavior}")
                lines.append(f"        // TODO: Implement test")
                lines.append(f"    }}")
                lines.append("")
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def _extract_parameters(
        self,
        method_signature: str,
        annotations: List[str]
    ) -> List[ParameterBoundary]:
        """Extract parameters from method signature."""
        parameters = []
        
        match = re.search(r'\(([^)]*)\)', method_signature)
        if not match:
            return parameters
        
        params_str = match.group(1)
        if not params_str.strip():
            return parameters
        
        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue
            
            parts = param.split()
            if len(parts) >= 2:
                param_type = parts[-2]
                param_name = parts[-1]
                
                if '<' in param_type:
                    base_type = param_type.split('<')[0]
                else:
                    base_type = param_type
                
                ptype = self._detect_parameter_type(base_type)
                
                boundaries = self._generate_boundaries(ptype)
                
                parameters.append(ParameterBoundary(
                    parameter_name=param_name,
                    parameter_type=ptype,
                    boundaries=boundaries,
                    constraints=[],
                    suggested_tests=[]
                ))
        
        return parameters
    
    def _detect_parameter_type(self, type_str: str) -> ParameterType:
        """Detect parameter type from string."""
        type_lower = type_str.lower()
        
        if type_lower in ('int', 'integer'):
            return ParameterType.INTEGER
        elif type_lower == 'long':
            return ParameterType.LONG
        elif type_lower in ('double', 'float'):
            return ParameterType.DOUBLE
        elif type_lower in ('string', 'charsequence'):
            return ParameterType.STRING
        elif type_lower == 'boolean':
            return ParameterType.BOOLEAN
        elif type_lower in ('list', 'set', 'collection', 'map'):
            return ParameterType.COLLECTION
        elif type_lower.endswith('[]'):
            return ParameterType.ARRAY
        elif 'date' in type_lower or 'localdate' in type_lower or 'localdatetime' in type_lower:
            return ParameterType.DATE
        elif type_lower in ('object', 'any'):
            return ParameterType.OBJECT
        else:
            return ParameterType.OBJECT
    
    def _generate_boundaries(self, param_type: ParameterType) -> List[BoundaryValue]:
        """Generate boundary values for a parameter type."""
        handler = self._type_handlers.get(param_type, self._generate_object_boundaries)
        return handler()
    
    def _generate_integer_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for integer type."""
        return [
            BoundaryValue(0, BoundaryType.ZERO, "zero value", "Should handle zero correctly", True, 1),
            BoundaryValue(1, BoundaryType.POSITIVE, "positive value", "Should handle positive value", True, 2),
            BoundaryValue(-1, BoundaryType.NEGATIVE, "negative value", "Should handle negative value", True, 2),
            BoundaryValue(2147483647, BoundaryType.MAX_VALUE, "Integer.MAX_VALUE", "Should handle max int", True, 1),
            BoundaryValue(-2147483648, BoundaryType.MIN_VALUE, "Integer.MIN_VALUE", "Should handle min int", True, 1),
            BoundaryValue(None, BoundaryType.NULL, "null value", "Should throw exception or handle gracefully", False, 1),
        ]
    
    def _generate_long_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for long type."""
        return [
            BoundaryValue(0, BoundaryType.ZERO, "zero value", "Should handle zero correctly", True, 1),
            BoundaryValue(1, BoundaryType.POSITIVE, "positive value", "Should handle positive value", True, 2),
            BoundaryValue(-1, BoundaryType.NEGATIVE, "negative value", "Should handle negative value", True, 2),
            BoundaryValue(9223372036854775807, BoundaryType.MAX_VALUE, "Long.MAX_VALUE", "Should handle max long", True, 1),
            BoundaryValue(-9223372036854775808, BoundaryType.MIN_VALUE, "Long.MIN_VALUE", "Should handle min long", True, 1),
            BoundaryValue(None, BoundaryType.NULL, "null value", "Should throw exception or handle gracefully", False, 1),
        ]
    
    def _generate_double_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for double type."""
        return [
            BoundaryValue(0.0, BoundaryType.ZERO, "zero value", "Should handle zero correctly", True, 1),
            BoundaryValue(0.000001, BoundaryType.POSITIVE, "very small positive", "Should handle small decimals", True, 2),
            BoundaryValue(-0.000001, BoundaryType.NEGATIVE, "very small negative", "Should handle small negatives", True, 2),
            BoundaryValue(float('inf'), BoundaryType.OVERFLOW, "positive infinity", "Should handle infinity", True, 1),
            BoundaryValue(float('-inf'), BoundaryType.UNDERFLOW, "negative infinity", "Should handle negative infinity", True, 1),
            BoundaryValue(float('nan'), BoundaryType.FORMAT, "NaN", "Should handle NaN", True, 1),
            BoundaryValue(None, BoundaryType.NULL, "null value", "Should throw exception or handle gracefully", False, 1),
        ]
    
    def _generate_string_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for string type."""
        return [
            BoundaryValue("", BoundaryType.EMPTY, "empty string", "Should handle empty string", True, 1),
            BoundaryValue(" ", BoundaryType.FORMAT, "whitespace only", "Should handle whitespace", True, 2),
            BoundaryValue("a", BoundaryType.MIN_VALUE, "single character", "Should handle single char", True, 2),
            BoundaryValue("a" * 1000, BoundaryType.MAX_VALUE, "very long string", "Should handle long strings", True, 2),
            BoundaryValue(None, BoundaryType.NULL, "null string", "Should throw exception or handle gracefully", False, 1),
            BoundaryValue("特殊字符!@#$%", BoundaryType.FORMAT, "special characters", "Should handle special chars", True, 3),
            BoundaryValue("<script>alert('xss')</script>", BoundaryType.FORMAT, "potential XSS", "Should sanitize input", False, 2),
        ]
    
    def _generate_boolean_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for boolean type."""
        return [
            BoundaryValue(True, BoundaryType.POSITIVE, "true value", "Should handle true", True, 1),
            BoundaryValue(False, BoundaryType.NEGATIVE, "false value", "Should handle false", True, 1),
            BoundaryValue(None, BoundaryType.NULL, "null value", "Should throw exception or handle gracefully", False, 1),
        ]
    
    def _generate_collection_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for collection type."""
        return [
            BoundaryValue([], BoundaryType.EMPTY, "empty collection", "Should handle empty collection", True, 1),
            BoundaryValue([None], BoundaryType.NULL, "collection with null", "Should handle null elements", True, 2),
            BoundaryValue([1], BoundaryType.MIN_VALUE, "single element", "Should handle single element", True, 2),
            BoundaryValue(list(range(1000)), BoundaryType.MAX_VALUE, "large collection", "Should handle large collections", True, 2),
            BoundaryValue(None, BoundaryType.NULL, "null collection", "Should throw exception or handle gracefully", False, 1),
        ]
    
    def _generate_array_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for array type."""
        return [
            BoundaryValue([], BoundaryType.EMPTY, "empty array", "Should handle empty array", True, 1),
            BoundaryValue([None], BoundaryType.NULL, "array with null", "Should handle null elements", True, 2),
            BoundaryValue([1], BoundaryType.MIN_VALUE, "single element array", "Should handle single element", True, 2),
            BoundaryValue([0] * 1000, BoundaryType.MAX_VALUE, "large array", "Should handle large arrays", True, 2),
            BoundaryValue(None, BoundaryType.NULL, "null array", "Should throw exception or handle gracefully", False, 1),
        ]
    
    def _generate_date_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for date type."""
        return [
            BoundaryValue("1970-01-01", BoundaryType.MIN_VALUE, "epoch date", "Should handle epoch", True, 1),
            BoundaryValue("2099-12-31", BoundaryType.MAX_VALUE, "far future date", "Should handle future dates", True, 2),
            BoundaryValue("2000-02-29", BoundaryType.FORMAT, "leap year date", "Should handle leap years", True, 2),
            BoundaryValue(None, BoundaryType.NULL, "null date", "Should throw exception or handle gracefully", False, 1),
            BoundaryValue("invalid-date", BoundaryType.FORMAT, "invalid format", "Should throw exception", False, 1),
        ]
    
    def _generate_object_boundaries(self) -> List[BoundaryValue]:
        """Generate boundary values for generic object type."""
        return [
            BoundaryValue(None, BoundaryType.NULL, "null object", "Should throw exception or handle gracefully", False, 1),
            BoundaryValue("valid_object", BoundaryType.POSITIVE, "valid object", "Should handle valid object", True, 1),
        ]
    
    def _extract_constraints(
        self,
        method_body: str,
        annotations: List[str],
        param_name: str
    ) -> List[str]:
        """Extract constraints from method body and annotations."""
        constraints = []
        
        all_text = method_body + " ".join(annotations)
        
        for constraint_name, pattern in self._constraint_patterns.items():
            if pattern.search(all_text):
                constraints.append(constraint_name)
        
        if re.search(rf'{param_name}\s*==\s*null|{param_name}\s*!=\s*null', all_text):
            constraints.append("null_check")
        
        if re.search(rf'{param_name}\.isEmpty\(\)|{param_name}\.length\(\)\s*==\s*0', all_text):
            constraints.append("empty_check")
        
        return constraints
    
    def _constraints_to_boundaries(
        self,
        constraints: List[str],
        param_type: ParameterType
    ) -> List[BoundaryValue]:
        """Convert constraints to boundary values."""
        boundaries = []
        
        if "not_null" in constraints:
            boundaries.append(BoundaryValue(
                None, BoundaryType.NULL, "null (violates @NotNull)",
                "Should throw validation error", False, 1
            ))
        
        if "not_empty" in constraints or "not_blank" in constraints:
            boundaries.append(BoundaryValue(
                "", BoundaryType.EMPTY, "empty (violates @NotEmpty)",
                "Should throw validation error", False, 1
            ))
        
        if "positive" in constraints:
            boundaries.append(BoundaryValue(
                -1, BoundaryType.NEGATIVE, "negative (violates @Positive)",
                "Should throw validation error", False, 1
            ))
        
        if "negative" in constraints:
            boundaries.append(BoundaryValue(
                1, BoundaryType.POSITIVE, "positive (violates @Negative)",
                "Should throw validation error", False, 1
            ))
        
        return boundaries
    
    def _generate_test_suggestions(self, param: ParameterBoundary) -> List[str]:
        """Generate test suggestions for a parameter."""
        suggestions = []
        
        for boundary in param.boundaries:
            if boundary.test_priority <= 2:
                suggestion = f"Test {param.parameter_name} with {boundary.description}"
                if not boundary.is_valid:
                    suggestion += " (expect exception or error)"
                suggestions.append(suggestion)
        
        if param.constraints:
            suggestions.append(f"Validate constraints: {', '.join(param.constraints)}")
        
        return suggestions
    
    def _calculate_coverage_score(self, parameters: List[ParameterBoundary]) -> float:
        """Calculate boundary coverage score."""
        if not parameters:
            return 0.0
        
        total_boundaries = 0
        covered_boundaries = 0
        
        for param in parameters:
            for boundary in param.boundaries:
                total_boundaries += 1
                if boundary.test_priority <= 2:
                    covered_boundaries += 1
        
        return covered_boundaries / total_boundaries if total_boundaries > 0 else 0.0
    
    def _generate_recommendations(self, parameters: List[ParameterBoundary]) -> List[str]:
        """Generate recommendations for boundary testing."""
        recommendations = []
        
        has_null_boundary = any(
            any(b.boundary_type == BoundaryType.NULL for b in p.boundaries)
            for p in parameters
        )
        if has_null_boundary:
            recommendations.append("Add null safety tests for all nullable parameters")
        
        has_empty_boundary = any(
            any(b.boundary_type == BoundaryType.EMPTY for b in p.boundaries)
            for p in parameters
        )
        if has_empty_boundary:
            recommendations.append("Add empty collection/string tests")
        
        numeric_params = [p for p in parameters if p.parameter_type in 
                         (ParameterType.INTEGER, ParameterType.LONG, ParameterType.DOUBLE)]
        if numeric_params:
            recommendations.append("Add boundary value tests for numeric parameters (min, max, zero)")
        
        string_params = [p for p in parameters if p.parameter_type == ParameterType.STRING]
        if string_params:
            recommendations.append("Add string validation tests (empty, whitespace, special chars)")
        
        return recommendations
    
    def _extract_method_name(self, signature: str) -> str:
        """Extract method name from signature."""
        match = re.search(r'(\w+)\s*\(', signature)
        return match.group(1) if match else "unknown"
    
    def _extract_annotations_before(self, source_code: str, position: int) -> List[str]:
        """Extract annotations before a position in source code."""
        before = source_code[:position]
        return re.findall(r'@\w+(?:\([^)]*\))?', before)
    
    def _extract_method_body(self, source_code: str, start: int) -> str:
        """Extract method body from source code."""
        brace_count = 0
        body_start = None
        
        for i, char in enumerate(source_code[start:], start):
            if char == '{':
                if body_start is None:
                    body_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and body_start is not None:
                    return source_code[body_start:i+1]
        
        return ""
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "null"
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, list):
            return f"[{len(value)} elements]"
        else:
            return str(value)
