"""Unit tests for BoundaryAnalyzer module."""

import pytest

from pyutagent.core.boundary_analyzer import (
    BoundaryAnalyzer,
    BoundaryType,
    ParameterType,
    BoundaryValue,
    ParameterBoundary,
    BoundaryAnalysisResult,
)


class TestBoundaryAnalyzer:
    """Tests for BoundaryAnalyzer class."""

    def test_init(self):
        """Test initialization."""
        analyzer = BoundaryAnalyzer()
        
        assert analyzer._type_handlers is not None
        assert analyzer._constraint_patterns is not None

    def test_analyze_method_simple(self):
        """Test analyzing a simple method."""
        analyzer = BoundaryAnalyzer()
        
        method_signature = "public int calculate(int value)"
        
        result = analyzer.analyze_method(method_signature)
        
        assert isinstance(result, BoundaryAnalysisResult)
        assert result.method_name == "calculate"
        assert len(result.parameters) == 1
        assert result.parameters[0].parameter_type == ParameterType.INTEGER

    def test_analyze_method_with_multiple_params(self):
        """Test analyzing method with multiple parameters."""
        analyzer = BoundaryAnalyzer()
        
        method_signature = "public void process(String name, int age, boolean active)"
        
        result = analyzer.analyze_method(method_signature)
        
        assert len(result.parameters) == 3
        param_types = [p.parameter_type for p in result.parameters]
        assert ParameterType.STRING in param_types
        assert ParameterType.INTEGER in param_types
        assert ParameterType.BOOLEAN in param_types

    def test_analyze_method_with_annotations(self):
        """Test analyzing method with validation annotations."""
        analyzer = BoundaryAnalyzer()
        
        method_signature = "public void setUser(@NotNull String name, @Min(0) @Max(150) int age)"
        annotations = ["@NotNull", "@Min(0)", "@Max(150)"]
        
        result = analyzer.analyze_method(method_signature, annotations=annotations)
        
        assert len(result.parameters) == 2

    def test_analyze_class(self):
        """Test analyzing a class."""
        analyzer = BoundaryAnalyzer()
        
        source_code = '''
public class UserService {
    public User findById(Long id) {
        return repository.findById(id);
    }
    
    public List<User> findByName(String name) {
        return repository.findByName(name);
    }
    
    public void updateUser(User user) {
        repository.save(user);
    }
}
'''
        
        results = analyzer.analyze_class(source_code)
        
        assert "findById" in results
        assert "findByName" in results
        assert "updateUser" in results

    def test_generate_integer_boundaries(self):
        """Test generating integer boundaries."""
        analyzer = BoundaryAnalyzer()
        
        boundaries = analyzer._generate_integer_boundaries()
        
        boundary_types = [b.boundary_type for b in boundaries]
        assert BoundaryType.ZERO in boundary_types
        assert BoundaryType.POSITIVE in boundary_types
        assert BoundaryType.NEGATIVE in boundary_types
        assert BoundaryType.MAX_VALUE in boundary_types
        assert BoundaryType.MIN_VALUE in boundary_types
        assert BoundaryType.NULL in boundary_types

    def test_generate_string_boundaries(self):
        """Test generating string boundaries."""
        analyzer = BoundaryAnalyzer()
        
        boundaries = analyzer._generate_string_boundaries()
        
        boundary_types = [b.boundary_type for b in boundaries]
        assert BoundaryType.EMPTY in boundary_types
        assert BoundaryType.NULL in boundary_types
        assert any(b.boundary_type == BoundaryType.FORMAT for b in boundaries)

    def test_generate_boolean_boundaries(self):
        """Test generating boolean boundaries."""
        analyzer = BoundaryAnalyzer()
        
        boundaries = analyzer._generate_boolean_boundaries()
        
        values = [b.value for b in boundaries]
        assert True in values
        assert False in values

    def test_generate_collection_boundaries(self):
        """Test generating collection boundaries."""
        analyzer = BoundaryAnalyzer()
        
        boundaries = analyzer._generate_collection_boundaries()
        
        boundary_types = [b.boundary_type for b in boundaries]
        assert BoundaryType.EMPTY in boundary_types
        assert BoundaryType.NULL in boundary_types

    def test_detect_parameter_type_integer(self):
        """Test detecting integer parameter type."""
        analyzer = BoundaryAnalyzer()
        
        assert analyzer._detect_parameter_type("int") == ParameterType.INTEGER
        assert analyzer._detect_parameter_type("Integer") == ParameterType.INTEGER

    def test_detect_parameter_type_string(self):
        """Test detecting string parameter type."""
        analyzer = BoundaryAnalyzer()
        
        assert analyzer._detect_parameter_type("String") == ParameterType.STRING
        assert analyzer._detect_parameter_type("CharSequence") == ParameterType.STRING

    def test_detect_parameter_type_collection(self):
        """Test detecting collection parameter type."""
        analyzer = BoundaryAnalyzer()
        
        assert analyzer._detect_parameter_type("List") == ParameterType.COLLECTION
        assert analyzer._detect_parameter_type("Set") == ParameterType.COLLECTION
        assert analyzer._detect_parameter_type("ArrayList") == ParameterType.COLLECTION

    def test_detect_parameter_type_date(self):
        """Test detecting date parameter type."""
        analyzer = BoundaryAnalyzer()
        
        assert analyzer._detect_parameter_type("LocalDate") == ParameterType.DATE
        assert analyzer._detect_parameter_type("Date") == ParameterType.DATE
        assert analyzer._detect_parameter_type("LocalDateTime") == ParameterType.DATE

    def test_extract_constraints_not_null(self):
        """Test extracting NotNull constraint."""
        analyzer = BoundaryAnalyzer()
        
        constraints = analyzer._extract_constraints(["@NotNull", "@NonNull"])
        
        assert "not_null" in constraints

    def test_extract_constraints_size(self):
        """Test extracting Size constraint."""
        analyzer = BoundaryAnalyzer()
        
        constraints = analyzer._extract_constraints(["@Size(min=1, max=100)"])
        
        assert "size" in constraints

    def test_extract_constraints_min_max(self):
        """Test extracting Min/Max constraints."""
        analyzer = BoundaryAnalyzer()
        
        constraints = analyzer._extract_constraints(["@Min(0)", "@Max(100)"])
        
        assert "min" in constraints
        assert "max" in constraints

    def test_extract_constraints_pattern(self):
        """Test extracting Pattern constraint."""
        analyzer = BoundaryAnalyzer()
        
        constraints = analyzer._extract_constraints(["@Pattern(regexp=\"[a-z]+\")"])
        
        assert "pattern" in constraints

    def test_constraints_to_boundaries_not_null(self):
        """Test converting NotNull constraint to boundaries."""
        analyzer = BoundaryAnalyzer()
        
        boundaries = analyzer._constraints_to_boundaries(["not_null"], ParameterType.STRING)
        
        null_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.NULL]
        assert len(null_boundaries) > 0
        assert not null_boundaries[0].is_valid

    def test_constraints_to_boundaries_positive(self):
        """Test converting Positive constraint to boundaries."""
        analyzer = BoundaryAnalyzer()
        
        boundaries = analyzer._constraints_to_boundaries(["positive"], ParameterType.INTEGER)
        
        negative_boundaries = [b for b in boundaries if b.boundary_type == BoundaryType.NEGATIVE]
        assert len(negative_boundaries) > 0
        assert not negative_boundaries[0].is_valid

    def test_generate_test_suggestions(self):
        """Test generating test suggestions."""
        analyzer = BoundaryAnalyzer()
        
        param = ParameterBoundary(
            parameter_name="age",
            parameter_type=ParameterType.INTEGER,
            boundaries=[
                BoundaryValue(0, BoundaryType.ZERO, "zero", "Should handle zero"),
                BoundaryValue(None, BoundaryType.NULL, "null", "Should throw exception", is_valid=False),
            ],
            constraints=["not_null"]
        )
        
        suggestions = analyzer._generate_test_suggestions(param)
        
        assert len(suggestions) > 0

    def test_calculate_coverage_score(self):
        """Test calculating coverage score."""
        analyzer = BoundaryAnalyzer()
        
        parameters = [
            ParameterBoundary(
                parameter_name="a",
                parameter_type=ParameterType.INTEGER,
                boundaries=[
                    BoundaryValue(0, BoundaryType.ZERO, "zero", "", test_priority=1),
                    BoundaryValue(1, BoundaryType.POSITIVE, "positive", "", test_priority=2),
                ],
                constraints=[],
                suggested_tests=[]
            )
        ]
        
        score = analyzer._calculate_coverage_score(parameters)
        
        assert 0.0 <= score <= 1.0

    def test_generate_recommendations(self):
        """Test generating recommendations."""
        analyzer = BoundaryAnalyzer()
        
        parameters = [
            ParameterBoundary(
                parameter_name="name",
                parameter_type=ParameterType.STRING,
                boundaries=[BoundaryValue(None, BoundaryType.NULL, "null", "")],
                constraints=[],
                suggested_tests=[]
            ),
            ParameterBoundary(
                parameter_name="age",
                parameter_type=ParameterType.INTEGER,
                boundaries=[BoundaryValue(-1, BoundaryType.NEGATIVE, "negative", "")],
                constraints=[],
                suggested_tests=[]
            )
        ]
        
        recommendations = analyzer._generate_recommendations(parameters)
        
        assert len(recommendations) > 0

    def test_extract_method_name(self):
        """Test extracting method name from signature."""
        analyzer = BoundaryAnalyzer()
        
        assert analyzer._extract_method_name("public void testMethod()") == "testMethod"
        assert analyzer._extract_method_name("private int calculate(int a)") == "calculate"

    def test_format_value(self):
        """Test formatting values for display."""
        analyzer = BoundaryAnalyzer()
        
        assert analyzer._format_value(None) == "null"
        assert analyzer._format_value("test") == '"test"'
        assert analyzer._format_value(True) == "true"
        assert analyzer._format_value(False) == "false"
        assert analyzer._format_value([1, 2, 3]) == "[3 elements]"


class TestBoundaryValue:
    """Tests for BoundaryValue dataclass."""

    def test_boundary_value_creation(self):
        """Test boundary value creation."""
        bv = BoundaryValue(
            value=0,
            boundary_type=BoundaryType.ZERO,
            description="Zero value",
            expected_behavior="Should handle zero correctly",
            is_valid=True,
            test_priority=1
        )
        
        assert bv.value == 0
        assert bv.boundary_type == BoundaryType.ZERO
        assert bv.is_valid is True
        assert bv.test_priority == 1

    def test_boundary_value_invalid(self):
        """Test invalid boundary value."""
        bv = BoundaryValue(
            value=None,
            boundary_type=BoundaryType.NULL,
            description="Null value",
            expected_behavior="Should throw exception",
            is_valid=False,
            test_priority=1
        )
        
        assert bv.is_valid is False


class TestParameterBoundary:
    """Tests for ParameterBoundary dataclass."""

    def test_parameter_boundary_creation(self):
        """Test parameter boundary creation."""
        pb = ParameterBoundary(
            parameter_name="age",
            parameter_type=ParameterType.INTEGER,
            boundaries=[
                BoundaryValue(0, BoundaryType.ZERO, "zero", ""),
            ],
            constraints=["min=0", "max=150"],
            suggested_tests=["Test with zero", "Test with negative"]
        )
        
        assert pb.parameter_name == "age"
        assert pb.parameter_type == ParameterType.INTEGER
        assert len(pb.boundaries) == 1
        assert len(pb.constraints) == 2


class TestBoundaryAnalysisResult:
    """Tests for BoundaryAnalysisResult dataclass."""

    def test_result_creation(self):
        """Test analysis result creation."""
        result = BoundaryAnalysisResult(
            method_name="calculate",
            parameters=[
                ParameterBoundary("a", ParameterType.INTEGER, [], [], [])
            ],
            total_test_cases=5,
            coverage_score=0.8,
            recommendations=["Add null test"]
        )
        
        assert result.method_name == "calculate"
        assert result.total_test_cases == 5
        assert result.coverage_score == 0.8


class TestBoundaryType:
    """Tests for BoundaryType enum."""

    def test_boundary_type_values(self):
        """Test boundary type enum values."""
        assert BoundaryType.NULL.value == "null"
        assert BoundaryType.EMPTY.value == "empty"
        assert BoundaryType.ZERO.value == "zero"
        assert BoundaryType.MAX_VALUE.value == "max_value"
        assert BoundaryType.MIN_VALUE.value == "min_value"


class TestParameterType:
    """Tests for ParameterType enum."""

    def test_parameter_type_values(self):
        """Test parameter type enum values."""
        assert ParameterType.INTEGER.value == "integer"
        assert ParameterType.STRING.value == "string"
        assert ParameterType.BOOLEAN.value == "boolean"
        assert ParameterType.COLLECTION.value == "collection"
        assert ParameterType.DATE.value == "date"
