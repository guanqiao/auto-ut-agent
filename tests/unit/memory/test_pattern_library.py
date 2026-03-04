"""Unit tests for PatternLibrary module."""

import pytest
import tempfile
import os

from pyutagent.memory.pattern_library import (
    PatternLibrary,
    TestPattern,
    PatternCategory,
    PatternComplexity,
    PatternMatch,
)


class TestPatternLibrary:
    """Tests for PatternLibrary class."""

    def test_init(self):
        """Test initialization."""
        library = PatternLibrary()
        
        assert library.db_path is not None
        assert library._cache is not None

    def test_init_custom_db(self):
        """Test initialization with custom database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_patterns.db")
            library = PatternLibrary(db_path=db_path)
            
            assert library.db_path == db_path

    def test_builtin_patterns_loaded(self):
        """Test that built-in patterns are loaded."""
        library = PatternLibrary()
        
        basic_pattern = library.get_pattern("builtin_basic_test")
        
        assert basic_pattern is not None
        assert basic_pattern.name == "Basic Test Method"

    def test_add_pattern(self):
        """Test adding a pattern."""
        library = PatternLibrary()
        
        pattern = TestPattern(
            pattern_id="custom_test_123",
            name="Custom Test Pattern",
            category=PatternCategory.TEST_STRUCTURE,
            description="A custom test pattern",
            template="@Test void test() { }",
            placeholders=[]
        )
        
        pattern_id = library.add_pattern(pattern)
        
        assert pattern_id == "custom_test_123"
        retrieved = library.get_pattern("custom_test_123")
        assert retrieved is not None

    def test_get_pattern_existing(self):
        """Test getting existing pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_basic_test")
        
        assert pattern is not None
        assert pattern.category == PatternCategory.TEST_STRUCTURE

    def test_get_pattern_non_existing(self):
        """Test getting non-existing pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("non_existing_pattern")
        
        assert pattern is None

    def test_find_patterns_by_category(self):
        """Test finding patterns by category."""
        library = PatternLibrary()
        
        patterns = library.find_patterns(category=PatternCategory.MOCK)
        
        assert len(patterns) > 0
        for p in patterns:
            assert p.category == PatternCategory.MOCK

    def test_find_patterns_by_tags(self):
        """Test finding patterns by tags."""
        library = PatternLibrary()
        
        patterns = library.find_patterns(tags=["mock", "mockito"])
        
        assert len(patterns) >= 0

    def test_find_patterns_by_complexity(self):
        """Test finding patterns by complexity."""
        library = PatternLibrary()
        
        patterns = library.find_patterns(complexity=PatternComplexity.SIMPLE)
        
        for p in patterns:
            assert p.complexity == PatternComplexity.SIMPLE

    def test_find_patterns_min_success_rate(self):
        """Test finding patterns with minimum success rate."""
        library = PatternLibrary()
        
        patterns = library.find_patterns(min_success_rate=0.7)
        
        for p in patterns:
            assert p.success_rate >= 0.7

    def test_match_pattern(self):
        """Test matching patterns against code."""
        library = PatternLibrary()
        
        code = '''
@Test
@DisplayName("should process data")
void shouldProcessData() {
    // Given
    Data data = new Data();
    
    // When
    Result result = processor.process(data);
    
    // Then
    assertNotNull(result);
}
'''
        
        matches = library.match_pattern(code)
        
        assert len(matches) > 0
        assert all(m.confidence > 0 for m in matches)

    def test_match_pattern_with_category_filter(self):
        """Test matching patterns with category filter."""
        library = PatternLibrary()
        
        code = '''
@Mock
private UserService userService;

@Test
void testWithMock() {
    when(userService.getUser()).thenReturn(new User());
}
'''
        
        matches = library.match_pattern(code, category=PatternCategory.MOCK)
        
        for m in matches:
            assert m.pattern.category == PatternCategory.MOCK

    def test_recommend_patterns(self):
        """Test recommending patterns."""
        library = PatternLibrary()
        
        context = {
            "has_dependencies": True,
            "has_exceptions": False,
            "complexity": "moderate",
            "tags": ["mock"]
        }
        
        recommendations = library.recommend_patterns(context, limit=5)
        
        assert len(recommendations) <= 5
        for pattern, score in recommendations:
            assert score >= 0

    def test_record_usage(self):
        """Test recording pattern usage."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_basic_test")
        if pattern:
            initial_count = pattern.usage_count
            
            library.record_usage("builtin_basic_test", success=True)
            
            updated = library.get_pattern("builtin_basic_test")
            assert updated.usage_count > initial_count

    def test_get_statistics(self):
        """Test getting statistics."""
        library = PatternLibrary()
        
        stats = library.get_statistics()
        
        assert "total_patterns" in stats
        assert "category_distribution" in stats
        assert stats["total_patterns"] > 0

    def test_create_custom_pattern(self):
        """Test creating custom pattern."""
        library = PatternLibrary()
        
        pattern = library.create_custom_pattern(
            name="My Custom Pattern",
            category=PatternCategory.TEST_STRUCTURE,
            description="A custom test pattern",
            template="@Test void {method_name}() { {body} }",
            tags=["custom", "test"],
            complexity=PatternComplexity.SIMPLE
        )
        
        assert pattern is not None
        assert pattern.name == "My Custom Pattern"
        assert "method_name" in pattern.placeholders
        assert "body" in pattern.placeholders


class TestTestPattern:
    """Tests for TestPattern dataclass."""

    def test_pattern_creation(self):
        """Test pattern creation."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test Pattern",
            category=PatternCategory.TEST_STRUCTURE,
            description="A test pattern",
            template="@Test void test() {}",
            placeholders=[],
            example_usage="Example usage",
            complexity=PatternComplexity.SIMPLE,
            tags=["test"],
            success_rate=0.9
        )
        
        assert pattern.pattern_id == "test-123"
        assert pattern.category == PatternCategory.TEST_STRUCTURE
        assert pattern.success_rate == 0.9

    def test_pattern_render(self):
        """Test pattern rendering."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test",
            category=PatternCategory.TEST_STRUCTURE,
            description="Test",
            template="Hello {name}, welcome to {place}!",
            placeholders=["name", "place"]
        )
        
        rendered = pattern.render({"name": "Alice", "place": "Wonderland"})
        
        assert rendered == "Hello Alice, welcome to Wonderland!"

    def test_pattern_render_missing_placeholder(self):
        """Test pattern rendering with missing placeholder."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test",
            category=PatternCategory.TEST_STRUCTURE,
            description="Test",
            template="Hello {name} from {place}",
            placeholders=["name", "place"]
        )
        
        rendered = pattern.render({"name": "Bob"})
        
        assert "Bob" in rendered
        assert "{place}" in rendered

    def test_pattern_to_dict(self):
        """Test pattern to dictionary conversion."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test",
            category=PatternCategory.MOCK,
            description="Test",
            template="Template",
            placeholders=["var"],
            complexity=PatternComplexity.COMPLEX,
            tags=["mock"]
        )
        
        d = pattern.to_dict()
        
        assert d["pattern_id"] == "test-123"
        assert d["category"] == "mock"
        assert d["complexity"] == "complex"

    def test_pattern_from_dict(self):
        """Test pattern from dictionary."""
        d = {
            "pattern_id": "test-123",
            "name": "Test",
            "category": "test_structure",
            "description": "Test",
            "template": "Template",
            "placeholders": ["var"],
            "example_usage": "Example",
            "complexity": "moderate",
            "tags": ["test"],
            "prerequisites": [],
            "success_rate": 0.8,
            "usage_count": 5,
            "created_at": "2024-01-01T00:00:00"
        }
        
        pattern = TestPattern.from_dict(d)
        
        assert pattern.pattern_id == "test-123"
        assert pattern.category == PatternCategory.TEST_STRUCTURE
        assert pattern.complexity == PatternComplexity.MODERATE


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_match_creation(self):
        """Test match creation."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test",
            category=PatternCategory.TEST_STRUCTURE,
            description="Test",
            template="Template",
            placeholders=[]
        )
        
        match = PatternMatch(
            pattern=pattern,
            confidence=0.85,
            matched_elements=["method_name", "class_name"],
            suggested_values={"method_name": "testMethod"},
            missing_elements=["description"]
        )
        
        assert match.confidence == 0.85
        assert len(match.matched_elements) == 2


class TestPatternCategory:
    """Tests for PatternCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert PatternCategory.TEST_STRUCTURE.value == "test_structure"
        assert PatternCategory.ASSERTION.value == "assertion"
        assert PatternCategory.MOCK.value == "mock"
        assert PatternCategory.EXCEPTION.value == "exception"
        assert PatternCategory.BOUNDARY.value == "boundary"


class TestPatternComplexity:
    """Tests for PatternComplexity enum."""

    def test_complexity_values(self):
        """Test complexity enum values."""
        assert PatternComplexity.SIMPLE.value == "simple"
        assert PatternComplexity.MODERATE.value == "moderate"
        assert PatternComplexity.COMPLEX.value == "complex"


class TestBuiltinPatterns:
    """Tests for built-in patterns."""

    def test_basic_test_pattern(self):
        """Test basic test pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_basic_test")
        
        assert pattern is not None
        assert "@Test" in pattern.template
        assert "@DisplayName" in pattern.template

    def test_mock_test_pattern(self):
        """Test mock test pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_mock_test")
        
        assert pattern is not None
        assert "@Mock" in pattern.template
        assert "when(" in pattern.template

    def test_exception_test_pattern(self):
        """Test exception test pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_exception_test")
        
        assert pattern is not None
        assert "assertThrows" in pattern.template

    def test_parameterized_test_pattern(self):
        """Test parameterized test pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_parameterized_test")
        
        assert pattern is not None
        assert "@ParameterizedTest" in pattern.template

    def test_boundary_test_pattern(self):
        """Test boundary test pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_boundary_test")
        
        assert pattern is not None
        assert pattern.category == PatternCategory.BOUNDARY


class TestPatternLibraryIntegration:
    """Integration tests for pattern library."""

    def test_full_workflow(self):
        """Test full workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            library = PatternLibrary(db_path=db_path)
            
            custom = library.create_custom_pattern(
                name="Integration Test Pattern",
                category=PatternCategory.INTEGRATION,
                description="Pattern for integration tests",
                template="@SpringBootTest class {class_name} { }",
                tags=["integration", "spring"]
            )
            
            patterns = library.find_patterns(category=PatternCategory.INTEGRATION)
            assert len(patterns) >= 1
            
            library.record_usage(custom.pattern_id, success=True)
            
            stats = library.get_statistics()
            assert stats["total_patterns"] > 0
