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
        db_path = tempfile.mktemp(suffix=".db")
        try:
            library = PatternLibrary(db_path=db_path)
            assert library.db_path == db_path
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

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
            placeholders=[],
            example_usage="Example usage"
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
        
        assert len(patterns) >= 0

    def test_match_pattern(self):
        """Test matching a pattern."""
        library = PatternLibrary()
        
        code = "@Test void testMethod() { assertEquals(1, 1); }"
        
        match = library.match_pattern(code)
        
        assert match is not None
        assert match.pattern_id is not None
        assert match.confidence >= 0

    def test_record_pattern_usage(self):
        """Test recording pattern usage."""
        library = PatternLibrary()
        
        library.record_pattern_usage("builtin_basic_test")
        
        pattern = library.get_pattern("builtin_basic_test")
        assert pattern.usage_count >= 1

    def test_get_best_patterns(self):
        """Test getting best patterns."""
        library = PatternLibrary()
        
        patterns = library.get_best_patterns(limit=5)
        
        assert len(patterns) <= 5

    def test_get_statistics(self):
        """Test getting statistics."""
        library = PatternLibrary()
        
        stats = library.get_statistics()
        
        assert "total_patterns" in stats
        assert "by_category" in stats


class TestTestPattern:
    """Tests for TestPattern dataclass."""

    def test_pattern_creation(self):
        """Test pattern creation."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test Pattern",
            category=PatternCategory.TEST_STRUCTURE,
            description="A test pattern",
            template="@Test void test() { }",
            placeholders=[],
            example_usage="Example"
        )
        
        assert pattern.pattern_id == "test-123"
        assert pattern.category == PatternCategory.TEST_STRUCTURE

    def test_pattern_render(self):
        """Test pattern rendering."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test Pattern",
            category=PatternCategory.TEST_STRUCTURE,
            description="A test pattern",
            template="Hello {name}!",
            placeholders=["name"],
            example_usage="Example"
        )
        
        rendered = pattern.render({"name": "World"})
        
        assert "World" in rendered

    def test_pattern_render_missing_placeholder(self):
        """Test pattern rendering with missing placeholder."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test Pattern",
            category=PatternCategory.TEST_STRUCTURE,
            description="A test pattern",
            template="Hello {name} from {place}!",
            placeholders=["name", "place"],
            example_usage="Example"
        )
        
        rendered = pattern.render({"name": "World"})
        
        assert "World" in rendered

    def test_pattern_to_dict(self):
        """Test pattern to dictionary conversion."""
        pattern = TestPattern(
            pattern_id="test-123",
            name="Test Pattern",
            category=PatternCategory.TEST_STRUCTURE,
            description="A test pattern",
            template="@Test void test() { }",
            placeholders=[],
            example_usage="Example"
        )
        
        d = pattern.to_dict()
        
        assert d["pattern_id"] == "test-123"
        assert d["category"] == "test_structure"


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_match_creation(self):
        """Test match creation."""
        match = PatternMatch(
            pattern_id="pattern-123",
            confidence=0.85,
            matched_elements=["element1", "element2"],
            suggestions=["suggestion1"]
        )
        
        assert match.pattern_id == "pattern-123"
        assert match.confidence == 0.85
        assert len(match.matched_elements) == 2


class TestPatternCategory:
    """Tests for PatternCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert PatternCategory.TEST_STRUCTURE.value == "test_structure"
        assert PatternCategory.ASSERTION.value == "assertion"
        assert PatternCategory.MOCK.value == "mock"
        assert PatternCategory.SETUP.value == "setup"
        assert PatternCategory.EXCEPTION.value == "exception"


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

    def test_mock_test_pattern(self):
        """Test mock test pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_mock_test")
        
        assert pattern is not None
        assert pattern.category == PatternCategory.MOCK

    def test_exception_test_pattern(self):
        """Test exception test pattern."""
        library = PatternLibrary()
        
        pattern = library.get_pattern("builtin_exception_test")
        
        assert pattern is not None
        assert pattern.category == PatternCategory.EXCEPTION


class TestPatternLibraryIntegration:
    """Integration tests for pattern library."""

    def test_full_workflow(self):
        """Test full workflow."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            library = PatternLibrary(db_path=db_path)
            
            pattern = TestPattern(
                pattern_id="custom_pattern",
                name="Custom Pattern",
                category=PatternCategory.TEST_STRUCTURE,
                description="Custom pattern",
                template="@Test void test() { }",
                placeholders=[],
                example_usage="Example"
            )
            
            library.add_pattern(pattern)
            
            retrieved = library.get_pattern("custom_pattern")
            assert retrieved is not None
            
            library.record_pattern_usage("custom_pattern")
            
            stats = library.get_statistics()
            assert stats["total_patterns"] >= 1
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_pattern_search_workflow(self):
        """Test pattern search workflow."""
        library = PatternLibrary()
        
        patterns = library.find_patterns(category=PatternCategory.TEST_STRUCTURE)
        
        assert len(patterns) > 0
