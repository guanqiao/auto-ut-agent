"""Code Pattern Library for test generation patterns and templates.

This module provides a library of reusable code patterns, test templates,
and best practices for intelligent test generation.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class PatternCategory(Enum):
    """Categories of code patterns."""
    TEST_STRUCTURE = "test_structure"
    ASSERTION = "assertion"
    MOCK = "mock"
    SETUP = "setup"
    PARAMETERIZED = "parameterized"
    EXCEPTION = "exception"
    BOUNDARY = "boundary"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"


class PatternComplexity(Enum):
    """Complexity levels of patterns."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class TestPattern:
    """A test pattern template."""
    pattern_id: str
    name: str
    category: PatternCategory
    description: str
    template: str
    placeholders: List[str]
    example_usage: str
    complexity: PatternComplexity = PatternComplexity.MODERATE
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    success_rate: float = 0.8
    usage_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def render(self, values: Dict[str, str]) -> str:
        """Render the pattern with provided values.
        
        Args:
            values: Dictionary mapping placeholders to values
            
        Returns:
            Rendered template string
        """
        result = self.template
        for placeholder in self.placeholders:
            value = values.get(placeholder, f"{{{placeholder}}}")
            result = result.replace(f"{{{placeholder}}}", value)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "template": self.template,
            "placeholders": self.placeholders,
            "example_usage": self.example_usage,
            "complexity": self.complexity.value,
            "tags": self.tags,
            "prerequisites": self.prerequisites,
            "success_rate": self.success_rate,
            "usage_count": self.usage_count,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestPattern':
        """Create from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            name=data["name"],
            category=PatternCategory(data["category"]),
            description=data["description"],
            template=data["template"],
            placeholders=data.get("placeholders", []),
            example_usage=data.get("example_usage", ""),
            complexity=PatternComplexity(data.get("complexity", "moderate")),
            tags=data.get("tags", []),
            prerequisites=data.get("prerequisites", []),
            success_rate=data.get("success_rate", 0.8),
            usage_count=data.get("usage_count", 0),
            created_at=data.get("created_at", datetime.now().isoformat())
        )


@dataclass
class PatternMatch:
    """Result of matching a pattern against code."""
    pattern: TestPattern
    confidence: float
    matched_elements: List[str]
    suggested_values: Dict[str, str]
    missing_elements: List[str]


class PatternLibrary:
    """Library of code patterns for test generation.
    
    Features:
    - Pattern storage and retrieval
    - Pattern matching against code
    - Template rendering
    - Pattern recommendation
    - Usage statistics
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize pattern library.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "pattern_library.db"
        
        self.db_path = str(db_path)
        self._init_database()
        self._cache: Dict[str, TestPattern] = {}
        self._category_index: Dict[PatternCategory, Set[str]] = defaultdict(set)
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        self._load_builtin_patterns()
        
        logger.info(f"[PatternLibrary] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT,
                    template TEXT NOT NULL,
                    placeholders TEXT,
                    example_usage TEXT,
                    complexity TEXT DEFAULT 'moderate',
                    tags TEXT,
                    prerequisites TEXT,
                    success_rate REAL DEFAULT 0.8,
                    usage_count INTEGER DEFAULT 0,
                    created_at TEXT
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_category ON patterns(category)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_name ON patterns(name)')
            
            conn.commit()
    
    def _load_builtin_patterns(self):
        """Load built-in test patterns."""
        builtin_patterns = self._get_builtin_patterns()
        
        for pattern in builtin_patterns:
            existing = self.get_pattern(pattern.pattern_id)
            if not existing:
                self.add_pattern(pattern, builtin=True)
    
    def _get_builtin_patterns(self) -> List[TestPattern]:
        """Get built-in test patterns."""
        patterns = []
        
        patterns.append(TestPattern(
            pattern_id="builtin_basic_test",
            name="Basic Test Method",
            category=PatternCategory.TEST_STRUCTURE,
            description="Basic JUnit 5 test method structure",
            template='''@Test
@DisplayName("should {description}")
void {method_name}() {
    // Given
    {given_code}
    
    // When
    {when_code}
    
    // Then
    {then_code}
}''',
            placeholders=["description", "method_name", "given_code", "when_code", "then_code"],
            example_usage="Basic AAA pattern test",
            complexity=PatternComplexity.SIMPLE,
            tags=["basic", "junit5", "aaa"]
        ))
        
        patterns.append(TestPattern(
            pattern_id="builtin_mock_test",
            name="Mock-based Test",
            category=PatternCategory.MOCK,
            description="Test using Mockito mocks",
            template='''@Test
@DisplayName("should {description}")
void {method_name}() {
    // Given
    {mock_setup}
    when({mock_call}).thenReturn({mock_return});
    
    // When
    {actual_type} result = {method_call};
    
    // Then
    assertEquals({expected}, result);
    verify({mock_object}).{mock_method}();
}''',
            placeholders=["description", "method_name", "mock_setup", "mock_call", 
                         "mock_return", "actual_type", "method_call", "expected", 
                         "mock_object", "mock_method"],
            example_usage="Test with mocked dependencies",
            complexity=PatternComplexity.MODERATE,
            tags=["mock", "mockito", "dependency"],
            prerequisites=["Mockito"]
        ))
        
        patterns.append(TestPattern(
            pattern_id="builtin_exception_test",
            name="Exception Test",
            category=PatternCategory.EXCEPTION,
            description="Test for expected exception",
            template='''@Test
@DisplayName("should throw {exception_type} when {condition}")
void {method_name}() {
    // Given
    {given_code}
    
    // When & Then
    {exception_type} exception = assertThrows(
        {exception_type}.class,
        () -> {method_call}
    );
    
    assertEquals("{expected_message}", exception.getMessage());
}''',
            placeholders=["exception_type", "condition", "method_name", "given_code", 
                         "method_call", "expected_message"],
            example_usage="Test for expected exceptions",
            complexity=PatternComplexity.SIMPLE,
            tags=["exception", "assertThrows"]
        ))
        
        patterns.append(TestPattern(
            pattern_id="builtin_parameterized_test",
            name="Parameterized Test",
            category=PatternCategory.PARAMETERIZED,
            description="JUnit 5 parameterized test",
            template='''@ParameterizedTest
@MethodSource("{method_source}")
@DisplayName("should {description}")
void {method_name}({param_declarations}) {
    // Given
    {given_code}
    
    // When
    {when_code}
    
    // Then
    {then_code}
}

private static Stream<Arguments> {method_source}() {
    return Stream.of(
        Arguments.of({test_cases})
    );
}''',
            placeholders=["method_source", "description", "method_name", "param_declarations",
                         "given_code", "when_code", "then_code", "test_cases"],
            example_usage="Parameterized test with multiple inputs",
            complexity=PatternComplexity.COMPLEX,
            tags=["parameterized", "junit5", "data-driven"],
            prerequisites=["JUnit Jupiter Params"]
        ))
        
        patterns.append(TestPattern(
            pattern_id="builtin_boundary_test",
            name="Boundary Value Test",
            category=PatternCategory.BOUNDARY,
            description="Test for boundary values",
            template='''@Test
@DisplayName("should handle {boundary_type} boundary for {parameter_name}")
void {method_name}() {
    // Given
    {given_code}
    
    // When
    {actual_type} result = {method_call};
    
    // Then
    {assertion_code}
}''',
            placeholders=["boundary_type", "parameter_name", "method_name", "given_code",
                         "actual_type", "method_call", "assertion_code"],
            example_usage="Test boundary conditions",
            complexity=PatternComplexity.MODERATE,
            tags=["boundary", "edge-case"]
        ))
        
        patterns.append(TestPattern(
            pattern_id="builtin_null_check_test",
            name="Null Input Test",
            category=PatternCategory.BOUNDARY,
            description="Test for null input handling",
            template='''@Test
@DisplayName("should throw NullPointerException when {parameter} is null")
void {method_name}() {
    // Given
    {given_code}
    
    // When & Then
    assertThrows(NullPointerException.class, () -> {
        {method_call};
    });
}''',
            placeholders=["parameter", "method_name", "given_code", "method_call"],
            example_usage="Test null input validation",
            complexity=PatternComplexity.SIMPLE,
            tags=["null", "validation", "boundary"]
        ))
        
        patterns.append(TestPattern(
            pattern_id="builtin_before_each",
            name="Test Setup with BeforeEach",
            category=PatternCategory.SETUP,
            description="Common test setup using @BeforeEach",
            template='''private {class_under_test} {instance_name};

@Mock
private {dependency_type} {dependency_name};

@BeforeEach
void setUp() {
    MockitoAnnotations.openMocks(this);
    {instance_name} = new {class_under_test}({constructor_args});
}

@Test
@DisplayName("should {description}")
void {method_name}() {
    // When
    {when_code}
    
    // Then
    {then_code}
}''',
            placeholders=["class_under_test", "instance_name", "dependency_type", 
                         "dependency_name", "constructor_args", "description", 
                         "method_name", "when_code", "then_code"],
            example_usage="Test class with common setup",
            complexity=PatternComplexity.MODERATE,
            tags=["setup", "beforeEach", "mockito"],
            prerequisites=["Mockito"]
        ))
        
        patterns.append(TestPattern(
            pattern_id="builtin_integration_test",
            name="Integration Test",
            category=PatternCategory.INTEGRATION,
            description="Integration test with Spring Boot",
            template='''@SpringBootTest
class {test_class_name} {
    
    @Autowired
    private {service_type} {service_name};
    
    @Test
    @DisplayName("should {description}")
    void {method_name}() {
        // Given
        {given_code}
        
        // When
        {when_code}
        
        // Then
        {then_code}
    }
}''',
            placeholders=["test_class_name", "service_type", "service_name", 
                         "description", "method_name", "given_code", "when_code", "then_code"],
            example_usage="Spring Boot integration test",
            complexity=PatternComplexity.COMPLEX,
            tags=["integration", "spring-boot", "autowired"],
            prerequisites=["Spring Boot Test"]
        ))
        
        return patterns
    
    def add_pattern(
        self,
        pattern: TestPattern,
        builtin: bool = False
    ) -> str:
        """Add a pattern to the library.
        
        Args:
            pattern: Pattern to add
            builtin: Whether this is a built-in pattern
            
        Returns:
            Pattern ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO patterns
                (pattern_id, name, category, description, template, placeholders,
                 example_usage, complexity, tags, prerequisites, success_rate, usage_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id,
                pattern.name,
                pattern.category.value,
                pattern.description,
                pattern.template,
                json.dumps(pattern.placeholders),
                pattern.example_usage,
                pattern.complexity.value,
                json.dumps(pattern.tags),
                json.dumps(pattern.prerequisites),
                pattern.success_rate,
                pattern.usage_count,
                pattern.created_at
            ))
            conn.commit()
        
        self._cache[pattern.pattern_id] = pattern
        self._category_index[pattern.category].add(pattern.pattern_id)
        for tag in pattern.tags:
            self._tag_index[tag].add(pattern.pattern_id)
        
        logger.debug(f"[PatternLibrary] Added pattern: {pattern.pattern_id} ({pattern.name})")
        return pattern.pattern_id
    
    def get_pattern(self, pattern_id: str) -> Optional[TestPattern]:
        """Get a pattern by ID."""
        if pattern_id in self._cache:
            return self._cache[pattern_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patterns WHERE pattern_id = ?', (pattern_id,))
            row = cursor.fetchone()
            
            if row:
                pattern = self._row_to_pattern(row)
                self._cache[pattern_id] = pattern
                return pattern
        
        return None
    
    def find_patterns(
        self,
        category: Optional[PatternCategory] = None,
        tags: Optional[List[str]] = None,
        complexity: Optional[PatternComplexity] = None,
        min_success_rate: float = 0.0
    ) -> List[TestPattern]:
        """Find patterns matching criteria.
        
        Args:
            category: Filter by category
            tags: Filter by tags (any match)
            complexity: Filter by complexity
            min_success_rate: Minimum success rate
            
        Returns:
            List of matching patterns
        """
        query = 'SELECT * FROM patterns WHERE success_rate >= ?'
        params = [min_success_rate]
        
        if category:
            query += ' AND category = ?'
            params.append(category.value)
        
        if complexity:
            query += ' AND complexity = ?'
            params.append(complexity.value)
        
        query += ' ORDER BY success_rate DESC, usage_count DESC'
        
        patterns = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                pattern = self._row_to_pattern(row)
                
                if tags:
                    if not any(tag in pattern.tags for tag in tags):
                        continue
                
                patterns.append(pattern)
        
        return patterns
    
    def match_pattern(
        self,
        code: str,
        category: Optional[PatternCategory] = None
    ) -> List[PatternMatch]:
        """Match patterns against code.
        
        Args:
            code: Source code to match
            category: Optional category filter
            
        Returns:
            List of pattern matches sorted by confidence
        """
        matches = []
        
        patterns = self.find_patterns(category=category)
        
        for pattern in patterns:
            match = self._match_single_pattern(pattern, code)
            if match and match.confidence > 0.3:
                matches.append(match)
        
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches
    
    def _match_single_pattern(
        self,
        pattern: TestPattern,
        code: str
    ) -> Optional[PatternMatch]:
        """Match a single pattern against code."""
        matched_elements = []
        missing_elements = []
        suggested_values = {}
        
        for placeholder in pattern.placeholders:
            placeholder_patterns = {
                "method_name": r'void\s+(\w+)\s*\(',
                "class_under_test": r'class\s+(\w+)',
                "exception_type": r'assertThrows\s*\(\s*(\w+)\.class',
                "description": r'@DisplayName\s*\(\s*"([^"]+)"\s*\)',
            }
            
            if placeholder in placeholder_patterns:
                match = re.search(placeholder_patterns[placeholder], code)
                if match:
                    matched_elements.append(placeholder)
                    suggested_values[placeholder] = match.group(1)
                else:
                    missing_elements.append(placeholder)
        
        pattern_indicators = {
            PatternCategory.MOCK: ['@Mock', 'Mockito', 'when(', 'verify('],
            PatternCategory.EXCEPTION: ['assertThrows', 'Exception'],
            PatternCategory.PARAMETERIZED: ['@ParameterizedTest', '@MethodSource', '@CsvSource'],
            PatternCategory.BOUNDARY: ['null', 'empty', 'boundary', 'edge'],
            PatternCategory.SETUP: ['@BeforeEach', '@AfterEach', 'setUp'],
        }
        
        category_matches = 0
        if pattern.category in pattern_indicators:
            indicators = pattern_indicators[pattern.category]
            category_matches = sum(1 for ind in indicators if ind in code)
        
        confidence = 0.0
        if matched_elements:
            confidence = len(matched_elements) / len(pattern.placeholders)
        if category_matches > 0:
            confidence = min(1.0, confidence + category_matches * 0.1)
        
        if confidence < 0.1:
            return None
        
        return PatternMatch(
            pattern=pattern,
            confidence=confidence,
            matched_elements=matched_elements,
            suggested_values=suggested_values,
            missing_elements=missing_elements
        )
    
    def recommend_patterns(
        self,
        context: Dict[str, Any],
        limit: int = 5
    ) -> List[Tuple[TestPattern, float]]:
        """Recommend patterns based on context.
        
        Args:
            context: Context dictionary with keys like:
                - has_dependencies: bool
                - has_exceptions: bool
                - needs_parameterization: bool
                - complexity: str
                - tags: List[str]
            limit: Maximum number of recommendations
            
        Returns:
            List of (pattern, score) tuples
        """
        scored_patterns = []
        
        patterns = self.find_patterns()
        
        for pattern in patterns:
            score = pattern.success_rate
            
            if context.get("has_dependencies") and pattern.category == PatternCategory.MOCK:
                score += 0.2
            
            if context.get("has_exceptions") and pattern.category == PatternCategory.EXCEPTION:
                score += 0.2
            
            if context.get("needs_parameterization") and pattern.category == PatternCategory.PARAMETERIZED:
                score += 0.2
            
            context_tags = context.get("tags", [])
            if context_tags:
                tag_overlap = len(set(pattern.tags) & set(context_tags))
                score += tag_overlap * 0.1
            
            complexity = context.get("complexity", "moderate")
            if pattern.complexity.value == complexity:
                score += 0.1
            
            scored_patterns.append((pattern, score))
        
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        return scored_patterns[:limit]
    
    def record_usage(self, pattern_id: str, success: bool):
        """Record pattern usage.
        
        Args:
            pattern_id: Pattern ID
            success: Whether the usage was successful
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT usage_count, success_rate FROM patterns WHERE pattern_id = ?',
                (pattern_id,)
            )
            row = cursor.fetchone()
            
            if row:
                usage_count = row[0] + 1
                old_rate = row[1]
                new_rate = old_rate + ((1.0 if success else 0.0) - old_rate) / usage_count
                
                cursor.execute('''
                    UPDATE patterns 
                    SET usage_count = ?, success_rate = ?
                    WHERE pattern_id = ?
                ''', (usage_count, new_rate, pattern_id))
                conn.commit()
                
                if pattern_id in self._cache:
                    self._cache[pattern_id].usage_count = usage_count
                    self._cache[pattern_id].success_rate = new_rate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM patterns')
            total_patterns = cursor.fetchone()[0]
            
            cursor.execute('SELECT category, COUNT(*) FROM patterns GROUP BY category')
            category_counts = dict(cursor.fetchall())
            
            cursor.execute('SELECT AVG(success_rate) FROM patterns')
            avg_success_rate = cursor.fetchone()[0] or 0.0
            
            cursor.execute('SELECT SUM(usage_count) FROM patterns')
            total_usage = cursor.fetchone()[0] or 0
            
            return {
                "total_patterns": total_patterns,
                "category_distribution": category_counts,
                "average_success_rate": avg_success_rate,
                "total_usage": total_usage,
                "cached_patterns": len(self._cache)
            }
    
    def create_custom_pattern(
        self,
        name: str,
        category: PatternCategory,
        description: str,
        template: str,
        tags: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
        complexity: PatternComplexity = PatternComplexity.MODERATE
    ) -> TestPattern:
        """Create a custom pattern from template.
        
        Args:
            name: Pattern name
            category: Pattern category
            description: Description
            template: Template string with {placeholders}
            tags: Optional tags
            prerequisites: Optional prerequisites
            complexity: Pattern complexity
            
        Returns:
            Created TestPattern
        """
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        pattern = TestPattern(
            pattern_id=f"custom_{uuid.uuid4().hex[:8]}",
            name=name,
            category=category,
            description=description,
            template=template,
            placeholders=placeholders,
            example_usage="Custom pattern",
            complexity=complexity,
            tags=tags or [],
            prerequisites=prerequisites or []
        )
        
        self.add_pattern(pattern)
        return pattern
    
    def _row_to_pattern(self, row) -> TestPattern:
        """Convert database row to TestPattern."""
        return TestPattern(
            pattern_id=row[0],
            name=row[1],
            category=PatternCategory(row[2]),
            description=row[3] or "",
            template=row[4],
            placeholders=json.loads(row[5]) if row[5] else [],
            example_usage=row[6] or "",
            complexity=PatternComplexity(row[7]) if row[7] else PatternComplexity.MODERATE,
            tags=json.loads(row[8]) if row[8] else [],
            prerequisites=json.loads(row[9]) if row[9] else [],
            success_rate=row[10] if row[10] is not None else 0.8,
            usage_count=row[11] or 0,
            created_at=row[12] or datetime.now().isoformat()
        )
