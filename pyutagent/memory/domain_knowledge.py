"""Domain Knowledge Base for Java/JUnit/Mockito testing knowledge.

This module provides domain-specific knowledge for test generation,
including Java patterns, JUnit best practices, and Mockito configurations.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class KnowledgeDomain(Enum):
    """Domains of knowledge."""
    JAVA_CORE = "java_core"
    JUNIT5 = "junit5"
    MOCKITO = "mockito"
    SPRING_BOOT = "spring_boot"
    MAVEN = "maven"
    ASSERTIONS = "assertions"
    PATTERNS = "patterns"
    BEST_PRACTICES = "best_practices"
    ANTI_PATTERNS = "anti_patterns"


class KnowledgeType(Enum):
    """Types of knowledge entries."""
    CONCEPT = "concept"
    PATTERN = "pattern"
    EXAMPLE = "example"
    RULE = "rule"
    TIP = "tip"
    WARNING = "warning"
    TEMPLATE = "template"


@dataclass
class KnowledgeEntry:
    """A knowledge entry in the domain knowledge base."""
    entry_id: str
    domain: KnowledgeDomain
    knowledge_type: KnowledgeType
    title: str
    content: str
    code_example: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    confidence: float = 1.0
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DomainKnowledgeBase:
    """Domain-specific knowledge base for test generation.
    
    Features:
    - Java/JUnit/Mockito knowledge
    - Best practices and patterns
    - Code examples
    - Quick reference
    - Search and retrieval
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize domain knowledge base.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "domain_knowledge.db"
        
        self.db_path = str(db_path)
        self._init_database()
        self._cache: Dict[str, KnowledgeEntry] = {}
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        self._load_builtin_knowledge()
        
        logger.info(f"[DomainKnowledgeBase] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_entries (
                    entry_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    knowledge_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    code_example TEXT,
                    tags TEXT,
                    related_entries TEXT,
                    references TEXT,
                    confidence REAL DEFAULT 1.0,
                    usage_count INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON knowledge_entries(domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON knowledge_entries(knowledge_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON knowledge_entries(title)')
            
            conn.commit()
    
    def _load_builtin_knowledge(self):
        """Load built-in domain knowledge."""
        builtin = self._get_builtin_knowledge()
        
        for entry in builtin:
            existing = self.get_entry(entry.entry_id)
            if not existing:
                self.add_entry(entry, builtin=True)
    
    def _get_builtin_knowledge(self) -> List[KnowledgeEntry]:
        """Get built-in knowledge entries."""
        entries = []
        
        entries.extend(self._get_junit5_knowledge())
        entries.extend(self._get_mockito_knowledge())
        entries.extend(self._get_java_patterns())
        entries.extend(self._get_best_practices())
        
        return entries
    
    def _get_junit5_knowledge(self) -> List[KnowledgeEntry]:
        """Get JUnit 5 knowledge entries."""
        entries = []
        
        entries.append(KnowledgeEntry(
            entry_id="junit5_test_annotation",
            domain=KnowledgeDomain.JUNIT5,
            knowledge_type=KnowledgeType.CONCEPT,
            title="@Test Annotation",
            content="The @Test annotation marks a method as a test method. JUnit 5 does not require methods to be public.",
            code_example='''@Test
@DisplayName("Should calculate sum correctly")
void shouldCalculateSumCorrectly() {
    // test code
}''',
            tags=["test", "annotation", "basic"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="junit5_displayname",
            domain=KnowledgeDomain.JUNIT5,
            knowledge_type=KnowledgeType.PATTERN,
            title="@DisplayName for Readable Tests",
            content="Use @DisplayName to provide human-readable test names. This improves test reports and documentation.",
            code_example='''@Test
@DisplayName("Should throw IllegalArgumentException when input is negative")
void shouldThrowExceptionForNegativeInput() {
    // test code
}''',
            tags=["displayname", "readability", "documentation"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="junit5_beforeeach",
            domain=KnowledgeDomain.JUNIT5,
            knowledge_type=KnowledgeType.PATTERN,
            title="@BeforeEach for Test Setup",
            content="Use @BeforeEach for common test setup. Runs before each test method.",
            code_example='''@BeforeEach
void setUp() {
    calculator = new Calculator();
    testData = new TestDataBuilder().build();
}''',
            tags=["setup", "beforeeach", "initialization"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="junit5_parameterized",
            domain=KnowledgeDomain.JUNIT5,
            knowledge_type=KnowledgeType.PATTERN,
            title="Parameterized Tests",
            content="Use @ParameterizedTest for data-driven testing with multiple inputs.",
            code_example='''@ParameterizedTest
@ValueSource(strings = {"hello", "world", "test"})
@DisplayName("Should process valid strings")
void shouldProcessValidStrings(String input) {
    assertNotNull(processor.process(input));
}''',
            tags=["parameterized", "data-driven", "value-source"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="junit5_assertions",
            domain=KnowledgeDomain.ASSERTIONS,
            knowledge_type=KnowledgeType.REFERENCE,
            title="Common JUnit 5 Assertions",
            content="Key assertion methods in JUnit 5 for verifying test outcomes.",
            code_example='''// Equality
assertEquals(expected, actual);
assertEquals(expected, actual, "Optional message");

// Boolean
assertTrue(condition);
assertFalse(condition);

// Null checks
assertNull(value);
assertNotNull(value);

// Exceptions
assertThrows(ExpectedException.class, () -> method());

// Arrays/Collections
assertArrayEquals(expectedArray, actualArray);
assertTrue(collection.contains(element));''',
            tags=["assertion", "verification", "reference"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="junit5_nested",
            domain=KnowledgeDomain.JUNIT5,
            knowledge_type=KnowledgeType.PATTERN,
            title="Nested Test Classes",
            content="Use @Nested to group related tests within a test class.",
            code_example='''@Nested
@DisplayName("When processing valid input")
class WhenProcessingValidInput {
    @Test
    void shouldReturnResult() {
        // test
    }
}

@Nested
@DisplayName("When processing invalid input")
class WhenProcessingInvalidInput {
    @Test
    void shouldThrowException() {
        // test
    }
}''',
            tags=["nested", "grouping", "organization"],
            confidence=1.0
        ))
        
        return entries
    
    def _get_mockito_knowledge(self) -> List[KnowledgeEntry]:
        """Get Mockito knowledge entries."""
        entries = []
        
        entries.append(KnowledgeEntry(
            entry_id="mockito_mock_annotation",
            domain=KnowledgeDomain.MOCKITO,
            knowledge_type=KnowledgeType.CONCEPT,
            title="@Mock Annotation",
            content="Use @Mock to create mock objects. Requires MockitoAnnotations.openMocks(this) or @ExtendWith(MockitoExtension.class).",
            code_example='''@ExtendWith(MockitoExtension.class)
class UserServiceTest {
    @Mock
    private UserRepository userRepository;
    
    @InjectMocks
    private UserService userService;
}''',
            tags=["mock", "annotation", "dependency"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="mockito_when_thenreturn",
            domain=KnowledgeDomain.MOCKITO,
            knowledge_type=KnowledgeType.PATTERN,
            title="Stubbing with when().thenReturn()",
            content="Use when().thenReturn() to define mock behavior for method calls.",
            code_example='''// Simple return
when(userRepository.findById(1L)).thenReturn(Optional.of(user));

// Return different values on consecutive calls
when(userRepository.save(any()))
    .thenReturn(user1)
    .thenReturn(user2);''',
            tags=["stubbing", "when", "thenReturn"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="mockito_verify",
            domain=KnowledgeDomain.MOCKITO,
            knowledge_type=KnowledgeType.PATTERN,
            title="Verifying Interactions",
            content="Use verify() to check that mock methods were called with expected arguments.",
            code_example='''// Verify method was called
verify(userRepository).save(any(User.class));

// Verify number of calls
verify(userRepository, times(2)).findById(anyLong());

// Verify no interactions
verifyNoInteractions(userRepository);

// Verify order
InOrder inOrder = inOrder(userRepository, emailService);
inOrder.verify(userRepository).save(any());
inOrder.verify(emailService).send(any());''',
            tags=["verify", "interaction", "assertion"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="mockito_argument_matchers",
            domain=KnowledgeDomain.MOCKITO,
            knowledge_type=KnowledgeType.REFERENCE,
            title="Argument Matchers",
            content="Use matchers to flexibly match arguments in stubbing and verification.",
            code_example='''// Any value
when(repo.findById(anyLong())).thenReturn(user);

// Specific type
when(repo.findByEmail(anyString())).thenReturn(Optional.of(user));

// Custom matcher
when(repo.findByCriteria(argThat(criteria -> criteria.isActive())))
    .thenReturn(users);

// Common matchers
any()
anyLong()
anyString()
anyList()
eq(specificValue)
isNull()
isNotNull()''',
            tags=["matcher", "argument", "flexible"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="mockito_throws",
            domain=KnowledgeDomain.MOCKITO,
            knowledge_type=KnowledgeType.PATTERN,
            title="Throwing Exceptions",
            content="Use thenThrow() to make mocks throw exceptions.",
            code_example='''// Throw exception
when(userRepository.findById(anyLong()))
    .thenThrow(new DatabaseException("Connection failed"));

// Throw on void method
doThrow(new RuntimeException())
    .when(emailService).send(any());''',
            tags=["exception", "throw", "error"],
            confidence=1.0
        ))
        
        return entries
    
    def _get_java_patterns(self) -> List[KnowledgeEntry]:
        """Get Java testing patterns."""
        entries = []
        
        entries.append(KnowledgeEntry(
            entry_id="java_aaa_pattern",
            domain=KnowledgeDomain.PATTERNS,
            knowledge_type=KnowledgeType.PATTERN,
            title="AAA Pattern (Arrange-Act-Assert)",
            content="Structure tests in three clear sections: Arrange (setup), Act (execute), Assert (verify).",
            code_example='''@Test
void shouldCalculateTotalPrice() {
    // Arrange
    ShoppingCart cart = new ShoppingCart();
    cart.addItem(new Item("Apple", 1.50));
    cart.addItem(new Item("Banana", 0.75));
    
    // Act
    double total = cart.calculateTotal();
    
    // Assert
    assertEquals(2.25, total, 0.01);
}''',
            tags=["aaa", "structure", "pattern"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="java_builder_pattern",
            domain=KnowledgeDomain.PATTERNS,
            knowledge_type=KnowledgeType.PATTERN,
            title="Test Data Builder Pattern",
            content="Use builder pattern to create test data for cleaner, more readable tests.",
            code_example='''// TestDataBuilder
public class UserBuilder {
    private String name = "Default Name";
    private String email = "default@test.com";
    
    public UserBuilder withName(String name) {
        this.name = name;
        return this;
    }
    
    public User build() {
        return new User(name, email);
    }
}

// Usage in test
User user = new UserBuilder()
    .withName("Test User")
    .build();''',
            tags=["builder", "test-data", "fixture"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="java_null_safety",
            domain=KnowledgeDomain.JAVA_CORE,
            knowledge_type=KnowledgeType.PATTERN,
            title="Null Safety Testing",
            content="Always test null input handling to ensure robust code.",
            code_example='''@Test
void shouldThrowExceptionWhenInputIsNull() {
    IllegalArgumentException exception = assertThrows(
        IllegalArgumentException.class,
        () -> processor.process(null)
    );
    assertEquals("Input cannot be null", exception.getMessage());
}

@Test
void shouldHandleOptionalEmpty() {
    Optional<String> result = service.findByName(null);
    assertTrue(result.isEmpty());
}''',
            tags=["null", "safety", "validation"],
            confidence=1.0
        ))
        
        return entries
    
    def _get_best_practices(self) -> List[KnowledgeEntry]:
        """Get testing best practices."""
        entries = []
        
        entries.append(KnowledgeEntry(
            entry_id="bp_naming_convention",
            domain=KnowledgeDomain.BEST_PRACTICES,
            knowledge_type=KnowledgeType.RULE,
            title="Test Naming Convention",
            content="Use descriptive names that indicate what is being tested and the expected outcome. Pattern: should_expectedBehavior_when_condition",
            code_example='''// Good
void shouldReturnEmptyList_whenNoUsersFound()
void shouldThrowException_whenInputIsNull()
void shouldCalculateTotal_whenMultipleItemsInCart()

// Bad
void test1()
void testUser()
void testMethod()''',
            tags=["naming", "convention", "readability"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="bp_single_assertion",
            domain=KnowledgeDomain.BEST_PRACTICES,
            knowledge_type=KnowledgeType.TIP,
            title="Single Concept per Test",
            content="Each test should verify one concept or behavior. Multiple assertions are okay if they verify the same concept.",
            code_example='''// Good - single concept
@Test
void shouldCreateUserWithAllFields() {
    User user = userService.create(dto);
    
    assertNotNull(user.getId());
    assertEquals(dto.getName(), user.getName());
    assertEquals(dto.getEmail(), user.getEmail());
}

// Bad - multiple concepts
@Test
void testUserOperations() {
    // Tests creation, update, and deletion
}''',
            tags=["single-responsibility", "focus", "clarity"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="bp_independence",
            domain=KnowledgeDomain.BEST_PRACTICES,
            knowledge_type=KnowledgeType.RULE,
            title="Test Independence",
            content="Tests should be independent and not rely on the execution order or state from other tests.",
            code_example='''// Good - each test is self-contained
@BeforeEach
void setUp() {
    userRepository = new InMemoryUserRepository();
    userService = new UserService(userRepository);
}

// Bad - relying on shared state
private static User sharedUser; // Don't do this''',
            tags=["independence", "isolation", "reliability"],
            confidence=1.0
        ))
        
        entries.append(KnowledgeEntry(
            entry_id="ap_testing_implementation",
            domain=KnowledgeDomain.ANTI_PATTERNS,
            knowledge_type=KnowledgeType.WARNING,
            title="Don't Test Implementation Details",
            content="Test behavior, not implementation. Tests that rely on implementation details break when refactoring.",
            code_example='''// Bad - testing implementation
@Test
void testInternalList() {
    assertEquals(ArrayList.class, service.getInternalList().getClass());
}

// Good - testing behavior
@Test
void shouldReturnAllItems() {
    service.addItem(item1);
    service.addItem(item2);
    
    List<Item> items = service.getAllItems();
    
    assertEquals(2, items.size());
    assertTrue(items.contains(item1));
    assertTrue(items.contains(item2));
}''',
            tags=["anti-pattern", "implementation", "behavior"],
            confidence=1.0
        ))
        
        return entries
    
    def add_entry(
        self,
        entry: KnowledgeEntry,
        builtin: bool = False
    ) -> str:
        """Add a knowledge entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge_entries
                (entry_id, domain, knowledge_type, title, content, code_example,
                 tags, related_entries, references, confidence, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.entry_id,
                entry.domain.value,
                entry.knowledge_type.value,
                entry.title,
                entry.content,
                entry.code_example,
                json.dumps(entry.tags),
                json.dumps(entry.related_entries),
                json.dumps(entry.references),
                entry.confidence,
                entry.usage_count
            ))
            conn.commit()
        
        self._cache[entry.entry_id] = entry
        for tag in entry.tags:
            self._tag_index[tag].add(entry.entry_id)
        
        return entry.entry_id
    
    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""
        if entry_id in self._cache:
            return self._cache[entry_id]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM knowledge_entries WHERE entry_id = ?', (entry_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_entry(row)
        
        return None
    
    def search(
        self,
        query: str,
        domain: Optional[KnowledgeDomain] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[KnowledgeEntry]:
        """Search knowledge entries."""
        sql = 'SELECT * FROM knowledge_entries WHERE 1=1'
        params = []
        
        if domain:
            sql += ' AND domain = ?'
            params.append(domain.value)
        
        if knowledge_type:
            sql += ' AND knowledge_type = ?'
            params.append(knowledge_type.value)
        
        if query:
            sql += ' AND (title LIKE ? OR content LIKE ?)'
            params.extend([f'%{query}%', f'%{query}%'])
        
        sql += ' ORDER BY usage_count DESC, confidence DESC LIMIT ?'
        params.append(limit)
        
        entries = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            
            for row in cursor.fetchall():
                entry = self._row_to_entry(row)
                
                if tags:
                    if not all(t in entry.tags for t in tags):
                        continue
                
                entries.append(entry)
        
        return entries
    
    def get_by_domain(self, domain: KnowledgeDomain) -> List[KnowledgeEntry]:
        """Get all entries for a domain."""
        return self.search(query="", domain=domain, limit=100)
    
    def get_quick_reference(self, domain: KnowledgeDomain) -> str:
        """Get a quick reference for a domain."""
        entries = self.get_by_domain(domain)
        
        lines = [f"# {domain.value.replace('_', ' ').title()} Quick Reference", ""]
        
        for entry in entries[:10]:
            lines.append(f"## {entry.title}")
            lines.append(entry.content)
            if entry.code_example:
                lines.append("")
                lines.append("```java")
                lines.append(entry.code_example)
                lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def record_usage(self, entry_id: str):
        """Record usage of a knowledge entry."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE knowledge_entries SET usage_count = usage_count + 1 WHERE entry_id = ?',
                (entry_id,)
            )
            conn.commit()
        
        if entry_id in self._cache:
            self._cache[entry_id].usage_count += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM knowledge_entries')
            total = cursor.fetchone()[0]
            
            cursor.execute('SELECT domain, COUNT(*) FROM knowledge_entries GROUP BY domain')
            domain_counts = dict(cursor.fetchall())
            
            cursor.execute('SELECT knowledge_type, COUNT(*) FROM knowledge_entries GROUP BY knowledge_type')
            type_counts = dict(cursor.fetchall())
            
            return {
                "total_entries": total,
                "domain_distribution": domain_counts,
                "type_distribution": type_counts,
                "cached_entries": len(self._cache)
            }
    
    def _row_to_entry(self, row) -> KnowledgeEntry:
        """Convert database row to KnowledgeEntry."""
        return KnowledgeEntry(
            entry_id=row[0],
            domain=KnowledgeDomain(row[1]),
            knowledge_type=KnowledgeType(row[2]),
            title=row[3],
            content=row[4],
            code_example=row[5],
            tags=json.loads(row[6]) if row[6] else [],
            related_entries=json.loads(row[7]) if row[7] else [],
            references=json.loads(row[8]) if row[8] else [],
            confidence=row[9] if row[9] is not None else 1.0,
            usage_count=row[10] or 0
        )
