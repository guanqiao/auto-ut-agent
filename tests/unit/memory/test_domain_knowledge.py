"""Unit tests for DomainKnowledgeBase module."""

import pytest
import tempfile
import os

from pyutagent.memory.domain_knowledge import (
    DomainKnowledgeBase,
    KnowledgeDomain,
    KnowledgeType,
    KnowledgeEntry,
)


class TestDomainKnowledgeBase:
    """Tests for DomainKnowledgeBase class."""

    def test_init(self):
        """Test initialization."""
        kb = DomainKnowledgeBase()
        
        assert kb.db_path is not None
        assert kb._cache is not None

    def test_init_custom_db(self):
        """Test initialization with custom database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_knowledge.db")
            kb = DomainKnowledgeBase(db_path=db_path)
            
            assert kb.db_path == db_path

    def test_builtin_knowledge_loaded(self):
        """Test that built-in knowledge is loaded."""
        kb = DomainKnowledgeBase()
        
        entry = kb.get_entry("junit5_test_annotation")
        
        assert entry is not None
        assert entry.title == "@Test Annotation"

    def test_add_entry(self):
        """Test adding an entry."""
        kb = DomainKnowledgeBase()
        
        entry = KnowledgeEntry(
            entry_id="custom_entry_123",
            domain=KnowledgeDomain.JAVA_CORE,
            knowledge_type=KnowledgeType.CONCEPT,
            title="Custom Entry",
            content="Custom content",
            tags=["custom"]
        )
        
        entry_id = kb.add_entry(entry)
        
        assert entry_id == "custom_entry_123"
        retrieved = kb.get_entry("custom_entry_123")
        assert retrieved is not None

    def test_get_entry_existing(self):
        """Test getting existing entry."""
        kb = DomainKnowledgeBase()
        
        entry = kb.get_entry("junit5_test_annotation")
        
        assert entry is not None
        assert entry.domain == KnowledgeDomain.JUNIT5

    def test_get_entry_non_existing(self):
        """Test getting non-existing entry."""
        kb = DomainKnowledgeBase()
        
        entry = kb.get_entry("non_existing_entry")
        
        assert entry is None

    def test_search_by_query(self):
        """Test searching by query."""
        kb = DomainKnowledgeBase()
        
        results = kb.search("assertion")
        
        assert len(results) > 0
        for entry in results:
            assert "assertion" in entry.content.lower() or "assertion" in entry.title.lower()

    def test_search_by_domain(self):
        """Test searching by domain."""
        kb = DomainKnowledgeBase()
        
        results = kb.search("", domain=KnowledgeDomain.JUNIT5)
        
        assert len(results) > 0
        for entry in results:
            assert entry.domain == KnowledgeDomain.JUNIT5

    def test_search_by_type(self):
        """Test searching by knowledge type."""
        kb = DomainKnowledgeBase()
        
        results = kb.search("", knowledge_type=KnowledgeType.PATTERN)
        
        for entry in results:
            assert entry.knowledge_type == KnowledgeType.PATTERN

    def test_search_by_tags(self):
        """Test searching by tags."""
        kb = DomainKnowledgeBase()
        
        results = kb.search("", tags=["mock", "mockito"])
        
        for entry in results:
            assert any(tag in entry.tags for tag in ["mock", "mockito"])

    def test_search_limit(self):
        """Test search with limit."""
        kb = DomainKnowledgeBase()
        
        results = kb.search("", limit=5)
        
        assert len(results) <= 5

    def test_get_by_domain(self):
        """Test getting entries by domain."""
        kb = DomainKnowledgeBase()
        
        entries = kb.get_by_domain(KnowledgeDomain.MOCKITO)
        
        assert len(entries) > 0
        for entry in entries:
            assert entry.domain == KnowledgeDomain.MOCKITO

    def test_get_quick_reference(self):
        """Test getting quick reference."""
        kb = DomainKnowledgeBase()
        
        reference = kb.get_quick_reference(KnowledgeDomain.JUNIT5)
        
        assert "JUnit 5" in reference
        assert "@Test" in reference

    def test_record_usage(self):
        """Test recording usage."""
        kb = DomainKnowledgeBase()
        
        entry = kb.get_entry("junit5_test_annotation")
        if entry:
            initial_count = entry.usage_count
            
            kb.record_usage("junit5_test_annotation")
            
            updated = kb.get_entry("junit5_test_annotation")
            assert updated.usage_count > initial_count

    def test_get_statistics(self):
        """Test getting statistics."""
        kb = DomainKnowledgeBase()
        
        stats = kb.get_statistics()
        
        assert "total_entries" in stats
        assert "domain_distribution" in stats
        assert stats["total_entries"] > 0


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry dataclass."""

    def test_entry_creation(self):
        """Test entry creation."""
        entry = KnowledgeEntry(
            entry_id="test-123",
            domain=KnowledgeDomain.JUNIT5,
            knowledge_type=KnowledgeType.CONCEPT,
            title="Test Entry",
            content="Test content",
            code_example="@Test void test() {}",
            tags=["test", "junit5"],
            confidence=0.95
        )
        
        assert entry.entry_id == "test-123"
        assert entry.domain == KnowledgeDomain.JUNIT5
        assert entry.confidence == 0.95

    def test_entry_to_dict(self):
        """Test entry to dictionary conversion."""
        entry = KnowledgeEntry(
            entry_id="test-123",
            domain=KnowledgeDomain.MOCKITO,
            knowledge_type=KnowledgeType.PATTERN,
            title="Test",
            content="Content",
            tags=["mock"]
        )
        
        d = entry.to_dict()
        
        assert d["entry_id"] == "test-123"
        assert d["domain"] == "mockito"
        assert d["knowledge_type"] == "pattern"


class TestKnowledgeDomain:
    """Tests for KnowledgeDomain enum."""

    def test_domain_values(self):
        """Test domain enum values."""
        assert KnowledgeDomain.JAVA_CORE.value == "java_core"
        assert KnowledgeDomain.JUNIT5.value == "junit5"
        assert KnowledgeDomain.MOCKITO.value == "mockito"
        assert KnowledgeDomain.SPRING_BOOT.value == "spring_boot"
        assert KnowledgeDomain.BEST_PRACTICES.value == "best_practices"


class TestKnowledgeType:
    """Tests for KnowledgeType enum."""

    def test_type_values(self):
        """Test type enum values."""
        assert KnowledgeType.CONCEPT.value == "concept"
        assert KnowledgeType.PATTERN.value == "pattern"
        assert KnowledgeType.EXAMPLE.value == "example"
        assert KnowledgeType.RULE.value == "rule"
        assert KnowledgeType.TIP.value == "tip"
        assert KnowledgeType.WARNING.value == "warning"


class TestBuiltinKnowledge:
    """Tests for built-in knowledge entries."""

    def test_junit5_knowledge_count(self):
        """Test JUnit 5 knowledge count."""
        kb = DomainKnowledgeBase()
        
        entries = kb.get_by_domain(KnowledgeDomain.JUNIT5)
        
        assert len(entries) >= 5

    def test_mockito_knowledge_count(self):
        """Test Mockito knowledge count."""
        kb = DomainKnowledgeBase()
        
        entries = kb.get_by_domain(KnowledgeDomain.MOCKITO)
        
        assert len(entries) >= 4

    def test_best_practices_knowledge(self):
        """Test best practices knowledge."""
        kb = DomainKnowledgeBase()
        
        entries = kb.get_by_domain(KnowledgeDomain.BEST_PRACTICES)
        
        assert len(entries) >= 2

    def test_junit5_displayname_entry(self):
        """Test JUnit 5 DisplayName entry."""
        kb = DomainKnowledgeBase()
        
        entry = kb.get_entry("junit5_displayname")
        
        assert entry is not None
        assert "@DisplayName" in entry.code_example

    def test_mockito_when_thenreturn_entry(self):
        """Test Mockito when-thenReturn entry."""
        kb = DomainKnowledgeBase()
        
        entry = kb.get_entry("mockito_when_thenreturn")
        
        assert entry is not None
        assert "when(" in entry.code_example
        assert "thenReturn" in entry.code_example

    def test_aaa_pattern_entry(self):
        """Test AAA pattern entry."""
        kb = DomainKnowledgeBase()
        
        entry = kb.get_entry("java_aaa_pattern")
        
        assert entry is not None
        assert "Arrange" in entry.content or "AAA" in entry.content


class TestDomainKnowledgeIntegration:
    """Integration tests for domain knowledge base."""

    def test_full_workflow(self):
        """Test full workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entry = KnowledgeEntry(
                entry_id="custom_test_entry",
                domain=KnowledgeDomain.JAVA_CORE,
                knowledge_type=KnowledgeType.TIP,
                title="Custom Tip",
                content="This is a custom tip for testing",
                tags=["custom", "test"]
            )
            
            kb.add_entry(entry)
            
            results = kb.search("custom tip")
            assert len(results) >= 1
            
            kb.record_usage("custom_test_entry")
            
            stats = kb.get_statistics()
            assert stats["total_entries"] > 0

    def test_search_relevance(self):
        """Test search relevance."""
        kb = DomainKnowledgeBase()
        
        results = kb.search("mock")
        
        mock_entries = [e for e in results if KnowledgeDomain.MOCKITO == e.domain]
        assert len(mock_entries) > 0

    def test_quick_reference_content(self):
        """Test quick reference content."""
        kb = DomainKnowledgeBase()
        
        reference = kb.get_quick_reference(KnowledgeDomain.MOCKITO)
        
        assert "Mockito" in reference
        assert "@Mock" in reference
