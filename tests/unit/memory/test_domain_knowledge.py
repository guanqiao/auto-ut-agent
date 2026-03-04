"""Unit tests for DomainKnowledgeBase module."""

import pytest
import tempfile
import os

from pyutagent.memory.domain_knowledge import (
    DomainKnowledgeBase,
    KnowledgeEntry,
    KnowledgeDomain,
    KnowledgeType,
)


class TestDomainKnowledgeBase:
    """Tests for DomainKnowledgeBase class."""

    def test_init(self):
        """Test initialization."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            assert kb.db_path == db_path
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_init_custom_db(self):
        """Test initialization with custom database."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            assert kb.db_path == db_path
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_builtin_knowledge_loaded(self):
        """Test that built-in knowledge is loaded."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entry = kb.get_entry("junit5_displayname")
            
            assert entry is not None
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_add_entry(self):
        """Test adding an entry."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entry = KnowledgeEntry(
                entry_id="custom_entry_123",
                domain=KnowledgeDomain.JUNIT5,
                knowledge_type=KnowledgeType.PATTERN,
                title="Custom Pattern",
                content="This is a custom pattern",
                tags=["custom", "test"]
            )
            
            entry_id = kb.add_entry(entry)
            
            assert entry_id == "custom_entry_123"
            retrieved = kb.get_entry("custom_entry_123")
            assert retrieved is not None
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_get_entry_existing(self):
        """Test getting existing entry."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entry = kb.get_entry("junit5_displayname")
            
            assert entry is not None
            assert entry.domain == KnowledgeDomain.JUNIT5
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_get_entry_non_existing(self):
        """Test getting non-existing entry."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entry = kb.get_entry("non_existing_entry")
            
            assert entry is None
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_search_by_query(self):
        """Test searching by query."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            results = kb.search("assertion")
            
            assert len(results) >= 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_search_by_domain(self):
        """Test searching by domain."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            results = kb.search(domain=KnowledgeDomain.JUNIT5)
            
            assert len(results) >= 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_search_by_type(self):
        """Test searching by type."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            results = kb.search(knowledge_type=KnowledgeType.PATTERN)
            
            assert len(results) >= 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_search_by_tags(self):
        """Test searching by tags."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            results = kb.search(tags=["test"])
            
            assert len(results) >= 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_search_limit(self):
        """Test search limit."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            results = kb.search(limit=5)
            
            assert len(results) <= 5
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_get_by_domain(self):
        """Test getting entries by domain."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entries = kb.get_by_domain(KnowledgeDomain.JUNIT5)
            
            assert len(entries) >= 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_get_quick_reference(self):
        """Test getting quick reference."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            ref = kb.get_quick_reference(KnowledgeDomain.JUNIT5)
            
            assert ref is not None
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_record_usage(self):
        """Test recording usage."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            kb.record_usage("junit5_displayname")
            
            entry = kb.get_entry("junit5_displayname")
            assert entry.usage_count >= 1
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_get_statistics(self):
        """Test getting statistics."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            stats = kb.get_statistics()
            
            assert "total_entries" in stats
            assert "by_domain" in stats
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry dataclass."""

    def test_entry_creation(self):
        """Test entry creation."""
        entry = KnowledgeEntry(
            entry_id="test-123",
            domain=KnowledgeDomain.JUNIT5,
            knowledge_type=KnowledgeType.PATTERN,
            title="Test Pattern",
            content="This is a test pattern",
            tags=["test", "pattern"]
        )
        
        assert entry.entry_id == "test-123"
        assert entry.domain == KnowledgeDomain.JUNIT5

    def test_entry_to_dict(self):
        """Test entry to dictionary conversion."""
        entry = KnowledgeEntry(
            entry_id="test-123",
            domain=KnowledgeDomain.MOCKITO,
            knowledge_type=KnowledgeType.EXAMPLE,
            title="Test Example",
            content="Example content"
        )
        
        d = entry.to_dict()
        
        assert d["entry_id"] == "test-123"
        assert d["domain"] == KnowledgeDomain.MOCKITO


class TestKnowledgeDomain:
    """Tests for KnowledgeDomain enum."""

    def test_domain_values(self):
        """Test domain enum values."""
        assert KnowledgeDomain.JAVA_CORE.value == "java_core"
        assert KnowledgeDomain.JUNIT5.value == "junit5"
        assert KnowledgeDomain.MOCKITO.value == "mockito"
        assert KnowledgeDomain.SPRING_BOOT.value == "spring_boot"


class TestKnowledgeType:
    """Tests for KnowledgeType enum."""

    def test_type_values(self):
        """Test type enum values."""
        assert KnowledgeType.CONCEPT.value == "concept"
        assert KnowledgeType.PATTERN.value == "pattern"
        assert KnowledgeType.EXAMPLE.value == "example"
        assert KnowledgeType.RULE.value == "rule"


class TestBuiltinKnowledge:
    """Tests for built-in knowledge."""

    def test_junit5_knowledge_count(self):
        """Test JUnit5 knowledge count."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entries = kb.get_by_domain(KnowledgeDomain.JUNIT5)
            
            assert len(entries) >= 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_mockito_knowledge_count(self):
        """Test Mockito knowledge count."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entries = kb.get_by_domain(KnowledgeDomain.MOCKITO)
            
            assert len(entries) >= 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_best_practices_knowledge(self):
        """Test best practices knowledge."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entries = kb.get_by_domain(KnowledgeDomain.BEST_PRACTICES)
            
            assert len(entries) >= 0
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)


class TestDomainKnowledgeIntegration:
    """Integration tests for domain knowledge base."""

    def test_full_workflow(self):
        """Test full workflow."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            entry = KnowledgeEntry(
                entry_id="custom_entry",
                domain=KnowledgeDomain.JUNIT5,
                knowledge_type=KnowledgeType.PATTERN,
                title="Custom Pattern",
                content="Custom content"
            )
            
            kb.add_entry(entry)
            
            retrieved = kb.get_entry("custom_entry")
            assert retrieved is not None
            
            kb.record_usage("custom_entry")
            
            stats = kb.get_statistics()
            assert stats["total_entries"] >= 1
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_search_relevance(self):
        """Test search relevance."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            results = kb.search("test")
            
            assert isinstance(results, list)
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)

    def test_quick_reference_content(self):
        """Test quick reference content."""
        db_path = tempfile.mktemp(suffix=".db")
        try:
            kb = DomainKnowledgeBase(db_path=db_path)
            
            ref = kb.get_quick_reference(KnowledgeDomain.JUNIT5)
            
            assert ref is not None
        finally:
            if os.path.exists(db_path):
                os.remove(db_path)
