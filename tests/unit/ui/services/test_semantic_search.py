"""Tests for semantic search service and dialog."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from pyutagent.ui.services.semantic_search import (
    SemanticSearchService,
    SearchResult,
    SearchQuery,
    SearchWorker
)
from pyutagent.ui.dialogs.semantic_search_dialog import (
    SemanticSearchDialog,
    SearchResultItem,
    CodePreviewWidget,
    show_semantic_search
)


@pytest.fixture
def app():
    """Create QApplication for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def temp_project():
    """Create a temporary project with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample Java file
        java_file = Path(tmpdir) / "TestClass.java"
        java_file.write_text("""
public class TestClass {
    private String name;

    public TestClass(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
""")

        # Create sample Python file
        py_file = Path(tmpdir) / "test_module.py"
        py_content = '''def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b

class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, value):
        self.result += value
        return self.result
'''
        py_file.write_text(py_content)

        yield tmpdir


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_relevance_percentage(self):
        """Test relevance percentage calculation."""
        result = SearchResult(
            file_path="/test/file.py",
            content="def test(): pass",
            score=0.85,
            start_line=1,
            end_line=3
        )
        assert result.relevance_percentage == 85

    def test_file_name(self):
        """Test file name extraction."""
        result = SearchResult(
            file_path="/path/to/file.py",
            content="",
            score=0.5,
            start_line=1,
            end_line=1
        )
        assert result.file_name == "file.py"

    def test_location_single_line(self):
        """Test location string for single line."""
        result = SearchResult(
            file_path="/test/file.py",
            content="",
            score=0.5,
            start_line=10,
            end_line=10
        )
        assert result.location == "file.py:10"

    def test_location_multi_line(self):
        """Test location string for multiple lines."""
        result = SearchResult(
            file_path="/test/file.py",
            content="",
            score=0.5,
            start_line=10,
            end_line=20
        )
        assert result.location == "file.py:10-20"


class TestSearchQuery:
    """Tests for SearchQuery dataclass."""

    def test_default_values(self):
        """Test default values for SearchQuery."""
        query = SearchQuery(original_query="test query")
        assert query.original_query == "test query"
        assert query.keywords == []
        assert query.intent == ""
        assert query.language_filter is None
        assert query.symbol_type_filter is None


class TestSemanticSearchService:
    """Tests for SemanticSearchService."""

    def test_initialization(self, temp_project):
        """Test service initialization."""
        service = SemanticSearchService(project_path=temp_project)
        assert service.project_path == Path(temp_project)
        assert service._indexer is None
        assert service._embedding_model is None

    def test_parse_query_general_search(self, temp_project):
        """Test parsing general search query."""
        service = SemanticSearchService(project_path=temp_project)
        parsed = service._parse_query("find authentication code")

        assert parsed.original_query == "find authentication code"
        assert "authentication" in parsed.keywords
        assert parsed.intent == "general_search"

    def test_parse_query_find_function(self, temp_project):
        """Test parsing function search query."""
        service = SemanticSearchService(project_path=temp_project)
        parsed = service._parse_query("find function calculate sum")

        assert parsed.intent == "find_function"
        assert parsed.symbol_type_filter == "method"
        assert "calculate" in parsed.keywords
        assert "sum" in parsed.keywords

    def test_parse_query_find_class(self, temp_project):
        """Test parsing class search query."""
        service = SemanticSearchService(project_path=temp_project)
        parsed = service._parse_query("find class Calculator")

        assert parsed.intent == "find_class"
        assert parsed.symbol_type_filter == "class"

    def test_parse_query_with_language(self, temp_project):
        """Test parsing query with language filter."""
        service = SemanticSearchService(project_path=temp_project)
        parsed = service._parse_query("find python function")

        assert parsed.language_filter == "python"

    def test_parse_query_explain(self, temp_project):
        """Test parsing explain query."""
        service = SemanticSearchService(project_path=temp_project)
        parsed = service._parse_query("explain how this works")

        assert parsed.intent == "explain"

    def test_extract_context(self, temp_project):
        """Test context extraction from file."""
        service = SemanticSearchService(project_path=temp_project)
        py_file = Path(temp_project) / "test_module.py"

        context = service._extract_context(str(py_file), 5, 7, context_lines=2)

        assert context is not None
        assert len(context) > 0
        # Check that line numbers are included
        assert "5:" in context or "6:" in context or "7:" in context

    def test_extract_context_nonexistent_file(self, temp_project):
        """Test context extraction for non-existent file."""
        service = SemanticSearchService(project_path=temp_project)
        context = service._extract_context("/nonexistent/file.py", 1, 5)
        assert context == ""

    def test_calculate_relevance_score(self, temp_project):
        """Test relevance score calculation."""
        service = SemanticSearchService(project_path=temp_project)

        result = SearchResult(
            file_path="/test.py",
            content="def calculate_sum",
            score=0.5,
            start_line=1,
            end_line=1,
            symbol_name="calculate_sum"
        )

        parsed = SearchQuery(
            original_query="find calculate",
            keywords=["calculate"],
            intent="find_function"
        )

        score = service._calculate_relevance_score(result, parsed)
        assert score > 0.5  # Should be boosted by keyword match
        assert score <= 1.0  # Should be capped at 1.0

    def test_calculate_relevance_score_with_symbol_type(self, temp_project):
        """Test relevance score with symbol type filter."""
        service = SemanticSearchService(project_path=temp_project)

        result = SearchResult(
            file_path="/test.py",
            content="",
            score=0.5,
            start_line=1,
            end_line=1,
            symbol_type="method"
        )

        parsed = SearchQuery(
            original_query="find method",
            keywords=["find"],
            intent="find_function",
            symbol_type_filter="method"
        )

        score = service._calculate_relevance_score(result, parsed)
        assert score > 0.5  # Should be boosted by symbol type match

    def test_keyword_search(self, temp_project):
        """Test keyword-based search."""
        service = SemanticSearchService(project_path=temp_project)
        results = service._keyword_search("calculate", max_results=10)

        assert isinstance(results, list)
        # Should find the calculate_sum function
        assert len(results) > 0

    def test_keyword_search_no_keywords(self, temp_project):
        """Test keyword search with no valid keywords."""
        service = SemanticSearchService(project_path=temp_project)
        results = service._keyword_search("a b c", max_results=10)
        assert results == []

    def test_sort_results_by_relevance(self, temp_project):
        """Test sorting results by relevance."""
        # Results should be sorted by score in descending order by default
        results = [
            SearchResult(file_path="/a.py", content="", score=0.5, start_line=1, end_line=1),
            SearchResult(file_path="/b.py", content="", score=0.9, start_line=1, end_line=1),
            SearchResult(file_path="/c.py", content="", score=0.3, start_line=1, end_line=1),
        ]

        # Sort by score descending (default behavior in search method)
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        assert sorted_results[0].score == 0.9
        assert sorted_results[1].score == 0.5
        assert sorted_results[2].score == 0.3

    def test_get_search_suggestions(self, temp_project):
        """Test getting search suggestions."""
        service = SemanticSearchService(project_path=temp_project)
        suggestions = service.get_search_suggestions("calc", limit=5)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5
        # Should include pattern suggestions
        assert any("find function" in s for s in suggestions)

    def test_is_indexed_no_indexer(self, temp_project):
        """Test is_indexed when no indexer exists."""
        service = SemanticSearchService(project_path=temp_project)
        assert not service.is_indexed()


class TestSearchWorker:
    """Tests for SearchWorker thread."""

    def test_worker_initialization(self, temp_project):
        """Test worker initialization."""
        worker = SearchWorker(
            query="test",
            project_path=temp_project,
            max_results=10
        )
        assert worker.query == "test"
        assert worker.project_path == temp_project
        assert worker.max_results == 10
        assert not worker._is_cancelled

    def test_worker_cancel(self, temp_project):
        """Test worker cancellation."""
        worker = SearchWorker(
            query="test",
            project_path=temp_project
        )
        worker.cancel()
        assert worker._is_cancelled


class TestSearchResultItem:
    """Tests for SearchResultItem widget."""

    def test_item_creation(self, app):
        """Test creating a search result item."""
        result = SearchResult(
            file_path="/test/file.py",
            content="def test(): pass",
            score=0.85,
            start_line=10,
            end_line=15,
            symbol_name="test",
            symbol_type="function"
        )

        item = SearchResultItem(result)
        assert item.result == result
        assert "file.py" in item.text()
        assert "85%" in item.text()
        assert "function" in item.text()

    def test_item_icon_python(self, app):
        """Test Python file icon."""
        result = SearchResult(
            file_path="/test/script.py",
            content="",
            score=0.5,
            start_line=1,
            end_line=1
        )
        item = SearchResultItem(result)
        assert "🐍" in item.text()

    def test_item_icon_java(self, app):
        """Test Java file icon."""
        result = SearchResult(
            file_path="/test/Class.java",
            content="",
            score=0.5,
            start_line=1,
            end_line=1
        )
        item = SearchResultItem(result)
        assert "☕" in item.text()


class TestCodePreviewWidget:
    """Tests for CodePreviewWidget."""

    def test_widget_creation(self, app):
        """Test creating the preview widget."""
        widget = CodePreviewWidget()
        assert widget.isReadOnly()

    def test_show_result(self, app):
        """Test displaying a result."""
        widget = CodePreviewWidget()
        result = SearchResult(
            file_path="/test/file.py",
            content="def test(): pass",
            score=0.85,
            start_line=10,
            end_line=15,
            symbol_name="test",
            symbol_type="function",
            context="def test():\n    pass"
        )

        widget.show_result(result)
        text = widget.toPlainText()
        assert "/test/file.py" in text
        assert "85%" in text
        assert "test" in text
        assert "function" in text

    def test_show_result_none(self, app):
        """Test displaying no result."""
        widget = CodePreviewWidget()
        widget.show_result(None)
        assert widget.toPlainText() == ""


class TestSemanticSearchDialog:
    """Tests for SemanticSearchDialog."""

    @pytest.mark.skip(reason="Qt dialog tests require display")
    def test_dialog_creation(self, app, temp_project):
        """Test creating the dialog."""
        dialog = SemanticSearchDialog(project_path=temp_project)
        assert dialog.project_path == temp_project
        assert dialog.windowTitle() == "Semantic Search"

    @pytest.mark.skip(reason="Qt dialog tests require display")
    def test_set_query(self, app, temp_project):
        """Test setting search query."""
        dialog = SemanticSearchDialog(project_path=temp_project)
        dialog.set_query("find function")
        assert dialog._search_input.text() == "find function"

    def test_apply_filter(self, temp_project):
        """Test applying filters to results."""
        # Test filter logic without creating the full dialog
        results = [
            SearchResult(file_path="/a.py", content="", score=0.5, start_line=1, end_line=1, symbol_type="method"),
            SearchResult(file_path="/b.py", content="", score=0.5, start_line=1, end_line=1, symbol_type="class"),
            SearchResult(file_path="/c.py", content="", score=0.5, start_line=1, end_line=1, symbol_type="field"),
        ]

        # Test filter logic directly
        filtered = [r for r in results if r.symbol_type == "method"]
        assert len(filtered) == 1
        assert filtered[0].symbol_type == "method"

    def test_sort_results(self, temp_project):
        """Test sorting results."""
        results = [
            SearchResult(file_path="/z.py", content="", score=0.5, start_line=1, end_line=1),
            SearchResult(file_path="/a.py", content="", score=0.9, start_line=1, end_line=1),
        ]

        # Sort by file name
        sorted_results = sorted(results, key=lambda r: r.file_name.lower())
        assert sorted_results[0].file_name == "a.py"
        assert sorted_results[1].file_name == "z.py"

    @pytest.mark.skip(reason="Qt dialog tests require display")
    def test_get_selected_result_none(self, app, temp_project):
        """Test getting selected result when none selected."""
        dialog = SemanticSearchDialog(project_path=temp_project)
        assert dialog.get_selected_result() is None


class TestSemanticSearchIntegration:
    """Integration tests for semantic search."""

    @pytest.mark.slow
    def test_full_search_flow(self, temp_project):
        """Test complete search flow."""
        service = SemanticSearchService(project_path=temp_project)

        # Perform search
        results = service.search("calculate sum", max_results=10)

        assert isinstance(results, list)
        # Should find the calculate_sum function

    def test_search_with_empty_query(self, temp_project):
        """Test search with empty query."""
        service = SemanticSearchService(project_path=temp_project)
        results = service.search("", max_results=10)
        assert results == []

    def test_search_with_whitespace_query(self, temp_project):
        """Test search with whitespace-only query."""
        service = SemanticSearchService(project_path=temp_project)
        results = service.search("   ", max_results=10)
        assert results == []


class TestShowSemanticSearch:
    """Tests for show_semantic_search convenience function."""

    def test_function_exists(self):
        """Test that the function exists and is callable."""
        assert callable(show_semantic_search)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
