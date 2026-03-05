"""Unit tests for ai_suggestion_provider module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from pyutagent.ui.editor.ai_suggestion_provider import (
    SuggestionType,
    CodeSuggestion,
    SuggestionContext,
    SuggestionCache,
    AISuggestionProvider,
    MockAISuggestionProvider,
    SuggestionManager
)


class TestSuggestionType:
    """Tests for SuggestionType enum."""
    
    def test_enum_values(self):
        """Test that enum values exist."""
        assert SuggestionType.COMPLETION is not None
        assert SuggestionType.INSERTION is not None
        assert SuggestionType.REPLACEMENT is not None
        
    def test_enum_uniqueness(self):
        """Test that enum values are unique."""
        values = [SuggestionType.COMPLETION, SuggestionType.INSERTION, SuggestionType.REPLACEMENT]
        assert len(values) == len(set(values))


class TestCodeSuggestion:
    """Tests for CodeSuggestion dataclass."""
    
    def test_basic_creation(self):
        """Test creating a basic code suggestion."""
        suggestion = CodeSuggestion(
            text="print('hello')",
            suggestion_type=SuggestionType.COMPLETION,
            start_line=1,
            start_column=0
        )
        
        assert suggestion.text == "print('hello')"
        assert suggestion.suggestion_type == SuggestionType.COMPLETION
        assert suggestion.start_line == 1
        assert suggestion.start_column == 0
        assert suggestion.end_line is None
        assert suggestion.end_column is None
        assert suggestion.confidence == 0.0
        assert suggestion.explanation == ""
        
    def test_creation_with_optional_fields(self):
        """Test creating suggestion with optional fields."""
        suggestion = CodeSuggestion(
            text="new code",
            suggestion_type=SuggestionType.REPLACEMENT,
            start_line=5,
            start_column=10,
            end_line=5,
            end_column=20,
            confidence=0.95,
            explanation="Better implementation"
        )
        
        assert suggestion.end_line == 5
        assert suggestion.end_column == 20
        assert suggestion.confidence == 0.95
        assert suggestion.explanation == "Better implementation"


class TestSuggestionContext:
    """Tests for SuggestionContext dataclass."""
    
    def test_creation(self):
        """Test creating a suggestion context."""
        context = SuggestionContext(
            file_path="/path/to/file.py",
            language="python",
            cursor_line=10,
            cursor_column=5,
            text_before_cursor="def func():",
            text_after_cursor="\n    pass",
            selected_text="",
            full_text="def func():\n    pass"
        )
        
        assert context.file_path == "/path/to/file.py"
        assert context.language == "python"
        assert context.cursor_line == 10
        assert context.cursor_column == 5
        assert context.text_before_cursor == "def func():"


class TestSuggestionCache:
    """Tests for SuggestionCache class."""
    
    @pytest.fixture
    def cache(self):
        """Create a SuggestionCache instance."""
        return SuggestionCache(max_size=10, ttl_seconds=1.0)
        
    @pytest.fixture
    def sample_context(self):
        """Create a sample context."""
        return SuggestionContext(
            file_path="test.py",
            language="python",
            cursor_line=1,
            cursor_column=0,
            text_before_cursor="def ",
            text_after_cursor="",
            selected_text="",
            full_text="def "
        )
        
    @pytest.fixture
    def sample_suggestion(self):
        """Create a sample suggestion."""
        return CodeSuggestion(
            text="hello():",
            suggestion_type=SuggestionType.COMPLETION,
            start_line=1,
            start_column=4
        )
        
    def test_initial_state(self, cache):
        """Test initial state of cache."""
        assert cache._cache == {}
        
    def test_set_and_get(self, cache, sample_context, sample_suggestion):
        """Test setting and getting from cache."""
        cache.set(sample_context, sample_suggestion)
        
        retrieved = cache.get(sample_context)
        assert retrieved == sample_suggestion
        
    def test_get_nonexistent(self, cache, sample_context):
        """Test getting non-existent key."""
        assert cache.get(sample_context) is None
        
    def test_cache_expiration(self, cache, sample_context, sample_suggestion):
        """Test cache entry expiration."""
        cache._ttl_seconds = 0.01  # Very short TTL
        cache.set(sample_context, sample_suggestion)
        
        # Should exist immediately
        assert cache.get(sample_context) == sample_suggestion
        
        # Wait for expiration
        time.sleep(0.02)
        
        # Should be expired
        assert cache.get(sample_context) is None
        
    def test_cache_max_size(self, cache):
        """Test cache max size enforcement."""
        cache._max_size = 2
        
        # Add multiple entries
        for i in range(3):
            context = SuggestionContext(
                file_path=f"test{i}.py",
                language="python",
                cursor_line=i,
                cursor_column=0,
                text_before_cursor=f"def {i}",
                text_after_cursor="",
                selected_text="",
                full_text=f"def {i}"
            )
            suggestion = CodeSuggestion(
                text=f"func{i}()",
                suggestion_type=SuggestionType.COMPLETION,
                start_line=i,
                start_column=0
            )
            cache.set(context, suggestion)
            
        # Should only have 2 entries (max_size)
        assert len(cache._cache) == 2
        
    def test_cache_clear(self, cache, sample_context, sample_suggestion):
        """Test clearing the cache."""
        cache.set(sample_context, sample_suggestion)
        assert len(cache._cache) == 1
        
        cache.clear()
        assert len(cache._cache) == 0
        
    def test_cache_key_consistency(self, cache, sample_context, sample_suggestion):
        """Test that same context produces same key."""
        cache.set(sample_context, sample_suggestion)
        
        # Same context should retrieve
        retrieved = cache.get(sample_context)
        assert retrieved == sample_suggestion


class TestAISuggestionProvider:
    """Tests for AISuggestionProvider class."""
    
    @pytest.fixture
    def provider(self, qtbot):
        """Create an AISuggestionProvider instance."""
        return AISuggestionProvider()
        
    @pytest.fixture
    def sample_context(self):
        """Create a sample context."""
        return SuggestionContext(
            file_path="test.py",
            language="python",
            cursor_line=1,
            cursor_column=4,
            text_before_cursor="def ",
            text_after_cursor="",
            selected_text="",
            full_text="def "
        )
        
    def test_initial_state(self, provider):
        """Test initial state of provider."""
        assert provider._cache is not None
        assert provider._debounce_delay_ms == 300
        assert not provider.is_generating()
        
    def test_set_debounce_delay(self, provider):
        """Test setting debounce delay."""
        provider.set_debounce_delay(500)
        assert provider._debounce_delay_ms == 500
        
    def test_set_debounce_delay_minimum(self, provider):
        """Test debounce delay minimum value."""
        provider.set_debounce_delay(10)  # Below minimum
        assert provider._debounce_delay_ms == 50  # Should be clamped
        
    def test_clear_cache(self, provider, sample_context):
        """Test clearing cache."""
        suggestion = CodeSuggestion(
            text="test",
            suggestion_type=SuggestionType.COMPLETION,
            start_line=1,
            start_column=0
        )
        provider._cache.set(sample_context, suggestion)
        
        provider.clear_cache()
        assert provider._cache.get(sample_context) is None
        
    def test_cancel_pending(self, provider):
        """Test canceling pending requests."""
        provider._pending_context = Mock()
        provider.cancel_pending()
        assert provider._pending_context is None
        
    def test_build_prompt_completion(self, provider, sample_context):
        """Test building completion prompt."""
        prompt = provider._build_prompt(sample_context, SuggestionType.COMPLETION)
        
        assert "Complete" in prompt or "complete" in prompt.lower()
        assert "python" in prompt.lower() or sample_context.language in prompt
        
    def test_build_prompt_insertion(self, provider, sample_context):
        """Test building insertion prompt."""
        prompt = provider._build_prompt(sample_context, SuggestionType.INSERTION)
        
        assert "Insert" in prompt or "insert" in prompt.lower()
        
    def test_build_prompt_replacement(self, provider, sample_context):
        """Test building replacement prompt."""
        prompt = provider._build_prompt(sample_context, SuggestionType.REPLACEMENT)
        
        assert "Replace" in prompt or "replace" in prompt.lower()
        
    def test_parse_suggestion_with_code_blocks(self, provider):
        """Test parsing suggestion with markdown code blocks."""
        response = """```python
def hello():
    pass
```"""
        
        context = Mock()
        result = provider._parse_suggestion(response, context, SuggestionType.COMPLETION)
        
        assert "```" not in result
        assert "def hello():" in result
        
    def test_parse_suggestion_without_code_blocks(self, provider):
        """Test parsing suggestion without code blocks."""
        response = "def hello():\n    pass"
        
        context = Mock()
        result = provider._parse_suggestion(response, context, SuggestionType.COMPLETION)
        
        assert result == "def hello():\n    pass"


class TestMockAISuggestionProvider:
    """Tests for MockAISuggestionProvider class."""
    
    @pytest.fixture
    def provider(self):
        """Create a MockAISuggestionProvider instance."""
        return MockAISuggestionProvider()
        
    def test_call_llm_def_pattern(self, provider):
        """Test mock response for def pattern."""
        prompt = "Complete the following: def "
        response = provider._call_llm(prompt)
        
        assert "(self, x):" in response or "(" in response
        
    def test_call_llm_if_pattern(self, provider):
        """Test mock response for if pattern."""
        prompt = "Complete the following: if "
        response = provider._call_llm(prompt)
        
        assert ":" in response
        
    def test_call_llm_for_pattern(self, provider):
        """Test mock response for for pattern."""
        prompt = "Complete the following: for "
        response = provider._call_llm(prompt)
        
        assert "in" in response
        
    def test_call_llm_class_pattern(self, provider):
        """Test mock response for class pattern."""
        prompt = "Complete the following: class "
        response = provider._call_llm(prompt)
        
        assert ":" in response
        
    def test_call_llm_import_pattern(self, provider):
        """Test mock response for import pattern."""
        prompt = "Complete the following: import"
        response = provider._call_llm(prompt)
        
        assert "os" in response or "import" in response
        
    def test_call_llm_return_pattern(self, provider):
        """Test mock response for return pattern."""
        prompt = "Complete the following: return"
        response = provider._call_llm(prompt)
        
        assert response.strip() != ""
        
    def test_set_mock_suggestion(self, provider):
        """Test setting custom mock suggestion."""
        provider.set_mock_suggestion("custom_pattern", "custom_result")
        
        response = provider._call_llm("test custom_pattern here")
        assert response == "custom_result"


class TestSuggestionManager:
    """Tests for SuggestionManager class."""
    
    @pytest.fixture
    def mock_editor(self):
        """Create a mock editor."""
        editor = Mock()
        editor.textCursor.return_value = Mock()
        editor.toPlainText.return_value = "def test():\n    pass"
        return editor
        
    @pytest.fixture
    def manager(self, mock_editor):
        """Create a SuggestionManager instance."""
        return SuggestionManager(mock_editor)
        
    def test_initial_state(self, manager):
        """Test initial state of manager."""
        assert manager._editor is not None
        assert manager._provider is not None
        assert manager._current_suggestion is None
        
    def test_get_current_suggestion_none(self, manager):
        """Test getting suggestion when none exists."""
        assert manager.get_current_suggestion() is None
        
    def test_accept_current_suggestion_none(self, manager):
        """Test accepting when no suggestion exists."""
        assert not manager.accept_current_suggestion()
        
    def test_reject_current_suggestion_none(self, manager):
        """Test rejecting when no suggestion exists."""
        assert not manager.reject_current_suggestion()
        
    def test_clear_suggestion(self, manager):
        """Test clearing suggestion."""
        manager._current_suggestion = CodeSuggestion(
            text="test",
            suggestion_type=SuggestionType.COMPLETION,
            start_line=1,
            start_column=0
        )
        
        manager.clear_suggestion()
        assert manager._current_suggestion is None
        
    def test_build_context(self, manager, mock_editor):
        """Test building suggestion context."""
        cursor = Mock()
        cursor.blockNumber.return_value = 0
        cursor.columnNumber.return_value = 4
        cursor.position.return_value = 4
        cursor.selectedText.return_value = ""
        mock_editor.textCursor.return_value = cursor
        
        context = manager._build_context()
        
        assert context.cursor_line == 1  # 0 + 1
        assert context.cursor_column == 4
        assert context.full_text == "def test():\n    pass"


class TestAISuggestionIntegration:
    """Integration tests for AI suggestion functionality."""
    
    def test_full_suggestion_flow(self, qtbot):
        """Test full suggestion flow."""
        # Create mock editor
        mock_editor = Mock()
        mock_editor.textCursor.return_value = Mock()
        mock_editor.toPlainText.return_value = "def "
        
        manager = SuggestionManager(mock_editor)
        
        # Simulate suggestion ready
        suggestion = CodeSuggestion(
            text="hello():",
            suggestion_type=SuggestionType.COMPLETION,
            start_line=1,
            start_column=4,
            confidence=0.9
        )
        
        manager._on_suggestion_ready(suggestion)
        assert manager.get_current_suggestion() == suggestion
        
        # Accept it
        result = manager.accept_current_suggestion()
        assert result
        assert manager.get_current_suggestion() is None
        
    def test_suggestion_error_handling(self, qtbot):
        """Test error handling in suggestion flow."""
        mock_editor = Mock()
        manager = SuggestionManager(mock_editor)
        
        # Simulate error
        manager._on_suggestion_error("Test error")
        assert manager.get_current_suggestion() is None
        
    def test_provider_signals_connected(self, qtbot):
        """Test that provider signals are properly connected."""
        mock_editor = Mock()
        manager = SuggestionManager(mock_editor)
        
        # Check that signals are connected
        assert manager._provider.receivers(manager._provider.suggestion_ready) >= 0
        assert manager._provider.receivers(manager._provider.suggestion_error) >= 0
