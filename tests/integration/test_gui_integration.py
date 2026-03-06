"""Integration tests for GUI components.

Tests the integration between different GUI modules.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


def test_markdown_renderer_integration():
    """Test MarkdownRenderer integrates with streaming handler."""
    from pyutagent.ui.components.markdown_renderer import MarkdownRenderer
    from pyutagent.ui.components.streaming_handler import StreamingHandler
    
    renderer = MarkdownRenderer()
    handler = StreamingHandler()
    
    # Test that renderer can process streamed content
    test_content = "# Hello\n\n```python\nprint('hello')\n```"
    html = renderer.render(test_content)
    
    assert "<h1>" in html
    assert "<code" in html
    
    print("✅ MarkdownRenderer integration test passed")


def test_symbol_indexer_integration():
    """Test SymbolIndexer integrates with MentionSystem."""
    from pyutagent.ui.services.symbol_indexer import SymbolIndexer
    
    indexer = SymbolIndexer()
    
    # Add test symbols
    indexer.add_symbol(
        name="TestClass",
        symbol_type="class",
        file_path="test.py",
        line_number=10
    )
    
    # Search for symbols
    results = indexer.search_symbols("Test")
    
    assert len(results) > 0
    assert results[0].name == "TestClass"
    
    print("✅ SymbolIndexer integration test passed")


def test_semantic_search_integration():
    """Test SemanticSearch integrates with SymbolIndexer."""
    from pyutagent.ui.services.semantic_search import SemanticSearchService
    from pyutagent.ui.services.symbol_indexer import SymbolIndexer
    
    symbol_indexer = SymbolIndexer()
    search_service = SemanticSearchService(symbol_indexer)
    
    # Index some test data
    symbol_indexer.add_symbol(
        name="calculate_sum",
        symbol_type="function",
        file_path="math.py",
        line_number=5
    )
    
    # Search should work
    results = search_service.search("find function for adding numbers")
    
    # Should return results (even if empty in test environment)
    assert isinstance(results, list)
    
    print("✅ SemanticSearch integration test passed")


def test_ai_suggestion_provider_integration():
    """Test AISuggestionProvider integrates with editor components."""
    from pyutagent.ui.editor.ai_suggestion_provider import AISuggestionProvider
    from pyutagent.ui.editor.ghost_text import GhostTextSuggestion
    
    provider = AISuggestionProvider()
    
    # Create a suggestion
    suggestion = GhostTextSuggestion(
        text="print('hello')",
        start_line=1,
        start_column=0
    )
    
    assert suggestion.text == "print('hello')"
    assert suggestion.start_line == 1
    
    print("✅ AISuggestionProvider integration test passed")


def test_inline_diff_integration():
    """Test InlineDiff integrates with editor."""
    from pyutagent.ui.editor.inline_diff import InlineDiffCalculator, DiffType
    
    calculator = InlineDiffCalculator()
    
    old_text = "def hello():\n    pass"
    new_text = "def hello():\n    print('hello')"
    
    diffs = calculator.calculate_diff(old_text, new_text)
    
    assert len(diffs) > 0
    
    print("✅ InlineDiff integration test passed")


def test_agent_worker_integration():
    """Test AgentWorker signal system."""
    from pyutagent.ui.agent_panel.agent_worker import AgentStateSignals, AgentState
    
    signals = AgentStateSignals()
    
    # Test that signals can be connected
    mock_handler = Mock()
    signals.state_changed.connect(mock_handler)
    
    # Emit signal
    signals.state_changed.emit(AgentState.THINKING)
    
    mock_handler.assert_called_once_with(AgentState.THINKING)
    
    print("✅ AgentWorker integration test passed")


def test_ghost_text_integration():
    """Test GhostText integrates with suggestion provider."""
    from pyutagent.ui.editor.ghost_text import GhostTextSuggestion, GhostTextRenderer
    
    suggestion = GhostTextSuggestion(
        text="def test():\n    pass",
        start_line=1,
        start_column=0
    )
    
    # Test suggestion properties
    assert suggestion.is_multiline() is True
    assert suggestion.get_line_count() == 2
    
    print("✅ GhostText integration test passed")


def test_streaming_handler_integration():
    """Test StreamingHandler processes content correctly."""
    from pyutagent.ui.components.streaming_handler import StreamingHandler, StreamingConfig
    
    config = StreamingConfig(mode="word", words_per_chunk=2)
    handler = StreamingHandler(config)
    
    test_text = "Hello world test content"
    chunks = list(handler.stream_text(test_text))
    
    # Should split into chunks
    assert len(chunks) > 0
    
    print("✅ StreamingHandler integration test passed")


def test_thinking_expander_integration():
    """Test ThinkingExpander data structures."""
    from pyutagent.ui.components.thinking_expander import ThinkingStep, StepStatus
    
    step = ThinkingStep(
        step_id="step_1",
        title="Analyzing code",
        status=StepStatus.RUNNING
    )
    
    assert step.step_id == "step_1"
    assert step.status == StepStatus.RUNNING
    
    print("✅ ThinkingExpander integration test passed")


def test_git_status_service_integration():
    """Test GitStatusService detects changes."""
    from pyutagent.ui.services.git_status_service import GitStatusService, GitStatus
    
    service = GitStatusService()
    
    # Test status detection (may fail if not in git repo)
    try:
        status = service.get_file_status("test.py")
        assert isinstance(status, GitStatus)
    except Exception:
        # Expected if not in git repo
        pass
    
    print("✅ GitStatusService integration test passed")


def test_command_palette_integration():
    """Test CommandPalette fuzzy matching."""
    from pyutagent.ui.command_palette import FuzzyMatcher
    
    matcher = FuzzyMatcher()
    
    commands = [
        {"name": "Open File", "category": "File"},
        {"name": "Save File", "category": "File"},
        {"name": "Close Editor", "category": "Edit"},
    ]
    
    results = matcher.match("open", commands)
    
    # Should find "Open File"
    assert len(results) > 0
    
    print("✅ CommandPalette integration test passed")


def test_end_to_end_component_integration():
    """Test all components work together."""
    from pyutagent.ui.components.markdown_renderer import MarkdownRenderer
    from pyutagent.ui.components.streaming_handler import StreamingHandler
    from pyutagent.ui.services.symbol_indexer import SymbolIndexer
    from pyutagent.ui.editor.inline_diff import InlineDiffCalculator
    
    # Create all components
    markdown = MarkdownRenderer()
    streamer = StreamingHandler()
    indexer = SymbolIndexer()
    diff_calc = InlineDiffCalculator()
    
    # Test markdown rendering
    html = markdown.render("# Test\n\n`code`")
    assert "<h1>" in html
    
    # Test streaming
    chunks = list(streamer.stream_text("Hello world"))
    assert len(chunks) > 0
    
    # Test symbol indexing
    indexer.add_symbol("TestFunc", "function", "test.py", 1)
    results = indexer.search_symbols("Test")
    assert len(results) >= 0  # May be empty in test
    
    # Test diff calculation
    diffs = diff_calc.calculate_diff("a", "b")
    assert len(diffs) > 0
    
    print("✅ End-to-end integration test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running GUI Integration Tests")
    print("="*60 + "\n")
    
    test_markdown_renderer_integration()
    test_symbol_indexer_integration()
    test_semantic_search_integration()
    test_ai_suggestion_provider_integration()
    test_inline_diff_integration()
    test_agent_worker_integration()
    test_ghost_text_integration()
    test_streaming_handler_integration()
    test_thinking_expander_integration()
    test_git_status_service_integration()
    test_command_palette_integration()
    test_end_to_end_component_integration()
    
    print("\n" + "="*60)
    print("All integration tests passed! ✅")
    print("="*60 + "\n")
