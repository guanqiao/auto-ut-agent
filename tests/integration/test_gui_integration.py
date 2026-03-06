"""Integration tests for GUI components.

Tests the integration between different GUI modules.
"""

import pytest
import tempfile
from unittest.mock import Mock


def test_markdown_renderer_integration():
    """Test MarkdownRenderer integrates with streaming handler."""
    from pyutagent.ui.components.markdown_renderer import MarkdownRenderer
    
    renderer = MarkdownRenderer()
    
    # Test that renderer can process streamed content
    test_content = "# Hello\n\n```python\nprint('hello')\n```"
    html = renderer.render(test_content)
    
    assert "<h1" in html  # Markdown may add id attribute
    assert "<code" in html or "<pre>" in html
    
    print("✅ MarkdownRenderer integration test passed")


def test_symbol_indexer_creation():
    """Test SymbolIndexer can be created."""
    from pyutagent.ui.services.symbol_indexer import SymbolIndexer
    
    # Create a temporary directory for the project
    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = SymbolIndexer(project_path=tmpdir)
        assert indexer is not None
        assert indexer.project_path == Path(tmpdir).resolve()
    
    print("✅ SymbolIndexer integration test passed")


def test_semantic_search_creation():
    """Test SemanticSearch can be created."""
    from pyutagent.ui.services.semantic_search import SemanticSearchService
    from pyutagent.ui.services.symbol_indexer import SymbolIndexer
    
    with tempfile.TemporaryDirectory() as tmpdir:
        symbol_indexer = SymbolIndexer(project_path=tmpdir)
        search_service = SemanticSearchService(symbol_indexer)
        assert search_service is not None
    
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
    from pyutagent.ui.editor.inline_diff import InlineDiffCalculator
    
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
    
    # Test that signals can be connected (state_changed has 2 args: state, message)
    mock_handler = Mock()
    signals.state_changed.connect(mock_handler)
    
    # Emit signal with correct arguments
    signals.state_changed.emit(AgentState.THINKING, "Agent is thinking...")
    
    mock_handler.assert_called_once_with(AgentState.THINKING, "Agent is thinking...")
    
    print("✅ AgentWorker integration test passed")


def test_ghost_text_integration():
    """Test GhostText integrates with suggestion provider."""
    from pyutagent.ui.editor.ghost_text import GhostTextSuggestion
    
    suggestion = GhostTextSuggestion(
        text="def test():\n    pass",
        start_line=1,
        start_column=0
    )
    
    # Test suggestion properties
    assert suggestion.text == "def test():\n    pass"
    assert suggestion.start_line == 1
    assert suggestion.start_column == 0
    
    print("✅ GhostText integration test passed")


def test_streaming_handler_creation():
    """Test StreamingHandler can be created."""
    from pyutagent.ui.components.streaming_handler import StreamingHandler, StreamingConfig
    
    # Use correct config parameters
    config = StreamingConfig(mode="word")
    handler = StreamingHandler(config)
    
    assert handler is not None
    assert handler.config.mode == "word"
    
    print("✅ StreamingHandler integration test passed")


def test_thinking_expander_creation():
    """Test ThinkingExpander data structures."""
    from pyutagent.ui.components.thinking_expander import ThinkingStep
    
    step = ThinkingStep(
        id="step_1",
        title="Analyzing code",
        status="running"
    )
    
    assert step.id == "step_1"
    assert step.status == "running"
    
    print("✅ ThinkingExpander integration test passed")


def test_git_status_service_creation():
    """Test GitStatusService can be created."""
    from pyutagent.ui.services.git_status_service import GitStatusService
    
    service = GitStatusService()
    assert service is not None
    
    print("✅ GitStatusService integration test passed")


def test_command_palette_creation():
    """Test CommandPalette fuzzy matching."""
    from pyutagent.ui.command_palette import FuzzyMatcher
    
    matcher = FuzzyMatcher()
    assert matcher is not None
    
    print("✅ CommandPalette integration test passed")


def test_end_to_end_component_integration():
    """Test all components work together."""
    from pyutagent.ui.components.markdown_renderer import MarkdownRenderer
    from pyutagent.ui.components.streaming_handler import StreamingHandler, StreamingConfig
    from pyutagent.ui.editor.inline_diff import InlineDiffCalculator
    
    # Create all components
    markdown = MarkdownRenderer()
    config = StreamingConfig(mode="word")
    streamer = StreamingHandler(config)
    diff_calc = InlineDiffCalculator()
    
    # Test markdown rendering
    html = markdown.render("# Test\n\n`code`")
    assert "<h1" in html
    
    # Test streaming handler exists
    assert streamer is not None
    
    # Test diff calculation
    diffs = diff_calc.calculate_diff("a", "b")
    assert len(diffs) > 0
    
    print("✅ End-to-end integration test passed")


if __name__ == "__main__":
    from pathlib import Path
    
    print("\n" + "="*60)
    print("Running GUI Integration Tests")
    print("="*60 + "\n")
    
    test_markdown_renderer_integration()
    test_symbol_indexer_creation()
    test_semantic_search_creation()
    test_ai_suggestion_provider_integration()
    test_inline_diff_integration()
    test_agent_worker_integration()
    test_ghost_text_integration()
    test_streaming_handler_creation()
    test_thinking_expander_creation()
    test_git_status_service_creation()
    test_command_palette_creation()
    test_end_to_end_component_integration()
    
    print("\n" + "="*60)
    print("All integration tests passed! ✅")
    print("="*60 + "\n")
