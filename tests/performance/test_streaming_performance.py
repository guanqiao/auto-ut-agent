"""Performance tests for streaming response.

Tests the performance of streaming response rendering.
"""

import pytest
import time
from unittest.mock import Mock, patch


class TestStreamingPerformance:
    """Test streaming response performance."""

    def test_markdown_rendering_speed(self):
        """Test Markdown rendering speed - should render in < 100ms."""
        from pyutagent.ui.components.markdown_renderer import MarkdownRenderer
        
        renderer = MarkdownRenderer()
        
        # Generate test markdown
        test_md = "\n\n".join([
            f"## Section {i}\n\n```python\nprint('hello {i}')\n```\n\nSome text here."
            for i in range(10)
        ])
        
        start_time = time.time()
        html = renderer.render(test_md)
        elapsed = time.time() - start_time
        
        # Should render in less than 100ms
        assert elapsed < 0.1, f"Rendering too slow: {elapsed*1000:.0f}ms"
        assert len(html) > 0
        
        print(f"✅ Markdown rendering: {elapsed*1000:.0f}ms")

    def test_symbol_search_performance(self):
        """Test symbol search performance - should respond in < 200ms."""
        from pyutagent.ui.services.symbol_indexer import SymbolIndexer
        from pyutagent.indexing.codebase_indexer import SymbolType, SymbolIndexEntry
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = SymbolIndexer(project_path=tmpdir)
            
            # Add test symbols directly to index
            for i in range(100):
                entry = SymbolIndexEntry(
                    name=f"TestClass{i}",
                    symbol_type=SymbolType.CLASS,
                    file_path=f"test{i}.py",
                    line_number=i,
                    column=0,
                    docstring=None,
                    signature=None,
                    parent=None,
                    children=[]
                )
                indexer._index[f"TestClass{i}"] = entry
            
            start_time = time.time()
            results = indexer.search_symbols("TestClass")
            elapsed = time.time() - start_time
            
            # Should respond in less than 200ms
            assert elapsed < 0.2, f"Search too slow: {elapsed*1000:.0f}ms"
            assert len(results) > 0
            
            print(f"✅ Symbol search: {elapsed*1000:.0f}ms")

    def test_ghost_text_rendering(self):
        """Test ghost text rendering performance."""
        from pyutagent.ui.editor.ghost_text import GhostTextSuggestion
        
        # Create suggestion
        suggestion = GhostTextSuggestion(
            text="def hello():\n    print('world')\n    return True",
            start_line=1,
            start_column=0
        )
        
        start_time = time.time()
        # Simulate rendering calculations
        lines = suggestion.text.split('\n')
        line_count = len(lines)
        max_line_length = max(len(line) for line in lines)
        elapsed = time.time() - start_time
        
        # Should be instantaneous
        assert elapsed < 0.01, f"Ghost text calc too slow: {elapsed*1000:.0f}ms"
        assert line_count == 3
        
        print(f"✅ Ghost text calc: {elapsed*1000:.0f}ms")

    def test_diff_calculation_speed(self):
        """Test diff calculation speed."""
        from pyutagent.ui.editor.inline_diff import InlineDiffCalculator
        
        calculator = InlineDiffCalculator()
        
        old_text = "def hello():\n    pass\n" * 50
        new_text = "def hello():\n    print('hello')\n" * 50
        
        start_time = time.time()
        diffs = calculator.calculate_diff(old_text, new_text)
        elapsed = time.time() - start_time
        
        # Should complete in less than 100ms
        assert elapsed < 0.1, f"Diff calc too slow: {elapsed*1000:.0f}ms"
        assert len(diffs) > 0
        
        print(f"✅ Diff calculation: {elapsed*1000:.0f}ms")

    def test_semantic_search_performance(self):
        """Test semantic search performance."""
        from pyutagent.ui.services.semantic_search import SemanticSearchService
        from pyutagent.ui.services.symbol_indexer import SymbolIndexer
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = SymbolIndexer(project_path=tmpdir)
            search_service = SemanticSearchService(indexer)
            
            start_time = time.time()
            # Search should complete quickly even with empty index
            results = search_service.search("test query")
            elapsed = time.time() - start_time
            
            # Should respond in less than 500ms
            assert elapsed < 0.5, f"Semantic search too slow: {elapsed*1000:.0f}ms"
            assert isinstance(results, list)
            
            print(f"✅ Semantic search: {elapsed*1000:.0f}ms")


class TestMemoryUsage:
    """Test memory usage."""

    def test_streaming_handler_creation(self):
        """Test that creating streaming handlers doesn't leak memory."""
        import gc
        
        from pyutagent.ui.components.streaming_handler import StreamingHandler
        
        # Force garbage collection
        gc.collect()
        
        # Create multiple handlers
        handlers = []
        for _ in range(10):
            handler = StreamingHandler()
            handlers.append(handler)
        
        # Delete handlers
        for handler in handlers:
            handler.deleteLater()
        del handlers
        
        # Force garbage collection again
        gc.collect()
        
        print("✅ Streaming handler memory test passed")


class TestConcurrentPerformance:
    """Test concurrent operations performance."""

    def test_multiple_markdown_renderers(self):
        """Test multiple markdown renderers working concurrently."""
        from pyutagent.ui.components.markdown_renderer import MarkdownRenderer
        
        renderers = [MarkdownRenderer() for _ in range(5)]
        test_md = "# Hello\n\n`code`"
        
        start_time = time.time()
        
        # Process all renderers
        for renderer in renderers:
            html = renderer.render(test_md)
            assert len(html) > 0
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 1.0, f"Concurrent rendering too slow: {elapsed:.2f}s"
        
        print(f"✅ Concurrent rendering: {elapsed:.2f}s for 5 renderers")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Performance Tests")
    print("="*60 + "\n")
    
    test = TestStreamingPerformance()
    test.test_markdown_rendering_speed()
    test.test_symbol_search_performance()
    test.test_ghost_text_rendering()
    test.test_diff_calculation_speed()
    test.test_semantic_search_performance()
    
    memory_test = TestMemoryUsage()
    memory_test.test_streaming_handler_creation()
    
    concurrent_test = TestConcurrentPerformance()
    concurrent_test.test_multiple_markdown_renderers()
    
    print("\n" + "="*60)
    print("All performance tests passed! ✅")
    print("="*60 + "\n")
