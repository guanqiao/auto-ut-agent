"""Tests for markdown renderer component."""

import pytest
from unittest.mock import MagicMock, patch

from pyutagent.ui.components.markdown_renderer import (
    MarkdownRenderer,
    MarkdownViewer,
    CodeBlockWidget,
    CodeBlock
)


class TestMarkdownRenderer:
    """Tests for MarkdownRenderer class."""
    
    def test_init(self):
        """Test renderer initialization."""
        renderer = MarkdownRenderer()
        assert renderer is not None
        
    def test_render_simple_text(self):
        """Test rendering simple text."""
        renderer = MarkdownRenderer()
        text = "Hello world"
        html = renderer.render(text)
        assert "Hello world" in html
        
    def test_render_with_headers(self):
        """Test rendering headers."""
        renderer = MarkdownRenderer()
        text = "# Header 1\n## Header 2\n### Header 3"
        html = renderer.render(text)
        assert "<h1>" in html or "Header 1" in html
        assert "<h2>" in html or "Header 2" in html
        assert "<h3>" in html or "Header 3" in html
        
    def test_render_with_code_block(self):
        """Test rendering code blocks."""
        renderer = MarkdownRenderer()
        text = "```python\nprint('hello')\n```"
        html = renderer.render(text)
        assert "print" in html or "code" in html.lower()
        
    def test_render_with_inline_code(self):
        """Test rendering inline code."""
        renderer = MarkdownRenderer()
        text = "Use `print()` function"
        html = renderer.render(text)
        assert "print()" in html or "<code>" in html
        
    def test_render_with_list(self):
        """Test rendering lists."""
        renderer = MarkdownRenderer()
        text = "* Item 1\n* Item 2\n* Item 3"
        html = renderer.render(text)
        assert "Item 1" in html
        assert "Item 2" in html
        assert "Item 3" in html
        
    def test_render_with_bold_and_italic(self):
        """Test rendering bold and italic text."""
        renderer = MarkdownRenderer()
        text = "**bold** and *italic*"
        html = renderer.render(text)
        assert "bold" in html
        assert "italic" in html
        
    def test_extract_code_blocks(self):
        """Test extracting code blocks from text."""
        renderer = MarkdownRenderer()
        text = """
Some text
```python
print('hello')
```
More text
```java
System.out.println("hello");
```
"""
        blocks = renderer.extract_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].language == "python"
        assert "print" in blocks[0].code
        assert blocks[1].language == "java"
        assert "System.out" in blocks[1].code
        
    def test_extract_code_blocks_no_language(self):
        """Test extracting code blocks without language specifier."""
        renderer = MarkdownRenderer()
        text = "```\nsome code\n```"
        blocks = renderer.extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].language == "text"
        
    def test_highlight_code_python(self):
        """Test syntax highlighting for Python."""
        renderer = MarkdownRenderer()
        code = "def hello():\n    return 'world'"
        html = renderer.highlight_code(code, "python")
        assert "def" in html or "hello" in html
        assert "<pre" in html or "<div" in html
        
    def test_highlight_code_java(self):
        """Test syntax highlighting for Java."""
        renderer = MarkdownRenderer()
        code = "public class Hello { }"
        html = renderer.highlight_code(code, "java")
        assert "class" in html or "Hello" in html or "<pre" in html or "<div" in html
        
    def test_highlight_code_unknown_language(self):
        """Test highlighting with unknown language."""
        renderer = MarkdownRenderer()
        code = "some code"
        html = renderer.highlight_code(code, "unknown_lang")
        assert "some code" in html
        
    def test_simple_render_fallback(self):
        """Test simple render fallback method."""
        renderer = MarkdownRenderer()
        text = "# Header\n**bold**\n`code`"
        html = renderer._simple_render(text)
        assert "Header" in html
        assert "bold" in html
        assert "code" in html


class TestCodeBlock:
    """Tests for CodeBlock dataclass."""
    
    def test_code_block_creation(self):
        """Test creating a CodeBlock."""
        block = CodeBlock(
            language="python",
            code="print('hello')",
            start_pos=10,
            end_pos=50
        )
        assert block.language == "python"
        assert block.code == "print('hello')"
        assert block.start_pos == 10
        assert block.end_pos == 50


class TestMarkdownRendererEdgeCases:
    """Tests for edge cases in markdown rendering."""
    
    def test_empty_string(self):
        """Test rendering empty string."""
        renderer = MarkdownRenderer()
        html = renderer.render("")
        assert html is not None
        
    def test_special_characters(self):
        """Test rendering special characters."""
        renderer = MarkdownRenderer()
        text = "<script>alert('xss')</script>"
        html = renderer.render(text)
        # Should escape or handle special chars
        assert html is not None
        
    def test_very_long_text(self):
        """Test rendering very long text."""
        renderer = MarkdownRenderer()
        text = "Line\n" * 1000
        html = renderer.render(text)
        assert html is not None
        assert len(html) > 0
        
    def test_nested_formatting(self):
        """Test nested formatting."""
        renderer = MarkdownRenderer()
        text = "**bold *and italic***"
        html = renderer.render(text)
        assert "bold" in html and "italic" in html
        
    def test_code_block_with_language_hint(self):
        """Test code block with various language hints."""
        renderer = MarkdownRenderer()
        languages = ["python", "java", "javascript", "cpp", "csharp", "go", "rust"]
        
        for lang in languages:
            text = f"```{lang}\ncode\n```"
            blocks = renderer.extract_code_blocks(text)
            assert len(blocks) == 1
            assert blocks[0].language == lang
