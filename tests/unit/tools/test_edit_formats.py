"""Tests for edit formats module."""

import pytest
from unittest.mock import Mock

from pyutagent.tools.edit_formats import (
    EditFormat,
    EditBlock,
    DiffParserFormat,
    UDiffParserFormat,
    WholeFileParserFormat,
    DiffFencedParserFormat,
    EditFormatRegistry,
    edit_format_registry,
)


class TestEditFormat:
    """Tests for EditFormat enum."""

    def test_edit_format_values(self):
        """Test that all edit formats are defined."""
        assert EditFormat.DIFF.value == "diff"
        assert EditFormat.UDIFF.value == "udiff"
        assert EditFormat.WHOLE.value == "whole"
        assert EditFormat.DIFF_FENCED.value == "diff-fenced"


class TestEditBlock:
    """Tests for EditBlock dataclass."""

    def test_edit_block_creation(self):
        """Test creating an EditBlock."""
        block = EditBlock(
            original="old code",
            modified="new code",
            file_path="/path/file.java",
            line_start=10,
            line_end=20
        )
        assert block.original == "old code"
        assert block.modified == "new code"
        assert block.file_path == "/path/file.java"
        assert block.line_start == 10
        assert block.line_end == 20

    def test_edit_block_defaults(self):
        """Test EditBlock with default values."""
        block = EditBlock(original="old", modified="new")
        assert block.file_path is None
        assert block.line_start is None
        assert block.line_end is None


class TestDiffParserFormat:
    """Tests for DiffParserFormat (SEARCH/REPLACE)."""

    @pytest.fixture
    def parser(self):
        """Create a DiffParserFormat instance."""
        return DiffParserFormat()

    def test_parse_single_block(self, parser):
        """Test parsing a single SEARCH/REPLACE block."""
        text = """<<<<<<< SEARCH
old code line 1
old code line 2
=======
new code line 1
new code line 2
>>>>>>> REPLACE"""

        edits = parser.parse(text)

        assert len(edits) == 1
        assert edits[0].original == "old code line 1\nold code line 2"
        assert edits[0].modified == "new code line 1\nnew code line 2"

    def test_parse_multiple_blocks(self, parser):
        """Test parsing multiple SEARCH/REPLACE blocks."""
        text = """<<<<<<< SEARCH
old code 1
=======
new code 1
>>>>>>> REPLACE

<<<<<<< SEARCH
old code 2
=======
new code 2
>>>>>>> REPLACE"""

        edits = parser.parse(text)

        assert len(edits) == 2
        assert edits[0].original == "old code 1"
        assert edits[0].modified == "new code 1"
        assert edits[1].original == "old code 2"
        assert edits[1].modified == "new code 2"

    def test_parse_with_file_path(self, parser):
        """Test parsing with file path context."""
        text = """<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE"""

        edits = parser.parse(text, file_path="/path/Test.java")

        assert len(edits) == 1
        assert edits[0].file_path == "/path/Test.java"

    def test_parse_no_blocks(self, parser):
        """Test parsing text with no SEARCH/REPLACE blocks."""
        text = "Some random text without blocks"
        edits = parser.parse(text)
        assert len(edits) == 0

    def test_format_prompt_instruction(self, parser):
        """Test getting prompt instruction."""
        instruction = parser.format_prompt_instruction()
        assert "SEARCH/REPLACE" in instruction
        assert "<<<<<<< SEARCH" in instruction
        assert "=======" in instruction
        assert ">>>>>>> REPLACE" in instruction


class TestUDiffParserFormat:
    """Tests for UDiffParserFormat (unified diff)."""

    @pytest.fixture
    def parser(self):
        """Create a UDiffParserFormat instance."""
        return UDiffParserFormat()

    def test_parse_simple_diff(self, parser):
        """Test parsing a simple unified diff."""
        text = """--- old.java
+++ new.java
@@ -10,3 +10,3 @@
 context line 1
-old line
+new line
 context line 2"""

        edits = parser.parse(text)

        assert len(edits) == 1
        assert "old line" in edits[0].original
        assert "new line" in edits[0].modified
        assert edits[0].line_start == 10

    def test_parse_multiple_hunks(self, parser):
        """Test parsing diff with multiple hunks."""
        text = """--- old.java
+++ new.java
@@ -10,2 +10,2 @@
-old1
+new1
@@ -20,2 +20,2 @@
-old2
+new2"""

        edits = parser.parse(text)

        assert len(edits) == 2
        assert edits[0].line_start == 10
        assert edits[1].line_start == 20

    def test_parse_no_newline_marker(self, parser):
        """Test parsing diff with no newline marker."""
        text = """--- old.java
+++ new.java
@@ -1,1 +1,1 @@
-old
+new
\\ No newline at end of file"""

        edits = parser.parse(text)

        assert len(edits) == 1
        assert "No newline" not in edits[0].original
        assert "No newline" not in edits[0].modified

    def test_format_prompt_instruction(self, parser):
        """Test getting prompt instruction."""
        instruction = parser.format_prompt_instruction()
        assert "unified diff" in instruction.lower()
        assert "---" in instruction
        assert "+++" in instruction
        assert "@@" in instruction


class TestWholeFileParserFormat:
    """Tests for WholeFileParserFormat."""

    @pytest.fixture
    def parser(self):
        """Create a WholeFileParserFormat instance."""
        return WholeFileParserFormat()

    def test_parse_fenced_code_block(self, parser):
        """Test parsing fenced code block."""
        text = """```java
public class Test {
    public void method() {}
}
```"""

        edits = parser.parse(text)

        assert len(edits) == 1
        assert edits[0].original == ""
        assert "public class Test" in edits[0].modified

    def test_parse_multiple_fenced_blocks(self, parser):
        """Test parsing multiple fenced code blocks."""
        text = """```java
class A {}
```

```java
class B {}
```"""

        edits = parser.parse(text)

        assert len(edits) == 2

    def test_parse_looks_like_code(self, parser):
        """Test parsing text that looks like code."""
        text = """public class Test {
    public void method() {}
}"""

        edits = parser.parse(text)

        assert len(edits) == 1
        assert edits[0].original == ""
        assert "public class Test" in edits[0].modified

    def test_parse_not_code(self, parser):
        """Test parsing text that doesn't look like code."""
        text = "This is just a description without code"
        edits = parser.parse(text)
        assert len(edits) == 0

    def test_looks_like_code_with_class(self, parser):
        """Test _looks_like_code with class keyword."""
        assert parser._looks_like_code("public class Test {}") is True
        assert parser._looks_like_code("private class Test {}") is True
        assert parser._looks_like_code("class Test {}") is True

    def test_looks_like_code_with_import(self, parser):
        """Test _looks_like_code with import."""
        assert parser._looks_like_code("import java.util.List;") is True

    def test_looks_like_code_with_package(self, parser):
        """Test _looks_like_code with package."""
        assert parser._looks_like_code("package com.example;") is True

    def test_format_prompt_instruction(self, parser):
        """Test getting prompt instruction."""
        instruction = parser.format_prompt_instruction()
        assert "complete" in instruction.lower()
        assert "```" in instruction


class TestDiffFencedParserFormat:
    """Tests for DiffFencedParserFormat."""

    @pytest.fixture
    def parser(self):
        """Create a DiffFencedParserFormat instance."""
        return DiffFencedParserFormat()

    def test_parse_fenced_diff(self, parser):
        """Test parsing fenced diff block."""
        text = """```diff
- removed line
+ added line
  context line
```"""

        edits = parser.parse(text)

        assert len(edits) == 1
        assert "removed line" in edits[0].original
        assert "added line" in edits[0].modified
        assert "context line" in edits[0].original
        assert "context line" in edits[0].modified

    def test_parse_multiple_fenced_diffs(self, parser):
        """Test parsing multiple fenced diff blocks."""
        text = """```diff
- old1
+ new1
```

```diff
- old2
+ new2
```"""

        edits = parser.parse(text)

        assert len(edits) == 2

    def test_format_prompt_instruction(self, parser):
        """Test getting prompt instruction."""
        instruction = parser.format_prompt_instruction()
        assert "diff" in instruction.lower()
        assert "```diff" in instruction
        assert "- " in instruction
        assert "+ " in instruction


class TestEditFormatRegistry:
    """Tests for EditFormatRegistry."""

    @pytest.fixture
    def registry(self):
        """Create an EditFormatRegistry instance."""
        return EditFormatRegistry()

    def test_initialization(self, registry):
        """Test registry initialization."""
        assert len(registry._parsers) == 4
        assert EditFormat.DIFF in registry._parsers
        assert EditFormat.UDIFF in registry._parsers
        assert EditFormat.WHOLE in registry._parsers
        assert EditFormat.DIFF_FENCED in registry._parsers

    def test_get_parser_diff(self, registry):
        """Test getting diff parser."""
        parser = registry.get_parser(EditFormat.DIFF)
        assert isinstance(parser, DiffParserFormat)

    def test_get_parser_udiff(self, registry):
        """Test getting udiff parser."""
        parser = registry.get_parser(EditFormat.UDIFF)
        assert isinstance(parser, UDiffParserFormat)

    def test_get_parser_whole(self, registry):
        """Test getting whole file parser."""
        parser = registry.get_parser(EditFormat.WHOLE)
        assert isinstance(parser, WholeFileParserFormat)

    def test_get_parser_diff_fenced(self, registry):
        """Test getting diff fenced parser."""
        parser = registry.get_parser(EditFormat.DIFF_FENCED)
        assert isinstance(parser, DiffFencedParserFormat)

    def test_get_parser_default(self, registry):
        """Test getting default parser for unknown format."""
        # Create a mock enum value
        mock_format = Mock()
        mock_format.value = "unknown"
        parser = registry.get_parser(mock_format)
        assert isinstance(parser, DiffParserFormat)  # Defaults to DIFF

    def test_get_preferred_format_openai(self, registry):
        """Test getting preferred format for OpenAI models."""
        assert registry.get_preferred_format("gpt-4") == EditFormat.DIFF
        assert registry.get_preferred_format("gpt-4-turbo") == EditFormat.UDIFF
        assert registry.get_preferred_format("gpt-4o") == EditFormat.DIFF
        assert registry.get_preferred_format("gpt-3.5-turbo") == EditFormat.DIFF

    def test_get_preferred_format_anthropic(self, registry):
        """Test getting preferred format for Anthropic models."""
        assert registry.get_preferred_format("claude-3-opus") == EditFormat.DIFF
        assert registry.get_preferred_format("claude-3-sonnet") == EditFormat.DIFF
        assert registry.get_preferred_format("claude-3-haiku") == EditFormat.DIFF

    def test_get_preferred_format_gemini(self, registry):
        """Test getting preferred format for Gemini models."""
        assert registry.get_preferred_format("gemini") == EditFormat.DIFF_FENCED
        assert registry.get_preferred_format("gemini-pro") == EditFormat.DIFF_FENCED

    def test_get_preferred_format_deepseek(self, registry):
        """Test getting preferred format for DeepSeek models."""
        assert registry.get_preferred_format("deepseek") == EditFormat.DIFF
        assert registry.get_preferred_format("deepseek-coder") == EditFormat.DIFF

    def test_get_preferred_format_ollama(self, registry):
        """Test getting preferred format for Ollama models."""
        assert registry.get_preferred_format("codellama") == EditFormat.WHOLE
        assert registry.get_preferred_format("llama2") == EditFormat.WHOLE

    def test_get_preferred_format_case_insensitive(self, registry):
        """Test that model name matching is case insensitive."""
        assert registry.get_preferred_format("GPT-4") == EditFormat.DIFF
        assert registry.get_preferred_format("Claude-3-Opus") == EditFormat.DIFF

    def test_get_preferred_format_partial_match(self, registry):
        """Test partial model name matching."""
        assert registry.get_preferred_format("my-gpt-4-model") == EditFormat.DIFF
        assert registry.get_preferred_format("custom-claude-3") == EditFormat.DIFF

    def test_get_preferred_format_default(self, registry):
        """Test default format for unknown models."""
        assert registry.get_preferred_format("unknown-model") == EditFormat.DIFF

    def test_get_preferred_format_mock_object(self, registry):
        """Test handling mock objects in tests."""
        mock_model = Mock()
        assert registry.get_preferred_format(mock_model) == EditFormat.DIFF

    def test_set_model_preference(self, registry):
        """Test setting model preference."""
        registry.set_model_preference("custom-model", EditFormat.WHOLE)
        assert registry.get_preferred_format("custom-model") == EditFormat.WHOLE

    def test_get_format_instruction(self, registry):
        """Test getting format instruction."""
        instruction = registry.get_format_instruction(EditFormat.DIFF)
        assert "SEARCH/REPLACE" in instruction

    def test_parse_edits(self, registry):
        """Test parsing edits with specific format."""
        text = """<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE"""

        edits = registry.parse_edits(text, EditFormat.DIFF)

        assert len(edits) == 1
        assert edits[0].original == "old"
        assert edits[0].modified == "new"

    def test_auto_detect_and_parse_diff(self, registry):
        """Test auto-detecting diff format."""
        text = """<<<<<<< SEARCH
old
=======
new
>>>>>>> REPLACE"""

        edits, fmt = registry.auto_detect_and_parse(text)

        assert len(edits) == 1
        assert fmt == EditFormat.DIFF

    def test_auto_detect_and_parse_fenced_diff(self, registry):
        """Test auto-detecting fenced diff format."""
        text = """```diff
- old
+ new
```"""

        edits, fmt = registry.auto_detect_and_parse(text)

        assert len(edits) == 1
        assert fmt == EditFormat.DIFF_FENCED

    def test_auto_detect_and_parse_whole(self, registry):
        """Test auto-detecting whole file format."""
        text = """```java
public class Test {}
```"""

        edits, fmt = registry.auto_detect_and_parse(text)

        assert len(edits) == 1
        assert fmt == EditFormat.WHOLE

    def test_auto_detect_and_parse_no_match(self, registry):
        """Test auto-detect when no format matches."""
        text = "Just some random text"

        edits, fmt = registry.auto_detect_and_parse(text)

        assert len(edits) == 0
        assert fmt == EditFormat.DIFF  # Default


class TestGlobalRegistry:
    """Tests for the global registry instance."""

    def test_global_registry_exists(self):
        """Test that global registry exists."""
        assert edit_format_registry is not None
        assert isinstance(edit_format_registry, EditFormatRegistry)

    def test_global_registry_has_parsers(self):
        """Test that global registry has all parsers."""
        assert len(edit_format_registry._parsers) == 4
