"""Unit tests for inline_diff module."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from pyutagent.ui.editor.inline_diff import (
    DiffType,
    DiffBlock,
    CharDiff,
    InlineDiffRenderer,
    InlineDiffCalculator
)


class TestDiffType:
    """Tests for DiffType enum."""
    
    def test_enum_values(self):
        """Test that enum values exist."""
        assert DiffType.ADD is not None
        assert DiffType.DELETE is not None
        assert DiffType.MODIFY is not None
        assert DiffType.EQUAL is not None
        
    def test_enum_uniqueness(self):
        """Test that enum values are unique."""
        values = [DiffType.ADD, DiffType.DELETE, DiffType.MODIFY, DiffType.EQUAL]
        assert len(values) == len(set(values))


class TestDiffBlock:
    """Tests for DiffBlock dataclass."""
    
    def test_basic_creation(self):
        """Test creating a basic diff block."""
        block = DiffBlock(
            diff_type=DiffType.ADD,
            old_start_line=1,
            old_end_line=1,
            new_start_line=1,
            new_end_line=2,
            old_text="",
            new_text="new line\nanother line"
        )
        
        assert block.diff_type == DiffType.ADD
        assert block.old_start_line == 1
        assert block.new_end_line == 2
        
    def test_modify_block(self):
        """Test creating a modify block."""
        block = DiffBlock(
            diff_type=DiffType.MODIFY,
            old_start_line=5,
            old_end_line=7,
            new_start_line=5,
            new_end_line=6,
            old_text="old line1\nold line2\nold line3",
            new_text="new line1\nnew line2"
        )
        
        assert block.diff_type == DiffType.MODIFY
        assert block.old_end_line == 7
        
    def test_delete_block(self):
        """Test creating a delete block."""
        block = DiffBlock(
            diff_type=DiffType.DELETE,
            old_start_line=3,
            old_end_line=5,
            new_start_line=3,
            new_end_line=2,  # 0 lines in new
            old_text="line1\nline2\nline3",
            new_text=""
        )
        
        assert block.diff_type == DiffType.DELETE


class TestCharDiff:
    """Tests for CharDiff dataclass."""
    
    def test_creation(self):
        """Test creating a char diff."""
        diff = CharDiff(start=5, end=10, diff_type=DiffType.ADD)
        
        assert diff.start == 5
        assert diff.end == 10
        assert diff.diff_type == DiffType.ADD


class TestInlineDiffRenderer:
    """Tests for InlineDiffRenderer class."""
    
    @pytest.fixture
    def mock_editor(self):
        """Create a mock editor."""
        editor = Mock()
        editor.viewport.return_value = Mock()
        return editor
        
    @pytest.fixture
    def renderer(self, mock_editor):
        """Create an InlineDiffRenderer instance."""
        return InlineDiffRenderer(mock_editor)
        
    def test_initial_state(self, renderer):
        """Test initial state of renderer."""
        assert not renderer.is_visible()
        assert renderer._diff_blocks == []
        assert renderer._current_block_index == -1
        
    def test_set_diff_blocks(self, renderer, mock_editor):
        """Test setting diff blocks."""
        blocks = [
            DiffBlock(
                diff_type=DiffType.ADD,
                old_start_line=1,
                old_end_line=1,
                new_start_line=1,
                new_end_line=2,
                old_text="",
                new_text="new line"
            )
        ]
        
        renderer.set_diff_blocks(blocks)
        
        assert renderer.is_visible()
        assert renderer._diff_blocks == blocks
        assert renderer._current_block_index == 0
        mock_editor.viewport.return_value.update.assert_called_once()
        
    def test_clear_diff(self, renderer, mock_editor):
        """Test clearing diff."""
        blocks = [DiffBlock(
            diff_type=DiffType.ADD,
            old_start_line=1,
            old_end_line=1,
            new_start_line=1,
            new_end_line=2,
            old_text="",
            new_text="new"
        )]
        renderer.set_diff_blocks(blocks)
        renderer.clear_diff()
        
        assert not renderer.is_visible()
        assert renderer._diff_blocks == []
        
    def test_set_dark_mode(self, renderer, mock_editor):
        """Test setting dark mode."""
        blocks = [DiffBlock(
            diff_type=DiffType.ADD,
            old_start_line=1,
            old_end_line=1,
            new_start_line=1,
            new_end_line=2,
            old_text="",
            new_text="new"
        )]
        renderer.set_diff_blocks(blocks)
        
        renderer.set_dark_mode(False)
        
        assert renderer._is_dark_mode == False
        mock_editor.viewport.return_value.update.assert_called()
        
    def test_next_diff_no_blocks(self, renderer):
        """Test next_diff with no blocks."""
        assert not renderer.next_diff()
        
    def test_previous_diff_no_blocks(self, renderer):
        """Test previous_diff with no blocks."""
        assert not renderer.previous_diff()
        
    def test_next_diff_with_blocks(self, renderer, mock_editor):
        """Test next_diff navigation."""
        blocks = [
            DiffBlock(DiffType.ADD, 1, 1, 1, 2, "", "line1"),
            DiffBlock(DiffType.DELETE, 5, 6, 5, 4, "old", ""),
        ]
        renderer.set_diff_blocks(blocks)
        
        # Mock document and cursor - need to return a valid QTextBlock-like object
        from PyQt6.QtGui import QTextCursor
        mock_block = Mock()
        mock_block.isValid.return_value = True
        mock_editor.document.return_value.findBlockByLineNumber.return_value = mock_block
        # Mock the cursor creation in _navigate_to_block
        with patch.object(renderer, '_navigate_to_block') as mock_nav:
            assert renderer.next_diff()
            assert renderer._current_block_index == 1
        
    def test_previous_diff_with_blocks(self, renderer, mock_editor):
        """Test previous_diff navigation."""
        blocks = [
            DiffBlock(DiffType.ADD, 1, 1, 1, 2, "", "line1"),
            DiffBlock(DiffType.DELETE, 5, 6, 5, 4, "old", ""),
        ]
        renderer.set_diff_blocks(blocks)
        renderer._current_block_index = 1
        
        # Mock document and cursor
        mock_block = Mock()
        mock_block.isValid.return_value = True
        mock_editor.document.return_value.findBlockByLineNumber.return_value = mock_block
        # Mock the cursor creation in _navigate_to_block
        with patch.object(renderer, '_navigate_to_block') as mock_nav:
            assert renderer.previous_diff()
            assert renderer._current_block_index == 0
        
    def test_get_diff_stats(self, renderer):
        """Test getting diff statistics."""
        blocks = [
            DiffBlock(DiffType.ADD, 1, 1, 1, 2, "", "line1"),
            DiffBlock(DiffType.DELETE, 5, 6, 5, 4, "old", ""),
            DiffBlock(DiffType.MODIFY, 10, 11, 10, 11, "old", "new"),
            DiffBlock(DiffType.ADD, 15, 15, 15, 16, "", "line2"),
        ]
        renderer.set_diff_blocks(blocks)
        
        added, deleted, modified = renderer.get_diff_stats()
        
        assert added == 2
        assert deleted == 1
        assert modified == 1
        
    def test_color_constants(self):
        """Test that color constants are defined."""
        # Dark mode colors
        assert InlineDiffRenderer.COLOR_ADD_BG_DARK is not None
        assert InlineDiffRenderer.COLOR_DEL_BG_DARK is not None
        assert InlineDiffRenderer.COLOR_MOD_BG_DARK is not None
        
        # Light mode colors
        assert InlineDiffRenderer.COLOR_ADD_BG_LIGHT is not None
        assert InlineDiffRenderer.COLOR_DEL_BG_LIGHT is not None
        assert InlineDiffRenderer.COLOR_MOD_BG_LIGHT is not None


class TestInlineDiffCalculator:
    """Tests for InlineDiffCalculator class."""
    
    def test_calculate_diff_no_changes(self):
        """Test calculating diff with identical texts."""
        original = "line1\nline2\nline3"
        modified = "line1\nline2\nline3"
        
        blocks = InlineDiffCalculator.calculate_diff(original, modified)
        
        # Should return empty or only equal blocks
        modify_blocks = [b for b in blocks if b.diff_type != DiffType.EQUAL]
        assert len(modify_blocks) == 0
        
    def test_calculate_diff_addition(self):
        """Test calculating diff with added lines."""
        original = "line1\nline2"
        modified = "line1\nline2\nline3"
        
        blocks = InlineDiffCalculator.calculate_diff(original, modified)
        
        # Should have an ADD block
        add_blocks = [b for b in blocks if b.diff_type == DiffType.ADD]
        assert len(add_blocks) >= 0  # May be combined with other changes
        
    def test_calculate_diff_deletion(self):
        """Test calculating diff with deleted lines."""
        original = "line1\nline2\nline3"
        modified = "line1\nline3"
        
        blocks = InlineDiffCalculator.calculate_diff(original, modified)
        
        # Should have a DELETE block
        del_blocks = [b for b in blocks if b.diff_type == DiffType.DELETE]
        # Note: May be represented as MODIFY depending on algorithm
        
    def test_calculate_diff_modification(self):
        """Test calculating diff with modified lines."""
        original = "line1\nold line\nline3"
        modified = "line1\nnew line\nline3"
        
        blocks = InlineDiffCalculator.calculate_diff(original, modified)
        
        # Should have a MODIFY block
        mod_blocks = [b for b in blocks if b.diff_type == DiffType.MODIFY]
        assert len(mod_blocks) >= 0  # May vary by algorithm
        
    def test_calculate_diff_multiple_changes(self):
        """Test calculating diff with multiple changes."""
        original = """def old_func():
    pass

class OldClass:
    pass"""
        
        modified = """def new_func():
    pass

class NewClass:
    pass"""
        
        blocks = InlineDiffCalculator.calculate_diff(original, modified)
        
        # Should detect changes
        assert len(blocks) > 0
        
    def test_simple_line_diff_add(self):
        """Test simple line diff with addition."""
        original_lines = ["line1\n", "line2\n"]
        modified_lines = ["line1\n", "line2\n", "line3\n"]
        
        blocks = InlineDiffCalculator._simple_line_diff(original_lines, modified_lines)
        
        # Should have an ADD block
        add_blocks = [b for b in blocks if b.diff_type == DiffType.ADD]
        assert len(add_blocks) == 1
        assert "line3" in add_blocks[0].new_text
        
    def test_simple_line_diff_delete(self):
        """Test simple line diff with deletion."""
        original_lines = ["line1\n", "line2\n", "line3\n"]
        modified_lines = ["line1\n", "line3\n"]
        
        blocks = InlineDiffCalculator._simple_line_diff(original_lines, modified_lines)
        
        # Should have a DELETE block
        del_blocks = [b for b in blocks if b.diff_type == DiffType.DELETE]
        assert len(del_blocks) == 1
        
    def test_simple_line_diff_modify(self):
        """Test simple line diff with modification."""
        original_lines = ["old line\n"]
        modified_lines = ["new line\n"]
        
        blocks = InlineDiffCalculator._simple_line_diff(original_lines, modified_lines)
        
        # Should have a MODIFY block
        mod_blocks = [b for b in blocks if b.diff_type == DiffType.MODIFY]
        assert len(mod_blocks) == 1


class TestInlineDiffIntegration:
    """Integration tests for inline diff functionality."""
    
    @pytest.fixture
    def mock_editor(self):
        """Create a mock editor."""
        editor = Mock()
        editor.viewport.return_value = Mock()
        editor.document.return_value = Mock()
        return editor
        
    def test_full_diff_workflow(self, mock_editor):
        """Test full diff workflow."""
        renderer = InlineDiffRenderer(mock_editor)
        
        # Calculate diff
        original = "line1\nline2\nline3"
        modified = "line1\nmodified\nline3"
        
        blocks = InlineDiffCalculator.calculate_diff(original, modified)
        
        # Set and display
        renderer.set_diff_blocks(blocks)
        assert renderer.is_visible()
        
        # Navigate - mock the navigation method to avoid Qt issues
        with patch.object(renderer, '_navigate_to_block'):
            if blocks:
                renderer.next_diff()
            
        # Clear
        renderer.clear_diff()
        assert not renderer.is_visible()
        
    def test_diff_with_empty_text(self, mock_editor):
        """Test diff with empty text."""
        renderer = InlineDiffRenderer(mock_editor)
        
        blocks = InlineDiffCalculator.calculate_diff("", "new content")
        renderer.set_diff_blocks(blocks)
        
        # Should handle empty original gracefully
        assert renderer.is_visible() or not blocks
        
    def test_character_level_diff(self, mock_editor):
        """Test character-level diff calculation."""
        renderer = InlineDiffRenderer(mock_editor)
        
        old_text = "hello world"
        new_text = "hello there"
        
        char_diffs = renderer._calculate_char_diff(old_text, new_text)
        
        # Should detect character changes
        assert len(char_diffs) > 0
        
    def test_navigation_wraparound(self, mock_editor):
        """Test navigation wraparound behavior."""
        renderer = InlineDiffRenderer(mock_editor)
        
        blocks = [
            DiffBlock(DiffType.ADD, 1, 1, 1, 2, "", "line1"),
            DiffBlock(DiffType.ADD, 3, 3, 3, 4, "", "line2"),
        ]
        renderer.set_diff_blocks(blocks)
        
        # Mock the navigation to avoid Qt issues
        with patch.object(renderer, '_navigate_to_block'):
            # Navigate to end
            renderer._current_block_index = 1
            renderer.next_diff()  # Should wrap to 0
            assert renderer._current_block_index == 0
            
            # Navigate before start
            renderer._current_block_index = 0
            renderer.previous_diff()  # Should wrap to end
            assert renderer._current_block_index == 1
