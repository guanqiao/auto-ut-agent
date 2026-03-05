"""Inline diff highlighting for code editor.

Provides visual diff highlighting directly in the code editor,
similar to Cursor's inline diff feature.
"""

import logging
from typing import Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum, auto
import difflib

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import (
    QPainter, QColor, QTextCursor, QTextFormat, 
    QTextCharFormat, QFontMetrics
)

logger = logging.getLogger(__name__)


class DiffType(Enum):
    """Type of diff change."""
    ADD = auto()      # Added lines (green)
    DELETE = auto()   # Deleted lines (red)
    MODIFY = auto()   # Modified lines (yellow/orange)
    EQUAL = auto()    # Unchanged lines


@dataclass
class DiffBlock:
    """Represents a block of diff changes."""
    diff_type: DiffType
    old_start_line: int
    old_end_line: int
    new_start_line: int
    new_end_line: int
    old_text: str
    new_text: str


@dataclass
class CharDiff:
    """Character-level diff within a line."""
    start: int
    end: int
    diff_type: DiffType


class InlineDiffRenderer:
    """Renders inline diff highlighting on the editor.
    
    Features:
    - Line background highlighting (green/red/yellow)
    - Character-level diff highlighting
    - Diff navigation (next/prev change)
    """
    
    # Colors for dark mode
    COLOR_ADD_BG_DARK = QColor(46, 125, 50, 60)      # Green with alpha
    COLOR_ADD_FG_DARK = QColor(129, 199, 132)
    COLOR_DEL_BG_DARK = QColor(211, 47, 47, 60)      # Red with alpha
    COLOR_DEL_FG_DARK = QColor(239, 154, 154)
    COLOR_MOD_BG_DARK = QColor(245, 124, 0, 60)      # Orange with alpha
    COLOR_MOD_FG_DARK = QColor(255, 183, 77)
    
    # Colors for light mode
    COLOR_ADD_BG_LIGHT = QColor(200, 230, 201, 150)
    COLOR_ADD_FG_LIGHT = QColor(27, 94, 32)
    COLOR_DEL_BG_LIGHT = QColor(255, 205, 210, 150)
    COLOR_DEL_FG_LIGHT = QColor(183, 28, 28)
    COLOR_MOD_BG_LIGHT = QColor(255, 224, 178, 150)
    COLOR_MOD_FG_LIGHT = QColor(230, 81, 0)
    
    def __init__(self, editor: 'CodeEditor'):
        self._editor = editor
        self._diff_blocks: List[DiffBlock] = []
        self._current_block_index: int = -1
        self._is_visible: bool = False
        self._is_dark_mode: bool = True
        
    def set_diff_blocks(self, blocks: List[DiffBlock]):
        """Set the diff blocks to display.
        
        Args:
            blocks: List of DiffBlock objects
        """
        self._diff_blocks = blocks
        self._current_block_index = -1 if not blocks else 0
        self._is_visible = len(blocks) > 0
        self._editor.viewport().update()
        logger.debug(f"Set {len(blocks)} diff blocks")
        
    def clear_diff(self):
        """Clear all diff highlighting."""
        self._diff_blocks = []
        self._current_block_index = -1
        self._is_visible = False
        self._editor.viewport().update()
        logger.debug("Diff cleared")
        
    def is_visible(self) -> bool:
        """Check if diff is currently visible."""
        return self._is_visible
        
    def set_dark_mode(self, dark_mode: bool):
        """Set the color scheme based on dark mode."""
        self._is_dark_mode = dark_mode
        if self._is_visible:
            self._editor.viewport().update()
            
    def render(self, painter: QPainter):
        """Render the diff highlighting.
        
        Args:
            painter: The QPainter to use
        """
        if not self._is_visible or not self._diff_blocks:
            return
            
        painter.save()
        
        for block in self._diff_blocks:
            self._render_diff_block(painter, block)
            
        painter.restore()
        
    def _render_diff_block(self, painter: QPainter, block: DiffBlock):
        """Render a single diff block."""
        # Select colors based on diff type and theme
        if block.diff_type == DiffType.ADD:
            bg_color = self.COLOR_ADD_BG_DARK if self._is_dark_mode else self.COLOR_ADD_BG_LIGHT
            fg_color = self.COLOR_ADD_FG_DARK if self._is_dark_mode else self.COLOR_ADD_FG_LIGHT
        elif block.diff_type == DiffType.DELETE:
            bg_color = self.COLOR_DEL_BG_DARK if self._is_dark_mode else self.COLOR_DEL_BG_LIGHT
            fg_color = self.COLOR_DEL_FG_DARK if self._is_dark_mode else self.COLOR_DEL_FG_LIGHT
        elif block.diff_type == DiffType.MODIFY:
            bg_color = self.COLOR_MOD_BG_DARK if self._is_dark_mode else self.COLOR_MOD_BG_LIGHT
            fg_color = self.COLOR_MOD_FG_DARK if self._is_dark_mode else self.COLOR_MOD_FG_LIGHT
        else:
            return
            
        # Draw line backgrounds
        if block.diff_type == DiffType.ADD:
            for line_num in range(block.new_start_line, block.new_end_line + 1):
                self._draw_line_background(painter, line_num, bg_color)
        elif block.diff_type == DiffType.DELETE:
            # For deleted lines, we show them as ghost lines or highlight the gap
            self._draw_deleted_lines_indicator(painter, block.old_start_line, block.old_end_line, bg_color)
        elif block.diff_type == DiffType.MODIFY:
            # Highlight modified lines
            for line_num in range(block.new_start_line, block.new_end_line + 1):
                self._draw_line_background(painter, line_num, bg_color)
                
            # Draw character-level diff
            self._draw_char_diff(painter, block)
            
    def _draw_line_background(self, painter: QPainter, line_num: int, color: QColor):
        """Draw background highlight for a line."""
        block = self._editor.document().findBlockByLineNumber(line_num - 1)
        if not block.isValid():
            return
            
        layout = block.layout()
        if not layout:
            return
            
        # Get the bounding rect of the line
        cursor = QTextCursor(block)
        rect = self._editor.cursorRect(cursor)
        
        # Extend to full width
        viewport_rect = self._editor.viewport().rect()
        full_rect = QRect(0, rect.top(), viewport_rect.width(), rect.height())
        
        painter.fillRect(full_rect, color)
        
    def _draw_deleted_lines_indicator(self, painter: QPainter, start_line: int, end_line: int, color: QColor):
        """Draw indicator for deleted lines."""
        # Draw a vertical bar or indicator where lines were deleted
        block = self._editor.document().findBlockByLineNumber(start_line - 1)
        if not block.isValid():
            return
            
        cursor = QTextCursor(block)
        rect = self._editor.cursorRect(cursor)
        
        # Draw a small indicator at the start of the line
        indicator_rect = QRect(0, rect.top(), 4, rect.height())
        painter.fillRect(indicator_rect, color)
        
        # Draw text showing number of deleted lines
        num_deleted = end_line - start_line + 1
        if num_deleted > 0:
            painter.setPen(color)
            font = self._editor.font()
            font.setItalic(True)
            painter.setFont(font)
            text = f" (-{num_deleted} lines) "
            painter.drawText(rect.left(), rect.top(), 
                           self._editor.fontMetrics().horizontalAdvance(text), 
                           rect.height(),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 
                           text)
        
    def _draw_char_diff(self, painter: QPainter, block: DiffBlock):
        """Draw character-level diff highlighting."""
        if not block.old_text or not block.new_text:
            return
            
        # Calculate character differences
        char_diffs = self._calculate_char_diff(block.old_text, block.new_text)
        
        for diff in char_diffs:
            if diff.diff_type == DiffType.EQUAL:
                continue
                
            # Get color for this diff type
            if diff.diff_type == DiffType.ADD:
                color = self.COLOR_ADD_BG_DARK if self._is_dark_mode else self.COLOR_ADD_BG_LIGHT
            else:
                color = self.COLOR_DEL_BG_DARK if self._is_dark_mode else self.COLOR_DEL_BG_LIGHT
                
            # Draw highlight for this character range
            self._draw_char_range_highlight(painter, block.new_start_line, diff.start, diff.end, color)
            
    def _calculate_char_diff(self, old_text: str, new_text: str) -> List[CharDiff]:
        """Calculate character-level differences between two strings."""
        char_diffs = []
        
        # Use difflib.SequenceMatcher for character-level diff
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                char_diffs.append(CharDiff(j1, j2, DiffType.EQUAL))
            elif tag == 'replace':
                char_diffs.append(CharDiff(j1, j2, DiffType.MODIFY))
            elif tag == 'delete':
                # Deletion in old text - corresponds to position in new text
                char_diffs.append(CharDiff(j1, j1, DiffType.DELETE))
            elif tag == 'insert':
                char_diffs.append(CharDiff(j1, j2, DiffType.ADD))
                
        return char_diffs
        
    def _draw_char_range_highlight(self, painter: QPainter, line_num: int, 
                                   start_col: int, end_col: int, color: QColor):
        """Draw highlight for a character range."""
        block = self._editor.document().findBlockByLineNumber(line_num - 1)
        if not block.isValid():
            return
            
        cursor = QTextCursor(block)
        
        # Move to start column
        cursor.movePosition(QTextCursor.MoveOperation.Right, 
                          QTextCursor.MoveMode.MoveAnchor, start_col)
        start_rect = self._editor.cursorRect(cursor)
        
        # Move to end column
        cursor.movePosition(QTextCursor.MoveOperation.Right,
                          QTextCursor.MoveMode.MoveAnchor, end_col - start_col)
        end_rect = self._editor.cursorRect(cursor)
        
        # Draw highlight rect
        highlight_rect = QRect(start_rect.left(), start_rect.top(),
                              end_rect.left() - start_rect.left(),
                              start_rect.height())
        painter.fillRect(highlight_rect, color)
        
    # Navigation methods
    def next_diff(self) -> bool:
        """Navigate to the next diff block.
        
        Returns:
            True if navigation was successful
        """
        if not self._diff_blocks:
            return False
            
        self._current_block_index = (self._current_block_index + 1) % len(self._diff_blocks)
        self._navigate_to_block(self._diff_blocks[self._current_block_index])
        return True
        
    def previous_diff(self) -> bool:
        """Navigate to the previous diff block.
        
        Returns:
            True if navigation was successful
        """
        if not self._diff_blocks:
            return False
            
        self._current_block_index = (self._current_block_index - 1) % len(self._diff_blocks)
        self._navigate_to_block(self._diff_blocks[self._current_block_index])
        return True
        
    def _navigate_to_block(self, block: DiffBlock):
        """Navigate the editor to show a specific diff block."""
        # Go to the start of the block
        line_num = block.new_start_line if block.diff_type != DiffType.DELETE else block.old_start_line
        cursor = QTextCursor(self._editor.document().findBlockByLineNumber(line_num - 1))
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()
        
        # Highlight the current block more prominently
        self._editor.viewport().update()
        logger.debug(f"Navigated to diff block at line {line_num}")
        
    def get_diff_stats(self) -> Tuple[int, int, int]:
        """Get diff statistics.
        
        Returns:
            Tuple of (added_lines, deleted_lines, modified_lines)
        """
        added = sum(1 for b in self._diff_blocks if b.diff_type == DiffType.ADD)
        deleted = sum(1 for b in self._diff_blocks if b.diff_type == DiffType.DELETE)
        modified = sum(1 for b in self._diff_blocks if b.diff_type == DiffType.MODIFY)
        return added, deleted, modified


class InlineDiffCalculator:
    """Calculates diff between original and modified text."""
    
    @staticmethod
    def calculate_diff(original_text: str, modified_text: str) -> List[DiffBlock]:
        """Calculate diff blocks between two texts.
        
        Args:
            original_text: The original text
            modified_text: The modified text
            
        Returns:
            List of DiffBlock objects
        """
        original_lines = original_text.splitlines(keepends=True)
        modified_lines = modified_text.splitlines(keepends=True)
        
        diff_blocks = []
        
        # Use difflib to get the diff
        diff = list(difflib.unified_diff(
            original_lines, modified_lines,
            lineterm='',
            n=3  # Context lines
        ))
        
        # Parse the unified diff
        i = 0
        while i < len(diff):
            line = diff[i]
            
            # Skip header lines
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                i += 1
                continue
                
            # Parse hunk header
            if line.startswith('@@'):
                # Parse the line numbers
                # Format: @@ -old_start,old_count +new_start,new_count @@
                import re
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    old_start = int(match.group(1))
                    old_count = int(match.group(2)) if match.group(2) else 1
                    new_start = int(match.group(3))
                    new_count = int(match.group(4)) if match.group(4) else 1
                    
                    old_end = old_start + old_count - 1
                    new_end = new_start + new_count - 1
                    
                    # Collect the lines in this hunk
                    old_text_lines = []
                    new_text_lines = []
                    
                    i += 1
                    while i < len(diff) and not diff[i].startswith('@@'):
                        diff_line = diff[i]
                        if diff_line.startswith('-'):
                            old_text_lines.append(diff_line[1:])
                        elif diff_line.startswith('+'):
                            new_text_lines.append(diff_line[1:])
                        elif diff_line.startswith(' '):
                            old_text_lines.append(diff_line[1:])
                            new_text_lines.append(diff_line[1:])
                        i += 1
                        
                    # Determine the diff type
                    if old_count == 0 and new_count > 0:
                        diff_type = DiffType.ADD
                    elif old_count > 0 and new_count == 0:
                        diff_type = DiffType.DELETE
                    elif old_count > 0 and new_count > 0:
                        diff_type = DiffType.MODIFY
                    else:
                        diff_type = DiffType.EQUAL
                        
                    diff_blocks.append(DiffBlock(
                        diff_type=diff_type,
                        old_start_line=old_start,
                        old_end_line=old_end,
                        new_start_line=new_start,
                        new_end_line=new_end,
                        old_text=''.join(old_text_lines),
                        new_text=''.join(new_text_lines)
                    ))
                    continue
                    
            i += 1
            
        # If unified diff parsing didn't work, use simple line-by-line comparison
        if not diff_blocks:
            diff_blocks = InlineDiffCalculator._simple_line_diff(original_lines, modified_lines)
            
        return diff_blocks
        
    @staticmethod
    def _simple_line_diff(original_lines: List[str], modified_lines: List[str]) -> List[DiffBlock]:
        """Simple line-by-line diff calculation."""
        diff_blocks = []
        
        matcher = difflib.SequenceMatcher(None, original_lines, modified_lines)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
                
            old_text = ''.join(original_lines[i1:i2])
            new_text = ''.join(modified_lines[j1:j2])
            
            if tag == 'replace':
                diff_type = DiffType.MODIFY
            elif tag == 'delete':
                diff_type = DiffType.DELETE
            elif tag == 'insert':
                diff_type = DiffType.ADD
            else:
                continue
                
            diff_blocks.append(DiffBlock(
                diff_type=diff_type,
                old_start_line=i1 + 1,
                old_end_line=i2,
                new_start_line=j1 + 1,
                new_end_line=j2,
                old_text=old_text,
                new_text=new_text
            ))
            
        return diff_blocks
