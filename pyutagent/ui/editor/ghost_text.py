"""Ghost text layer for inline code suggestions.

Provides semi-transparent text overlay for AI code suggestions,
similar to Cursor's Tab completion feature.
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QPainter, QColor, QFont, QFontMetrics, QTextCursor

logger = logging.getLogger(__name__)


@dataclass
class GhostTextSuggestion:
    """Represents a ghost text suggestion."""
    text: str
    start_line: int
    start_column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None


class GhostTextRenderer:
    """Renders ghost text overlay on the editor.
    
    Features:
    - Semi-transparent gray text display
    - Multi-line ghost text support
    - Position calculation based on cursor/selection
    """
    
    def __init__(self, editor: 'CodeEditor'):
        self._editor = editor
        self._suggestion: Optional[GhostTextSuggestion] = None
        self._opacity = 0.6
        self._color_light = QColor(150, 150, 150)
        self._color_dark = QColor(100, 100, 100)
        
    def set_suggestion(self, suggestion: GhostTextSuggestion):
        """Set the current ghost text suggestion."""
        self._suggestion = suggestion
        self._editor.viewport().update()
        logger.debug(f"Ghost text set: {len(suggestion.text)} chars at L{suggestion.start_line}:C{suggestion.start_column}")
        
    def clear_suggestion(self):
        """Clear the current suggestion."""
        if self._suggestion:
            self._suggestion = None
            self._editor.viewport().update()
            logger.debug("Ghost text cleared")
            
    def has_suggestion(self) -> bool:
        """Check if there's an active suggestion."""
        return self._suggestion is not None
        
    def get_suggestion(self) -> Optional[GhostTextSuggestion]:
        """Get the current suggestion."""
        return self._suggestion
        
    def render(self, painter: QPainter, is_dark_mode: bool = True):
        """Render the ghost text overlay.
        
        Args:
            painter: The QPainter to use
            is_dark_mode: Whether the editor is in dark mode
        """
        if not self._suggestion:
            return
            
        painter.save()
        
        # Set semi-transparent color
        color = self._color_dark if is_dark_mode else self._color_light
        color.setAlphaF(self._opacity)
        painter.setPen(color)
        
        # Use same font as editor
        font = self._editor.font()
        painter.setFont(font)
        
        # Calculate position
        positions = self._calculate_text_positions()
        
        # Draw each line
        for text, rect in positions:
            painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            
        painter.restore()
        
    def _calculate_text_positions(self) -> List[Tuple[str, QRect]]:
        """Calculate the screen positions for each line of ghost text.
        
        Returns:
            List of (text, rect) tuples
        """
        if not self._suggestion:
            return []
            
        positions = []
        lines = self._suggestion.text.split('\n')
        
        # Get starting position
        cursor = QTextCursor(self._editor.document())
        block = self._editor.document().findBlockByLineNumber(self._suggestion.start_line - 1)
        cursor.setPosition(block.position() + self._suggestion.start_column)
        
        # Get the initial position
        cursor_rect = self._editor.cursorRect(cursor)
        x = cursor_rect.left()
        y = cursor_rect.top()
        
        font_metrics = QFontMetrics(self._editor.font())
        line_height = font_metrics.height()
        
        for i, line_text in enumerate(lines):
            if i == 0:
                # First line starts at cursor position
                rect = QRect(x, y, font_metrics.horizontalAdvance(line_text), line_height)
            else:
                # Subsequent lines start at left margin
                x_start = self._editor.contentOffset().x() + self._editor.line_number_area_width() if hasattr(self._editor, 'line_number_area_width') else 0
                y += line_height
                rect = QRect(int(x_start), y, font_metrics.horizontalAdvance(line_text), line_height)
                
            positions.append((line_text, rect))
            
        return positions
        
    def get_full_suggestion_text(self) -> str:
        """Get the full suggestion text."""
        return self._suggestion.text if self._suggestion else ""
        
    def accept_suggestion(self) -> bool:
        """Accept the current suggestion and insert it into the editor.
        
        Returns:
            True if suggestion was accepted
        """
        if not self._suggestion:
            return False
            
        cursor = self._editor.textCursor()
        cursor.beginEditBlock()
        
        # Position cursor at the insertion point
        block = self._editor.document().findBlockByLineNumber(self._suggestion.start_line - 1)
        cursor.setPosition(block.position() + self._suggestion.start_column)
        
        # Insert the suggestion text
        cursor.insertText(self._suggestion.text)
        
        cursor.endEditBlock()
        
        self.clear_suggestion()
        logger.info("Suggestion accepted")
        return True
        
    def get_next_word(self) -> str:
        """Get the next word from the suggestion (for partial acceptance).
        
        Returns:
            The next word to accept
        """
        if not self._suggestion:
            return ""
            
        text = self._suggestion.text
        # Find the next word boundary
        for i, char in enumerate(text):
            if char in ' \t\n.,;:!?()[]{}':
                return text[:i+1]
        return text
        
    def accept_next_word(self) -> bool:
        """Accept only the next word from the suggestion.
        
        Returns:
            True if a word was accepted
        """
        if not self._suggestion:
            return False
            
        next_word = self.get_next_word()
        if not next_word:
            return False
            
        cursor = self._editor.textCursor()
        cursor.beginEditBlock()
        
        # Position cursor at the insertion point
        block = self._editor.document().findBlockByLineNumber(self._suggestion.start_line - 1)
        cursor.setPosition(block.position() + self._suggestion.start_column)
        
        # Insert the word
        cursor.insertText(next_word)
        
        cursor.endEditBlock()
        
        # Update suggestion to remove the accepted part
        remaining = self._suggestion.text[len(next_word):]
        if remaining:
            self._suggestion.text = remaining
            # Update position for next insertion
            lines_accepted = next_word.count('\n')
            if lines_accepted > 0:
                self._suggestion.start_line += lines_accepted
                last_newline = next_word.rfind('\n')
                self._suggestion.start_column = len(next_word) - last_newline - 1
            else:
                self._suggestion.start_column += len(next_word)
            self._editor.viewport().update()
        else:
            self.clear_suggestion()
            
        logger.debug(f"Accepted word: '{next_word}'")
        return True


class GhostTextWidget(QWidget):
    """Widget for displaying ghost text as an overlay.
    
    Alternative implementation using a separate widget.
    """
    
    def __init__(self, parent: 'CodeEditor'):
        super().__init__(parent)
        self._editor = parent
        self._suggestion: Optional[GhostTextSuggestion] = None
        self._opacity = 0.6
        
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("background: transparent;")
        
    def set_suggestion(self, suggestion: GhostTextSuggestion):
        """Set the ghost text suggestion."""
        self._suggestion = suggestion
        self.update()
        
    def clear_suggestion(self):
        """Clear the suggestion."""
        self._suggestion = None
        self.update()
        
    def paintEvent(self, event):
        """Paint the ghost text."""
        if not self._suggestion:
            return
            
        painter = QPainter(self)
        painter.setOpacity(self._opacity)
        
        # Determine color based on editor theme
        is_dark = getattr(self._editor, '_dark_mode', True)
        color = QColor(100, 100, 100) if is_dark else QColor(150, 150, 150)
        painter.setPen(color)
        painter.setFont(self._editor.font())
        
        # Calculate and draw positions
        renderer = GhostTextRenderer(self._editor)
        renderer._suggestion = self._suggestion
        positions = renderer._calculate_text_positions()
        
        for text, rect in positions:
            painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
