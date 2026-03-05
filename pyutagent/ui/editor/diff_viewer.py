"""Diff viewer for comparing code changes."""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel,
    QPushButton, QSplitter, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of change in a diff."""
    UNCHANGED = "unchanged"
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass
class DiffLine:
    """Represents a single line in a diff."""
    old_line_num: Optional[int]
    new_line_num: Optional[int]
    content: str
    change_type: ChangeType


class DiffViewer(QWidget):
    """Diff viewer widget for comparing code changes.
    
    Features:
    - Side-by-side diff view
    - Inline diff view
    - Syntax highlighting
    - Accept/Reject changes
    """
    
    # Signals
    change_accepted = pyqtSignal()  # All changes accepted
    change_rejected = pyqtSignal()  # All changes rejected
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._old_content: str = ""
        self._new_content: str = ""
        self._diff_lines: List[DiffLine] = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the diff viewer UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        header.setFixedHeight(40)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        self._title_label = QLabel("Diff View")
        self._title_label.setStyleSheet("color: #CCCCCC; font-weight: bold;")
        header_layout.addWidget(self._title_label)
        
        header_layout.addStretch()
        
        # Stats
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #858585; font-size: 11px;")
        header_layout.addWidget(self._stats_label)
        
        header_layout.addSpacing(16)
        
        # Action buttons
        self._btn_reject = QPushButton("✗ Reject All")
        self._btn_reject.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)
        self._btn_reject.clicked.connect(self._on_reject)
        header_layout.addWidget(self._btn_reject)
        
        self._btn_accept = QPushButton("✓ Accept All")
        self._btn_accept.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self._btn_accept.clicked.connect(self._on_accept)
        header_layout.addWidget(self._btn_accept)
        
        layout.addWidget(header)
        
        # Diff content
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Old version
        old_container = QWidget()
        old_layout = QVBoxLayout(old_container)
        old_layout.setContentsMargins(0, 0, 0, 0)
        old_layout.setSpacing(0)
        
        old_header = QLabel("📄 Original")
        old_header.setStyleSheet("""
            background-color: #1E1E1E;
            color: #CCCCCC;
            padding: 8px;
            border-bottom: 1px solid #3C3C3C;
        """)
        old_layout.addWidget(old_header)
        
        self._old_editor = QTextEdit()
        self._old_editor.setReadOnly(True)
        self._old_editor.setFont(QFont("Consolas", 11))
        self._old_editor.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
            }
        """)
        old_layout.addWidget(self._old_editor)
        
        self._splitter.addWidget(old_container)
        
        # New version
        new_container = QWidget()
        new_layout = QVBoxLayout(new_container)
        new_layout.setContentsMargins(0, 0, 0, 0)
        new_layout.setSpacing(0)
        
        new_header = QLabel("✏️ Modified")
        new_header.setStyleSheet("""
            background-color: #1E1E1E;
            color: #CCCCCC;
            padding: 8px;
            border-bottom: 1px solid #3C3C3C;
        """)
        new_layout.addWidget(new_header)
        
        self._new_editor = QTextEdit()
        self._new_editor.setReadOnly(True)
        self._new_editor.setFont(QFont("Consolas", 11))
        self._new_editor.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
            }
        """)
        new_layout.addWidget(self._new_editor)
        
        self._splitter.addWidget(new_container)
        
        # Set equal sizes
        self._splitter.setSizes([500, 500])
        
        layout.addWidget(self._splitter, stretch=1)
        
    def set_diff(self, old_content: str, new_content: str, title: str = ""):
        """Set the diff content.
        
        Args:
            old_content: Original content
            new_content: Modified content
            title: Optional title for the diff
        """
        self._old_content = old_content
        self._new_content = new_content
        
        if title:
            self._title_label.setText(f"Diff: {title}")
        
        # Compute and display diff
        self._compute_diff()
        self._display_diff()
        self._update_stats()
        
    def _compute_diff(self):
        """Compute the diff between old and new content."""
        try:
            # Try to use difflib for unified diff
            import difflib
            
            old_lines = self._old_content.splitlines(keepends=True)
            new_lines = self._new_content.splitlines(keepends=True)
            
            # Use SequenceMatcher for better diff
            sm = difflib.SequenceMatcher(None, old_lines, new_lines)
            
            self._diff_lines = []
            old_line_num = 1
            new_line_num = 1
            
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == 'equal':
                    for i in range(i1, i2):
                        line = old_lines[i].rstrip('\n\r')
                        self._diff_lines.append(DiffLine(
                            old_line_num=old_line_num,
                            new_line_num=new_line_num,
                            content=line,
                            change_type=ChangeType.UNCHANGED
                        ))
                        old_line_num += 1
                        new_line_num += 1
                        
                elif tag == 'delete':
                    for i in range(i1, i2):
                        line = old_lines[i].rstrip('\n\r')
                        self._diff_lines.append(DiffLine(
                            old_line_num=old_line_num,
                            new_line_num=None,
                            content=line,
                            change_type=ChangeType.REMOVED
                        ))
                        old_line_num += 1
                        
                elif tag == 'insert':
                    for j in range(j1, j2):
                        line = new_lines[j].rstrip('\n\r')
                        self._diff_lines.append(DiffLine(
                            old_line_num=None,
                            new_line_num=new_line_num,
                            content=line,
                            change_type=ChangeType.ADDED
                        ))
                        new_line_num += 1
                        
                elif tag == 'replace':
                    # Mark as modified
                    max_len = max(i2 - i1, j2 - j1)
                    for k in range(max_len):
                        old_line = old_lines[i1 + k].rstrip('\n\r') if i1 + k < i2 else None
                        new_line = new_lines[j1 + k].rstrip('\n\r') if j1 + k < j2 else None
                        
                        if old_line:
                            self._diff_lines.append(DiffLine(
                                old_line_num=old_line_num,
                                new_line_num=None,
                                content=old_line,
                                change_type=ChangeType.REMOVED
                            ))
                            old_line_num += 1
                            
                        if new_line:
                            self._diff_lines.append(DiffLine(
                                old_line_num=None,
                                new_line_num=new_line_num,
                                content=new_line,
                                change_type=ChangeType.ADDED
                            ))
                            new_line_num += 1
                            
        except Exception as e:
            logger.exception("Failed to compute diff")
            # Fallback: just show both versions
            self._diff_lines = []
            
    def _display_diff(self):
        """Display the computed diff."""
        # Clear editors
        self._old_editor.clear()
        self._new_editor.clear()
        
        # Build content for each side
        old_lines = []
        new_lines = []
        
        for diff_line in self._diff_lines:
            if diff_line.change_type == ChangeType.UNCHANGED:
                old_lines.append(self._format_line(diff_line, 'old'))
                new_lines.append(self._format_line(diff_line, 'new'))
            elif diff_line.change_type == ChangeType.REMOVED:
                old_lines.append(self._format_line(diff_line, 'old'))
                new_lines.append(self._format_empty_line())
            elif diff_line.change_type == ChangeType.ADDED:
                old_lines.append(self._format_empty_line())
                new_lines.append(self._format_line(diff_line, 'new'))
        
        # Set content
        self._old_editor.setHtml('<pre>' + '\n'.join(old_lines) + '</pre>')
        self._new_editor.setHtml('<pre>' + '\n'.join(new_lines) + '</pre>')
        
    def _format_line(self, diff_line: DiffLine, side: str) -> str:
        """Format a diff line for display.
        
        Args:
            diff_line: The diff line
            side: 'old' or 'new'
            
        Returns:
            HTML formatted line
        """
        # Escape HTML
        content = diff_line.content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Get line number
        if side == 'old':
            line_num = diff_line.old_line_num
        else:
            line_num = diff_line.new_line_num
        
        num_str = f"{line_num:4d}" if line_num else "    "
        
        # Apply color based on change type
        if diff_line.change_type == ChangeType.UNCHANGED:
            bg_color = "transparent"
            text_color = "#D4D4D4"
        elif diff_line.change_type == ChangeType.ADDED:
            bg_color = "#1E4620"  # Dark green
            text_color = "#B5F2A6"
        elif diff_line.change_type == ChangeType.REMOVED:
            bg_color = "#4A1E1E"  # Dark red
            text_color = "#F2A6A6"
        else:
            bg_color = "transparent"
            text_color = "#D4D4D4"
        
        return f'<span style="background-color: {bg_color}; color: {text_color};">{num_str} │ {content}</span>'
        
    def _format_empty_line(self) -> str:
        """Format an empty line placeholder."""
        return '<span style="color: #555;">    │ </span>'
        
    def _update_stats(self):
        """Update diff statistics."""
        added = sum(1 for line in self._diff_lines if line.change_type == ChangeType.ADDED)
        removed = sum(1 for line in self._diff_lines if line.change_type == ChangeType.REMOVED)
        
        stats_text = f"<span style='color: #4CAF50;'>+{added}</span> <span style='color: #F44336;'>-{removed}</span>"
        self._stats_label.setText(f"Changes: {stats_text}")
        
    def _on_accept(self):
        """Handle accept button click."""
        self.change_accepted.emit()
        logger.info("Changes accepted")
        
    def _on_reject(self):
        """Handle reject button click."""
        self.change_rejected.emit()
        logger.info("Changes rejected")
        
    def get_new_content(self) -> str:
        """Get the new/modified content."""
        return self._new_content
        
    def clear(self):
        """Clear the diff view."""
        self._old_content = ""
        self._new_content = ""
        self._diff_lines = []
        self._old_editor.clear()
        self._new_editor.clear()
        self._stats_label.setText("")
        self._title_label.setText("Diff View")


class InlineDiffViewer(QTextEdit):
    """Inline diff viewer showing changes in a single editor."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 11))
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
            }
        """)
        
    def set_diff(self, old_content: str, new_content: str):
        """Set the diff content inline."""
        self.clear()
        
        # Simple inline diff
        import difflib
        
        diff = list(difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile='Original',
            tofile='Modified',
            lineterm=''
        ))
        
        if not diff:
            self.setPlainText("No changes")
            return
        
        # Format diff with colors
        html_lines = ['<pre style="margin: 0;">']
        
        for line in diff[2:]:  # Skip header
            escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            if line.startswith('+'):
                html_lines.append(f'<span style="background-color: #1E4620; color: #B5F2A6;">{escaped}</span>')
            elif line.startswith('-'):
                html_lines.append(f'<span style="background-color: #4A1E1E; color: #F2A6A6;">{escaped}</span>')
            elif line.startswith('@@'):
                html_lines.append(f'<span style="color: #569CD6;">{escaped}</span>')
            else:
                html_lines.append(f'<span style="color: #D4D4D4;">{escaped}</span>')
        
        html_lines.append('</pre>')
        self.setHtml('\n'.join(html_lines))
