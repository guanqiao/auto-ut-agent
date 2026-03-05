"""Code editor component with syntax highlighting."""

import logging
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel,
    QScrollBar, QFrame, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QSize
from PyQt6.QtGui import (
    QFont, QFontMetrics, QColor, QPainter, QTextFormat,
    QTextCursor, QAction, QKeyEvent
)

from .syntax_highlighter import SyntaxHighlighter, CodeEditorStyle

logger = logging.getLogger(__name__)


class LineNumberArea(QWidget):
    """Line number area for code editor."""
    
    def __init__(self, editor: 'CodeEditor'):
        super().__init__(editor)
        self._editor = editor
        
    def sizeHint(self) -> QSize:
        """Return the size hint."""
        return QSize(self._editor.line_number_area_width(), 0)
        
    def paintEvent(self, event):
        """Paint the line numbers."""
        self._editor.line_number_area_paint_event(event)


class CodeEditor(QTextEdit):
    """Code editor with syntax highlighting and line numbers.
    
    Features:
    - Syntax highlighting (via Pygments)
    - Line numbers
    - Dark/light theme support
    - Current line highlighting
    """
    
    # Signals
    cursor_position_changed = pyqtSignal(int, int)  # line, column
    file_loaded = pyqtSignal(str)  # file_path
    text_modified = pyqtSignal(bool)  # is_modified
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._file_path: Optional[str] = None
        self._language: str = ""
        self._dark_mode: bool = True
        self._highlighter: Optional[SyntaxHighlighter] = None
        self._line_number_area: Optional[LineNumberArea] = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the editor UI."""
        # Set font
        font = QFont("Consolas", 11)
        font.setFixedPitch(True)
        self.setFont(font)
        
        # Set tab width
        self.setTabStopDistance(QFontMetrics(self.font()).horizontalAdvance(' ') * 4)
        
        # Setup line number area
        self._line_number_area = LineNumberArea(self)
        
        # Connect signals
        self.textChanged.connect(self._on_text_changed)
        self.cursorPositionChanged.connect(self._on_cursor_position_changed)
        
        # Setup context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        
        # Apply initial style
        self.set_dark_mode(True)
        
    def line_number_area_width(self) -> int:
        """Calculate the width of the line number area."""
        digits = 1
        count = max(1, self.document().blockCount())
        while count >= 10:
            count //= 10
            digits += 1
        
        space = 10 + self.fontMetrics().horizontalAdvance('9') * digits
        return space
        
    def update_line_number_area_width(self):
        """Update the line number area width."""
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)
        
    def update_line_number_area(self, rect: QRect, dy: int):
        """Update the line number area."""
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(), self._line_number_area.width(), rect.height())
        
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width()
            
    def line_number_area_paint_event(self, event):
        """Paint the line numbers."""
        painter = QPainter(self._line_number_area)
        
        # Background
        bg_color = QColor('#1E1E1E') if self._dark_mode else QColor('#F0F0F0')
        painter.fillRect(event.rect(), bg_color)
        
        # Text color
        text_color = QColor('#858585') if self._dark_mode else QColor('#6E7681')
        painter.setPen(text_color)
        
        # Draw line numbers
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.drawText(0, top, self._line_number_area.width() - 5,
                               self.fontMetrics().height(),
                               Qt.AlignmentFlag.AlignRight, number)
            
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1
            
    def resizeEvent(self, event):
        """Handle resize event."""
        super().resizeEvent(event)
        
        cr = self.contentsRect()
        self._line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )
        
    def load_file(self, file_path: str) -> bool:
        """Load a file into the editor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if loaded successfully
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return False
            
            content = path.read_text(encoding='utf-8', errors='ignore')
            self.setPlainText(content)
            
            self._file_path = file_path
            
            # Detect language
            self._detect_language(file_path)
            
            # Setup syntax highlighting
            self._setup_highlighter()
            
            self.file_loaded.emit(file_path)
            logger.info(f"Loaded file: {file_path}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to load file: {file_path}")
            return False
            
    def _detect_language(self, file_path: str):
        """Detect language from file path."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.java': 'java',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'css',
            '.sql': 'sql',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown',
            '.sh': 'bash',
            '.bash': 'bash',
            '.ps1': 'powershell',
        }
        
        self._language = language_map.get(ext, '')
        logger.debug(f"Detected language: {self._language}")
        
    def _setup_highlighter(self):
        """Setup syntax highlighter."""
        if self._highlighter:
            self._highlighter.setDocument(None)
            self._highlighter = None
        
        if self._language:
            self._highlighter = SyntaxHighlighter(
                self.document(),
                language=self._language,
                dark_mode=self._dark_mode
            )
            
    def set_dark_mode(self, dark_mode: bool):
        """Set dark mode.
        
        Args:
            dark_mode: Whether to use dark theme
        """
        self._dark_mode = dark_mode
        
        style = CodeEditorStyle.get_dark_style() if dark_mode else CodeEditorStyle.get_light_style()
        self.setStyleSheet(style)
        
        # Update highlighter
        if self._highlighter:
            self._highlighter.set_dark_mode(dark_mode)
            
    def _on_text_changed(self):
        """Handle text changes."""
        self.text_modified.emit(True)
        
    def _on_cursor_position_changed(self):
        """Handle cursor position changes."""
        cursor = self.textCursor()
        line = cursor.blockNumber() + 1
        column = cursor.columnNumber() + 1
        self.cursor_position_changed.emit(line, column)
        
    def _show_context_menu(self, position):
        """Show context menu."""
        menu = self.createStandardContextMenu()
        
        # Add custom actions
        menu.addSeparator()
        
        # AI actions
        ai_menu = menu.addMenu("🤖 AI Actions")
        
        explain_action = QAction("Explain Code", self)
        explain_action.triggered.connect(self._explain_selection)
        ai_menu.addAction(explain_action)
        
        refactor_action = QAction("Refactor", self)
        refactor_action.triggered.connect(self._refactor_selection)
        ai_menu.addAction(refactor_action)
        
        generate_tests_action = QAction("Generate Tests", self)
        generate_tests_action.triggered.connect(self._generate_tests)
        ai_menu.addAction(generate_tests_action)
        
        menu.exec(self.mapToGlobal(position))
        
    def _explain_selection(self):
        """Explain selected code."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            logger.info(f"Explain selection: {len(selected_text)} chars")
            # TODO: Send to AI for explanation
            
    def _refactor_selection(self):
        """Refactor selected code."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            logger.info(f"Refactor selection: {len(selected_text)} chars")
            # TODO: Send to AI for refactoring
            
    def _generate_tests(self):
        """Generate tests for current file."""
        if self._file_path:
            logger.info(f"Generate tests for: {self._file_path}")
            # TODO: Send to AI for test generation
            
    def get_file_path(self) -> Optional[str]:
        """Get the current file path."""
        return self._file_path
        
    def get_language(self) -> str:
        """Get the detected language."""
        return self._language
        
    def get_selected_text(self) -> str:
        """Get the selected text."""
        cursor = self.textCursor()
        return cursor.selectedText()
        
    def get_current_line(self) -> str:
        """Get the current line text."""
        cursor = self.textCursor()
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        return cursor.selectedText()
        
    def go_to_line(self, line_number: int):
        """Go to a specific line.
        
        Args:
            line_number: 1-based line number
        """
        cursor = QTextCursor(self.document().findBlockByLineNumber(line_number - 1))
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
    def highlight_line(self, line_number: int):
        """Highlight a specific line.
        
        Args:
            line_number: 1-based line number
        """
        # TODO: Implement line highlighting
        pass


class CodeEditorWidget(QWidget):
    """Code editor widget with toolbar and status bar."""
    
    file_loaded = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        toolbar.setFixedHeight(32)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        
        self._file_label = QLabel("No file loaded")
        self._file_label.setStyleSheet("color: #CCCCCC;")
        toolbar_layout.addWidget(self._file_label)
        
        toolbar_layout.addStretch()
        
        self._language_label = QLabel("")
        self._language_label.setStyleSheet("color: #858585; font-size: 11px;")
        toolbar_layout.addWidget(self._language_label)
        
        toolbar_layout.addSpacing(16)
        
        self._position_label = QLabel("1:1")
        self._position_label.setStyleSheet("color: #858585; font-size: 11px;")
        toolbar_layout.addWidget(self._position_label)
        
        layout.addWidget(toolbar)
        
        # Editor
        self._editor = CodeEditor()
        self._editor.file_loaded.connect(self._on_file_loaded)
        self._editor.cursor_position_changed.connect(self._on_cursor_changed)
        layout.addWidget(self._editor, stretch=1)
        
    def load_file(self, file_path: str) -> bool:
        """Load a file."""
        result = self._editor.load_file(file_path)
        if result:
            self.file_loaded.emit(file_path)
        return result
        
    def _on_file_loaded(self, file_path: str):
        """Handle file loaded."""
        from pathlib import Path
        self._file_label.setText(f"📄 {Path(file_path).name}")
        
        language = self._editor.get_language()
        if language:
            self._language_label.setText(f"[{language.upper()}]")
        else:
            self._language_label.setText("")
            
    def _on_cursor_changed(self, line: int, column: int):
        """Handle cursor position change."""
        self._position_label.setText(f"Ln {line}, Col {column}")
        
    def get_editor(self) -> CodeEditor:
        """Get the code editor."""
        return self._editor
