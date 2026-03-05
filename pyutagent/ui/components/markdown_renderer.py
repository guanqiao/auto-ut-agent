"""Markdown renderer component with syntax highlighting."""

import logging
import re
from typing import Optional, Callable
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QSizePolicy, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QTextCursor, QColor, QTextCharFormat, QFont, QKeyEvent

logger = logging.getLogger(__name__)


@dataclass
class CodeBlock:
    """Represents a code block in markdown."""
    language: str
    code: str
    start_pos: int
    end_pos: int


class MarkdownRenderer:
    """Markdown renderer with syntax highlighting support.
    
    Features:
    - Convert markdown to HTML
    - Syntax highlighting for code blocks
    - Support for tables, lists, quotes
    """
    
    def __init__(self):
        self._markdown = None
        try:
            import markdown
            self._markdown = markdown.Markdown(
                extensions=[
                    'tables',
                    'fenced_code',
                    'toc',
                    'nl2br',
                ]
            )
        except ImportError:
            logger.warning("markdown library not installed, using fallback")
    
    def render(self, text: str) -> str:
        """Render markdown text to HTML.
        
        Args:
            text: Markdown text
            
        Returns:
            HTML string
        """
        if self._markdown:
            try:
                self._markdown.reset()
                return self._markdown.convert(text)
            except Exception as e:
                logger.error(f"Markdown rendering failed: {e}")
        
        # Fallback: simple HTML conversion
        return self._simple_render(text)
    
    def _simple_render(self, text: str) -> str:
        """Simple markdown to HTML conversion as fallback."""
        html = text
        
        # Escape HTML
        html = html.replace('&', '&amp;')
        html = html.replace('<', '&lt;')
        html = html.replace('>', '&gt;')
        
        # Code blocks
        html = re.sub(
            r'```(\w+)?\n(.*?)```',
            lambda m: f'<pre><code class="language-{m.group(1) or "text"}">{m.group(2)}</code></pre>',
            html,
            flags=re.DOTALL
        )
        
        # Inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)
        
        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
        # Bold and italic
        html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
        # Lists
        html = re.sub(r'^\* (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^(<li>.+</li>\n)+', r'<ul>\g<0></ul>', html, flags=re.MULTILINE)
        
        # Line breaks
        html = html.replace('\n', '<br>')
        
        return f'<div>{html}</div>'
    
    def extract_code_blocks(self, text: str) -> list[CodeBlock]:
        """Extract code blocks from markdown text.
        
        Args:
            text: Markdown text
            
        Returns:
            List of CodeBlock objects
        """
        blocks = []
        pattern = r'```(\w+)?\n(.*?)```'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or 'text'
            code = match.group(2)
            blocks.append(CodeBlock(
                language=language,
                code=code,
                start_pos=match.start(),
                end_pos=match.end()
            ))
        
        return blocks
    
    def highlight_code(self, code: str, language: str = 'text') -> str:
        """Highlight code using Pygments.
        
        Args:
            code: Code to highlight
            language: Programming language
            
        Returns:
            HTML with syntax highlighting
        """
        try:
            from pygments import highlight
            from pygments.lexers import get_lexer_by_name, guess_lexer
            from pygments.formatters import HtmlFormatter
            
            try:
                lexer = get_lexer_by_name(language)
            except Exception:
                try:
                    lexer = guess_lexer(code)
                except Exception:
                    lexer = get_lexer_by_name('text')
            
            formatter = HtmlFormatter(
                style='default',
                noclasses=True,
                prestyles='background-color: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto;'
            )
            
            return highlight(code, lexer, formatter)
        except ImportError:
            logger.debug("Pygments not available for syntax highlighting")
            return f'<pre><code>{code}</code></pre>'
        except Exception as e:
            logger.error(f"Code highlighting failed: {e}")
            return f'<pre><code>{code}</code></pre>'


class CodeBlockWidget(QFrame):
    """Widget for displaying a code block with copy/insert buttons."""
    
    copy_requested = pyqtSignal(str)
    insert_requested = pyqtSignal(str)
    
    def __init__(self, code: str, language: str = 'text', 
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._code = code
        self._language = language
        self._renderer = MarkdownRenderer()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the code block widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            CodeBlockWidget {
                background-color: #263238;
                border-radius: 8px;
                border: 1px solid #37474F;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with language and buttons
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #37474F;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)
        header_layout.setSpacing(8)
        
        # Language label
        lang_label = QLabel(self._language.upper() if self._language else 'CODE')
        lang_label.setStyleSheet("""
            color: #90A4AE;
            font-size: 11px;
            font-weight: bold;
            font-family: 'Consolas', 'Monaco', monospace;
        """)
        header_layout.addWidget(lang_label)
        
        header_layout.addStretch()
        
        # Copy button
        copy_btn = QPushButton("📋 Copy")
        copy_btn.setFixedHeight(24)
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #455A64;
                color: #ECEFF1;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        copy_btn.clicked.connect(self._on_copy)
        header_layout.addWidget(copy_btn)
        
        # Insert button
        insert_btn = QPushButton("⬇️ Insert")
        insert_btn.setFixedHeight(24)
        insert_btn.setStyleSheet("""
            QPushButton {
                background-color: #455A64;
                color: #ECEFF1;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #546E7A;
            }
        """)
        insert_btn.clicked.connect(self._on_insert)
        header_layout.addWidget(insert_btn)
        
        layout.addWidget(header)
        
        # Code content
        self._code_text = QTextEdit()
        self._code_text.setReadOnly(True)
        self._code_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._code_text.setStyleSheet("""
            QTextEdit {
                background-color: #263238;
                color: #EEFFFF;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 13px;
                border: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
                padding: 12px;
            }
        """)
        
        # Set highlighted code
        highlighted = self._renderer.highlight_code(self._code, self._language)
        self._code_text.setHtml(highlighted)
        
        # Set fixed height based on content
        doc_height = self._code_text.document().size().height()
        self._code_text.setFixedHeight(int(min(doc_height + 20, 400)))
        
        layout.addWidget(self._code_text)
        
    def _on_copy(self):
        """Handle copy button click."""
        self.copy_requested.emit(self._code)
        
    def _on_insert(self):
        """Handle insert button click."""
        self.insert_requested.emit(self._code)
        
    def get_code(self) -> str:
        """Get the code content."""
        return self._code


class MarkdownViewer(QWidget):
    """Widget for displaying rendered markdown content.
    
    Features:
    - Real-time markdown rendering
    - Syntax highlighted code blocks
    - Copy/Insert buttons for code blocks
    - Streaming content support
    """
    
    code_copy_requested = pyqtSignal(str)
    code_insert_requested = pyqtSignal(str)
    link_clicked = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._renderer = MarkdownRenderer()
        self._content = ""
        self._code_blocks: list[CodeBlockWidget] = []
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the markdown viewer UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Content container
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(8)
        self._container_layout.addStretch()
        
        layout.addWidget(self._container)
        
    def set_content(self, text: str):
        """Set markdown content.
        
        Args:
            text: Markdown text to render
        """
        self._content = text
        self._render_content()
        
    def append_content(self, text: str):
        """Append content to existing markdown.
        
        Args:
            text: Text to append
        """
        self._content += text
        self._render_content()
        
    def _render_content(self):
        """Render the current content."""
        # Clear existing widgets
        while self._container_layout.count() > 1:
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._code_blocks.clear()
        
        # Split content by code blocks
        parts = re.split(r'(```[\w]*\n.*?```)', self._content, flags=re.DOTALL)
        
        for part in parts:
            if part.startswith('```'):
                # Extract code block
                match = re.match(r'```(\w+)?\n(.*?)```', part, re.DOTALL)
                if match:
                    language = match.group(1) or 'text'
                    code = match.group(2)
                    
                    code_widget = CodeBlockWidget(code, language)
                    code_widget.copy_requested.connect(self.code_copy_requested.emit)
                    code_widget.insert_requested.connect(self.code_insert_requested.emit)
                    
                    self._code_blocks.append(code_widget)
                    index = self._container_layout.count() - 1
                    self._container_layout.insertWidget(index, code_widget)
            else:
                # Regular markdown content
                if part.strip():
                    html = self._renderer.render(part)
                    
                    text_widget = QTextEdit()
                    text_widget.setReadOnly(True)
                    text_widget.setFrameStyle(QFrame.Shape.NoFrame)
                    text_widget.setStyleSheet("""
                        QTextEdit {
                            background-color: transparent;
                            color: #333;
                            font-size: 14px;
                            line-height: 1.6;
                        }
                    """)
                    text_widget.setHtml(f"""
                        <style>
                            p {{ margin: 8px 0; line-height: 1.6; }}
                            h1, h2, h3 {{ margin: 16px 0 8px 0; color: #1976D2; }}
                            ul, ol {{ margin: 8px 0; padding-left: 24px; }}
                            li {{ margin: 4px 0; }}
                            code {{ 
                                background-color: #f5f5f5; 
                                padding: 2px 6px; 
                                border-radius: 3px;
                                font-family: 'Consolas', 'Monaco', monospace;
                                font-size: 13px;
                            }}
                            pre {{
                                background-color: #f5f5f5;
                                padding: 12px;
                                border-radius: 6px;
                                overflow-x: auto;
                            }}
                            blockquote {{
                                border-left: 4px solid #1976D2;
                                margin: 8px 0;
                                padding: 8px 16px;
                                background-color: #E3F2FD;
                                color: #1565C0;
                            }}
                            table {{
                                border-collapse: collapse;
                                width: 100%;
                                margin: 8px 0;
                            }}
                            th, td {{
                                border: 1px solid #ddd;
                                padding: 8px;
                                text-align: left;
                            }}
                            th {{
                                background-color: #f5f5f5;
                                font-weight: bold;
                            }}
                        </style>
                        {html}
                    """)
                    
                    text_widget.setVerticalScrollBarPolicy(
                        Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                    )
                    
                    # Adjust height to content
                    doc_height = text_widget.document().size().height()
                    text_widget.setFixedHeight(int(doc_height + 20))
                    
                    index = self._container_layout.count() - 1
                    self._container_layout.insertWidget(index, text_widget)
        
    def clear(self):
        """Clear all content."""
        self._content = ""
        self._render_content()
        
    def get_content(self) -> str:
        """Get current markdown content."""
        return self._content
    
    def get_code_blocks(self) -> list[CodeBlockWidget]:
        """Get all code block widgets."""
        return self._code_blocks.copy()
