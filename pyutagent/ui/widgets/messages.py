"""Unified message widgets for chat and agent panels.

This module provides a unified set of message widgets that can be used
across different parts of the application.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QTextBrowser, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

from ..styles import get_style_manager

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Role of a message."""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


@dataclass
class Message:
    """Represents a chat message."""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMessageWidget(QFrame):
    """Base class for message widgets."""

    copy_requested = pyqtSignal(str)

    ROLE_COLORS = {
        MessageRole.USER: {'light': '#E3F2FD', 'dark': '#1E3A5F'},
        MessageRole.AGENT: {'light': '#F5F5F5', 'dark': '#2D2D2D'},
        MessageRole.SYSTEM: {'light': '#FFF3E0', 'dark': '#3E2723'}
    }

    ROLE_ICONS = {
        MessageRole.USER: '👤',
        MessageRole.AGENT: '🤖',
        MessageRole.SYSTEM: 'ℹ️'
    }

    ROLE_LABELS = {
        MessageRole.USER: 'You',
        MessageRole.AGENT: 'Assistant',
        MessageRole.SYSTEM: 'System'
    }

    def __init__(self, message: Message, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._message = message
        self._style_manager = get_style_manager()
        self.setup_ui()

    def setup_ui(self):
        """Setup the message widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._apply_theme_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()

        icon = self.ROLE_ICONS.get(self._message.role, '💬')
        role_label = QLabel(f"{icon} {self.ROLE_LABELS.get(self._message.role, 'Message')}")
        role_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.addWidget(role_label)

        header.addStretch()

        time_str = self._message.timestamp.strftime("%H:%M:%S")
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #999; font-size: 10px;")
        header.addWidget(time_label)

        copy_btn = QPushButton("📋")
        copy_btn.setFixedSize(24, 24)
        copy_btn.setToolTip("Copy message")
        copy_btn.clicked.connect(lambda: self.copy_requested.emit(self._message.content))
        header.addWidget(copy_btn)

        layout.addLayout(header)

        self._content_widget = self._create_content_widget()
        layout.addWidget(self._content_widget)

    def _create_content_widget(self) -> QWidget:
        """Create the content widget. Override in subclasses."""
        label = QLabel(self._message.content)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        return label

    def _apply_theme_style(self):
        """Apply theme-based styles."""
        is_dark = self._style_manager.current_theme == "dark"
        colors = self.ROLE_COLORS.get(self._message.role, {'light': '#FFFFFF', 'dark': '#2D2D2D'})
        bg_color = colors['dark'] if is_dark else colors['light']
        
        border_color = "#2196F3" if self._message.role == MessageRole.AGENT else "#9E9E9E"
        
        self.setStyleSheet(f"""
            {self.__class__.__name__} {{
                background-color: {bg_color};
                border-radius: 8px;
                border-left: 3px solid {border_color};
                margin: 5px 40px 5px 5px;
            }}
        """)

    def update_content(self, content: str):
        """Update message content."""
        self._message.content = content
        if isinstance(self._content_widget, QLabel):
            self._content_widget.setText(content)


class ChatMessageWidget(BaseMessageWidget):
    """Widget for displaying a chat message with markdown support."""

    def _create_content_widget(self) -> QWidget:
        """Create content widget with markdown support."""
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        browser.setMinimumHeight(30)
        browser.setStyleSheet("""
            QTextBrowser {
                border: none;
                background: transparent;
            }
        """)
        browser.setHtml(self._format_content(self._message.content))
        return browser

    def update_content(self, content: str):
        """Update message content."""
        self._message.content = content
        if isinstance(self._content_widget, QTextBrowser):
            self._content_widget.setHtml(self._format_content(content))

    def _format_content(self, content: str) -> str:
        """Format content with markdown support."""
        is_dark = self._style_manager.current_theme == "dark"
        text_color = "#E0E0E0" if is_dark else "#212121"
        code_bg = "#1E1E1E" if is_dark else "#F5F5F5"

        formatted = content

        formatted = formatted.replace('&', '&amp;')
        formatted = formatted.replace('<', '&lt;')
        formatted = formatted.replace('>', '&gt;')

        import re

        def replace_code_block(match):
            lang = match.group(1) or 'text'
            code = match.group(2)
            return f'''<div style="background-color: {code_bg}; border-radius: 8px; margin: 10px 0; overflow: hidden;">
                <div style="background-color: {"#2D2D2D" if is_dark else "#E0E0E0"}; padding: 5px 10px; font-size: 12px; color: {"#999" if is_dark else "#666"}; font-family: monospace;">{lang}</div>
                <pre style="padding: 10px; margin: 0; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; line-height: 1.5;"><code>{code}</code></pre>
            </div>'''

        formatted = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, formatted, flags=re.DOTALL)

        formatted = re.sub(
            r'`([^`]+)`',
            rf'<code style="background-color: {code_bg}; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 90%;">\1</code>',
            formatted
        )

        formatted = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', formatted)
        formatted = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', formatted)

        formatted = re.sub(r'^### (.+)$', r'<h3 style="margin: 10px 0; color: #2196F3;">\1</h3>', formatted, flags=re.MULTILINE)
        formatted = re.sub(r'^## (.+)$', r'<h2 style="margin: 12px 0; color: #2196F3;">\1</h2>', formatted, flags=re.MULTILINE)
        formatted = re.sub(r'^# (.+)$', r'<h1 style="margin: 15px 0; color: #2196F3;">\1</h1>', formatted, flags=re.MULTILINE)

        formatted = re.sub(r'^- (.+)$', r'<li style="margin: 5px 0;">\1</li>', formatted, flags=re.MULTILINE)
        formatted = re.sub(r'(<li>.*</li>\n)+', r'<ul style="margin: 10px 0; padding-left: 20px;">\g<0></ul>', formatted, flags=re.DOTALL)

        formatted = re.sub(
            r'\[([^\]]+)\]\(([^)]+)\)',
            r'<a href="\2" style="color: #2196F3; text-decoration: none;">\1</a>',
            formatted
        )

        formatted = formatted.replace('\n', '<br>')

        font_family = self._style_manager.get_font_family()
        return f'<div style="font-family: {font_family}; color: {text_color}; line-height: 1.6;">{formatted}</div>'


class StreamingMessageWidget(ChatMessageWidget):
    """Widget for streaming message display."""

    def __init__(self, role: MessageRole = MessageRole.AGENT, parent: Optional[QWidget] = None):
        message = Message(
            id="",
            role=role,
            content=""
        )
        super().__init__(message, parent)
        self._content = ""

    def append_chunk(self, chunk: str):
        """Append a chunk of streaming content."""
        self._content += chunk
        self._update_display()

    def update_content(self, content: str):
        """Update full content."""
        self._content = content
        self._update_display()

    def _update_display(self):
        """Update the display with formatted content."""
        if isinstance(self._content_widget, QTextBrowser):
            self._content_widget.setHtml(self._format_content(self._content))
            QTimer.singleShot(10, self._adjust_height)

    def _adjust_height(self):
        """Adjust height to fit content."""
        if isinstance(self._content_widget, QTextBrowser):
            doc = self._content_widget.document()
            doc.setTextWidth(self._content_widget.width())
            height = doc.size().height() + 20
            self._content_widget.setMinimumHeight(int(height))

    def get_content(self) -> str:
        """Get the raw content."""
        return self._content


class SystemMessageWidget(BaseMessageWidget):
    """Widget for system messages."""

    def _apply_theme_style(self):
        """Apply theme-based styles for system messages."""
        is_dark = self._style_manager.current_theme == "dark"
        bg_color = "#3E2723" if is_dark else "#FFF3E0"
        
        self.setStyleSheet(f"""
            {self.__class__.__name__} {{
                background-color: {bg_color};
                border-radius: 8px;
                border-left: 3px solid #FF9800;
                margin: 5px 40px 5px 5px;
            }}
        """)
