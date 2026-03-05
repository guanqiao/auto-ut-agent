"""Chat mode for Agent panel - traditional chat interface."""

import logging
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QScrollArea, QLabel, QFrame, QSizePolicy,
    QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QTextCursor, QColor, QFont, QTextCharFormat

from ..components import MarkdownViewer, StreamingHandler, StreamingConfig, StreamingMode

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    id: str
    role: str  # 'user', 'agent', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class ChatMessageWidget(QFrame):
    """Widget for displaying a single chat message with Markdown support."""
    
    copy_requested = pyqtSignal(str)  # message_content
    code_copy_requested = pyqtSignal(str)  # code_content
    code_insert_requested = pyqtSignal(str)  # code_content
    
    # Role colors
    ROLE_COLORS = {
        'user': '#E3F2FD',
        'agent': '#F5F5F5',
        'system': '#FFF3E0'
    }
    
    ROLE_ICONS = {
        'user': '👤',
        'agent': '🤖',
        'system': 'ℹ️'
    }
    
    def __init__(self, message: ChatMessage, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._message = message
        self._markdown_viewer: Optional[MarkdownViewer] = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the message widget UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        # Set background color based on role
        bg_color = self.ROLE_COLORS.get(self._message.role, '#FFFFFF')
        self.setStyleSheet(f"""
            ChatMessageWidget {{
                background-color: {bg_color};
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        # Header
        header = QHBoxLayout()
        
        # Icon and role
        icon = self.ROLE_ICONS.get(self._message.role, '💬')
        role_label = QLabel(f"{icon} {self._message.role.capitalize()}")
        role_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.addWidget(role_label)
        
        header.addStretch()
        
        # Timestamp
        time_str = self._message.timestamp.strftime("%H:%M:%S")
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #999; font-size: 10px;")
        header.addWidget(time_label)
        
        # Copy button
        copy_btn = QPushButton("📋")
        copy_btn.setFixedSize(24, 24)
        copy_btn.setToolTip("Copy message")
        copy_btn.clicked.connect(lambda: self.copy_requested.emit(self._message.content))
        header.addWidget(copy_btn)
        
        layout.addLayout(header)
        
        # Content - use Markdown viewer for agent messages, plain text for others
        if self._message.role == 'agent':
            self._markdown_viewer = MarkdownViewer()
            self._markdown_viewer.set_content(self._message.content)
            self._markdown_viewer.code_copy_requested.connect(self.code_copy_requested.emit)
            self._markdown_viewer.code_insert_requested.connect(self.code_insert_requested.emit)
            layout.addWidget(self._markdown_viewer)
        else:
            self._content_label = QLabel()
            self._content_label.setWordWrap(True)
            self._content_label.setTextFormat(Qt.TextFormat.PlainText)
            self._content_label.setText(self._message.content)
            layout.addWidget(self._content_label)
        
    def update_content(self, content: str):
        """Update message content."""
        self._message.content = content
        if self._markdown_viewer:
            self._markdown_viewer.set_content(content)
        else:
            self._content_label.setText(content)


class ChatMode(QWidget):
    """Chat mode widget for Agent panel.
    
    Traditional chat interface with:
    - Message history
    - Input area
    - Message actions (copy, etc.)
    - Markdown rendering
    - Streaming response support
    """
    
    message_sent = pyqtSignal(str)  # message_text
    message_edited = pyqtSignal(str, str)  # message_id, new_content
    code_copy_requested = pyqtSignal(str)  # code_content
    code_insert_requested = pyqtSignal(str)  # code_content
    streaming_started = pyqtSignal()
    streaming_finished = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._messages: List[ChatMessage] = []
        self._message_widgets: dict = {}  # id -> widget
        self._current_streaming_message: Optional[str] = None
        self._streaming_handler: Optional[StreamingHandler] = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the chat mode UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Messages area
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self._messages_container = QWidget()
        self._messages_layout = QVBoxLayout(self._messages_container)
        self._messages_layout.setContentsMargins(8, 8, 8, 8)
        self._messages_layout.setSpacing(8)
        self._messages_layout.addStretch()
        
        self._scroll_area.setWidget(self._messages_container)
        layout.addWidget(self._scroll_area, stretch=1)
        
        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border-top: 1px solid #E0E0E0;
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(8)
        
        # Quick actions
        quick_actions = QHBoxLayout()
        
        self._btn_clear = QPushButton("🗑️ Clear")
        self._btn_clear.clicked.connect(self.clear_messages)
        quick_actions.addWidget(self._btn_clear)
        
        quick_actions.addStretch()
        
        self._btn_generate = QPushButton("🚀 Generate Tests")
        self._btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self._btn_generate.clicked.connect(self._on_generate_clicked)
        quick_actions.addWidget(self._btn_generate)
        
        input_layout.addLayout(quick_actions)
        
        # Input row
        input_row = QHBoxLayout()
        
        self._input = QLineEdit()
        self._input.setPlaceholderText("Type your message... (Use @ to mention files)")
        self._input.returnPressed.connect(self._send_message)
        input_row.addWidget(self._input)
        
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(80)
        self._send_btn.clicked.connect(self._send_message)
        input_row.addWidget(self._send_btn)
        
        input_layout.addLayout(input_row)
        
        layout.addWidget(input_frame)
        
        # Welcome message
        self._add_system_message(
            "Welcome! I'm your AI coding assistant.\n"
            "I can help you with:\n"
            "• Generating unit tests\n"
            "• Explaining code\n"
            "• Refactoring\n"
            "• Code review\n\n"
            "Select a file and click 'Generate Tests' to get started!"
        )
        
    def _send_message(self):
        """Send user message."""
        text = self._input.text().strip()
        if not text:
            return
        
        # Add user message
        self.add_user_message(text)
        
        # Clear input
        self._input.clear()
        
        # Emit signal
        self.message_sent.emit(text)
        
    def add_user_message(self, content: str) -> ChatMessage:
        """Add a user message."""
        import uuid
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role='user',
            content=content
        )
        self._add_message(message)
        return message
        
    def add_agent_message(self, content: str) -> ChatMessage:
        """Add an agent message."""
        import uuid
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role='agent',
            content=content
        )
        self._add_message(message)
        return message
        
    def _add_system_message(self, content: str):
        """Add a system message."""
        import uuid
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role='system',
            content=content
        )
        self._add_message(message)
        
    def _add_message(self, message: ChatMessage):
        """Add a message to the chat."""
        self._messages.append(message)
        
        widget = ChatMessageWidget(message)
        widget.copy_requested.connect(self._copy_to_clipboard)
        widget.code_copy_requested.connect(self.code_copy_requested.emit)
        widget.code_insert_requested.connect(self.code_insert_requested.emit)
        
        # Insert before stretch
        index = self._messages_layout.count() - 1
        self._messages_layout.insertWidget(index, widget)
        
        self._message_widgets[message.id] = widget
        
        # Scroll to bottom
        QTimer.singleShot(50, self._scroll_to_bottom)
        
    def start_streaming_response(self) -> str:
        """Start a streaming response.
        
        Returns:
            Message ID for updating
        """
        import uuid
        message_id = str(uuid.uuid4())
        
        message = ChatMessage(
            id=message_id,
            role='agent',
            content=''
        )
        
        self._messages.append(message)
        
        widget = ChatMessageWidget(message)
        widget.copy_requested.connect(self._copy_to_clipboard)
        widget.code_copy_requested.connect(self.code_copy_requested.emit)
        widget.code_insert_requested.connect(self.code_insert_requested.emit)
        
        index = self._messages_layout.count() - 1
        self._messages_layout.insertWidget(index, widget)
        
        self._message_widgets[message_id] = widget
        self._current_streaming_message = message_id
        
        # Setup streaming handler
        config = StreamingConfig(mode=StreamingMode.CHUNK, chunk_delay_ms=10)
        self._streaming_handler = StreamingHandler(config)
        self._streaming_handler.content_updated.connect(
            lambda content: self._update_streaming_content(message_id, content)
        )
        self._streaming_handler.streaming_finished.connect(self.streaming_finished.emit)
        self._streaming_handler.start_streaming()
        
        self.streaming_started.emit()
        QTimer.singleShot(50, self._scroll_to_bottom)
        
        return message_id
        
    def _update_streaming_content(self, message_id: str, content: str):
        """Update streaming message content."""
        if message_id in self._message_widgets:
            widget = self._message_widgets[message_id]
            widget.update_content(content)
            QTimer.singleShot(10, self._scroll_to_bottom)
        
    def append_to_streaming(self, text: str):
        """Append text to the current streaming message."""
        if not self._current_streaming_message:
            return
        
        if self._streaming_handler:
            self._streaming_handler.append_chunk(text)
        else:
            # Fallback to direct update
            message_id = self._current_streaming_message
            if message_id in self._message_widgets:
                widget = self._message_widgets[message_id]
                current_content = widget._message.content
                widget.update_content(current_content + text)
            
    def finish_streaming(self):
        """Finish the current streaming message."""
        if self._streaming_handler:
            self._streaming_handler.stop_streaming()
            self._streaming_handler = None
        self._current_streaming_message = None
        
    def update_message(self, message_id: str, content: str):
        """Update a message's content."""
        if message_id in self._message_widgets:
            self._message_widgets[message_id].update_content(content)
            
    def clear_messages(self):
        """Clear all messages."""
        # Remove all widgets except stretch
        while self._messages_layout.count() > 1:
            item = self._messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._messages.clear()
        self._message_widgets.clear()
        self._current_streaming_message = None
        
        if self._streaming_handler:
            self._streaming_handler.stop_streaming()
            self._streaming_handler = None
        
    def _scroll_to_bottom(self):
        """Scroll to the bottom of messages."""
        scrollbar = self._scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
    def _on_generate_clicked(self):
        """Handle generate button click."""
        # This will be connected to the main window
        pass
        
    def set_generate_callback(self, callback):
        """Set the generate button callback."""
        self._btn_generate.clicked.connect(callback)
        
    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)
        
    def get_messages(self) -> List[ChatMessage]:
        """Get all messages."""
        return self._messages.copy()
        
    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._streaming_handler is not None and self._streaming_handler.is_streaming()
