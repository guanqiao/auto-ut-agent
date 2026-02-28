"""Chat widget for Agent conversation."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QScrollArea, QLabel,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QColor, QFont


class ChatMessageWidget(QFrame):
    """Single chat message widget."""
    
    def __init__(self, role: str, content: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Avatar/role indicator
        avatar_label = QLabel("🤖" if role == "agent" else "👤")
        avatar_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(avatar_label)
        
        # Message content
        self.content_label = QLabel(content)
        self.content_label.setWordWrap(True)
        self.content_label.setTextFormat(Qt.TextFormat.PlainText)
        self.content_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        
        # Style based on role
        if role == "agent":
            self.setStyleSheet("""
                ChatMessageWidget {
                    background-color: #e3f2fd;
                    border-radius: 10px;
                    margin: 5px 50px 5px 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                ChatMessageWidget {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    margin: 5px 5px 5px 50px;
                }
            """)
        
        layout.addWidget(self.content_label, stretch=1)
    
    def update_content(self, content: str):
        """Update message content."""
        self.content_label.setText(content)


class ChatWidget(QWidget):
    """Chat widget for Agent conversation."""
    
    message_sent = pyqtSignal(str)  # User message signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.messages: list[ChatMessageWidget] = []
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel("💬 对话")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        # Message area (scrollable)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.messages_layout = QVBoxLayout(self.scroll_content)
        self.messages_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)
        
        # Quick action buttons
        quick_buttons_layout = QHBoxLayout()
        
        quick_actions = [
            ("⏸ 暂停", self.on_pause),
            ("▶ 继续", self.on_resume),
            ("📊 状态", self.on_status),
            ("🗑 清空", self.on_clear),
        ]
        
        for text, callback in quick_actions:
            btn = QPushButton(text)
            btn.setMaximumWidth(80)
            btn.clicked.connect(callback)
            quick_buttons_layout.addWidget(btn)
        
        quick_buttons_layout.addStretch()
        layout.addLayout(quick_buttons_layout)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入消息... (例如: 生成 UserService 的测试)")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("发送")
        self.send_button.setMaximumWidth(60)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
    
    def add_message(self, role: str, content: str) -> ChatMessageWidget:
        """Add a message to the chat."""
        msg_widget = ChatMessageWidget(role, content)
        # Insert before the stretch
        self.messages_layout.insertWidget(
            self.messages_layout.count() - 1,
            msg_widget
        )
        self.messages.append(msg_widget)
        
        # Scroll to bottom
        self.scroll_to_bottom()
        
        return msg_widget
    
    def update_last_message(self, content: str):
        """Update the last message (for streaming)."""
        if self.messages:
            current_text = self.messages[-1].content_label.text()
            self.messages[-1].update_content(current_text + content)
            self.scroll_to_bottom()
    
    def send_message(self):
        """Send user message."""
        text = self.input_field.text().strip()
        if text:
            self.add_message("user", text)
            self.input_field.clear()
            self.message_sent.emit(text)
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the chat."""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_pause(self):
        """Handle pause button."""
        self.input_field.setText("暂停")
        self.send_message()
    
    def on_resume(self):
        """Handle resume button."""
        self.input_field.setText("继续")
        self.send_message()
    
    def on_status(self):
        """Handle status button."""
        self.input_field.setText("状态")
        self.send_message()
    
    def on_clear(self):
        """Handle clear button."""
        self.clear_chat()
    
    def clear_chat(self):
        """Clear all messages."""
        for msg in self.messages:
            msg.deleteLater()
        self.messages.clear()
    
    def add_agent_message(self, content: str) -> ChatMessageWidget:
        """Add an agent message."""
        return self.add_message("agent", content)
    
    def set_input_enabled(self, enabled: bool):
        """Enable/disable input."""
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
