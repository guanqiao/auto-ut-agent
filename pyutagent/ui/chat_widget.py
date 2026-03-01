"""Chat widget for Agent conversation with streaming support."""

import logging
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QScrollArea, QLabel,
    QFrame, QSizePolicy, QTextBrowser
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QColor, QFont, QTextCursor

logger = logging.getLogger(__name__)


class StreamingMessageWidget(QFrame):
    """Widget for streaming message display with code highlighting."""
    
    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._role = role
        self._content = ""
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        avatar_label = QLabel("🤖" if role == "agent" else "👤")
        avatar_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(avatar_label)
        
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        self.content_browser.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        self.content_browser.setMinimumHeight(30)
        self.content_browser.setStyleSheet("""
            QTextBrowser {
                border: none;
                background: transparent;
            }
        """)
        
        layout.addWidget(self.content_browser, stretch=1)
        
        if role == "agent":
            self.setStyleSheet("""
                StreamingMessageWidget {
                    background-color: #e3f2fd;
                    border-radius: 10px;
                    margin: 5px 50px 5px 5px;
                }
            """)
        else:
            self.setStyleSheet("""
                StreamingMessageWidget {
                    background-color: #f5f5f5;
                    border-radius: 10px;
                    margin: 5px 5px 5px 50px;
                }
            """)
    
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
        formatted = self._format_content(self._content)
        self.content_browser.setHtml(formatted)
        
        QTimer.singleShot(10, self._adjust_height)
    
    def _format_content(self, content: str) -> str:
        """Format content with basic highlighting."""
        formatted = content
        
        formatted = formatted.replace('&', '&amp;')
        formatted = formatted.replace('<', '&lt;')
        formatted = formatted.replace('>', '&gt;')
        
        import re
        
        code_block_pattern = r'```(\w*)\n(.*?)```'
        def replace_code_block(match):
            lang = match.group(1) or 'java'
            code = match.group(2)
            return f'<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto;"><code class="{lang}">{code}</code></pre>'
        formatted = re.sub(code_block_pattern, replace_code_block, formatted, flags=re.DOTALL)
        
        inline_code_pattern = r'`([^`]+)`'
        formatted = re.sub(inline_code_pattern, r'<code style="background-color: #e0e0e0; padding: 2px 5px; border-radius: 3px;">\1</code>', formatted)
        
        bold_pattern = r'\*\*([^*]+)\*\*'
        formatted = re.sub(bold_pattern, r'<b>\1</b>', formatted)
        
        formatted = formatted.replace('\n', '<br>')
        
        return f'<div style="font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;">{formatted}</div>'
    
    def _adjust_height(self):
        """Adjust height to fit content."""
        doc = self.content_browser.document()
        doc.setTextWidth(self.content_browser.width())
        height = doc.size().height() + 20
        self.content_browser.setMinimumHeight(int(height))
    
    def get_content(self) -> str:
        """Get the raw content."""
        return self._content


class ChatMessageWidget(QFrame):
    """Single chat message widget."""
    
    def __init__(self, role: str, content: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        avatar_label = QLabel("🤖" if role == "agent" else "👤")
        avatar_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(avatar_label)
        
        self.content_label = QLabel(content)
        self.content_label.setWordWrap(True)
        self.content_label.setTextFormat(Qt.TextFormat.PlainText)
        self.content_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        
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
    """Chat widget for Agent conversation with streaming support."""
    
    message_sent = pyqtSignal(str)
    generate_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    resume_clicked = pyqtSignal()
    terminate_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.messages: list = []
        self._is_running = False
        self._is_paused = False
        self._current_streaming_widget: Optional[StreamingMessageWidget] = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        header = QLabel("💬 对话")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.messages_layout = QVBoxLayout(self.scroll_content)
        self.messages_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)
        
        # Quick action buttons
        quick_buttons_layout = QHBoxLayout()
        
        # Generate button
        self.generate_btn = QPushButton("▶ 生成测试")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.generate_btn.clicked.connect(self.on_generate)
        quick_buttons_layout.addWidget(self.generate_btn)
        
        # Pause button
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.pause_btn.clicked.connect(self.on_pause)
        self.pause_btn.setEnabled(False)  # Disabled by default
        quick_buttons_layout.addWidget(self.pause_btn)
        
        # Resume button
        self.resume_btn = QPushButton("▶ 恢复")
        self.resume_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.resume_btn.clicked.connect(self.on_resume)
        self.resume_btn.setEnabled(False)  # Disabled by default
        quick_buttons_layout.addWidget(self.resume_btn)
        
        # Terminate button
        self.terminate_btn = QPushButton("⏹ 终止")
        self.terminate_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.terminate_btn.clicked.connect(self.on_terminate)
        self.terminate_btn.setEnabled(False)  # Disabled by default
        quick_buttons_layout.addWidget(self.terminate_btn)
        
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
    
    def add_message(self, role: str, content: str):
        """Add a message to the chat."""
        try:
            msg_widget = ChatMessageWidget(role, content)
            self.messages_layout.insertWidget(
                self.messages_layout.count() - 1,
                msg_widget
            )
            self.messages.append(msg_widget)

            self.scroll_to_bottom()

            logger.debug(f"Added {role} message to chat")
            return msg_widget
        except Exception as e:
            logger.exception("Failed to add message to chat")
            raise
    
    def add_streaming_message(self, role: str = "agent") -> StreamingMessageWidget:
        """Add a streaming message widget.
        
        Args:
            role: Role for the message ('agent' or 'user')
            
        Returns:
            StreamingMessageWidget for appending chunks
        """
        try:
            msg_widget = StreamingMessageWidget(role)
            self.messages_layout.insertWidget(
                self.messages_layout.count() - 1,
                msg_widget
            )
            self.messages.append(msg_widget)
            self._current_streaming_widget = msg_widget
            
            self.scroll_to_bottom()
            
            logger.debug(f"Added streaming {role} message to chat")
            return msg_widget
        except Exception as e:
            logger.exception("Failed to add streaming message to chat")
            raise
    
    def append_to_streaming(self, chunk: str):
        """Append a chunk to the current streaming message.
        
        Args:
            chunk: Text chunk to append
        """
        if self._current_streaming_widget:
            self._current_streaming_widget.append_chunk(chunk)
            self.scroll_to_bottom()
    
    def finish_streaming(self, final_content: Optional[str] = None):
        """Finish the current streaming message.
        
        Args:
            final_content: Optional final content to set
        """
        if self._current_streaming_widget:
            if final_content:
                self._current_streaming_widget.update_content(final_content)
            self._current_streaming_widget = None
    
    def update_last_message(self, content: str):
        """Update the last message (for streaming)."""
        if self.messages:
            last = self.messages[-1]
            if isinstance(last, StreamingMessageWidget):
                last.append_chunk(content)
            elif isinstance(last, ChatMessageWidget):
                current_text = last.content_label.text()
                last.update_content(current_text + content)
            self.scroll_to_bottom()
    
    def send_message(self):
        """Send user message."""
        try:
            text = self.input_field.text().strip()
            if text:
                self.add_message("user", text)
                self.input_field.clear()
                self.message_sent.emit(text)
                logger.info(f"User message sent: {text[:50]}...")
        except Exception as e:
            logger.exception("Failed to send message")
    
    def scroll_to_bottom(self):
        """Scroll to the bottom of the chat."""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_generate(self):
        """Handle generate button."""
        self.generate_clicked.emit()
    
    def on_pause(self):
        """Handle pause button."""
        self.pause_clicked.emit()
    
    def on_resume(self):
        """Handle resume button."""
        self.resume_clicked.emit()
    
    def on_terminate(self):
        """Handle terminate button."""
        self.terminate_clicked.emit()
    
    def set_running_state(self, is_running: bool, is_paused: bool = False):
        """Update button states based on running state.
        
        Args:
            is_running: Whether the agent is currently running
            is_paused: Whether the agent is currently paused
        """
        self._is_running = is_running
        self._is_paused = is_paused
        
        if not is_running:
            # Not running - only generate button enabled
            self.generate_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.terminate_btn.setEnabled(False)
        elif is_paused:
            # Paused - resume and terminate enabled
            self.generate_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(True)
            self.terminate_btn.setEnabled(True)
        else:
            # Running - pause and terminate enabled
            self.generate_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.resume_btn.setEnabled(False)
            self.terminate_btn.setEnabled(True)
        
        logger.debug(f"[ChatWidget] Button states updated - running: {is_running}, paused: {is_paused}")
    
    def on_status(self):
        """Handle status button."""
        self.input_field.setText("状态")
        self.send_message()
    
    def on_clear(self):
        """Handle clear button."""
        self.clear_chat()
    
    def clear_chat(self):
        """Clear all messages."""
        try:
            for msg in self.messages:
                msg.deleteLater()
            self.messages.clear()
            logger.info("Chat cleared")
        except Exception as e:
            logger.exception("Failed to clear chat")
    
    def add_agent_message(self, content: str) -> ChatMessageWidget:
        """Add an agent message."""
        try:
            result = self.add_message("agent", content)
            logger.debug(f"Agent message added: {content[:50]}...")
            return result
        except Exception as e:
            logger.exception("Failed to add agent message")
            raise
    
    def set_input_enabled(self, enabled: bool):
        """Enable/disable input."""
        self.input_field.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
