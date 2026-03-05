"""Main Agent panel container with multiple modes."""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

from .chat_mode import ChatMode
from .agent_mode import AgentMode
from .context_manager import ContextManager

logger = logging.getLogger(__name__)


class AgentPanel(QWidget):
    """Main Agent panel with multiple modes.
    
    Modes:
    - Chat: Traditional chat interface
    - Agent: Shows thinking chain and tool calls
    
    Features:
    - Mode switching
    - Context management
    - Session management
    """
    
    # Signals
    message_sent = pyqtSignal(str)  # message_text
    generate_requested = pyqtSignal()  # Generate button clicked
    context_changed = pyqtSignal()  # Context items changed
    
    # Mode constants
    MODE_CHAT = 0
    MODE_AGENT = 1
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_mode = self.MODE_CHAT
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the agent panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with mode switcher
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border-bottom: 1px solid #E0E0E0;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(4)
        
        # Title
        title = QLabel("🤖 Agent")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Mode buttons
        self._btn_chat = QPushButton("💬 Chat")
        self._btn_chat.setCheckable(True)
        self._btn_chat.setChecked(True)
        self._btn_chat.clicked.connect(lambda: self.set_mode(self.MODE_CHAT))
        header_layout.addWidget(self._btn_chat)
        
        self._btn_agent = QPushButton("🧠 Agent")
        self._btn_agent.setCheckable(True)
        self._btn_agent.clicked.connect(lambda: self.set_mode(self.MODE_AGENT))
        header_layout.addWidget(self._btn_agent)
        
        layout.addWidget(header)
        
        # Context manager (collapsible)
        self._context_manager = ContextManager()
        self._context_manager.setMaximumHeight(250)
        self._context_manager.context_changed.connect(self.context_changed.emit)
        layout.addWidget(self._context_manager)
        
        # Mode stack
        self._stack = QStackedWidget()
        
        # Chat mode
        self._chat_mode = ChatMode()
        self._chat_mode.message_sent.connect(self.message_sent.emit)
        self._chat_mode._btn_generate.clicked.connect(self.generate_requested.emit)
        self._stack.addWidget(self._chat_mode)
        
        # Agent mode
        self._agent_mode = AgentMode()
        self._stack.addWidget(self._agent_mode)
        
        layout.addWidget(self._stack, stretch=1)
        
    def set_mode(self, mode: int):
        """Set the display mode.
        
        Args:
            mode: One of MODE_CHAT or MODE_AGENT
        """
        if mode not in [self.MODE_CHAT, self.MODE_AGENT]:
            logger.warning(f"Invalid mode: {mode}")
            return
        
        self._current_mode = mode
        self._stack.setCurrentIndex(mode)
        
        # Update button states
        self._btn_chat.setChecked(mode == self.MODE_CHAT)
        self._btn_agent.setChecked(mode == self.MODE_AGENT)
        
        logger.debug(f"Agent panel mode changed to: {mode}")
        
    def get_chat_mode(self) -> ChatMode:
        """Get the chat mode widget."""
        return self._chat_mode
        
    def get_agent_mode(self) -> AgentMode:
        """Get the agent mode widget."""
        return self._agent_mode
        
    def get_context_manager(self) -> ContextManager:
        """Get the context manager."""
        return self._context_manager
        
    def add_user_message(self, content: str):
        """Add a user message to chat."""
        self._chat_mode.add_user_message(content)
        
    def add_agent_message(self, content: str):
        """Add an agent message to chat."""
        self._chat_mode.add_agent_message(content)
        
    def start_streaming_response(self) -> str:
        """Start a streaming response in chat mode."""
        return self._chat_mode.start_streaming_response()
        
    def append_to_streaming(self, text: str):
        """Append to streaming response."""
        self._chat_mode.append_to_streaming(text)
        
    def finish_streaming(self):
        """Finish streaming response."""
        self._chat_mode.finish_streaming()
        
    def start_agent_task(self, task_name: str):
        """Start an agent task.
        
        Automatically switches to agent mode.
        """
        self.set_mode(self.MODE_AGENT)
        self._agent_mode.start_task(task_name)
        
    def set_agent_status(self, status: str, status_type: str = "info"):
        """Set agent status."""
        self._agent_mode.set_status(status, status_type)
        
    def add_file_to_context(self, file_path: str) -> bool:
        """Add a file to context."""
        return self._context_manager.add_file(file_path)
        
    def clear_context(self):
        """Clear all context."""
        self._context_manager.clear_context()
        
    def get_current_mode(self) -> int:
        """Get current mode."""
        return self._current_mode
