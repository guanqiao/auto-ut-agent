"""Slash command system for quick AI actions."""

import logging
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QFrame, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QKeyEvent

logger = logging.getLogger(__name__)


class CommandCategory(Enum):
    """Category of slash command."""
    CODE = "code"
    TEST = "test"
    EXPLAIN = "explain"
    REFACTOR = "refactor"
    DOCUMENT = "document"
    REVIEW = "review"
    FIX = "fix"
    OTHER = "other"


@dataclass
class SlashCommand:
    """Represents a slash command."""
    name: str
    description: str
    category: CommandCategory
    icon: str = "💬"
    shortcut: Optional[str] = None
    handler: Optional[Callable] = None
    args_hint: Optional[str] = None
    example: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        """Get display name with slash."""
        return f"/{self.name}"


# Default slash commands
DEFAULT_COMMANDS: List[SlashCommand] = [
    # Code generation
    SlashCommand(
        name="generate",
        description="Generate code based on description",
        category=CommandCategory.CODE,
        icon="✨",
        args_hint="<description>",
        example="/generate a function to calculate fibonacci"
    ),
    SlashCommand(
        name="test",
        description="Generate unit tests for selected code or file",
        category=CommandCategory.TEST,
        icon="🧪",
        args_hint="[file_path]",
        example="/test UserService.java"
    ),
    SlashCommand(
        name="explain",
        description="Explain the selected code or concept",
        category=CommandCategory.EXPLAIN,
        icon="📖",
        args_hint="[code or concept]",
        example="/explain what does this function do?"
    ),
    SlashCommand(
        name="refactor",
        description="Refactor the selected code",
        category=CommandCategory.REFACTOR,
        icon="🔧",
        args_hint="[instructions]",
        example="/refactor extract this into a separate method"
    ),
    SlashCommand(
        name="doc",
        description="Generate documentation for code",
        category=CommandCategory.DOCUMENT,
        icon="📝",
        args_hint="[file_path]",
        example="/doc UserService.java"
    ),
    SlashCommand(
        name="review",
        description="Review code for issues and improvements",
        category=CommandCategory.REVIEW,
        icon="👀",
        args_hint="[file_path]",
        example="/review UserService.java"
    ),
    SlashCommand(
        name="fix",
        description="Fix bugs or issues in code",
        category=CommandCategory.FIX,
        icon="🐛",
        args_hint="[description of issue]",
        example="/fix this null pointer exception"
    ),
    SlashCommand(
        name="optimize",
        description="Optimize code for performance",
        category=CommandCategory.REFACTOR,
        icon="⚡",
        args_hint="[file_path]",
        example="/optimize DatabaseQuery.java"
    ),
    SlashCommand(
        name="type",
        description="Add type annotations or hints",
        category=CommandCategory.CODE,
        icon="🏷️",
        args_hint="[file_path]",
        example="/type utils.py"
    ),
    SlashCommand(
        name="translate",
        description="Translate code to another language",
        category=CommandCategory.CODE,
        icon="🌐",
        args_hint="<target_language>",
        example="/translate Python"
    ),
    SlashCommand(
        name="regex",
        description="Generate or explain regular expressions",
        category=CommandCategory.CODE,
        icon="🔍",
        args_hint="<description>",
        example="/regex match email addresses"
    ),
    SlashCommand(
        name="sql",
        description="Generate or optimize SQL queries",
        category=CommandCategory.CODE,
        icon="🗄️",
        args_hint="<description>",
        example="/sql select users with active orders"
    ),
    SlashCommand(
        name="mock",
        description="Generate mock data or fixtures",
        category=CommandCategory.TEST,
        icon="🎭",
        args_hint="<data_type>",
        example="/mock 10 user records"
    ),
    SlashCommand(
        name="commit",
        description="Generate commit message",
        category=CommandCategory.OTHER,
        icon="💾",
        args_hint="[changes_description]",
        example="/commit added user authentication"
    ),
    SlashCommand(
        name="readme",
        description="Generate README for project",
        category=CommandCategory.DOCUMENT,
        icon="📄",
        args_hint="",
        example="/readme"
    ),
    SlashCommand(
        name="diagram",
        description="Generate architecture diagram description",
        category=CommandCategory.DOCUMENT,
        icon="📊",
        args_hint="[component_description]",
        example="/diagram system architecture"
    ),
    SlashCommand(
        name="security",
        description="Check code for security issues",
        category=CommandCategory.REVIEW,
        icon="🔒",
        args_hint="[file_path]",
        example="/security AuthController.java"
    ),
    SlashCommand(
        name="clean",
        description="Clean up code (remove unused imports, etc.)",
        category=CommandCategory.REFACTOR,
        icon="🧹",
        args_hint="[file_path]",
        example="/clean Main.java"
    ),
]


class SlashCommandPopup(QFrame):
    """Popup widget for displaying slash command suggestions."""
    
    command_selected = pyqtSignal(SlashCommand)  # selected command
    dismissed = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._commands: List[SlashCommand] = []
        self._filtered_commands: List[SlashCommand] = []
        self._selected_index = 0
        
        self.setup_ui()
        self.hide()
        
    def setup_ui(self):
        """Setup the popup UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            SlashCommandPopup {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                border-radius: 6px;
            }
        """)
        self.setMaximumHeight(300)
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header
        header = QLabel("Commands")
        header.setStyleSheet("color: #858585; font-size: 11px; padding-bottom: 4px;")
        layout.addWidget(header)
        
        # Command list
        self._list = QListWidget()
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                background-color: transparent;
                border-radius: 4px;
                padding: 8px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:hover {
                background-color: #2A2D2E;
            }
        """)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)
        
        # Help text
        self._help_label = QLabel("Use ↑↓ to navigate, Enter to select, Esc to close")
        self._help_label.setStyleSheet("color: #666; font-size: 10px; padding-top: 4px;")
        layout.addWidget(self._help_label)
        
    def set_commands(self, commands: List[SlashCommand]):
        """Set available commands."""
        self._commands = commands
        self._filtered_commands = commands
        self._update_list()
        
    def filter_commands(self, query: str):
        """Filter commands based on query."""
        if not query:
            self._filtered_commands = self._commands
        else:
            query_lower = query.lower()
            self._filtered_commands = [
                cmd for cmd in self._commands
                if query_lower in cmd.name.lower() 
                or query_lower in cmd.description.lower()
            ]
        
        self._selected_index = 0
        self._update_list()
        
    def _update_list(self):
        """Update the command list display."""
        self._list.clear()
        
        for cmd in self._filtered_commands:
            item = QListWidgetItem()
            
            # Create display text
            display_text = f"{cmd.icon} {cmd.display_name}"
            if cmd.args_hint:
                display_text += f" {cmd.args_hint}"
            display_text += f"\n   {cmd.description}"
            
            item.setText(display_text)
            item.setData(Qt.ItemDataRole.UserRole, cmd)
            item.setFont(QFont("Segoe UI", 10))
            
            self._list.addItem(item)
        
        # Select first item
        if self._filtered_commands:
            self._list.setCurrentRow(0)
            
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        cmd = item.data(Qt.ItemDataRole.UserRole)
        if cmd:
            self.command_selected.emit(cmd)
            self.hide()
            
    def select_next(self):
        """Select next command."""
        current = self._list.currentRow()
        if current < self._list.count() - 1:
            self._list.setCurrentRow(current + 1)
            
    def select_previous(self):
        """Select previous command."""
        current = self._list.currentRow()
        if current > 0:
            self._list.setCurrentRow(current - 1)
            
    def get_selected_command(self) -> Optional[SlashCommand]:
        """Get currently selected command."""
        item = self._list.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None
        
    def has_commands(self) -> bool:
        """Check if there are commands to display."""
        return len(self._filtered_commands) > 0
        
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.dismissed.emit()
            self.hide()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            cmd = self.get_selected_command()
            if cmd:
                self.command_selected.emit(cmd)
                self.hide()
        elif event.key() == Qt.Key.Key_Down:
            self.select_next()
        elif event.key() == Qt.Key.Key_Up:
            self.select_previous()
        else:
            super().keyPressEvent(event)


class SlashCommandHandler:
    """Handler for slash commands in chat input."""
    
    command_triggered = pyqtSignal(str, str)  # command_name, args
    
    def __init__(self):
        self._commands: Dict[str, SlashCommand] = {}
        self._popup: Optional[SlashCommandPopup] = None
        self._current_input: Optional[QLineEdit] = None
        self._is_active = False
        
        # Register default commands
        self._register_default_commands()
        
    def _register_default_commands(self):
        """Register default slash commands."""
        for cmd in DEFAULT_COMMANDS:
            self._commands[cmd.name] = cmd
            
    def register_command(self, command: SlashCommand):
        """Register a custom command.
        
        Args:
            command: Command to register
        """
        self._commands[command.name] = command
        logger.info(f"Registered slash command: {command.name}")
        
    def unregister_command(self, name: str):
        """Unregister a command.
        
        Args:
            name: Command name
        """
        if name in self._commands:
            del self._commands[name]
            logger.info(f"Unregistered slash command: {name}")
            
    def attach_to_input(self, input_widget: QLineEdit, parent_widget: QWidget):
        """Attach handler to an input widget.
        
        Args:
            input_widget: The input line edit
            parent_widget: Parent for the popup
        """
        self._current_input = input_widget
        
        # Create popup
        self._popup = SlashCommandPopup(parent_widget)
        self._popup.set_commands(list(self._commands.values()))
        self._popup.command_selected.connect(self._on_command_selected)
        self._popup.dismissed.connect(self._on_popup_dismissed)
        
        # Connect input signals
        input_widget.textChanged.connect(self._on_text_changed)
        
    def _on_text_changed(self, text: str):
        """Handle input text change."""
        if not self._popup:
            return
            
        # Check if starting with slash
        if text.startswith('/'):
            self._is_active = True
            
            # Extract command query
            query = text[1:]  # Remove leading slash
            parts = query.split(' ', 1)
            cmd_query = parts[0]
            
            # Filter and show popup
            self._popup.filter_commands(cmd_query)
            
            if self._popup.has_commands():
                # Position popup below input
                if self._current_input:
                    pos = self._current_input.mapToGlobal(
                        self._current_input.rect().bottomLeft()
                    )
                    self._popup.move(pos)
                    self._popup.show()
            else:
                self._popup.hide()
        else:
            self._is_active = False
            if self._popup:
                self._popup.hide()
                
    def _on_command_selected(self, command: SlashCommand):
        """Handle command selection."""
        if self._current_input:
            # Update input with selected command
            current_text = self._current_input.text()
            parts = current_text.split(' ', 1)
            
            # Replace command part
            new_text = f"/{command.name}"
            if len(parts) > 1:
                new_text += f" {parts[1]}"
            
            self._current_input.setText(new_text)
            self._current_input.setFocus()
            
            # Move cursor to end of command name
            self._current_input.setCursorPosition(len(new_text))
            
        self._is_active = False
        
    def _on_popup_dismissed(self):
        """Handle popup dismissal."""
        self._is_active = False
        
    def handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle key press event.
        
        Returns:
            True if event was handled
        """
        if not self._is_active or not self._popup:
            return False
            
        if event.key() in [Qt.Key.Key_Up, Qt.Key.Key_Down, 
                          Qt.Key.Key_Return, Qt.Key.Key_Enter,
                          Qt.Key.Key_Escape]:
            self._popup.keyPressEvent(event)
            return True
            
        return False
        
    def execute_command(self, text: str) -> bool:
        """Execute a slash command from text.
        
        Args:
            text: Command text (e.g., "/test UserService.java")
            
        Returns:
            True if command was executed
        """
        if not text.startswith('/'):
            return False
            
        # Parse command
        parts = text[1:].split(' ', 1)
        cmd_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd_name in self._commands:
            self.command_triggered.emit(cmd_name, args)
            logger.info(f"Executed command: {cmd_name} with args: {args}")
            return True
            
        return False
        
    def get_command_help(self, name: str) -> Optional[str]:
        """Get help text for a command.
        
        Args:
            name: Command name (without slash)
            
        Returns:
            Help text or None
        """
        cmd = self._commands.get(name)
        if cmd:
            help_text = f"{cmd.icon} {cmd.display_name}"
            if cmd.args_hint:
                help_text += f" {cmd.args_hint}"
            help_text += f"\n\n{cmd.description}"
            if cmd.example:
                help_text += f"\n\nExample: {cmd.example}"
            return help_text
        return None
        
    def get_all_commands(self) -> List[SlashCommand]:
        """Get all registered commands."""
        return list(self._commands.values())
        
    def is_command(self, text: str) -> bool:
        """Check if text is a command."""
        if not text.startswith('/'):
            return False
            
        parts = text[1:].split(' ', 1)
        return parts[0] in self._commands
