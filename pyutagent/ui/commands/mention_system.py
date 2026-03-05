"""Mention system for referencing files, symbols, and context in chat."""

import logging
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QFrame, QAbstractItemView, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QKeyEvent

logger = logging.getLogger(__name__)


@dataclass
class MentionItem:
    """Represents a mentionable item."""
    id: str
    name: str
    item_type: str  # 'file', 'folder', 'symbol', 'current', 'selection'
    icon: str
    path: str = ""
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MentionPopup(QFrame):
    """Popup widget for displaying mention suggestions."""
    
    item_selected = pyqtSignal(MentionItem)  # selected item
    dismissed = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._items: List[MentionItem] = []
        self._filtered_items: List[MentionItem] = []
        
        self.setup_ui()
        self.hide()
        
    def setup_ui(self):
        """Setup the popup UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            MentionPopup {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                border-radius: 6px;
            }
        """)
        self.setMaximumHeight(350)
        self.setMinimumWidth(350)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        
        # Header
        self._header = QLabel("Mention")
        self._header.setStyleSheet("color: #858585; font-size: 11px; padding-bottom: 4px;")
        layout.addWidget(self._header)
        
        # Search within popup
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Filter...")
        self._search_box.setStyleSheet("""
            QLineEdit {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 4px 8px;
                border-radius: 3px;
            }
        """)
        self._search_box.textChanged.connect(self._on_search)
        layout.addWidget(self._search_box)
        
        # Item list
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
                padding: 6px;
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
        help_label = QLabel("Use ↑↓ to navigate, Enter to select, Esc to close")
        help_label.setStyleSheet("color: #666; font-size: 10px; padding-top: 4px;")
        layout.addWidget(help_label)
        
    def set_items(self, items: List[MentionItem], title: str = "Mention"):
        """Set available items."""
        self._items = items
        self._filtered_items = items
        self._header.setText(title)
        self._update_list()
        
    def _on_search(self, text: str):
        """Filter items based on search."""
        if not text:
            self._filtered_items = self._items
        else:
            text_lower = text.lower()
            self._filtered_items = [
                item for item in self._items
                if text_lower in item.name.lower()
                or text_lower in item.description.lower()
            ]
        self._update_list()
        
    def _update_list(self):
        """Update the item list display."""
        self._list.clear()
        
        for item in self._filtered_items:
            list_item = QListWidgetItem()
            
            # Display text
            display_text = f"{item.icon} {item.name}"
            if item.description:
                display_text += f"\n   {item.description}"
            
            list_item.setText(display_text)
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            list_item.setFont(QFont("Segoe UI", 10))
            
            self._list.addItem(list_item)
        
        # Select first item
        if self._filtered_items:
            self._list.setCurrentRow(0)
            
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        mention_item = item.data(Qt.ItemDataRole.UserRole)
        if mention_item:
            self.item_selected.emit(mention_item)
            self.hide()
            
    def select_next(self):
        """Select next item."""
        current = self._list.currentRow()
        if current < self._list.count() - 1:
            self._list.setCurrentRow(current + 1)
            
    def select_previous(self):
        """Select previous item."""
        current = self._list.currentRow()
        if current > 0:
            self._list.setCurrentRow(current - 1)
            
    def get_selected_item(self) -> Optional[MentionItem]:
        """Get currently selected item."""
        item = self._list.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None
        
    def has_items(self) -> bool:
        """Check if there are items to display."""
        return len(self._filtered_items) > 0
        
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.dismissed.emit()
            self.hide()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            item = self.get_selected_item()
            if item:
                self.item_selected.emit(item)
                self.hide()
        elif event.key() == Qt.Key.Key_Down:
            self.select_next()
        elif event.key() == Qt.Key.Key_Up:
            self.select_previous()
        else:
            super().keyPressEvent(event)


class MentionSystem:
    """System for handling @mentions in chat input."""
    
    mention_triggered = pyqtSignal(str, str)  # mention_type, mention_id
    
    def __init__(self):
        self._popup: Optional[MentionPopup] = None
        self._current_input: Optional[QLineEdit] = None
        self._is_active = False
        self._mention_start_pos = -1
        self._project_path: Optional[str] = None
        
    def attach_to_input(self, input_widget: QLineEdit, parent_widget: QWidget):
        """Attach mention system to an input widget.
        
        Args:
            input_widget: The input line edit
            parent_widget: Parent for the popup
        """
        self._current_input = input_widget
        
        # Create popup
        self._popup = MentionPopup(parent_widget)
        self._popup.item_selected.connect(self._on_item_selected)
        self._popup.dismissed.connect(self._on_popup_dismissed)
        
        # Connect signals
        input_widget.textChanged.connect(self._on_text_changed)
        
    def set_project_path(self, path: str):
        """Set the current project path for file discovery."""
        self._project_path = path
        
    def _on_text_changed(self, text: str):
        """Handle input text change."""
        if not self._popup:
            return
        
        # Find @ symbol
        cursor_pos = self._current_input.cursorPosition()
        text_before_cursor = text[:cursor_pos]
        
        # Check if we're in a mention context
        last_at = text_before_cursor.rfind('@')
        
        if last_at != -1:
            # Check if @ is preceded by space or is at start
            if last_at == 0 or text[last_at - 1] == ' ':
                self._mention_start_pos = last_at
                self._is_active = True
                
                # Get query after @
                query = text_before_cursor[last_at + 1:]
                
                # Show appropriate items based on query
                self._show_mention_suggestions(query)
                return
        
        # Not in mention context
        self._is_active = False
        self._popup.hide()
        
    def _show_mention_suggestions(self, query: str):
        """Show mention suggestions based on query."""
        items = self._get_mention_items(query)
        
        if items:
            title = self._get_suggestion_title(query)
            self._popup.set_items(items, title)
            
            # Position popup
            if self._current_input:
                pos = self._current_input.mapToGlobal(
                    self._current_input.rect().bottomLeft()
                )
                self._popup.move(pos)
                self._popup.show()
        else:
            self._popup.hide()
            
    def _get_suggestion_title(self, query: str) -> str:
        """Get title for suggestion popup based on query."""
        if not query:
            return "Mention"
        
        first_char = query[0].lower() if query else ''
        
        if first_char == 'f':
            return "Files"
        elif first_char == 's':
            return "Symbols"
        elif first_char == 'c':
            return "Current"
        elif first_char == 'w':
            return "Workspace"
        else:
            return "Mention"
            
    def _get_mention_items(self, query: str) -> List[MentionItem]:
        """Get mention items based on query."""
        items = []
        
        # Always add special mentions
        items.extend([
            MentionItem(
                id="current",
                name="current",
                item_type="current",
                icon="📄",
                description="Current file"
            ),
            MentionItem(
                id="selection",
                name="selection",
                item_type="selection",
                icon="✂️",
                description="Selected code"
            ),
            MentionItem(
                id="workspace",
                name="workspace",
                item_type="workspace",
                icon="📁",
                description="Entire workspace"
            ),
        ])
        
        # Add files from project
        if self._project_path:
            files = self._get_project_files(query)
            for file_path in files[:20]:  # Limit to 20 files
                path = Path(file_path)
                icon = self._get_file_icon(path.suffix)
                items.append(MentionItem(
                    id=file_path,
                    name=path.name,
                    item_type="file",
                    icon=icon,
                    path=file_path,
                    description=str(path.parent)
                ))
        
        # Filter by query if provided
        if query:
            query_lower = query.lower()
            items = [
                item for item in items
                if query_lower in item.name.lower()
                or query_lower in item.item_type.lower()
            ]
        
        return items
        
    def _get_project_files(self, query: str) -> List[str]:
        """Get files from project matching query."""
        if not self._project_path:
            return []
        
        files = []
        project_path = Path(self._project_path)
        
        try:
            # Search for files
            for file_path in project_path.rglob('*'):
                if file_path.is_file():
                    # Skip hidden and common ignore patterns
                    if any(part.startswith('.') for part in file_path.parts):
                        continue
                    if any(ignore in str(file_path) for ignore in ['node_modules', '__pycache__', 'target', 'build']):
                        continue
                    
                    # Check if matches query
                    if not query or query.lower() in file_path.name.lower():
                        files.append(str(file_path))
                        
                    if len(files) >= 50:  # Limit search
                        break
                        
        except Exception as e:
            logger.warning(f"Failed to get project files: {e}")
        
        return files
        
    def _get_file_icon(self, extension: str) -> str:
        """Get icon for file extension."""
        icons = {
            '.py': '🐍',
            '.java': '☕',
            '.js': '📜',
            '.ts': '🔷',
            '.go': '🐹',
            '.rs': '🦀',
            '.cpp': '🔧',
            '.c': '🔧',
            '.cs': '🔵',
            '.rb': '💎',
            '.php': '🐘',
            '.html': '🌐',
            '.css': '🎨',
            '.json': '📋',
            '.md': '📝',
            '.sql': '🗄️',
            '.xml': '📋',
            '.yaml': '⚙️',
            '.yml': '⚙️',
        }
        return icons.get(extension.lower(), '📄')
        
    def _on_item_selected(self, item: MentionItem):
        """Handle item selection."""
        if self._current_input and self._mention_start_pos >= 0:
            text = self._current_input.text()
            cursor_pos = self._current_input.cursorPosition()
            
            # Replace the mention text with the selected item
            before_mention = text[:self._mention_start_pos]
            after_mention = text[cursor_pos:]
            
            new_text = f"{before_mention}@{item.name}{after_mention}"
            self._current_input.setText(new_text)
            
            # Move cursor after the mention
            new_cursor_pos = len(before_mention) + len(item.name) + 1  # +1 for @
            self._current_input.setCursorPosition(new_cursor_pos)
            
            # Emit signal
            self.mention_triggered.emit(item.item_type, item.id)
            
        self._is_active = False
        self._mention_start_pos = -1
        
    def _on_popup_dismissed(self):
        """Handle popup dismissal."""
        self._is_active = False
        self._mention_start_pos = -1
        
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
        
    def parse_mentions(self, text: str) -> List[Dict[str, str]]:
        """Parse mentions from text.
        
        Returns:
            List of dicts with 'type' and 'id' keys
        """
        import re
        mentions = []
        
        # Find @mentions
        pattern = r'@(\w+)'
        for match in re.finditer(pattern, text):
            name = match.group(1)
            
            # Determine type based on name
            if name in ['current', 'selection', 'workspace']:
                mentions.append({
                    'type': name,
                    'id': name,
                    'text': match.group(0)
                })
            else:
                # Assume it's a file
                mentions.append({
                    'type': 'file',
                    'id': name,
                    'text': match.group(0)
                })
        
        return mentions
        
    def is_active(self) -> bool:
        """Check if mention system is active."""
        return self._is_active
