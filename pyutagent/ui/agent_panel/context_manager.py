"""Context manager for Agent panel - manages files and context selection."""

import logging
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QMenu, QAbstractItemView,
    QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QFont

logger = logging.getLogger(__name__)


@dataclass
class ContextItem:
    """Represents a context item (file, folder, or code snippet)."""
    id: str
    name: str
    path: str
    item_type: str  # 'file', 'folder', 'snippet', 'current_file', 'selection'
    icon: str = "📄"
    token_count: int = 0
    priority: int = 0  # Higher = more important
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, ContextItem):
            return self.id == other.id
        return False


class ContextManager(QWidget):
    """Manages context for Agent conversations.
    
    Features:
    - Add/remove files from context
    - Show token usage
    - Priority management
    - @mention support
    """
    
    context_changed = pyqtSignal()  # Emitted when context changes
    item_removed = pyqtSignal(str)  # item_id
    item_clicked = pyqtSignal(str)  # item_id
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._items: Dict[str, ContextItem] = {}
        self._max_context_items = 20
        self._estimated_tokens = 0
        self._max_tokens = 128000  # Default max context
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the context manager UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header
        header = QHBoxLayout()
        
        self._title_label = QLabel("📋 Context")
        self._title_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.addWidget(self._title_label)
        
        header.addStretch()
        
        # Token usage label
        self._token_label = QLabel("0 tokens")
        self._token_label.setStyleSheet("color: #666; font-size: 11px;")
        header.addWidget(self._token_label)
        
        # Clear button
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedHeight(24)
        self._clear_btn.clicked.connect(self.clear_context)
        header.addWidget(self._clear_btn)
        
        layout.addLayout(header)
        
        # Context list
        self._list_widget = QListWidget()
        self._list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list_widget.customContextMenuRequested.connect(self._on_context_menu)
        self._list_widget.itemClicked.connect(self._on_item_clicked)
        self._list_widget.setMaximumHeight(200)
        layout.addWidget(self._list_widget)
        
        # Quick add buttons
        quick_add = QHBoxLayout()
        
        self._add_current_btn = QPushButton("📄 Current")
        self._add_current_btn.setToolTip("Add current file")
        self._add_current_btn.clicked.connect(self.add_current_file)
        quick_add.addWidget(self._add_current_btn)
        
        self._add_selection_btn = QPushButton("✂️ Selection")
        self._add_selection_btn.setToolTip("Add selected code")
        self._add_selection_btn.clicked.connect(self.add_selection)
        quick_add.addWidget(self._add_selection_btn)
        
        quick_add.addStretch()
        
        layout.addLayout(quick_add)
        
        # Help text
        help_label = QLabel("💡 Use @ to mention files in chat")
        help_label.setStyleSheet("color: #999; font-size: 10px;")
        layout.addWidget(help_label)
        
        layout.addStretch()
        
    def add_item(self, item: ContextItem) -> bool:
        """Add an item to context.
        
        Args:
            item: Context item to add
            
        Returns:
            True if added, False if at limit or already exists
        """
        if item.id in self._items:
            logger.debug(f"Item already in context: {item.id}")
            return False
        
        if len(self._items) >= self._max_context_items:
            logger.warning("Context limit reached")
            return False
        
        self._items[item.id] = item
        self._add_list_item(item)
        self._update_token_count()
        self.context_changed.emit()
        
        logger.info(f"Added to context: {item.name}")
        return True
        
    def _add_list_item(self, item: ContextItem):
        """Add item to list widget."""
        list_item = QListWidgetItem(f"{item.icon} {item.name}")
        list_item.setData(Qt.ItemDataRole.UserRole, item.id)
        list_item.setToolTip(f"{item.path}\nTokens: ~{item.token_count}")
        
        # Set color based on token count
        if item.token_count > 4000:
            list_item.setForeground(Qt.GlobalColor.red)
        elif item.token_count > 1000:
            list_item.setForeground(Qt.GlobalColor.darkYellow)
        
        self._list_widget.addItem(list_item)
        
    def remove_item(self, item_id: str):
        """Remove an item from context."""
        if item_id not in self._items:
            return
        
        item = self._items.pop(item_id)
        
        # Remove from list
        for i in range(self._list_widget.count()):
            list_item = self._list_widget.item(i)
            if list_item.data(Qt.ItemDataRole.UserRole) == item_id:
                self._list_widget.takeItem(i)
                break
        
        self._update_token_count()
        self.item_removed.emit(item_id)
        self.context_changed.emit()
        
        logger.info(f"Removed from context: {item.name}")
        
    def clear_context(self):
        """Clear all context items."""
        self._items.clear()
        self._list_widget.clear()
        self._update_token_count()
        self.context_changed.emit()
        
        logger.info("Context cleared")
        
    def add_file(self, file_path: str, priority: int = 0) -> bool:
        """Add a file to context.
        
        Args:
            file_path: Path to the file
            priority: Priority level (higher = more important)
            
        Returns:
            True if added
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return False
        
        # Estimate token count (rough approximation)
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            token_count = len(content) // 4  # Rough estimate: 4 chars per token
        except Exception:
            token_count = 0
        
        # Determine icon based on file type
        icon = "📄"
        if path.suffix in ['.py']:
            icon = "🐍"
        elif path.suffix in ['.java']:
            icon = "☕"
        elif path.suffix in ['.js', '.ts']:
            icon = "📜"
        
        item = ContextItem(
            id=f"file:{file_path}",
            name=path.name,
            path=file_path,
            item_type="file",
            icon=icon,
            token_count=token_count,
            priority=priority
        )
        
        return self.add_item(item)
        
    def add_current_file(self):
        """Add current file to context."""
        # TODO: Get current file from editor
        logger.info("Add current file requested")
        
    def add_selection(self):
        """Add selected code to context."""
        # TODO: Get selection from editor
        logger.info("Add selection requested")
        
    def add_folder(self, folder_path: str) -> bool:
        """Add a folder to context.
        
        Args:
            folder_path: Path to the folder
            
        Returns:
            True if added
        """
        path = Path(folder_path)
        if not path.exists() or not path.is_dir():
            return False
        
        item = ContextItem(
            id=f"folder:{folder_path}",
            name=path.name + "/",
            path=folder_path,
            item_type="folder",
            icon="📁",
            token_count=0,  # Folders don't have direct token count
            priority=0
        )
        
        return self.add_item(item)
        
    def add_snippet(self, name: str, content: str, language: str = "") -> bool:
        """Add a code snippet to context.
        
        Args:
            name: Name of the snippet
            content: Code content
            language: Programming language
            
        Returns:
            True if added
        """
        token_count = len(content) // 4
        
        icon = "✂️"
        if language == "python":
            icon = "🐍"
        elif language == "java":
            icon = "☕"
        
        item = ContextItem(
            id=f"snippet:{name}:{hash(content)}",
            name=name,
            path="",
            item_type="snippet",
            icon=icon,
            token_count=token_count,
            priority=1  # Snippets are usually high priority
        )
        
        return self.add_item(item)
        
    def get_context_items(self) -> List[ContextItem]:
        """Get all context items."""
        return list(self._items.values())
        
    def get_context_files(self) -> List[str]:
        """Get file paths in context."""
        return [
            item.path for item in self._items.values()
            if item.item_type == 'file'
        ]
        
    def _update_token_count(self):
        """Update token count display."""
        total = sum(item.token_count for item in self._items.values())
        self._estimated_tokens = total
        
        self._token_label.setText(f"~{total:,} tokens")
        
        # Color code based on usage
        if total > self._max_tokens * 0.9:
            self._token_label.setStyleSheet("color: #F44336; font-weight: bold;")
        elif total > self._max_tokens * 0.7:
            self._token_label.setStyleSheet("color: #FF9800;")
        else:
            self._token_label.setStyleSheet("color: #4CAF50;")
            
    def set_max_tokens(self, max_tokens: int):
        """Set maximum token limit."""
        self._max_tokens = max_tokens
        self._update_token_count()
        
    def _on_context_menu(self, position):
        """Show context menu for list item."""
        item = self._list_widget.itemAt(position)
        if not item:
            return
        
        item_id = item.data(Qt.ItemDataRole.UserRole)
        
        menu = QMenu(self)
        
        # Remove action
        remove_action = menu.addAction("🗑️ Remove from context")
        remove_action.triggered.connect(lambda: self.remove_item(item_id))
        
        # Priority actions
        menu.addSeparator()
        
        high_priority = menu.addAction("⭐ High Priority")
        high_priority.triggered.connect(lambda: self._set_priority(item_id, 2))
        
        normal_priority = menu.addAction("🔵 Normal Priority")
        normal_priority.triggered.connect(lambda: self._set_priority(item_id, 1))
        
        low_priority = menu.addAction("⚪ Low Priority")
        low_priority.triggered.connect(lambda: self._set_priority(item_id, 0))
        
        menu.exec(self._list_widget.viewport().mapToGlobal(position))
        
    def _set_priority(self, item_id: str, priority: int):
        """Set priority for an item."""
        if item_id in self._items:
            self._items[item_id].priority = priority
            self.context_changed.emit()
            
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        item_id = item.data(Qt.ItemDataRole.UserRole)
        self.item_clicked.emit(item_id)
        
    def has_item(self, item_id: str) -> bool:
        """Check if item is in context."""
        return item_id in self._items
        
    def get_item_count(self) -> int:
        """Get number of context items."""
        return len(self._items)
        
    def get_total_tokens(self) -> int:
        """Get estimated total tokens."""
        return self._estimated_tokens
