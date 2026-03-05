"""Context manager for Agent panel - manages files and context selection."""

import logging
from typing import Optional, List, Dict, Set
from dataclasses import dataclass, field
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QMenu, QAbstractItemView,
    QFrame, QSizePolicy, QToolTip, QTextEdit, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont, QCursor

from pyutagent.indexing.codebase_indexer import SymbolType
from pyutagent.ui.services.symbol_indexer import SymbolIndexer, SymbolIndexEntry

logger = logging.getLogger(__name__)


@dataclass
class ContextItem:
    """Represents a context item (file, folder, code snippet, or symbol)."""
    id: str
    name: str
    path: str
    item_type: str  # 'file', 'folder', 'snippet', 'current_file', 'selection', 'symbol'
    icon: str = "📄"
    token_count: int = 0
    priority: int = 0  # Higher = more important
    symbol_data: Optional[Dict] = None  # For symbol context items
    preview_text: str = ""  # Preview text for symbols
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, ContextItem):
            return self.id == other.id
        return False


class SymbolPreviewDialog(QDialog):
    """Dialog for previewing symbol definitions."""
    
    def __init__(self, symbol_name: str, file_path: str, start_line: int, 
                 end_line: int, signature: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Symbol Preview: {symbol_name}")
        self.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Signature label
        if signature:
            sig_label = QLabel(f"<code>{signature}</code>")
            sig_label.setStyleSheet("""
                background-color: #2D2D30;
                padding: 8px;
                border-radius: 4px;
                font-family: Consolas, monospace;
                font-size: 12px;
            """)
            layout.addWidget(sig_label)
        
        # Location label
        loc_label = QLabel(f"📍 {file_path}:{start_line}-{end_line}")
        loc_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(loc_label)
        
        # Code preview
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._text_edit)
        
        # Load code
        self._load_code(file_path, start_line, end_line)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _load_code(self, file_path: str, start_line: int, end_line: int):
        """Load code from file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Show context around symbol
                context_start = max(0, start_line - 6)
                context_end = min(len(lines), end_line + 5)
                
                code_lines = []
                for i in range(context_start, context_end):
                    line_num = i + 1
                    prefix = ">>> " if start_line <= line_num <= end_line else "    "
                    code_lines.append(f"{prefix}{line_num:4d}: {lines[i]}")
                
                self._text_edit.setText(''.join(code_lines))
        except Exception as e:
            self._text_edit.setText(f"Error loading file: {e}")


class ContextManager(QWidget):
    """Manages context for Agent conversations.
    
    Features:
    - Add/remove files from context
    - Add/remove symbols from context
    - Show token usage
    - Priority management
    - @mention support
    - Symbol preview on hover/click
    """
    
    context_changed = pyqtSignal()  # Emitted when context changes
    item_removed = pyqtSignal(str)  # item_id
    item_clicked = pyqtSignal(str)  # item_id
    symbol_selected = pyqtSignal(str)  # symbol_id
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._items: Dict[str, ContextItem] = {}
        self._max_context_items = 20
        self._estimated_tokens = 0
        self._max_tokens = 128000  # Default max context
        self._symbol_indexer: Optional[SymbolIndexer] = None
        
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
        self._list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._list_widget.setMaximumHeight(200)
        self._list_widget.setStyleSheet("""
            QListWidget {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:hover {
                background-color: #2A2D2E;
            }
        """)
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
        
        self._add_symbol_btn = QPushButton("🔷 Symbol")
        self._add_symbol_btn.setToolTip("Add symbol (@symbol)")
        self._add_symbol_btn.clicked.connect(self._show_symbol_selector)
        quick_add.addWidget(self._add_symbol_btn)
        
        quick_add.addStretch()
        
        layout.addLayout(quick_add)
        
        # Help text
        help_label = QLabel("💡 Use @ to mention files, folders, or symbols in chat")
        help_label.setStyleSheet("color: #999; font-size: 10px;")
        layout.addWidget(help_label)
        
        layout.addStretch()
        
    def set_symbol_indexer(self, indexer: SymbolIndexer):
        """Set the symbol indexer for symbol context support."""
        self._symbol_indexer = indexer
        
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
        
        logger.info(f"Added to context: {item.name} ({item.item_type})")
        return True
        
    def _add_list_item(self, item: ContextItem):
        """Add item to list widget."""
        display_text = f"{item.icon} {item.name}"
        if item.item_type == 'symbol' and item.preview_text:
            # Truncate preview for display
            preview = item.preview_text[:50] + "..." if len(item.preview_text) > 50 else item.preview_text
            display_text += f"\n    {preview}"
        
        list_item = QListWidgetItem(display_text)
        list_item.setData(Qt.ItemDataRole.UserRole, item.id)
        
        # Build tooltip
        tooltip = f"{item.path}\nTokens: ~{item.token_count}"
        if item.item_type == 'symbol' and item.symbol_data:
            tooltip += f"\nType: {item.symbol_data.get('symbol_type', 'Unknown')}"
        list_item.setToolTip(tooltip)
        
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
        
    def add_symbol(self, symbol: SymbolIndexEntry) -> bool:
        """Add a symbol to context.
        
        Args:
            symbol: Symbol index entry to add
            
        Returns:
            True if added
        """
        # Estimate token count for symbol (smaller than full file)
        try:
            with open(symbol.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                start = max(0, symbol.start_line - 1)
                end = min(len(lines), symbol.end_line)
                content = ''.join(lines[start:end])
                token_count = len(content) // 4
        except Exception:
            token_count = 50  # Default estimate
        
        # Get icon based on symbol type
        icon = self._get_symbol_icon(symbol.symbol_type)
        
        item = ContextItem(
            id=f"symbol:{symbol.id}",
            name=symbol.name,
            path=symbol.file_path,
            item_type="symbol",
            icon=icon,
            token_count=token_count,
            priority=2,  # Symbols are high priority
            symbol_data=symbol.to_dict(),
            preview_text=symbol.signature
        )
        
        return self.add_item(item)
        
    def _get_symbol_icon(self, symbol_type: SymbolType) -> str:
        """Get icon for symbol type."""
        icons = {
            SymbolType.CLASS: '🏛️',
            SymbolType.INTERFACE: '🔌',
            SymbolType.ENUM: '📋',
            SymbolType.METHOD: '⚡',
            SymbolType.CONSTRUCTOR: '🏗️',
            SymbolType.FIELD: '📦',
            SymbolType.ANNOTATION: '🏷️',
            SymbolType.PACKAGE: '📦',
            SymbolType.MODULE: '📁',
        }
        return icons.get(symbol_type, '🔷')
        
    def _show_symbol_selector(self):
        """Show symbol selector dialog."""
        if not self._symbol_indexer:
            logger.warning("Symbol indexer not available")
            return
            
        # Get recent symbols
        recent = self._symbol_indexer.get_recent_symbols(limit=20)
        
        if not recent:
            logger.info("No symbols available")
            return
            
        # TODO: Show a proper symbol picker dialog
        # For now, just add the most recent symbol
        if recent:
            self.add_symbol(recent[0])
            
    def get_context_items(self) -> List[ContextItem]:
        """Get all context items."""
        return list(self._items.values())
        
    def get_context_files(self) -> List[str]:
        """Get file paths in context."""
        return [
            item.path for item in self._items.values()
            if item.item_type == 'file'
        ]
        
    def get_context_symbols(self) -> List[Dict]:
        """Get symbol data in context."""
        return [
            item.symbol_data for item in self._items.values()
            if item.item_type == 'symbol' and item.symbol_data
        ]
        
    def _update_token_count(self):
        """Update token count display."""
        total = sum(item.token_count for item in self._items.values())
        self._estimated_tokens = total
        
        # Count items by type
        file_count = sum(1 for item in self._items.values() if item.item_type == 'file')
        symbol_count = sum(1 for item in self._items.values() if item.item_type == 'symbol')
        other_count = len(self._items) - file_count - symbol_count
        
        count_parts = []
        if file_count:
            count_parts.append(f"{file_count} files")
        if symbol_count:
            count_parts.append(f"{symbol_count} symbols")
        if other_count:
            count_parts.append(f"{other_count} other")
            
        count_text = ", ".join(count_parts) if count_parts else "0 items"
        self._token_label.setText(f"~{total:,} tokens ({count_text})")
        
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
        context_item = self._items.get(item_id)
        
        menu = QMenu(self)
        
        # Preview action for symbols
        if context_item and context_item.item_type == 'symbol' and context_item.symbol_data:
            preview_action = menu.addAction("👁️ Preview Symbol")
            preview_action.triggered.connect(lambda: self._preview_symbol(item_id))
            menu.addSeparator()
        
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
        
    def _preview_symbol(self, item_id: str):
        """Show symbol preview dialog."""
        item = self._items.get(item_id)
        if not item or not item.symbol_data:
            return
            
        data = item.symbol_data
        dialog = SymbolPreviewDialog(
            symbol_name=data.get('name', 'Unknown'),
            file_path=data.get('file_path', ''),
            start_line=data.get('start_line', 1),
            end_line=data.get('end_line', 1),
            signature=data.get('signature', ''),
            parent=self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Symbol was confirmed/selected
            self.symbol_selected.emit(item_id)
        
    def _set_priority(self, item_id: str, priority: int):
        """Set priority for an item."""
        if item_id in self._items:
            self._items[item_id].priority = priority
            self.context_changed.emit()
            
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        item_id = item.data(Qt.ItemDataRole.UserRole)
        self.item_clicked.emit(item_id)
        
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle item double click - show preview for symbols."""
        item_id = item.data(Qt.ItemDataRole.UserRole)
        context_item = self._items.get(item_id)
        
        if context_item and context_item.item_type == 'symbol':
            self._preview_symbol(item_id)
        else:
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
        
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of context for LLM prompts."""
        summary = {
            "total_items": len(self._items),
            "total_tokens": self._estimated_tokens,
            "files": [],
            "symbols": [],
            "snippets": [],
        }
        
        for item in self._items.values():
            if item.item_type == 'file':
                summary["files"].append({
                    "name": item.name,
                    "path": item.path,
                    "tokens": item.token_count
                })
            elif item.item_type == 'symbol' and item.symbol_data:
                summary["symbols"].append({
                    "name": item.name,
                    "type": item.symbol_data.get('symbol_type'),
                    "signature": item.symbol_data.get('signature'),
                    "path": item.path
                })
            elif item.item_type == 'snippet':
                summary["snippets"].append({
                    "name": item.name,
                    "tokens": item.token_count
                })
                
        return summary
