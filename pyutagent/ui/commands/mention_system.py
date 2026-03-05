"""Mention system for referencing files, symbols, and context in chat.

This module provides a Cursor-like @-mention system for precise code referencing:
- @file:path/to/File.java - Reference specific files
- @folder:path/to/folder - Reference entire folders
- @symbol:ClassName.methodName - Reference code symbols
- @code:symbol_name - Reference any code symbol

参考 Cursor 的 @-symbol 引用设计:
- https://www.cursor.com/blog/llm-chat-code-context
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any, Tuple, Set

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QLineEdit, QFrame, QAbstractItemView, QScrollArea,
    QApplication, QTextEdit, QStyledItemDelegate, QStyle
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QObject, QSize
from PyQt6.QtGui import QFont, QKeyEvent, QTextCursor, QTextCharFormat, QColor, QIcon, QPixmap

from pyutagent.indexing.codebase_indexer import CodebaseIndexer, CodeSymbol, SymbolType
from pyutagent.ui.services.symbol_indexer import SymbolIndexer, SymbolIndexEntry, Language

logger = logging.getLogger(__name__)


class MentionType(Enum):
    """Types of mentions supported."""
    FILE = "file"
    FOLDER = "folder"
    CODE = "code"
    SYMBOL = "symbol"
    CURRENT = "current"
    SELECTION = "selection"
    WORKSPACE = "workspace"


@dataclass
class MentionItem:
    """Represents a mentionable item."""
    id: str
    name: str
    mention_type: MentionType
    icon: str
    path: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    symbol: Optional[CodeSymbol] = None
    symbol_entry: Optional[SymbolIndexEntry] = None
    priority_score: float = 0.0
    match_score: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def display_text(self) -> str:
        """Get display text for the mention."""
        if self.mention_type == MentionType.FILE:
            return f"@{self.mention_type.value}:{self.path}"
        elif self.mention_type == MentionType.FOLDER:
            return f"@{self.mention_type.value}:{self.path}"
        elif self.mention_type in (MentionType.CODE, MentionType.SYMBOL):
            return f"@{self.mention_type.value}:{self.name}"
        else:
            return f"@{self.name}"

    @property
    def preview_text(self) -> str:
        """Get preview text for display."""
        if self.symbol_entry:
            return self.symbol_entry.signature or self.name
        if self.symbol:
            return self.symbol.signature or self.name
        return self.description or self.name

    @property
    def sort_key(self) -> Tuple[float, float, str]:
        """Get sort key for ordering items."""
        # Sort by: match score (desc), priority score (desc), name (asc)
        return (-self.match_score, -self.priority_score, self.name.lower())


class GroupedListItem(QListWidgetItem):
    """List item with grouping support."""
    
    def __init__(self, item: MentionItem, group: str):
        super().__init__()
        self.mention_item = item
        self.group = group
        
        # Set display text
        type_indicator = self._get_type_icon(item.mention_type)
        display_text = f"{type_indicator} {item.name}"
        if item.description:
            display_text += f"\n   <span style='color: #888; font-size: 10px;'>{item.description}</span>"
        
        self.setText(display_text)
        self.setData(Qt.ItemDataRole.UserRole, item)
        self.setFont(QFont("Segoe UI", 10))
        
    def _get_type_icon(self, mention_type: MentionType) -> str:
        """Get icon for mention type."""
        icons = {
            MentionType.FILE: "📄",
            MentionType.FOLDER: "📁",
            MentionType.CODE: "⚡",
            MentionType.SYMBOL: "🔷",
            MentionType.CURRENT: "📄",
            MentionType.SELECTION: "✂️",
            MentionType.WORKSPACE: "🗂️",
        }
        return icons.get(mention_type, "📎")


class MentionPopup(QFrame):
    """Popup widget for displaying mention suggestions with smart autocomplete."""

    item_selected = pyqtSignal(MentionItem)
    dismissed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._items: List[MentionItem] = []
        self._filtered_items: List[MentionItem] = []
        self._group_by_type: bool = True
        self._max_items: int = 50
        self._search_timer: Optional[QTimer] = None
        
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
        self.setMaximumHeight(450)
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header with count
        header_layout = QHBoxLayout()
        self._header = QLabel("Mention")
        self._header.setStyleSheet("color: #858585; font-size: 11px; padding-bottom: 4px;")
        header_layout.addWidget(self._header)
        
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #666; font-size: 10px;")
        header_layout.addWidget(self._count_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)

        # Search within popup
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Filter symbols...")
        self._search_box.setStyleSheet("""
            QLineEdit {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 4px 8px;
                border-radius: 3px;
            }
        """)
        self._search_box.textChanged.connect(self._on_search_delayed)
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
        self._update_count_label()
        self._update_list()

    def _update_count_label(self):
        """Update the count label."""
        total = len(self._items)
        showing = len(self._filtered_items)
        if showing < total:
            self._count_label.setText(f"({showing}/{total})")
        else:
            self._count_label.setText(f"({total})")

    def _on_search_delayed(self, text: str):
        """Handle search with debouncing."""
        if self._search_timer is None:
            self._search_timer = QTimer(self)
            self._search_timer.setSingleShot(True)
            self._search_timer.timeout.connect(lambda: self._on_search(text))
        self._search_timer.start(50)  # 50ms debounce

    def _on_search(self, text: str):
        """Filter items based on search with fuzzy matching."""
        start_time = time.time()
        
        if not text:
            self._filtered_items = self._items
        else:
            text_lower = text.lower()
            scored_items = []
            
            for item in self._items:
                score = self._calculate_fuzzy_score(item, text_lower)
                if score > 0:
                    item.match_score = score
                    scored_items.append(item)
            
            # Sort by score and priority
            scored_items.sort(key=lambda x: x.sort_key)
            self._filtered_items = scored_items
        
        self._update_count_label()
        self._update_list()
        
        elapsed = (time.time() - start_time) * 1000
        if elapsed > 200:
            logger.warning(f"[MentionPopup] Search took {elapsed:.1f}ms")

    def _calculate_fuzzy_score(self, item: MentionItem, query: str) -> float:
        """Calculate fuzzy match score."""
        score = 0.0
        
        # Name matching
        name_lower = item.name.lower()
        if query == name_lower:
            score += 100.0  # Exact match
        elif query in name_lower:
            score += 50.0  # Substring match
        elif self._fuzzy_match(query, name_lower):
            score += 20.0  # Fuzzy match
        
        # Path matching
        if item.path and query in item.path.lower():
            score += 10.0
        
        # Description matching
        if item.description and query in item.description.lower():
            score += 5.0
        
        # Boost recent/priority items
        score += item.priority_score * 10.0
        
        return score

    def _fuzzy_match(self, query: str, target: str) -> bool:
        """Check if query fuzzy matches target."""
        query_idx = 0
        target_idx = 0
        
        while query_idx < len(query) and target_idx < len(target):
            if query[query_idx] == target[target_idx]:
                query_idx += 1
            target_idx += 1
        
        return query_idx == len(query)

    def _update_list(self):
        """Update the item list display with grouping."""
        self._list.clear()
        
        if self._group_by_type:
            self._update_list_grouped()
        else:
            self._update_list_flat()
        
        # Select first item
        if self._filtered_items:
            self._list.setCurrentRow(0)

    def _update_list_grouped(self):
        """Update list with type grouping."""
        # Group items by type
        groups: Dict[str, List[MentionItem]] = {
            "Recent": [],
            "Classes": [],
            "Methods": [],
            "Functions": [],
            "Files": [],
            "Folders": [],
            "Other": [],
        }
        
        for item in self._filtered_items[:self._max_items]:
            if item.priority_score > 0.5:
                groups["Recent"].append(item)
            elif item.mention_type == MentionType.FILE:
                groups["Files"].append(item)
            elif item.mention_type == MentionType.FOLDER:
                groups["Folders"].append(item)
            elif item.symbol_entry:
                if item.symbol_entry.symbol_type == SymbolType.CLASS:
                    groups["Classes"].append(item)
                elif item.symbol_entry.symbol_type in (SymbolType.METHOD, SymbolType.CONSTRUCTOR):
                    groups["Methods"].append(item)
                else:
                    groups["Other"].append(item)
            elif item.symbol:
                if item.symbol.symbol_type == SymbolType.CLASS:
                    groups["Classes"].append(item)
                elif item.symbol.symbol_type in (SymbolType.METHOD, SymbolType.CONSTRUCTOR):
                    groups["Methods"].append(item)
                else:
                    groups["Other"].append(item)
            else:
                groups["Other"].append(item)
        
        # Add items by group
        for group_name, items in groups.items():
            if not items:
                continue
            
            # Add group header
            header = QListWidgetItem(f"  {group_name}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setBackground(QColor("#2D2D30"))
            header.setForeground(QColor("#808080"))
            header.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            self._list.addItem(header)
            
            # Add items in group
            for item in items:
                list_item = GroupedListItem(item, group_name)
                self._list.addItem(list_item)

    def _update_list_flat(self):
        """Update list without grouping."""
        for item in self._filtered_items[:self._max_items]:
            list_item = GroupedListItem(item, "")
            self._list.addItem(list_item)

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        # Skip headers
        if item.flags() == Qt.ItemFlag.NoItemFlags:
            return
            
        mention_item = item.data(Qt.ItemDataRole.UserRole)
        if mention_item:
            self.item_selected.emit(mention_item)
            self.hide()

    def select_next(self):
        """Select next item, skipping headers."""
        current = self._list.currentRow()
        for i in range(current + 1, self._list.count()):
            item = self._list.item(i)
            if item.flags() != Qt.ItemFlag.NoItemFlags:
                self._list.setCurrentRow(i)
                return

    def select_previous(self):
        """Select previous item, skipping headers."""
        current = self._list.currentRow()
        for i in range(current - 1, -1, -1):
            item = self._list.item(i)
            if item.flags() != Qt.ItemFlag.NoItemFlags:
                self._list.setCurrentRow(i)
                return

    def get_selected_item(self) -> Optional[MentionItem]:
        """Get currently selected item."""
        item = self._list.currentItem()
        if item and item.flags() != Qt.ItemFlag.NoItemFlags:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    def has_items(self) -> bool:
        """Check if there are items to display."""
        return len(self._filtered_items) > 0

    def set_group_by_type(self, enabled: bool):
        """Enable/disable type grouping."""
        self._group_by_type = enabled
        self._update_list()

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Escape:
            self.dismissed.emit()
            self.hide()
        elif event.key() in [Qt.Key.Key_Return, Qt.Key.Key_Enter]:
            item = self.get_selected_item()
            if item:
                self.item_selected.emit(item)
                self.hide()
        elif event.key() == Qt.Key.Key_Down:
            self.select_next()
        elif event.key() == Qt.Key.Key_Up:
            self.select_previous()
        elif event.key() == Qt.Key.Key_Tab:
            # Toggle grouping
            self.set_group_by_type(not self._group_by_type)
        else:
            super().keyPressEvent(event)


class MentionHighlighter(QObject):
    """Highlights @-mentions in text editors."""

    def __init__(self, text_edit: QTextEdit):
        super().__init__()
        self._text_edit = text_edit
        self._mention_format = QTextCharFormat()
        self._mention_format.setBackground(QColor("#094771"))
        self._mention_format.setForeground(QColor("#FFFFFF"))

    def highlight_mentions(self):
        """Highlight all @-mentions in the document."""
        cursor = self._text_edit.textCursor()
        document = self._text_edit.document()

        # Clear existing formatting
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.setCharFormat(QTextCharFormat())
        cursor.clearSelection()

        # Find and highlight mentions
        text = document.toPlainText()
        mention_pattern = r'@\w+(?::[^\s@]*)?'

        for match in re.finditer(mention_pattern, text):
            start = match.start()
            end = match.end()

            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(self._mention_format)


class EnhancedMentionSystem:
    """Enhanced system for handling @mentions with codebase indexing support.

    Features:
    - @file:path/to/File.java - Reference specific files
    - @folder:path/to/folder - Reference entire folders
    - @symbol:ClassName.methodName - Reference code symbols
    - @code:symbol_name - Reference any code symbol
    - @current, @selection, @workspace - Special references
    - Smart autocomplete with fuzzy matching
    - Type grouping display
    - Recent usage priority sorting
    """

    mention_triggered = pyqtSignal(str, str, dict)  # mention_type, mention_id, metadata

    def __init__(self):
        self._popup: Optional[MentionPopup] = None
        self._current_input: Optional[QLineEdit] = None
        self._is_active = False
        self._mention_start_pos = -1
        self._project_path: Optional[str] = None
        self._codebase_indexer: Optional[CodebaseIndexer] = None
        self._symbol_indexer: Optional[SymbolIndexer] = None
        self._highlighter: Optional[MentionHighlighter] = None
        self._search_cache: Dict[str, List[MentionItem]] = {}
        self._cache_timeout = 30  # seconds

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

    def attach_to_text_edit(self, text_edit: QTextEdit, parent_widget: QWidget):
        """Attach mention system to a text edit widget.

        Args:
            text_edit: The text edit widget
            parent_widget: Parent for the popup
        """
        self._text_edit = text_edit
        self._highlighter = MentionHighlighter(text_edit)

        # Create popup
        self._popup = MentionPopup(parent_widget)
        self._popup.item_selected.connect(self._on_item_selected_text_edit)
        self._popup.dismissed.connect(self._on_popup_dismissed)

        # Connect signals
        text_edit.textChanged.connect(self._on_text_edit_changed)

    def set_project_path(self, path: str):
        """Set the current project path for file discovery."""
        self._project_path = path
        
        # Initialize symbol indexer
        if path:
            self._symbol_indexer = SymbolIndexer(path)

    def set_codebase_indexer(self, indexer: CodebaseIndexer):
        """Set the codebase indexer for symbol resolution."""
        self._codebase_indexer = indexer

    def set_symbol_indexer(self, indexer: SymbolIndexer):
        """Set the symbol indexer for enhanced symbol search."""
        self._symbol_indexer = indexer

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

    def _on_text_edit_changed(self):
        """Handle text edit change."""
        if not hasattr(self, '_text_edit') or not self._popup:
            return

        text = self._text_edit.toPlainText()
        cursor = self._text_edit.textCursor()
        cursor_pos = cursor.position()
        text_before_cursor = text[:cursor_pos]

        # Highlight mentions
        if self._highlighter:
            self._highlighter.highlight_mentions()

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
        start_time = time.time()
        
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
            elif hasattr(self, '_text_edit'):
                cursor = self._text_edit.textCursor()
                rect = self._text_edit.cursorRect(cursor)
                pos = self._text_edit.mapToGlobal(rect.bottomLeft())
                self._popup.move(pos)
                self._popup.show()
        else:
            self._popup.hide()
        
        elapsed = (time.time() - start_time) * 1000
        if elapsed > 200:
            logger.warning(f"[EnhancedMentionSystem] Suggestion generation took {elapsed:.1f}ms")

    def _get_suggestion_title(self, query: str) -> str:
        """Get title for suggestion popup based on query."""
        if not query:
            return "Mention (@file, @folder, @symbol)"

        # Check for type prefix
        if query.startswith("file:") or query.startswith("f:"):
            return "Files"
        elif query.startswith("folder:") or query.startswith("fol:"):
            return "Folders"
        elif query.startswith("symbol:") or query.startswith("s:"):
            return "Symbols"
        elif query.startswith("code:") or query.startswith("c:"):
            return "Code Symbols"

        # Check first character
        first_char = query[0].lower() if query else ''

        if first_char == 'f':
            return "Files & Folders"
        elif first_char == 's':
            return "Symbols"
        elif first_char == 'c':
            return "Classes & Methods"
        else:
            return "Mention"

    def _get_mention_items(self, query: str) -> List[MentionItem]:
        """Get mention items based on query."""
        items = []

        # Parse query for type prefix
        mention_type = None
        search_query = query

        if ':' in query:
            prefix, search_query = query.split(':', 1)
            prefix_lower = prefix.lower()
            if prefix_lower in ['file', 'f']:
                mention_type = MentionType.FILE
            elif prefix_lower in ['folder', 'fol', 'dir']:
                mention_type = MentionType.FOLDER
            elif prefix_lower in ['symbol', 's']:
                mention_type = MentionType.SYMBOL
            elif prefix_lower in ['code', 'c']:
                mention_type = MentionType.CODE

        # Always add special mentions if no specific type
        if not mention_type or mention_type in [MentionType.CURRENT, MentionType.SELECTION, MentionType.WORKSPACE]:
            items.extend([
                MentionItem(
                    id="current",
                    name="current",
                    mention_type=MentionType.CURRENT,
                    icon="📄",
                    description="Current file"
                ),
                MentionItem(
                    id="selection",
                    name="selection",
                    mention_type=MentionType.SELECTION,
                    icon="✂️",
                    description="Selected code"
                ),
                MentionItem(
                    id="workspace",
                    name="workspace",
                    mention_type=MentionType.WORKSPACE,
                    icon="📁",
                    description="Entire workspace"
                ),
            ])

        # Add files from project
        if not mention_type or mention_type == MentionType.FILE:
            files = self._get_project_files(search_query)
            for file_path in files[:20]:  # Limit to 20 files
                path = Path(file_path)
                icon = self._get_file_icon(path.suffix)
                rel_path = str(path.relative_to(self._project_path)) if self._project_path else file_path
                items.append(MentionItem(
                    id=file_path,
                    name=path.name,
                    mention_type=MentionType.FILE,
                    icon=icon,
                    path=rel_path,
                    description=str(path.parent)
                ))

        # Add folders
        if not mention_type or mention_type == MentionType.FOLDER:
            folders = self._get_project_folders(search_query)
            for folder_path in folders[:10]:  # Limit to 10 folders
                path = Path(folder_path)
                rel_path = str(path.relative_to(self._project_path)) if self._project_path else folder_path
                items.append(MentionItem(
                    id=folder_path,
                    name=path.name,
                    mention_type=MentionType.FOLDER,
                    icon="📁",
                    path=rel_path,
                    description=f"Folder with {self._count_files_in_folder(folder_path)} files"
                ))

        # Add symbols from symbol indexer (new enhanced symbol search)
        if self._symbol_indexer and (not mention_type or mention_type in [MentionType.SYMBOL, MentionType.CODE]):
            try:
                symbol_type = None
                if mention_type == MentionType.CODE:
                    # Filter to classes and methods only
                    pass  # We'll filter after search
                
                symbols = self._symbol_indexer.search(
                    search_query, 
                    symbol_type=symbol_type,
                    limit=20,
                    fuzzy=True
                )
                
                for symbol in symbols:
                    # Filter for @code to only show classes and methods
                    if mention_type == MentionType.CODE and symbol.symbol_type not in (
                        SymbolType.CLASS, SymbolType.METHOD, SymbolType.CONSTRUCTOR, SymbolType.INTERFACE
                    ):
                        continue
                    
                    icon = self._get_symbol_icon(symbol.symbol_type)
                    description = symbol.signature or f"{symbol.symbol_type.value} in {Path(symbol.file_path).name}"
                    
                    items.append(MentionItem(
                        id=symbol.id,
                        name=symbol.name,
                        mention_type=MentionType.CODE if mention_type == MentionType.CODE else MentionType.SYMBOL,
                        icon=icon,
                        path=symbol.file_path,
                        description=description,
                        symbol_entry=symbol,
                        priority_score=symbol.priority_score
                    ))
            except Exception as e:
                logger.warning(f"[EnhancedMentionSystem] Symbol search failed: {e}")

        # Add symbols from codebase indexer (fallback)
        if self._codebase_indexer and (not mention_type or mention_type in [MentionType.CODE, MentionType.SYMBOL]):
            try:
                symbols = self._codebase_indexer.search_symbols(search_query, limit=15)
                for symbol in symbols:
                    # Skip if already added from symbol indexer
                    if any(item.id == symbol.id for item in items):
                        continue
                    
                    icon = self._get_symbol_icon(symbol.symbol_type)
                    description = symbol.signature or f"{symbol.symbol_type.value} in {Path(symbol.file_path).name}"
                    items.append(MentionItem(
                        id=symbol.id,
                        name=symbol.name,
                        mention_type=MentionType.CODE if mention_type == MentionType.CODE else MentionType.SYMBOL,
                        icon=icon,
                        path=symbol.file_path,
                        description=description,
                        symbol=symbol
                    ))
            except Exception as e:
                logger.warning(f"[EnhancedMentionSystem] Codebase indexer search failed: {e}")

        # Filter by query if provided (for non-prefixed queries)
        if query and ':' not in query:
            query_lower = query.lower()
            filtered_items = []
            for item in items:
                score = self._calculate_item_score(item, query_lower)
                if score > 0:
                    item.match_score = score
                    filtered_items.append(item)
            items = filtered_items
            
            # Sort by score and priority
            items.sort(key=lambda x: x.sort_key)

        return items

    def _calculate_item_score(self, item: MentionItem, query: str) -> float:
        """Calculate match score for an item."""
        score = 0.0
        
        # Name match
        name_lower = item.name.lower()
        if query == name_lower:
            score += 100.0
        elif query in name_lower:
            score += 50.0
        elif self._fuzzy_match(query, name_lower):
            score += 20.0
        
        # Path match
        if item.path and query in item.path.lower():
            score += 10.0
        
        # Description match
        if item.description and query in item.description.lower():
            score += 5.0
        
        # Boost priority items
        score += item.priority_score * 10.0
        
        return score

    def _fuzzy_match(self, query: str, target: str) -> bool:
        """Check if query fuzzy matches target."""
        query_idx = 0
        target_idx = 0
        
        while query_idx < len(query) and target_idx < len(target):
            if query[query_idx] == target[target_idx]:
                query_idx += 1
            target_idx += 1
        
        return query_idx == len(query)

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
                    if any(ignore in str(file_path) for ignore in ['node_modules', '__pycache__', 'target', 'build', '.git']):
                        continue

                    # Check if matches query
                    if not query or query.lower() in file_path.name.lower():
                        files.append(str(file_path))

                    if len(files) >= 50:  # Limit search
                        break

        except Exception as e:
            logger.warning(f"Failed to get project files: {e}")

        return files

    def _get_project_folders(self, query: str) -> List[str]:
        """Get folders from project matching query."""
        if not self._project_path:
            return []

        folders = []
        project_path = Path(self._project_path)

        try:
            for folder_path in project_path.rglob('*'):
                if folder_path.is_dir():
                    # Skip hidden and common ignore patterns
                    if any(part.startswith('.') for part in folder_path.parts):
                        continue
                    if any(ignore in str(folder_path) for ignore in ['node_modules', '__pycache__', 'target', 'build', '.git']):
                        continue

                    # Check if matches query
                    if not query or query.lower() in folder_path.name.lower():
                        folders.append(str(folder_path))

                    if len(folders) >= 30:
                        break

        except Exception as e:
            logger.warning(f"Failed to get project folders: {e}")

        return folders

    def _count_files_in_folder(self, folder_path: str) -> int:
        """Count files in a folder."""
        try:
            path = Path(folder_path)
            return len([f for f in path.rglob('*') if f.is_file()])
        except Exception:
            return 0

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

    def _on_item_selected(self, item: MentionItem):
        """Handle item selection for line edit."""
        if self._current_input and self._mention_start_pos >= 0:
            text = self._current_input.text()
            cursor_pos = self._current_input.cursorPosition()

            # Replace the mention text with the selected item
            before_mention = text[:self._mention_start_pos]
            after_mention = text[cursor_pos:]

            new_text = f"{before_mention}{item.display_text}{after_mention}"
            self._current_input.setText(new_text)

            # Move cursor after the mention
            new_cursor_pos = len(before_mention) + len(item.display_text)
            self._current_input.setCursorPosition(new_cursor_pos)

            # Emit signal
            metadata = {
                "path": item.path,
                "symbol": item.symbol.to_dict() if item.symbol else None,
                "symbol_entry": item.symbol_entry.to_dict() if item.symbol_entry else None,
            }
            self.mention_triggered.emit(item.mention_type.value, item.id, metadata)

        self._is_active = False
        self._mention_start_pos = -1

    def _on_item_selected_text_edit(self, item: MentionItem):
        """Handle item selection for text edit."""
        if hasattr(self, '_text_edit') and self._mention_start_pos >= 0:
            cursor = self._text_edit.textCursor()
            text = self._text_edit.toPlainText()

            # Replace the mention text with the selected item
            before_mention = text[:self._mention_start_pos]
            after_mention = text[cursor.position():]

            new_text = f"{before_mention}{item.display_text}{after_mention}"
            self._text_edit.setPlainText(new_text)

            # Move cursor after the mention
            new_cursor_pos = len(before_mention) + len(item.display_text)
            cursor.setPosition(new_cursor_pos)
            self._text_edit.setTextCursor(cursor)

            # Re-highlight
            if self._highlighter:
                self._highlighter.highlight_mentions()

            # Emit signal
            metadata = {
                "path": item.path,
                "symbol": item.symbol.to_dict() if item.symbol else None,
                "symbol_entry": item.symbol_entry.to_dict() if item.symbol_entry else None,
            }
            self.mention_triggered.emit(item.mention_type.value, item.id, metadata)

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
                          Qt.Key.Key_Escape, Qt.Key.Key_Tab]:
            self._popup.keyPressEvent(event)
            return True

        return False

    def parse_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Parse mentions from text.

        Supports formats:
        - @file:path/to/file.java
        - @folder:path/to/folder
        - @symbol:ClassName.methodName
        - @code:symbol_name
        - @current, @selection, @workspace

        Returns:
            List of dicts with 'type', 'id', 'text', and 'resolved' keys
        """
        mentions = []

        # Find @mentions with optional type prefix
        pattern = r'@(\w+)(?::([^\s@]+))?'
        for match in re.finditer(pattern, text):
            full_match = match.group(0)
            prefix = match.group(1)
            value = match.group(2) if match.group(2) else ""

            mention_data = {
                'text': full_match,
                'raw': match.group(0)[1:],  # Without @
            }

            # Determine type based on prefix or value
            if prefix in ['file', 'folder', 'code', 'symbol']:
                mention_data['type'] = prefix
                mention_data['id'] = value
            elif prefix in ['current', 'selection', 'workspace']:
                mention_data['type'] = prefix
                mention_data['id'] = prefix
            else:
                # Try to guess type
                if '.' in prefix and not value:
                    mention_data['type'] = 'code'
                    mention_data['id'] = prefix
                else:
                    mention_data['type'] = 'symbol'
                    mention_data['id'] = prefix

            # Try to resolve the mention if indexer is available
            if self._symbol_indexer:
                resolved = self._resolve_symbol_mention(mention_data['raw'])
                mention_data['resolved'] = resolved
            elif self._codebase_indexer:
                resolved = self._codebase_indexer.resolve_reference(mention_data['raw'])
                mention_data['resolved'] = resolved

            mentions.append(mention_data)

        return mentions

    def _resolve_symbol_mention(self, ref: str) -> Optional[Dict[str, Any]]:
        """Resolve a symbol mention using the symbol indexer."""
        if not self._symbol_indexer:
            return None

        # Parse reference
        if ':' in ref:
            ref_type, ref_value = ref.split(':', 1)
        else:
            ref_type = 'symbol'
            ref_value = ref

        if ref_type in ('symbol', 'code'):
            # Search for symbol
            symbols = self._symbol_indexer.search(ref_value, limit=1, fuzzy=False)
            if symbols:
                symbol = symbols[0]
                return {
                    'type': 'symbol',
                    'symbol': symbol.to_dict(),
                    'file_path': symbol.file_path,
                }

        return None

    def resolve_mention(self, mention_text: str) -> Optional[Dict[str, Any]]:
        """Resolve a single mention to its target.

        Args:
            mention_text: The mention text (without @ prefix)

        Returns:
            Resolved reference info or None
        """
        # Try symbol indexer first
        if self._symbol_indexer:
            resolved = self._resolve_symbol_mention(mention_text)
            if resolved:
                return resolved

        # Fall back to codebase indexer
        if self._codebase_indexer:
            return self._codebase_indexer.resolve_reference(mention_text)

        return None

    def get_mention_context(self, mention_text: str) -> str:
        """Get context for a mention to include in LLM prompts.

        Args:
            mention_text: The mention text (without @ prefix)

        Returns:
            Context string for the mention
        """
        resolved = self.resolve_mention(mention_text)

        if not resolved:
            return f"[Could not resolve: @{mention_text}]"

        context_parts = []

        if resolved['type'] == 'file':
            file_path = resolved['path']
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                context_parts.append(f"File: {file_path}")
                context_parts.append("```")
                context_parts.append(content)
                context_parts.append("```")
            except Exception as e:
                context_parts.append(f"[Error reading file {file_path}: {e}]")

        elif resolved['type'] == 'folder':
            folder_path = resolved['path']
            context_parts.append(f"Folder: {folder_path}")
            context_parts.append(f"Contains {resolved.get('file_count', 0)} files")

        elif resolved['type'] == 'symbol':
            # Handle symbol from symbol indexer
            if 'symbol' in resolved:
                symbol_data = resolved['symbol']
                context_parts.append(f"Symbol: {symbol_data.get('name', 'Unknown')}")
                context_parts.append(f"Type: {symbol_data.get('symbol_type', 'Unknown')}")
                context_parts.append(f"Location: {symbol_data.get('file_path', 'Unknown')}:{symbol_data.get('start_line', 0)}")
                if symbol_data.get('signature'):
                    context_parts.append(f"Signature: {symbol_data['signature']}")

                # Get file content around the symbol
                file_path = symbol_data.get('file_path')
                start_line = symbol_data.get('start_line', 1)
                end_line = symbol_data.get('end_line', start_line)
                
                if file_path:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            start = max(0, start_line - 5)
                            end = min(len(lines), end_line + 5)
                            context = ''.join(lines[start:end])
                            context_parts.append("```")
                            context_parts.append(context)
                            context_parts.append("```")
                    except Exception:
                        pass

        return "\n".join(context_parts)

    def is_active(self) -> bool:
        """Check if mention system is active."""
        return self._is_active

    def refresh_symbol_index(self) -> None:
        """Refresh the symbol index."""
        if self._symbol_indexer:
            self._symbol_indexer.refresh()

    def get_symbol_stats(self) -> Dict[str, Any]:
        """Get symbol indexing statistics."""
        if self._symbol_indexer:
            stats = self._symbol_indexer.get_stats()
            return stats.to_dict()
        return {}


# Backward compatibility - keep the old class name
MentionSystem = EnhancedMentionSystem
