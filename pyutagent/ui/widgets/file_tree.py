"""Enhanced file tree widget with multi-language support."""

import logging
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QLabel, QMenu, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont

from ..language.file_icons import FileIconProvider
from ..language.language_support import LanguageSupport

logger = logging.getLogger(__name__)


class FileTree(QWidget):
    """Enhanced file tree widget with search and multi-language support.
    
    Features:
    - Multi-language file detection and icons
    - Real-time search filtering
    - Context menu with language-specific actions
    - File selection signals
    """
    
    # Signals
    file_selected = pyqtSignal(str)  # file_path
    folder_selected = pyqtSignal(str)  # folder_path
    file_activated = pyqtSignal(str)  # file_path (double-click)
    context_menu_requested = pyqtSignal(str, object)  # file_path, menu
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._project_path: str = ""
        self._all_items: List[tuple] = []  # (name_lower, item, path)
        self._icon_provider = FileIconProvider()
        self._language_support = LanguageSupport()
        self._file_filter: Optional[Callable[[str], bool]] = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the file tree UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("📁 Project Files")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        # Search box
        search_layout = QHBoxLayout()
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("🔍 Search files... (Ctrl+F)")
        self._search_box.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self._search_box)
        
        # Clear button
        self._clear_btn = QLabel("✕")
        self._clear_btn.setStyleSheet("color: #999; cursor: pointer; padding: 0 5px;")
        self._clear_btn.mousePressEvent = lambda e: self._search_box.clear()
        search_layout.addWidget(self._clear_btn)
        
        layout.addLayout(search_layout)
        
        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.setAnimated(True)
        self._tree.setIndentation(16)
        layout.addWidget(self._tree)
        
        # Stats label
        self._stats_label = QLabel("No project loaded")
        self._stats_label.setStyleSheet("color: #757575; font-size: 11px; padding: 5px;")
        layout.addWidget(self._stats_label)
        
        # Debounce timer for search
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._apply_search)
        
    def load_project(self, project_path: str):
        """Load project structure into tree.
        
        Args:
            project_path: Path to the project directory
        """
        self._tree.clear()
        self._all_items.clear()
        self._project_path = project_path
        
        project_name = Path(project_path).name
        root = QTreeWidgetItem(self._tree, [f"📦 {project_name}"])
        root.setData(0, Qt.ItemDataRole.UserRole, project_path)
        root.setIcon(0, self._icon_provider.get_folder_icon("project"))
        root.setExpanded(True)
        
        # Detect project type and get source directories
        project_info = self._language_support.detect_project(project_path)
        src_dirs = project_info.get('source_dirs', [])
        
        if not src_dirs:
            # Fallback: load root directory
            src_dirs = [Path(project_path)]
        
        file_count = 0
        for src_dir in src_dirs:
            if src_dir.exists():
                file_count += self._add_directory(src_dir, root)
        
        # Update stats
        lang_name = project_info.get('language', 'Unknown')
        self._stats_label.setText(f"📊 {file_count} files | {lang_name}")
        
        logger.info(f"Loaded project: {project_path} with {file_count} files")
        
    def _add_directory(self, dir_path: Path, parent_item: QTreeWidgetItem) -> int:
        """Recursively add directory contents.
        
        Args:
            dir_path: Directory path
            parent_item: Parent tree item
            
        Returns:
            Number of files added
        """
        file_count = 0
        
        try:
            # Sort: directories first, then files
            items = sorted(dir_path.iterdir(), 
                          key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for item in items:
                # Skip hidden files and common ignore patterns
                if item.name.startswith('.') or item.name in ['node_modules', '__pycache__', 'target', 'build', 'dist']:
                    continue
                    
                if item.is_dir():
                    tree_item = QTreeWidgetItem(parent_item, [item.name])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item))
                    tree_item.setIcon(0, self._icon_provider.get_folder_icon(item.name))
                    file_count += self._add_directory(item, tree_item)
                else:
                    # Check if file should be shown
                    if self._file_filter and not self._file_filter(str(item)):
                        continue
                        
                    tree_item = QTreeWidgetItem(parent_item, [item.name])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item))
                    tree_item.setIcon(0, self._icon_provider.get_file_icon(item.suffix))
                    self._all_items.append((item.name.lower(), tree_item, str(item)))
                    file_count += 1
                    
        except PermissionError:
            logger.warning(f"Permission denied: {dir_path}")
        except Exception as e:
            logger.exception(f"Failed to add directory: {dir_path}")
        
        return file_count
        
    def set_file_filter(self, filter_func: Optional[Callable[[str], bool]]):
        """Set a filter function for files.
        
        Args:
            filter_func: Function that takes file path and returns True to include
        """
        self._file_filter = filter_func
        
    def _on_search_changed(self, text: str):
        """Handle search text change with debounce."""
        self._search_timer.stop()
        self._search_timer.start(150)  # 150ms debounce
        
    def _apply_search(self):
        """Apply search filter."""
        text = self._search_box.text().lower().strip()
        
        if not text:
            self._set_all_items_visible(True)
            return
        
        # Filter items
        for name, item, path in self._all_items:
            matches = text in name
            item.setHidden(not matches)
            
            # Expand parent if match found
            if matches:
                parent = item.parent()
                while parent:
                    parent.setExpanded(True)
                    parent.setHidden(False)
                    parent = parent.parent()
                    
        # Hide folders that have no visible children
        self._prune_empty_folders(self._tree.invisibleRootItem())
        
    def _prune_empty_folders(self, parent_item: QTreeWidgetItem):
        """Hide folders that have no visible children."""
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if child.childCount() > 0:
                self._prune_empty_folders(child)
                # Check if all children are hidden
                has_visible = any(not child.child(j).isHidden() 
                                 for j in range(child.childCount()))
                child.setHidden(not has_visible)
                
    def _set_all_items_visible(self, visible: bool):
        """Set visibility of all items."""
        for _, item, _ in self._all_items:
            item.setHidden(not visible)
        
        # Also reset folder visibility
        root = self._tree.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setHidden(False)
            
    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click."""
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
            
        if Path(path).is_file():
            self.file_selected.emit(path)
        else:
            self.folder_selected.emit(path)
            
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item double-click."""
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and Path(path).is_file():
            self.file_activated.emit(path)
            
    def _on_context_menu(self, position):
        """Show context menu."""
        item = self._tree.itemAt(position)
        if not item:
            return
            
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path:
            return
            
        menu = QMenu(self)
        
        # Add common actions
        if Path(path).is_file():
            menu.addAction("📋 Copy Path", lambda: self._copy_to_clipboard(path))
            menu.addSeparator()
            
        # Emit signal for custom actions
        self.context_menu_requested.emit(path, menu)
        
        if not menu.isEmpty():
            menu.exec(self._tree.viewport().mapToGlobal(position))
            
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
    def get_selected_path(self) -> Optional[str]:
        """Get the currently selected file/folder path."""
        item = self._tree.currentItem()
        if item:
            return item.data(0, Qt.ItemDataRole.UserRole)
        return None
        
    def select_path(self, path: str):
        """Select an item by path."""
        for name, item, item_path in self._all_items:
            if item_path == path:
                self._tree.setCurrentItem(item)
                break
                
    def expand_all(self):
        """Expand all items."""
        self._tree.expandAll()
        
    def collapse_all(self):
        """Collapse all items."""
        self._tree.collapseAll()
        
    def refresh(self):
        """Refresh the tree."""
        if self._project_path:
            self.load_project(self._project_path)
            
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._search_box.setFocus()
            self._search_box.selectAll()
        elif event.key() == Qt.Key.Key_Escape:
            self._search_box.clear()
        else:
            super().keyPressEvent(event)
