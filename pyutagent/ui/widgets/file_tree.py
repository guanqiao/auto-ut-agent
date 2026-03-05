"""Enhanced file tree widget with multi-language support, Git status, and drag-drop."""

import logging
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QLabel, QMenu, QAbstractItemView, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QMimeData, QByteArray
from PyQt6.QtGui import QIcon, QFont, QColor, QDrag, QKeyEvent

from ..language.file_icons import FileIconProvider
from ..language.language_support import LanguageSupport
from ..services.git_status_service import GitStatusService, GitStatus

logger = logging.getLogger(__name__)


class FileTreeItem(QTreeWidgetItem):
    """Custom tree item with Git status support."""
    
    def __init__(self, parent=None, texts=None):
        super().__init__(parent, texts or [])
        self._git_status: Optional[GitStatus] = None
        self._original_text = texts[0] if texts else ""
        self._is_highlighted = False
        
    def set_git_status(self, status: Optional[GitStatus]):
        """Set Git status for this item."""
        self._git_status = status
        self._update_display()
        
    def get_git_status(self) -> Optional[GitStatus]:
        """Get Git status."""
        return self._git_status
        
    def set_highlighted(self, highlighted: bool):
        """Set highlight state for search matches."""
        self._is_highlighted = highlighted
        self._update_display()
        
    def _update_display(self):
        """Update item display based on status."""
        text = self._original_text
        
        # Add Git status indicator
        if self._git_status and self._git_status != GitStatus.UNMODIFIED:
            status_icon = self._git_status.icon
            text = f"{text} [{status_icon}]"
            
        self.setText(0, text)
        
        # Set foreground color
        if self._git_status and self._git_status != GitStatus.UNMODIFIED:
            color = QColor(self._git_status.color)
            self.setForeground(0, color)
        elif self._is_highlighted:
            self.setForeground(0, QColor("#4EC9B0"))
        else:
            self.setForeground(0, QColor("#CCCCCC"))
            
    def setText(self, column: int, text: str):
        """Override to track original text."""
        if column == 0 and not text.endswith("]"):
            self._original_text = text
        super().setText(column, text)


class FileTree(QWidget):
    """Enhanced file tree widget with search, Git status, and drag-drop support.
    
    Features:
    - Multi-language file detection and icons
    - Real-time search filtering with fuzzy matching
    - Git status display (modified/added/deleted)
    - Drag and drop support
    - Context menu with language-specific actions
    """
    
    # Signals
    file_selected = pyqtSignal(str)  # file_path
    folder_selected = pyqtSignal(str)  # folder_path
    file_activated = pyqtSignal(str)  # file_path (double-click)
    context_menu_requested = pyqtSignal(str, object)  # file_path, menu
    file_dragged = pyqtSignal(str)  # file_path (drag started)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._project_path: str = ""
        self._all_items: List[tuple] = []  # (name_lower, item, path)
        self._icon_provider = FileIconProvider()
        self._language_support = LanguageSupport()
        self._git_service = GitStatusService()
        self._file_filter: Optional[Callable[[str], bool]] = None
        self._search_text: str = ""
        
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
        
        # Enable drag and drop
        self._tree.setDragEnabled(True)
        self._tree.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self._tree.setDefaultDropAction(Qt.DropAction.CopyAction)
        self._tree.startDrag = self._start_drag
        
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
        
        # Detect Git repository
        is_git_repo = self._git_service.detect_repo(project_path)
        if is_git_repo:
            self._git_service.refresh_all_status()
        
        project_name = Path(project_path).name
        root = FileTreeItem(self._tree, [f"📦 {project_name}"])
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
        git_info = " | Git" if is_git_repo else ""
        self._stats_label.setText(f"📊 {file_count} files | {lang_name}{git_info}")
        
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
                    tree_item = FileTreeItem(parent_item, [item.name])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item))
                    tree_item.setIcon(0, self._icon_provider.get_folder_icon(item.name))
                    file_count += self._add_directory(item, tree_item)
                else:
                    # Check if file should be shown
                    if self._file_filter and not self._file_filter(str(item)):
                        continue
                    
                    file_path = str(item)
                    tree_item = FileTreeItem(parent_item, [item.name])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, file_path)
                    tree_item.setIcon(0, self._icon_provider.get_file_icon(item.suffix))
                    
                    # Set Git status
                    git_status = self._git_service.get_file_status(file_path)
                    if git_status:
                        tree_item.set_git_status(git_status)
                    
                    self._all_items.append((item.name.lower(), tree_item, file_path))
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
        self._search_text = text.lower().strip()
        self._search_timer.stop()
        self._search_timer.start(150)  # 150ms debounce
        
    def _apply_search(self):
        """Apply search filter with fuzzy matching."""
        text = self._search_text
        
        if not text:
            self._set_all_items_visible(True)
            self._clear_highlights()
            return
        
        # Filter items with fuzzy matching
        for name, item, path in self._all_items:
            # Simple fuzzy matching: all characters in text must appear in name in order
            matches = self._fuzzy_match(text, name)
            item.setHidden(not matches)
            item.set_highlighted(matches)
            
            # Expand parent if match found
            if matches:
                parent = item.parent()
                while parent:
                    parent.setExpanded(True)
                    parent.setHidden(False)
                    parent = parent.parent()
                    
        # Hide folders that have no visible children
        self._prune_empty_folders(self._tree.invisibleRootItem())
        
    def _fuzzy_match(self, pattern: str, text: str) -> bool:
        """Perform fuzzy matching.
        
        Args:
            pattern: Search pattern
            text: Text to match against
            
        Returns:
            True if pattern fuzzy matches text
        """
        pattern = pattern.lower()
        text = text.lower()
        
        # Simple substring match first
        if pattern in text:
            return True
            
        # Fuzzy match: check if all characters in pattern appear in text in order
        pattern_idx = 0
        for char in text:
            if pattern_idx < len(pattern) and char == pattern[pattern_idx]:
                pattern_idx += 1
        
        return pattern_idx == len(pattern)
        
    def _clear_highlights(self):
        """Clear all search highlights."""
        for _, item, _ in self._all_items:
            item.set_highlighted(False)
        
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
            
            # Add Git status info if available
            if isinstance(item, FileTreeItem):
                git_status = item.get_git_status()
                if git_status:
                    menu.addSeparator()
                    menu.addAction(f"Git: {git_status.display_name}")
            
            menu.addSeparator()
            
        # Emit signal for custom actions
        self.context_menu_requested.emit(path, menu)
        
        if not menu.isEmpty():
            menu.exec(self._tree.viewport().mapToGlobal(position))
            
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
    def _start_drag(self, supportedActions):
        """Handle drag start."""
        item = self._tree.currentItem()
        if not item:
            return
            
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path or not Path(path).is_file():
            return
            
        # Create mime data
        mime_data = QMimeData()
        mime_data.setText(path)
        mime_data.setUrls([path])  # For external drops
        
        # Also set custom data for internal use
        mime_data.setData("application/x-filetree-item", path.encode())
        
        # Create drag
        drag = QDrag(self._tree)
        drag.setMimeData(mime_data)
        
        # Emit signal
        self.file_dragged.emit(path)
        
        # Execute drag
        drag.exec(supportedActions)
        
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
            self._git_service.clear_cache()
            self.load_project(self._project_path)
            
    def refresh_git_status(self):
        """Refresh only Git status without reloading entire tree."""
        if self._project_path:
            self._git_service.refresh_all_status()
            for name, item, path in self._all_items:
                if isinstance(item, FileTreeItem):
                    git_status = self._git_service.get_file_status(path)
                    item.set_git_status(git_status)
            
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._search_box.setFocus()
            self._search_box.selectAll()
        elif event.key() == Qt.Key.Key_Escape:
            self._search_box.clear()
        else:
            super().keyPressEvent(event)
            
    def get_git_service(self) -> GitStatusService:
        """Get the Git status service."""
        return self._git_service
