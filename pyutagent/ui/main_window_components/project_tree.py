"""Project tree widget for displaying project structure."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QLabel, QMessageBox, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon

from ..styles import get_style_manager
from ..components import get_notification_manager
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class ProjectTreeWidget(QWidget):
    """Enhanced tree widget for displaying project structure with search."""

    file_selected = pyqtSignal(str)
    generate_incremental = pyqtSignal(str, bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_path: str = ""
        self._all_items: list = []
        self._style_manager = get_style_manager()
        self._notification_manager = get_notification_manager()
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        header = QLabel("📁 Project Files")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Search files... (Ctrl+F)")
        self.search_box.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_box)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMaximumWidth(350)
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.tree)
        
        self.stats_label = QLabel("No project loaded")
        self.stats_label.setStyleSheet("color: #757575; font-size: 11px; padding: 5px;")
        layout.addWidget(self.stats_label)

    def load_project(self, project_path: str):
        """Load project structure into tree."""
        self.tree.clear()
        self._all_items.clear()
        self.project_path = project_path

        project_name = Path(project_path).name
        root = QTreeWidgetItem(self.tree, [f"📦 {project_name}"])
        root.setData(0, Qt.ItemDataRole.UserRole, project_path)
        root.setIcon(0, self._get_icon("project"))

        settings = get_settings()
        src_dir = Path(project_path) / settings.project_paths.src_main_java
        
        file_count = 0
        if src_dir.exists():
            file_count = self._add_directory(src_dir, root)

        root.setExpanded(True)
        
        self.stats_label.setText(f"📊 {file_count} Java files found")
        
        logger.info(f"Loaded project: {project_path} with {file_count} files")

    def _add_directory(self, dir_path: Path, parent_item: QTreeWidgetItem) -> int:
        """Recursively add directory contents. Returns file count."""
        file_count = 0
        try:
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    tree_item = QTreeWidgetItem(parent_item, [f"📂 {item.name}"])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item))
                    tree_item.setIcon(0, self._get_icon("folder"))
                    file_count += self._add_directory(item, tree_item)
                elif item.suffix == '.java':
                    tree_item = QTreeWidgetItem(parent_item, [f"☕ {item.name}"])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item))
                    tree_item.setIcon(0, self._get_icon("java"))
                    self._all_items.append((item.name.lower(), tree_item))
                    file_count += 1
        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {dir_path}")
        except Exception as e:
            logger.exception(f"Failed to add directory: {dir_path}")
        
        return file_count

    def _get_icon(self, icon_type: str) -> QIcon:
        """Get icon for file type (using emoji as fallback)."""
        return QIcon()

    def on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click."""
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and path.endswith('.java'):
            self.file_selected.emit(path)

    def get_selected_file(self) -> str:
        """Get currently selected file path."""
        item = self.tree.currentItem()
        if item:
            path = item.data(0, Qt.ItemDataRole.UserRole)
            if path and path.endswith('.java'):
                return path
        return ""

    def _on_search_changed(self, text: str):
        """Handle search text change."""
        text = text.lower().strip()
        
        if not text:
            self._set_all_items_visible(True)
            return
        
        for name, item in self._all_items:
            item.setHidden(text not in name)
            
            if text in name:
                parent = item.parent()
                while parent:
                    parent.setExpanded(True)
                    parent = parent.parent()

    def _set_all_items_visible(self, visible: bool):
        """Set visibility of all items."""
        for _, item in self._all_items:
            item.setHidden(not visible)

    def _on_context_menu(self, position):
        """Show context menu for tree item."""
        item = self.tree.itemAt(position)
        if not item:
            return
        
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if not path or not path.endswith('.java'):
            return
        
        menu = QMenu(self)
        
        generate_action = menu.addAction("🧪 Generate Tests")
        generate_action.triggered.connect(lambda: self.file_selected.emit(path))
        
        generate_incr_action = menu.addAction("🔄 Generate Tests (Incremental)")
        generate_incr_action.setToolTip("Preserve existing passing tests")
        generate_incr_action.triggered.connect(lambda: self.generate_incremental.emit(path, True, False))
        
        generate_skip_action = menu.addAction("⚡ Generate Tests (Skip Analysis)")
        generate_skip_action.setToolTip("Skip running existing tests, just analyze file content")
        generate_skip_action.triggered.connect(lambda: self.generate_incremental.emit(path, True, True))
        
        menu.addSeparator()
        
        copy_path_action = menu.addAction("📋 Copy Path")
        copy_path_action.triggered.connect(lambda: self._copy_to_clipboard(path))
        
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard."""
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        self._notification_manager.show_info("Path copied to clipboard", duration=2000)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_F and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.search_box.setFocus()
            self.search_box.selectAll()
        else:
            super().keyPressEvent(event)
