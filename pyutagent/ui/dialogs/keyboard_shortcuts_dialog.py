"""Keyboard shortcuts configuration dialog."""

import logging
from typing import Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QMessageBox, QWidget, QComboBox, QStyledItemDelegate,
    QKeySequenceEdit, QGroupBox, QSplitter, QTextEdit,
    QApplication, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QKeyCombination
from PyQt6.QtGui import QKeySequence, QFont, QColor, QIcon, QKeyEvent

from ..styles import get_style_manager

logger = logging.getLogger(__name__)


class ShortcutCategory(Enum):
    """Shortcut categories."""
    FILE = "File"
    EDIT = "Edit"
    VIEW = "View"
    SEARCH = "Search"
    AI = "AI"
    SETTINGS = "Settings"
    THEME = "Theme"
    HELP = "Help"


@dataclass
class ShortcutDefinition:
    """Definition of a keyboard shortcut."""
    id: str
    name: str
    description: str
    default_key: str
    category: ShortcutCategory
    callback: Optional[Callable] = None
    current_key: str = field(default="")

    def __post_init__(self):
        if not self.current_key:
            self.current_key = self.default_key

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "current_key": self.current_key
        }

    @classmethod
    def from_dict(cls, data: dict, default_shortcut: 'ShortcutDefinition') -> 'ShortcutDefinition':
        """Create from dictionary."""
        shortcut = cls(
            id=data.get("id", default_shortcut.id),
            name=default_shortcut.name,
            description=default_shortcut.description,
            default_key=default_shortcut.default_key,
            category=default_shortcut.category,
            callback=default_shortcut.callback
        )
        shortcut.current_key = data.get("current_key", default_shortcut.default_key)
        return shortcut


class KeySequenceEdit(QKeySequenceEdit):
    """Custom key sequence edit with better UX."""

    editing_finished = pyqtSignal()
    escape_pressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setClearButtonEnabled(True)
        self._recording = False

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press for recording shortcut."""
        key = event.key()

        # Handle Escape to cancel editing
        if key == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            return

        # Handle Backspace/Delete to clear
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.clear()
            return

        # Don't record modifier-only keys
        if key in (
            Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt,
            Qt.Key.Key_Meta, Qt.Key.Key_AltGr
        ):
            return

        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        """Handle focus out."""
        super().focusOutEvent(event)
        self.editing_finished.emit()


class ShortcutTableItemDelegate(QStyledItemDelegate):
    """Custom delegate for shortcut table items."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_dark = True

    def set_dark_mode(self, is_dark: bool):
        """Set dark mode."""
        self._is_dark = is_dark

    def paint(self, painter, option, index):
        """Paint the item."""
        # Get the shortcut definition
        shortcut = index.data(Qt.ItemDataRole.UserRole)
        if not shortcut:
            super().paint(painter, option, index)
            return

        painter.save()

        # Determine colors
        is_selected = option.state & QStyle.StateFlag.State_Selected

        if self._is_dark:
            bg_color = QColor("#094771") if is_selected else QColor("#2D2D2D")
            text_color = QColor("#E0E0E0")
            modified_color = QColor("#FFD700")  # Gold for modified shortcuts
        else:
            bg_color = QColor("#E3F2FD") if is_selected else QColor("#FFFFFF")
            text_color = QColor("#212121")
            modified_color = QColor("#1976D2")

        painter.fillRect(option.rect, bg_color)

        # Get column
        column = index.column()

        if column == 0:  # Category
            text = shortcut.category.value
        elif column == 1:  # Name
            text = shortcut.name
        elif column == 2:  # Shortcut
            text = shortcut.current_key if shortcut.current_key else "None"
            # Use different color if modified from default
            if shortcut.current_key != shortcut.default_key:
                painter.setPen(modified_color)
        else:  # Description
            text = shortcut.description

        if column != 2 or shortcut.current_key == shortcut.default_key:
            painter.setPen(text_color)

        # Draw text with padding
        padding = 8
        rect = option.rect.adjusted(padding, 0, -padding, 0)
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        painter.restore()


class ConflictDetector:
    """Detects conflicts between keyboard shortcuts."""

    @staticmethod
    def find_conflicts(shortcuts: List[ShortcutDefinition]) -> List[Tuple[ShortcutDefinition, ShortcutDefinition]]:
        """
        Find all shortcut conflicts.

        Returns:
            List of conflicting shortcut pairs
        """
        conflicts = []
        key_map: Dict[str, List[ShortcutDefinition]] = {}

        for shortcut in shortcuts:
            if not shortcut.current_key:
                continue

            # Normalize key sequence
            normalized = ConflictDetector._normalize_key(shortcut.current_key)

            if normalized in key_map:
                for existing in key_map[normalized]:
                    conflicts.append((existing, shortcut))
                key_map[normalized].append(shortcut)
            else:
                key_map[normalized] = [shortcut]

        return conflicts

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize key sequence for comparison."""
        # Convert to standard format
        key = key.upper()
        parts = key.replace("+", " ").replace("-", " ").split()

        # Sort modifiers
        modifiers = []
        main_key = ""

        for part in parts:
            if part in ("CTRL", "CONTROL"):
                modifiers.append("CTRL")
            elif part in ("SHIFT",):
                modifiers.append("SHIFT")
            elif part in ("ALT",):
                modifiers.append("ALT")
            elif part in ("META", "CMD", "COMMAND", "WIN"):
                modifiers.append("META")
            else:
                main_key = part

        modifiers.sort()
        return "+".join(modifiers + [main_key]) if main_key else "+".join(modifiers)

    @staticmethod
    def check_validity(key: str) -> Tuple[bool, str]:
        """
        Check if a key sequence is valid.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not key:
            return True, ""  # Empty is valid (means no shortcut)

        # Check for reserved system shortcuts
        reserved = [
            "ALT+F4",  # Close window
            "CTRL+ALT+DELETE",  # System menu
            "CTRL+SHIFT+ESC",  # Task manager
            "WIN+L",  # Lock screen
        ]

        normalized = ConflictDetector._normalize_key(key)
        for reserved_key in reserved:
            if normalized == ConflictDetector._normalize_key(reserved_key):
                return False, f"'{key}' is a reserved system shortcut"

        # Check if it's a valid key sequence
        sequence = QKeySequence(key)
        if sequence.isEmpty():
            return False, f"'{key}' is not a valid key sequence"

        return True, ""


class KeyboardShortcutsDialog(QDialog):
    """Dialog for viewing and editing keyboard shortcuts."""

    shortcuts_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(900, 600)

        self._style_manager = get_style_manager()
        self._shortcuts: Dict[str, ShortcutDefinition] = {}
        self._conflict_detector = ConflictDetector()
        self._config_path = Path.home() / ".pyutagent" / "shortcuts.json"

        self.setup_ui()
        self.register_default_shortcuts()
        self.load_shortcuts()
        self.apply_styles()

    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header
        header = QLabel("<h2>Keyboard Shortcuts</h2>")
        header.setStyleSheet("margin-bottom: 10px;")
        layout.addWidget(header)

        # Description
        description = QLabel(
            "Customize keyboard shortcuts for PyUT Agent. "
            "Double-click a shortcut to edit it. "
            "Modified shortcuts are shown in gold/blue color."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #808080; margin-bottom: 10px;")
        layout.addWidget(description)

        # Search bar
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_layout.addWidget(search_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter by name, category, or shortcut...")
        self.search_box.textChanged.connect(self.on_search_changed)
        search_layout.addWidget(self.search_box)

        layout.addLayout(search_layout)

        # Splitter for table and detail panel
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table container
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        # Shortcuts table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Category", "Command", "Shortcut", "Description"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self.on_cell_double_clicked)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)

        # Set custom delegate
        self._item_delegate = ShortcutTableItemDelegate(self)
        self._item_delegate.set_dark_mode(self._style_manager.current_theme == "dark")
        self.table.setItemDelegate(self._item_delegate)

        table_layout.addWidget(self.table)
        splitter.addWidget(table_container)

        # Detail panel
        detail_panel = QWidget()
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(10, 0, 0, 0)

        # Edit section
        edit_group = QGroupBox("Edit Shortcut")
        edit_layout = QVBoxLayout(edit_group)

        # Current selection info
        self.selected_command_label = QLabel("Select a command to edit")
        self.selected_command_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        edit_layout.addWidget(self.selected_command_label)

        self.selected_description_label = QLabel("")
        self.selected_description_label.setStyleSheet("color: #808080;")
        self.selected_description_label.setWordWrap(True)
        edit_layout.addWidget(self.selected_description_label)

        edit_layout.addSpacing(10)

        # Default shortcut display
        default_layout = QHBoxLayout()
        default_layout.addWidget(QLabel("Default:"))
        self.default_shortcut_label = QLabel("-")
        self.default_shortcut_label.setStyleSheet("color: #808080;")
        default_layout.addWidget(self.default_shortcut_label)
        default_layout.addStretch()
        edit_layout.addLayout(default_layout)

        # Key sequence editor
        key_layout = QHBoxLayout()
        key_layout.addWidget(QLabel("New shortcut:"))

        self.key_sequence_edit = KeySequenceEdit()
        self.key_sequence_edit.setEnabled(False)
        self.key_sequence_edit.editing_finished.connect(self.on_shortcut_edited)
        self.key_sequence_edit.escape_pressed.connect(self.on_shortcut_edit_cancelled)
        key_layout.addWidget(self.key_sequence_edit)

        edit_layout.addLayout(key_layout)

        # Conflict warning
        self.conflict_label = QLabel("")
        self.conflict_label.setStyleSheet("color: #FF6B6B; font-weight: bold;")
        self.conflict_label.setWordWrap(True)
        edit_layout.addWidget(self.conflict_label)

        # Buttons
        button_layout = QHBoxLayout()

        self.clear_button = QPushButton("Clear")
        self.clear_button.setEnabled(False)
        self.clear_button.clicked.connect(self.on_clear_shortcut)
        button_layout.addWidget(self.clear_button)

        self.reset_button = QPushButton("Reset to Default")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.on_reset_shortcut)
        button_layout.addWidget(self.reset_button)

        button_layout.addStretch()

        self.apply_button = QPushButton("Apply")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.on_apply_shortcut)
        button_layout.addWidget(self.apply_button)

        edit_layout.addLayout(button_layout)

        detail_layout.addWidget(edit_group)

        # Conflicts section
        conflicts_group = QGroupBox("Conflicts")
        conflicts_layout = QVBoxLayout(conflicts_group)

        self.conflicts_text = QTextEdit()
        self.conflicts_text.setReadOnly(True)
        self.conflicts_text.setMaximumHeight(100)
        conflicts_layout.addWidget(self.conflicts_text)

        detail_layout.addWidget(conflicts_group)

        # Legend
        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout(legend_group)

        legend_text = QLabel(
            "<span style='color: #FFD700;'>●</span> Modified from default<br>"
            "<span style='color: #FF6B6B;'>●</span> Has conflicts"
        )
        legend_layout.addWidget(legend_text)

        detail_layout.addWidget(legend_group)
        detail_layout.addStretch()

        splitter.addWidget(detail_panel)
        splitter.setSizes([600, 300])

        layout.addWidget(splitter)

        # Bottom buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        self.reset_all_button = QPushButton("Reset All to Defaults")
        self.reset_all_button.clicked.connect(self.on_reset_all)
        bottom_layout.addWidget(self.reset_all_button)

        bottom_layout.addSpacing(20)

        self.import_button = QPushButton("Import...")
        self.import_button.clicked.connect(self.on_import)
        bottom_layout.addWidget(self.import_button)

        self.export_button = QPushButton("Export...")
        self.export_button.clicked.connect(self.on_export)
        bottom_layout.addWidget(self.export_button)

        bottom_layout.addSpacing(20)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        bottom_layout.addWidget(close_button)

        layout.addLayout(bottom_layout)

    def register_default_shortcuts(self):
        """Register default shortcuts."""
        defaults = [
            # File
            ShortcutDefinition("file.open", "Open Project", "Open a project directory", "Ctrl+O", ShortcutCategory.FILE),
            ShortcutDefinition("file.new_session", "New Session", "Create a new chat session", "Ctrl+N", ShortcutCategory.FILE),
            ShortcutDefinition("file.session_history", "Session History", "View and restore previous sessions", "Ctrl+H", ShortcutCategory.FILE),
            ShortcutDefinition("file.exit", "Exit Application", "Close the application", "Alt+F4", ShortcutCategory.FILE),

            # Edit
            ShortcutDefinition("edit.undo", "Undo", "Undo last action", "Ctrl+Z", ShortcutCategory.EDIT),
            ShortcutDefinition("edit.redo", "Redo", "Redo last undone action", "Ctrl+Y", ShortcutCategory.EDIT),
            ShortcutDefinition("edit.copy", "Copy", "Copy selected text", "Ctrl+C", ShortcutCategory.EDIT),
            ShortcutDefinition("edit.paste", "Paste", "Paste from clipboard", "Ctrl+V", ShortcutCategory.EDIT),

            # View
            ShortcutDefinition("view.toggle_sidebar", "Toggle Sidebar", "Show or hide the sidebar panel", "Ctrl+B", ShortcutCategory.VIEW),
            ShortcutDefinition("view.toggle_agent", "Toggle Agent Panel", "Show or hide the AI agent panel", "Ctrl+J", ShortcutCategory.VIEW),
            ShortcutDefinition("view.terminal", "Show Terminal", "Open integrated terminal", "Ctrl+`", ShortcutCategory.VIEW),

            # Search
            ShortcutDefinition("search.semantic", "Semantic Search", "Search code using natural language", "Ctrl+Shift+F", ShortcutCategory.SEARCH),
            ShortcutDefinition("search.files", "Find in Files", "Search text across all project files", "Ctrl+Shift+H", ShortcutCategory.SEARCH),
            ShortcutDefinition("search.command_palette", "Command Palette", "Quick access to all commands", "Ctrl+Shift+P", ShortcutCategory.SEARCH),

            # AI
            ShortcutDefinition("ai.generate_tests", "Generate Tests", "Generate unit tests for selected file", "Ctrl+G", ShortcutCategory.AI),
            ShortcutDefinition("ai.review", "Review Code", "Get AI code review", "Ctrl+Shift+R", ShortcutCategory.AI),

            # Settings
            ShortcutDefinition("settings.llm", "LLM Settings", "Configure LLM provider and API keys", "Ctrl+,", ShortcutCategory.SETTINGS),
            ShortcutDefinition("settings.shortcuts", "Keyboard Shortcuts", "View and customize keyboard shortcuts", "Ctrl+/", ShortcutCategory.SETTINGS),

            # Theme
            ShortcutDefinition("theme.light", "Light Theme", "Switch to light color theme", "", ShortcutCategory.THEME),
            ShortcutDefinition("theme.dark", "Dark Theme", "Switch to dark color theme", "", ShortcutCategory.THEME),

            # Help
            ShortcutDefinition("help.slash_commands", "Slash Commands Help", "Show available slash commands", "", ShortcutCategory.HELP),
            ShortcutDefinition("help.about", "About", "Show application information", "", ShortcutCategory.HELP),
        ]

        for shortcut in defaults:
            self._shortcuts[shortcut.id] = shortcut

    def load_shortcuts(self):
        """Load shortcuts from configuration file."""
        if not self._config_path.exists():
            self.populate_table()
            return

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for shortcut_data in data.get("shortcuts", []):
                shortcut_id = shortcut_data.get("id")
                if shortcut_id in self._shortcuts:
                    self._shortcuts[shortcut_id].current_key = shortcut_data.get("current_key", "")

            logger.info(f"Loaded shortcuts from {self._config_path}")
        except Exception as e:
            logger.error(f"Failed to load shortcuts: {e}")

        self.populate_table()
        self.update_conflicts()

    def save_shortcuts(self):
        """Save shortcuts to configuration file."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "shortcuts": [s.to_dict() for s in self._shortcuts.values()]
            }

            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved shortcuts to {self._config_path}")
            self.shortcuts_changed.emit()
        except Exception as e:
            logger.error(f"Failed to save shortcuts: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save shortcuts: {e}")

    def populate_table(self):
        """Populate the shortcuts table."""
        # Filter shortcuts based on search
        search_text = self.search_box.text().lower()

        filtered_shortcuts = []
        for shortcut in self._shortcuts.values():
            if search_text:
                if (search_text not in shortcut.name.lower() and
                    search_text not in shortcut.description.lower() and
                    search_text not in shortcut.category.value.lower() and
                    search_text not in shortcut.current_key.lower()):
                    continue
            filtered_shortcuts.append(shortcut)

        # Sort by category and name
        filtered_shortcuts.sort(key=lambda s: (s.category.value, s.name))

        self.table.setRowCount(len(filtered_shortcuts))

        for i, shortcut in enumerate(filtered_shortcuts):
            # Store shortcut in first column's item
            category_item = QTableWidgetItem(shortcut.category.value)
            category_item.setData(Qt.ItemDataRole.UserRole, shortcut)
            self.table.setItem(i, 0, category_item)

            self.table.setItem(i, 1, QTableWidgetItem(shortcut.name))
            self.table.setItem(i, 2, QTableWidgetItem(shortcut.current_key if shortcut.current_key else "None"))
            self.table.setItem(i, 3, QTableWidgetItem(shortcut.description))

    def update_conflicts(self):
        """Update conflicts display."""
        conflicts = self._conflict_detector.find_conflicts(list(self._shortcuts.values()))

        if conflicts:
            conflict_text = "<b>Conflicting shortcuts detected:</b><br>"
            for s1, s2 in conflicts:
                conflict_text += f"• '{s1.name}' and '{s2.name}' both use '{s1.current_key}'<br>"
            self.conflicts_text.setHtml(conflict_text)
        else:
            self.conflicts_text.setHtml("<span style='color: #4CAF50;'>No conflicts detected</span>")

        return conflicts

    def on_search_changed(self, text: str):
        """Handle search text change."""
        self.populate_table()

    def on_cell_double_clicked(self, index):
        """Handle cell double click to edit shortcut."""
        if index.column() == 2:  # Shortcut column
            self.start_editing_shortcut(index.row())

    def on_selection_changed(self):
        """Handle selection change."""
        selected = self.table.selectedItems()
        if not selected:
            return

        # Get shortcut from first column
        row = selected[0].row()
        shortcut = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)

        if shortcut:
            self.selected_command_label.setText(shortcut.name)
            self.selected_description_label.setText(shortcut.description)
            self.default_shortcut_label.setText(shortcut.default_key if shortcut.default_key else "None")

            # Enable editing
            self.key_sequence_edit.setEnabled(True)
            self.key_sequence_edit.setKeySequence(QKeySequence(shortcut.current_key))
            self.clear_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.apply_button.setEnabled(True)

            self._current_editing_shortcut = shortcut
            self.conflict_label.setText("")

    def start_editing_shortcut(self, row: int):
        """Start editing a shortcut."""
        self.table.selectRow(row)
        self.key_sequence_edit.setFocus()

    def on_shortcut_edited(self):
        """Handle shortcut edit completion."""
        if not hasattr(self, '_current_editing_shortcut'):
            return

        key_sequence = self.key_sequence_edit.keySequence()
        key_text = key_sequence.toString()

        # Check validity
        is_valid, error_msg = self._conflict_detector.check_validity(key_text)
        if not is_valid:
            self.conflict_label.setText(f"⚠ {error_msg}")
            return

        # Check for conflicts
        shortcut = self._current_editing_shortcut
        conflicts = []
        for other in self._shortcuts.values():
            if other.id != shortcut.id and other.current_key == key_text and key_text:
                conflicts.append(other)

        if conflicts:
            conflict_names = ", ".join([c.name for c in conflicts])
            self.conflict_label.setText(f"⚠ Conflicts with: {conflict_names}")
        else:
            self.conflict_label.setText("")

    def on_shortcut_edit_cancelled(self):
        """Handle shortcut edit cancellation."""
        if hasattr(self, '_current_editing_shortcut'):
            shortcut = self._current_editing_shortcut
            self.key_sequence_edit.setKeySequence(QKeySequence(shortcut.current_key))
            self.conflict_label.setText("")

    def on_clear_shortcut(self):
        """Clear the current shortcut."""
        if hasattr(self, '_current_editing_shortcut'):
            self.key_sequence_edit.clear()
            self.conflict_label.setText("")

    def on_reset_shortcut(self):
        """Reset current shortcut to default."""
        if hasattr(self, '_current_editing_shortcut'):
            shortcut = self._current_editing_shortcut
            self.key_sequence_edit.setKeySequence(QKeySequence(shortcut.default_key))
            self.conflict_label.setText("")

    def on_apply_shortcut(self):
        """Apply the edited shortcut."""
        if not hasattr(self, '_current_editing_shortcut'):
            return

        shortcut = self._current_editing_shortcut
        key_sequence = self.key_sequence_edit.keySequence()
        key_text = key_sequence.toString()

        # Check validity
        is_valid, error_msg = self._conflict_detector.check_validity(key_text)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Shortcut", error_msg)
            return

        # Update shortcut
        shortcut.current_key = key_text

        # Refresh table
        self.populate_table()
        self.update_conflicts()
        self.save_shortcuts()

        # Show confirmation
        self._notification_manager = None
        try:
            from ..components import get_notification_manager
            self._notification_manager = get_notification_manager()
            self._notification_manager.show_success(
                f"Updated shortcut for '{shortcut.name}'",
                duration=2000
            )
        except:
            pass

    def on_reset_all(self):
        """Reset all shortcuts to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset All Shortcuts",
            "Are you sure you want to reset all shortcuts to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            for shortcut in self._shortcuts.values():
                shortcut.current_key = shortcut.default_key

            self.populate_table()
            self.update_conflicts()
            self.save_shortcuts()

    def on_import(self):
        """Import shortcuts from file."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Shortcuts",
            "",
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            imported_count = 0
            for shortcut_data in data.get("shortcuts", []):
                shortcut_id = shortcut_data.get("id")
                if shortcut_id in self._shortcuts:
                    self._shortcuts[shortcut_id].current_key = shortcut_data.get("current_key", "")
                    imported_count += 1

            self.populate_table()
            self.update_conflicts()
            self.save_shortcuts()

            QMessageBox.information(
                self,
                "Import Successful",
                f"Imported {imported_count} shortcuts."
            )
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Failed to import shortcuts: {e}")

    def on_export(self):
        """Export shortcuts to file."""
        from PyQt6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Shortcuts",
            "pyutagent_shortcuts.json",
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            data = {
                "shortcuts": [s.to_dict() for s in self._shortcuts.values()]
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            QMessageBox.information(
                self,
                "Export Successful",
                f"Shortcuts exported to {file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export shortcuts: {e}")

    def apply_styles(self):
        """Apply theme styles."""
        is_dark = self._style_manager.current_theme == "dark"
        self._item_delegate.set_dark_mode(is_dark)

        bg_color = "#2D2D2D" if is_dark else "#FFFFFF"
        text_color = "#E0E0E0" if is_dark else "#212121"
        border_color = "#3C3C3C" if is_dark else "#E0E0E0"

        self.setStyleSheet(f"""
            QDialog {{
                background-color: {bg_color};
            }}
            QTableWidget {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                gridline-color: {border_color};
            }}
            QTableWidget::item {{
                padding: 6px;
            }}
            QHeaderView::section {{
                background-color: {"#3C3C3C" if is_dark else "#F5F5F5"};
                color: {text_color};
                padding: 6px;
                border: 1px solid {border_color};
            }}
            QGroupBox {{
                color: {text_color};
                border: 1px solid {border_color};
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLineEdit, QTextEdit, QKeySequenceEdit {{
                background-color: {"#1E1E1E" if is_dark else "#F5F5F5"};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 6px;
            }}
        """)

    def get_shortcut(self, shortcut_id: str) -> Optional[ShortcutDefinition]:
        """Get a shortcut definition by ID."""
        return self._shortcuts.get(shortcut_id)

    def get_all_shortcuts(self) -> List[ShortcutDefinition]:
        """Get all shortcut definitions."""
        return list(self._shortcuts.values())

    def update_shortcut(self, shortcut_id: str, new_key: str) -> bool:
        """Update a shortcut key programmatically."""
        if shortcut_id not in self._shortcuts:
            return False

        self._shortcuts[shortcut_id].current_key = new_key
        self.populate_table()
        self.update_conflicts()
        self.save_shortcuts()
        return True


def show_keyboard_shortcuts_dialog(parent=None) -> Optional[KeyboardShortcutsDialog]:
    """Show the keyboard shortcuts dialog.

    Args:
        parent: Parent widget

    Returns:
        The dialog instance or None
    """
    dialog = KeyboardShortcutsDialog(parent)
    dialog.exec()
    return dialog


# Global shortcuts manager instance
_shortcuts_dialog_instance: Optional[KeyboardShortcutsDialog] = None


def get_shortcuts_dialog() -> Optional[KeyboardShortcutsDialog]:
    """Get the global shortcuts dialog instance."""
    return _shortcuts_dialog_instance


def load_shortcuts_config() -> Dict[str, str]:
    """Load shortcuts configuration and return as a mapping.

    Returns:
        Dictionary mapping shortcut_id to current_key
    """
    config_path = Path.home() / ".pyutagent" / "shortcuts.json"

    if not config_path.exists():
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            s["id"]: s.get("current_key", "")
            for s in data.get("shortcuts", [])
        }
    except Exception as e:
        logger.error(f"Failed to load shortcuts config: {e}")
        return {}
