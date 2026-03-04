"""Keyboard shortcuts management for PyUT Agent."""

import logging
from typing import Dict, Callable, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtWidgets import QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt, QKeyCombination
from PyQt6.QtGui import QKeySequence, QAction, QShortcut

logger = logging.getLogger(__name__)


class ShortcutCategory(Enum):
    """Shortcut categories."""
    FILE = "File"
    EDIT = "Edit"
    VIEW = "View"
    GENERATE = "Generate"
    TOOLS = "Tools"
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
    
    def __post_init__(self):
        self.current_key = self.default_key


class ShortcutsManager:
    """Manages keyboard shortcuts for the application."""
    
    _instance: Optional['ShortcutsManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self._shortcuts: Dict[str, ShortcutDefinition] = {}
        self._qshortcuts: Dict[str, QShortcut] = {}
        self._parent: Optional[QWidget] = None
        
        self._register_default_shortcuts()
    
    def _register_default_shortcuts(self):
        """Register default shortcuts."""
        defaults = [
            # File
            ShortcutDefinition("file.open", "Open Project", "Open a Maven project", "Ctrl+O", ShortcutCategory.FILE),
            ShortcutDefinition("file.exit", "Exit", "Exit application", "Alt+F4", ShortcutCategory.FILE),
            
            # Generate
            ShortcutDefinition("generate.single", "Generate Tests", "Generate tests for selected file", "Ctrl+G", ShortcutCategory.GENERATE),
            ShortcutDefinition("generate.batch", "Batch Generate", "Generate tests for multiple files", "Ctrl+B", ShortcutCategory.GENERATE),
            
            # View
            ShortcutDefinition("view.history", "Test History", "Open test history", "Ctrl+H", ShortcutCategory.VIEW),
            ShortcutDefinition("view.config", "Configuration", "Open configuration", "Ctrl+Comma", ShortcutCategory.VIEW),
            
            # Tools
            ShortcutDefinition("tools.stats", "Project Statistics", "Show project statistics", "Ctrl+Shift+S", ShortcutCategory.TOOLS),
            
            # Help
            ShortcutDefinition("help.shortcuts", "Keyboard Shortcuts", "Show keyboard shortcuts", "Ctrl+Slash", ShortcutCategory.HELP),
        ]
        
        for shortcut in defaults:
            self._shortcuts[shortcut.id] = shortcut
    
    def set_parent(self, parent: QWidget):
        """Set the parent widget for shortcuts."""
        self._parent = parent
    
    def register_callback(self, shortcut_id: str, callback: Callable):
        """Register a callback for a shortcut.
        
        Args:
            shortcut_id: The shortcut ID
            callback: The callback function
        """
        if shortcut_id in self._shortcuts:
            self._shortcuts[shortcut_id].callback = callback
            self._create_qshortcut(shortcut_id)
        else:
            logger.warning(f"Shortcut not found: {shortcut_id}")
    
    def _create_qshortcut(self, shortcut_id: str):
        """Create QShortcut for a shortcut definition.
        
        Args:
            shortcut_id: The shortcut ID
        """
        if not self._parent:
            return
        
        shortcut = self._shortcuts[shortcut_id]
        
        # Remove existing shortcut if any
        if shortcut_id in self._qshortcuts:
            self._qshortcuts[shortcut_id].deleteLater()
        
        # Create new shortcut
        key_sequence = QKeySequence(shortcut.current_key)
        qshortcut = QShortcut(key_sequence, self._parent)
        qshortcut.activated.connect(lambda: self._on_activated(shortcut_id))
        
        self._qshortcuts[shortcut_id] = qshortcut
        logger.debug(f"Created shortcut: {shortcut_id} = {shortcut.current_key}")
    
    def _on_activated(self, shortcut_id: str):
        """Handle shortcut activation.
        
        Args:
            shortcut_id: The activated shortcut ID
        """
        shortcut = self._shortcuts.get(shortcut_id)
        if shortcut and shortcut.callback:
            logger.debug(f"Shortcut activated: {shortcut_id}")
            shortcut.callback()
    
    def get_shortcut(self, shortcut_id: str) -> Optional[ShortcutDefinition]:
        """Get a shortcut definition.
        
        Args:
            shortcut_id: The shortcut ID
            
        Returns:
            The shortcut definition or None
        """
        return self._shortcuts.get(shortcut_id)
    
    def get_all_shortcuts(self) -> List[ShortcutDefinition]:
        """Get all shortcut definitions.
        
        Returns:
            List of all shortcuts
        """
        return list(self._shortcuts.values())
    
    def get_shortcuts_by_category(self, category: ShortcutCategory) -> List[ShortcutDefinition]:
        """Get shortcuts by category.
        
        Args:
            category: The category
            
        Returns:
            List of shortcuts in the category
        """
        return [s for s in self._shortcuts.values() if s.category == category]
    
    def update_shortcut(self, shortcut_id: str, new_key: str) -> bool:
        """Update a shortcut key.
        
        Args:
            shortcut_id: The shortcut ID
            new_key: The new key sequence
            
        Returns:
            True if updated successfully
        """
        if shortcut_id not in self._shortcuts:
            return False
        
        self._shortcuts[shortcut_id].current_key = new_key
        self._create_qshortcut(shortcut_id)
        
        logger.info(f"Updated shortcut: {shortcut_id} = {new_key}")
        return True
    
    def reset_to_defaults(self):
        """Reset all shortcuts to defaults."""
        for shortcut in self._shortcuts.values():
            shortcut.current_key = shortcut.default_key
            self._create_qshortcut(shortcut.id)
        
        logger.info("Reset all shortcuts to defaults")


class ShortcutsDialog(QDialog):
    """Dialog for viewing and editing keyboard shortcuts."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(600, 500)
        
        self._shortcuts_manager = ShortcutsManager()
        
        self.setup_ui()
        self.load_shortcuts()
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        
        # Info label
        info = QLabel("<p>Keyboard shortcuts for PyUT Agent. Click on a shortcut to edit it.</p>")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Category", "Name", "Shortcut", "Description"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(self.table.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.on_reset)
        button_layout.addWidget(reset_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def load_shortcuts(self):
        """Load shortcuts into the table."""
        shortcuts = self._shortcuts_manager.get_all_shortcuts()
        
        # Sort by category and name
        shortcuts.sort(key=lambda s: (s.category.value, s.name))
        
        self.table.setRowCount(len(shortcuts))
        
        for i, shortcut in enumerate(shortcuts):
            self.table.setItem(i, 0, QTableWidgetItem(shortcut.category.value))
            self.table.setItem(i, 1, QTableWidgetItem(shortcut.name))
            self.table.setItem(i, 2, QTableWidgetItem(shortcut.current_key))
            self.table.setItem(i, 3, QTableWidgetItem(shortcut.description))
    
    def on_reset(self):
        """Reset shortcuts to defaults."""
        self._shortcuts_manager.reset_to_defaults()
        self.load_shortcuts()


def get_shortcuts_manager() -> ShortcutsManager:
    """Get the singleton shortcuts manager instance."""
    return ShortcutsManager()


def setup_main_window_shortcuts(main_window: QMainWindow):
    """Setup shortcuts for the main window.
    
    Args:
        main_window: The main window instance
    """
    manager = get_shortcuts_manager()
    manager.set_parent(main_window)
    
    # Register callbacks
    manager.register_callback("file.open", main_window.on_open_project)
    manager.register_callback("generate.single", main_window.on_generate_tests)
    manager.register_callback("generate.batch", main_window.on_generate_all_tests)
    manager.register_callback("view.history", main_window.on_test_history)
    manager.register_callback("view.config", main_window.on_llm_config)
    manager.register_callback("tools.stats", main_window.on_project_stats)
    manager.register_callback("help.shortcuts", lambda: ShortcutsDialog(main_window).exec())
    
    logger.info("Main window shortcuts setup complete")
