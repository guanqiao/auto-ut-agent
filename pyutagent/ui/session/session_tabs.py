"""Session tabs widget for switching between sessions."""

import logging
from typing import Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QMenu, QAbstractItemView,
    QFrame, QLineEdit, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QAction

from .session_manager import SessionManager, ChatSession

logger = logging.getLogger(__name__)


class SessionListItem(QWidget):
    """Custom widget for session list items."""
    
    def __init__(self, session: ChatSession, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._session = session
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the item UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Favorite icon
        fav_icon = "⭐" if self._session.is_favorite else "  "
        self._fav_label = QLabel(fav_icon)
        self._fav_label.setFixedWidth(20)
        layout.addWidget(self._fav_label)
        
        # Title
        self._title_label = QLabel(self._session.title)
        self._title_label.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self._title_label, stretch=1)
        
        # Message count
        msg_count = len(self._session.messages)
        self._count_label = QLabel(f"({msg_count})")
        self._count_label.setStyleSheet("color: #858585; font-size: 10px;")
        layout.addWidget(self._count_label)
        
    def update_display(self):
        """Update the display."""
        fav_icon = "⭐" if self._session.is_favorite else "  "
        self._fav_label.setText(fav_icon)
        self._title_label.setText(self._session.title)
        self._count_label.setText(f"({len(self._session.messages)})")


class SessionTabs(QWidget):
    """Session tabs widget for managing and switching sessions.
    
    Features:
    - List of sessions
    - Search/filter
    - New session button
    - Context menu actions
    """
    
    # Signals
    session_selected = pyqtSignal(str)  # session_id
    session_created = pyqtSignal()
    session_renamed = pyqtSignal(str, str)  # session_id, new_title
    session_deleted = pyqtSignal(str)  # session_id
    session_favorited = pyqtSignal(str, bool)  # session_id, is_favorite
    session_exported = pyqtSignal(str)  # session_id
    session_duplicated = pyqtSignal(str)  # session_id
    
    def __init__(self, session_manager: SessionManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._session_manager = session_manager
        self._current_session_id: Optional[str] = None
        
        self.setup_ui()
        self._connect_signals()
        self.refresh_list()
        
    def setup_ui(self):
        """Setup the session tabs UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        header.setFixedHeight(40)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)
        
        title = QLabel("💬 Sessions")
        title.setStyleSheet("color: #CCCCCC; font-weight: bold;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # New session button
        self._btn_new = QPushButton("+ New")
        self._btn_new.setFixedHeight(24)
        self._btn_new.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
        """)
        self._btn_new.clicked.connect(self._on_new_session)
        header_layout.addWidget(self._btn_new)
        
        layout.addWidget(header)
        
        # Search box
        search_frame = QFrame()
        search_frame.setStyleSheet("background-color: #252526;")
        search_layout = QHBoxLayout(search_frame)
        search_layout.setContentsMargins(8, 4, 8, 4)
        
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("🔍 Search sessions...")
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
        search_layout.addWidget(self._search_box)
        
        layout.addWidget(search_frame)
        
        # Session list
        self._list_widget = QListWidget()
        self._list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list_widget.customContextMenuRequested.connect(self._on_context_menu)
        self._list_widget.itemClicked.connect(self._on_item_clicked)
        self._list_widget.setStyleSheet("""
            QListWidget {
                background-color: #1E1E1E;
                border: none;
                outline: none;
            }
            QListWidget::item {
                background-color: transparent;
                border-bottom: 1px solid #2C2C2C;
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:hover {
                background-color: #2A2D2E;
            }
        """)
        layout.addWidget(self._list_widget, stretch=1)
        
        # Stats footer
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-top: 1px solid #3C3C3C;
            }
        """)
        footer.setFixedHeight(28)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(8, 4, 8, 4)
        
        self._stats_label = QLabel("0 sessions")
        self._stats_label.setStyleSheet("color: #858585; font-size: 11px;")
        footer_layout.addWidget(self._stats_label)
        
        layout.addWidget(footer)
        
    def _connect_signals(self):
        """Connect to session manager signals."""
        self._session_manager.session_list_changed.connect(self.refresh_list)
        self._session_manager.session_saved.connect(self._on_session_saved)
        
    def refresh_list(self):
        """Refresh the session list."""
        self._list_widget.clear()
        
        sessions = self._session_manager.get_all_sessions()
        
        for session in sessions:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            
            # Create custom widget
            widget = SessionListItem(session)
            
            item.setSizeHint(widget.sizeHint())
            self._list_widget.addItem(item)
            self._list_widget.setItemWidget(item, widget)
            
            # Select current session
            if session.id == self._current_session_id:
                item.setSelected(True)
        
        # Update stats
        self._stats_label.setText(f"{len(sessions)} session{'s' if len(sessions) != 1 else ''}")
        
    def _on_search(self, text: str):
        """Handle search text change."""
        if not text:
            self.refresh_list()
            return
        
        # Search sessions
        sessions = self._session_manager.search_sessions(text)
        
        self._list_widget.clear()
        for session in sessions:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            
            widget = SessionListItem(session)
            item.setSizeHint(widget.sizeHint())
            self._list_widget.addItem(item)
            self._list_widget.setItemWidget(item, widget)
            
        self._stats_label.setText(f"{len(sessions)} found")
        
    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle item click."""
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id:
            self._current_session_id = session_id
            self.session_selected.emit(session_id)
            
    def _on_context_menu(self, position):
        """Show context menu."""
        item = self._list_widget.itemAt(position)
        if not item:
            return
        
        session_id = item.data(Qt.ItemDataRole.UserRole)
        session = self._session_manager.get_session(session_id)
        if not session:
            return
        
        menu = QMenu(self)
        
        # Rename action
        rename_action = QAction("✏️ Rename", self)
        rename_action.triggered.connect(lambda: self._on_rename(session_id))
        menu.addAction(rename_action)
        
        # Favorite action
        fav_text = "⭐ Unfavorite" if session.is_favorite else "⭐ Favorite"
        fav_action = QAction(fav_text, self)
        fav_action.triggered.connect(lambda: self._on_toggle_favorite(session_id))
        menu.addAction(fav_action)
        
        menu.addSeparator()
        
        # Duplicate action
        duplicate_action = QAction("📋 Duplicate", self)
        duplicate_action.triggered.connect(lambda: self._on_duplicate(session_id))
        menu.addAction(duplicate_action)
        
        # Export action
        export_action = QAction("💾 Export", self)
        export_action.triggered.connect(lambda: self._on_export(session_id))
        menu.addAction(export_action)
        
        menu.addSeparator()
        
        # Delete action
        delete_action = QAction("🗑️ Delete", self)
        delete_action.triggered.connect(lambda: self._on_delete(session_id))
        menu.addAction(delete_action)
        
        menu.exec(self._list_widget.viewport().mapToGlobal(position))
        
    def _on_new_session(self):
        """Handle new session button click."""
        self.session_created.emit()
        
    def _on_rename(self, session_id: str):
        """Handle rename action."""
        # TODO: Show rename dialog
        logger.info(f"Rename session: {session_id}")
        
    def _on_toggle_favorite(self, session_id: str):
        """Handle toggle favorite action."""
        is_fav = self._session_manager.toggle_favorite(session_id)
        self.session_favorited.emit(session_id, is_fav)
        self.refresh_list()
        
    def _on_duplicate(self, session_id: str):
        """Handle duplicate action."""
        new_session = self._session_manager.duplicate_session(session_id)
        if new_session:
            self.session_duplicated.emit(new_session.id)
            self.refresh_list()
            
    def _on_export(self, session_id: str):
        """Handle export action."""
        self.session_exported.emit(session_id)
        
    def _on_delete(self, session_id: str):
        """Handle delete action."""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Delete Session",
            "Are you sure you want to delete this session?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self._session_manager.delete_session(session_id):
                self.session_deleted.emit(session_id)
                
    def _on_session_saved(self, session_id: str):
        """Handle session saved signal."""
        # Update the item display
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == session_id:
                widget = self._list_widget.itemWidget(item)
                if isinstance(widget, SessionListItem):
                    widget.update_display()
                break
                
    def set_current_session(self, session_id: str):
        """Set the current session."""
        self._current_session_id = session_id
        
        # Update selection
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == session_id:
                item.setSelected(True)
                break
                
    def get_current_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._current_session_id
