"""Session history dialog for managing and viewing past sessions."""

import logging
from typing import Optional, List
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMenu,
    QAbstractItemView, QMessageBox, QFileDialog, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QFont

from ..session.session_manager import SessionManager, ChatSession

logger = logging.getLogger(__name__)


class SessionHistoryDialog(QDialog):
    """Dialog for viewing and managing session history.
    
    Features:
    - List all sessions with details
    - Search and filter
    - Import/Export
    - Delete multiple sessions
    """
    
    # Signals
    session_selected = pyqtSignal(str)  # session_id
    session_deleted = pyqtSignal(str)  # session_id
    
    def __init__(self, session_manager: SessionManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._session_manager = session_manager
        self._selected_session_id: Optional[str] = None
        
        self.setWindowTitle("Session History")
        self.setMinimumSize(800, 600)
        
        self.setup_ui()
        self.refresh_sessions()
        
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("📚 Session History")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Stats
        self._stats_label = QLabel("0 sessions")
        self._stats_label.setStyleSheet("color: #858585;")
        header.addWidget(self._stats_label)
        
        layout.addLayout(header)
        
        # Filter bar
        filter_layout = QHBoxLayout()
        
        # Search
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("🔍 Search sessions...")
        self._search_box.textChanged.connect(self._on_search)
        filter_layout.addWidget(self._search_box, stretch=2)
        
        # Filter by type
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All Sessions", "Favorites", "Recent (7 days)", "Recent (30 days)"])
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._filter_combo, stretch=1)
        
        layout.addLayout(filter_layout)
        
        # Sessions table
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "⭐", "Title", "Messages", "Project", "Last Updated", "Created"
        ])
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 30)
        self._table.setColumnWidth(2, 80)
        self._table.setColumnWidth(3, 150)
        self._table.setColumnWidth(4, 150)
        self._table.setColumnWidth(5, 150)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_context_menu)
        self._table.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E1E;
                border: 1px solid #3C3C3C;
                gridline-color: #3C3C3C;
            }
            QTableWidget::item {
                padding: 8px;
                color: #CCCCCC;
            }
            QTableWidget::item:selected {
                background-color: #094771;
            }
            QHeaderView::section {
                background-color: #252526;
                color: #CCCCCC;
                padding: 8px;
                border: 1px solid #3C3C3C;
            }
        """)
        layout.addWidget(self._table, stretch=1)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Left side buttons
        self._btn_import = QPushButton("📥 Import")
        self._btn_import.clicked.connect(self._on_import)
        button_layout.addWidget(self._btn_import)
        
        self._btn_export = QPushButton("📤 Export")
        self._btn_export.clicked.connect(self._on_export)
        button_layout.addWidget(self._btn_export)
        
        button_layout.addStretch()
        
        # Right side buttons
        self._btn_delete = QPushButton("🗑️ Delete")
        self._btn_delete.clicked.connect(self._on_delete_selected)
        button_layout.addWidget(self._btn_delete)
        
        self._btn_load = QPushButton("📂 Load Session")
        self._btn_load.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
        """)
        self._btn_load.clicked.connect(self._on_load_selected)
        button_layout.addWidget(self._btn_load)
        
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        button_layout.addWidget(self._btn_close)
        
        layout.addLayout(button_layout)
        
    def refresh_sessions(self):
        """Refresh the sessions list."""
        self._table.setRowCount(0)
        
        sessions = self._get_filtered_sessions()
        
        for session in sessions:
            self._add_session_row(session)
        
        # Update stats
        total = self._session_manager.get_session_count()
        self._stats_label.setText(f"{len(sessions)} of {total} sessions")
        
    def _get_filtered_sessions(self) -> List[ChatSession]:
        """Get sessions based on current filter."""
        filter_text = self._filter_combo.currentText()
        
        if filter_text == "Favorites":
            return self._session_manager.get_favorite_sessions()
        elif filter_text == "Recent (7 days)":
            # Filter sessions from last 7 days
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            return [s for s in self._session_manager.get_all_sessions() 
                   if s.updated_at > cutoff]
        elif filter_text == "Recent (30 days)":
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            return [s for s in self._session_manager.get_all_sessions() 
                   if s.updated_at > cutoff]
        else:
            return self._session_manager.get_all_sessions()
        
    def _add_session_row(self, session: ChatSession):
        """Add a session row to the table."""
        row = self._table.rowCount()
        self._table.insertRow(row)
        
        # Favorite
        fav_item = QTableWidgetItem("⭐" if session.is_favorite else "")
        fav_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 0, fav_item)
        
        # Title
        title_item = QTableWidgetItem(session.title)
        title_item.setData(Qt.ItemDataRole.UserRole, session.id)
        self._table.setItem(row, 1, title_item)
        
        # Message count
        count_item = QTableWidgetItem(str(len(session.messages)))
        count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 2, count_item)
        
        # Project
        project = ""
        if session.context.project_path:
            from pathlib import Path
            project = Path(session.context.project_path).name
        project_item = QTableWidgetItem(project)
        self._table.setItem(row, 3, project_item)
        
        # Last updated
        updated = self._format_datetime(session.updated_at)
        updated_item = QTableWidgetItem(updated)
        self._table.setItem(row, 4, updated_item)
        
        # Created
        created = self._format_datetime(session.created_at)
        created_item = QTableWidgetItem(created)
        self._table.setItem(row, 5, created_item)
        
    def _format_datetime(self, iso_string: str) -> str:
        """Format ISO datetime string."""
        try:
            dt = datetime.fromisoformat(iso_string)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return iso_string
            
    def _on_search(self, text: str):
        """Handle search."""
        if not text:
            self.refresh_sessions()
            return
        
        self._table.setRowCount(0)
        sessions = self._session_manager.search_sessions(text)
        
        for session in sessions:
            self._add_session_row(session)
            
        self._stats_label.setText(f"{len(sessions)} found")
        
    def _on_filter_changed(self):
        """Handle filter change."""
        self.refresh_sessions()
        
    def _on_context_menu(self, position):
        """Show context menu."""
        item = self._table.itemAt(position)
        if not item:
            return
        
        row = item.row()
        session_id = self._table.item(row, 1).data(Qt.ItemDataRole.UserRole)
        session = self._session_manager.get_session(session_id)
        if not session:
            return
        
        menu = QMenu(self)
        
        # Load action
        load_action = QAction("📂 Load Session", self)
        load_action.triggered.connect(lambda: self._load_session(session_id))
        menu.addAction(load_action)
        
        menu.addSeparator()
        
        # Favorite action
        fav_text = "⭐ Unfavorite" if session.is_favorite else "⭐ Favorite"
        fav_action = QAction(fav_text, self)
        fav_action.triggered.connect(lambda: self._toggle_favorite(session_id))
        menu.addAction(fav_action)
        
        # Rename action
        rename_action = QAction("✏️ Rename", self)
        rename_action.triggered.connect(lambda: self._rename_session(session_id))
        menu.addAction(rename_action)
        
        menu.addSeparator()
        
        # Duplicate action
        duplicate_action = QAction("📋 Duplicate", self)
        duplicate_action.triggered.connect(lambda: self._duplicate_session(session_id))
        menu.addAction(duplicate_action)
        
        # Export action
        export_action = QAction("💾 Export", self)
        export_action.triggered.connect(lambda: self._export_session(session_id))
        menu.addAction(export_action)
        
        menu.addSeparator()
        
        # Delete action
        delete_action = QAction("🗑️ Delete", self)
        delete_action.triggered.connect(lambda: self._delete_session(session_id))
        menu.addAction(delete_action)
        
        menu.exec(self._table.viewport().mapToGlobal(position))
        
    def _on_item_double_clicked(self, item: QTableWidgetItem):
        """Handle double click."""
        row = item.row()
        session_id = self._table.item(row, 1).data(Qt.ItemDataRole.UserRole)
        self._load_session(session_id)
        
    def _get_selected_session_id(self) -> Optional[str]:
        """Get the selected session ID."""
        selected = self._table.selectedItems()
        if selected:
            row = selected[0].row()
            return self._table.item(row, 1).data(Qt.ItemDataRole.UserRole)
        return None
        
    def _on_load_selected(self):
        """Handle load selected button."""
        session_id = self._get_selected_session_id()
        if session_id:
            self._load_session(session_id)
            
    def _load_session(self, session_id: str):
        """Load a session."""
        self._selected_session_id = session_id
        self.session_selected.emit(session_id)
        self.accept()
        
    def _on_delete_selected(self):
        """Handle delete selected button."""
        session_id = self._get_selected_session_id()
        if session_id:
            self._delete_session(session_id)
            
    def _delete_session(self, session_id: str):
        """Delete a session."""
        reply = QMessageBox.question(
            self,
            "Delete Session",
            "Are you sure you want to delete this session?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self._session_manager.delete_session(session_id):
                self.session_deleted.emit(session_id)
                self.refresh_sessions()
                
    def _toggle_favorite(self, session_id: str):
        """Toggle favorite status."""
        self._session_manager.toggle_favorite(session_id)
        self.refresh_sessions()
        
    def _rename_session(self, session_id: str):
        """Rename a session."""
        session = self._session_manager.get_session(session_id)
        if not session:
            return
        
        from PyQt6.QtWidgets import QInputDialog
        new_title, ok = QInputDialog.getText(
            self,
            "Rename Session",
            "Enter new title:",
            text=session.title
        )
        
        if ok and new_title:
            session.update_title(new_title)
            self._session_manager.save_session(session_id)
            self.refresh_sessions()
            
    def _duplicate_session(self, session_id: str):
        """Duplicate a session."""
        new_session = self._session_manager.duplicate_session(session_id)
        if new_session:
            self.refresh_sessions()
            
    def _on_import(self):
        """Handle import button."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Session",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            session = self._session_manager.import_session(file_path)
            if session:
                self.refresh_sessions()
                QMessageBox.information(self, "Import Successful", 
                                       f"Session '{session.title}' imported successfully.")
            else:
                QMessageBox.warning(self, "Import Failed", 
                                   "Failed to import session.")
                
    def _on_export(self):
        """Handle export button."""
        session_id = self._get_selected_session_id()
        if not session_id:
            QMessageBox.information(self, "Select Session", 
                                   "Please select a session to export.")
            return
        
        session = self._session_manager.get_session(session_id)
        if not session:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Session",
            f"{session.title}.json",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            if self._session_manager.export_session(session_id, file_path):
                QMessageBox.information(self, "Export Successful", 
                                       f"Session exported to:\n{file_path}")
            else:
                QMessageBox.warning(self, "Export Failed", 
                                   "Failed to export session.")
                
    def _export_session(self, session_id: str):
        """Export a specific session."""
        self._on_export()
        
    def get_selected_session_id(self) -> Optional[str]:
        """Get the selected session ID."""
        return self._selected_session_id
