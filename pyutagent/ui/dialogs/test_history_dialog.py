"""Test history dialog for viewing and managing test generation history."""

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QLineEdit, QGroupBox, QFormLayout, QMessageBox, QMenu,
    QAbstractItemView, QSplitter, QTextEdit, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush

from ...core.test_history import (
    get_test_history, save_test_history, TestGenerationRecord,
    TestGenerationStatus, TestHistory
)
from ..styles import get_style_manager
from ..components import get_notification_manager

logger = logging.getLogger(__name__)


class TestHistoryDialog(QDialog):
    """Dialog for viewing and managing test generation history."""
    
    regenerate_requested = pyqtSignal(str)  # source_file path
    
    def __init__(self, current_project: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Generation History")
        self.setMinimumSize(1000, 700)
        
        self.current_project = current_project
        self._style_manager = get_style_manager()
        self._notification_manager = get_notification_manager()
        self._history = get_test_history()
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header with stats
        self.stats_group = QGroupBox("Statistics")
        stats_layout = QHBoxLayout(self.stats_group)
        
        self.total_label = QLabel("Total: 0")
        self.success_label = QLabel("Successful: 0")
        self.failed_label = QLabel("Failed: 0")
        self.success_rate_label = QLabel("Success Rate: 0%")
        self.avg_coverage_label = QLabel("Avg Coverage: 0%")
        self.avg_duration_label = QLabel("Avg Duration: 0s")
        
        for label in [self.total_label, self.success_label, self.failed_label,
                      self.success_rate_label, self.avg_coverage_label, self.avg_duration_label]:
            stats_layout.addWidget(label)
        
        stats_layout.addStretch()
        layout.addWidget(self.stats_group)
        
        # Filter controls
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.status_filter = QComboBox()
        self.status_filter.addItem("All Status", "")
        self.status_filter.addItem("Success", TestGenerationStatus.SUCCESS.value)
        self.status_filter.addItem("Failed", TestGenerationStatus.FAILED.value)
        self.status_filter.addItem("Partial", TestGenerationStatus.PARTIAL.value)
        self.status_filter.addItem("Cancelled", TestGenerationStatus.CANCELLED.value)
        self.status_filter.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.status_filter)
        
        self.project_filter = QComboBox()
        self.project_filter.addItem("All Projects", "")
        self.project_filter.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.project_filter)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by file name...")
        self.search_box.textChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.search_box)
        
        filter_layout.addStretch()
        
        # Clear buttons
        self.clear_filtered_btn = QPushButton("Clear Filtered")
        self.clear_filtered_btn.clicked.connect(self.on_clear_filtered)
        filter_layout.addWidget(self.clear_filtered_btn)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.on_clear_all)
        filter_layout.addWidget(self.clear_all_btn)
        
        layout.addLayout(filter_layout)
        
        # Splitter for table and details
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Time", "File", "Status", "Coverage", "Iterations", "Duration", "Model", "ID"
        ])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.on_context_menu)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.setAlternatingRowColors(True)
        table_layout.addWidget(self.table)
        
        splitter.addWidget(table_container)
        
        # Details panel
        details_container = QWidget()
        details_container.setMaximumWidth(400)
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        
        details_group = QGroupBox("Details")
        self.details_layout = QFormLayout(details_group)
        
        self.detail_time = QLabel("-")
        self.detail_file = QLabel("-")
        self.detail_status = QLabel("-")
        self.detail_coverage = QLabel("-")
        self.detail_iterations = QLabel("-")
        self.detail_duration = QLabel("-")
        self.detail_model = QLabel("-")
        self.detail_tokens = QLabel("-")
        self.detail_error = QTextEdit()
        self.detail_error.setReadOnly(True)
        self.detail_error.setMaximumHeight(100)
        
        self.details_layout.addRow("Time:", self.detail_time)
        self.details_layout.addRow("File:", self.detail_file)
        self.details_layout.addRow("Status:", self.detail_status)
        self.details_layout.addRow("Coverage:", self.detail_coverage)
        self.details_layout.addRow("Iterations:", self.detail_iterations)
        self.details_layout.addRow("Duration:", self.detail_duration)
        self.details_layout.addRow("Model:", self.detail_model)
        self.details_layout.addRow("Tokens:", self.detail_tokens)
        self.details_layout.addRow("Error:", self.detail_error)
        
        details_layout.addWidget(details_group)
        
        # Action buttons
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.regenerate_btn = QPushButton("🔄 Regenerate Tests")
        self.regenerate_btn.clicked.connect(self.on_regenerate)
        self.regenerate_btn.setEnabled(False)
        actions_layout.addWidget(self.regenerate_btn)
        
        self.view_test_btn = QPushButton("📄 View Test File")
        self.view_test_btn.clicked.connect(self.on_view_test)
        self.view_test_btn.setEnabled(False)
        actions_layout.addWidget(self.view_test_btn)
        
        self.delete_btn = QPushButton("🗑️ Delete Record")
        self.delete_btn.clicked.connect(self.on_delete)
        self.delete_btn.setEnabled(False)
        actions_layout.addWidget(self.delete_btn)
        
        details_layout.addWidget(actions_group)
        details_layout.addStretch()
        
        splitter.addWidget(details_container)
        splitter.setSizes([600, 300])
        
        layout.addWidget(splitter)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def load_data(self):
        """Load data into the dialog."""
        # Update project filter
        projects = set()
        for record in self._history.records:
            projects.add(record.project_path)
        
        self.project_filter.clear()
        self.project_filter.addItem("All Projects", "")
        for project in sorted(projects):
            project_name = Path(project).name if project else "Unknown"
            self.project_filter.addItem(project_name, project)
        
        # Update stats
        self.update_stats()
        
        # Load table
        self.refresh_table()
    
    def update_stats(self):
        """Update statistics display."""
        stats = self._history.get_stats()
        
        self.total_label.setText(f"📊 Total: {stats['total']}")
        self.success_label.setText(f"✅ Successful: {stats.get('successful', 0)}")
        self.failed_label.setText(f"❌ Failed: {stats.get('failed', 0)}")
        self.success_rate_label.setText(f"📈 Success Rate: {stats['success_rate']:.1%}")
        self.avg_coverage_label.setText(f"🎯 Avg Coverage: {stats['avg_coverage']:.1%}")
        self.avg_duration_label.setText(f"⏱️ Avg Duration: {stats['avg_duration']:.1f}s")
    
    def refresh_table(self):
        """Refresh the table with current filter."""
        status_filter = self.status_filter.currentData()
        project_filter = self.project_filter.currentData()
        search_text = self.search_box.text().lower()
        
        filtered_records = []
        for record in self._history.records:
            # Status filter
            if status_filter and record.status != status_filter:
                continue
            
            # Project filter
            if project_filter and record.project_path != project_filter:
                continue
            
            # Search filter
            if search_text and search_text not in record.source_file_name.lower():
                continue
            
            filtered_records.append(record)
        
        self.table.setRowCount(len(filtered_records))
        
        for i, record in enumerate(filtered_records):
            # Time
            self.table.setItem(i, 0, QTableWidgetItem(record.formatted_timestamp))
            
            # File
            self.table.setItem(i, 1, QTableWidgetItem(record.source_file_name))
            
            # Status
            status_item = QTableWidgetItem(record.status.capitalize())
            if record.status == TestGenerationStatus.SUCCESS.value:
                status_item.setForeground(QBrush(QColor("#4CAF50")))
            elif record.status == TestGenerationStatus.FAILED.value:
                status_item.setForeground(QBrush(QColor("#F44336")))
            elif record.status == TestGenerationStatus.PARTIAL.value:
                status_item.setForeground(QBrush(QColor("#FF9800")))
            self.table.setItem(i, 2, status_item)
            
            # Coverage
            coverage_text = f"{record.coverage:.1%}" if record.coverage > 0 else "-"
            self.table.setItem(i, 3, QTableWidgetItem(coverage_text))
            
            # Iterations
            self.table.setItem(i, 4, QTableWidgetItem(str(record.iterations)))
            
            # Duration
            self.table.setItem(i, 5, QTableWidgetItem(record.formatted_duration))
            
            # Model
            model_text = record.model_used[:20] + "..." if len(record.model_used) > 20 else record.model_used
            self.table.setItem(i, 6, QTableWidgetItem(model_text))
            
            # ID (hidden)
            id_item = QTableWidgetItem(record.id)
            id_item.setData(Qt.ItemDataRole.UserRole, record)
            self.table.setItem(i, 7, id_item)
        
        self.table.resizeColumnsToContents()
    
    def on_filter_changed(self):
        """Handle filter changes."""
        self.refresh_table()
    
    def on_selection_changed(self):
        """Handle table selection change."""
        selected = self.table.selectedItems()
        if not selected:
            self.clear_details()
            self.regenerate_btn.setEnabled(False)
            self.view_test_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            return
        
        row = selected[0].row()
        id_item = self.table.item(row, 7)
        record = id_item.data(Qt.ItemDataRole.UserRole)
        
        if record:
            self.show_details(record)
            self.regenerate_btn.setEnabled(True)
            self.view_test_btn.setEnabled(bool(record.test_file and Path(record.test_file).exists()))
            self.delete_btn.setEnabled(True)
    
    def show_details(self, record: TestGenerationRecord):
        """Show record details."""
        self.detail_time.setText(record.formatted_timestamp)
        self.detail_file.setText(record.source_file_name)
        
        status_text = record.status.capitalize()
        if record.status == TestGenerationStatus.SUCCESS.value:
            status_text = f"<span style='color: #4CAF50;'>✅ {status_text}</span>"
        elif record.status == TestGenerationStatus.FAILED.value:
            status_text = f"<span style='color: #F44336;'>❌ {status_text}</span>"
        elif record.status == TestGenerationStatus.PARTIAL.value:
            status_text = f"<span style='color: #FF9800;'>⚠️ {status_text}</span>"
        self.detail_status.setText(status_text)
        
        self.detail_coverage.setText(f"{record.coverage:.1%} / {record.target_coverage:.1%}")
        self.detail_iterations.setText(str(record.iterations))
        self.detail_duration.setText(record.formatted_duration)
        self.detail_model.setText(record.model_used or "Unknown")
        self.detail_tokens.setText(f"{record.total_tokens:,} ({record.prompt_tokens:,} / {record.completion_tokens:,})")
        self.detail_error.setText(record.error_message or "None")
    
    def clear_details(self):
        """Clear details panel."""
        self.detail_time.setText("-")
        self.detail_file.setText("-")
        self.detail_status.setText("-")
        self.detail_coverage.setText("-")
        self.detail_iterations.setText("-")
        self.detail_duration.setText("-")
        self.detail_model.setText("-")
        self.detail_tokens.setText("-")
        self.detail_error.setText("")
    
    def on_context_menu(self, position):
        """Show context menu."""
        item = self.table.itemAt(position)
        if not item:
            return
        
        row = item.row()
        id_item = self.table.item(row, 7)
        record = id_item.data(Qt.ItemDataRole.UserRole)
        
        if not record:
            return
        
        menu = QMenu(self)
        
        regenerate_action = menu.addAction("🔄 Regenerate Tests")
        regenerate_action.triggered.connect(lambda: self.regenerate_requested.emit(record.source_file))
        
        if record.test_file and Path(record.test_file).exists():
            view_action = menu.addAction("📄 View Test File")
            view_action.triggered.connect(lambda: self._view_file(record.test_file))
        
        menu.addSeparator()
        
        delete_action = menu.addAction("🗑️ Delete Record")
        delete_action.triggered.connect(lambda: self._delete_record(record.id))
        
        menu.exec(self.table.viewport().mapToGlobal(position))
    
    def on_regenerate(self):
        """Handle regenerate button."""
        selected = self.table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        id_item = self.table.item(row, 7)
        record = id_item.data(Qt.ItemDataRole.UserRole)
        
        if record:
            self.regenerate_requested.emit(record.source_file)
            self._notification_manager.show_info(f"Regenerating tests for {record.source_file_name}")
            self.accept()
    
    def on_view_test(self):
        """Handle view test button."""
        selected = self.table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        id_item = self.table.item(row, 7)
        record = id_item.data(Qt.ItemDataRole.UserRole)
        
        if record and record.test_file:
            self._view_file(record.test_file)
    
    def _view_file(self, file_path: str):
        """View a file."""
        try:
            import subprocess
            import os
            
            if Path(file_path).exists():
                if os.name == 'nt':  # Windows
                    subprocess.run(['notepad', file_path])
                else:  # macOS/Linux
                    subprocess.run(['open', file_path])
            else:
                self._notification_manager.show_warning("Test file not found")
        except Exception as e:
            logger.error(f"Failed to open file: {e}")
            self._notification_manager.show_error(f"Failed to open file: {e}")
    
    def on_delete(self):
        """Handle delete button."""
        selected = self.table.selectedItems()
        if not selected:
            return
        
        row = selected[0].row()
        id_item = self.table.item(row, 7)
        record = id_item.data(Qt.ItemDataRole.UserRole)
        
        if record:
            self._delete_record(record.id)
    
    def _delete_record(self, record_id: str):
        """Delete a record."""
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            "Are you sure you want to delete this record?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self._history.delete_record(record_id):
                save_test_history(self._history)
                self.refresh_table()
                self.update_stats()
                self.clear_details()
                self._notification_manager.show_success("Record deleted")
            else:
                self._notification_manager.show_error("Failed to delete record")
    
    def on_clear_filtered(self):
        """Clear filtered records."""
        status_filter = self.status_filter.currentData()
        project_filter = self.project_filter.currentData()
        search_text = self.search_box.text().lower()
        
        # Count records to delete
        to_delete = []
        for record in self._history.records:
            if status_filter and record.status != status_filter:
                continue
            if project_filter and record.project_path != project_filter:
                continue
            if search_text and search_text not in record.source_file_name.lower():
                continue
            to_delete.append(record.id)
        
        if not to_delete:
            self._notification_manager.show_info("No records to delete")
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            f"Are you sure you want to delete {len(to_delete)} filtered records?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            for record_id in to_delete:
                self._history.delete_record(record_id)
            
            save_test_history(self._history)
            self.refresh_table()
            self.update_stats()
            self._notification_manager.show_success(f"Deleted {len(to_delete)} records")
    
    def on_clear_all(self):
        """Clear all records."""
        if not self._history.records:
            self._notification_manager.show_info("No records to delete")
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            f"Are you sure you want to delete all {len(self._history.records)} records?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._history.clear()
            save_test_history(self._history)
            self.refresh_table()
            self.update_stats()
            self.clear_details()
            self._notification_manager.show_success("All records cleared")
