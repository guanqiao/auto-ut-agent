"""Batch generate dialog for GUI."""

import logging
import time
from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QSpinBox, QCheckBox, QGroupBox, QAbstractItemView, QMessageBox,
    QWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

logger = logging.getLogger(__name__)


class BatchGenerateWorker(QThread):
    """Worker thread for batch test generation."""
    
    progress_updated = pyqtSignal(str, str, float, int, float, str)
    file_completed = pyqtSignal(str, bool, float, int, str, float)
    all_completed = pyqtSignal(int, int, float)
    error = pyqtSignal(str)
    
    def __init__(
        self,
        llm_client,
        project_path: str,
        files: List[str],
        parallel_workers: int = 1,
        timeout: int = 300,
        coverage_target: int = 80,
        max_iterations: int = 10,
        continue_on_error: bool = True
    ):
        super().__init__()
        self.llm_client = llm_client
        self.project_path = project_path
        self.files = files
        self.parallel_workers = parallel_workers
        self.timeout = timeout
        self.coverage_target = coverage_target
        self.max_iterations = max_iterations
        self.continue_on_error = continue_on_error
        self._stop_requested = False
    
    def run(self):
        """Run batch generation."""
        try:
            from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
            
            config = BatchConfig(
                parallel_workers=self.parallel_workers,
                timeout_per_file=self.timeout,
                continue_on_error=self.continue_on_error,
                coverage_target=self.coverage_target,
                max_iterations=self.max_iterations
            )
            
            def progress_callback(batch_progress):
                if self._stop_requested:
                    return
                current = batch_progress.current_file
                status = batch_progress.current_status
                self.progress_updated.emit(
                    current, status, 
                    batch_progress.progress_percent,
                    batch_progress.completed_files + batch_progress.failed_files,
                    0.0, ""
                )
            
            generator = BatchGenerator(
                llm_client=self.llm_client,
                project_path=self.project_path,
                config=config,
                progress_callback=progress_callback
            )
            
            self._stop_requested = False
            
            result = generator.generate_all_sync(self.files)
            
            for file_result in result.results:
                self.file_completed.emit(
                    file_result.file_path,
                    file_result.success,
                    file_result.coverage,
                    file_result.iterations,
                    file_result.test_file or "",
                    file_result.duration
                )
            
            self.all_completed.emit(
                result.success_count,
                result.failed_count,
                result.total_duration
            )
            
        except Exception as e:
            logger.exception("Batch generation error")
            self.error.emit(str(e))
    
    def stop(self):
        """Stop the generation."""
        self._stop_requested = True


class BatchGenerateDialog(QDialog):
    """Dialog for batch test generation with parallel execution support."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Generate Tests")
        self.setMinimumSize(900, 700)
        
        self.llm_client = None
        self.project_path = ""
        self.java_files: List[str] = []
        self.worker: Optional[BatchGenerateWorker] = None
        self._is_running = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        
        config_group = QGroupBox("Configuration")
        config_layout = QHBoxLayout(config_group)
        
        parallel_label = QLabel("Parallel Workers:")
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(0, 20)
        self.parallel_spin.setValue(1)
        self.parallel_spin.setToolTip("Number of parallel workers (0 = unlimited)")
        config_layout.addWidget(parallel_label)
        config_layout.addWidget(self.parallel_spin)
        
        config_layout.addSpacing(20)
        
        timeout_label = QLabel("Timeout (s):")
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(60, 3600)
        self.timeout_spin.setValue(300)
        self.timeout_spin.setSingleStep(60)
        config_layout.addWidget(timeout_label)
        config_layout.addWidget(self.timeout_spin)
        
        config_layout.addSpacing(20)
        
        coverage_label = QLabel("Coverage Target (%):")
        self.coverage_spin = QSpinBox()
        self.coverage_spin.setRange(0, 100)
        self.coverage_spin.setValue(80)
        config_layout.addWidget(coverage_label)
        config_layout.addWidget(self.coverage_spin)
        
        config_layout.addSpacing(20)
        
        self.continue_check = QCheckBox("Continue on error")
        self.continue_check.setChecked(True)
        config_layout.addWidget(self.continue_check)
        
        config_layout.addStretch()
        layout.addWidget(config_group)
        
        files_group = QGroupBox("Java Files")
        files_layout = QVBoxLayout(files_group)
        
        select_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        self.file_count_label = QLabel("0 files selected")
        select_layout.addWidget(self.select_all_btn)
        select_layout.addWidget(self.deselect_all_btn)
        select_layout.addStretch()
        select_layout.addWidget(self.file_count_label)
        files_layout.addLayout(select_layout)
        
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(6)
        self.files_table.setHorizontalHeaderLabels([
            "Select", "File", "Status", "Coverage", "Iterations", "Time"
        ])
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.files_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.files_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.files_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.files_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.files_table.setColumnWidth(0, 50)
        self.files_table.setColumnWidth(2, 100)
        self.files_table.setColumnWidth(3, 80)
        self.files_table.setColumnWidth(4, 80)
        self.files_table.setColumnWidth(5, 80)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        files_layout.addWidget(self.files_table)
        
        layout.addWidget(files_group)
        
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        self.time_label = QLabel("Elapsed: 0s")
        status_layout.addWidget(self.time_label)
        progress_layout.addLayout(status_layout)
        
        layout.addWidget(progress_group)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_generation)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 20px;")
        button_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_generation)
        self.pause_btn.setEnabled(False)
        button_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px 20px;")
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self._start_time = 0
        self._timer_id = None
    
    def set_files(self, project_path: str, java_files: List[str], llm_client):
        """Set the files to generate tests for."""
        self.project_path = project_path
        self.java_files = java_files
        self.llm_client = llm_client
        
        self.files_table.setRowCount(len(java_files))
        for i, file_path in enumerate(java_files):
            check_item = QTableWidgetItem()
            check_item.setCheckState(Qt.CheckState.Checked)
            check_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            self.files_table.setItem(i, 0, check_item)
            
            file_name = Path(file_path).name
            self.files_table.setItem(i, 1, QTableWidgetItem(file_name))
            self.files_table.setItem(i, 2, QTableWidgetItem("Pending"))
            self.files_table.setItem(i, 3, QTableWidgetItem("-"))
            self.files_table.setItem(i, 4, QTableWidgetItem("-"))
            self.files_table.setItem(i, 5, QTableWidgetItem("-"))
            
            self.files_table.item(i, 1).setData(Qt.ItemDataRole.UserRole, file_path)
        
        self.update_file_count()
        self.files_table.itemChanged.connect(self.on_item_changed)
    
    def on_item_changed(self, item):
        """Handle item changes."""
        if item.column() == 0:
            self.update_file_count()
    
    def update_file_count(self):
        """Update the file count label."""
        count = sum(
            1 for i in range(self.files_table.rowCount())
            if self.files_table.item(i, 0).checkState() == Qt.CheckState.Checked
        )
        self.file_count_label.setText(f"{count} files selected")
    
    def select_all(self):
        """Select all files."""
        for i in range(self.files_table.rowCount()):
            self.files_table.item(i, 0).setCheckState(Qt.CheckState.Checked)
    
    def deselect_all(self):
        """Deselect all files."""
        for i in range(self.files_table.rowCount()):
            self.files_table.item(i, 0).setCheckState(Qt.CheckState.Unchecked)
    
    def get_selected_files(self) -> List[str]:
        """Get list of selected files."""
        files = []
        for i in range(self.files_table.rowCount()):
            if self.files_table.item(i, 0).checkState() == Qt.CheckState.Checked:
                file_path = self.files_table.item(i, 1).data(Qt.ItemDataRole.UserRole)
                files.append(file_path)
        return files
    
    def start_generation(self):
        """Start batch generation."""
        selected_files = self.get_selected_files()
        
        if not selected_files:
            QMessageBox.warning(self, "Warning", "Please select at least one file")
            return
        
        if not self.llm_client:
            QMessageBox.warning(self, "Warning", "LLM client not configured")
            return
        
        self._is_running = True
        self._start_time = time.time()
        
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.close_btn.setEnabled(False)
        self.parallel_spin.setEnabled(False)
        self.timeout_spin.setEnabled(False)
        self.coverage_spin.setEnabled(False)
        self.continue_check.setEnabled(False)
        
        for i in range(self.files_table.rowCount()):
            self.files_table.item(i, 0).setFlags(Qt.ItemFlag.NoItemFlags)
            self.files_table.item(i, 2).setText("Pending")
            self.files_table.item(i, 2).setForeground(QColor("#666666"))
            self.files_table.item(i, 3).setText("-")
            self.files_table.item(i, 4).setText("-")
            self.files_table.item(i, 5).setText("-")
        
        self._timer_id = self.startTimer(1000)
        
        self.worker = BatchGenerateWorker(
            llm_client=self.llm_client,
            project_path=self.project_path,
            files=selected_files,
            parallel_workers=self.parallel_spin.value(),
            timeout=self.timeout_spin.value(),
            coverage_target=self.coverage_spin.value(),
            max_iterations=10,
            continue_on_error=self.continue_check.isChecked()
        )
        
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.file_completed.connect(self.on_file_completed)
        self.worker.all_completed.connect(self.on_all_completed)
        self.worker.error.connect(self.on_error)
        
        self.worker.start()
        self.status_label.setText("Running...")
        self.status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
    
    def pause_generation(self):
        """Pause/resume generation."""
        if self.worker and self.worker.isRunning():
            if self.worker._stop_requested:
                self.worker._stop_requested = False
                self.pause_btn.setText("Pause")
                self.status_label.setText("Running...")
                self.status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            else:
                self.worker._stop_requested = True
                self.pause_btn.setText("Resume")
                self.status_label.setText("Paused")
                self.status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
    
    def stop_generation(self):
        """Stop generation."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        
        self._is_running = False
        self.on_generation_finished()
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
    
    def timerEvent(self, event):
        """Handle timer event."""
        if self._is_running:
            elapsed = time.time() - self._start_time
            self.time_label.setText(f"Elapsed: {int(elapsed)}s")
    
    def on_progress_updated(self, current_file, status, progress, completed, duration, error):
        """Handle progress update."""
        self.progress_bar.setValue(int(progress))
        
        for i in range(self.files_table.rowCount()):
            item = self.files_table.item(i, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == current_file:
                self.files_table.item(i, 2).setText(status)
                self.files_table.item(i, 2).setForeground(QColor("#2196F3"))
                break
    
    def on_file_completed(self, file_path, success, coverage, iterations, test_file, duration):
        """Handle file completion."""
        for i in range(self.files_table.rowCount()):
            item = self.files_table.item(i, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == file_path:
                if success:
                    self.files_table.item(i, 2).setText("Done")
                    self.files_table.item(i, 2).setForeground(QColor("#4CAF50"))
                    self.files_table.item(i, 3).setText(f"{coverage:.1f}%")
                    self.files_table.item(i, 3).setForeground(QColor("#4CAF50"))
                else:
                    self.files_table.item(i, 2).setText("Failed")
                    self.files_table.item(i, 2).setForeground(QColor("#f44336"))
                    self.files_table.item(i, 3).setText("-")
                    self.files_table.item(i, 3).setForeground(QColor("#f44336"))
                
                self.files_table.item(i, 4).setText(str(iterations) if iterations > 0 else "-")
                self.files_table.item(i, 5).setText(f"{duration:.1f}s")
                break
    
    def on_all_completed(self, success_count, failed_count, total_duration):
        """Handle all completed."""
        self._is_running = False
        self.on_generation_finished()
        
        total = success_count + failed_count
        if failed_count == 0:
            self.status_label.setText(f"Completed: {success_count}/{total} files")
            self.status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif success_count == 0:
            self.status_label.setText(f"Failed: {failed_count}/{total} files")
            self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
        else:
            self.status_label.setText(f"Completed: {success_count} success, {failed_count} failed")
            self.status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        
        self.progress_bar.setValue(100)
    
    def on_error(self, error_message):
        """Handle error."""
        self._is_running = False
        self.on_generation_finished()
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
        QMessageBox.critical(self, "Error", error_message)
    
    def on_generation_finished(self):
        """Handle generation finished."""
        if self._timer_id:
            self.killTimer(self._timer_id)
            self._timer_id = None
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.stop_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.parallel_spin.setEnabled(True)
        self.timeout_spin.setEnabled(True)
        self.coverage_spin.setEnabled(True)
        self.continue_check.setEnabled(True)
        
        for i in range(self.files_table.rowCount()):
            item = self.files_table.item(i, 0)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
    
    def closeEvent(self, event):
        """Handle close event."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm",
                "Generation is in progress. Stop and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.stop()
                self.worker.wait(2000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
