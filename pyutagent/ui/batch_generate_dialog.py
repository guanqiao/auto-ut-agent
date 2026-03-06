"""Batch generate dialog for GUI."""

import logging
import time
from pathlib import Path
from typing import List, Optional

from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QSpinBox, QCheckBox, QGroupBox, QAbstractItemView, QMessageBox,
    QWidget, QTextEdit, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

from pyutagent.core.config import get_settings
from pyutagent.ui.log_handler import LogEmitter

logger = logging.getLogger(__name__)


class BatchGenerateWorker(QThread):
    """Worker thread for batch test generation."""
    
    progress_updated = pyqtSignal(str, str, float, int, float, str, float, float)
    file_completed = pyqtSignal(str, bool, float, int, str, float, int, int, int)
    all_completed = pyqtSignal(int, int, float, object, int, int, int)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str, str)
    
    def __init__(
        self,
        llm_client,
        project_path: str,
        files: List[str],
        parallel_workers: int = 1,
        timeout: int = 300,
        coverage_target: int = 80,
        max_iterations: int = 2,
        continue_on_error: bool = True,
        defer_compilation: bool = False,
        compile_only_at_end: bool = False,
        incremental_mode: bool = False,
        skip_test_analysis: bool = False
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
        self.defer_compilation = defer_compilation
        self.compile_only_at_end = compile_only_at_end
        self.incremental_mode = incremental_mode
        self.skip_test_analysis = skip_test_analysis
        self._stop_requested = False
        self._log_emitter: Optional[LogEmitter] = None
    
    def run(self):
        """Run batch generation."""
        self._log_emitter = LogEmitter()
        self._log_emitter.log_message.connect(self._on_log)
        self._log_emitter.install_handler('pyutagent')
        
        try:
            from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
            
            config = BatchConfig(
                parallel_workers=self.parallel_workers,
                timeout_per_file=self.timeout,
                continue_on_error=self.continue_on_error,
                coverage_target=self.coverage_target,
                max_iterations=self.max_iterations,
                defer_compilation=self.defer_compilation,
                compile_only_at_end=self.compile_only_at_end,
                incremental_mode=self.incremental_mode,
                skip_test_analysis=self.skip_test_analysis
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
                    batch_progress.current_coverage,
                    batch_progress.coverage_source,
                    batch_progress.coverage_confidence,
                    0.0  # elapsed_time - not used in this context
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
                    file_result.duration,
                    file_result.preserved_tests,
                    file_result.new_tests,
                    file_result.fixed_tests
                )
            
            self.all_completed.emit(
                result.success_count,
                result.failed_count,
                result.total_duration,
                result.compilation_result,
                result.total_preserved_tests,
                result.total_new_tests,
                result.total_fixed_tests
            )
            
        except Exception as e:
            logger.exception("Batch generation error")
            self.error.emit(str(e))
        finally:
            if self._log_emitter:
                self._log_emitter.uninstall_handler('pyutagent')
    
    def _on_log(self, message: str, level: str):
        """Handle log message from LogEmitter."""
        self.log_message.emit(message, level)
    
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
        
        # Left column
        left_col = QVBoxLayout()
        
        parallel_label = QLabel("Parallel Workers:")
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(0, 20)
        self.parallel_spin.setValue(1)
        self.parallel_spin.setToolTip("Number of parallel workers (0 = unlimited)")
        parallel_layout = QHBoxLayout()
        parallel_layout.addWidget(parallel_label)
        parallel_layout.addWidget(self.parallel_spin)
        parallel_layout.addStretch()
        left_col.addLayout(parallel_layout)
        
        timeout_label = QLabel("Timeout (s):")
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(60, 3600)
        self.timeout_spin.setValue(300)
        self.timeout_spin.setSingleStep(60)
        timeout_layout = QHBoxLayout()
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self.timeout_spin)
        timeout_layout.addStretch()
        left_col.addLayout(timeout_layout)
        
        coverage_label = QLabel("Coverage Target (%):")
        self.coverage_spin = QSpinBox()
        self.coverage_spin.setRange(0, 100)
        self.coverage_spin.setValue(80)
        coverage_layout = QHBoxLayout()
        coverage_layout.addWidget(coverage_label)
        coverage_layout.addWidget(self.coverage_spin)
        coverage_layout.addStretch()
        left_col.addLayout(coverage_layout)
        
        config_layout.addLayout(left_col)
        
        config_layout.addSpacing(20)
        
        # Right column
        right_col = QVBoxLayout()
        
        self.continue_check = QCheckBox("Continue on error")
        self.continue_check.setChecked(True)
        self.continue_check.setToolTip("Continue generating tests for other files if one fails")
        right_col.addWidget(self.continue_check)
        
        self.incremental_check = QCheckBox("Incremental Mode")
        self.incremental_check.setChecked(False)
        self.incremental_check.setToolTip("Preserve existing passing tests when regenerating")
        self.incremental_check.stateChanged.connect(self.on_incremental_changed)
        right_col.addWidget(self.incremental_check)
        
        self.skip_analysis_check = QCheckBox("Skip Test Analysis")
        self.skip_analysis_check.setChecked(False)
        self.skip_analysis_check.setToolTip("Skip running existing tests, just analyze file content")
        self.skip_analysis_check.setEnabled(False)
        right_col.addWidget(self.skip_analysis_check)
        
        # Compilation strategy group
        compile_strategy_group = QGroupBox("Compilation Strategy")
        compile_strategy_layout = QVBoxLayout(compile_strategy_group)
        
        self.standard_radio = QPushButton("Standard Mode")
        self.standard_radio.setCheckable(True)
        self.standard_radio.setChecked(True)
        self.standard_radio.setToolTip("Generate → Compile → Test → Fix for each file")
        self.standard_radio.clicked.connect(lambda: self.on_compilation_strategy_changed("standard"))
        compile_strategy_layout.addWidget(self.standard_radio)
        
        self.defer_radio = QPushButton("Defer Compilation")
        self.defer_radio.setCheckable(True)
        self.defer_radio.setChecked(False)
        self.defer_radio.setToolTip("Generate all tests first, then compile all at once")
        self.defer_radio.clicked.connect(lambda: self.on_compilation_strategy_changed("defer"))
        compile_strategy_layout.addWidget(self.defer_radio)
        
        self.fast_radio = QPushButton("Fast Mode (Compile Only at End)")
        self.fast_radio.setCheckable(True)
        self.fast_radio.setChecked(False)
        self.fast_radio.setToolTip("Generate all without verification, compile once at the end")
        self.fast_radio.clicked.connect(lambda: self.on_compilation_strategy_changed("fast"))
        compile_strategy_layout.addWidget(self.fast_radio)
        
        # Strategy description label
        self.strategy_desc_label = QLabel(
            "Standard: Verify each file immediately (recommended for quality)"
        )
        self.strategy_desc_label.setWordWrap(True)
        self.strategy_desc_label.setStyleSheet("color: #666; font-size: 11px;")
        compile_strategy_layout.addWidget(self.strategy_desc_label)
        
        right_col.addWidget(compile_strategy_group)
        right_col.addStretch()
        
        config_layout.addLayout(right_col)
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
        
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        
        log_header_layout = QHBoxLayout()
        log_header_layout.addStretch()
        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.setMaximumWidth(60)
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_header_layout.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_header_layout)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        self.log_area.setMaximumHeight(200)
        log_layout.addWidget(self.log_area)
        
        layout.addWidget(log_group)
        
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
    
    def on_compilation_strategy_changed(self, strategy: str):
        """Handle compilation strategy change."""
        self.standard_radio.setChecked(strategy == "standard")
        self.defer_radio.setChecked(strategy == "defer")
        self.fast_radio.setChecked(strategy == "fast")
        
        if strategy == "standard":
            self.strategy_desc_label.setText(
                "Standard: Verify each file immediately (recommended for quality)"
            )
        elif strategy == "defer":
            self.strategy_desc_label.setText(
                "Defer: Generate all → Compile all (faster, errors shown at end)"
            )
        elif strategy == "fast":
            self.strategy_desc_label.setText(
                "Fast: Generate all without checks → Compile once (fastest, manual fix may be needed)"
            )
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message to the log area."""
        self.on_log_message(message, level)
    
    def on_incremental_changed(self, state):
        """Handle incremental mode checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        self.skip_analysis_check.setEnabled(enabled)
        if enabled:
            self.log("Incremental mode enabled: Existing passing tests will be preserved")
        else:
            self.skip_analysis_check.setChecked(False)
            self.log("Incremental mode disabled: All tests will be regenerated")
    
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
        
        # Determine compilation strategy
        defer_compilation = False
        compile_only_at_end = False
        
        if self.defer_radio.isChecked():
            defer_compilation = True
            compile_only_at_end = False
        elif self.fast_radio.isChecked():
            defer_compilation = True
            compile_only_at_end = True
        # else: standard mode (both False)
        
        settings = get_settings()
        self.worker = BatchGenerateWorker(
            llm_client=self.llm_client,
            project_path=self.project_path,
            files=selected_files,
            parallel_workers=self.parallel_spin.value(),
            timeout=self.timeout_spin.value(),
            coverage_target=self.coverage_spin.value(),
            max_iterations=settings.coverage.max_iterations,
            continue_on_error=self.continue_check.isChecked(),
            defer_compilation=defer_compilation,
            compile_only_at_end=compile_only_at_end,
            incremental_mode=self.incremental_check.isChecked(),
            skip_test_analysis=self.skip_analysis_check.isChecked()
        )
        
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.file_completed.connect(self.on_file_completed)
        self.worker.all_completed.connect(self.on_all_completed)
        self.worker.error.connect(self.on_error)
        self.worker.log_message.connect(self.on_log_message)
        
        self.worker.start()
        self.status_label.setText("Running...")
        self.status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        
        # Log incremental mode status
        if self.incremental_check.isChecked():
            self.on_log_message("🔄 Incremental mode: ENABLED (will preserve existing passing tests)", "INFO")
            if self.skip_analysis_check.isChecked():
                self.on_log_message("⚡ Skip test analysis: ENABLED (will not run existing tests)", "INFO")
    
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
    
    def on_log_message(self, message: str, level: str):
        """Handle log message from worker."""
        colors = {
            "DEBUG": "#808080",
            "INFO": "#4FC1FF",
            "WARNING": "#FFCC00",
            "ERROR": "#FF6B6B",
            "CRITICAL": "#FF0000",
        }
        color = colors.get(level.upper(), "#d4d4d4")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] [{level.upper()}] {message}"
        
        self.log_area.append(f'<span style="color: {color};">{formatted_message}</span>')
        
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """Clear log area."""
        self.log_area.clear()
    
    def on_progress_updated(self, current_file, status, progress, completed, coverage, coverage_source, coverage_confidence, elapsed_time):
        """Handle progress update."""
        self.progress_bar.setValue(int(progress))
        
        for i in range(self.files_table.rowCount()):
            item = self.files_table.item(i, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == current_file:
                self.files_table.item(i, 2).setText(status)
                self.files_table.item(i, 2).setForeground(QColor("#2196F3"))
                
                if coverage > 0:
                    if coverage_source == "llm_estimated":
                        coverage_text = f"{coverage:.1f}% (LLM)"
                    else:
                        coverage_text = f"{coverage:.1f}%"
                    self.files_table.item(i, 3).setText(coverage_text)
                    self.files_table.item(i, 3).setForeground(QColor("#4CAF50"))
                break
    
    def on_file_completed(self, file_path, success, coverage, iterations, test_file, duration, preserved_tests=0, new_tests=0, fixed_tests=0):
        """Handle file completion."""
        for i in range(self.files_table.rowCount()):
            item = self.files_table.item(i, 1)
            if item and item.data(Qt.ItemDataRole.UserRole) == file_path:
                if success:
                    self.files_table.item(i, 2).setText("Done")
                    self.files_table.item(i, 2).setForeground(QColor("#4CAF50"))
                    
                    if coverage > 0:
                        self.files_table.item(i, 3).setText(f"{coverage:.1f}%")
                        self.files_table.item(i, 3).setForeground(QColor("#4CAF50"))
                    else:
                        self.files_table.item(i, 3).setText("-")
                    
                    if self.incremental_check.isChecked() and (preserved_tests > 0 or new_tests > 0 or fixed_tests > 0):
                        stats_msg = f"📊 Incremental stats for {Path(file_path).name}:"
                        if preserved_tests > 0:
                            stats_msg += f" preserved={preserved_tests}"
                        if new_tests > 0:
                            stats_msg += f" new={new_tests}"
                        if fixed_tests > 0:
                            stats_msg += f" fixed={fixed_tests}"
                        self.on_log_message(stats_msg, "INFO")
                else:
                    self.files_table.item(i, 2).setText("Failed")
                    self.files_table.item(i, 2).setForeground(QColor("#f44336"))
                    self.files_table.item(i, 3).setText("-")
                    self.files_table.item(i, 3).setForeground(QColor("#f44336"))
                
                self.files_table.item(i, 4).setText(str(iterations) if iterations > 0 else "-")
                self.files_table.item(i, 5).setText(f"{duration:.1f}s")
                break
    
    def on_all_completed(self, success_count, failed_count, total_duration, compilation_result, total_preserved_tests=0, total_new_tests=0, total_fixed_tests=0):
        """Handle all completed."""
        self._is_running = False
        self.on_generation_finished()
        
        total = success_count + failed_count
        
        # Build status message
        status_parts = []
        
        if compilation_result:
            # Two-phase mode completed
            if compilation_result.success:
                status_parts.append(f"Generated: {success_count}/{total}")
                status_parts.append(f"Compiled: ✓ Success")
                self.status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
            else:
                status_parts.append(f"Generated: {success_count}/{total}")
                status_parts.append(f"Compiled: ✗ {compilation_result.failed_files} failed")
                self.status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
            
            # Show compilation time
            status_parts.append(f"Compile time: {compilation_result.duration:.1f}s")
        else:
            # Standard mode
            if failed_count == 0:
                status_parts.append(f"Completed: {success_count}/{total} files")
                self.status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
            elif success_count == 0:
                status_parts.append(f"Failed: {failed_count}/{total} files")
                self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
            else:
                status_parts.append(f"Completed: {success_count} success, {failed_count} failed")
                self.status_label.setStyleSheet("font-weight: bold; color: #FF9800;")
        
        # Add incremental mode statistics to status
        if self.incremental_check.isChecked() and (total_preserved_tests > 0 or total_new_tests > 0 or total_fixed_tests > 0):
            incr_stats = []
            if total_preserved_tests > 0:
                incr_stats.append(f"preserved={total_preserved_tests}")
            if total_new_tests > 0:
                incr_stats.append(f"new={total_new_tests}")
            if total_fixed_tests > 0:
                incr_stats.append(f"fixed={total_fixed_tests}")
            status_parts.append(f"🔄 {', '.join(incr_stats)}")
        
        self.status_label.setText(" | ".join(status_parts))
        self.progress_bar.setValue(100)
        
        # Show compilation details if available
        if compilation_result and self.isVisible():
            # Build incremental stats message
            incr_msg = ""
            if self.incremental_check.isChecked() and (total_preserved_tests > 0 or total_new_tests > 0 or total_fixed_tests > 0):
                incr_msg = "\n\n🔄 Incremental Mode Statistics:"
                if total_preserved_tests > 0:
                    incr_msg += f"\n  ✓ Preserved: {total_preserved_tests} tests"
                if total_new_tests > 0:
                    incr_msg += f"\n  ✓ New: {total_new_tests} tests"
                if total_fixed_tests > 0:
                    incr_msg += f"\n  ✓ Fixed: {total_fixed_tests} tests"
            
            if compilation_result.success:
                QMessageBox.information(
                    self,
                    "Batch Generation Complete",
                    f"✓ All tests generated and compiled successfully!\n\n"
                    f"Generated: {success_count}/{total} files\n"
                    f"Compilation: Successful\n"
                    f"Total time: {total_duration:.1f}s"
                    f"{incr_msg}"
                )
            else:
                error_details = "\n".join(compilation_result.errors[:5]) if compilation_result.errors else "Unknown errors"
                QMessageBox.warning(
                    self,
                    "Batch Generation Complete with Errors",
                    f"⚠ Tests generated but compilation failed\n\n"
                    f"Generated: {success_count}/{total} files\n"
                    f"Compilation errors: {compilation_result.failed_files}\n"
                    f"Compile time: {compilation_result.duration:.1f}s\n\n"
                    f"First errors:\n{error_details}"
                    f"{incr_msg}"
                )
        elif self.isVisible() and self.incremental_check.isChecked() and (total_preserved_tests > 0 or total_new_tests > 0 or total_fixed_tests > 0):
            # Show incremental stats even without compilation result
            incr_msg = "\n\n🔄 Incremental Mode Statistics:"
            if total_preserved_tests > 0:
                incr_msg += f"\n  ✓ Preserved: {total_preserved_tests} tests"
            if total_new_tests > 0:
                incr_msg += f"\n  ✓ New: {total_new_tests} tests"
            if total_fixed_tests > 0:
                incr_msg += f"\n  ✓ Fixed: {total_fixed_tests} tests"
            
            if failed_count == 0:
                QMessageBox.information(
                    self,
                    "Batch Generation Complete",
                    f"✓ All tests generated successfully!\n\n"
                    f"Generated: {success_count}/{total} files\n"
                    f"Total time: {total_duration:.1f}s"
                    f"{incr_msg}"
                )
    
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
