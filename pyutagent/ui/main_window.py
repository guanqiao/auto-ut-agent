"""Main window for PyUT Agent."""

import asyncio
import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QMenuBar,
    QMenu, QFileDialog, QLabel, QProgressBar, QTextEdit,
    QStatusBar, QMessageBox, QDialog, QPushButton, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction

from .chat_widget import ChatWidget
from .dialogs.llm_config_dialog import LLMConfigDialog
from .dialogs.aider_config_dialog import AiderConfigDialog
from .batch_generate_dialog import BatchGenerateDialog
from ..core.config import (
    LLMConfig,
    LLMConfigCollection,
    AiderConfig,
    AppState,
    load_llm_config,
    save_llm_config,
    load_aider_config,
    save_aider_config,
    load_app_state,
    save_app_state,
    get_settings,
)
from ..llm.client import LLMClient
from ..agent.react_agent import ReActAgent, AgentState
from ..memory.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


class AgentWorker(QThread):
    """Worker thread for running Agent."""

    progress_updated = pyqtSignal(dict)
    state_changed = pyqtSignal(str, str)
    log_message = pyqtSignal(str)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        llm_client: LLMClient,
        project_path: str,
        target_file: str,
        target_coverage: float = 0.8,
        max_iterations: int = 10
    ):
        super().__init__()
        self.llm_client = llm_client
        self.project_path = project_path
        self.target_file = target_file
        self.target_coverage = target_coverage
        self.max_iterations = max_iterations
        self._is_running = True

    def run(self):
        """Run the agent."""
        try:
            working_memory = WorkingMemory(
                target_coverage=self.target_coverage,
                max_iterations=self.max_iterations,
                current_file=self.target_file
            )

            agent = ReActAgent(
                llm_client=self.llm_client,
                working_memory=working_memory,
                project_path=self.project_path,
                progress_callback=self._on_progress
            )

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        agent.generate_tests(self.target_file)
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    agent.generate_tests(self.target_file)
                )

            if result.success:
                self.completed.emit({
                    "success": True,
                    "message": result.message,
                    "test_file": result.test_file,
                    "coverage": result.coverage,
                    "iterations": result.iterations
                })
            else:
                self.error.emit(result.message)

        except Exception as e:
            logger.exception("Agent worker failed")
            self.error.emit(str(e))

    def _on_progress(self, progress_info: dict):
        """Handle progress updates."""
        self.progress_updated.emit(progress_info)
        self.state_changed.emit(
            progress_info.get("state", ""),
            progress_info.get("message", "")
        )

    def stop(self):
        """Stop the worker."""
        self._is_running = False
        self.wait(1000)


class ProjectTreeWidget(QTreeWidget):
    """Tree widget for displaying project structure."""

    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabel("Project Files")
        self.setMaximumWidth(300)
        self.itemClicked.connect(self.on_item_clicked)
        self.project_path: str = ""

    def load_project(self, project_path: str):
        """Load project structure into tree."""
        self.clear()
        self.project_path = project_path

        project_name = Path(project_path).name
        root = QTreeWidgetItem(self, [project_name])
        root.setData(0, Qt.ItemDataRole.UserRole, project_path)

        settings = get_settings()
        src_dir = Path(project_path) / settings.project_paths.src_main_java
        if src_dir.exists():
            self._add_directory(src_dir, root)

        root.setExpanded(True)

    def _add_directory(self, dir_path: Path, parent_item: QTreeWidgetItem):
        """Recursively add directory contents."""
        try:
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    tree_item = QTreeWidgetItem(parent_item, [item.name])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item))
                    self._add_directory(item, tree_item)
                elif item.suffix == '.java':
                    tree_item = QTreeWidgetItem(parent_item, [item.name])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item))
        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {dir_path}")
        except Exception as e:
            logger.exception(f"Failed to add directory: {dir_path}")

    def on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click."""
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path and path.endswith('.java'):
            self.file_selected.emit(path)

    def get_selected_file(self) -> str:
        """Get currently selected file path."""
        item = self.currentItem()
        if item:
            path = item.data(0, Qt.ItemDataRole.UserRole)
            if path and path.endswith('.java'):
                return path
        return ""


class ProgressWidget(QWidget):
    """Widget displaying generation progress and logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Setup the UI with vertical splitter for progress and logs."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create vertical splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Upper section: Progress info
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(10, 10, 10, 10)

        header = QLabel("Progress")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        progress_layout.addWidget(header)

        self.state_label = QLabel("Status: Ready")
        self.state_label.setStyleSheet("font-weight: bold; color: #666;")
        progress_layout.addWidget(self.state_label)

        self.status_label = QLabel("Waiting to start...")
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.coverage_label = QLabel("Coverage: -")
        progress_layout.addWidget(self.coverage_label)

        self.iteration_label = QLabel("Iteration: -")
        progress_layout.addWidget(self.iteration_label)

        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        progress_layout.addWidget(self.details_label)

        progress_layout.addStretch()
        splitter.addWidget(progress_container)

        # Lower section: Log display
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(10, 10, 10, 10)

        # Log header with title and clear button
        log_header_layout = QHBoxLayout()
        log_header = QLabel("Logs")
        log_header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        log_header_layout.addWidget(log_header)
        log_header_layout.addStretch()

        self.clear_log_btn = QPushButton("Clear")
        self.clear_log_btn.setMaximumWidth(60)
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_header_layout.addWidget(self.clear_log_btn)
        log_layout.addLayout(log_header_layout)

        # Log display area
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
        log_layout.addWidget(self.log_area)

        splitter.addWidget(log_container)

        # Set initial sizes (progress:logs = 40:60)
        splitter.setSizes([200, 300])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def update_progress(self, value: int, status: str = ""):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)

    def update_state(self, state: str, message: str = ""):
        """Update state indicator."""
        state_colors = {
            "IDLE": "#666",
            "PARSING": "#2196F3",
            "GENERATING": "#9C27B0",
            "COMPILING": "#FF9800",
            "TESTING": "#00BCD4",
            "ANALYZING": "#3F51B5",
            "FIXING": "#F44336",
            "OPTIMIZING": "#4CAF50",
            "COMPLETED": "#4CAF50",
            "FAILED": "#F44336",
            "PAUSED": "#FF9800",
        }
        color = state_colors.get(state, "#666")
        self.state_label.setText(f"Status: {state}")
        self.state_label.setStyleSheet(f"font-weight: bold; color: {color};")

        if message:
            self.status_label.setText(message)

    def update_coverage(self, coverage: float, target: float):
        """Update coverage display."""
        self.coverage_label.setText(f"Coverage: {coverage:.1%} / Target: {target:.1%}")

        if coverage >= target:
            self.coverage_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif coverage >= target * 0.8:
            self.coverage_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.coverage_label.setStyleSheet("color: #F44336; font-weight: bold;")

    def update_iteration(self, current: int, max_iter: int):
        """Update iteration display."""
        self.iteration_label.setText(f"Iteration: {current} / {max_iter}")

    def update_details(self, details: str):
        """Update details text."""
        self.details_label.setText(details)

    def add_log(self, message: str, level: str = "INFO"):
        """Add log message with color coding.

        Args:
            message: Log message
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color mapping for different log levels
        colors = {
            "DEBUG": "#808080",      # Gray
            "INFO": "#4FC1FF",       # Light blue
            "WARNING": "#FFCC00",    # Yellow/Orange
            "ERROR": "#FF6B6B",      # Red
            "CRITICAL": "#FF0000",   # Bright red
        }
        color = colors.get(level.upper(), "#d4d4d4")

        # Format: [HH:MM:SS] [LEVEL] message
        formatted_message = f"[{timestamp}] [{level.upper()}] {message}"

        # Append with color
        self.log_area.append(f'<span style="color: {color};">{formatted_message}</span>')

        # Auto-scroll to bottom
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        """Clear log area."""
        self.log_area.clear()


class MainWindow(QMainWindow):
    """Main application window."""

    project_opened = pyqtSignal(str)
    generate_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyUT Agent - AI Unit Test Generator")
        self.setGeometry(100, 100, 1400, 900)

        self.current_project: str = ""
        self.config_collection: LLMConfigCollection = load_llm_config()
        self.aider_config: AiderConfig = load_aider_config()
        self.app_state: AppState = load_app_state()
        self.llm_client: LLMClient = None
        self.agent_worker: AgentWorker = None
        self.recent_project_actions: list = []

        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        self.setup_llm_client()

    def setup_ui(self):
        """Setup the main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.project_tree = ProjectTreeWidget()
        self.project_tree.file_selected.connect(self.on_file_selected)
        splitter.addWidget(self.project_tree)

        self.chat_widget = ChatWidget()
        self.chat_widget.message_sent.connect(self.on_message_sent)
        self.chat_widget.generate_clicked.connect(self.on_generate_tests)
        self.chat_widget.pause_clicked.connect(self.on_pause_generation)
        splitter.addWidget(self.chat_widget)

        self.progress_widget = ProgressWidget()
        splitter.addWidget(self.progress_widget)

        splitter.setSizes([300, 700, 400])
        layout.addWidget(splitter)

    def setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Project...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.on_open_project)
        file_menu.addAction(open_action)

        # Recent projects submenu
        self.recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self.recent_menu)
        self._update_recent_projects_menu()

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        settings_menu = menubar.addMenu("&Settings")

        llm_config_action = QAction("&LLM Configuration...", self)
        llm_config_action.triggered.connect(self.on_llm_config)
        settings_menu.addAction(llm_config_action)

        aider_config_action = QAction("&Aider Advanced Configuration...", self)
        aider_config_action.triggered.connect(self.on_aider_config)
        settings_menu.addAction(aider_config_action)

        tools_menu = menubar.addMenu("&Tools")

        scan_action = QAction("&Scan Project", self)
        scan_action.triggered.connect(self.on_scan_project)
        tools_menu.addAction(scan_action)

        generate_action = QAction("&Generate Tests", self)
        generate_action.setShortcut("Ctrl+G")
        generate_action.triggered.connect(self.on_generate_tests)
        tools_menu.addAction(generate_action)

        generate_all_action = QAction("Generate &All Tests", self)
        generate_all_action.setShortcut("Ctrl+Shift+G")
        generate_all_action.triggered.connect(self.on_generate_all_tests)
        tools_menu.addAction(generate_all_action)

        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Please open a Maven project")

    def setup_llm_client(self):
        """Setup LLM client using default configuration."""
        try:
            default_config = self.config_collection.get_default_config()
            if default_config:
                self.llm_client = LLMClient.from_config(default_config)
                logger.info(f"LLM client initialized with config: {default_config.name}")
            else:
                self.status_bar.showMessage("Please configure LLM model")
                logger.warning("No default LLM config found")
        except Exception as e:
            logger.exception("Failed to initialize LLM client")
            self.status_bar.showMessage(f"LLM client initialization failed: {e}")

    def on_open_project(self):
        """Handle open project action."""
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select Maven Project",
                "",
                QFileDialog.Option.ShowDirsOnly
            )

            if dir_path:
                self._open_project(dir_path)
        except Exception as e:
            logger.exception("Failed to open project")

    def _open_project(self, dir_path: str) -> bool:
        """Open a project by path.

        Args:
            dir_path: Path to the project directory

        Returns:
            True if project was opened successfully
        """
        try:
            pom_path = Path(dir_path) / "pom.xml"
            if not pom_path.exists():
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Selected directory is not a Maven project (pom.xml not found)"
                )
                logger.warning(f"Selected directory is not a Maven project: {dir_path}")
                return False

            self.current_project = dir_path
            self.project_tree.load_project(dir_path)
            self.project_opened.emit(dir_path)
            self.status_bar.showMessage(f"Project: {dir_path}")
            logger.info(f"Project opened: {dir_path}")

            # Save to app state
            self.app_state.add_project(dir_path)
            save_app_state(self.app_state)
            self._update_recent_projects_menu()

            self.chat_widget.add_agent_message(
                f"Project opened: {Path(dir_path).name}\n"
                "Please select a Java file, then click Generate Tests or send a message."
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to open project: {dir_path}")
            return False

    def _update_recent_projects_menu(self):
        """Update the recent projects submenu."""
        # Clear existing actions
        self.recent_menu.clear()
        self.recent_project_actions.clear()

        # Get valid recent projects
        valid_projects = self.app_state.get_recent_project_paths()

        if not valid_projects:
            no_recent_action = QAction("No recent projects", self)
            no_recent_action.setEnabled(False)
            self.recent_menu.addAction(no_recent_action)
            return

        # Add project actions
        for i, project_path in enumerate(valid_projects[:10], 1):
            project_name = Path(project_path).name
            action = QAction(f"&{i}. {project_name}", self)
            action.setToolTip(project_path)
            action.triggered.connect(lambda checked, path=project_path: self._on_recent_project_selected(path))
            self.recent_menu.addAction(action)
            self.recent_project_actions.append(action)

        # Add separator and clear action
        self.recent_menu.addSeparator()
        clear_action = QAction("Clear History", self)
        clear_action.triggered.connect(self._clear_recent_projects)
        self.recent_menu.addAction(clear_action)

    def _on_recent_project_selected(self, project_path: str):
        """Handle selection of a recent project.

        Args:
            project_path: Path to the project
        """
        if not Path(project_path).exists():
            QMessageBox.warning(
                self,
                "Project Not Found",
                f"The project no longer exists:\n{project_path}\n\nIt will be removed from recent projects."
            )
            self.app_state.remove_project(project_path)
            save_app_state(self.app_state)
            self._update_recent_projects_menu()
            return

        self._open_project(project_path)

    def _clear_recent_projects(self):
        """Clear all recent projects."""
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear all recent projects?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.app_state.recent_projects.clear()
            self.app_state.last_project_path = None
            save_app_state(self.app_state)
            self._update_recent_projects_menu()
            logger.info("Recent projects history cleared")

    def load_last_project(self) -> bool:
        """Load the last opened project if available.

        Returns:
            True if a project was loaded
        """
        if not self.app_state.last_project_path:
            logger.info("No last project to load")
            return False

        if not Path(self.app_state.last_project_path).exists():
            logger.warning(f"Last project no longer exists: {self.app_state.last_project_path}")
            # Remove invalid project from history
            self.app_state.remove_project(self.app_state.last_project_path)
            save_app_state(self.app_state)
            self._update_recent_projects_menu()
            return False

        logger.info(f"Auto-loading last project: {self.app_state.last_project_path}")
        return self._open_project(self.app_state.last_project_path)

    def on_llm_config(self):
        """Handle LLM config action."""
        try:
            dialog = LLMConfigDialog(self.config_collection, self.aider_config, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.config_collection = dialog.get_config_collection()
                self.aider_config = dialog.aider_config

                save_llm_config(self.config_collection)

                self.setup_llm_client()

                default_config = self.config_collection.get_default_config()
                if default_config:
                    self.status_bar.showMessage(
                        f"LLM configuration updated: {default_config.get_display_name()}"
                    )
                    logger.info(f"LLM config updated: {default_config.get_display_name()}")
                else:
                    self.status_bar.showMessage("LLM configuration updated")
                    logger.info("LLM config updated")
        except Exception as e:
            logger.exception("Failed to open LLM config dialog")

    def on_aider_config(self):
        """Handle Aider config action."""
        try:
            dialog = AiderConfigDialog(self.aider_config, self.config_collection, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.aider_config = dialog.config

                save_aider_config(self.aider_config)

                features = []
                if self.aider_config.use_architect_editor:
                    features.append("Architect/Editor")
                    if self.aider_config.architect_model_id and self.aider_config.editor_model_id:
                        arch_config = self.config_collection.get_config(self.aider_config.architect_model_id)
                        edit_config = self.config_collection.get_config(self.aider_config.editor_model_id)
                        if arch_config and edit_config:
                            features.append(f"Architect: {arch_config.name or arch_config.model}")
                            features.append(f"Editor: {edit_config.name or edit_config.model}")
                if self.aider_config.enable_multi_file:
                    features.append("Multi-file editing")
                if self.aider_config.auto_detect_format:
                    features.append("Auto format detection")

                if features:
                    self.status_bar.showMessage(
                        f"Aider configuration updated: {', '.join(features)}"
                    )
                    logger.info(f"Aider config updated: {', '.join(features)}")
                else:
                    self.status_bar.showMessage("Aider configuration updated")
                    logger.info("Aider config updated")
        except Exception as e:
            logger.exception("Failed to open Aider config dialog")

    def on_scan_project(self):
        """Handle scan project action."""
        try:
            if not self.current_project:
                QMessageBox.information(
                    self,
                    "Information",
                    "Please open a project first"
                )
                return

            self.project_tree.load_project(self.current_project)
            self.status_bar.showMessage("Project refreshed")
            logger.info(f"Project rescanned: {self.current_project}")
        except Exception as e:
            logger.exception("Failed to scan project")

    def on_generate_tests(self):
        """Handle generate tests action."""
        try:
            selected_file = self.project_tree.get_selected_file()

            if not selected_file:
                QMessageBox.information(
                    self,
                    "Information",
                    "Please select a Java file from the left panel first"
                )
                return

            if not self.llm_client:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "LLM client not initialized, please configure LLM first"
                )
                return

            self.start_generation(selected_file)
            logger.info(f"Test generation started for: {selected_file}")
        except Exception as e:
            logger.exception("Failed to start test generation")

    def on_generate_all_tests(self):
        """Handle generate all tests action."""
        try:
            if not self.current_project:
                QMessageBox.information(
                    self,
                    "Information",
                    "Please open a project first"
                )
                return

            if not self.llm_client:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "LLM client not initialized, please configure LLM first"
                )
                return

            settings = get_settings()
            src_dir = Path(self.current_project) / settings.project_paths.src_main_java
            
            if not src_dir.exists():
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Java source directory not found: {src_dir}"
                )
                return

            java_files = [str(f.relative_to(self.current_project)) for f in src_dir.rglob('*.java')]
            
            if not java_files:
                QMessageBox.information(
                    self,
                    "Information",
                    "No Java files found in the project"
                )
                return

            dialog = BatchGenerateDialog(self)
            dialog.set_files(self.current_project, java_files, self.llm_client)
            dialog.exec()
            
            logger.info(f"Batch generate dialog opened with {len(java_files)} files")
        except Exception as e:
            logger.exception("Failed to open batch generate dialog")

    def on_pause_generation(self):
        """Handle pause generation."""
        try:
            if self.agent_worker and self.agent_worker.isRunning():
                self.agent_worker.stop()
                self.progress_widget.update_state("PAUSED", "Generation paused")
                self.chat_widget.add_agent_message("Generation paused. Click continue to resume.")
                logger.info("Test generation paused")
        except Exception as e:
            logger.exception("Failed to pause generation")

    def start_generation(self, target_file: str):
        """Start test generation.

        Args:
            target_file: Target Java file path
        """
        try:
            if self.agent_worker and self.agent_worker.isRunning():
                self.agent_worker.stop()

            self.progress_widget.clear_log()
            self.progress_widget.update_progress(0)

            settings = get_settings()
            file_name = Path(target_file).name
            
            # 添加初始日志信息
            self.progress_widget.add_log(f"🚀 Starting test generation for {file_name}", "INFO")
            self.progress_widget.add_log(f"📊 Target coverage: {settings.coverage.target_coverage:.1%}", "INFO")
            self.progress_widget.add_log(f"🔄 Max iterations: {settings.coverage.max_iterations}", "INFO")
            self.progress_widget.add_log("⏳ This process may take several minutes. Please wait...", "WARNING")
            
            self.agent_worker = AgentWorker(
                llm_client=self.llm_client,
                project_path=self.current_project,
                target_file=target_file,
                target_coverage=settings.coverage.target_coverage,
                max_iterations=settings.coverage.max_iterations
            )

            self.agent_worker.progress_updated.connect(self.on_agent_progress)
            self.agent_worker.state_changed.connect(self.on_agent_state_changed)
            self.agent_worker.log_message.connect(self.on_agent_log)
            self.agent_worker.completed.connect(self.on_agent_completed)
            self.agent_worker.error.connect(self.on_agent_error)

            self.agent_worker.start()

            self.chat_widget.add_agent_message(f"🚀 Starting test generation for {file_name}...")
            self.status_bar.showMessage(f"Generating tests: {file_name}")
            logger.info(f"Started generation for: {target_file}")
        except Exception as e:
            logger.exception("Failed to start generation")

    def on_agent_progress(self, progress_info: dict):
        """Handle agent progress updates."""
        progress = progress_info.get("progress", {})

        coverage_str = progress.get("coverage", "0%").replace("%", "")
        try:
            coverage = float(coverage_str) / 100
            target_str = progress.get("target", "80%").replace("%", "")
            target = float(target_str) / 100
            self.progress_widget.update_coverage(coverage, target)
        except ValueError:
            pass

        iteration_str = progress.get("iteration", "0/10")
        try:
            current, max_iter = iteration_str.split("/")
            self.progress_widget.update_iteration(int(current), int(max_iter))
        except ValueError:
            pass

        try:
            current_iter = int(progress.get("iteration", "0/10").split("/")[0])
            max_iter = int(progress.get("iteration", "0/10").split("/")[1])
            progress_pct = min(100, int((current_iter / max_iter) * 100))
            self.progress_widget.update_progress(progress_pct)
        except (ValueError, IndexError):
            pass

    def on_agent_state_changed(self, state: str, message: str):
        """Handle agent state changes."""
        self.progress_widget.update_state(state, message)
        # 只记录非重复的状态变更日志
        if message and not message.startswith("["):
            self.progress_widget.add_log(f"[{state}] {message}", "INFO")

    def on_agent_log(self, message: str, level: str = "INFO"):
        """Handle agent log messages."""
        self.progress_widget.add_log(message, level)

    def on_agent_completed(self, result: dict):
        """Handle agent completion."""
        try:
            success = result.get("success", False)
            message = result.get("message", "")
            test_file = result.get("test_file", "")
            coverage = result.get("coverage", 0.0)
            iterations = result.get("iterations", 0)

            if success:
                self.progress_widget.add_log(f"🎉 Test generation completed successfully!", "INFO")
                self.progress_widget.add_log(f"📁 Test file: {test_file}", "INFO")
                self.progress_widget.add_log(f"📊 Coverage: {coverage:.1%}", "INFO")
                self.progress_widget.add_log(f"🔄 Iterations: {iterations}", "INFO")
                
                self.chat_widget.add_agent_message(
                    f"🎉 Test generation completed!\n\n"
                    f"{message}\n"
                    f"Test file: {test_file}\n"
                    f"Iterations: {iterations}"
                )
                self.progress_widget.update_progress(100, "✅ Completed")
                logger.info(f"Test generation completed successfully: {test_file}, coverage: {coverage:.1%}")
            else:
                self.progress_widget.add_log(f"❌ Generation failed: {message}", "ERROR")
                self.chat_widget.add_agent_message(f"❌ Generation failed: {message}")
                logger.warning(f"Test generation failed: {message}")

            self.status_bar.showMessage("Ready")
        except Exception as e:
            logger.exception("Failed to handle agent completion")

    def on_agent_error(self, error_message: str):
        """Handle agent errors."""
        try:
            self.progress_widget.add_log(f"❌ Error: {error_message}", "ERROR")
            self.chat_widget.add_agent_message(f"❌ Error: {error_message}")
            self.progress_widget.update_state("FAILED", f"❌ {error_message}")
            self.status_bar.showMessage("Generation failed")
            logger.error(f"Agent error: {error_message}")
        except Exception as e:
            logger.exception("Failed to handle agent error")

    def on_file_selected(self, file_path: str):
        """Handle file selection."""
        try:
            self.status_bar.showMessage(f"Selected: {file_path}")
            logger.info(f"File selected: {file_path}")
        except Exception as e:
            logger.exception("Failed to handle file selection")

    def on_message_sent(self, message: str):
        """Handle user message."""
        try:
            if "generate" in message.lower() or "test" in message.lower():
                selected_file = self.project_tree.get_selected_file()
                if selected_file:
                    self.start_generation(selected_file)
                else:
                    self.chat_widget.add_agent_message(
                        "Please select a Java file from the left panel first, then tell me to generate tests."
                    )
            else:
                self.chat_widget.add_agent_message(
                    "I am PyUT Agent, I can help you generate Java unit tests.\n"
                    "Please select a Java file, then click the Generate Tests button."
                )
            logger.info(f"User message sent: {message}")
        except Exception as e:
            logger.exception("Failed to handle user message")

    def on_about(self):
        """Show about dialog."""
        try:
            QMessageBox.about(
                self,
                "About PyUT Agent",
                "<h2>PyUT Agent</h2>"
                "<p>AI-powered Java Unit Test Generator</p>"
                "<p>Based on ReAct Agent architecture with self-feedback loop</p>"
                "<p>Features:</p>"
                "<ul>"
                "<li>Automatic JUnit 5 test code generation</li>"
                "<li>Automatic compilation error fixing</li>"
                "<li>Automatic test failure fixing</li>"
                "<li>Coverage-driven iterative optimization</li>"
                "<li>Pause/Resume support</li>"
                "</ul>"
                "<p>Version: 0.1.0</p>"
            )
            logger.info("About dialog shown")
        except Exception as e:
            logger.exception("Failed to show about dialog")

    def update_progress(self, value: int, status: str = ""):
        """Update progress display."""
        try:
            self.progress_widget.update_progress(value, status)
        except Exception as e:
            logger.exception("Failed to update progress")

    def add_log(self, message: str):
        """Add log message."""
        try:
            self.progress_widget.add_log(message)
        except Exception as e:
            logger.exception("Failed to add log")

    def set_status(self, message: str):
        """Update status bar."""
        try:
            self.status_bar.showMessage(message)
        except Exception as e:
            logger.exception("Failed to set status")

    def add_agent_message(self, message: str):
        """Add agent message to chat."""
        try:
            self.chat_widget.add_agent_message(message)
        except Exception as e:
            logger.exception("Failed to add agent message")

    def closeEvent(self, event):
        """Handle window close."""
        try:
            if self.agent_worker and self.agent_worker.isRunning():
                self.agent_worker.stop()
            event.accept()
            logger.info("Main window closed")
        except Exception as e:
            logger.exception("Failed to handle close event")
            event.accept()
