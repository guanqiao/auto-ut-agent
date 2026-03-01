"""Main window for PyUT Agent."""

import asyncio
import logging
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QMenuBar,
    QMenu, QFileDialog, QLabel, QProgressBar, QTextEdit,
    QStatusBar, QMessageBox, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from pathlib import Path

from .chat_widget import ChatWidget
from .dialogs.llm_config_dialog import LLMConfigDialog
from .dialogs.aider_config_dialog import AiderConfigDialog
from ..llm.config import LLMConfig, LLMConfigCollection
from ..llm.client import LLMClient
from ..agent.react_agent import ReActAgent, AgentState
from ..memory.working_memory import WorkingMemory
from ..tools.aider_integration import AiderConfig
from ..config import load_llm_config, save_llm_config

logger = logging.getLogger(__name__)


class AgentWorker(QThread):
    """Worker thread for running Agent."""
    
    progress_updated = pyqtSignal(dict)  # Progress info
    state_changed = pyqtSignal(str, str)  # state, message
    log_message = pyqtSignal(str)  # log message
    completed = pyqtSignal(dict)  # result
    error = pyqtSignal(str)  # error message
    
    def __init__(
        self,
        llm_client: LLMClient,
        project_path: str,
        target_file: str,
        target_coverage: float = 0.8,
        max_iterations: int = 10
    ):
        """Initialize worker.
        
        Args:
            llm_client: LLM client
            project_path: Project path
            target_file: Target Java file
            target_coverage: Target coverage
            max_iterations: Max iterations
        """
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
            # Create working memory
            working_memory = WorkingMemory(
                target_coverage=self.target_coverage,
                max_iterations=self.max_iterations,
                current_file=self.target_file
            )
            
            # Create agent with progress callback
            agent = ReActAgent(
                llm_client=self.llm_client,
                working_memory=working_memory,
                project_path=self.project_path,
                progress_callback=self._on_progress
            )
            
            # Run agent
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                agent.generate_tests(self.target_file)
            )
            loop.close()
            
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
    
    file_selected = pyqtSignal(str)  # Emits file path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabel("📁 项目文件")
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
        
        # Scan for Java files
        src_dir = Path(project_path) / "src" / "main" / "java"
        if src_dir.exists():
            self._add_directory(src_dir, root)
        
        root.setExpanded(True)
    
    def _add_directory(self, dir_path: Path, parent_item: QTreeWidgetItem):
        """Recursively add directory contents."""
        try:
            for item in sorted(dir_path.iterdir()):
                if item.is_dir():
                    tree_item = QTreeWidgetItem(parent_item, [f"📁 {item.name}"])
                    tree_item.setData(0, Qt.ItemDataRole.UserRole, str(item))
                    self._add_directory(item, tree_item)
                elif item.suffix == '.java':
                    tree_item = QTreeWidgetItem(parent_item, [f"📄 {item.name}"])
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
    """Widget displaying generation progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel("📊 进度")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        # State indicator
        self.state_label = QLabel("状态: 就绪")
        self.state_label.setStyleSheet("font-weight: bold; color: #666;")
        layout.addWidget(self.state_label)
        
        # Status
        self.status_label = QLabel("等待开始...")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # Coverage info
        self.coverage_label = QLabel("覆盖率: -")
        layout.addWidget(self.coverage_label)
        
        # Iteration info
        self.iteration_label = QLabel("迭代: -")
        layout.addWidget(self.iteration_label)
        
        # Details
        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        
        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(200)
        layout.addWidget(self.log_area)
        
        layout.addStretch()
    
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
        self.state_label.setText(f"状态: {state}")
        self.state_label.setStyleSheet(f"font-weight: bold; color: {color};")
        
        if message:
            self.status_label.setText(message)
    
    def update_coverage(self, coverage: float, target: float):
        """Update coverage display."""
        self.coverage_label.setText(f"覆盖率: {coverage:.1%} / 目标: {target:.1%}")
        
        # Color code based on target
        if coverage >= target:
            self.coverage_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif coverage >= target * 0.8:
            self.coverage_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.coverage_label.setStyleSheet("color: #F44336; font-weight: bold;")
    
    def update_iteration(self, current: int, max_iter: int):
        """Update iteration display."""
        self.iteration_label.setText(f"迭代: {current} / {max_iter}")
    
    def update_details(self, details: str):
        """Update details text."""
        self.details_label.setText(details)
    
    def add_log(self, message: str):
        """Add log message."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {message}")
    
    def clear_log(self):
        """Clear log area."""
        self.log_area.clear()


class MainWindow(QMainWindow):
    """Main application window."""
    
    project_opened = pyqtSignal(str)  # Emits project path
    generate_requested = pyqtSignal(str)  # Emits file path
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyUT Agent - AI 单元测试生成器")
        self.setGeometry(100, 100, 1400, 900)

        self.current_project: str = ""
        self.config_collection: LLMConfigCollection = load_llm_config()
        self.aider_config: AiderConfig = AiderConfig()
        self.llm_client: LLMClient = None
        self.agent_worker: AgentWorker = None

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
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Project tree
        self.project_tree = ProjectTreeWidget()
        self.project_tree.file_selected.connect(self.on_file_selected)
        splitter.addWidget(self.project_tree)
        
        # Middle: Chat widget
        self.chat_widget = ChatWidget()
        self.chat_widget.message_sent.connect(self.on_message_sent)
        self.chat_widget.generate_clicked.connect(self.on_generate_tests)
        self.chat_widget.pause_clicked.connect(self.on_pause_generation)
        splitter.addWidget(self.chat_widget)
        
        # Right: Progress and details
        self.progress_widget = ProgressWidget()
        splitter.addWidget(self.progress_widget)
        
        # Set splitter sizes
        splitter.setSizes([300, 700, 400])
        layout.addWidget(splitter)
    
    def setup_menu(self):
        """Setup menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("文件(&F)")
        
        # Open Project action
        open_action = QAction("打开项目(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.on_open_project)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menubar.addMenu("设置(&S)")

        # LLM Config action
        llm_config_action = QAction("LLM 配置(&L)...", self)
        llm_config_action.triggered.connect(self.on_llm_config)
        settings_menu.addAction(llm_config_action)

        # Aider Config action
        aider_config_action = QAction("Aider 高级配置(&A)...", self)
        aider_config_action.triggered.connect(self.on_aider_config)
        settings_menu.addAction(aider_config_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("工具(&T)")
        
        # Scan Project action
        scan_action = QAction("扫描项目(&S)", self)
        scan_action.triggered.connect(self.on_scan_project)
        tools_menu.addAction(scan_action)
        
        # Generate Tests action
        generate_action = QAction("生成测试(&G)", self)
        generate_action.setShortcut("Ctrl+G")
        generate_action.triggered.connect(self.on_generate_tests)
        tools_menu.addAction(generate_action)
        
        # Help menu
        help_menu = menubar.addMenu("帮助(&H)")
        
        # About action
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请打开一个 Maven 项目")
    
    def setup_llm_client(self):
        """Setup LLM client using default configuration."""
        try:
            default_config = self.config_collection.get_default_config()
            if default_config:
                self.llm_client = LLMClient.from_config(default_config)
                logger.info(f"LLM client initialized with config: {default_config.name}")
            else:
                self.status_bar.showMessage("请配置 LLM 模型")
                logger.warning("No default LLM config found")
        except Exception as e:
            logger.exception("Failed to initialize LLM client")
            self.status_bar.showMessage(f"LLM 客户端初始化失败: {e}")
    
    def on_open_project(self):
        """Handle open project action."""
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "选择 Maven 项目",
                "",
                QFileDialog.Option.ShowDirsOnly
            )

            if dir_path:
                # Check if it's a Maven project
                pom_path = Path(dir_path) / "pom.xml"
                if not pom_path.exists():
                    QMessageBox.warning(
                        self,
                        "警告",
                        "选择的目录不是 Maven 项目（未找到 pom.xml）"
                    )
                    logger.warning(f"Selected directory is not a Maven project: {dir_path}")
                    return

                self.current_project = dir_path
                self.project_tree.load_project(dir_path)
                self.project_opened.emit(dir_path)
                self.status_bar.showMessage(f"项目: {dir_path}")
                logger.info(f"Project opened: {dir_path}")

                # Add welcome message
                self.chat_widget.add_agent_message(
                    f"已打开项目: {Path(dir_path).name}\n"
                    "请选择一个 Java 文件，然后点击生成测试按钮或发送消息。"
                )
        except Exception as e:
            logger.exception("Failed to open project")
    
    def on_llm_config(self):
        """Handle LLM config action."""
        try:
            dialog = LLMConfigDialog(self.config_collection, self.aider_config, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.config_collection = dialog.get_config_collection()
                self.aider_config = dialog.aider_config

                # Save config to file
                save_llm_config(self.config_collection)

                self.setup_llm_client()

                # Show status message
                default_config = self.config_collection.get_default_config()
                if default_config:
                    self.status_bar.showMessage(
                        f"LLM 配置已更新: {default_config.get_display_name()}"
                    )
                    logger.info(f"LLM config updated: {default_config.get_display_name()}")
                else:
                    self.status_bar.showMessage("LLM 配置已更新")
                    logger.info("LLM config updated")
        except Exception as e:
            logger.exception("Failed to open LLM config dialog")

    def on_aider_config(self):
        """Handle Aider config action."""
        try:
            dialog = AiderConfigDialog(self.aider_config, self.config_collection, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.aider_config = dialog.config
                # Show status message with enabled features
                features = []
                if self.aider_config.use_architect_editor:
                    features.append("Architect/Editor")
                    # Show which models are selected
                    if self.aider_config.architect_model_id and self.aider_config.editor_model_id:
                        arch_config = self.config_collection.get_config(self.aider_config.architect_model_id)
                        edit_config = self.config_collection.get_config(self.aider_config.editor_model_id)
                        if arch_config and edit_config:
                            features.append(f"Architect: {arch_config.name or arch_config.model}")
                            features.append(f"Editor: {edit_config.name or edit_config.model}")
                if self.aider_config.enable_multi_file:
                    features.append("多文件编辑")
                if self.aider_config.auto_detect_format:
                    features.append("自动格式检测")

                if features:
                    self.status_bar.showMessage(
                        f"Aider 配置已更新: {', '.join(features)}"
                    )
                    logger.info(f"Aider config updated: {', '.join(features)}")
                else:
                    self.status_bar.showMessage("Aider 配置已更新")
                    logger.info("Aider config updated")
        except Exception as e:
            logger.exception("Failed to open Aider config dialog")
    
    def on_scan_project(self):
        """Handle scan project action."""
        try:
            if not self.current_project:
                QMessageBox.information(
                    self,
                    "提示",
                    "请先打开一个项目"
                )
                return

            self.project_tree.load_project(self.current_project)
            self.status_bar.showMessage("项目已刷新")
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
                    "提示",
                    "请先在左侧选择一个 Java 文件"
                )
                return

            if not self.llm_client:
                QMessageBox.warning(
                    self,
                    "警告",
                    "LLM 客户端未初始化，请先配置 LLM"
                )
                return

            # Start generation
            self.start_generation(selected_file)
            logger.info(f"Test generation started for: {selected_file}")
        except Exception as e:
            logger.exception("Failed to start test generation")
    
    def on_pause_generation(self):
        """Handle pause generation."""
        try:
            if self.agent_worker and self.agent_worker.isRunning():
                self.agent_worker.stop()
                self.progress_widget.update_state("PAUSED", "生成已暂停")
                self.chat_widget.add_agent_message("生成已暂停。点击继续可恢复。")
                logger.info("Test generation paused")
        except Exception as e:
            logger.exception("Failed to pause generation")
    
    def start_generation(self, target_file: str):
        """Start test generation.

        Args:
            target_file: Target Java file path
        """
        try:
            # Stop any existing worker
            if self.agent_worker and self.agent_worker.isRunning():
                self.agent_worker.stop()

            # Clear previous progress
            self.progress_widget.clear_log()
            self.progress_widget.update_progress(0)

            # Create and start worker
            self.agent_worker = AgentWorker(
                llm_client=self.llm_client,
                project_path=self.current_project,
                target_file=target_file,
                target_coverage=0.8,
                max_iterations=10
            )

            # Connect signals
            self.agent_worker.progress_updated.connect(self.on_agent_progress)
            self.agent_worker.state_changed.connect(self.on_agent_state_changed)
            self.agent_worker.log_message.connect(self.on_agent_log)
            self.agent_worker.completed.connect(self.on_agent_completed)
            self.agent_worker.error.connect(self.on_agent_error)

            # Start
            self.agent_worker.start()

            # Update UI
            file_name = Path(target_file).name
            self.chat_widget.add_agent_message(f"开始为 {file_name} 生成测试...")
            self.status_bar.showMessage(f"生成测试中: {file_name}")
            logger.info(f"Started generation for: {target_file}")
        except Exception as e:
            logger.exception("Failed to start generation")
    
    def on_agent_progress(self, progress_info: dict):
        """Handle agent progress updates."""
        progress = progress_info.get("progress", {})
        
        # Update coverage
        coverage_str = progress.get("coverage", "0%").replace("%", "")
        try:
            coverage = float(coverage_str) / 100
            target_str = progress.get("target", "80%").replace("%", "")
            target = float(target_str) / 100
            self.progress_widget.update_coverage(coverage, target)
        except ValueError:
            pass
        
        # Update iteration
        iteration_str = progress.get("iteration", "0/10")
        try:
            current, max_iter = iteration_str.split("/")
            self.progress_widget.update_iteration(int(current), int(max_iter))
        except ValueError:
            pass
        
        # Calculate overall progress
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
        self.progress_widget.add_log(f"[{state}] {message}")
    
    def on_agent_log(self, message: str):
        """Handle agent log messages."""
        self.progress_widget.add_log(message)
    
    def on_agent_completed(self, result: dict):
        """Handle agent completion."""
        try:
            success = result.get("success", False)
            message = result.get("message", "")
            test_file = result.get("test_file", "")
            coverage = result.get("coverage", 0.0)
            iterations = result.get("iterations", 0)

            if success:
                self.chat_widget.add_agent_message(
                    f"✅ 测试生成完成！\n\n"
                    f"{message}\n"
                    f"测试文件: {test_file}\n"
                    f"迭代次数: {iterations}"
                )
                self.progress_widget.update_progress(100, "完成")
                logger.info(f"Test generation completed successfully: {test_file}, coverage: {coverage:.1%}")
            else:
                self.chat_widget.add_agent_message(f"❌ 生成失败: {message}")
                logger.warning(f"Test generation failed: {message}")

            self.status_bar.showMessage("就绪")
        except Exception as e:
            logger.exception("Failed to handle agent completion")
    
    def on_agent_error(self, error_message: str):
        """Handle agent errors."""
        try:
            self.chat_widget.add_agent_message(f"❌ 错误: {error_message}")
            self.progress_widget.update_state("FAILED", error_message)
            self.status_bar.showMessage("生成失败")
            logger.error(f"Agent error: {error_message}")
        except Exception as e:
            logger.exception("Failed to handle agent error")
    
    def on_file_selected(self, file_path: str):
        """Handle file selection."""
        try:
            self.status_bar.showMessage(f"选中: {file_path}")
            logger.info(f"File selected: {file_path}")
        except Exception as e:
            logger.exception("Failed to handle file selection")

    def on_message_sent(self, message: str):
        """Handle user message."""
        try:
            # Check if it's a generation request
            if "生成" in message or "测试" in message:
                selected_file = self.project_tree.get_selected_file()
                if selected_file:
                    self.start_generation(selected_file)
                else:
                    self.chat_widget.add_agent_message(
                        "请先在左侧选择一个 Java 文件，然后告诉我生成测试。"
                    )
            else:
                # General chat - add a simple response
                self.chat_widget.add_agent_message(
                    "我是 PyUT Agent，可以帮助你生成 Java 单元测试。\n"
                    "请选择一个 Java 文件，然后点击生成测试按钮。"
                )
            logger.info(f"User message sent: {message}")
        except Exception as e:
            logger.exception("Failed to handle user message")

    def on_about(self):
        """Show about dialog."""
        try:
            QMessageBox.about(
                self,
                "关于 PyUT Agent",
                "<h2>PyUT Agent</h2>"
                "<p>AI 驱动的 Java 单元测试生成器</p>"
                "<p>基于 ReAct Agent 架构，支持自我反馈闭环</p>"
                "<p>功能特点:</p>"
                "<ul>"
                "<li>自动生成 JUnit 5 测试代码</li>"
                "<li>自动修复编译错误</li>"
                "<li>自动修复测试失败</li>"
                "<li>覆盖率驱动的迭代优化</li>"
                "<li>支持暂停/恢复</li>"
                "</ul>"
                "<p>版本: 0.1.0</p>"
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
            # Stop any running worker
            if self.agent_worker and self.agent_worker.isRunning():
                self.agent_worker.stop()
            event.accept()
            logger.info("Main window closed")
        except Exception as e:
            logger.exception("Failed to handle close event")
            event.accept()