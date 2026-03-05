"""JaCoCo configuration dialog for automatic setup.

This dialog provides a user-friendly interface for:
- Checking current JaCoCo configuration status
- Generating configuration using LLM
- Previewing configuration changes
- Applying configuration to pom.xml
- Installing dependencies
"""

import logging
from pathlib import Path
from typing import Optional, Any

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QTextEdit, QCheckBox, QMessageBox, QProgressBar,
    QSplitter, QWidget, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

from ...services.jacoco_config_service import JacocoConfigService, JacocoConfigResult
from ...core.config import get_settings

logger = logging.getLogger(__name__)


class JacocoConfigWorker(QThread):
    """Worker thread for JaCoCo configuration operations."""
    
    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    config_generated = pyqtSignal(dict)
    config_applied = pyqtSignal(object)  # JacocoConfigResult
    error_occurred = pyqtSignal(str)
    finished_success = pyqtSignal()
    
    def __init__(self, project_path: str, llm_client: Optional[Any] = None, operation: str = "auto"):
        super().__init__()
        self.project_path = project_path
        self.llm_client = llm_client
        self.operation = operation  # "detect", "generate", "apply", "auto"
        self.service = JacocoConfigService(project_path, llm_client)
        self.generated_config: Optional[dict] = None
        
    def run(self):
        """Run the configuration operation."""
        try:
            if self.operation == "detect":
                self._detect_configuration()
            elif self.operation == "generate":
                self._generate_configuration()
            elif self.operation == "apply":
                self._apply_configuration()
            elif self.operation == "auto":
                self._auto_configure()
        except Exception as e:
            logger.exception(f"[JacocoConfigWorker] Error: {e}")
            self.error_occurred.emit(str(e))
    
    def _detect_configuration(self):
        """Detect current JaCoCo configuration."""
        self.status_update.emit("正在检测 JaCoCo 配置...")
        self.progress_update.emit(10)
        
        result = self.service.check_jacoco_configured()
        
        self.progress_update.emit(100)
        
        if result.is_configured:
            version = result.plugin_version or "未知"
            self.status_update.emit(f"✅ JaCoCo 已配置 (版本: {version})")
        else:
            self.status_update.emit("⚠️ JaCoCo 未配置")
        
        self.finished_success.emit()
    
    def _generate_configuration(self):
        """Generate configuration using LLM."""
        self.status_update.emit("正在使用 LLM 生成配置...")
        self.progress_update.emit(20)
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        config = loop.run_until_complete(
            self.service.generate_config_with_llm()
        )
        
        self.generated_config = config
        self.progress_update.emit(80)
        
        self.config_generated.emit(config)
        self.status_update.emit("✅ 配置生成完成")
        self.progress_update.emit(100)
        self.finished_success.emit()
    
    def _apply_configuration(self):
        """Apply configuration to pom.xml."""
        if self.generated_config is None:
            self.error_occurred.emit("没有可应用的配置，请先生成配置")
            return
        
        self.status_update.emit("正在应用配置到 pom.xml...")
        self.progress_update.emit(30)
        
        result = self.service.apply_config(self.generated_config)
        
        self.progress_update.emit(70)
        
        if result.success:
            self.status_update.emit("✅ 配置已应用到 pom.xml")
            
            # Install dependencies
            self.status_update.emit("正在安装依赖...")
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            deps_success, deps_msg = loop.run_until_complete(
                self.service.install_dependencies()
            )
            
            if deps_success:
                result.dependencies_installed = True
                self.status_update.emit("✅ 依赖安装完成")
            else:
                self.status_update.emit(f"⚠️ 依赖安装警告: {deps_msg}")
            
            self.progress_update.emit(100)
            self.config_applied.emit(result)
        else:
            self.error_occurred.emit(f"应用配置失败: {result.message}")
        
        self.finished_success.emit()
    
    async def _auto_configure_async(self):
        """Async auto-configuration."""
        return await self.service.auto_configure(skip_if_exists=False)
    
    def _auto_configure(self):
        """Run full auto-configuration."""
        self.status_update.emit("开始自动配置 JaCoCo...")
        self.progress_update.emit(10)
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(self._auto_configure_async())
        
        self.progress_update.emit(100)
        
        if result.success:
            if result.applied:
                self.status_update.emit("✅ JaCoCo 自动配置完成")
            else:
                self.status_update.emit(f"ℹ️ {result.message}")
            self.config_applied.emit(result)
        else:
            self.error_occurred.emit(f"自动配置失败: {result.message}")
        
        self.finished_success.emit()


class JacocoConfigDialog(QDialog):
    """Dialog for JaCoCo configuration.
    
    Provides a user-friendly interface for configuring JaCoCo
    with automatic LLM-powered configuration generation.
    """
    
    def __init__(self, project_path: str, llm_client: Optional[Any] = None, parent=None):
        super().__init__(parent)
        self.project_path = project_path
        self.llm_client = llm_client
        self.service = JacocoConfigService(project_path, llm_client)
        self.current_config: Optional[dict] = None
        self.last_result: Optional[JacocoConfigResult] = None
        self.worker: Optional[JacocoConfigWorker] = None
        
        self.setWindowTitle("JaCoCo 配置")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)
        
        self.setup_ui()
        self.check_initial_status()
        
        logger.info("[JacocoConfigDialog] Initialized")
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title_label = QLabel("🔧 JaCoCo 代码覆盖率工具配置")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Status group
        status_group = QGroupBox("当前状态")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("正在检测...")
        self.status_label.setStyleSheet("font-size: 13px; padding: 10px;")
        status_layout.addWidget(self.status_label)
        
        self.status_details = QLabel("")
        self.status_details.setStyleSheet("color: #666; font-size: 11px;")
        status_layout.addWidget(self.status_details)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Configuration preview group
        preview_group = QGroupBox("配置预览")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(250)
        self.preview_text.setPlaceholderText("点击\"生成配置\"按钮查看配置预览...")
        preview_layout.addWidget(self.preview_text)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Options group
        options_group = QGroupBox("配置选项")
        options_layout = QVBoxLayout()
        
        self.auto_use_llm_check = QCheckBox("使用 LLM 智能生成配置（推荐）")
        self.auto_use_llm_check.setChecked(True)
        self.auto_use_llm_check.setToolTip("使用 LLM 分析项目结构并生成最优配置")
        options_layout.addWidget(self.auto_use_llm_check)
        
        self.skip_if_exists_check = QCheckBox("如果已配置则跳过")
        self.skip_if_exists_check.setChecked(True)
        self.skip_if_exists_check.setToolTip("如果检测到 JaCoCo 已配置，则跳过配置过程")
        options_layout.addWidget(self.skip_if_exists_check)
        
        self.create_backup_check = QCheckBox("创建备份")
        self.create_backup_check.setChecked(True)
        self.create_backup_check.setToolTip("修改 pom.xml 前自动创建备份")
        options_layout.addWidget(self.create_backup_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.detect_btn = QPushButton("🔍 检测配置")
        self.detect_btn.clicked.connect(self.on_detect)
        button_layout.addWidget(self.detect_btn)
        
        self.generate_btn = QPushButton("📝 生成配置")
        self.generate_btn.clicked.connect(self.on_generate)
        button_layout.addWidget(self.generate_btn)
        
        self.apply_btn = QPushButton("✅ 应用配置")
        self.apply_btn.clicked.connect(self.on_apply)
        self.apply_btn.setEnabled(False)
        button_layout.addWidget(self.apply_btn)
        
        self.auto_btn = QPushButton("🚀 一键配置")
        self.auto_btn.clicked.connect(self.on_auto_configure)
        self.auto_btn.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(self.auto_btn)
        
        button_layout.addStretch()
        
        self.close_btn = QPushButton("❌ 关闭")
        self.close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Help text
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setMaximumHeight(120)
        help_text.setText(
            "📚 JaCoCo 配置说明:\n\n"
            "1. 检测配置 - 检查当前项目是否已配置 JaCoCo\n"
            "2. 生成配置 - 使用 LLM 分析项目并生成最优配置\n"
            "3. 应用配置 - 将生成的配置写入 pom.xml\n"
            "4. 一键配置 - 自动完成检测、生成、应用全流程\n\n"
            "💡 提示: 配置完成后，运行 'mvn test jacoco:report' 生成覆盖率报告"
        )
        help_text.setStyleSheet("color: #666; font-size: 11px; background-color: #f5f5f5;")
        layout.addWidget(help_text)
    
    def check_initial_status(self):
        """Check initial JaCoCo configuration status."""
        self.on_detect()
    
    def set_buttons_enabled(self, enabled: bool):
        """Enable/disable buttons during operations."""
        self.detect_btn.setEnabled(enabled)
        self.generate_btn.setEnabled(enabled)
        self.apply_btn.setEnabled(enabled and self.current_config is not None)
        self.auto_btn.setEnabled(enabled)
    
    def start_worker(self, operation: str):
        """Start a worker thread for the operation."""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "警告", "有操作正在进行中，请稍候...")
            return
        
        self.set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = JacocoConfigWorker(
            self.project_path,
            self.llm_client if self.auto_use_llm_check.isChecked() else None,
            operation
        )
        
        self.worker.status_update.connect(self.on_status_update)
        self.worker.progress_update.connect(self.on_progress_update)
        self.worker.config_generated.connect(self.on_config_generated)
        self.worker.config_applied.connect(self.on_config_applied)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.finished_success.connect(self.on_worker_finished)
        
        self.worker.start()
    
    def on_status_update(self, message: str):
        """Handle status update from worker."""
        self.status_label.setText(message)
        logger.info(f"[JacocoConfigDialog] Status: {message}")
    
    def on_progress_update(self, value: int):
        """Handle progress update from worker."""
        self.progress_bar.setValue(value)
    
    def on_config_generated(self, config: dict):
        """Handle config generation completion."""
        self.current_config = config
        preview = self.service.generate_config_preview(config)
        self.preview_text.setText(preview)
        self.apply_btn.setEnabled(True)
    
    def on_config_applied(self, result: JacocoConfigResult):
        """Handle config application completion."""
        self.last_result = result
        
        if result.success:
            details = []
            if result.backup_path:
                details.append(f"备份路径: {result.backup_path}")
            if result.dependencies_installed:
                details.append("依赖已安装")
            
            self.status_details.setText("\n".join(details))
            
            QMessageBox.information(
                self,
                "配置成功",
                f"JaCoCo 配置已成功应用！\n\n{result.message}"
            )
        else:
            QMessageBox.warning(
                self,
                "配置失败",
                f"配置应用失败:\n{result.message}"
            )
    
    def on_error(self, error_message: str):
        """Handle error from worker."""
        logger.error(f"[JacocoConfigDialog] Error: {error_message}")
        QMessageBox.critical(self, "错误", f"操作失败:\n{error_message}")
    
    def on_worker_finished(self):
        """Handle worker completion."""
        self.set_buttons_enabled(True)
        self.worker = None
    
    def on_detect(self):
        """Handle detect button click."""
        self.start_worker("detect")
    
    def on_generate(self):
        """Handle generate button click."""
        self.start_worker("generate")
    
    def on_apply(self):
        """Handle apply button click."""
        if self.current_config is None:
            QMessageBox.warning(self, "警告", "请先生成配置")
            return
        
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "确认应用",
            "确定要将 JaCoCo 配置应用到 pom.xml 吗？\n\n"
            "此操作将修改 pom.xml 文件。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.worker = JacocoConfigWorker(
                self.project_path,
                self.llm_client if self.auto_use_llm_check.isChecked() else None,
                "apply"
            )
            self.worker.generated_config = self.current_config
            
            self.worker.status_update.connect(self.on_status_update)
            self.worker.progress_update.connect(self.on_progress_update)
            self.worker.config_applied.connect(self.on_config_applied)
            self.worker.error_occurred.connect(self.on_error)
            self.worker.finished_success.connect(self.on_worker_finished)
            
            self.set_buttons_enabled(False)
            self.progress_bar.setValue(0)
            self.worker.start()
    
    def on_auto_configure(self):
        """Handle auto-configure button click."""
        # Confirm with user
        reply = QMessageBox.question(
            self,
            "确认一键配置",
            "确定要自动配置 JaCoCo 吗？\n\n"
            "此操作将:\n"
            "1. 检测当前配置\n"
            "2. 使用 LLM 生成配置（如果启用）\n"
            "3. 应用配置到 pom.xml\n"
            "4. 安装依赖\n\n"
            "pom.xml 将被修改，建议先创建备份。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.start_worker("auto")
    
    def get_result(self) -> Optional[JacocoConfigResult]:
        """Get the last configuration result.
        
        Returns:
            Last JacocoConfigResult or None
        """
        return self.last_result
