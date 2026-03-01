"""LLM configuration dialog with multi-config support."""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QTabWidget, QWidget, QTextEdit, QListWidget, QListWidgetItem,
    QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path

from ...core.config import (
    LLMConfig, LLMProvider, LLMConfigCollection,
    get_default_endpoint, get_available_models, AiderConfig
)
from ...llm.client import LLMClient
from .aider_config_dialog import AiderConfigDialog

logger = logging.getLogger(__name__)


class LLMTestThread(QThread):
    """Thread for testing LLM connection."""
    
    test_completed = pyqtSignal(bool, str)
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
    
    def run(self):
        """Run connection test."""
        import asyncio

        try:
            logger.info(f"Starting LLM connection test for config: {self.config.name}")
            logger.debug(f"Config details - Provider: {self.config.provider}, Model: {self.config.model}, Endpoint: {self.config.endpoint}")

            client = LLMClient.from_config(self.config)
            logger.debug("LLM client created successfully")

            success, message = asyncio.run(client.test_connection())

            if success:
                logger.info(f"LLM connection test successful: {message}")
            else:
                logger.warning(f"LLM connection test failed: {message}")

            self.test_completed.emit(success, message)
        except Exception as e:
            logger.exception(f"LLM connection test failed with exception: {e}")
            self.test_completed.emit(False, f"测试失败: {str(e)}")


class LLMConfigDialog(QDialog):
    """Dialog for configuring multiple LLM settings."""

    def __init__(self, config_collection: LLMConfigCollection = None, 
                 aider_config: AiderConfig = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LLM 配置管理")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(700)

        self.config_collection = config_collection or LLMConfigCollection()
        self.aider_config = aider_config or AiderConfig()
        self.current_config_id: str = None
        self.test_thread: LLMTestThread = None

        # Ensure at least one config exists
        if self.config_collection.is_empty():
            self.config_collection.create_default_config()
        
        # Set current config to default
        default_config = self.config_collection.get_default_config()
        if default_config:
            self.current_config_id = default_config.id

        self.setup_ui()
        self.refresh_config_list()
        self.load_current_config()
    
    def setup_ui(self):
        """Setup the UI with split view."""
        layout = QVBoxLayout(self)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Config list
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Config details
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes (30% left, 70% right)
        splitter.setSizes([300, 700])
        layout.addWidget(splitter)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.aider_config_btn = QPushButton("🔧 Aider 高级配置...")
        self.aider_config_btn.setToolTip("配置 Architect/Editor 双模型、多文件编辑等高级功能")
        self.aider_config_btn.clicked.connect(self.on_aider_config)
        button_layout.addWidget(self.aider_config_btn)
        
        button_layout.addStretch()
        
        self.save_btn = QPushButton("💾 保存")
        self.save_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("❌ 取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)
    
    def _create_left_panel(self) -> QWidget:
        """Create left panel with config list."""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel("📋 配置列表")
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header)
        
        # Config list
        self.config_list = QListWidget()
        self.config_list.currentItemChanged.connect(self.on_config_selected)
        layout.addWidget(self.config_list)
        
        # List buttons
        btn_layout = QHBoxLayout()
        
        self.add_btn = QPushButton("➕ 添加")
        self.add_btn.clicked.connect(self.on_add_config)
        btn_layout.addWidget(self.add_btn)
        
        self.delete_btn = QPushButton("🗑️ 删除")
        self.delete_btn.clicked.connect(self.on_delete_config)
        btn_layout.addWidget(self.delete_btn)
        
        self.default_btn = QPushButton("⭐ 设为默认")
        self.default_btn.clicked.connect(self.on_set_default)
        btn_layout.addWidget(self.default_btn)
        
        layout.addLayout(btn_layout)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create right panel with config details."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Config name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("配置名称:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("输入配置名称（如：OpenAI GPT-4）")
        name_layout.addWidget(self.name_input)
        layout.addLayout(name_layout)
        
        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #ccc;")
        layout.addWidget(line)
        
        # Provider selection
        provider_group = QGroupBox("提供商")
        provider_layout = QFormLayout()
        
        self.provider_combo = QComboBox()
        for provider in LLMProvider:
            self.provider_combo.addItem(provider.value, provider)
        self.provider_combo.currentIndexChanged.connect(self.on_provider_changed)
        provider_layout.addRow("提供商:", self.provider_combo)
        
        self.endpoint_input = QLineEdit()
        self.endpoint_input.setPlaceholderText("https://api.openai.com/v1")
        provider_layout.addRow("Endpoint:", self.endpoint_input)
        
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        provider_layout.addRow("模型:", self.model_combo)
        
        provider_group.setLayout(provider_layout)
        layout.addWidget(provider_group)
        
        # Authentication
        auth_group = QGroupBox("认证")
        auth_layout = QFormLayout()
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("输入 API Key")
        auth_layout.addRow("API Key:", self.api_key_input)
        
        # Show/Hide API key
        self.show_key_btn = QPushButton("显示")
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self.toggle_api_key_visibility)
        auth_layout.addRow("", self.show_key_btn)
        
        auth_group.setLayout(auth_layout)
        layout.addWidget(auth_group)
        
        # CA Certificate
        cert_group = QGroupBox("CA 证书配置 (可选)")
        cert_layout = QHBoxLayout()
        
        self.ca_cert_input = QLineEdit()
        self.ca_cert_input.setPlaceholderText("选择 CA 证书文件路径（用于自定义 SSL 证书）")
        cert_layout.addWidget(self.ca_cert_input)
        
        self.browse_cert_btn = QPushButton("浏览...")
        self.browse_cert_btn.clicked.connect(self.on_browse_cert)
        cert_layout.addWidget(self.browse_cert_btn)
        
        cert_group.setLayout(cert_layout)
        layout.addWidget(cert_group)
        
        # Parameters
        params_group = QGroupBox("参数")
        params_layout = QFormLayout()
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 600)
        self.timeout_spin.setValue(300)
        self.timeout_spin.setSuffix(" 秒")
        params_layout.addRow("超时:", self.timeout_spin)
        
        self.retries_spin = QSpinBox()
        self.retries_spin.setRange(0, 10)
        self.retries_spin.setValue(5)
        params_layout.addRow("重试次数:", self.retries_spin)
        
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setValue(0.7)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(1)
        params_layout.addRow("Temperature:", self.temperature_spin)
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(100, 32000)
        self.max_tokens_spin.setValue(4096)
        self.max_tokens_spin.setSingleStep(100)
        params_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Test result
        self.test_result_label = QLabel("")
        self.test_result_label.setWordWrap(True)
        layout.addWidget(self.test_result_label)

        # Test button
        test_layout = QHBoxLayout()
        self.test_btn = QPushButton("🧪 测试连接")
        self.test_btn.clicked.connect(self.test_connection)
        test_layout.addWidget(self.test_btn)
        test_layout.addStretch()
        layout.addLayout(test_layout)
        
        layout.addStretch()
        return panel
    
    def refresh_config_list(self):
        """Refresh the config list widget."""
        self.config_list.clear()
        
        for config in self.config_collection.configs:
            item = QListWidgetItem()
            display_name = config.get_display_name()
            
            # Mark default config
            if config.id == self.config_collection.default_config_id:
                display_name = "⭐ " + display_name
            
            item.setText(display_name)
            item.setData(Qt.ItemDataRole.UserRole, config.id)
            self.config_list.addItem(item)
            
            # Select current config
            if config.id == self.current_config_id:
                self.config_list.setCurrentItem(item)
    
    def load_current_config(self):
        """Load current config into UI."""
        try:
            config = self.config_collection.get_config(self.current_config_id)
            if not config:
                return
            
            # Name
            self.name_input.setText(config.name)
            
            # Provider
            index = self.provider_combo.findData(config.provider)
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)
            
            # Endpoint
            self.endpoint_input.setText(config.endpoint)
            
            # Model
            self.update_model_list()
            index = self.model_combo.findText(config.model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            else:
                self.model_combo.setCurrentText(config.model)
            
            # API Key
            api_key = config.api_key.get_secret_value()
            if api_key:
                self.api_key_input.setText(api_key)
            else:
                self.api_key_input.clear()
            
            # CA Cert
            if config.ca_cert:
                self.ca_cert_input.setText(str(config.ca_cert))
            else:
                self.ca_cert_input.clear()
            
            # Parameters
            self.timeout_spin.setValue(config.timeout)
            self.retries_spin.setValue(config.max_retries)
            self.temperature_spin.setValue(config.temperature)
            self.max_tokens_spin.setValue(config.max_tokens)
            
            # Clear test result
            self.test_result_label.clear()
        except Exception as e:
            logger.exception("Failed to load current config")
    
    def save_current_config(self):
        """Save current UI state to config."""
        try:
            if not self.current_config_id:
                return
            
            config = self.config_collection.get_config(self.current_config_id)
            if not config:
                return
            
            from pydantic import SecretStr
            
            config.name = self.name_input.text()
            config.provider = self.provider_combo.currentData()
            config.endpoint = self.endpoint_input.text()
            config.model = self.model_combo.currentText()
            config.api_key = SecretStr(self.api_key_input.text())
            
            ca_cert_text = self.ca_cert_input.text()
            config.ca_cert = Path(ca_cert_text) if ca_cert_text else None
            
            config.timeout = self.timeout_spin.value()
            config.max_retries = self.retries_spin.value()
            config.temperature = self.temperature_spin.value()
            config.max_tokens = self.max_tokens_spin.value()
        except Exception as e:
            logger.exception("Failed to save current config")
    
    def on_config_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle config selection change."""
        try:
            if previous:
                # Save previous config
                self.save_current_config()
            
            if current:
                self.current_config_id = current.data(Qt.ItemDataRole.UserRole)
                self.load_current_config()
        except Exception as e:
            logger.exception("Failed to handle config selection change")
    
    def on_add_config(self):
        """Add a new configuration."""
        try:
            # Save current first
            self.save_current_config()
            
            # Create new config
            new_config = LLMConfig(
                name=f"新配置 {len(self.config_collection.configs) + 1}",
                provider=LLMProvider.OPENAI,
                model="gpt-4"
            )
            self.config_collection.add_config(new_config)
            self.current_config_id = new_config.id
            
            self.refresh_config_list()
            
            # Select the new config
            for i in range(self.config_list.count()):
                item = self.config_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == new_config.id:
                    self.config_list.setCurrentItem(item)
                    break
            logger.info(f"Added new config: {new_config.name}")
        except Exception as e:
            logger.exception("Failed to add new config")
    
    def on_delete_config(self):
        """Delete current configuration."""
        try:
            if len(self.config_collection.configs) <= 1:
                QMessageBox.warning(
                    self,
                    "无法删除",
                    "至少需要保留一个配置"
                )
                return
            
            config = self.config_collection.get_config(self.current_config_id)
            if not config:
                return
            
            reply = QMessageBox.question(
                self,
                "确认删除",
                f"确定要删除配置 \"{config.get_display_name()}\" 吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                config_name = config.get_display_name()
                self.config_collection.remove_config(self.current_config_id)
                
                # Select default or first config
                default_config = self.config_collection.get_default_config()
                self.current_config_id = default_config.id if default_config else None
                
                self.refresh_config_list()
                self.load_current_config()
                logger.info(f"Deleted config: {config_name}")
        except Exception as e:
            logger.exception("Failed to delete config")
    
    def on_set_default(self):
        """Set current config as default."""
        try:
            if self.current_config_id:
                self.config_collection.set_default_config(self.current_config_id)
                self.refresh_config_list()
                logger.info(f"Set default config: {self.current_config_id}")
        except Exception as e:
            logger.exception("Failed to set default config")
    
    def update_model_list(self):
        """Update model list based on provider."""
        provider = self.provider_combo.currentData()
        models = get_available_models(provider)
        
        self.model_combo.clear()
        for model in models:
            self.model_combo.addItem(model)
        
        # Add custom option
        self.model_combo.addItem("自定义...")
    
    def on_provider_changed(self, index: int):
        """Handle provider change."""
        provider = self.provider_combo.currentData()
        
        # Update endpoint
        default_endpoint = get_default_endpoint(provider)
        self.endpoint_input.setText(default_endpoint)
        
        # Update model list
        self.update_model_list()
    
    def toggle_api_key_visibility(self, checked: bool):
        """Toggle API key visibility."""
        if checked:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.show_key_btn.setText("隐藏")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.show_key_btn.setText("显示")
    
    def on_browse_cert(self):
        """Browse for CA certificate file."""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择 CA 证书文件",
                "",
                "Certificate Files (*.crt *.pem *.cer);;All Files (*.*)"
            )
            if file_path:
                self.ca_cert_input.setText(file_path)
                logger.info(f"Selected CA certificate: {file_path}")
        except Exception as e:
            logger.exception("Failed to browse for CA certificate")
    
    def test_connection(self):
        """Test LLM connection."""
        try:
            # Save current config first
            self.save_current_config()
            
            config = self.config_collection.get_config(self.current_config_id)
            if not config:
                return
            
            if not config.is_configured():
                QMessageBox.warning(
                    self,
                    "配置不完整",
                    "请先填写 API Key 和 Endpoint"
                )
                return
            
            self.test_btn.setEnabled(False)
            self.test_btn.setText("测试中...")
            self.test_result_label.setText("正在测试连接...")
            
            self.test_thread = LLMTestThread(config)
            self.test_thread.test_completed.connect(self.on_test_completed)
            self.test_thread.start()
            logger.info(f"Started connection test for config: {config.name}")
        except Exception as e:
            logger.exception("Failed to start connection test")
    
    def on_test_completed(self, success: bool, message: str):
        """Handle test completion."""
        try:
            self.test_btn.setEnabled(True)
            self.test_btn.setText("🧪 测试连接")
            
            if success:
                self.test_result_label.setText(f"✅ {message}")
                self.test_result_label.setStyleSheet("color: green;")
                logger.info(f"Connection test successful: {message}")
            else:
                self.test_result_label.setText(f"❌ {message}")
                self.test_result_label.setStyleSheet("color: red;")
                logger.warning(f"Connection test failed: {message}")
        except Exception as e:
            logger.exception("Failed to handle test completion")
    
    def get_config_collection(self) -> LLMConfigCollection:
        """Get configuration collection from UI."""
        self.save_current_config()
        return self.config_collection
    
    def accept(self):
        """Accept dialog."""
        self.save_current_config()
        super().accept()
    
    def on_aider_config(self):
        """Open Aider configuration dialog."""
        try:
            # Save current config first
            self.save_current_config()
            
            dialog = AiderConfigDialog(
                self.aider_config, 
                self.config_collection,
                self
            )
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.aider_config = dialog.config
                logger.info("Aider config updated")
        except Exception as e:
            logger.exception("Failed to open Aider config dialog")

    def closeEvent(self, event):
        """Handle close event."""
        try:
            if self.test_thread and self.test_thread.isRunning():
                self.test_thread.terminate()
                self.test_thread.wait()
            event.accept()
            logger.info("LLM config dialog closed")
        except Exception as e:
            logger.exception("Failed to handle close event")
            event.accept()
