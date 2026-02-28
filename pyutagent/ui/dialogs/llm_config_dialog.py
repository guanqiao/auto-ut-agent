"""LLM configuration dialog."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QTabWidget, QWidget, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from pathlib import Path

from ...llm.config import (
    LLMConfig, LLMProvider, 
    get_default_endpoint, get_available_models
)
from ...llm.client import LLMClient


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
            client = LLMClient.from_config(self.config)
            success, message = asyncio.run(client.test_connection())
            self.test_completed.emit(success, message)
        except Exception as e:
            self.test_completed.emit(False, f"测试失败: {str(e)}")


class LLMConfigDialog(QDialog):
    """Dialog for configuring LLM settings."""
    
    def __init__(self, config: LLMConfig = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LLM 配置")
        self.setMinimumWidth(500)
        
        self.config = config or LLMConfig()
        self.test_thread: LLMTestThread = None
        
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        
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
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("🧪 测试连接")
        self.test_btn.clicked.connect(self.test_connection)
        button_layout.addWidget(self.test_btn)
        
        button_layout.addStretch()
        
        self.save_btn = QPushButton("💾 保存")
        self.save_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("❌ 取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def load_config(self):
        """Load configuration into UI."""
        # Provider
        index = self.provider_combo.findData(self.config.provider)
        if index >= 0:
            self.provider_combo.setCurrentIndex(index)
        
        # Endpoint
        self.endpoint_input.setText(self.config.endpoint)
        
        # Model
        self.update_model_list()
        index = self.model_combo.findText(self.config.model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        else:
            self.model_combo.setCurrentText(self.config.model)
        
        # API Key
        api_key = self.config.api_key.get_secret_value()
        if api_key:
            self.api_key_input.setText(api_key)
        
        # Parameters
        self.timeout_spin.setValue(self.config.timeout)
        self.retries_spin.setValue(self.config.max_retries)
        self.temperature_spin.setValue(self.config.temperature)
        self.max_tokens_spin.setValue(self.config.max_tokens)
    
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
    
    def test_connection(self):
        """Test LLM connection."""
        config = self.get_config()
        
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
    
    def on_test_completed(self, success: bool, message: str):
        """Handle test completion."""
        self.test_btn.setEnabled(True)
        self.test_btn.setText("🧪 测试连接")
        
        if success:
            self.test_result_label.setText(f"✅ {message}")
            self.test_result_label.setStyleSheet("color: green;")
        else:
            self.test_result_label.setText(f"❌ {message}")
            self.test_result_label.setStyleSheet("color: red;")
    
    def get_config(self) -> LLMConfig:
        """Get configuration from UI."""
        return LLMConfig(
            provider=self.provider_combo.currentData(),
            endpoint=self.endpoint_input.text(),
            api_key=self.api_key_input.text(),
            model=self.model_combo.currentText(),
            timeout=self.timeout_spin.value(),
            max_retries=self.retries_spin.value(),
            temperature=self.temperature_spin.value(),
            max_tokens=self.max_tokens_spin.value(),
        )
    
    def accept(self):
        """Accept dialog."""
        self.config = self.get_config()
        super().accept()
    
    def closeEvent(self, event):
        """Handle close event."""
        if self.test_thread and self.test_thread.isRunning():
            self.test_thread.terminate()
            self.test_thread.wait()
        event.accept()
