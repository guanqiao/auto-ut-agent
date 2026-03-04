"""Setup wizard for first-time configuration."""

import logging
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QWidget, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QProgressBar, QTextEdit, QGroupBox,
    QFormLayout, QFileDialog, QMessageBox, QCheckBox,
    QWizard, QWizardPage
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

from ...core.config import (
    LLMConfig, LLMProvider, LLMConfigCollection,
    AiderConfig, save_llm_config, save_aider_config,
    get_settings
)
from ...llm.client import LLMClient
from ..styles import get_style_manager
from ..components import get_notification_manager

logger = logging.getLogger(__name__)


class TestConnectionThread(QThread):
    """Thread for testing LLM connection."""
    
    finished = pyqtSignal(bool, str)
    
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
    
    def run(self):
        """Test connection."""
        try:
            import asyncio
            client = LLMClient.from_config(self.config)
            success, message = asyncio.run(client.test_connection())
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, str(e))


class WelcomePage(QWizardPage):
    """Welcome page."""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to PyUT Agent")
        self.setSubTitle("Let's set up your AI unit test generator")
        
        layout = QVBoxLayout(self)
        
        intro = QLabel(
            "<h2>🚀 PyUT Agent Setup Wizard</h2>"
            "<p>This wizard will guide you through the initial configuration:</p>"
            "<ul>"
            "<li>Configure LLM (OpenAI, Claude, etc.)</li>"
            "<li>Set up JDK and Maven paths</li>"
            "<li>Configure test generation preferences</li>"
            "</ul>"
            "<p>Click <b>Next</b> to begin.</p>"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        
        layout.addStretch()


class LLMConfigPage(QWizardPage):
    """LLM configuration page."""
    
    def __init__(self):
        super().__init__()
        self.setTitle("LLM Configuration")
        self.setSubTitle("Configure your AI model")
        
        self._config_collection: Optional[LLMConfigCollection] = None
        self._test_thread: Optional[TestConnectionThread] = None
        
        layout = QVBoxLayout(self)
        
        # Preset selection
        preset_group = QGroupBox("Quick Setup")
        preset_layout = QVBoxLayout(preset_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Select a preset...", "")
        self.preset_combo.addItem("OpenAI GPT-4", "openai_gpt4")
        self.preset_combo.addItem("OpenAI GPT-3.5", "openai_gpt35")
        self.preset_combo.addItem("Claude (Anthropic)", "claude")
        self.preset_combo.addItem("DeepSeek", "deepseek")
        self.preset_combo.addItem("Ollama (Local)", "ollama")
        self.preset_combo.addItem("Custom", "custom")
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        
        layout.addWidget(preset_group)
        
        # Manual configuration
        manual_group = QGroupBox("Manual Configuration")
        form_layout = QFormLayout(manual_group)
        
        self.provider_combo = QComboBox()
        for provider in LLMProvider:
            self.provider_combo.addItem(provider.value, provider)
        form_layout.addRow("Provider:", self.provider_combo)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("My LLM Config")
        form_layout.addRow("Name:", self.name_input)
        
        self.endpoint_input = QLineEdit()
        self.endpoint_input.setPlaceholderText("https://api.openai.com/v1")
        form_layout.addRow("Endpoint:", self.endpoint_input)
        
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("gpt-4")
        form_layout.addRow("Model:", self.model_input)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("sk-...")
        form_layout.addRow("API Key:", self.api_key_input)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(10, 600)
        self.timeout_spin.setValue(300)
        self.timeout_spin.setSuffix("s")
        form_layout.addRow("Timeout:", self.timeout_spin)
        
        layout.addWidget(manual_group)
        
        # Test connection
        test_layout = QHBoxLayout()
        self.test_btn = QPushButton("🧪 Test Connection")
        self.test_btn.clicked.connect(self.on_test_connection)
        test_layout.addWidget(self.test_btn)
        
        self.test_result = QLabel("")
        test_layout.addWidget(self.test_result)
        test_layout.addStretch()
        
        layout.addLayout(test_layout)
        layout.addStretch()
    
    def on_preset_changed(self, preset_name: str):
        """Handle preset selection."""
        preset = self.preset_combo.currentData()
        
        presets = {
            "openai_gpt4": {
                "provider": LLMProvider.OPENAI,
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-4",
                "name": "OpenAI GPT-4"
            },
            "openai_gpt35": {
                "provider": LLMProvider.OPENAI,
                "endpoint": "https://api.openai.com/v1",
                "model": "gpt-3.5-turbo",
                "name": "OpenAI GPT-3.5"
            },
            "claude": {
                "provider": LLMProvider.ANTHROPIC,
                "endpoint": "https://api.anthropic.com",
                "model": "claude-3-opus-20240229",
                "name": "Claude 3 Opus"
            },
            "deepseek": {
                "provider": LLMProvider.DEEPSEEK,
                "endpoint": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "name": "DeepSeek"
            },
            "ollama": {
                "provider": LLMProvider.OLLAMA,
                "endpoint": "http://localhost:11434",
                "model": "llama2",
                "name": "Ollama Local"
            }
        }
        
        if preset in presets:
            p = presets[preset]
            self.provider_combo.setCurrentText(p["provider"].value)
            self.endpoint_input.setText(p["endpoint"])
            self.model_input.setText(p["model"])
            self.name_input.setText(p["name"])
    
    def on_test_connection(self):
        """Test LLM connection."""
        config = self.get_config()
        if not config:
            self.test_result.setText("❌ Please fill in all required fields")
            return
        
        self.test_btn.setEnabled(False)
        self.test_result.setText("🔄 Testing...")
        
        self._test_thread = TestConnectionThread(config)
        self._test_thread.finished.connect(self.on_test_finished)
        self._test_thread.start()
    
    def on_test_finished(self, success: bool, message: str):
        """Handle test completion."""
        self.test_btn.setEnabled(True)
        if success:
            self.test_result.setText(f"✅ {message}")
        else:
            self.test_result.setText(f"❌ {message}")
    
    def get_config(self) -> Optional[LLMConfig]:
        """Get LLM config from form."""
        name = self.name_input.text().strip()
        endpoint = self.endpoint_input.text().strip()
        model = self.model_input.text().strip()
        api_key = self.api_key_input.text().strip()
        
        if not all([name, endpoint, model, api_key]):
            return None
        
        provider = self.provider_combo.currentData()
        
        return LLMConfig(
            name=name,
            provider=provider,
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            timeout=self.timeout_spin.value()
        )
    
    def validatePage(self) -> bool:
        """Validate page before proceeding."""
        config = self.get_config()
        if not config:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please fill in all required fields."
            )
            return False
        return True


class JDKMavenPage(QWizardPage):
    """JDK and Maven configuration page."""
    
    def __init__(self):
        super().__init__()
        self.setTitle("JDK & Maven")
        self.setSubTitle("Configure Java development environment")
        
        layout = QVBoxLayout(self)
        
        # JDK
        jdk_group = QGroupBox("JDK Configuration")
        jdk_layout = QFormLayout(jdk_group)
        
        jdk_path_layout = QHBoxLayout()
        self.jdk_input = QLineEdit()
        self.jdk_input.setPlaceholderText("Auto-detect")
        jdk_path_layout.addWidget(self.jdk_input)
        
        jdk_browse_btn = QPushButton("Browse...")
        jdk_browse_btn.clicked.connect(self.on_browse_jdk)
        jdk_path_layout.addWidget(jdk_browse_btn)
        
        jdk_layout.addRow("JAVA_HOME:", jdk_path_layout)
        
        self.auto_jdk_check = QCheckBox("Auto-detect JDK")
        self.auto_jdk_check.setChecked(True)
        self.auto_jdk_check.toggled.connect(self.on_auto_jdk_toggled)
        jdk_layout.addRow("", self.auto_jdk_check)
        
        layout.addWidget(jdk_group)
        
        # Maven
        maven_group = QGroupBox("Maven Configuration")
        maven_layout = QFormLayout(maven_group)
        
        maven_path_layout = QHBoxLayout()
        self.maven_input = QLineEdit()
        self.maven_input.setPlaceholderText("Auto-detect")
        maven_path_layout.addWidget(self.maven_input)
        
        maven_browse_btn = QPushButton("Browse...")
        maven_browse_btn.clicked.connect(self.on_browse_maven)
        maven_path_layout.addWidget(maven_browse_btn)
        
        maven_layout.addRow("Maven Path:", maven_path_layout)
        
        self.auto_maven_check = QCheckBox("Auto-detect Maven")
        self.auto_maven_check.setChecked(True)
        self.auto_maven_check.toggled.connect(self.on_auto_maven_toggled)
        maven_layout.addRow("", self.auto_maven_check)
        
        layout.addWidget(maven_group)
        layout.addStretch()
    
    def on_browse_jdk(self):
        """Browse for JDK."""
        path = QFileDialog.getExistingDirectory(self, "Select JDK Directory")
        if path:
            self.jdk_input.setText(path)
            self.auto_jdk_check.setChecked(False)
    
    def on_browse_maven(self):
        """Browse for Maven."""
        path = QFileDialog.getExistingDirectory(self, "Select Maven Directory")
        if path:
            self.maven_input.setText(path)
            self.auto_maven_check.setChecked(False)
    
    def on_auto_jdk_toggled(self, checked: bool):
        """Handle auto JDK toggle."""
        self.jdk_input.setEnabled(not checked)
    
    def on_auto_maven_toggled(self, checked: bool):
        """Handle auto Maven toggle."""
        self.maven_input.setEnabled(not checked)
    
    def get_jdk_path(self) -> str:
        """Get JDK path."""
        if self.auto_jdk_check.isChecked():
            return ""
        return self.jdk_input.text().strip()
    
    def get_maven_path(self) -> str:
        """Get Maven path."""
        if self.auto_maven_check.isChecked():
            return ""
        return self.maven_input.text().strip()


class TestPreferencesPage(QWizardPage):
    """Test generation preferences page."""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Test Preferences")
        self.setSubTitle("Configure test generation settings")
        
        layout = QVBoxLayout(self)
        
        prefs_group = QGroupBox("Generation Settings")
        form_layout = QFormLayout(prefs_group)
        
        self.coverage_spin = QDoubleSpinBox()
        self.coverage_spin.setRange(0.1, 1.0)
        self.coverage_spin.setValue(0.8)
        self.coverage_spin.setSingleStep(0.1)
        self.coverage_spin.setSuffix(" (80%)")
        self.coverage_spin.valueChanged.connect(self.on_coverage_changed)
        form_layout.addRow("Target Coverage:", self.coverage_spin)
        
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 10)
        self.iterations_spin.setValue(2)
        form_layout.addRow("Max Iterations:", self.iterations_spin)
        
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(60, 1800)
        self.timeout_spin.setValue(300)
        self.timeout_spin.setSuffix("s")
        form_layout.addRow("Timeout per File:", self.timeout_spin)
        
        layout.addWidget(prefs_group)
        
        # Info
        info = QLabel(
            "<p><b>Tips:</b></p>"
            "<ul>"
            "<li><b>Target Coverage:</b> Higher values may take longer but produce more comprehensive tests</li>"
            "<li><b>Max Iterations:</b> Number of attempts to improve coverage</li>"
            "<li><b>Timeout:</b> Maximum time spent on each file</li>"
            "</ul>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        layout.addStretch()
    
    def on_coverage_changed(self, value: float):
        """Handle coverage change."""
        self.coverage_spin.setSuffix(f" ({value:.0%})")
    
    def get_coverage_target(self) -> float:
        """Get coverage target."""
        return self.coverage_spin.value()
    
    def get_max_iterations(self) -> int:
        """Get max iterations."""
        return self.iterations_spin.value()
    
    def get_timeout(self) -> int:
        """Get timeout."""
        return self.timeout_spin.value()


class SummaryPage(QWizardPage):
    """Summary page."""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Summary")
        self.setSubTitle("Review your configuration")
        
        layout = QVBoxLayout(self)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
    
    def initializePage(self):
        """Initialize page with summary."""
        wizard = self.wizard()
        
        summary = "<h2>Configuration Summary</h2>"
        
        # LLM
        llm_page = wizard.page(1)
        if llm_page:
            config = llm_page.get_config()
            if config:
                summary += f"""
                <h3>LLM Configuration</h3>
                <ul>
                <li><b>Name:</b> {config.name}</li>
                <li><b>Provider:</b> {config.provider}</li>
                <li><b>Model:</b> {config.model}</li>
                <li><b>Endpoint:</b> {config.endpoint}</li>
                </ul>
                """
        
        # JDK/Maven
        jdk_page = wizard.page(2)
        if jdk_page:
            jdk_path = jdk_page.get_jdk_path()
            maven_path = jdk_page.get_maven_path()
            summary += f"""
            <h3>Environment</h3>
            <ul>
            <li><b>JDK:</b> {jdk_path or "Auto-detect"}</li>
            <li><b>Maven:</b> {maven_path or "Auto-detect"}</li>
            </ul>
            """
        
        # Preferences
        prefs_page = wizard.page(3)
        if prefs_page:
            summary += f"""
            <h3>Test Preferences</h3>
            <ul>
            <li><b>Target Coverage:</b> {prefs_page.get_coverage_target():.0%}</li>
            <li><b>Max Iterations:</b> {prefs_page.get_max_iterations()}</li>
            <li><b>Timeout:</b> {prefs_page.get_timeout()}s</li>
            </ul>
            """
        
        summary += "<p>Click <b>Finish</b> to save these settings.</p>"
        
        self.summary_text.setHtml(summary)


class SetupWizard(QWizard):
    """Setup wizard for first-time configuration."""
    
    finished = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("PyUT Agent - Setup Wizard")
        self.setMinimumSize(700, 500)
        
        self._style_manager = get_style_manager()
        
        # Add pages
        self.addPage(WelcomePage())
        self.addPage(LLMConfigPage())
        self.addPage(JDKMavenPage())
        self.addPage(TestPreferencesPage())
        self.addPage(SummaryPage())
        
        # Set options
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setOption(QWizard.WizardOption.IndependentPages, False)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setOption(QWizard.WizardOption.NoCancelButtonOnLastPage, False)
    
    def accept(self):
        """Handle finish button."""
        try:
            self.save_configuration()
            get_notification_manager().show_success("Configuration saved successfully!")
            self.finished.emit()
            super().accept()
        except Exception as e:
            logger.exception("Failed to save configuration")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save configuration: {e}"
            )
    
    def save_configuration(self):
        """Save all configurations."""
        # LLM Config
        llm_page = self.page(1)
        if llm_page:
            config = llm_page.get_config()
            if config:
                collection = LLMConfigCollection()
                collection.add_config(config)
                collection.set_default_config(config.id)
                save_llm_config(collection)
                logger.info(f"Saved LLM config: {config.name}")
        
        # JDK/Maven
        jdk_page = self.page(2)
        if jdk_page:
            settings = get_settings()
            
            jdk_path = jdk_page.get_jdk_path()
            if jdk_path:
                settings.jdk.java_home = jdk_path
            
            maven_path = jdk_page.get_maven_path()
            if maven_path:
                settings.maven.maven_path = maven_path
            
            settings.save()
            logger.info("Saved JDK/Maven settings")
        
        # Test Preferences
        prefs_page = self.page(3)
        if prefs_page:
            settings = get_settings()
            settings.coverage.target_coverage = prefs_page.get_coverage_target()
            settings.coverage.max_iterations = prefs_page.get_max_iterations()
            settings.coverage.timeout_per_file = prefs_page.get_timeout()
            settings.save()
            logger.info("Saved test preferences")


def run_setup_wizard(parent=None) -> bool:
    """Run the setup wizard.
    
    Args:
        parent: Parent widget
        
    Returns:
        True if wizard was completed
    """
    wizard = SetupWizard(parent)
    result = wizard.exec()
    return result == QWizard.DialogCode.Accepted
