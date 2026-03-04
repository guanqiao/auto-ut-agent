"""Config overview dialog for PyUT Agent."""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class ConfigOverviewDialog(QDialog):
    """Dialog showing configuration overview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration Overview")
        self.setMinimumSize(700, 500)

        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.llm_tab = QWidget()
        self.setup_llm_tab()
        self.tab_widget.addTab(self.llm_tab, "LLM")

        self.maven_tab = QWidget()
        self.setup_maven_tab()
        self.tab_widget.addTab(self.maven_tab, "Maven")

        self.jdk_tab = QWidget()
        self.setup_jdk_tab()
        self.tab_widget.addTab(self.jdk_tab, "JDK")

        self.coverage_tab = QWidget()
        self.setup_coverage_tab()
        self.tab_widget.addTab(self.coverage_tab, "Coverage")

        self.aider_tab = QWidget()
        self.setup_aider_tab()
        self.tab_widget.addTab(self.aider_tab, "Aider")

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.copy_button = QPushButton("Copy All")
        self.copy_button.clicked.connect(self.copy_all)
        button_layout.addWidget(self.copy_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)

    def setup_llm_tab(self):
        """Setup LLM configuration tab."""
        layout = QVBoxLayout(self.llm_tab)

        self.llm_text = QTextEdit()
        self.llm_text.setReadOnly(True)
        self.llm_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.llm_text)

    def setup_maven_tab(self):
        """Setup Maven configuration tab."""
        layout = QVBoxLayout(self.maven_tab)

        self.maven_text = QTextEdit()
        self.maven_text.setReadOnly(True)
        self.maven_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.maven_text)

    def setup_jdk_tab(self):
        """Setup JDK configuration tab."""
        layout = QVBoxLayout(self.jdk_tab)

        self.jdk_text = QTextEdit()
        self.jdk_text.setReadOnly(True)
        self.jdk_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.jdk_text)

    def setup_coverage_tab(self):
        """Setup Coverage configuration tab."""
        layout = QVBoxLayout(self.coverage_tab)

        self.coverage_text = QTextEdit()
        self.coverage_text.setReadOnly(True)
        self.coverage_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.coverage_text)

    def setup_aider_tab(self):
        """Setup Aider configuration tab."""
        layout = QVBoxLayout(self.aider_tab)

        self.aider_text = QTextEdit()
        self.aider_text.setReadOnly(True)
        self.aider_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.aider_text)

    def load_config(self):
        """Load and display configuration."""
        try:
            from ...core.config import (
                load_llm_config,
                load_aider_config,
                get_settings,
                get_data_dir
            )

            data_dir = get_data_dir()

            self._load_llm_config()
            self._load_maven_config()
            self._load_jdk_config()
            self._load_coverage_config()
            self._load_aider_config()

        except Exception as e:
            logger.exception(f"Failed to load config: {e}")

    def _load_llm_config(self):
        """Load LLM configuration."""
        try:
            from ...core.config import load_llm_config

            collection = load_llm_config()

            lines = []
            lines.append(f"Total configurations: {len(collection.configs)}")
            lines.append(f"Default config ID: {collection.default_config_id or 'None'}")
            lines.append("")

            for i, config in enumerate(collection.configs, 1):
                is_default = " (default)" if config.id == collection.default_config_id else ""
                lines.append(f"Configuration {i}{is_default}:")
                lines.append(f"  ID: {config.id}")
                lines.append(f"  Name: {config.name or 'Unnamed'}")
                lines.append(f"  Provider: {config.provider}")
                lines.append(f"  Model: {config.model}")
                lines.append(f"  Endpoint: {config.endpoint}")
                lines.append(f"  API Key: {config.get_masked_api_key()}")
                lines.append(f"  Timeout: {config.timeout}s")
                lines.append(f"  Max Retries: {config.max_retries}")
                lines.append(f"  Temperature: {config.temperature}")
                lines.append(f"  Max Tokens: {config.max_tokens}")
                if config.ca_cert:
                    lines.append(f"  CA Cert: {config.ca_cert}")
                lines.append("")

            self.llm_text.setPlainText("\n".join(lines))

        except Exception as e:
            self.llm_text.setPlainText(f"Error loading LLM config: {e}")

    def _load_maven_config(self):
        """Load Maven configuration."""
        try:
            from ...core.config import get_settings

            settings = get_settings()

            lines = []
            lines.append("Maven Configuration:")
            lines.append("")
            if settings.maven.maven_path:
                lines.append(f"  Maven Path: {settings.maven.maven_path}")
            else:
                lines.append("  Maven Path: Auto-detect")
            lines.append("")

            self.maven_text.setPlainText("\n".join(lines))

        except Exception as e:
            self.maven_text.setPlainText(f"Error loading Maven config: {e}")

    def _load_jdk_config(self):
        """Load JDK configuration."""
        try:
            from ...core.config import get_settings

            settings = get_settings()

            lines = []
            lines.append("JDK Configuration:")
            lines.append("")
            if settings.jdk.java_home:
                lines.append(f"  JAVA_HOME: {settings.jdk.java_home}")
            else:
                lines.append("  JAVA_HOME: Auto-detect")
            lines.append("")

            self.jdk_text.setPlainText("\n".join(lines))

        except Exception as e:
            self.jdk_text.setPlainText(f"Error loading JDK config: {e}")

    def _load_coverage_config(self):
        """Load Coverage configuration."""
        try:
            from ...core.config import get_settings

            settings = get_settings()

            lines = []
            lines.append("Coverage Configuration:")
            lines.append("")
            lines.append(f"  Target Coverage: {settings.coverage.target_coverage:.1%}")
            lines.append(f"  Min Coverage: {settings.coverage.min_coverage:.1%}")
            lines.append(f"  Max Iterations: {settings.coverage.max_iterations}")
            lines.append(f"  Max Step Attempts: {settings.coverage.max_step_attempts}")
            lines.append(f"  Max Compilation Attempts: {settings.coverage.max_compilation_attempts}")
            lines.append(f"  Max Test Attempts: {settings.coverage.max_test_attempts}")
            lines.append("")

            self.coverage_text.setPlainText("\n".join(lines))

        except Exception as e:
            self.coverage_text.setPlainText(f"Error loading Coverage config: {e}")

    def _load_aider_config(self):
        """Load Aider configuration."""
        try:
            from ...core.config import load_aider_config

            config = load_aider_config()

            lines = []
            lines.append("Aider Configuration:")
            lines.append("")
            lines.append("Core Settings:")
            lines.append(f"  Max Attempts: {config.max_attempts}")
            lines.append(f"  Enable Fallback: {config.enable_fallback}")
            lines.append(f"  Enable Circuit Breaker: {config.enable_circuit_breaker}")
            lines.append(f"  Timeout: {config.timeout_seconds}s")
            lines.append("")
            lines.append("Architect/Editor Settings:")
            lines.append(f"  Use Architect/Editor: {config.use_architect_editor}")
            if config.use_architect_editor:
                lines.append(f"    Architect Model ID: {config.architect_model_id or 'Not set'}")
                lines.append(f"    Editor Model ID: {config.editor_model_id or 'Not set'}")
                lines.append(f"    Mode: {config.architect_mode.value}")
            lines.append("")
            lines.append("Multi-file Settings:")
            lines.append(f"  Enable Multi-file: {config.enable_multi_file}")
            if config.enable_multi_file:
                lines.append(f"    Max Files Per Edit: {config.max_files_per_edit}")
            lines.append("")
            lines.append("Edit Format Settings:")
            lines.append(f"  Auto Detect Format: {config.auto_detect_format}")
            lines.append(f"  Preferred Format: {config.preferred_format or 'Auto'}")
            lines.append("")
            lines.append("Cost Tracking:")
            lines.append(f"  Track Costs: {config.track_costs}")
            lines.append("")

            self.aider_text.setPlainText("\n".join(lines))

        except Exception as e:
            self.aider_text.setPlainText(f"Error loading Aider config: {e}")

    def copy_all(self):
        """Copy all configuration to clipboard."""
        try:
            all_config = []
            all_config.append("=" * 60)
            all_config.append("PyUT Agent Configuration Overview")
            all_config.append("=" * 60)
            all_config.append("")

            all_config.append("LLM Configuration:")
            all_config.append(self.llm_text.toPlainText())
            all_config.append("")

            all_config.append("Maven Configuration:")
            all_config.append(self.maven_text.toPlainText())
            all_config.append("")

            all_config.append("JDK Configuration:")
            all_config.append(self.jdk_text.toPlainText())
            all_config.append("")

            all_config.append("Coverage Configuration:")
            all_config.append(self.coverage_text.toPlainText())
            all_config.append("")

            all_config.append("Aider Configuration:")
            all_config.append(self.aider_text.toPlainText())

            from PyQt6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText("\n".join(all_config))

            logger.info("Configuration copied to clipboard")

        except Exception as e:
            logger.exception(f"Failed to copy config: {e}")
