"""Maven configuration dialog for setting Maven executable path."""

import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QFileDialog,
    QMessageBox, QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt

from ...core.config import MavenSettings, get_settings, save_app_config
from ...tools.maven_tools import find_maven_executable

logger = logging.getLogger(__name__)


class MavenConfigDialog(QDialog):
    """Dialog for configuring Maven executable path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Maven 配置")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self.settings = get_settings()
        self.detected_maven: str = ""
        try:
            self.setup_ui()
            self.load_settings()
            self.detect_maven()
            logger.info("Maven config dialog initialized")
        except Exception as e:
            logger.exception("Failed to initialize maven config dialog")

    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # Maven path group
        maven_group = QGroupBox("Maven 可执行文件配置")
        maven_layout = QFormLayout()

        self.maven_path_edit = QLineEdit()
        self.maven_path_edit.setPlaceholderText("留空则自动检测 Maven 路径")
        maven_layout.addRow("Maven 路径:", self.maven_path_edit)

        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_maven)
        maven_layout.addRow(self.browse_btn)

        maven_group.setLayout(maven_layout)
        layout.addWidget(maven_group)

        # Auto-detection info group
        detection_group = QGroupBox("自动检测结果")
        detection_layout = QVBoxLayout()

        self.detection_label = QLabel("正在检测 Maven...")
        self.detection_label.setStyleSheet("color: #666; font-weight: bold;")
        detection_layout.addWidget(self.detection_label)

        self.detection_path_label = QLabel("")
        self.detection_path_label.setStyleSheet("color: #0066cc; font-family: Consolas, monospace;")
        self.detection_path_label.setWordWrap(True)
        detection_layout.addWidget(self.detection_path_label)

        self.use_detected_checkbox = QCheckBox("使用检测到的 Maven 路径")
        self.use_detected_checkbox.setChecked(True)
        self.use_detected_checkbox.stateChanged.connect(self.on_use_detected_changed)
        detection_layout.addWidget(self.use_detected_checkbox)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # Description
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setMaximumHeight(200)
        desc.setText(
            "📦 Maven 配置说明:\n\n"
            "• Maven 路径：指定 Maven 可执行文件 (mvn 或 mvn.cmd) 的完整路径\n"
            "• 自动检测：系统会自动在以下位置查找 Maven:\n"
            "  - PATH 环境变量\n"
            "  - M2_HOME, M3_HOME, MAVEN_HOME 环境变量\n"
            "  - Windows: Program Files, Chocolatey, Scoop 等常见安装位置\n"
            "  - Unix/Linux: /usr/share/maven, /opt/maven 等\n\n"
            "💡 使用建议:\n\n"
            "• 如果已安装 Maven 并配置好环境变量，建议留空使用自动检测\n"
            "• 如果使用多个 Maven 版本，可以指定具体版本的路径\n"
            "• 如果 Maven 不在标准位置，需要手动指定路径\n"
            "• 配置后将优先使用指定路径，找不到时才会回退到自动检测"
        )
        desc.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(desc)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_btn = QPushButton("💾 保存")
        self.save_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("🔄 重置为自动检测")
        self.reset_btn.clicked.connect(self.reset_to_auto_detect)
        button_layout.addWidget(self.reset_btn)

        self.cancel_btn = QPushButton("❌ 取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def detect_maven(self):
        """Detect Maven executable."""
        try:
            self.detected_maven = find_maven_executable()
            if self.detected_maven:
                self.detection_label.setText("✅ 检测到 Maven:")
                self.detection_label.setStyleSheet("color: #00aa00; font-weight: bold;")
                self.detection_path_label.setText(self.detected_maven)
                self.use_detected_checkbox.setEnabled(True)
                logger.info(f"Maven detected: {self.detected_maven}")
            else:
                self.detection_label.setText("⚠️ 未检测到 Maven")
                self.detection_label.setStyleSheet("color: #ff6600; font-weight: bold;")
                self.detection_path_label.setText("请在下方手动指定 Maven 路径，或安装 Maven 并配置环境变量")
                self.use_detected_checkbox.setEnabled(False)
                self.use_detected_checkbox.setChecked(False)
                logger.warning("Maven not detected")
        except Exception as e:
            logger.exception("Failed to detect Maven")
            self.detection_label.setText("❌ 检测失败")
            self.detection_path_label.setText(f"错误：{str(e)}")
            self.use_detected_checkbox.setEnabled(False)

    def browse_maven(self):
        """Open file browser to select Maven executable."""
        try:
            file_filter = "Maven 可执行文件 (mvn mvn.cmd mvn.bat);;所有文件 (*.*)"
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择 Maven 可执行文件",
                "",
                file_filter
            )

            if file_path:
                self.maven_path_edit.setText(file_path)
                self.use_detected_checkbox.setChecked(False)
                logger.info(f"Maven path selected via browser: {file_path}")
        except Exception as e:
            logger.exception("Failed to browse for Maven")
            QMessageBox.critical(
                self,
                "错误",
                f"无法浏览文件：{str(e)}"
            )

    def on_use_detected_changed(self, state):
        """Handle use detected checkbox state change."""
        if state == Qt.CheckState.Checked:
            if self.detected_maven:
                self.maven_path_edit.setText(self.detected_maven)
                self.maven_path_edit.setEnabled(False)
                self.browse_btn.setEnabled(False)
        else:
            self.maven_path_edit.setEnabled(True)
            self.browse_btn.setEnabled(True)
            if self.maven_path_edit.text() == self.detected_maven:
                self.maven_path_edit.clear()

    def load_settings(self):
        """Load current settings into UI controls."""
        try:
            maven_path = self.settings.maven.maven_path
            if maven_path:
                self.maven_path_edit.setText(maven_path)
                self.use_detected_checkbox.setChecked(False)
                self.maven_path_edit.setEnabled(True)
                self.browse_btn.setEnabled(True)
                logger.debug(f"Loaded maven path from settings: {maven_path}")
            else:
                if self.detected_maven:
                    self.maven_path_edit.setText(self.detected_maven)
                    self.use_detected_checkbox.setChecked(True)
                    self.maven_path_edit.setEnabled(False)
                    self.browse_btn.setEnabled(False)
        except Exception as e:
            logger.exception("Failed to load maven settings")

    def save_settings(self) -> bool:
        """Save settings from UI controls."""
        try:
            if self.use_detected_checkbox.isChecked():
                self.settings.maven.maven_path = ""
                logger.info("Maven path set to auto-detect (empty)")
            else:
                maven_path = self.maven_path_edit.text().strip()
                if maven_path and maven_path != self.detected_maven:
                    self.settings.maven.maven_path = maven_path
                    logger.info(f"Maven path set to: {maven_path}")
                else:
                    self.settings.maven.maven_path = ""
                    logger.info("Maven path set to auto-detect (empty or same as detected)")

            save_app_config(self.settings)
            logger.info("Saved maven settings")
            return True
        except Exception as e:
            logger.exception("Failed to save maven settings")
            QMessageBox.critical(
                self,
                "保存失败",
                f"无法保存配置：{str(e)}"
            )
            return False

    def reset_to_auto_detect(self):
        """Reset to auto-detection mode."""
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置为自动检测模式吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.maven_path_edit.clear()
            if self.detected_maven:
                self.maven_path_edit.setText(self.detected_maven)
                self.use_detected_checkbox.setChecked(True)
                self.maven_path_edit.setEnabled(False)
                self.browse_btn.setEnabled(False)
            else:
                self.use_detected_checkbox.setChecked(False)
                self.maven_path_edit.setEnabled(True)
                self.browse_btn.setEnabled(True)
            logger.info("Maven config reset to auto-detect")

    def validate_maven_path(self) -> bool:
        """Validate the configured Maven path."""
        if self.use_detected_checkbox.isChecked():
            if not self.detected_maven:
                QMessageBox.warning(
                    self,
                    "警告",
                    "未检测到 Maven，请先安装 Maven 或手动指定路径"
                )
                return False
            return True

        maven_path = self.maven_path_edit.text().strip()
        if not maven_path:
            return True

        path = Path(maven_path)
        if not path.exists():
            QMessageBox.warning(
                self,
                "警告",
                f"Maven 路径不存在:\n{maven_path}"
            )
            return False

        if not path.is_file():
            QMessageBox.warning(
                self,
                "警告",
                f"指定的路径不是文件:\n{maven_path}"
            )
            return False

        if not path.name.lower() in ["mvn", "mvn.cmd", "mvn.bat"]:
            reply = QMessageBox.question(
                self,
                "确认",
                f"文件名看起来不像 Maven 可执行文件:\n{path.name}\n\n"
                f"确定要继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        return True

    def accept(self):
        """Handle save button click."""
        if not self.validate_maven_path():
            return

        if self.save_settings():
            QMessageBox.information(
                self,
                "保存成功",
                "Maven 配置已保存，将在下次运行时生效。"
            )
            super().accept()
