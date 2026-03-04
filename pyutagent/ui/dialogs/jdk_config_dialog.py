"""JDK configuration dialog for setting Java home path."""

import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QLineEdit, QFileDialog,
    QMessageBox, QTextEdit, QCheckBox
)
from PyQt6.QtCore import Qt

from ...core.config import JDKSettings, get_settings, save_app_config
from ...tools.java_tools import find_java_home, find_java_executable, find_javac_executable, get_java_info

logger = logging.getLogger(__name__)


class JDKConfigDialog(QDialog):
    """Dialog for configuring JDK home path."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JDK 配置")
        self.setMinimumWidth(600)
        self.setMinimumHeight(550)

        self.settings = get_settings()
        self.detected_java_home: str = ""
        self.detected_java_info = None
        try:
            self.setup_ui()
            self.load_settings()
            self.detect_java()
            logger.info("JDK config dialog initialized")
        except Exception as e:
            logger.exception("Failed to initialize JDK config dialog")

    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        java_home_group = QGroupBox("JDK 主目录配置")
        java_home_layout = QFormLayout()

        self.java_home_edit = QLineEdit()
        self.java_home_edit.setPlaceholderText("留空则自动检测 JDK 路径")
        java_home_layout.addRow("JAVA_HOME:", self.java_home_edit)

        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_java_home)
        java_home_layout.addRow(self.browse_btn)

        java_home_group.setLayout(java_home_layout)
        layout.addWidget(java_home_group)

        detection_group = QGroupBox("自动检测结果")
        detection_layout = QVBoxLayout()

        self.detection_label = QLabel("正在检测 Java...")
        self.detection_label.setStyleSheet("color: #666; font-weight: bold;")
        detection_layout.addWidget(self.detection_label)

        self.detection_path_label = QLabel("")
        self.detection_path_label.setStyleSheet("color: #0066cc; font-family: Consolas, monospace;")
        self.detection_path_label.setWordWrap(True)
        detection_layout.addWidget(self.detection_path_label)

        self.version_label = QLabel("")
        self.version_label.setStyleSheet("color: #666;")
        detection_layout.addWidget(self.version_label)

        self.javac_label = QLabel("")
        self.javac_label.setStyleSheet("color: #666;")
        detection_layout.addWidget(self.javac_label)

        self.use_detected_checkbox = QCheckBox("使用检测到的 JDK 路径")
        self.use_detected_checkbox.setChecked(True)
        self.use_detected_checkbox.stateChanged.connect(self.on_use_detected_changed)
        detection_layout.addWidget(self.use_detected_checkbox)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setMaximumHeight(220)
        desc.setText(
            "☕ JDK 配置说明:\n\n"
            "• JAVA_HOME：指定 JDK 安装目录（包含 bin 目录的父目录）\n"
            "• 自动检测：系统会自动在以下位置查找 JDK:\n"
            "  - JAVA_HOME, JDK_HOME, JRE_HOME 环境变量\n"
            "  - PATH 环境变量\n"
            "  - Windows: Program Files, Chocolatey, Scoop 等\n"
            "  - macOS: /Library/Java/JavaVirtualMachines, Homebrew\n"
            "  - Linux: /usr/lib/jvm, /opt/java 等\n\n"
            "💡 使用建议:\n\n"
            "• 如果已安装 JDK 并配置好环境变量，建议留空使用自动检测\n"
            "• 如果使用多个 JDK 版本，可以指定具体版本的路径\n"
            "• 需要完整的 JDK（包含 javac），而不是 JRE\n"
            "• 配置后将优先使用指定路径，找不到时才会回退到自动检测"
        )
        desc.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(desc)

        layout.addStretch()

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

    def detect_java(self):
        """Detect Java installation."""
        try:
            self.detected_java_info = get_java_info()
            
            if self.detected_java_info:
                self.detected_java_home = self.detected_java_info.java_home
                
                if self.detected_java_info.is_jdk:
                    self.detection_label.setText("✅ 检测到 JDK:")
                    self.detection_label.setStyleSheet("color: #00aa00; font-weight: bold;")
                else:
                    self.detection_label.setText("⚠️ 检测到 JRE (无编译器):")
                    self.detection_label.setStyleSheet("color: #ff6600; font-weight: bold;")
                
                self.detection_path_label.setText(self.detected_java_home or self.detected_java_info.java_path)
                
                version_text = f"版本: {self.detected_java_info.java_version} | 供应商: {self.detected_java_info.vendor}"
                self.version_label.setText(version_text)
                
                if self.detected_java_info.javac_path:
                    self.javac_label.setText(f"✅ javac: {self.detected_java_info.javac_path}")
                    self.javac_label.setStyleSheet("color: #00aa00;")
                else:
                    self.javac_label.setText("⚠️ 未找到 javac 编译器，将无法编译 Java 代码")
                    self.javac_label.setStyleSheet("color: #ff6600;")
                
                self.use_detected_checkbox.setEnabled(True)
                logger.info(f"Java detected: {self.detected_java_home}, version: {self.detected_java_info.java_version}")
            else:
                self.detection_label.setText("❌ 未检测到 Java")
                self.detection_label.setStyleSheet("color: #ff0000; font-weight: bold;")
                self.detection_path_label.setText("请安装 JDK 或手动指定 JAVA_HOME 路径")
                self.version_label.setText("")
                self.javac_label.setText("")
                self.use_detected_checkbox.setEnabled(False)
                self.use_detected_checkbox.setChecked(False)
                logger.warning("Java not detected")
        except Exception as e:
            logger.exception("Failed to detect Java")
            self.detection_label.setText("❌ 检测失败")
            self.detection_path_label.setText(f"错误：{str(e)}")
            self.use_detected_checkbox.setEnabled(False)

    def browse_java_home(self):
        """Open directory browser to select JAVA_HOME."""
        try:
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "选择 JDK 主目录 (JAVA_HOME)",
                "",
                QFileDialog.Option.ShowDirsOnly
            )

            if dir_path:
                java_home = Path(dir_path)
                bin_dir = java_home / "bin"
                
                if not bin_dir.exists():
                    reply = QMessageBox.question(
                        self,
                        "确认",
                        f"所选目录不包含 bin 子目录:\n{dir_path}\n\n"
                        "确定要使用此目录吗？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                
                if platform := __import__('platform').system() == "Windows":
                    java_exe = bin_dir / "java.exe"
                    javac_exe = bin_dir / "javac.exe"
                else:
                    java_exe = bin_dir / "java"
                    javac_exe = bin_dir / "javac"
                
                if not java_exe.exists():
                    QMessageBox.warning(
                        self,
                        "警告",
                        f"所选目录不包含 java 可执行文件:\n{java_exe}"
                    )
                    return
                
                self.java_home_edit.setText(dir_path)
                self.use_detected_checkbox.setChecked(False)
                
                if not javac_exe.exists():
                    QMessageBox.warning(
                        self,
                        "警告",
                        "所选目录不包含 javac 编译器。\n"
                        "这可能是一个 JRE 而非 JDK，将无法编译 Java 代码。"
                    )
                
                logger.info(f"JAVA_HOME selected via browser: {dir_path}")
        except Exception as e:
            logger.exception("Failed to browse for JAVA_HOME")
            QMessageBox.critical(
                self,
                "错误",
                f"无法浏览目录：{str(e)}"
            )

    def on_use_detected_changed(self, state):
        """Handle use detected checkbox state change."""
        if state == Qt.CheckState.Checked:
            if self.detected_java_home:
                self.java_home_edit.setText(self.detected_java_home)
                self.java_home_edit.setEnabled(False)
                self.browse_btn.setEnabled(False)
        else:
            self.java_home_edit.setEnabled(True)
            self.browse_btn.setEnabled(True)
            if self.java_home_edit.text() == self.detected_java_home:
                self.java_home_edit.clear()

    def load_settings(self):
        """Load current settings into UI controls."""
        try:
            java_home = self.settings.jdk.java_home
            if java_home:
                self.java_home_edit.setText(java_home)
                self.use_detected_checkbox.setChecked(False)
                self.java_home_edit.setEnabled(True)
                self.browse_btn.setEnabled(True)
                logger.debug(f"Loaded java_home from settings: {java_home}")
        except Exception as e:
            logger.exception("Failed to load JDK settings")

    def save_settings(self) -> bool:
        """Save settings from UI controls."""
        try:
            if self.use_detected_checkbox.isChecked():
                self.settings.jdk.java_home = ""
                logger.info("JAVA_HOME set to auto-detect (empty)")
            else:
                java_home = self.java_home_edit.text().strip()
                if java_home and java_home != self.detected_java_home:
                    self.settings.jdk.java_home = java_home
                    logger.info(f"JAVA_HOME set to: {java_home}")
                else:
                    self.settings.jdk.java_home = ""
                    logger.info("JAVA_HOME set to auto-detect (empty or same as detected)")

            save_app_config(self.settings)
            logger.info("Saved JDK settings")
            return True
        except Exception as e:
            logger.exception("Failed to save JDK settings")
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
            self.java_home_edit.clear()
            if self.detected_java_home:
                self.java_home_edit.setText(self.detected_java_home)
                self.use_detected_checkbox.setChecked(True)
                self.java_home_edit.setEnabled(False)
                self.browse_btn.setEnabled(False)
            else:
                self.use_detected_checkbox.setChecked(False)
                self.java_home_edit.setEnabled(True)
                self.browse_btn.setEnabled(True)
            logger.info("JDK config reset to auto-detect")

    def validate_java_home(self) -> bool:
        """Validate the configured JAVA_HOME."""
        if self.use_detected_checkbox.isChecked():
            if not self.detected_java_home:
                QMessageBox.warning(
                    self,
                    "警告",
                    "未检测到 Java，请先安装 JDK 或手动指定路径"
                )
                return False
            if self.detected_java_info and not self.detected_java_info.is_jdk:
                reply = QMessageBox.question(
                    self,
                    "确认",
                    "检测到的是 JRE 而非 JDK，将无法编译 Java 代码。\n\n"
                    "确定要继续吗？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return False
            return True

        java_home = self.java_home_edit.text().strip()
        if not java_home:
            return True

        path = Path(java_home)
        if not path.exists():
            QMessageBox.warning(
                self,
                "警告",
                f"JAVA_HOME 路径不存在:\n{java_home}"
            )
            return False

        if not path.is_dir():
            QMessageBox.warning(
                self,
                "警告",
                f"指定的路径不是目录:\n{java_home}"
            )
            return False

        bin_dir = path / "bin"
        if not bin_dir.exists():
            reply = QMessageBox.question(
                self,
                "确认",
                f"所选目录不包含 bin 子目录:\n{java_home}\n\n"
                "确定要继续吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        return True

    def accept(self):
        """Handle save button click."""
        if not self.validate_java_home():
            return

        if self.save_settings():
            QMessageBox.information(
                self,
                "保存成功",
                "JDK 配置已保存，将在下次运行时生效。"
            )
            super().accept()
