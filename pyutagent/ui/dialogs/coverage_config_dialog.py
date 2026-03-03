"""Coverage settings dialog for test generation configuration."""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QMessageBox, QWidget, QTextEdit
)
from PyQt6.QtCore import Qt

from ...core.config import CoverageSettings, get_settings, save_app_config

logger = logging.getLogger(__name__)


class CoverageConfigDialog(QDialog):
    """Dialog for configuring test generation coverage settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("测试生成配置")
        self.setMinimumWidth(500)
        self.setMinimumHeight(450)

        self.settings = get_settings()
        try:
            self.setup_ui()
            self.load_settings()
            logger.info("Coverage config dialog initialized")
        except Exception as e:
            logger.exception("Failed to initialize coverage config dialog")

    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # Coverage settings group
        coverage_group = QGroupBox("覆盖率设置")
        coverage_layout = QFormLayout()

        self.target_coverage_spin = QDoubleSpinBox()
        self.target_coverage_spin.setRange(0.1, 1.0)
        self.target_coverage_spin.setSingleStep(0.05)
        self.target_coverage_spin.setSuffix(" (80%)")
        self.target_coverage_spin.setDecimals(2)
        coverage_layout.addRow("目标覆盖率:", self.target_coverage_spin)

        self.min_coverage_spin = QDoubleSpinBox()
        self.min_coverage_spin.setRange(0.0, 0.9)
        self.min_coverage_spin.setSingleStep(0.05)
        self.min_coverage_spin.setSuffix(" (50%)")
        self.min_coverage_spin.setDecimals(2)
        coverage_layout.addRow("最低覆盖率:", self.min_coverage_spin)

        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(1, 20)
        self.max_iterations_spin.setValue(10)
        self.max_iterations_spin.setSuffix(" 次")
        coverage_layout.addRow("最大迭代次数:", self.max_iterations_spin)

        coverage_group.setLayout(coverage_layout)
        layout.addWidget(coverage_group)

        # Attempt limits group
        attempts_group = QGroupBox("尝试次数限制")
        attempts_layout = QFormLayout()

        self.max_compilation_attempts_spin = QSpinBox()
        self.max_compilation_attempts_spin.setRange(1, 10)
        self.max_compilation_attempts_spin.setValue(3)
        self.max_compilation_attempts_spin.setSuffix(" 次")
        attempts_layout.addRow("最大编译尝试次数:", self.max_compilation_attempts_spin)

        self.max_test_attempts_spin = QSpinBox()
        self.max_test_attempts_spin.setRange(1, 10)
        self.max_test_attempts_spin.setValue(3)
        self.max_test_attempts_spin.setSuffix(" 次")
        attempts_layout.addRow("最大测试尝试次数:", self.max_test_attempts_spin)

        attempts_group.setLayout(attempts_layout)
        layout.addWidget(attempts_group)

        # Description
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setMaximumHeight(150)
        desc.setText(
            "📊 覆盖率设置说明:\n\n"
            "• 目标覆盖率：测试生成的目标代码覆盖率，达到此值后停止优化\n"
            "• 最低覆盖率：可接受的最低覆盖率阈值\n"
            "• 最大迭代次数：覆盖率优化的最大迭代轮数\n\n"
            "🔄 尝试次数限制说明:\n\n"
            "• 最大编译尝试次数：编译失败时的最大修复尝试次数，超过后需要人工干预\n"
            "• 最大测试尝试次数：测试失败时的最大修复尝试次数，超过后需要人工干预\n\n"
            "💡 建议：较小的尝试次数可以更快失败，避免长时间等待；较大的尝试次数可以提高成功率。"
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

        self.reset_btn = QPushButton("🔄 重置为默认值")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_btn)

        self.cancel_btn = QPushButton("❌ 取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def load_settings(self):
        """Load current settings into UI controls."""
        coverage = self.settings.coverage
        
        self.target_coverage_spin.setValue(coverage.target_coverage)
        self.min_coverage_spin.setValue(coverage.min_coverage)
        self.max_iterations_spin.setValue(coverage.max_iterations)
        self.max_compilation_attempts_spin.setValue(coverage.max_compilation_attempts)
        self.max_test_attempts_spin.setValue(coverage.max_test_attempts)

        logger.debug(f"Loaded coverage settings: target={coverage.target_coverage}, "
                    f"min={coverage.min_coverage}, max_iter={coverage.max_iterations}, "
                    f"max_comp={coverage.max_compilation_attempts}, "
                    f"max_test={coverage.max_test_attempts}")

    def save_settings(self) -> bool:
        """Save settings from UI controls."""
        try:
            coverage = self.settings.coverage
            coverage.target_coverage = self.target_coverage_spin.value()
            coverage.min_coverage = self.min_coverage_spin.value()
            coverage.max_iterations = self.max_iterations_spin.value()
            coverage.max_compilation_attempts = self.max_compilation_attempts_spin.value()
            coverage.max_test_attempts = self.max_test_attempts_spin.value()

            save_app_config(self.settings)
            
            logger.info(f"Saved coverage settings: target={coverage.target_coverage}, "
                       f"min={coverage.min_coverage}, max_iter={coverage.max_iterations}, "
                       f"max_comp={coverage.max_compilation_attempts}, "
                       f"max_test={coverage.max_test_attempts}")
            return True
        except Exception as e:
            logger.exception("Failed to save coverage settings")
            QMessageBox.critical(
                self,
                "保存失败",
                f"无法保存配置：{str(e)}"
            )
            return False

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        reply = QMessageBox.question(
            self,
            "确认重置",
            "确定要重置为默认值吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.target_coverage_spin.setValue(0.8)
            self.min_coverage_spin.setValue(0.5)
            self.max_iterations_spin.setValue(10)
            self.max_compilation_attempts_spin.setValue(3)
            self.max_test_attempts_spin.setValue(3)
            logger.info("Coverage settings reset to defaults")

    def accept(self):
        """Handle save button click."""
        if self.save_settings():
            QMessageBox.information(
                self,
                "保存成功",
                "配置已保存，将在下次生成测试时生效。"
            )
            super().accept()
