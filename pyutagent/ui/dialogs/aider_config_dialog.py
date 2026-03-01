"""Aider configuration dialog for advanced editing features."""

import logging
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QCheckBox, QPushButton, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QLineEdit, QMessageBox,
    QTabWidget, QWidget, QTextEdit
)
from PyQt6.QtCore import Qt
from typing import Optional

from ...core.config import AiderConfig, LLMConfigCollection
from ...tools.edit_formats import EditFormat
from ...tools.architect_editor import ArchitectMode

logger = logging.getLogger(__name__)


class AiderConfigDialog(QDialog):
    """Dialog for configuring Aider advanced features."""

    def __init__(self, config: Optional[AiderConfig] = None,
                 config_collection: Optional[LLMConfigCollection] = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Aider 高级配置")
        self.setMinimumWidth(550)
        self.setMinimumHeight(600)

        self.config = config or AiderConfig()
        self.config_collection = config_collection
        try:
            self.setup_ui()
            self.load_config()
            logger.info("Aider config dialog initialized")
        except Exception as e:
            logger.exception("Failed to initialize Aider config dialog")

    def setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()

        # General tab
        self.general_tab = self._create_general_tab()
        self.tab_widget.addTab(self.general_tab, "常规")

        # Architect/Editor tab
        self.architect_tab = self._create_architect_tab()
        self.tab_widget.addTab(self.architect_tab, "Architect/Editor")

        # Multi-file tab
        self.multifile_tab = self._create_multifile_tab()
        self.tab_widget.addTab(self.multifile_tab, "多文件编辑")

        # Edit Format tab
        self.format_tab = self._create_format_tab()
        self.tab_widget.addTab(self.format_tab, "编辑格式")

        layout.addWidget(self.tab_widget)

        # Info label
        info_label = QLabel(
            "💡 Aider 高级功能提供更智能的代码编辑能力，"
            "包括 Architect/Editor 双模型模式和多文件批量编辑。"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(info_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_btn = QPushButton("💾 保存")
        self.save_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("❌ 取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def _create_general_tab(self) -> QWidget:
        """Create general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Core settings group
        core_group = QGroupBox("核心设置")
        core_layout = QFormLayout()

        self.max_attempts_spin = QSpinBox()
        self.max_attempts_spin.setRange(1, 10)
        self.max_attempts_spin.setValue(3)
        self.max_attempts_spin.setSuffix(" 次")
        core_layout.addRow("最大尝试次数:", self.max_attempts_spin)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 600)
        self.timeout_spin.setValue(120)
        self.timeout_spin.setSuffix(" 秒")
        core_layout.addRow("超时时间:", self.timeout_spin)

        self.enable_fallback_check = QCheckBox("启用降级策略")
        self.enable_fallback_check.setChecked(True)
        core_layout.addRow("", self.enable_fallback_check)

        self.enable_circuit_breaker_check = QCheckBox("启用熔断保护")
        self.enable_circuit_breaker_check.setChecked(True)
        core_layout.addRow("", self.enable_circuit_breaker_check)

        self.track_costs_check = QCheckBox("跟踪成本")
        self.track_costs_check.setChecked(True)
        core_layout.addRow("", self.track_costs_check)

        core_group.setLayout(core_layout)
        layout.addWidget(core_group)

        # Description
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setMaximumHeight(100)
        desc.setText(
            "核心设置控制 Aider 代码修复的基本行为。\n\n"
            "• 最大尝试次数: 修复失败时的重试次数\n"
            "• 超时时间: 每次 LLM 调用的最大等待时间\n"
            "• 降级策略: 失败时尝试替代修复方法\n"
            "• 熔断保护: 防止连续失败的保护机制"
        )
        layout.addWidget(desc)

        layout.addStretch()
        return tab

    def _create_architect_tab(self) -> QWidget:
        """Create Architect/Editor settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Enable Architect/Editor
        self.enable_architect_check = QCheckBox("启用 Architect/Editor 双模型模式")
        self.enable_architect_check.setChecked(False)
        self.enable_architect_check.stateChanged.connect(self._on_architect_toggled)
        layout.addWidget(self.enable_architect_check)

        # Architect/Editor settings group
        self.architect_group = QGroupBox("Architect/Editor 配置")
        architect_layout = QFormLayout()

        # Warning label if no config collection
        if not self.config_collection or self.config_collection.is_empty():
            warning_label = QLabel(
                "⚠️ 请先配置 LLM 模型，然后在 LLM 配置管理中添加模型配置。"
            )
            warning_label.setStyleSheet("color: orange;")
            warning_label.setWordWrap(True)
            architect_layout.addRow(warning_label)
        
        # Architect model selection
        self.architect_model_combo = QComboBox()
        self.architect_model_combo.setToolTip(
            "选择 Architect 模型（负责分析问题和制定修复计划）"
        )
        self._populate_model_combo(self.architect_model_combo)
        architect_layout.addRow("Architect 模型:", self.architect_model_combo)

        # Editor model selection
        self.editor_model_combo = QComboBox()
        self.editor_model_combo.setToolTip(
            "选择 Editor 模型（负责执行具体的代码修改）"
        )
        self._populate_model_combo(self.editor_model_combo)
        architect_layout.addRow("Editor 模型:", self.editor_model_combo)

        # Mode selection
        self.architect_mode_combo = QComboBox()
        self.architect_mode_combo.addItem("双模型模式", ArchitectMode.DUAL_MODEL)
        self.architect_mode_combo.addItem("单模型模式", ArchitectMode.SINGLE_MODEL)
        architect_layout.addRow("工作模式:", self.architect_mode_combo)

        self.architect_group.setLayout(architect_layout)
        layout.addWidget(self.architect_group)

        # Description
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setMaximumHeight(150)
        desc.setText(
            "Architect/Editor 模式使用两个不同的模型来完成代码修复任务：\n\n"
            "• Architect (强大模型): 分析代码问题并制定修复计划\n"
            "• Editor (快速/便宜模型): 将计划转换为具体的代码修改\n\n"
            "优点：\n"
            "• 更高质量的修复结果\n"
            "• 更低的 API 调用成本\n"
            "• 更好的可解释性\n\n"
            "注意：选择的模型将使用其在 LLM 配置中设置的 CA 证书。"
        )
        layout.addWidget(desc)

        layout.addStretch()
        return tab

    def _populate_model_combo(self, combo: QComboBox):
        """Populate model combo box with available configurations."""
        combo.clear()
        combo.addItem("-- 选择模型 --", None)
        
        if self.config_collection and not self.config_collection.is_empty():
            for config_id, display_name in self.config_collection.get_config_names():
                combo.addItem(display_name, config_id)
        
        # Set minimum width for better display
        combo.setMinimumWidth(250)

    def _create_multifile_tab(self) -> QWidget:
        """Create multi-file editing settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Enable multi-file
        self.enable_multifile_check = QCheckBox("启用多文件批量编辑")
        self.enable_multifile_check.setChecked(False)
        self.enable_multifile_check.stateChanged.connect(self._on_multifile_toggled)
        layout.addWidget(self.enable_multifile_check)

        # Multi-file settings group
        self.multifile_group = QGroupBox("多文件编辑配置")
        multifile_layout = QFormLayout()

        self.max_files_spin = QSpinBox()
        self.max_files_spin.setRange(2, 20)
        self.max_files_spin.setValue(5)
        self.max_files_spin.setSuffix(" 个")
        multifile_layout.addRow("最大文件数:", self.max_files_spin)

        self.multifile_group.setLayout(multifile_layout)
        layout.addWidget(self.multifile_group)

        # Description
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setMaximumHeight(150)
        desc.setText(
            "多文件批量编辑功能可以：\n\n"
            "• 分析文件间的依赖关系\n"
            "• 按照依赖顺序自动排序编辑\n"
            "• 支持批量验证和回滚\n"
            "• 处理跨文件的代码修改\n\n"
            "适用于需要同时修改多个相关文件的场景。"
        )
        layout.addWidget(desc)

        layout.addStretch()
        return tab

    def _create_format_tab(self) -> QWidget:
        """Create edit format settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Edit format group
        format_group = QGroupBox("编辑格式配置")
        format_layout = QFormLayout()

        # Preferred format
        self.preferred_format_combo = QComboBox()
        self.preferred_format_combo.addItem("自动检测", None)
        self.preferred_format_combo.addItem("Diff (SEARCH/REPLACE)", EditFormat.DIFF)
        self.preferred_format_combo.addItem("Unified Diff", EditFormat.UDIFF)
        self.preferred_format_combo.addItem("Whole File", EditFormat.WHOLE)
        self.preferred_format_combo.addItem("Diff Fenced (Gemini)", EditFormat.DIFF_FENCED)
        format_layout.addRow("首选格式:", self.preferred_format_combo)

        self.auto_detect_check = QCheckBox("自动检测模型最优格式")
        self.auto_detect_check.setChecked(True)
        format_layout.addRow("", self.auto_detect_check)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Format descriptions
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setMaximumHeight(200)
        desc.setText(
            "支持的编辑格式：\n\n"
            "1. Diff (SEARCH/REPLACE)\n"
            "   Aider 的标准格式，精确匹配和替换代码块\n\n"
            "2. Unified Diff\n"
            "   标准的统一差异格式，适用于大多数模型\n\n"
            "3. Whole File\n"
            "   返回完整文件内容，适用于本地模型\n\n"
            "4. Diff Fenced (Gemini)\n"
            "   Google Gemini 模型兼容的格式\n\n"
            "建议：使用自动检测让系统根据模型选择最优格式。"
        )
        layout.addWidget(desc)

        layout.addStretch()
        return tab

    def _on_architect_toggled(self, state: int):
        """Handle Architect/Editor toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.architect_group.setEnabled(enabled)

    def _on_multifile_toggled(self, state: int):
        """Handle multi-file toggle."""
        enabled = state == Qt.CheckState.Checked.value
        self.multifile_group.setEnabled(enabled)

    def load_config(self):
        """Load configuration into UI."""
        try:
            # General tab
            self.max_attempts_spin.setValue(self.config.max_attempts)
            self.timeout_spin.setValue(int(self.config.timeout_seconds))
            self.enable_fallback_check.setChecked(self.config.enable_fallback)
            self.enable_circuit_breaker_check.setChecked(self.config.enable_circuit_breaker)
            self.track_costs_check.setChecked(self.config.track_costs)

            # Architect tab
            self.enable_architect_check.setChecked(self.config.use_architect_editor)
            self.architect_group.setEnabled(self.config.use_architect_editor)

            # Set architect model
            if self.config.architect_model_id:
                index = self.architect_model_combo.findData(self.config.architect_model_id)
                if index >= 0:
                    self.architect_model_combo.setCurrentIndex(index)

            # Set editor model
            if self.config.editor_model_id:
                index = self.editor_model_combo.findData(self.config.editor_model_id)
                if index >= 0:
                    self.editor_model_combo.setCurrentIndex(index)

            index = self.architect_mode_combo.findData(self.config.architect_mode)
            if index >= 0:
                self.architect_mode_combo.setCurrentIndex(index)

            # Multi-file tab
            self.enable_multifile_check.setChecked(self.config.enable_multi_file)
            self.multifile_group.setEnabled(self.config.enable_multi_file)
            self.max_files_spin.setValue(self.config.max_files_per_edit)

            # Format tab
            if self.config.preferred_format:
                index = self.preferred_format_combo.findData(self.config.preferred_format)
                if index >= 0:
                    self.preferred_format_combo.setCurrentIndex(index)
            self.auto_detect_check.setChecked(self.config.auto_detect_format)
            logger.info("Aider config loaded into UI")
        except Exception as e:
            logger.exception("Failed to load Aider config")

    def get_config(self) -> AiderConfig:
        """Get configuration from UI."""
        try:
            # Get preferred format
            preferred_format = self.preferred_format_combo.currentData()

            # Get selected model IDs
            architect_model_id = self.architect_model_combo.currentData()
            editor_model_id = self.editor_model_combo.currentData()

            config = AiderConfig(
                # General
                max_attempts=self.max_attempts_spin.value(),
                timeout_seconds=float(self.timeout_spin.value()),
                enable_fallback=self.enable_fallback_check.isChecked(),
                enable_circuit_breaker=self.enable_circuit_breaker_check.isChecked(),
                track_costs=self.track_costs_check.isChecked(),

                # Architect/Editor
                use_architect_editor=self.enable_architect_check.isChecked(),
                architect_model_id=architect_model_id,
                editor_model_id=editor_model_id,
                architect_mode=self.architect_mode_combo.currentData(),

                # Multi-file
                enable_multi_file=self.enable_multifile_check.isChecked(),
                max_files_per_edit=self.max_files_spin.value(),

                # Format
                preferred_format=preferred_format,
                auto_detect_format=self.auto_detect_check.isChecked(),
            )
            logger.info("Aider config retrieved from UI")
            return config
        except Exception as e:
            logger.exception("Failed to get Aider config from UI")
            raise

    def accept(self):
        """Accept dialog."""
        try:
            # Validate Architect/Editor config if enabled
            if self.enable_architect_check.isChecked():
                architect_id = self.architect_model_combo.currentData()
                editor_id = self.editor_model_combo.currentData()

                if not architect_id:
                    QMessageBox.warning(
                        self,
                        "配置不完整",
                        "请为 Architect 选择一个模型"
                    )
                    return

                if not editor_id:
                    QMessageBox.warning(
                        self,
                        "配置不完整",
                        "请为 Editor 选择一个模型"
                    )
                    return

            self.config = self.get_config()
            logger.info("Aider config dialog accepted")
            super().accept()
        except Exception as e:
            logger.exception("Failed to accept Aider config dialog")
