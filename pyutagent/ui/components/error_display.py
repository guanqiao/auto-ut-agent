"""Error Display - 错误处理可视化组件.

提供红色高亮错误、重试按钮和跳过按钮.
"""

import logging
from typing import Optional, Callable, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTextEdit, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)


class ErrorDisplayWidget(QFrame):
    """错误显示组件.

    显示错误信息，提供重试和跳过功能.
    """

    retry_requested = pyqtSignal(str)
    skip_requested = pyqtSignal(str)
    details_expanded = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._error_id: Optional[str] = None
        self._expanded = False
        self._on_retry: Optional[Callable[[str], None]] = None
        self._on_skip: Optional[Callable[[str], None]] = None

        self.setup_ui()

    def setup_ui(self):
        """设置 UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            ErrorDisplayWidget {
                background-color: #FFEBEE;
                border: 2px solid #F44336;
                border-radius: 6px;
                margin: 4px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # 错误头部
        header = QHBoxLayout()

        # 错误图标和标题
        self._icon_label = QLabel("⚠️")
        self._icon_label.setStyleSheet("font-size: 18px;")
        header.addWidget(self._icon_label)

        self._title_label = QLabel("Error Occurred")
        self._title_label.setStyleSheet("""
            font-weight: bold;
            color: #C62828;
            font-size: 14px;
        """)
        header.addWidget(self._title_label)

        header.addStretch()

        # 展开/折叠按钮
        self._expand_btn = QPushButton("▼")
        self._expand_btn.setFixedSize(24, 24)
        self._expand_btn.setFlat(True)
        self._expand_btn.setStyleSheet("""
            QPushButton {
                color: #C62828;
                font-weight: bold;
            }
        """)
        self._expand_btn.clicked.connect(self._toggle_details)
        header.addWidget(self._expand_btn)

        layout.addLayout(header)

        # 错误摘要
        self._summary_label = QLabel("An error occurred during execution")
        self._summary_label.setStyleSheet("color: #D32F2F;")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        # 详情区域
        self._details_container = QWidget()
        details_layout = QVBoxLayout(self._details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)

        # 错误详情文本
        self._details_text = QTextEdit()
        self._details_text.setReadOnly(True)
        self._details_text.setMaximumHeight(120)
        self._details_text.setStyleSheet("""
            QTextEdit {
                background-color: #FFCDD2;
                border: 1px solid #EF9A9A;
                border-radius: 4px;
                color: #B71C1C;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        details_layout.addWidget(self._details_text)

        # 上下文信息
        self._context_label = QLabel()
        self._context_label.setStyleSheet("color: #666; font-size: 11px;")
        self._context_label.setWordWrap(True)
        details_layout.addWidget(self._context_label)

        self._details_container.hide()
        layout.addWidget(self._details_container)

        # 操作按钮
        buttons_layout = QHBoxLayout()

        self._retry_btn = QPushButton("🔄 Retry")
        self._retry_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BBDEFB;
            }
        """)
        self._retry_btn.clicked.connect(self._on_retry_clicked)
        buttons_layout.addWidget(self._retry_btn)

        self._skip_btn = QPushButton("⏭️ Skip")
        self._skip_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
        """)
        self._skip_btn.clicked.connect(self._on_skip_clicked)
        buttons_layout.addWidget(self._skip_btn)

        buttons_layout.addStretch()

        layout.addLayout(buttons_layout)

    def set_error(self, error_id: str, error_message: str,
                  error_type: str = "Execution Error",
                  context: Optional[Dict[str, Any]] = None,
                  retryable: bool = True):
        """设置错误信息.

        Args:
            error_id: 错误唯一标识
            error_message: 错误消息
            error_type: 错误类型
            context: 错误上下文
            retryable: 是否可重试
        """
        self._error_id = error_id

        # 更新标题
        self._title_label.setText(f"⚠️ {error_type}")

        # 更新摘要（显示第一行）
        summary = error_message.split('\n')[0][:100]
        if len(error_message) > 100:
            summary += "..."
        self._summary_label.setText(summary)

        # 更新详情
        self._details_text.setText(error_message)

        # 更新上下文
        if context:
            context_text = "\n".join(f"{k}: {v}" for k, v in context.items())
            self._context_label.setText(f"Context:\n{context_text}")
            self._context_label.show()
        else:
            self._context_label.hide()

        # 更新重试按钮状态
        self._retry_btn.setEnabled(retryable)
        if not retryable:
            self._retry_btn.setToolTip("This error cannot be retried")

        logger.debug(f"Error displayed: {error_id} - {error_type}")

    def set_callbacks(self, on_retry: Optional[Callable[[str], None]] = None,
                     on_skip: Optional[Callable[[str], None]] = None):
        """设置回调函数.

        Args:
            on_retry: 重试回调
            on_skip: 跳过回调
        """
        self._on_retry = on_retry
        self._on_skip = on_skip

    def clear(self):
        """清除错误显示."""
        self._error_id = None
        self._title_label.setText("⚠️ Error Occurred")
        self._summary_label.setText("An error occurred during execution")
        self._details_text.clear()
        self._context_label.clear()
        self._retry_btn.setEnabled(True)

        if self._expanded:
            self._toggle_details()

    def get_error_id(self) -> Optional[str]:
        """获取错误 ID."""
        return self._error_id

    def is_retryable(self) -> bool:
        """检查是否可重试."""
        return self._retry_btn.isEnabled()

    def _toggle_details(self):
        """切换详情展开/折叠."""
        self._expanded = not self._expanded
        self._details_container.setVisible(self._expanded)
        self._expand_btn.setText("▼" if self._expanded else "▶")
        self.details_expanded.emit(self._expanded)

    def _on_retry_clicked(self):
        """处理重试按钮点击."""
        if self._error_id:
            logger.info(f"Retry requested for error: {self._error_id}")
            self.retry_requested.emit(self._error_id)
            if self._on_retry:
                self._on_retry(self._error_id)

    def _on_skip_clicked(self):
        """处理跳过按钮点击."""
        if self._error_id:
            logger.info(f"Skip requested for error: {self._error_id}")
            self.skip_requested.emit(self._error_id)
            if self._on_skip:
                self._on_skip(self._error_id)


class ErrorListWidget(QScrollArea):
    """错误列表组件.

    显示多个错误，支持滚动.
    """

    retry_requested = pyqtSignal(str)
    skip_requested = pyqtSignal(str)
    error_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._errors: Dict[str, ErrorDisplayWidget] = {}

        self.setup_ui()

    def setup_ui(self):
        """设置 UI."""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMaximumHeight(300)
        self.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        # 容器
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self._layout.addStretch()

        self.setWidget(self._container)

    def add_error(self, error_id: str, error_message: str,
                  error_type: str = "Execution Error",
                  context: Optional[Dict[str, Any]] = None,
                  retryable: bool = True) -> ErrorDisplayWidget:
        """添加错误.

        Args:
            error_id: 错误唯一标识
            error_message: 错误消息
            error_type: 错误类型
            context: 错误上下文
            retryable: 是否可重试

        Returns:
            错误显示组件
        """
        if error_id in self._errors:
            # 更新现有错误
            widget = self._errors[error_id]
            widget.set_error(error_id, error_message, error_type, context, retryable)
            return widget

        # 创建新错误组件
        widget = ErrorDisplayWidget()
        widget.set_error(error_id, error_message, error_type, context, retryable)
        widget.retry_requested.connect(self._on_retry)
        widget.skip_requested.connect(self._on_skip)

        self._errors[error_id] = widget

        # 插入到布局（在 stretch 之前）
        index = self._layout.count() - 1
        self._layout.insertWidget(index, widget)

        logger.debug(f"Error added to list: {error_id}")
        return widget

    def remove_error(self, error_id: str):
        """移除错误.

        Args:
            error_id: 错误 ID
        """
        if error_id in self._errors:
            widget = self._errors.pop(error_id)
            widget.deleteLater()
            logger.debug(f"Error removed from list: {error_id}")

    def clear_errors(self):
        """清除所有错误."""
        for widget in self._errors.values():
            widget.deleteLater()
        self._errors.clear()
        logger.debug("All errors cleared")

    def get_error_count(self) -> int:
        """获取错误数量."""
        return len(self._errors)

    def has_errors(self) -> bool:
        """检查是否有错误."""
        return len(self._errors) > 0

    def get_error(self, error_id: str) -> Optional[ErrorDisplayWidget]:
        """获取错误组件.

        Args:
            error_id: 错误 ID

        Returns:
            错误显示组件或 None
        """
        return self._errors.get(error_id)

    def _on_retry(self, error_id: str):
        """处理重试请求."""
        self.retry_requested.emit(error_id)

    def _on_skip(self, error_id: str):
        """处理跳过请求."""
        self.skip_requested.emit(error_id)


def create_error_display(parent: Optional[QWidget] = None) -> ErrorDisplayWidget:
    """创建 ErrorDisplayWidget 实例.

    Args:
        parent: 父组件

    Returns:
        ErrorDisplayWidget 实例
    """
    return ErrorDisplayWidget(parent)


def create_error_list(parent: Optional[QWidget] = None) -> ErrorListWidget:
    """创建 ErrorListWidget 实例.

    Args:
        parent: 父组件

    Returns:
        ErrorListWidget 实例
    """
    return ErrorListWidget(parent)
