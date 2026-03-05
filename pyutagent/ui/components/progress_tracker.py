"""Progress Tracker - 执行进度条组件.

显示当前步骤/总步骤、进度百分比和步骤详情.
"""

import logging
from typing import Optional, List, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QFrame, QPushButton, QTextEdit, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

logger = logging.getLogger(__name__)


class ProgressTracker(QWidget):
    """进度追踪器组件.

    显示:
    - 进度条和百分比
    - 当前步骤/总步骤
    - 步骤详情（可展开）
    """

    step_clicked = pyqtSignal(int)
    details_expanded = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_step = 0
        self._total_steps = 0
        self._step_details: List[Dict[str, Any]] = []
        self._expanded = False

        self.setup_ui()

    def setup_ui(self):
        """设置 UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # 顶部信息栏
        header = QHBoxLayout()

        # 状态标签
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("font-weight: bold; color: #333;")
        header.addWidget(self._status_label)

        header.addStretch()

        # 步骤计数
        self._step_label = QLabel("0/0")
        self._step_label.setStyleSheet("color: #666; font-size: 12px;")
        header.addWidget(self._step_label)

        layout.addLayout(header)

        # 进度条
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p%")
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                text-align: center;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
            QProgressBar::chunk[success="true"] {
                background-color: #4CAF50;
            }
            QProgressBar::chunk[error="true"] {
                background-color: #F44336;
            }
        """)
        layout.addWidget(self._progress_bar)

        # 展开/折叠按钮
        self._expand_btn = QPushButton("▼ Details")
        self._expand_btn.setFlat(True)
        self._expand_btn.setStyleSheet("""
            QPushButton {
                color: #2196F3;
                text-align: left;
                padding: 4px;
            }
            QPushButton:hover {
                color: #1976D2;
            }
        """)
        self._expand_btn.clicked.connect(self._toggle_details)
        layout.addWidget(self._expand_btn)

        # 详情区域
        self._details_container = QFrame()
        self._details_container.setFrameShape(QFrame.Shape.StyledPanel)
        self._details_container.setStyleSheet("""
            QFrame {
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
            }
        """)
        details_layout = QVBoxLayout(self._details_container)
        details_layout.setContentsMargins(8, 8, 8, 8)

        # 步骤列表
        self._steps_text = QTextEdit()
        self._steps_text.setReadOnly(True)
        self._steps_text.setMaximumHeight(150)
        self._steps_text.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        details_layout.addWidget(self._steps_text)

        self._details_container.hide()
        layout.addWidget(self._details_container)

        layout.addStretch()

    def set_progress(self, current: int, total: int, message: str = ""):
        """设置进度.

        Args:
            current: 当前步骤
            total: 总步骤
            message: 状态消息
        """
        self._current_step = current
        self._total_steps = total

        # 更新进度条
        percentage = int((current / total) * 100) if total > 0 else 0
        self._progress_bar.setValue(percentage)

        # 更新标签
        self._step_label.setText(f"{current}/{total}")
        if message:
            self._status_label.setText(message)

        # 更新样式
        if current >= total and total > 0:
            self._progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #E0E0E0;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #F5F5F5;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
            """)
            self._status_label.setText("Completed")
            self._status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")

        self._update_details_display()

    def set_status(self, status: str, status_type: str = "info"):
        """设置状态.

        Args:
            status: 状态文本
            status_type: 状态类型 (info, success, warning, error)
        """
        self._status_label.setText(status)

        colors = {
            "info": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336"
        }
        color = colors.get(status_type, "#2196F3")
        self._status_label.setStyleSheet(f"font-weight: bold; color: {color};")

        # 更新进度条颜色
        if status_type == "error":
            self._progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #E0E0E0;
                    border-radius: 4px;
                    text-align: center;
                    background-color: #F5F5F5;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 3px;
                }}
            """)

    def add_step_detail(self, step_number: int, description: str,
                       status: str = "pending"):
        """添加步骤详情.

        Args:
            step_number: 步骤编号
            description: 步骤描述
            status: 步骤状态
        """
        self._step_details.append({
            "number": step_number,
            "description": description,
            "status": status
        })
        self._update_details_display()

    def update_step_status(self, step_number: int, status: str):
        """更新步骤状态.

        Args:
            step_number: 步骤编号
            status: 新状态
        """
        for detail in self._step_details:
            if detail["number"] == step_number:
                detail["status"] = status
                break
        self._update_details_display()

    def clear(self):
        """清除所有进度信息."""
        self._current_step = 0
        self._total_steps = 0
        self._step_details.clear()
        self._progress_bar.setValue(0)
        self._step_label.setText("0/0")
        self._status_label.setText("Ready")
        self._status_label.setStyleSheet("font-weight: bold; color: #333;")
        self._steps_text.clear()

        # 重置样式
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                text-align: center;
                background-color: #F5F5F5;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                border-radius: 3px;
            }
        """)

    def _toggle_details(self):
        """切换详情展开/折叠."""
        self._expanded = not self._expanded
        self._details_container.setVisible(self._expanded)
        self._expand_btn.setText("▼ Details" if self._expanded else "▶ Details")
        self.details_expanded.emit(self._expanded)

    def _update_details_display(self):
        """更新详情显示."""
        if not self._step_details:
            self._steps_text.setText("No steps recorded")
            return

        lines = []
        for detail in self._step_details:
            status_icon = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✓",
                "failed": "✗",
                "skipped": "⊘"
            }.get(detail["status"], "•")

            lines.append(
                f"{status_icon} Step {detail['number']}: {detail['description']}"
            )

        self._steps_text.setText("\n".join(lines))

    def get_progress_percent(self) -> float:
        """获取进度百分比."""
        if self._total_steps == 0:
            return 0.0
        return (self._current_step / self._total_steps) * 100

    def is_completed(self) -> bool:
        """检查是否已完成."""
        return self._total_steps > 0 and self._current_step >= self._total_steps


class CircularProgress(QWidget):
    """圆形进度指示器."""

    def __init__(self, diameter: int = 40, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._diameter = diameter
        self._progress = 0
        self._color = "#2196F3"

        self.setFixedSize(diameter, diameter)
        self.setStyleSheet(f"""
            CircularProgress {{
                background-color: transparent;
            }}
        """)

    def set_progress(self, progress: int):
        """设置进度 (0-100)."""
        self._progress = max(0, min(100, progress))
        self.update()

    def set_color(self, color: str):
        """设置颜色."""
        self._color = color
        self.update()

    def paintEvent(self, event):
        """绘制事件."""
        from PyQt6.QtGui import QPainter, QPen, QColor
        from PyQt6.QtCore import QRectF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景圆
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(QColor("#E0E0E0"))
        painter.setPen(pen)

        rect = QRectF(4, 4, self._diameter - 8, self._diameter - 8)
        painter.drawArc(rect, 0, 360 * 16)

        # 进度弧
        pen.setColor(QColor(self._color))
        painter.setPen(pen)

        span_angle = -int(self._progress * 3.6 * 16)
        painter.drawArc(rect, 90 * 16, span_angle)

        painter.end()


def create_progress_tracker(parent: Optional[QWidget] = None) -> ProgressTracker:
    """创建 ProgressTracker 实例.

    Args:
        parent: 父组件

    Returns:
        ProgressTracker 实例
    """
    return ProgressTracker(parent)
