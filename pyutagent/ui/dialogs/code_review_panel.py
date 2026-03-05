"""Code review panel for displaying review results in the UI."""

import logging
from typing import List, Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QScrollArea, QFrame, QSplitter,
    QTextEdit, QProgressBar, QCheckBox, QComboBox, QGroupBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


class IssueListItem(QFrame):
    """Widget for displaying a single issue."""

    clicked = pyqtSignal(str)  # issue_id

    def __init__(self, issue_data: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._issue_data = issue_data
        self._setup_ui()

    def _setup_ui(self):
        severity_colors = {
            "critical": "#F44336",
            "high": "#FF9800",
            "medium": "#FFC107",
            "low": "#2196F3",
            "info": "#9E9E9E"
        }

        severity = self._issue_data.get("severity", "info")
        color = severity_colors.get(severity, "#9E9E9E")

        self.setStyleSheet(f"""
            QFrame {{
                background-color: #2D2D2D;
                border-left: 4px solid {color};
                border-radius: 4px;
                padding: 8px;
                margin: 2px;
            }}
            QFrame:hover {{
                background-color: #3C3C3C;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()

        severity_label = QLabel(f"🔴 {severity.upper()}")
        severity_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
        header.addWidget(severity_label)

        category_label = QLabel(self._issue_data.get("category", "misc"))
        category_label.setStyleSheet("color: #999; font-size: 11px;")
        header.addWidget(category_label)

        header.addStretch()

        line_label = QLabel(f"Line {self._issue_data.get('line_start', '?')}")
        line_label.setStyleSheet("color: #666; font-size: 11px;")
        header.addWidget(line_label)

        layout.addLayout(header)

        message = QLabel(self._issue_data.get("message", ""))
        message.setStyleSheet("color: #DDD; font-size: 12px;")
        message.setWordWrap(True)
        layout.addWidget(message)

        if self._issue_data.get("code_snippet"):
            code = QLabel(f"  {self._issue_data.get('code_snippet')}")
            code.setStyleSheet("""
                color: #888;
                font-family: Consolas, monospace;
                font-size: 11px;
                background-color: #1E1E1E;
                padding: 4px;
                border-radius: 2px;
            """)
            code.setWordWrap(True)
            layout.addWidget(code)

        if self._issue_data.get("suggestion"):
            suggestion = QLabel(f"💡 {self._issue_data.get('suggestion')}")
            suggestion.setStyleSheet("color: #4CAF50; font-size: 11px;")
            suggestion.setWordWrap(True)
            layout.addWidget(suggestion)


class CodeReviewPanel(QWidget):
    """Panel for displaying code review results."""

    issue_selected = pyqtSignal(dict)  # issue data
    apply_fix_requested = pyqtSignal(str)  # issue_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reviews: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        header.setFixedHeight(50)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 8, 12, 8)

        title = QLabel("🔍 Code Review")
        title.setStyleSheet("color: #CCC; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self._score_label = QLabel("Score: --")
        self._score_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 13px;")
        header_layout.addWidget(self._score_label)

        layout.addWidget(header)

        controls = QFrame()
        controls.setStyleSheet("""
            QFrame {
                background-color: #2D2D2D;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 6, 12, 6)

        self._severity_filter = QComboBox()
        self._severity_filter.addItems(["All", "Critical", "High", "Medium", "Low", "Info"])
        self._severity_filter.setStyleSheet("""
            QComboBox {
                background-color: #3C3C3C;
                color: #CCC;
                border: 1px solid #555;
                padding: 4px 8px;
            }
        """)
        controls_layout.addWidget(QLabel("Severity:"))
        controls_layout.addWidget(self._severity_filter)

        self._category_filter = QComboBox()
        self._category_filter.addItems(["All", "Bug", "Security", "Performance", "Style", "Best Practice", "Design"])
        self._category_filter.setStyleSheet("""
            QComboBox {
                background-color: #3C3C3C;
                color: #CCC;
                border: 1px solid #555;
                padding: 4px 8px;
            }
        """)
        controls_layout.addWidget(QLabel("Category:"))
        controls_layout.addWidget(self._category_filter)

        controls_layout.addStretch()

        self._btn_refresh = QPushButton("🔄 Refresh")
        self._btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
            }
        """)
        controls_layout.addWidget(self._btn_refresh)

        layout.addWidget(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._issue_list = QListWidget()
        self._issue_list.setStyleSheet("""
            QListWidget {
                background-color: #1E1E1E;
                border: none;
            }
            QListWidget::item {
                padding: 4px;
            }
        """)
        splitter.addWidget(self._issue_list)

        detail_area = QScrollArea()
        detail_area.setStyleSheet("""
            QScrollArea {
                background-color: #252526;
                border: none;
            }
        """)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)

        detail_title = QLabel("Issue Details")
        detail_title.setStyleSheet("color: #CCC; font-weight: bold; padding: 8px;")
        detail_layout.addWidget(detail_title)

        self._detail_content = QTextEdit()
        self._detail_content.setReadOnly(True)
        self._detail_content.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        detail_layout.addWidget(self._detail_content)

        detail_area.setWidget(detail_widget)
        splitter.addWidget(detail_area)

        splitter.setSizes([300, 400])
        layout.addWidget(splitter)

        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-top: 1px solid #3C3C3C;
            }
        """)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 8, 12, 8)

        self._stats_label = QLabel("No issues")
        self._stats_label.setStyleSheet("color: #888; font-size: 11px;")
        footer_layout.addWidget(self._stats_label)

        footer_layout.addStretch()

        self._btn_apply_fix = QPushButton("🛠️ Apply Fix")
        self._btn_apply_fix.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 3px;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self._btn_apply_fix.setEnabled(False)
        footer_layout.addWidget(self._btn_apply_fix)

        layout.addWidget(footer)

    def set_review_report(self, report: Dict[str, Any]):
        """Set the review report to display."""
        self._reviews = report
        self._update_display()

    def _update_display(self):
        """Update the display with current review data."""
        self._issue_list.clear()

        if not self._reviews:
            self._score_label.setText("Score: --")
            self._stats_label.setText("No issues")
            return

        score = self._reviews.get("score", 100)
        self._score_label.setText(f"Score: {score:.1f}")

        if score >= 80:
            self._score_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif score >= 60:
            self._score_label.setStyleSheet("color: #FFC107; font-weight: bold;")
        else:
            self._score_label.setStyleSheet("color: #F44336; font-weight: bold;")

        issues = self._reviews.get("issues", [])

        severity_filter = self._severity_filter.currentText().lower()
        category_filter = self._category_filter.currentText().lower()

        filtered_issues = issues
        if severity_filter != "all":
            filtered_issues = [i for i in filtered_issues if i.get("severity", "").lower() == severity_filter]
        if category_filter != "all":
            filtered_issues = [i for i in filtered_issues if i.get("category", "").lower() == category_filter.replace(" ", "_")]

        for issue in filtered_issues:
            item = QListWidgetItem()
            widget = IssueListItem(issue)
            item.setSizeHint(widget.sizeHint())
            self._issue_list.addItem(item)
            self._issue_list.setItemWidget(item, widget)

        by_severity = self._reviews.get("issues_by_severity", {})
        stats = ", ".join(f"{k}: {v}" for k, v in by_severity.items())
        self._stats_label.setText(stats or "No issues")

    def get_selected_issue(self) -> Optional[Dict[str, Any]]:
        """Get currently selected issue."""
        current = self._issue_list.currentItem()
        if current:
            return self._issue_list.itemWidget(current)._issue_data
        return None


class CodeReviewDialog(QWidget):
    """Dialog for running code review."""

    review_completed = pyqtSignal(dict)  # review report

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Code Review")
        self.setMinimumSize(800, 600)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E1E;
            }
        """)

        layout = QVBoxLayout(self)

        title = QLabel("🔍 Code Review")
        title.setStyleSheet("""
            color: #4CAF50;
            font-size: 20px;
            font-weight: bold;
            padding: 12px;
        """)
        layout.addWidget(title)

        info = QLabel(
            "This tool analyzes your code for common issues like bugs, "
            "security vulnerabilities, performance problems, and style violations."
        )
        info.setStyleSheet("color: #999; padding: 0 12px 12px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self._review_panel = CodeReviewPanel()
        layout.addWidget(self._review_panel)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCC;
                border: 1px solid #555;
                padding: 8px 20px;
                border-radius: 4px;
            }
        """)
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def set_review_data(self, data: Dict[str, Any]):
        """Set review data to display."""
        self._review_panel.set_review_report(data)
