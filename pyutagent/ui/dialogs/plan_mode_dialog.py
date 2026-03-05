"""Plan mode dialog for interactive task planning with clarification."""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QListWidgetItem, QScrollArea, QWidget,
    QFrame, QProgressBar, QCheckBox, QRadioButton, QButtonGroup,
    QGroupBox, QComboBox, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


@dataclass
class PlanModeConfig:
    """Configuration for plan mode."""
    auto_clarify: bool = True
    max_questions: int = 5
    confidence_threshold: float = 0.7


class ClarificationQuestionWidget(QFrame):
    """Widget for displaying a single clarification question."""

    answered = pyqtSignal(str, str)  # question_id, response

    def __init__(self, question, parent=None):
        super().__init__(parent)
        self._question = question
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #2D2D2D;
                border: 1px solid #3C3C3C;
                border-radius: 6px;
                padding: 12px;
                margin: 4px;
            }
        """)

        layout = QVBoxLayout(self)

        question_label = QLabel(self._question.question_text)
        question_label.setStyleSheet("color: #DDDDDD; font-size: 14px; font-weight: bold;")
        question_label.setWordWrap(True)
        layout.addWidget(question_label)

        if self._question.options:
            self._options_group = QButtonGroup(self)
            for option in self._question.options:
                radio = QRadioButton(option)
                radio.setStyleSheet("color: #CCCCCC;")
                radio.toggled.connect(lambda checked, opt=option: self._on_option_selected(opt) if checked else None)
                self._options_group.addButton(radio)
                layout.addWidget(radio)
        else:
            self._input = QLineEdit()
            self._input.setPlaceholderText("Type your answer...")
            self._input.setStyleSheet("""
                QLineEdit {
                    background-color: #3C3C3C;
                    color: #DDDDDD;
                    border: 1px solid #555;
                    padding: 8px;
                    border-radius: 4px;
                }
            """)
            self._input.returnPressed.connect(self._on_input_submitted)
            layout.addWidget(self._input)

    def _on_option_selected(self, option: str):
        self.answered.emit(self._question.question_id, option)

    def _on_input_submitted(self):
        text = self._input.text().strip()
        if text:
            self.answered.emit(self._question.question_id, text)


class PlanModeDialog(QDialog):
    """Dialog for interactive plan mode with clarification questions."""

    plan_confirmed = pyqtSignal(dict)  # refined plan data

    def __init__(self, task_description: str = "", config: PlanModeConfig = None, parent=None):
        super().__init__(parent)
        self._task_description = task_description
        self._config = config or PlanModeConfig()
        self._clarifier = None
        self._question_widgets: List[ClarificationQuestionWidget] = []

        self.setWindowTitle("Plan Mode - Clarification")
        self.setMinimumSize(700, 600)

        self._setup_ui()
        self._initialize_clarifier()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
            }
        """)

        layout = QVBoxLayout(self)

        header = QLabel("📋 Plan Mode - Task Clarification")
        header.setStyleSheet("""
            color: #4CAF50;
            font-size: 18px;
            font-weight: bold;
            padding: 12px;
        """)
        layout.addWidget(header)

        task_label = QLabel(f"Task: {self._task_description}")
        task_label.setStyleSheet("""
            color: #CCCCCC;
            font-size: 13px;
            padding: 8px;
            background-color: #252526;
            border-radius: 4px;
        """)
        task_label.setWordWrap(True)
        layout.addWidget(task_label)

        info_label = QLabel(
            "Please answer the following questions to help me understand your requirements better:"
        )
        info_label.setStyleSheet("color: #999; font-size: 12px; padding: 8px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        self._questions_scroll = QScrollArea()
        self._questions_scroll.setWidgetResizable(True)
        self._questions_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1E1E1E;
            }
        """)

        self._questions_container = QWidget()
        self._questions_layout = QVBoxLayout(self._questions_container)
        self._questions_scroll.setWidget(self._questions_container)

        layout.addWidget(self._questions_scroll, stretch=1)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #858585; font-size: 11px; padding: 8px;")
        layout.addWidget(self._status_label)

        button_layout = QHBoxLayout()

        self._btn_skip = QPushButton("Skip Clarification")
        self._btn_skip.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 10px 20px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
            }
        """)
        self._btn_skip.clicked.connect(self._on_skip)
        button_layout.addWidget(self._btn_skip)

        button_layout.addStretch()

        self._btn_confirm = QPushButton("Confirm Plan")
        self._btn_confirm.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self._btn_confirm.clicked.connect(self._on_confirm)
        button_layout.addWidget(self._btn_confirm)

        layout.addLayout(button_layout)

    def _initialize_clarifier(self):
        """Initialize the plan clarifier."""
        try:
            from pyutagent.agent.enhanced_planner import PlanClarifier
            self._clarifier = PlanClarifier()
            self._clarifier.analyze_task(self._task_description)
            self._display_questions()
        except Exception as e:
            logger.error(f"Failed to initialize clarifier: {e}")

    def _display_questions(self):
        """Display pending questions."""
        for i in reversed(range(self._questions_layout.count())):
            widget = self._questions_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        self._question_widgets.clear()

        if not self._clarifier:
            return

        pending = self._clarifier.get_current_plan()
        if not pending:
            return

        answered_ids = {r.question_id for r in pending.user_responses}

        for question in pending.questions_asked:
            if question.question_id not in answered_ids:
                widget = ClarificationQuestionWidget(question)
                widget.answered.connect(self._on_question_answered)
                self._questions_layout.addWidget(widget)
                self._question_widgets.append(widget)

        self._update_status()

    def _on_question_answered(self, question_id: str, response: str):
        """Handle question answered."""
        if self._clarifier:
            self._clarifier.add_response(question_id, response)
            self._display_questions()

    def _update_status(self):
        """Update status label."""
        if not self._clarifier:
            return

        plan = self._clarifier.get_current_plan()
        if not plan:
            return

        answered = len(plan.user_responses)
        total = len(plan.questions_asked)

        if answered >= total:
            self._status_label.setText(f"✅ All questions answered. Ready to confirm.")
            self._btn_confirm.setEnabled(True)
        else:
            self._status_label.setText(f"Question {answered + 1} of {total} ({answered} answered)")
            self._btn_confirm.setEnabled(answered > 0)

    def _on_skip(self):
        """Handle skip button."""
        logger.info("Plan mode clarification skipped")
        self.accept()

    def _on_confirm(self):
        """Handle confirm button."""
        if not self._clarifier:
            self.accept()
            return

        plan = self._clarifier.get_current_plan()
        if plan:
            logger.info(f"Plan confirmed with {len(plan.user_responses)} clarifications")
            self.plan_confirmed.emit(plan.to_dict())

        self.accept()

    def get_refined_plan(self) -> Optional[Dict[str, Any]]:
        """Get the refined plan data."""
        if self._clarifier:
            plan = self._clarifier.get_current_plan()
            if plan:
                return plan.to_dict()
        return None


class QuickPlanDialog(QDialog):
    """Quick plan dialog for simple planning."""

    plan_created = pyqtSignal(str, list)  # task, steps

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quick Plan")
        self.setMinimumSize(500, 400)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
            }
        """)

        layout = QVBoxLayout(self)

        title = QLabel("⚡ Quick Plan")
        title.setStyleSheet("color: #4CAF50; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        task_label = QLabel("What would you like to do?")
        task_label.setStyleSheet("color: #CCCCCC;")
        layout.addWidget(task_label)

        self._task_input = QTextEdit()
        self._task_input.setPlaceholderText("Describe your task...")
        self._task_input.setStyleSheet("""
            QTextEdit {
                background-color: #2D2D2D;
                color: #DDDDDD;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        self._task_input.setMinimumHeight(100)
        layout.addWidget(self._task_input)

        quick_actions = QLabel("Quick Actions:")
        quick_actions.setStyleSheet("color: #999; font-size: 12px; margin-top: 12px;")
        layout.addWidget(quick_actions)

        buttons_layout = QHBoxLayout()

        test_btn = QPushButton("🧪 Generate Tests")
        test_btn.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1177BB; }
        """)
        test_btn.clicked.connect(lambda: self._quick_action("Generate unit tests"))
        buttons_layout.addWidget(test_btn)

        refactor_btn = QPushButton("🔧 Refactor Code")
        refactor_btn.setStyleSheet("""
            QPushButton {
                background-color: #6A1B9A;
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #7B1FA2; }
        """)
        refactor_btn.clicked.connect(lambda: self._quick_action("Refactor code"))
        buttons_layout.addWidget(refactor_btn)

        fix_btn = QPushButton("🐛 Fix Bug")
        fix_btn.setStyleSheet("""
            QPushButton {
                background-color: #C62828;
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #D32F2F; }
        """)
        fix_btn.clicked.connect(lambda: self._quick_action("Fix bug"))
        buttons_layout.addWidget(fix_btn)

        layout.addLayout(buttons_layout)

        layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 8px 16px;
                border-radius: 4px;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        create_btn = QPushButton("Create Plan")
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        create_btn.clicked.connect(self._create_plan)
        button_layout.addWidget(create_btn)

        layout.addLayout(button_layout)

    def _quick_action(self, action: str):
        """Handle quick action button."""
        self._task_input.setText(action)

    def _create_plan(self):
        """Create and emit plan."""
        task = self._task_input.toPlainText().strip()
        if task:
            self.plan_created.emit(task, [])
            self.accept()
