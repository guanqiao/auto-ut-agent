"""Enhanced diff approval system with per-file and per-line review capabilities.

This module provides:
- ApprovalDiffViewer: Main diff viewer with approval workflow
- ChangeApproval: Individual change approval tracking
- ApprovalMode: Manual vs Autonomous mode
- ApprovalManager: Manages approval workflow state
"""

import logging
from typing import Optional, List, Dict, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import difflib

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel,
    QPushButton, QSplitter, QFrame, QScrollArea, QCheckBox,
    QButtonGroup, QRadioButton, QGroupBox, QListWidget, QListWidgetItem,
    QStackedWidget, QProgressBar, QToolButton, QMenu, QDialog,
    QDialogButtonBox, QScrollBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QTextCursor, QAction

logger = logging.getLogger(__name__)


class ApprovalMode(Enum):
    """Approval mode for changes."""
    MANUAL = "manual"  # User approves each change
    AUTONOMOUS = "autonomous"  # Agent works autonomously


class ChangeStatus(Enum):
    """Status of a change."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class ApprovalLevel(Enum):
    """Level of approval granularity."""
    FILE = "file"  # Per-file approval
    CHUNK = "chunk"  # Per-chunk (section) approval
    LINE = "line"  # Per-line approval


@dataclass
class FileChange:
    """Represents a file change."""
    file_path: str
    old_content: str
    new_content: str
    change_status: ChangeStatus = ChangeStatus.PENDING
    approved_lines: Set[int] = field(default_factory=set)
    rejected_lines: Set[int] = field(default_factory=set)

    @property
    def has_changes(self) -> bool:
        return self.old_content != self.new_content

    @property
    def change_count(self) -> int:
        if not self.has_changes:
            return 0
        old_lines = self.old_content.splitlines()
        new_lines = self.new_content.splitlines()
        sm = difflib.SequenceMatcher(None, old_lines, new_lines)
        changes = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ('delete', 'insert', 'replace'):
                changes += max(i2 - i1, j2 - j1)
        return changes


@dataclass
class LineChange:
    """Represents a single line change."""
    line_num: int
    old_content: Optional[str]
    new_content: Optional[str]
    change_type: str  # 'added', 'removed', 'modified', 'unchanged'
    status: ChangeStatus = ChangeStatus.PENDING

    def is_addition(self) -> bool:
        return self.change_type == 'added'

    def is_deletion(self) -> bool:
        return self.change_type == 'removed'

    def is_modification(self) -> bool:
        return self.change_type == 'modified'


@dataclass
class ChunkChange:
    """Represents a chunk of changes (e.g., a function or block)."""
    start_line: int
    end_line: int
    lines: List[LineChange]
    status: ChangeStatus = ChangeStatus.PENDING

    @property
    def change_count(self) -> int:
        return sum(1 for line in self.lines if line.change_type != 'unchanged')


class ApprovalManager:
    """Manages the approval workflow state."""

    def __init__(self):
        self._file_changes: Dict[str, FileChange] = {}
        self._approval_mode: ApprovalMode = ApprovalMode.MANUAL
        self._approval_level: ApprovalLevel = ApprovalLevel.FILE
        self._on_change_callbacks: List[Callable] = []

        self._stats = {
            'total_files': 0,
            'approved_files': 0,
            'rejected_files': 0,
            'pending_files': 0,
            'total_changes': 0,
            'approved_changes': 0,
            'rejected_changes': 0,
        }

    def add_file_change(self, file_path: str, old_content: str, new_content: str):
        """Add a file change to track."""
        change = FileChange(
            file_path=file_path,
            old_content=old_content,
            new_content=new_content
        )
        self._file_changes[file_path] = change
        self._update_stats()
        self._notify_change()

    def remove_file_change(self, file_path: str):
        """Remove a file change from tracking."""
        if file_path in self._file_changes:
            del self._file_changes[file_path]
            self._update_stats()
            self._notify_change()

    def get_file_change(self, file_path: str) -> Optional[FileChange]:
        """Get file change by path."""
        return self._file_changes.get(file_path)

    def get_all_file_changes(self) -> List[FileChange]:
        """Get all file changes."""
        return list(self._file_changes.values())

    def approve_file(self, file_path: str) -> bool:
        """Approve a file change."""
        if file_path in self._file_changes:
            self._file_changes[file_path].change_status = ChangeStatus.APPROVED
            self._update_stats()
            self._notify_change()
            return True
        return False

    def reject_file(self, file_path: str) -> bool:
        """Reject a file change."""
        if file_path in self._file_changes:
            self._file_changes[file_path].change_status = ChangeStatus.REJECTED
            self._update_stats()
            self._notify_change()
            return True
        return False

    def approve_line(self, file_path: str, line_num: int) -> bool:
        """Approve a specific line in a file."""
        if file_path in self._file_changes:
            fc = self._file_changes[file_path]
            fc.approved_lines.add(line_num)
            if line_num in fc.rejected_lines:
                fc.rejected_lines.remove(line_num)
            self._update_stats()
            self._notify_change()
            return True
        return False

    def reject_line(self, file_path: str, line_num: int) -> bool:
        """Reject a specific line in a file."""
        if file_path in self._file_changes:
            fc = self._file_changes[file_path]
            fc.rejected_lines.add(line_num)
            if line_num in fc.approved_lines:
                fc.approved_lines.remove(line_num)
            self._update_stats()
            self._notify_change()
            return True
        return False

    def approve_all(self):
        """Approve all file changes."""
        for fc in self._file_changes.values():
            fc.change_status = ChangeStatus.APPROVED
        self._update_stats()
        self._notify_change()

    def reject_all(self):
        """Reject all file changes."""
        for fc in self._file_changes.values():
            fc.change_status = ChangeStatus.REJECTED
        self._update_stats()
        self._notify_change()

    def reset_all(self):
        """Reset all approvals."""
        for fc in self._file_changes.values():
            fc.change_status = ChangeStatus.PENDING
            fc.approved_lines.clear()
            fc.rejected_lines.clear()
        self._update_stats()
        self._notify_change()

    def get_approved_files(self) -> List[str]:
        """Get list of approved file paths."""
        return [
            fc.file_path for fc in self._file_changes.values()
            if fc.change_status == ChangeStatus.APPROVED
        ]

    def get_rejected_files(self) -> List[str]:
        """Get list of rejected file paths."""
        return [
            fc.file_path for fc in self._file_changes.values()
            if fc.change_status == ChangeStatus.REJECTED
        ]

    def get_pending_files(self) -> List[str]:
        """Get list of pending file paths."""
        return [
            fc.file_path for fc in self._file_changes.values()
            if fc.change_status == ChangeStatus.PENDING
        ]

    def get_stats(self) -> Dict[str, int]:
        """Get approval statistics."""
        return self._stats.copy()

    def set_approval_mode(self, mode: ApprovalMode):
        """Set approval mode."""
        self._approval_mode = mode
        self._notify_change()

    def get_approval_mode(self) -> ApprovalMode:
        """Get current approval mode."""
        return self._approval_mode

    def set_approval_level(self, level: ApprovalLevel):
        """Set approval granularity level."""
        self._approval_level = level
        self._notify_change()

    def get_approval_level(self) -> ApprovalLevel:
        """Get current approval level."""
        return self._approval_level

    def on_change(self, callback: Callable):
        """Register a callback for changes."""
        self._on_change_callbacks.append(callback)

    def _update_stats(self):
        """Update statistics."""
        self._stats['total_files'] = len(self._file_changes)
        self._stats['approved_files'] = sum(
            1 for fc in self._file_changes.values()
            if fc.change_status == ChangeStatus.APPROVED
        )
        self._stats['rejected_files'] = sum(
            1 for fc in self._file_changes.values()
            if fc.change_status == ChangeStatus.REJECTED
        )
        self._stats['pending_files'] = sum(
            1 for fc in self._file_changes.values()
            if fc.change_status == ChangeStatus.PENDING
        )
        self._stats['total_changes'] = sum(
            fc.change_count for fc in self._file_changes.values()
        )
        self._stats['approved_changes'] = sum(
            len(fc.approved_lines) for fc in self._file_changes.values()
        )
        self._stats['rejected_changes'] = sum(
            len(fc.rejected_lines) for fc in self._file_changes.values()
        )

    def _notify_change(self):
        """Notify all registered callbacks."""
        for callback in self._on_change_callbacks:
            try:
                callback()
            except Exception as e:
                logger.exception(f"Error in approval change callback: {e}")

    def clear(self):
        """Clear all changes."""
        self._file_changes.clear()
        self._update_stats()
        self._notify_change()


class ApprovalDiffViewer(QWidget):
    """Enhanced diff viewer with approval workflow.

    Features:
    - Multi-file diff display
    - Per-file and per-line approval
    - Manual and autonomous modes
    - Approval statistics
    - Batch operations
    """

    approval_completed = pyqtSignal(list, list)  # approved files, rejected files
    approval_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._manager = ApprovalManager()
        self._current_file: Optional[str] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with controls
        header = self._create_header()
        layout.addWidget(header)

        # Main content area
        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(1)

        # File list sidebar
        self._file_list = self._create_file_list()
        content_layout.addWidget(self._file_list, 1)

        # Diff view area
        self._diff_stack = QStackedWidget()
        content_layout.addWidget(self._diff_stack, 4)

        layout.addWidget(content, stretch=1)

        # Footer with stats and actions
        footer = self._create_footer()
        layout.addWidget(footer)

    def _create_header(self) -> QFrame:
        """Create header with mode controls."""
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

        # Title
        self._title_label = QLabel("📝 Code Change Review")
        self._title_label.setStyleSheet("color: #CCCCCC; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(self._title_label)

        header_layout.addStretch()

        # Mode selector
        mode_group = QButtonGroup()
        mode_group.setId(ApprovalMode.MANUAL, 0)
        mode_group.setId(ApprovalMode.AUTONOMOUS, 1)

        self._mode_manual = QRadioButton("👤 Manual")
        self._mode_manual.setStyleSheet("color: #CCCCCC;")
        self._mode_manual.setChecked(True)
        mode_group.addButton(self._mode_manual, 0)

        self._mode_autonomous = QRadioButton("🤖 Autonomous")
        self._mode_autonomous.setStyleSheet("color: #CCCCCC;")
        mode_group.addButton(self._mode_autonomous, 1)

        header_layout.addWidget(self._mode_manual)
        header_layout.addWidget(self._mode_autonomous)

        header_layout.addSpacing(16)

        # Level selector
        self._level_combo = QLabel("Granularity:")
        self._level_combo.setStyleSheet("color: #858585;")
        header_layout.addWidget(self._level_combo)

        self._btn_level_file = QPushButton("File")
        self._btn_level_file.setCheckable(True)
        self._btn_level_file.setChecked(True)
        self._btn_level_file.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #0E639C;
            }
        """)

        self._btn_level_line = QPushButton("Line")
        self._btn_level_line.setCheckable(True)
        self._btn_level_line.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QPushButton:checked {
                background-color: #0E639C;
            }
        """)

        header_layout.addWidget(self._btn_level_file)
        header_layout.addWidget(self._btn_level_line)

        return header

    def _create_file_list(self) -> QWidget:
        """Create file list sidebar."""
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-right: 1px solid #3C3C3C;
            }
        """)
        container.setFixedWidth(280)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # File list header
        file_header = QLabel("Changed Files")
        file_header.setStyleSheet("""
            background-color: #2D2D2D;
            color: #CCCCCC;
            padding: 8px;
            font-weight: bold;
        """)
        layout.addWidget(file_header)

        # File list
        self._file_list_widget = QListWidget()
        self._file_list_widget.setStyleSheet("""
            QListWidget {
                background-color: #252526;
                color: #CCCCCC;
                border: none;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3C3C3C;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QListWidget::item:hover {
                background-color: #2A2D2E;
            }
        """)
        layout.addWidget(self._file_list_widget)

        return container

    def _create_diff_viewer(self, file_path: str) -> QWidget:
        """Create a diff viewer for a specific file."""
        viewer = QWidget()
        layout = QVBoxLayout(viewer)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QFrame()
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #2D2D2D;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        toolbar.setFixedHeight(40)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)

        # File path label
        file_label = QLabel(f"📄 {file_path}")
        file_label.setStyleSheet("color: #CCCCCC;")
        toolbar_layout.addWidget(file_label)

        toolbar_layout.addStretch()

        # Per-file buttons (for file-level approval)
        self._btn_approve_file = QPushButton("✓ Approve File")
        self._btn_approve_file.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self._btn_reject_file = QPushButton("✗ Reject File")
        self._btn_reject_file.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 4px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)

        toolbar_layout.addWidget(self._btn_approve_file)
        toolbar_layout.addWidget(self._btn_reject_file)

        layout.addWidget(toolbar)

        # Diff content
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Old content
        old_container = QWidget()
        old_layout = QVBoxLayout(old_container)
        old_layout.setContentsMargins(0, 0, 0, 0)

        old_header = QLabel("📄 Original")
        old_header.setStyleSheet("""
            background-color: #1E1E1E;
            color: #CCCCCC;
            padding: 6px;
            border-bottom: 1px solid #3C3C3C;
        """)
        old_layout.addWidget(old_header)

        self._old_editor = QTextEdit()
        self._old_editor.setReadOnly(True)
        self._old_editor.setFont(QFont("Consolas", 11))
        self._old_editor.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
            }
        """)
        old_layout.addWidget(self._old_editor)

        splitter.addWidget(old_container)

        # New content
        new_container = QWidget()
        new_layout = QVBoxLayout(new_container)
        new_layout.setContentsMargins(0, 0, 0, 0)

        new_header = QLabel("✏️ Modified")
        new_header.setStyleSheet("""
            background-color: #1E1E1E;
            color: #CCCCCC;
            padding: 6px;
            border-bottom: 1px solid #3C3C3C;
        """)
        new_layout.addWidget(new_header)

        self._new_editor = QTextEdit()
        self._new_editor.setReadOnly(True)
        self._new_editor.setFont(QFont("Consolas", 11))
        self._new_editor.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
            }
        """)
        new_layout.addWidget(self._new_editor)

        splitter.addWidget(new_container)

        splitter.setSizes([400, 400])
        layout.addWidget(splitter)

        return viewer

    def _create_footer(self) -> QFrame:
        """Create footer with stats and actions."""
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-top: 1px solid #3C3C3C;
            }
        """)
        footer.setFixedHeight(50)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 8, 12, 8)

        # Stats
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #858585; font-size: 12px;")
        footer_layout.addWidget(self._stats_label)

        footer_layout.addStretch()

        # Action buttons
        self._btn_reset = QPushButton("🔄 Reset")
        self._btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 6px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
            }
        """)

        self._btn_reject_all = QPushButton("✗ Reject All")
        self._btn_reject_all.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)

        self._btn_accept_all = QPushButton("✓ Accept All")
        self._btn_accept_all.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self._btn_apply = QPushButton("🚀 Apply Changes")
        self._btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 6px 20px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
        """)

        footer_layout.addWidget(self._btn_reset)
        footer_layout.addWidget(self._btn_reject_all)
        footer_layout.addWidget(self._btn_accept_all)
        footer_layout.addWidget(self._btn_apply)

        return footer

    def _connect_signals(self):
        """Connect signals and slots."""
        self._file_list_widget.currentRowChanged.connect(self._on_file_selected)

        self._mode_manual.toggled.connect(self._on_mode_changed)
        self._btn_level_file.clicked.connect(lambda: self._on_level_changed(ApprovalLevel.FILE))
        self._btn_level_line.clicked.connect(lambda: self._on_level_changed(ApprovalLevel.LINE))

        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_reject_all.clicked.connect(self._on_reject_all)
        self._btn_accept_all.clicked.connect(self._on_accept_all)
        self._btn_apply.clicked.connect(self._on_apply)

        self._manager.on_change(self._on_approval_changed)

    def add_file_change(self, file_path: str, old_content: str, new_content: str):
        """Add a file change to review."""
        self._manager.add_file_change(file_path, old_content, new_content)

        # Add to file list
        item = QListWidgetItem(f"📝 {Path(file_path).name}")
        item.setData(Qt.ItemDataRole.UserRole, file_path)

        # Set color based on change type
        change = self._manager.get_file_change(file_path)
        if change and change.change_count > 0:
            item.setText(f"📝 {Path(file_path).name} ({change.change_count} changes)")
        else:
            item.setText(f"📝 {Path(file_path).name}")

        self._file_list_widget.addItem(item)

        # Create diff viewer
        viewer = self._create_diff_viewer(file_path)
        viewer.set_old_content = lambda old: viewer.findChild(QTextEdit, "_old_editor" if hasattr(viewer, '_old_editor') else None)
        self._diff_stack.addWidget(viewer)

        self._update_stats()

    def _on_file_selected(self, row: int):
        """Handle file selection."""
        if row >= 0:
            item = self._file_list_widget.item(row)
            if item:
                file_path = item.data(Qt.ItemDataRole.UserRole)
                self._current_file = file_path
                self._diff_stack.setCurrentIndex(row)

                # Update diff content
                change = self._manager.get_file_change(file_path)
                if change:
                    self._display_diff(change)

    def _display_diff(self, change: FileChange):
        """Display diff for a file."""
        viewer = self._diff_stack.currentWidget()
        if not viewer:
            return

        # Find editors
        old_editor = viewer.findChild(QTextEdit, "_old_editor")
        new_editor = viewer.findChild(QTextEdit, "_new_editor")

        if not old_editor or not new_editor:
            return

        # Compute diff
        diff_lines = self._compute_diff(change.old_content, change.new_content)

        # Display
        self._render_diff(old_editor, new_editor, diff_lines, change)

    def _compute_diff(self, old_content: str, new_content: str) -> List[Dict]:
        """Compute diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        sm = difflib.SequenceMatcher(None, old_lines, new_lines)

        diff_lines = []
        old_line_num = 1
        new_line_num = 1

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for i in range(i1, i2):
                    diff_lines.append({
                        'old_line': old_line_num,
                        'new_line': new_line_num,
                        'content': old_lines[i].rstrip('\n\r'),
                        'type': 'unchanged'
                    })
                    old_line_num += 1
                    new_line_num += 1

            elif tag == 'delete':
                for i in range(i1, i2):
                    diff_lines.append({
                        'old_line': old_line_num,
                        'new_line': None,
                        'content': old_lines[i].rstrip('\n\r'),
                        'type': 'removed'
                    })
                    old_line_num += 1

            elif tag == 'insert':
                for j in range(j1, j2):
                    diff_lines.append({
                        'old_line': None,
                        'new_line': new_line_num,
                        'content': new_lines[j].rstrip('\n\r'),
                        'type': 'added'
                    })
                    new_line_num += 1

            elif tag == 'replace':
                max_len = max(i2 - i1, j2 - j1)
                for k in range(max_len):
                    if i1 + k < i2:
                        diff_lines.append({
                            'old_line': old_line_num,
                            'new_line': None,
                            'content': old_lines[i1 + k].rstrip('\n\r'),
                            'type': 'removed'
                        })
                        old_line_num += 1

                    if j1 + k < j2:
                        diff_lines.append({
                            'old_line': None,
                            'new_line': new_line_num,
                            'content': new_lines[j1 + k].rstrip('\n\r'),
                            'type': 'added'
                        })
                        new_line_num += 1

        return diff_lines

    def _render_diff(self, old_editor: QTextEdit, new_editor: QTextEdit,
                     diff_lines: List[Dict], change: FileChange):
        """Render diff with syntax highlighting."""
        old_html = ['<pre style="margin: 0; font-family: Consolas, monospace; font-size: 11px;">']
        new_html = ['<pre style="margin: 0; font-family: Consolas, monospace; font-size: 11px;">']

        for line in diff_lines:
            old_num = f"{line['old_line']:4d}" if line['old_line'] else "    "
            new_num = f"{line['new_line']:4d}" if line['new_line'] else "    "
            content = line['content'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            if line['type'] == 'unchanged':
                old_html.append(f'<span style="color: #D4D4D4;">{old_num} │ {content}</span>')
                new_html.append(f'<span style="color: #D4D4D4;">{new_num} │ {content}</span>')
            elif line['type'] == 'added':
                old_html.append('<span style="color: #555;">    │ </span>')
                bg = "#1E4620" if line['new_line'] not in change.rejected_lines else "#4A1E1E"
                text = "#B5F2A6" if line['new_line'] not in change.rejected_lines else "#F2A6A6"
                new_html.append(f'<span style="background-color: {bg}; color: {text};">{new_num} │ +{content}</span>')
            elif line['type'] == 'removed':
                bg = "#4A1E1E" if line['old_line'] not in change.approved_lines else "#1E4620"
                text = "#F2A6A6" if line['old_line'] not in change.approved_lines else "#B5F2A6"
                old_html.append(f'<span style="background-color: {bg}; color: {text};">{old_num} │ -{content}</span>')
                new_html.append('<span style="color: #555;">    │ </span>')

        old_html.append('</pre>')
        new_html.append('</pre>')

        old_editor.setHtml(''.join(old_html))
        new_editor.setHtml(''.join(new_html))

    def _on_mode_changed(self, checked: bool):
        """Handle mode change."""
        if checked and self._mode_manual.isChecked():
            self._manager.set_approval_mode(ApprovalMode.MANUAL)
            self._update_ui_for_mode()
        elif checked and self._mode_autonomous.isChecked():
            self._manager.set_approval_mode(ApprovalMode.AUTONOMOUS)
            self._update_ui_for_mode()

    def _on_level_changed(self, level: ApprovalLevel):
        """Handle approval level change."""
        self._btn_level_file.setChecked(level == ApprovalLevel.FILE)
        self._btn_level_line.setChecked(level == ApprovalLevel.LINE)
        self._manager.set_approval_level(level)
        self._update_ui_for_level()

    def _update_ui_for_mode(self):
        """Update UI based on current mode."""
        mode = self._manager.get_approval_mode()

        if mode == ApprovalMode.AUTONOMOUS:
            self._btn_accept_all.setEnabled(True)
            self._btn_reject_all.setEnabled(True)
            self._btn_apply.setEnabled(True)
        else:
            self._btn_accept_all.setEnabled(True)
            self._btn_reject_all.setEnabled(True)
            self._btn_apply.setEnabled(True)

    def _update_ui_for_level(self):
        """Update UI based on approval level."""
        level = self._manager.get_approval_level()

        # Show/hide per-line buttons
        self._btn_approve_file.setVisible(level == ApprovalLevel.FILE)
        self._btn_reject_file.setVisible(level == ApprovalLevel.FILE)

    def _on_reset(self):
        """Handle reset button click."""
        self._manager.reset_all()
        self._update_file_list_colors()

    def _on_reject_all(self):
        """Handle reject all button click."""
        self._manager.reject_all()
        self._update_file_list_colors()
        self._update_stats()

    def _on_accept_all(self):
        """Handle accept all button click."""
        self._manager.approve_all()
        self._update_file_list_colors()
        self._update_stats()

    def _on_apply(self):
        """Handle apply button click."""
        approved = self._manager.get_approved_files()
        rejected = self._manager.get_rejected_files()
        self.approval_completed.emit(approved, rejected)

    def _on_approval_changed(self):
        """Handle approval change."""
        self._update_file_list_colors()
        self._update_stats()
        self.approval_changed.emit()

    def _update_file_list_colors(self):
        """Update file list item colors based on status."""
        for i in range(self._file_list_widget.count()):
            item = self._file_list_widget.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            change = self._manager.get_file_change(file_path)

            if change:
                if change.change_status == ChangeStatus.APPROVED:
                    item.setBackground(QColor("#1E4620"))
                elif change.change_status == ChangeStatus.REJECTED:
                    item.setBackground(QColor("#4A1E1E"))
                else:
                    item.setBackground(QColor("#252526"))

    def _update_stats(self):
        """Update statistics display."""
        stats = self._manager.get_stats()

        if stats['total_files'] == 0:
            self._stats_label.setText("No changes to review")
            return

        approved = stats['approved_files']
        rejected = stats['rejected_files']
        pending = stats['pending_files']
        total = stats['total_files']

        self._stats_label.setText(
            f"<span style='color: #4CAF50;'>✓ {approved}</span> approved, "
            f"<span style='color: #F44336;'>✗ {rejected}</span> rejected, "
            f"<span style='color: #FFC107;'>⏳ {pending}</span> pending | "
            f"Total: {total} files"
        )

    def get_manager(self) -> ApprovalManager:
        """Get the approval manager."""
        return self._manager

    def clear(self):
        """Clear all changes."""
        self._manager.clear()
        self._file_list_widget.clear()

        while self._diff_stack.count() > 0:
            widget = self._diff_stack.widget(0)
            self._diff_stack.removeWidget(widget)
            widget.deleteLater()

        self._current_file = None
        self._update_stats()


class ApprovalDialog(QDialog):
    """Dialog for reviewing and approving changes."""

    def __init__(self, parent: Optional[QWidget] = None, title: str = "Review Changes"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 700)

        self._viewer = ApprovalDiffViewer(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._viewer)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(button_box)

        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-top: 1px solid #3C3C3C;
            }
        """)
        footer.setFixedHeight(50)
        footer.setLayout(btn_layout)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._viewer, stretch=1)
        main_layout.addWidget(footer)

        self._viewer.approval_completed.connect(self._on_approval_completed)

    def add_file_change(self, file_path: str, old_content: str, new_content: str):
        """Add a file change to review."""
        self._viewer.add_file_change(file_path, old_content, new_content)

    def get_approved_files(self) -> List[str]:
        """Get approved file paths."""
        return self._viewer.get_manager().get_approved_files()

    def get_rejected_files(self) -> List[str]:
        """Get rejected file paths."""
        return self._viewer.get_manager().get_rejected_files()

    def _on_approval_completed(self, approved: List[str], rejected: List[str]):
        """Handle approval completion."""
        logger.info(f"Approval completed: {len(approved)} approved, {len(rejected)} rejected")
