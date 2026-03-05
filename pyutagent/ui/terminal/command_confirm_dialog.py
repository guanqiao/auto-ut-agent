"""Command confirmation dialog for dangerous operations."""

import logging
from typing import Optional, List, Set
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QCheckBox, QFrame, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


class CommandRisk(Enum):
    """Risk level of a command."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CommandInfo:
    """Information about a command to be executed."""
    command: str
    description: str = ""
    risk_level: CommandRisk = CommandRisk.LOW
    affected_files: List[str] = None
    can_undo: bool = False


class CommandConfirmDialog(QDialog):
    """Dialog for confirming potentially dangerous commands.
    
    Features:
    - Risk level indication
    - Command preview
    - Affected files display
    - Trust this command option
    - Remember choice option
    """
    
    # Signals
    command_allowed = pyqtSignal(str)  # command
    command_denied = pyqtSignal(str)  # command
    
    # Risk level colors
    RISK_COLORS = {
        CommandRisk.LOW: '#4CAF50',
        CommandRisk.MEDIUM: '#FF9800',
        CommandRisk.HIGH: '#F44336',
        CommandRisk.CRITICAL: '#9C27B0'
    }
    
    RISK_ICONS = {
        CommandRisk.LOW: '✓',
        CommandRisk.MEDIUM: '⚠️',
        CommandRisk.HIGH: '⚠️',
        CommandRisk.CRITICAL: '🚫'
    }
    
    def __init__(self, command_info: CommandInfo, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._command_info = command_info
        self._trust_checked = False
        
        self.setWindowTitle("Confirm Command Execution")
        self.setMinimumWidth(500)
        self.setModal(True)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with risk indicator
        header = QFrame()
        risk_color = self.RISK_COLORS.get(self._command_info.risk_level, '#757575')
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {risk_color}20;
                border: 1px solid {risk_color};
                border-radius: 6px;
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        
        # Risk icon
        icon = self.RISK_ICONS.get(self._command_info.risk_level, '?')
        icon_label = QLabel(icon)
        icon_label.setStyleSheet(f"font-size: 24px;")
        header_layout.addWidget(icon_label)
        
        header_layout.addSpacing(12)
        
        # Risk text
        risk_text = f"Risk Level: {self._command_info.risk_level.value.upper()}"
        risk_label = QLabel(risk_text)
        risk_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: bold;
            color: {risk_color};
        """)
        header_layout.addWidget(risk_label)
        
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # Warning message
        if self._command_info.risk_level in [CommandRisk.HIGH, CommandRisk.CRITICAL]:
            warning = QLabel("⚠️ This command could be dangerous. Please review carefully before proceeding.")
            warning.setStyleSheet("color: #F44336; font-weight: bold;")
            warning.setWordWrap(True)
            layout.addWidget(warning)
        
        # Command preview
        command_frame = QFrame()
        command_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
            }
        """)
        command_layout = QVBoxLayout(command_frame)
        command_layout.setContentsMargins(12, 12, 12, 12)
        
        command_label = QLabel("Command to execute:")
        command_label.setStyleSheet("color: #858585; font-size: 11px;")
        command_layout.addWidget(command_label)
        
        command_display = QTextEdit()
        command_display.setPlainText(self._command_info.command)
        command_display.setReadOnly(True)
        command_display.setFont(QFont("Consolas", 11))
        command_display.setStyleSheet("""
            QTextEdit {
                background-color: transparent;
                color: #D4D4D4;
                border: none;
            }
        """)
        command_display.setMaximumHeight(80)
        command_layout.addWidget(command_display)
        
        layout.addWidget(command_frame)
        
        # Description
        if self._command_info.description:
            desc_label = QLabel(self._command_info.description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #CCCCCC;")
            layout.addWidget(desc_label)
        
        # Affected files
        if self._command_info.affected_files:
            files_frame = QFrame()
            files_frame.setStyleSheet("""
                QFrame {
                    background-color: #252526;
                    border: 1px solid #3C3C3C;
                    border-radius: 4px;
                }
            """)
            files_layout = QVBoxLayout(files_frame)
            files_layout.setContentsMargins(12, 12, 12, 12)
            
            files_label = QLabel(f"Affected files ({len(self._command_info.affected_files)}):")
            files_label.setStyleSheet("color: #858585; font-size: 11px;")
            files_layout.addWidget(files_label)
            
            files_text = QTextEdit()
            files_text.setPlainText('\n'.join(self._command_info.affected_files[:20]))
            if len(self._command_info.affected_files) > 20:
                files_text.append(f"\n... and {len(self._command_info.affected_files) - 20} more files")
            files_text.setReadOnly(True)
            files_text.setFont(QFont("Consolas", 10))
            files_text.setStyleSheet("""
                QTextEdit {
                    background-color: transparent;
                    color: #D4D4D4;
                    border: none;
                }
            """)
            files_text.setMaximumHeight(100)
            files_layout.addWidget(files_text)
            
            layout.addWidget(files_frame)
        
        # Undo warning
        if not self._command_info.can_undo:
            undo_warning = QLabel("⚠️ This action cannot be easily undone.")
            undo_warning.setStyleSheet("color: #FF9800;")
            layout.addWidget(undo_warning)
        
        # Trust checkbox (for medium+ risk)
        if self._command_info.risk_level in [CommandRisk.MEDIUM, CommandRisk.HIGH]:
            self._trust_checkbox = QCheckBox("Trust this command and don't ask again for similar commands")
            self._trust_checkbox.setStyleSheet("color: #858585;")
            layout.addWidget(self._trust_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self._btn_deny = QPushButton("❌ Cancel")
        self._btn_deny.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
            }
        """)
        self._btn_deny.clicked.connect(self._on_deny)
        button_layout.addWidget(self._btn_deny)
        
        self._btn_allow = QPushButton("✓ Execute")
        self._btn_allow.setStyleSheet(f"""
            QPushButton {{
                background-color: {risk_color};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {risk_color}CC;
            }}
        """)
        self._btn_allow.clicked.connect(self._on_allow)
        button_layout.addWidget(self._btn_allow)
        
        layout.addLayout(button_layout)
        
    def _on_allow(self):
        """Handle allow button click."""
        if hasattr(self, '_trust_checkbox'):
            self._trust_checked = self._trust_checkbox.isChecked()
        
        self.command_allowed.emit(self._command_info.command)
        self.accept()
        
    def _on_deny(self):
        """Handle deny button click."""
        self.command_denied.emit(self._command_info.command)
        self.reject()
        
    def is_trusted(self) -> bool:
        """Check if user chose to trust this command."""
        return self._trust_checked


class CommandSafetyChecker:
    """Checks command safety and determines risk level."""
    
    # Dangerous commands that require confirmation
    DANGEROUS_PATTERNS = {
        # File operations
        'rm': CommandRisk.HIGH,
        'del': CommandRisk.HIGH,
        'rmdir': CommandRisk.HIGH,
        'rd': CommandRisk.HIGH,
        'format': CommandRisk.CRITICAL,
        'mkfs': CommandRisk.CRITICAL,
        
        # System operations
        'shutdown': CommandRisk.HIGH,
        'reboot': CommandRisk.HIGH,
        'restart': CommandRisk.HIGH,
        'kill': CommandRisk.MEDIUM,
        'taskkill': CommandRisk.MEDIUM,
        
        # Network operations
        'curl': CommandRisk.MEDIUM,
        'wget': CommandRisk.MEDIUM,
        ' Invoke-WebRequest': CommandRisk.MEDIUM,
        
        # Permission changes
        'chmod': CommandRisk.MEDIUM,
        'chown': CommandRisk.MEDIUM,
        'cacls': CommandRisk.MEDIUM,
        'icacls': CommandRisk.MEDIUM,
        
        # Registry (Windows)
        'reg': CommandRisk.HIGH,
        'regedit': CommandRisk.CRITICAL,
        
        # Package managers (can modify system)
        'pip install': CommandRisk.LOW,
        'npm install': CommandRisk.LOW,
        'npm uninstall': CommandRisk.MEDIUM,
        'mvn': CommandRisk.LOW,
        'gradle': CommandRisk.LOW,
    }
    
    # Patterns that indicate destructive operations
    DESTRUCTIVE_PATTERNS = [
        '-rf', '/f', '/s', '/q',  # Force/recursive delete
        '> /dev/null', '2>&1',     # Output redirection
        '| sh', '| bash',          # Piping to shell
        'curl.*|.*sh',             # curl pipe to shell
    ]
    
    @classmethod
    def check_command(cls, command: str) -> CommandInfo:
        """Check a command and return its risk assessment.
        
        Args:
            command: The command to check
            
        Returns:
            CommandInfo with risk assessment
        """
        command_lower = command.lower().strip()
        
        # Default: low risk
        risk_level = CommandRisk.LOW
        description = ""
        affected_files = []
        
        # Check for dangerous patterns
        for pattern, risk in cls.DANGEROUS_PATTERNS.items():
            if pattern.lower() in command_lower:
                if risk.value > risk_level.value:
                    risk_level = risk
                    description = f"Command contains potentially dangerous operation: {pattern}"
        
        # Check for destructive patterns
        for pattern in cls.DESTRUCTIVE_PATTERNS:
            import re
            if re.search(pattern, command, re.IGNORECASE):
                risk_level = CommandRisk.HIGH
                description = "Command contains potentially destructive pattern"
        
        # Special checks for file deletion
        if any(cmd in command_lower for cmd in ['rm', 'del', 'rmdir']):
            # Check if deleting specific files or wildcards
            if '*' in command or '?' in command:
                risk_level = CommandRisk.HIGH
                description = "Command uses wildcards which may affect multiple files"
            
            # Try to extract affected files
            affected_files = cls._extract_files_from_command(command)
        
        # Check for system-wide operations
        if any(path in command for path in ['/usr/', 'C:\\Windows', 'C:\\Program Files', '/etc/', '/sys/']):
            risk_level = CommandRisk.CRITICAL
            description = "Command targets system directories"
        
        return CommandInfo(
            command=command,
            description=description,
            risk_level=risk_level,
            affected_files=affected_files,
            can_undo=False
        )
    
    @classmethod
    def _extract_files_from_command(cls, command: str) -> List[str]:
        """Try to extract file paths from a command."""
        files = []
        parts = command.split()
        
        for part in parts:
            # Skip flags
            if part.startswith('-') or part.startswith('/'):
                continue
            
            # Check if it looks like a file path
            if '/' in part or '\\' in part or '.' in part:
                from pathlib import Path
                try:
                    path = Path(part)
                    if path.exists():
                        files.append(str(path))
                except:
                    pass
        
        return files
    
    @classmethod
    def should_confirm(cls, command: str) -> bool:
        """Check if a command should require confirmation.
        
        Args:
            command: The command to check
            
        Returns:
            True if confirmation is required
        """
        info = cls.check_command(command)
        return info.risk_level in [CommandRisk.MEDIUM, CommandRisk.HIGH, CommandRisk.CRITICAL]


class TrustedCommandsManager:
    """Manages trusted commands that don't require confirmation."""
    
    def __init__(self):
        self._trusted_patterns: Set[str] = set()
        self._trusted_commands: Set[str] = set()
        
    def add_trusted_pattern(self, pattern: str):
        """Add a trusted command pattern."""
        self._trusted_patterns.add(pattern.lower())
        
    def add_trusted_command(self, command: str):
        """Add a specific trusted command."""
        self._trusted_commands.add(command.lower())
        
    def is_trusted(self, command: str) -> bool:
        """Check if a command is trusted."""
        command_lower = command.lower()
        
        # Check exact match
        if command_lower in self._trusted_commands:
            return True
        
        # Check pattern match
        for pattern in self._trusted_patterns:
            if pattern in command_lower:
                return True
        
        return False
    
    def remove_trusted(self, command: str):
        """Remove a command from trusted list."""
        self._trusted_commands.discard(command.lower())
        
    def clear(self):
        """Clear all trusted commands."""
        self._trusted_patterns.clear()
        self._trusted_commands.clear()
