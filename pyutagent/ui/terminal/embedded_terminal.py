"""Embedded terminal widget for running shell commands."""

import logging
import subprocess
import os
import sys
import re
from typing import Optional, List, Callable, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QFrame, QComboBox, QMenu, QDialog,
    QPlainTextEdit, QSplitter, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QProcess, QTimer, QThread
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QAction, QTextCursor

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can be detected in terminal output."""
    PYTHON_SYNTAX = "python_syntax"
    PYTHON_RUNTIME = "python_runtime"
    PYTHON_IMPORT = "python_import"
    JAVA_COMPILE = "java_compile"
    JAVA_RUNTIME = "java_runtime"
    MAVEN_BUILD = "maven_build"
    GRADLE_BUILD = "gradle_build"
    NPM_ERROR = "npm_error"
    GENERIC_ERROR = "generic_error"


@dataclass
class TerminalError:
    """Represents a detected terminal error."""
    error_type: ErrorType
    message: str
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    raw_output: str = ""
    start_pos: int = 0
    end_pos: int = 0


class ErrorPattern:
    """Pattern for detecting errors in terminal output."""
    
    PATTERNS = {
        ErrorType.PYTHON_SYNTAX: [
            (r'File "([^"]+)", line (\d+)', 'file_path', 'line_number'),
            (r'SyntaxError: (.+)', 'message'),
            (r'IndentationError: (.+)', 'message'),
        ],
        ErrorType.PYTHON_IMPORT: [
            (r'ModuleNotFoundError: No module named [\'"]([^\'"]+)[\'"]', 'module'),
            (r'ImportError: (.+)', 'message'),
        ],
        ErrorType.PYTHON_RUNTIME: [
            (r'Traceback \(most recent call last\):', 'traceback_start'),
            (r'File "([^"]+)", line (\d+)', 'file_path', 'line_number'),
        ],
        ErrorType.JAVA_COMPILE: [
            (r'([^\s]+\.java):(\d+): error: (.+)', 'file_path', 'line_number', 'message'),
        ],
        ErrorType.JAVA_RUNTIME: [
            (r'Exception in thread "[^"]+" (.+)', 'exception'),
            (r'([^\s]+Exception): (.+)', 'error_type', 'message'),
            (r'at ([^\(]+)\(([^:]+):(\d+)\)', 'method', 'file_path', 'line_number'),
        ],
        ErrorType.MAVEN_BUILD: [
            (r'\[ERROR\] (.+)', 'message'),
            (r'BUILD FAILURE', 'build_failure'),
        ],
        ErrorType.GRADLE_BUILD: [
            (r'FAILURE: Build failed', 'build_failure'),
            (r'\* What went wrong:\s*\n(.+)', 'message'),
        ],
        ErrorType.NPM_ERROR: [
            (r'npm ERR! (.+)', 'message'),
            (r'error (.+)', 'message'),
        ],
        ErrorType.GENERIC_ERROR: [
            (r'(?i)^error:\s*(.+)', 'message'),
            (r'(?i)^fatal:\s*(.+)', 'message'),
        ],
    }
    
    @classmethod
    def detect_errors(cls, output: str) -> List[TerminalError]:
        """Detect errors in terminal output."""
        errors = []
        lines = output.split('\n')
        
        for error_type, patterns in cls.PATTERNS.items():
            for i, line in enumerate(lines):
                for pattern_info in patterns:
                    pattern = pattern_info[0]
                    match = re.search(pattern, line)
                    if match:
                        error = cls._create_error(
                            error_type, match, lines, i, output
                        )
                        if error and not any(e.raw_output == error.raw_output for e in errors):
                            errors.append(error)
        
        return errors
    
    @classmethod
    def _create_error(cls, error_type: ErrorType, match: re.Match, 
                      lines: List[str], line_idx: int, full_output: str) -> Optional[TerminalError]:
        """Create a TerminalError from regex match."""
        message = ""
        file_path = None
        line_number = None
        
        # Extract information based on pattern
        groups = match.groups()
        if groups:
            message = groups[-1] if groups else "Unknown error"
            
            # Try to extract file path and line number
            for i, group in enumerate(groups):
                if group and ('/' in group or '\\' in group or group.endswith('.py') or 
                             group.endswith('.java') or group.endswith('.js')):
                    file_path = group
                elif group and group.isdigit():
                    line_number = int(group)
        
        # Get surrounding context
        start_idx = max(0, line_idx - 2)
        end_idx = min(len(lines), line_idx + 3)
        raw_output = '\n'.join(lines[start_idx:end_idx])
        
        # Calculate positions in full output
        start_pos = sum(len(l) + 1 for l in lines[:start_idx])
        end_pos = start_pos + len(raw_output)
        
        return TerminalError(
            error_type=error_type,
            message=message or match.group(0),
            line_number=line_number,
            file_path=file_path,
            raw_output=raw_output,
            start_pos=start_pos,
            end_pos=end_pos
        )


class TerminalWidget(QFrame):
    """Terminal widget for executing shell commands.
    
    Features:
    - Execute shell commands
    - Show command output in real-time
    - Support for different shells (PowerShell, CMD, Bash)
    - Command history
    - Copy/Paste support
    - Error detection and AI fix suggestions
    """
    
    # Signals
    command_executed = pyqtSignal(str, int)  # command, exit_code
    command_started = pyqtSignal(str)  # command
    command_finished = pyqtSignal(str, int)  # command, exit_code
    error_detected = pyqtSignal(list)  # List[TerminalError]
    ask_ai_requested = pyqtSignal(str)  # error_text
    
    def __init__(self, parent: Optional[QWidget] = None, cwd: Optional[str] = None):
        super().__init__(parent)
        self._cwd = cwd or os.getcwd()
        self._process: Optional[QProcess] = None
        self._command_history: List[str] = []
        self._history_index = -1
        self._current_command = ""
        self._detected_errors: List[TerminalError] = []
        self._error_highlights: List[Tuple[int, int]] = []  # (start, end) positions
        
        self.setup_ui()
        self._detect_shell()
        
    def setup_ui(self):
        """Setup the terminal UI."""
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            TerminalWidget {
                background-color: #1E1E1E;
                border: 1px solid #3C3C3C;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        header.setFixedHeight(32)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        # Shell selector
        self._shell_label = QLabel("Shell:")
        self._shell_label.setStyleSheet("color: #CCCCCC;")
        header_layout.addWidget(self._shell_label)
        
        self._shell_combo = QComboBox()
        self._shell_combo.setFixedWidth(120)
        self._shell_combo.setStyleSheet("""
            QComboBox {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 2px 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        header_layout.addWidget(self._shell_combo)
        
        # Current directory
        header_layout.addSpacing(16)
        
        self._cwd_label = QLabel(f"📁 {self._cwd}")
        self._cwd_label.setStyleSheet("color: #858585; font-size: 11px;")
        header_layout.addWidget(self._cwd_label)
        
        header_layout.addStretch()
        
        # Clear button
        self._btn_clear = QPushButton("🗑️")
        self._btn_clear.setFixedSize(24, 24)
        self._btn_clear.setToolTip("Clear terminal")
        self._btn_clear.clicked.connect(self.clear)
        header_layout.addWidget(self._btn_clear)
        
        layout.addWidget(header)
        
        # Output area
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont("Consolas", 10))
        self._output.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
                padding: 8px;
            }
        """)
        self._output.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._output.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self._output, stretch=1)
        
        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #252526;
                border-top: 1px solid #3C3C3C;
            }
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 4, 8, 4)
        input_layout.setSpacing(8)
        
        # Prompt
        self._prompt_label = QLabel("$")
        self._prompt_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        input_layout.addWidget(self._prompt_label)
        
        # Command input
        self._input = QLineEdit()
        self._input.setFont(QFont("Consolas", 10))
        self._input.setStyleSheet("""
            QLineEdit {
                background-color: #3C3C3C;
                color: #D4D4D4;
                border: 1px solid #555;
                padding: 4px 8px;
                border-radius: 3px;
            }
        """)
        self._input.returnPressed.connect(self._execute_command)
        self._input.setPlaceholderText("Type command and press Enter...")
        input_layout.addWidget(self._input, stretch=1)
        
        # Execute button
        self._btn_execute = QPushButton("▶")
        self._btn_execute.setFixedSize(28, 28)
        self._btn_execute.setStyleSheet("""
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
        """)
        self._btn_execute.clicked.connect(self._execute_command)
        input_layout.addWidget(self._btn_execute)
        
        # Stop button
        self._btn_stop = QPushButton("⏹")
        self._btn_stop.setFixedSize(28, 28)
        self._btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_command)
        input_layout.addWidget(self._btn_stop)
        
        layout.addWidget(input_frame)
        
        # Print welcome message
        self._print_welcome()
        
    def _detect_shell(self):
        """Detect available shells."""
        shells = []
        
        if sys.platform == 'win32':
            # Windows
            if self._check_command('powershell'):
                shells.append(('PowerShell', 'powershell'))
            if self._check_command('cmd'):
                shells.append(('CMD', 'cmd'))
            if self._check_command('pwsh'):
                shells.insert(0, ('PowerShell Core', 'pwsh'))
        else:
            # Unix-like
            if self._check_command('bash'):
                shells.append(('Bash', 'bash'))
            if self._check_command('zsh'):
                shells.append(('Zsh', 'zsh'))
            if self._check_command('fish'):
                shells.append(('Fish', 'fish'))
        
        self._shell_combo.clear()
        for name, cmd in shells:
            self._shell_combo.addItem(name, cmd)
        
        if shells:
            self._shell = shells[0][1]
        else:
            self._shell = 'cmd' if sys.platform == 'win32' else 'bash'
            
    def _check_command(self, command: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([command, '--version'], 
                          capture_output=True, 
                          timeout=2)
            return True
        except:
            return False
            
    def _print_welcome(self):
        """Print welcome message."""
        welcome = f"""
╔══════════════════════════════════════════════════════════════╗
║  PyUT Agent Terminal                                          ║
║  Current directory: {self._cwd:<45}║
║  Shell: {self._shell_combo.currentText():<51}║
╚══════════════════════════════════════════════════════════════╝

Type 'help' for available commands, or enter any shell command.

"""
        self._output.append(welcome.strip())
        
    def _execute_command(self):
        """Execute the command in the input field."""
        command = self._input.text().strip()
        if not command:
            return
        
        # Add to history
        if command not in self._command_history:
            self._command_history.append(command)
        self._history_index = len(self._command_history)
        
        # Print command
        self._output.append(f"\n$ {command}\n")
        
        # Handle built-in commands
        if self._handle_builtin(command):
            self._input.clear()
            return
        
        # Execute shell command
        self._current_command = command
        self._run_shell_command(command)
        self._input.clear()
        
    def _handle_builtin(self, command: str) -> bool:
        """Handle built-in commands.
        
        Returns:
            True if handled
        """
        cmd_lower = command.lower().strip()
        
        if cmd_lower == 'help':
            self._print_help()
            return True
        elif cmd_lower == 'clear':
            self.clear()
            return True
        elif cmd_lower.startswith('cd '):
            path = command[3:].strip()
            self._change_directory(path)
            return True
        elif cmd_lower == 'pwd':
            self._output.append(self._cwd)
            return True
        elif cmd_lower == 'ls' or cmd_lower == 'dir':
            self._list_directory()
            return True
        elif cmd_lower == 'exit':
            self._output.append("Use the UI to close the terminal.")
            return True
            
        return False
        
    def _print_help(self):
        """Print help message."""
        help_text = """
Built-in Commands:
  help          Show this help message
  clear         Clear the terminal
  cd <path>     Change directory
  pwd           Print working directory
  ls/dir        List directory contents
  exit          Close terminal (use UI button)

Shell Commands:
  Any shell command can be executed directly.
  Examples: git status, npm install, mvn test, etc.

Keyboard Shortcuts:
  Up/Down       Navigate command history
  Ctrl+C        Copy selected text
  Ctrl+V        Paste text
"""
        self._output.append(help_text)
        
    def _change_directory(self, path: str):
        """Change current directory."""
        try:
            new_path = Path(self._cwd) / path
            new_path = new_path.resolve()
            
            if new_path.exists() and new_path.is_dir():
                self._cwd = str(new_path)
                self._cwd_label.setText(f"📁 {self._cwd}")
                self._output.append(f"Changed to: {self._cwd}")
            else:
                self._output.append(f"Error: Directory not found: {path}")
        except Exception as e:
            self._output.append(f"Error: {e}")
            
    def _list_directory(self):
        """List directory contents."""
        try:
            items = sorted(Path(self._cwd).iterdir(), 
                          key=lambda x: (not x.is_dir(), x.name.lower()))
            
            lines = [f"\nContents of {self._cwd}:\n"]
            for item in items:
                if item.is_dir():
                    lines.append(f"  📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    size_str = self._format_size(size)
                    lines.append(f"  📄 {item.name:<40} {size_str:>10}")
            
            self._output.append('\n'.join(lines))
        except Exception as e:
            self._output.append(f"Error: {e}")
            
    def _format_size(self, size: int) -> str:
        """Format file size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
        
    def _run_shell_command(self, command: str):
        """Run a shell command."""
        self.command_started.emit(command)
        
        # Create process
        self._process = QProcess(self)
        self._process.setWorkingDirectory(self._cwd)
        
        # Connect signals
        self._process.readyReadStandardOutput.connect(self._on_stdout)
        self._process.readyReadStandardError.connect(self._on_stderr)
        self._process.finished.connect(self._on_finished)
        
        # Start process
        shell = self._shell_combo.currentData()
        
        if sys.platform == 'win32':
            if shell == 'cmd':
                self._process.start('cmd', ['/c', command])
            else:
                self._process.start(shell, ['-Command', command])
        else:
            self._process.start(shell, ['-c', command])
        
        # Update UI
        self._btn_execute.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._input.setEnabled(False)
        
    def _on_stdout(self):
        """Handle stdout data."""
        if self._process:
            data = self._process.readAllStandardOutput().data().decode('utf-8', errors='replace')
            self._output.insertPlainText(data)
            self._scroll_to_bottom()
            
    def _on_stderr(self):
        """Handle stderr data."""
        if self._process:
            data = self._process.readAllStandardError().data().decode('utf-8', errors='replace')
            # Format error in red
            cursor = self._output.textCursor()
            format = QTextCharFormat()
            format.setForeground(QColor('#F44336'))
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.insertText(data, format)
            self._scroll_to_bottom()
            
            # Detect errors in stderr
            self._detect_errors_in_output(data)
            
    def _detect_errors_in_output(self, output: str):
        """Detect errors in terminal output and highlight them."""
        errors = ErrorPattern.detect_errors(output)
        if errors:
            self._detected_errors.extend(errors)
            self._highlight_errors()
            self.error_detected.emit(self._detected_errors)
            
    def _highlight_errors(self):
        """Highlight error lines in the output."""
        cursor = self._output.textCursor()
        full_text = self._output.toPlainText()
        
        for error in self._detected_errors:
            # Find the error text in the output
            if error.raw_output in full_text:
                start_pos = full_text.find(error.raw_output)
                if start_pos >= 0:
                    end_pos = start_pos + len(error.raw_output)
                    self._error_highlights.append((start_pos, end_pos))
                    
                    # Highlight the error
                    cursor.setPosition(start_pos)
                    cursor.setPosition(end_pos, QTextCursor.MoveMode.KeepAnchor)
                    
                    format = QTextCharFormat()
                    format.setBackground(QColor('#F44336'))  # Red background
                    format.setForeground(QColor('#FFFFFF'))  # White text
                    cursor.mergeCharFormat(format)
                    
    def get_detected_errors(self) -> List[TerminalError]:
        """Get all detected errors."""
        return self._detected_errors.copy()
        
    def clear_errors(self):
        """Clear all detected errors and highlights."""
        self._detected_errors.clear()
        self._error_highlights.clear()
        # Re-render output without highlights
        current_text = self._output.toPlainText()
        self._output.clear()
        self._output.setPlainText(current_text)
            
    def _on_finished(self, exit_code: int, exit_status):
        """Handle process finish."""
        self._btn_execute.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._input.setEnabled(True)
        self._input.setFocus()
        
        if exit_code != 0:
            self._output.append(f"\n[Exit code: {exit_code}]")
        
        self.command_finished.emit(self._current_command, exit_code)
        self.command_executed.emit(self._current_command, exit_code)
        
        self._process = None
        self._current_command = ""
        
    def _stop_command(self):
        """Stop the running command."""
        if self._process and self._process.state() == QProcess.ProcessState.Running:
            self._process.terminate()
            self._output.append("\n[Command terminated]")
            
    def _scroll_to_bottom(self):
        """Scroll output to bottom."""
        scrollbar = self._output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def _show_context_menu(self, position):
        """Show context menu."""
        menu = QMenu(self)
        
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self._output.copy)
        menu.addAction(copy_action)
        
        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self._output.selectAll)
        menu.addAction(select_all_action)
        
        # Add Ask AI option if errors detected
        if self._detected_errors:
            menu.addSeparator()
            ask_ai_action = QAction("🤖 Ask AI to Fix", self)
            ask_ai_action.triggered.connect(self._on_ask_ai_clicked)
            menu.addAction(ask_ai_action)
        
        menu.addSeparator()
        
        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.clear)
        menu.addAction(clear_action)
        
        menu.exec(self._output.mapToGlobal(position))
        
    def _on_ask_ai_clicked(self):
        """Handle Ask AI button click."""
        if self._detected_errors:
            error_text = self._format_errors_for_ai()
            self.ask_ai_requested.emit(error_text)
            
    def _format_errors_for_ai(self) -> str:
        """Format detected errors for AI analysis."""
        lines = ["Detected Errors:"]
        for i, error in enumerate(self._detected_errors, 1):
            lines.append(f"\n{i}. Type: {error.error_type.value}")
            lines.append(f"   Message: {error.message}")
            if error.file_path:
                lines.append(f"   File: {error.file_path}")
            if error.line_number:
                lines.append(f"   Line: {error.line_number}")
            lines.append(f"   Output:\n{error.raw_output}")
        return '\n'.join(lines)
        
    def clear(self):
        """Clear the terminal output."""
        self._output.clear()
        self._print_welcome()
        
    def set_working_directory(self, cwd: str):
        """Set the working directory."""
        self._cwd = cwd
        self._cwd_label.setText(f"📁 {self._cwd}")
        
    def execute_command(self, command: str) -> bool:
        """Execute a command programmatically.
        
        Args:
            command: Command to execute
            
        Returns:
            True if command started
        """
        if self._process and self._process.state() == QProcess.ProcessState.Running:
            return False
        
        self._input.setText(command)
        self._execute_command()
        return True
        
    def is_running(self) -> bool:
        """Check if a command is running."""
        return self._process is not None and self._process.state() == QProcess.ProcessState.Running


class AIFixDialog(QDialog):
    """Dialog for displaying AI fix suggestions and applying them.
    
    Features:
    - Display error details
    - Show AI analysis
    - Display diff preview
    - One-click apply fix
    """
    
    fix_applied = pyqtSignal(str, str)  # file_path, fixed_content
    
    def __init__(self, error_text: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._error_text = error_text
        self._ai_response = ""
        self._original_content = ""
        self._fixed_content = ""
        self._target_file = ""
        
        self.setWindowTitle("🤖 AI Fix Suggestion")
        self.setMinimumSize(800, 600)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Error section
        error_group = QFrame()
        error_group.setStyleSheet("""
            QFrame {
                background-color: #FFEBEE;
                border: 1px solid #EF9A9A;
                border-radius: 6px;
            }
        """)
        error_layout = QVBoxLayout(error_group)
        
        error_label = QLabel("❌ Detected Error:")
        error_label.setStyleSheet("font-weight: bold; color: #C62828;")
        error_layout.addWidget(error_label)
        
        self._error_display = QPlainTextEdit()
        self._error_display.setPlainText(self._error_text)
        self._error_display.setReadOnly(True)
        self._error_display.setMaximumHeight(120)
        self._error_display.setStyleSheet("""
            QPlainTextEdit {
                background-color: #FFCDD2;
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        error_layout.addWidget(self._error_display)
        layout.addWidget(error_group)
        
        # AI Analysis section
        analysis_group = QFrame()
        analysis_group.setStyleSheet("""
            QFrame {
                background-color: #E3F2FD;
                border: 1px solid #90CAF9;
                border-radius: 6px;
            }
        """)
        analysis_layout = QVBoxLayout(analysis_group)
        
        analysis_header = QHBoxLayout()
        self._analysis_label = QLabel("🤖 AI Analysis:")
        self._analysis_label.setStyleSheet("font-weight: bold; color: #1565C0;")
        analysis_header.addWidget(self._analysis_label)
        
        self._loading_label = QLabel("Analyzing...")
        self._loading_label.setStyleSheet("color: #666;")
        self._loading_label.hide()
        analysis_header.addWidget(self._loading_label)
        analysis_header.addStretch()
        
        analysis_layout.addLayout(analysis_header)
        
        self._analysis_display = QPlainTextEdit()
        self._analysis_display.setReadOnly(True)
        self._analysis_display.setPlaceholderText("AI analysis will appear here...")
        self._analysis_display.setStyleSheet("""
            QPlainTextEdit {
                background-color: #BBDEFB;
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        analysis_layout.addWidget(self._analysis_display)
        layout.addWidget(analysis_group)
        
        # Diff section
        diff_group = QFrame()
        diff_group.setStyleSheet("""
            QFrame {
                background-color: #F3E5F5;
                border: 1px solid #CE93D8;
                border-radius: 6px;
            }
        """)
        diff_layout = QVBoxLayout(diff_group)
        
        diff_label = QLabel("📝 Proposed Fix (Diff):")
        diff_label.setStyleSheet("font-weight: bold; color: #6A1B9A;")
        diff_layout.addWidget(diff_label)
        
        self._diff_display = QPlainTextEdit()
        self._diff_display.setReadOnly(True)
        self._diff_display.setPlaceholderText("Diff preview will appear here...")
        self._diff_display.setStyleSheet("""
            QPlainTextEdit {
                background-color: #E1BEE7;
                border: none;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        diff_layout.addWidget(self._diff_display)
        layout.addWidget(diff_group, stretch=1)
        
        # File info
        self._file_info_label = QLabel("Target file: Not determined")
        self._file_info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._file_info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self._refresh_btn = QPushButton("🔄 Refresh Analysis")
        self._refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self._refresh_btn.clicked.connect(self._request_ai_analysis)
        button_layout.addWidget(self._refresh_btn)
        
        button_layout.addStretch()
        
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)
        
        self._apply_btn = QPushButton("✅ Apply Fix")
        self._apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 24px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #A5D6A7;
            }
        """)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply_fix)
        button_layout.addWidget(self._apply_btn)
        
        layout.addLayout(button_layout)
        
    def set_ai_response(self, response: str, original_content: str = "", 
                        fixed_content: str = "", target_file: str = ""):
        """Set the AI response and update UI."""
        self._ai_response = response
        self._original_content = original_content
        self._fixed_content = fixed_content
        self._target_file = target_file
        
        self._analysis_display.setPlainText(response)
        self._loading_label.hide()
        
        if target_file:
            self._file_info_label.setText(f"Target file: {target_file}")
        
        if original_content and fixed_content:
            diff = self._generate_diff(original_content, fixed_content)
            self._diff_display.setPlainText(diff)
            self._apply_btn.setEnabled(True)
        else:
            # Try to parse fix from AI response
            self._try_parse_fix_from_response(response)
            
    def _try_parse_fix_from_response(self, response: str):
        """Try to parse code fix from AI response."""
        # Look for code blocks in markdown
        import re
        
        # Pattern for ```language ... ``` code blocks
        code_pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            # Use the last code block as the fix
            self._fixed_content = matches[-1].strip()
            
            # Try to extract file path from response
            file_pattern = r'(?:file|path)[:\s]+["\']?([^"\'\n]+)["\']?'
            file_matches = re.findall(file_pattern, response, re.IGNORECASE)
            if file_matches:
                self._target_file = file_matches[-1]
                self._file_info_label.setText(f"Target file: {self._target_file}")
                
            # Generate a simple diff view
            self._diff_display.setPlainText(
                f"Suggested fix:\n{'='*50}\n{self._fixed_content}"
            )
            self._apply_btn.setEnabled(True)
            
    def _generate_diff(self, original: str, modified: str) -> str:
        """Generate unified diff between original and modified content."""
        import difflib
        
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines, 
            modified_lines,
            fromfile='original',
            tofile='fixed',
            lineterm=''
        )
        
        return ''.join(diff)
        
    def _request_ai_analysis(self):
        """Request AI analysis (placeholder for actual implementation)."""
        self._loading_label.show()
        self._analysis_display.setPlainText("Requesting AI analysis...")
        # This would be connected to actual AI service
        
    def _apply_fix(self):
        """Apply the fix."""
        if self._target_file and self._fixed_content:
            self.fix_applied.emit(self._target_file, self._fixed_content)
            self.accept()
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Cannot Apply Fix", 
                              "No target file or fix content available.")
                              
    def get_fix_info(self) -> Tuple[str, str]:
        """Get the fix information.
        
        Returns:
            Tuple of (target_file, fixed_content)
        """
        return self._target_file, self._fixed_content


class AIFixWorker(QThread):
    """Worker thread for AI fix requests."""
    
    analysis_ready = pyqtSignal(str, str, str, str)  # response, original, fixed, target_file
    analysis_error = pyqtSignal(str)
    
    def __init__(self, error_text: str, file_content: str = "", 
                 file_path: str = "", parent=None):
        super().__init__(parent)
        self._error_text = error_text
        self._file_content = file_content
        self._file_path = file_path
        
    def run(self):
        """Run AI analysis in background thread."""
        try:
            # This is a placeholder - actual implementation would call LLM
            response = self._call_llm_for_fix()
            self.analysis_ready.emit(response, self._file_content, "", self._file_path)
        except Exception as e:
            self.analysis_error.emit(str(e))
            
    def _call_llm_for_fix(self) -> str:
        """Call LLM to get fix suggestion.
        
        This is a placeholder. Actual implementation would:
        1. Build a prompt with error context
        2. Call the LLM API
        3. Parse the response
        """
        prompt = f"""Analyze the following error and suggest a fix:

{self._error_text}

Please provide:
1. Analysis of the error cause
2. The fixed code
3. Explanation of the changes
"""
        # Placeholder response
        return f"""## Error Analysis

The error indicates an issue that needs to be fixed.

## Suggested Fix

```python
# Fixed code would go here
pass
```

## Explanation

This fix addresses the error by correcting the identified issue.
"""


class EmbeddedTerminal(QWidget):
    """Embedded terminal container with AI integration."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._ai_dialog: Optional[AIFixDialog] = None
        self._ai_worker: Optional[AIFixWorker] = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the embedded terminal UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Terminal widget
        self._terminal = TerminalWidget()
        self._terminal.error_detected.connect(self._on_errors_detected)
        self._terminal.ask_ai_requested.connect(self._on_ask_ai_requested)
        layout.addWidget(self._terminal)
        
        # Floating Ask AI button (shown when errors detected)
        self._ask_ai_btn = QPushButton("🤖 Ask AI", self)
        self._ask_ai_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        self._ask_ai_btn.setFixedHeight(36)
        self._ask_ai_btn.hide()
        self._ask_ai_btn.clicked.connect(self._show_ai_dialog)
        
    def resizeEvent(self, event):
        """Handle resize to position floating button."""
        super().resizeEvent(event)
        self._position_ask_ai_button()
        
    def _position_ask_ai_button(self):
        """Position the Ask AI button in the bottom-right corner."""
        btn_width = 100
        btn_height = 36
        margin = 16
        self._ask_ai_btn.setGeometry(
            self.width() - btn_width - margin,
            self.height() - btn_height - margin - 50,  # Above input area
            btn_width,
            btn_height
        )
        
    def _on_errors_detected(self, errors: List[TerminalError]):
        """Handle detected errors."""
        if errors:
            self._ask_ai_btn.show()
            self._position_ask_ai_button()
        else:
            self._ask_ai_btn.hide()
            
    def _on_ask_ai_requested(self, error_text: str):
        """Handle Ask AI request from terminal."""
        self._show_ai_dialog()
        
    def _show_ai_dialog(self):
        """Show AI fix dialog."""
        errors = self._terminal.get_detected_errors()
        if not errors:
            return
            
        error_text = self._terminal._format_errors_for_ai()
        
        self._ai_dialog = AIFixDialog(error_text, self)
        self._ai_dialog.fix_applied.connect(self._on_fix_applied)
        
        # Start AI analysis
        self._request_ai_analysis(error_text)
        
        self._ai_dialog.exec()
        
    def _request_ai_analysis(self, error_text: str):
        """Request AI analysis for the error."""
        # Try to read the file if available
        file_content = ""
        file_path = ""
        
        errors = self._terminal.get_detected_errors()
        if errors and errors[0].file_path:
            file_path = errors[0].file_path
            try:
                # Try to read the file
                full_path = Path(self._terminal._cwd) / file_path
                if full_path.exists():
                    with open(full_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                
        # Create and start worker
        self._ai_worker = AIFixWorker(error_text, file_content, file_path)
        self._ai_worker.analysis_ready.connect(self._on_analysis_ready)
        self._ai_worker.analysis_error.connect(self._on_analysis_error)
        self._ai_worker.start()
        
    def _on_analysis_ready(self, response: str, original: str, fixed: str, target_file: str):
        """Handle AI analysis completion."""
        if self._ai_dialog:
            self._ai_dialog.set_ai_response(response, original, fixed, target_file)
            
    def _on_analysis_error(self, error: str):
        """Handle AI analysis error."""
        logger.error(f"AI analysis error: {error}")
        if self._ai_dialog:
            self._ai_dialog.set_ai_response(f"Error getting AI analysis: {error}")
            
    def _on_fix_applied(self, file_path: str, fixed_content: str):
        """Handle fix applied."""
        logger.info(f"Fix applied to {file_path}")
        # Emit signal or notify parent
        
    def get_terminal(self) -> TerminalWidget:
        """Get the terminal widget."""
        return self._terminal
        
    def execute_command(self, command: str) -> bool:
        """Execute a command in the terminal."""
        return self._terminal.execute_command(command)
        
    def set_working_directory(self, cwd: str):
        """Set the working directory."""
        self._terminal.set_working_directory(cwd)
