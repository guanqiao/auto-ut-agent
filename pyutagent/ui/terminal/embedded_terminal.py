"""Embedded terminal widget for running shell commands."""

import logging
import subprocess
import os
import sys
from typing import Optional, List, Callable, Dict
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit,
    QPushButton, QLabel, QFrame, QComboBox, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QProcess, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QAction

logger = logging.getLogger(__name__)


class TerminalWidget(QFrame):
    """Terminal widget for executing shell commands.
    
    Features:
    - Execute shell commands
    - Show command output in real-time
    - Support for different shells (PowerShell, CMD, Bash)
    - Command history
    - Copy/Paste support
    """
    
    # Signals
    command_executed = pyqtSignal(str, int)  # command, exit_code
    command_started = pyqtSignal(str)  # command
    command_finished = pyqtSignal(str, int)  # command, exit_code
    
    def __init__(self, parent: Optional[QWidget] = None, cwd: Optional[str] = None):
        super().__init__(parent)
        self._cwd = cwd or os.getcwd()
        self._process: Optional[QProcess] = None
        self._command_history: List[str] = []
        self._history_index = -1
        self._current_command = ""
        
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
        
        menu.addSeparator()
        
        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.clear)
        menu.addAction(clear_action)
        
        menu.exec(self._output.mapToGlobal(position))
        
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


class EmbeddedTerminal(QWidget):
    """Embedded terminal container with tabs support."""
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the embedded terminal UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Single terminal for now (can be extended to tabs)
        self._terminal = TerminalWidget()
        layout.addWidget(self._terminal)
        
    def get_terminal(self) -> TerminalWidget:
        """Get the terminal widget."""
        return self._terminal
        
    def execute_command(self, command: str) -> bool:
        """Execute a command in the terminal."""
        return self._terminal.execute_command(command)
        
    def set_working_directory(self, cwd: str):
        """Set the working directory."""
        self._terminal.set_working_directory(cwd)
