"""CLI Output Formatter - Clean and concise output for CLI mode.

This module provides output formatting capabilities for CLI,
inspired by Claude Code's concise output style.
"""

import logging
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OutputVerbosity(Enum):
    """Output verbosity levels."""
    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"


class OutputFormat(Enum):
    """Output format types."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class OutputConfig:
    """Configuration for output formatting."""
    verbosity: OutputVerbosity = OutputVerbosity.NORMAL
    format: OutputFormat = OutputFormat.TEXT
    color: bool = True
    max_lines: int = 4
    show_timestamps: bool = False
    show_progress: bool = True


class OutputFormatter:
    """Formatter for CLI output with concise style."""
    
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    
    def __init__(self, config: Optional[OutputConfig] = None):
        """Initialize output formatter.
        
        Args:
            config: Output configuration
        """
        self.config = config or OutputConfig()
        self._color_enabled = self.config.color and self._supports_color()
    
    def _supports_color(self) -> bool:
        """Check if terminal supports color."""
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text."""
        if not self._color_enabled:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def format_result(
        self,
        success: bool,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a result message.
        
        Args:
            success: Whether operation succeeded
            message: Result message
            data: Optional additional data
            
        Returns:
            Formatted output string
        """
        if self.config.format == OutputFormat.JSON:
            return self._format_json(success, message, data)
        
        if self.config.verbosity == OutputVerbosity.QUIET:
            return "" if success else self._colorize(f"Error: {message}", "red")
        
        status = self._colorize("✓", "green") if success else self._colorize("✗", "red")
        output = f"{status} {message}"
        
        if data and self.config.verbosity in (OutputVerbosity.VERBOSE, OutputVerbosity.DEBUG):
            output += f"\n{self._format_data(data)}"
        
        return output
    
    def format_task_result(
        self,
        task_type: str,
        result: Dict[str, Any]
    ) -> str:
        """Format task execution result.
        
        Args:
            task_type: Type of task
            result: Task result
            
        Returns:
            Formatted output string
        """
        success = result.get("success", False)
        message = result.get("message", "Task completed")
        
        if self.config.verbosity == OutputVerbosity.QUIET:
            return ""
        
        if self.config.max_lines <= 1:
            return self.format_result(success, message)
        
        lines = [self.format_result(success, message)]
        
        if self.config.verbosity in (OutputVerbosity.VERBOSE, OutputVerbosity.DEBUG):
            if "artifacts" in result:
                lines.append(self._format_artifacts(result["artifacts"]))
            if "metrics" in result:
                lines.append(self._format_metrics(result["metrics"]))
        
        return "\n".join(lines[:self.config.max_lines])
    
    def format_progress(
        self,
        current: int,
        total: int,
        message: str = ""
    ) -> str:
        """Format progress indicator.
        
        Args:
            current: Current progress
            total: Total items
            message: Progress message
            
        Returns:
            Formatted progress string
        """
        if not self.config.show_progress:
            return ""
        
        percentage = (current / total * 100) if total > 0 else 0
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        progress_str = f"[{bar}] {current}/{total} ({percentage:.0f}%)"
        
        if message:
            progress_str += f" {message}"
        
        return self._colorize(progress_str, "cyan")
    
    def format_step(
        self,
        step_num: int,
        total_steps: int,
        description: str,
        status: str = "running"
    ) -> str:
        """Format a step indicator.
        
        Args:
            step_num: Current step number
            total_steps: Total steps
            description: Step description
            status: Step status (running, done, failed)
            
        Returns:
            Formatted step string
        """
        if self.config.verbosity == OutputVerbosity.QUIET:
            return ""
        
        if status == "running":
            icon = self._colorize("●", "yellow")
        elif status == "done":
            icon = self._colorize("✓", "green")
        elif status == "failed":
            icon = self._colorize("✗", "red")
        else:
            icon = "○"
        
        return f"{icon} Step {step_num}/{total_steps}: {description}"
    
    def format_error(
        self,
        error: str,
        suggestion: Optional[str] = None
    ) -> str:
        """Format an error message.
        
        Args:
            error: Error message
            suggestion: Optional suggestion
            
        Returns:
            Formatted error string
        """
        output = self._colorize(f"Error: {error}", "red")
        
        if suggestion and self.config.verbosity != OutputVerbosity.QUIET:
            output += f"\n{self._colorize(f'Suggestion: {suggestion}', 'yellow')}"
        
        return output
    
    def format_warning(self, message: str) -> str:
        """Format a warning message."""
        return self._colorize(f"Warning: {message}", "yellow")
    
    def format_info(self, message: str) -> str:
        """Format an info message."""
        if self.config.verbosity == OutputVerbosity.QUIET:
            return ""
        return self._colorize(message, "blue")
    
    def format_success(self, message: str) -> str:
        """Format a success message."""
        return self._colorize(f"✓ {message}", "green")
    
    def format_list(
        self,
        items: List[str],
        title: Optional[str] = None
    ) -> str:
        """Format a list of items.
        
        Args:
            items: List items
            title: Optional title
            
        Returns:
            Formatted list string
        """
        if not items:
            return ""
        
        lines = []
        
        if title:
            lines.append(self._colorize(title, "bold"))
        
        for item in items:
            lines.append(f"  • {item}")
        
        return "\n".join(lines)
    
    def format_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        title: Optional[str] = None
    ) -> str:
        """Format a table.
        
        Args:
            headers: Table headers
            rows: Table rows
            title: Optional title
            
        Returns:
            Formatted table string
        """
        if not rows:
            return ""
        
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        lines = []
        
        if title:
            lines.append(self._colorize(title, "bold"))
        
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        lines.append(self._colorize(header_line, "bold"))
        lines.append("-+-".join("-" * w for w in col_widths))
        
        for row in rows:
            line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_json(
        self,
        success: bool,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format as JSON."""
        import json
        
        output = {
            "success": success,
            "message": message,
        }
        
        if data:
            output["data"] = data
        
        return json.dumps(output, indent=2)
    
    def _format_data(self, data: Dict[str, Any]) -> str:
        """Format additional data."""
        lines = []
        
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                import json
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)
            
            lines.append(f"  {key}: {value_str}")
        
        return "\n".join(lines)
    
    def _format_artifacts(self, artifacts: Dict[str, Any]) -> str:
        """Format artifacts."""
        lines = [self._colorize("Artifacts:", "cyan")]
        
        for name, path in artifacts.items():
            lines.append(f"  {name}: {path}")
        
        return "\n".join(lines)
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics."""
        lines = [self._colorize("Metrics:", "cyan")]
        
        for name, value in metrics.items():
            lines.append(f"  {name}: {value}")
        
        return "\n".join(lines)


class ProgressIndicator:
    """Progress indicator for CLI."""
    
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def __init__(
        self,
        formatter: Optional[OutputFormatter] = None,
        enabled: bool = True
    ):
        """Initialize progress indicator.
        
        Args:
            formatter: Output formatter
            enabled: Whether progress is enabled
        """
        self.formatter = formatter or OutputFormatter()
        self.enabled = enabled
        self._frame_idx = 0
        self._last_message = ""
    
    def start(self, message: str = "Processing"):
        """Start progress indicator."""
        if not self.enabled:
            return
        
        self._last_message = message
        self._update()
    
    def update(self, message: str):
        """Update progress message."""
        if not self.enabled:
            return
        
        self._last_message = message
        self._update()
    
    def _update(self):
        """Update display."""
        frame = self.SPINNER_FRAMES[self._frame_idx % len(self.SPINNER_FRAMES)]
        self._frame_idx += 1
        
        output = f"{frame} {self._last_message}"
        
        if self.formatter._color_enabled:
            output = self.formatter._colorize(output, "cyan")
        
        print(f"\r{output}", end="", flush=True)
    
    def complete(self, message: str = "Done"):
        """Complete progress indicator."""
        if not self.enabled:
            return
        
        print(f"\r{self.formatter.format_success(message)}" + " " * 20)
    
    def fail(self, message: str = "Failed"):
        """Fail progress indicator."""
        if not self.enabled:
            return
        
        print(f"\r{self.formatter.format_error(message)}" + " " * 20)


def create_formatter(
    verbosity: str = "normal",
    format: str = "text",
    color: bool = True
) -> OutputFormatter:
    """Create output formatter.
    
    Args:
        verbosity: Verbosity level
        format: Output format
        color: Enable color output
        
    Returns:
        OutputFormatter instance
    """
    config = OutputConfig(
        verbosity=OutputVerbosity(verbosity),
        format=OutputFormat(format),
        color=color
    )
    return OutputFormatter(config)
