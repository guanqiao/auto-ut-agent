"""Logging System.

This module provides:
- Structured logging
- Log levels and formatting
- Log aggregation
- Performance logging
"""

import logging
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat(Enum):
    """Log formats."""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


@dataclass
class LogEntry:
    """A log entry."""
    timestamp: datetime
    level: str
    logger: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    stack_trace: Optional[str] = None


class LogFormatter(logging.Formatter):
    """Custom log formatter."""

    def __init__(self, log_format: LogFormat = LogFormat.TEXT):
        """Initialize formatter.

        Args:
            log_format: Log format
        """
        super().__init__()
        self.log_format = log_format

    def format(self, record: logging.LogRecord) -> str:
        """Format log record."""
        entry = LogEntry(
            timestamp=datetime.fromtimestamp(record.created),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            error=str(record.exc_info) if record.exc_info else None,
            stack_trace=record.exc_text
        )

        if self.log_format == LogFormat.JSON:
            return json.dumps({
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level,
                "logger": entry.logger,
                "message": entry.message,
                "error": entry.error
            }, default=str)

        return super().format(record)


class StructuredLogger:
    """Structured logger for the agent.

    Features:
    - Structured logging
    - Performance tracking
    - Error tracking
    - Log aggregation
    """

    def __init__(
        self,
        name: str,
        log_file: Optional[Path] = None,
        level: int = logging.INFO,
        log_format: LogFormat = LogFormat.TEXT
    ):
        """Initialize structured logger.

        Args:
            name: Logger name
            log_file: Log file path
            level: Log level
            log_format: Log format
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        formatter = LogFormatter(log_format)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self._log_entries: List[LogEntry] = []

    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self.logger.log(LogLevel.TRACE.value, message, extra=kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message."""
        if error:
            self.logger.exception(message, extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True
    ):
        """Log performance metrics.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
        """
        level = logging.INFO if success else logging.WARNING
        self.logger.log(
            level,
            f"Performance: {operation} - {duration_ms:.2f}ms",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "success": success,
                "type": "performance"
            }
        )

    def log_tool_execution(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None
    ):
        """Log tool execution.

        Args:
            tool_name: Tool name
            success: Whether execution succeeded
            duration_ms: Duration in milliseconds
            error: Error message if failed
        """
        level = logging.INFO if success else logging.WARNING
        self.logger.log(
            level,
            f"Tool: {tool_name} - {'OK' if success else 'FAILED'} ({duration_ms:.2f}ms)",
            extra={
                "tool_name": tool_name,
                "success": success,
                "duration_ms": duration_ms,
                "error": error,
                "type": "tool_execution"
            }
        )

    def log_agent_action(
        self,
        agent_name: str,
        action: str,
        state: str,
        iteration: int
    ):
        """Log agent action.

        Args:
            agent_name: Agent name
            action: Action taken
            state: Current state
            iteration: Current iteration
        """
        self.logger.info(
            f"Agent: {agent_name} - {action} (state: {state}, iter: {iteration})",
            extra={
                "agent_name": agent_name,
                "action": action,
                "state": state,
                "iteration": iteration,
                "type": "agent_action"
            }
        )

    def log_llm_call(
        self,
        model: str,
        tokens_used: int,
        duration_ms: float,
        success: bool
    ):
        """Log LLM API call.

        Args:
            model: Model name
            tokens_used: Number of tokens used
            duration_ms: Duration in milliseconds
            success: Whether call succeeded
        """
        level = logging.INFO if success else logging.WARNING
        self.logger.log(
            level,
            f"LLM: {model} - {tokens_used} tokens, {duration_ms:.2f}ms",
            extra={
                "model": model,
                "tokens_used": tokens_used,
                "duration_ms": duration_ms,
                "success": success,
                "type": "llm_call"
            }
        )


class LogAggregator:
    """Aggregates and analyzes logs.

    Features:
    - Log aggregation
    - Error tracking
    - Performance analysis
    - Log search
    """

    def __init__(self):
        """Initialize log aggregator."""
        self._entries: List[LogEntry] = []

    def add_entry(self, entry: LogEntry):
        """Add log entry.

        Args:
            entry: Log entry
        """
        self._entries.append(entry)

    def get_errors(self, limit: int = 100) -> List[LogEntry]:
        """Get error entries.

        Args:
            limit: Maximum entries

        Returns:
            Error entries
        """
        return [
            e for e in self._entries
            if e.level in ("ERROR", "CRITICAL")
        ][:limit]

    def get_performance_logs(self) -> List[LogEntry]:
        """Get performance log entries.

        Returns:
            Performance entries
        """
        return [
            e for e in self._entries
            if e.extra.get("type") == "performance"
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get log summary.

        Returns:
            Summary statistics
        """
        levels = {}
        for entry in self._entries:
            levels[entry.level] = levels.get(entry.level, 0) + 1

        return {
            "total": len(self._entries),
            "by_level": levels,
            "time_range": {
                "start": self._entries[0].timestamp if self._entries else None,
                "end": self._entries[-1].timestamp if self._entries else None
            }
        }


def create_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    json_format: bool = False
) -> StructuredLogger:
    """Create a structured logger.

    Args:
        name: Logger name
        log_dir: Log directory
        level: Log level
        json_format: Use JSON format

    Returns:
        StructuredLogger instance
    """
    log_format = LogFormat.JSON if json_format else LogFormat.TEXT

    log_file = None
    if log_dir:
        log_file = log_dir / f"{name}.log"

    return StructuredLogger(
        name=name,
        log_file=log_file,
        level=level,
        log_format=log_format
    )
