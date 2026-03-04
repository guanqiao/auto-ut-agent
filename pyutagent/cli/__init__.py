"""CLI module for PyUT Agent.

This module provides command-line interface for the PyUT Agent.

Usage:
    $ pyutagent generate --target com.example.MyClass
    $ pyutagent analyze --project /path/to/project
    $ pyutagent test --file TargetClass.java
"""

__version__ = "0.1.0"

from .main import main, cli
from .output_formatter import (
    OutputFormatter,
    OutputConfig,
    OutputVerbosity,
    OutputFormat,
    ProgressIndicator,
    create_formatter,
)
from .concise_prompts import (
    CONCISE_OUTPUT_SYSTEM_PROMPT,
    get_output_prompt,
    format_concise_output,
    OUTPUT_FORMATS,
    MAX_OUTPUT_LINES,
)

__all__ = [
    "__version__",
    "main",
    "cli",
    "OutputFormatter",
    "OutputConfig",
    "OutputVerbosity",
    "OutputFormat",
    "ProgressIndicator",
    "create_formatter",
    "CONCISE_OUTPUT_SYSTEM_PROMPT",
    "get_output_prompt",
    "format_concise_output",
    "OUTPUT_FORMATS",
    "MAX_OUTPUT_LINES",
]
