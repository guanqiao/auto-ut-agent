"""CLI module for PyUT Agent.

This module provides command-line interface for the PyUT Agent.

Usage:
    $ pyutagent generate --target com.example.MyClass
    $ pyutagent analyze --project /path/to/project
    $ pyutagent test --file TargetClass.java

Features:
    - Interactive mode with natural language support
    - Command mode for scripted usage
    - Batch mode with JSON output for CI/CD
    - Shared state with GUI (config, project history)

Example:
    >>> from pyutagent.cli import get_cli_context, InteractiveSession
    >>> ctx = get_cli_context()
    >>> session = InteractiveSession()
    >>> await session.start()
"""

__version__ = "0.1.0"

# Main entry points
from .main import main, cli

# Context and state management
from .main import (
    CLIContext,
    CLIState,
    OutputFormat,
    get_cli_context,
    save_shared_config,
    InteractiveSession,
    BatchOutput,
)

# Output formatting
from .output_formatter import (
    OutputFormatter,
    OutputConfig,
    OutputVerbosity,
    ProgressIndicator,
    create_formatter,
)

# Concise prompts
from .concise_prompts import (
    CONCISE_OUTPUT_SYSTEM_PROMPT,
    get_output_prompt,
    format_concise_output,
    OUTPUT_FORMATS,
    MAX_OUTPUT_LINES,
)

__all__ = [
    # Version
    "__version__",
    
    # Main entry points
    "main",
    "cli",
    
    # Context and state management
    "CLIContext",
    "CLIState",
    "OutputFormat",
    "get_cli_context",
    "save_shared_config",
    "InteractiveSession",
    "BatchOutput",
    
    # Output formatting
    "OutputFormatter",
    "OutputConfig",
    "OutputVerbosity",
    "ProgressIndicator",
    "create_formatter",
    
    # Concise prompts
    "CONCISE_OUTPUT_SYSTEM_PROMPT",
    "get_output_prompt",
    "format_concise_output",
    "OUTPUT_FORMATS",
    "MAX_OUTPUT_LINES",
]
