"""CLI module for PyUT Agent.

This module provides command-line interface for the PyUT Agent.

Usage:
    $ pyutagent generate --target com.example.MyClass
    $ pyutagent analyze --project /path/to/project
    $ pyutagent test --file TargetClass.java
"""

__version__ = "0.1.0"

from .main import main, cli

__all__ = [
    "__version__",
    "main",
    "cli",
]
