"""Retry manager - DEPRECATED.

This module is deprecated. Use pyutagent.core.retry_manager instead.
"""

import warnings

# Import everything from core module
from ..core.retry_manager import *  # noqa: F401,F403

# Issue deprecation warning
warnings.warn(
    "Importing from pyutagent.tools.retry_manager is deprecated. "
    "Use pyutagent.core.retry_manager instead.",
    DeprecationWarning,
    stacklevel=2
)
