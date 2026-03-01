"""Error recovery - DEPRECATED.

This module is deprecated. Use pyutagent.core.error_recovery instead.
"""

import warnings

# Import everything from core module
from ..core.error_recovery import *  # noqa: F401,F403

# Issue deprecation warning
warnings.warn(
    "Importing from pyutagent.agent.error_recovery is deprecated. "
    "Use pyutagent.core.error_recovery instead.",
    DeprecationWarning,
    stacklevel=2
)
