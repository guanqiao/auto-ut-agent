"""Error recovery module for tools.

This module provides backward-compatible imports from core.error_recovery.
All error recovery classes have been consolidated into core.error_recovery.
"""

from ..core.error_recovery import (
    ErrorRecoveryManager,
    ErrorCategory,
    RecoveryStrategy,
    RecoveryResult,
    RecoveryContext,
    RecoveryAttempt,
    ErrorContext,
    ErrorClassifier,
    StatePreserver,
    GracefulDegradation,
    RecoveryManager,
    is_retryable_error,
    classify_error,
    create_recovery_manager,
    safe_execution_context,
)

__all__ = [
    "ErrorRecoveryManager",
    "ErrorCategory",
    "RecoveryStrategy",
    "RecoveryResult",
    "RecoveryContext",
    "RecoveryAttempt",
    "ErrorContext",
    "ErrorClassifier",
    "StatePreserver",
    "GracefulDegradation",
    "RecoveryManager",
    "is_retryable_error",
    "classify_error",
    "create_recovery_manager",
    "safe_execution_context",
]
