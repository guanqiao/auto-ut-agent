"""Unified termination condition checker.

This module provides a centralized termination condition checker to ensure
consistent behavior and prevent scattered logic across multiple files.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TerminationReason(Enum):
    """Reasons for termination."""
    MAX_ITERATIONS = auto()
    TARGET_COVERAGE_REACHED = auto()
    USER_STOPPED = auto()
    USER_TERMINATED = auto()
    MAX_ATTEMPTS_EXCEEDED = auto()
    ERROR_THRESHOLD_EXCEEDED = auto()
    TIMEOUT_EXCEEDED = auto()
    NO_UNCOVERED_LINES = auto()


@dataclass
class TerminationState:
    """Current termination state."""
    should_stop: bool = False
    reason: Optional[TerminationReason] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class TerminationChecker:
    """Unified termination condition checker.
    
    Centralizes all termination condition checks to ensure consistency
    and prevent scattered logic across multiple files.
    
    Features:
    - Multiple termination conditions
    - Timeout support
    - Error threshold
    - Callback notifications
    - State tracking
    
    Attributes:
        max_iterations: Maximum number of iterations
        target_coverage: Target coverage percentage
        max_total_attempts: Maximum total attempts
        max_error_count: Maximum error count before termination
        timeout_seconds: Optional timeout in seconds
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        target_coverage: float = 0.8,
        max_total_attempts: int = 50,
        max_error_count: int = 10,
        timeout_seconds: Optional[float] = None
    ):
        """Initialize termination checker.
        
        Args:
            max_iterations: Maximum number of iterations
            target_coverage: Target coverage percentage (0.0-1.0)
            max_total_attempts: Maximum total attempts
            max_error_count: Maximum error count before termination
            timeout_seconds: Optional timeout in seconds
        """
        self.max_iterations = max_iterations
        self.target_coverage = target_coverage
        self.max_total_attempts = max_total_attempts
        self.max_error_count = max_error_count
        self.timeout_seconds = timeout_seconds
        
        self._start_time: Optional[float] = None
        self._error_count = 0
        self._total_attempts = 0
        self._callbacks: List[Callable[[TerminationState], None]] = []
        self._last_check_result: Optional[TerminationState] = None
    
    def start(self):
        """Start the timer."""
        self._start_time = time.time()
        logger.debug(f"[TerminationChecker] Started - MaxIterations: {self.max_iterations}, TargetCoverage: {self.target_coverage:.1%}")
    
    def register_callback(self, callback: Callable[[TerminationState], None]):
        """Register a callback for termination events.
        
        Args:
            callback: Function to call when termination occurs
        """
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[TerminationState], None]):
        """Unregister a callback.
        
        Args:
            callback: Function to unregister
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def record_error(self, count: int = 1):
        """Record error occurrence(s).
        
        Args:
            count: Number of errors to record
        """
        self._error_count += count
    
    def record_attempt(self, count: int = 1):
        """Record attempt(s).
        
        Args:
            count: Number of attempts to record
        """
        self._total_attempts += count
    
    @property
    def error_count(self) -> int:
        """Get current error count."""
        return self._error_count
    
    @property
    def total_attempts(self) -> int:
        """Get total attempts."""
        return self._total_attempts
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    def check(
        self,
        current_iteration: int,
        current_coverage: float,
        is_stopped: bool = False,
        is_terminated: bool = False,
        has_uncovered_lines: bool = True
    ) -> TerminationState:
        """Check all termination conditions.
        
        Args:
            current_iteration: Current iteration number
            current_coverage: Current coverage percentage (0.0-1.0)
            is_stopped: Whether user requested stop
            is_terminated: Whether user requested termination
            has_uncovered_lines: Whether there are uncovered lines
            
        Returns:
            TerminationState with stop decision and reason
        """
        if is_terminated:
            return self._create_state(True, TerminationReason.USER_TERMINATED, "User terminated")
        
        if is_stopped:
            return self._create_state(True, TerminationReason.USER_STOPPED, "User stopped")
        
        if current_iteration > self.max_iterations:
            return self._create_state(
                True, 
                TerminationReason.MAX_ITERATIONS,
                f"Max iterations ({self.max_iterations}) reached",
                {"iterations": current_iteration}
            )
        
        if current_coverage >= self.target_coverage:
            return self._create_state(
                True,
                TerminationReason.TARGET_COVERAGE_REACHED,
                f"Target coverage ({self.target_coverage:.1%}) reached",
                {"coverage": current_coverage}
            )
        
        if not has_uncovered_lines:
            return self._create_state(
                True,
                TerminationReason.NO_UNCOVERED_LINES,
                "No uncovered lines found",
                {"coverage": current_coverage}
            )
        
        if self._total_attempts >= self.max_total_attempts:
            return self._create_state(
                True,
                TerminationReason.MAX_ATTEMPTS_EXCEEDED,
                f"Max attempts ({self.max_total_attempts}) exceeded",
                {"attempts": self._total_attempts}
            )
        
        if self._error_count >= self.max_error_count:
            return self._create_state(
                True,
                TerminationReason.ERROR_THRESHOLD_EXCEEDED,
                f"Error threshold ({self.max_error_count}) exceeded",
                {"errors": self._error_count}
            )
        
        if self.timeout_seconds and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed >= self.timeout_seconds:
                return self._create_state(
                    True,
                    TerminationReason.TIMEOUT_EXCEEDED,
                    f"Timeout ({self.timeout_seconds}s) exceeded",
                    {"elapsed": elapsed}
                )
        
        return TerminationState(should_stop=False)
    
    def check_iteration(
        self,
        current_iteration: int,
        current_coverage: float,
        is_stopped: bool = False,
        is_terminated: bool = False
    ) -> bool:
        """Simple check for iteration loop termination.
        
        Args:
            current_iteration: Current iteration number
            current_coverage: Current coverage percentage
            is_stopped: Whether user requested stop
            is_terminated: Whether user requested termination
            
        Returns:
            True if should stop
        """
        state = self.check(current_iteration, current_coverage, is_stopped, is_terminated)
        return state.should_stop
    
    def _create_state(
        self, 
        should_stop: bool, 
        reason: TerminationReason, 
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> TerminationState:
        """Create termination state and notify callbacks.
        
        Args:
            should_stop: Whether should stop
            reason: Reason for termination
            message: Human-readable message
            details: Additional details
            
        Returns:
            TerminationState instance
        """
        state = TerminationState(
            should_stop=should_stop,
            reason=reason,
            message=message,
            details=details or {}
        )
        
        self._last_check_result = state
        
        if should_stop:
            logger.info(f"[TerminationChecker] Stopping - Reason: {reason.name}, Message: {message}")
            for callback in self._callbacks:
                try:
                    callback(state)
                except Exception as e:
                    logger.warning(f"[TerminationChecker] Callback error: {e}")
        
        return state
    
    def reset(self):
        """Reset the checker state."""
        self._start_time = None
        self._error_count = 0
        self._total_attempts = 0
        self._last_check_result = None
        logger.debug("[TerminationChecker] Reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current state.
        
        Returns:
            Dictionary with current state summary
        """
        return {
            "max_iterations": self.max_iterations,
            "target_coverage": self.target_coverage,
            "max_total_attempts": self.max_total_attempts,
            "max_error_count": self.max_error_count,
            "timeout_seconds": self.timeout_seconds,
            "current_error_count": self._error_count,
            "current_total_attempts": self._total_attempts,
            "elapsed_time": self.elapsed_time,
            "last_check_result": {
                "should_stop": self._last_check_result.should_stop,
                "reason": self._last_check_result.reason.name if self._last_check_result and self._last_check_result.reason else None,
                "message": self._last_check_result.message if self._last_check_result else None,
            } if self._last_check_result else None,
        }


def create_termination_checker(
    max_iterations: int = 10,
    target_coverage: float = 0.8,
    **kwargs
) -> TerminationChecker:
    """Create a termination checker with common settings.
    
    Args:
        max_iterations: Maximum iterations
        target_coverage: Target coverage
        **kwargs: Additional arguments
        
    Returns:
        Configured TerminationChecker
    """
    return TerminationChecker(
        max_iterations=max_iterations,
        target_coverage=target_coverage,
        **kwargs
    )
