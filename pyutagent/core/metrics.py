"""Performance metrics collection and monitoring.

This module provides performance monitoring capabilities:
- Execution time tracking
- Memory usage monitoring
- LLM call statistics
- Error rate tracking
- Performance reports
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MetricRecord:
    """A single metric record."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingRecord:
    """A timing record for an operation."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallStats:
    """Statistics for LLM calls."""
    total_calls: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    avg_time: float = 0.0
    avg_tokens: float = 0.0
    
    def record_call(self, tokens: int, time_taken: float, success: bool):
        self.total_calls += 1
        self.total_tokens += tokens
        self.total_time += time_taken
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.avg_time = self.total_time / self.total_calls
        self.avg_tokens = self.total_tokens / self.total_calls


@dataclass
class ErrorStats:
    """Statistics for errors."""
    total_errors: int = 0
    by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_step: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recovery_success: int = 0
    recovery_failure: int = 0
    
    def record_error(self, category: str, step: str, recovered: bool):
        self.total_errors += 1
        self.by_category[category] += 1
        self.by_step[step] += 1
        if recovered:
            self.recovery_success += 1
        else:
            self.recovery_failure += 1


class MetricsCollector:
    """Collects and aggregates performance metrics.
    
    Features:
    - Operation timing
    - Memory tracking
    - LLM call statistics
    - Error tracking
    - Report generation
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        
        self._metrics: List[MetricRecord] = []
        self._timings: List[TimingRecord] = []
        self._llm_stats = LLMCallStats()
        self._error_stats = ErrorStats()
        
        self._active_timings: Dict[str, TimingRecord] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        
        self._session_start = time.time()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value."""
        if not self.enabled:
            return
        
        record = MetricRecord(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            metadata=metadata or {}
        )
        self._metrics.append(record)
        logger.debug(f"[Metrics] Recorded {name}={value}")
    
    def increment_counter(self, name: str, delta: int = 1):
        """Increment a counter."""
        self._counters[name] += delta
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        self._gauges[name] = value
    
    def start_timer(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start timing an operation.
        
        Args:
            operation: Name of the operation
            metadata: Optional metadata
            
        Returns:
            Timer ID for stopping
        """
        if not self.enabled:
            return operation
        
        timer_id = f"{operation}_{time.time_ns()}"
        
        record = TimingRecord(
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self._active_timings[timer_id] = record
        logger.debug(f"[Metrics] Started timer: {operation}")
        
        return timer_id
    
    def stop_timer(
        self,
        timer_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> Optional[float]:
        """Stop a timer and record the duration.
        
        Args:
            timer_id: Timer ID from start_timer
            success: Whether operation succeeded
            error: Optional error message
            
        Returns:
            Duration in seconds, or None if timer not found
        """
        if not self.enabled:
            return None
        
        record = self._active_timings.pop(timer_id, None)
        
        if not record:
            logger.warning(f"[Metrics] Timer not found: {timer_id}")
            return None
        
        record.end_time = time.time()
        record.duration = record.end_time - record.start_time
        record.success = success
        record.error = error
        
        self._timings.append(record)
        
        logger.debug(
            f"[Metrics] Stopped timer: {record.operation} - "
            f"Duration: {record.duration:.3f}s, Success: {success}"
        )
        
        return record.duration
    
    def record_llm_call(
        self,
        tokens: int,
        time_taken: float,
        success: bool = True
    ):
        """Record an LLM API call."""
        self._llm_stats.record_call(tokens, time_taken, success)
        logger.debug(
            f"[Metrics] LLM call recorded - Tokens: {tokens}, "
            f"Time: {time_taken:.2f}s, Success: {success}"
        )
    
    def record_error(
        self,
        category: str,
        step: str,
        recovered: bool = False
    ):
        """Record an error occurrence."""
        self._error_stats.record_error(category, step, recovered)
        logger.debug(
            f"[Metrics] Error recorded - Category: {category}, "
            f"Step: {step}, Recovered: {recovered}"
        )
    
    def time_operation(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'OperationTimer':
        """Context manager for timing an operation.
        
        Args:
            operation: Name of the operation
            metadata: Optional metadata
            
        Returns:
            OperationTimer context manager
        """
        return OperationTimer(self, operation, metadata)
    
    def timed(self, operation: str) -> Callable:
        """Decorator for timing a function.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                timer_id = self.start_timer(operation)
                try:
                    result = await func(*args, **kwargs)
                    self.stop_timer(timer_id, success=True)
                    return result
                except Exception as e:
                    self.stop_timer(timer_id, success=False, error=str(e))
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                timer_id = self.start_timer(operation)
                try:
                    result = func(*args, **kwargs)
                    self.stop_timer(timer_id, success=True)
                    return result
                except Exception as e:
                    self.stop_timer(timer_id, success=False, error=str(e))
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
        
        return decorator
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for an operation.
        
        Args:
            operation: Operation name
            
        Returns:
            Statistics dictionary
        """
        timings = [t for t in self._timings if t.operation == operation]
        
        if not timings:
            return {"operation": operation, "count": 0}
        
        durations = [t.duration for t in timings if t.duration is not None]
        success_count = sum(1 for t in timings if t.success)
        
        return {
            "operation": operation,
            "count": len(timings),
            "success_count": success_count,
            "failure_count": len(timings) - success_count,
            "success_rate": success_count / len(timings) if timings else 0,
            "total_time": sum(durations),
            "avg_time": statistics.mean(durations) if durations else 0,
            "min_time": min(durations) if durations else 0,
            "max_time": max(durations) if durations else 0,
            "median_time": statistics.median(durations) if durations else 0,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.
        
        Returns:
            Summary dictionary
        """
        session_duration = time.time() - self._session_start
        
        operations = set(t.operation for t in self._timings)
        operation_stats = {op: self.get_operation_stats(op) for op in operations}
        
        return {
            "session_id": self._session_id,
            "session_duration": session_duration,
            "total_operations": len(self._timings),
            "total_metrics": len(self._metrics),
            "llm_stats": {
                "total_calls": self._llm_stats.total_calls,
                "total_tokens": self._llm_stats.total_tokens,
                "total_time": self._llm_stats.total_time,
                "success_rate": (
                    self._llm_stats.success_count / self._llm_stats.total_calls
                    if self._llm_stats.total_calls > 0 else 0
                ),
                "avg_time": self._llm_stats.avg_time,
                "avg_tokens": self._llm_stats.avg_tokens,
            },
            "error_stats": {
                "total_errors": self._error_stats.total_errors,
                "recovery_rate": (
                    self._error_stats.recovery_success / self._error_stats.total_errors
                    if self._error_stats.total_errors > 0 else 0
                ),
                "by_category": dict(self._error_stats.by_category),
                "by_step": dict(self._error_stats.by_step),
            },
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "operations": operation_stats,
        }
    
    def generate_report(self) -> str:
        """Generate a human-readable report.
        
        Returns:
            Report string
        """
        summary = self.get_summary()
        
        lines = [
            "=" * 60,
            "Performance Report",
            "=" * 60,
            f"Session ID: {summary['session_id']}",
            f"Session Duration: {summary['session_duration']:.2f}s",
            "",
            "LLM Statistics:",
            f"  Total Calls: {summary['llm_stats']['total_calls']}",
            f"  Total Tokens: {summary['llm_stats']['total_tokens']}",
            f"  Total Time: {summary['llm_stats']['total_time']:.2f}s",
            f"  Success Rate: {summary['llm_stats']['success_rate']:.1%}",
            "",
            "Error Statistics:",
            f"  Total Errors: {summary['error_stats']['total_errors']}",
            f"  Recovery Rate: {summary['error_stats']['recovery_rate']:.1%}",
            "",
            "Operation Statistics:",
        ]
        
        for op, stats in summary['operations'].items():
            lines.extend([
                f"  {op}:",
                f"    Count: {stats['count']}",
                f"    Success Rate: {stats['success_rate']:.1%}",
                f"    Avg Time: {stats['avg_time']:.3f}s",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, path: Union[str, Path]):
        """Save report to a file.
        
        Args:
            path: File path to save
        """
        report = self.generate_report()
        Path(path).write_text(report, encoding='utf-8')
        logger.info(f"[Metrics] Report saved to {path}")
    
    def reset(self):
        """Reset all metrics."""
        self._metrics.clear()
        self._timings.clear()
        self._llm_stats = LLMCallStats()
        self._error_stats = ErrorStats()
        self._active_timings.clear()
        self._counters.clear()
        self._gauges.clear()
        self._session_start = time.time()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("[Metrics] All metrics reset")


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.collector = collector
        self.operation = operation
        self.metadata = metadata
        self._timer_id: Optional[str] = None
        self._duration: Optional[float] = None
    
    def __enter__(self):
        self._timer_id = self.collector.start_timer(self.operation, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = str(exc_val) if exc_val else None
        self._duration = self.collector.stop_timer(self._timer_id, success, error)
        return False
    
    @property
    def duration(self) -> Optional[float]:
        return self._duration


_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


def setup_metrics(enabled: bool = True) -> MetricsCollector:
    """Setup the global metrics collector.
    
    Args:
        enabled: Whether metrics collection is enabled
        
    Returns:
        MetricsCollector instance
    """
    global _metrics_instance
    _metrics_instance = MetricsCollector(enabled=enabled)
    return _metrics_instance
