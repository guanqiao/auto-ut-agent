"""Performance Monitoring and Optimization.

This module provides:
- Performance metrics collection
- Resource usage tracking
- Optimization suggestions
- Performance alerts
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    LATENCY = auto()
    THROUGHPUT = auto()
    ERROR_RATE = auto()
    CPU_USAGE = auto()
    MEMORY_USAGE = auto()
    TOKEN_USAGE = auto()
    API_CALLS = auto()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


@dataclass
class Metric:
    """A performance metric."""
    metric_type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """Performance report."""
    period_start: datetime
    period_end: datetime
    metrics: Dict[MetricType, Dict[str, float]]
    alerts: List[str]
    recommendations: List[str]


@dataclass
class PerformanceThresholds:
    """Performance thresholds."""
    max_latency_ms: float = 1000.0
    max_error_rate: float = 0.1
    max_memory_mb: float = 1024.0
    max_token_per_minute: int = 100000


class PerformanceMonitor:
    """Performance monitoring for agent operations.

    Features:
    - Metrics collection
    - Threshold monitoring
    - Alert generation
    - Performance reporting
    """

    def __init__(self, thresholds: Optional[PerformanceThresholds] = None):
        """Initialize performance monitor.

        Args:
            thresholds: Performance thresholds
        """
        self.thresholds = thresholds or PerformanceThresholds()
        self._metrics: List[Metric] = []
        self._alerts: List[tuple] = []
        self._start_time = datetime.now()

    def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric.

        Args:
            metric_type: Metric type
            value: Metric value
            tags: Optional tags
        """
        metric = Metric(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self._metrics.append(metric)
        self._check_thresholds(metric)

    def record_latency(
        self,
        operation: str,
        latency_ms: float
    ):
        """Record operation latency.

        Args:
            operation: Operation name
            latency_ms: Latency in milliseconds
        """
        self.record_metric(
            MetricType.LATENCY,
            latency_ms,
            {"operation": operation}
        )

    def record_error(
        self,
        operation: str,
        error: bool
    ):
        """Record operation result.

        Args:
            operation: Operation name
            error: Whether operation failed
        """
        self.record_metric(
            MetricType.ERROR_RATE,
            1.0 if error else 0.0,
            {"operation": operation}
        )

    def record_token_usage(
        self,
        tokens: int,
        model: Optional[str] = None
    ):
        """Record token usage.

        Args:
            tokens: Number of tokens
            model: Model name
        """
        tags = {}
        if model:
            tags["model"] = model

        self.record_metric(
            MetricType.TOKEN_USAGE,
            float(tokens),
            tags
        )

    def _check_thresholds(self, metric: Metric):
        """Check if metric exceeds thresholds."""
        if metric.metric_type == MetricType.LATENCY:
            if metric.value > self.thresholds.max_latency_ms:
                self._add_alert(
                    AlertSeverity.WARNING,
                    f"High latency: {metric.value:.2f}ms"
                )

        elif metric.metric_type == MetricType.ERROR_RATE:
            if metric.value > self.thresholds.max_error_rate:
                self._add_alert(
                    AlertSeverity.CRITICAL,
                    f"High error rate: {metric.value:.2%}"
                )

    def _add_alert(self, severity: AlertSeverity, message: str):
        """Add an alert."""
        self._alerts.append((severity, message, datetime.now()))
        logger.warning(f"[PerformanceMonitor] {severity.name}: {message}")

    def get_metrics_summary(
        self,
        metric_type: MetricType,
        minutes: int = 5
    ) -> Dict[str, float]:
        """Get metrics summary.

        Args:
            metric_type: Metric type
            minutes: Time window in minutes

        Returns:
            Summary statistics
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        relevant = [
            m for m in self._metrics
            if m.metric_type == metric_type and m.timestamp > cutoff
        ]

        if not relevant:
            return {"count": 0, "avg": 0, "min": 0, "max": 0}

        values = [m.value for m in relevant]

        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "total": sum(values)
        }

    def generate_report(
        self,
        period_minutes: int = 10
    ) -> PerformanceReport:
        """Generate performance report.

        Args:
            period_minutes: Report period

        Returns:
            Performance report
        """
        period_end = datetime.now()
        period_start = period_end - timedelta(minutes=period_minutes)

        metrics_by_type = {}
        for metric_type in MetricType:
            metrics_by_type[metric_type] = self.get_metrics_summary(
                metric_type, period_minutes
            )

        alerts = [
            f"[{sev.name}] {msg}"
            for sev, msg, _ in self._alerts
            if _ > period_start
        ]

        recommendations = self._generate_recommendations(metrics_by_type)

        return PerformanceReport(
            period_start=period_start,
            period_end=period_end,
            metrics=metrics_by_type,
            alerts=alerts,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        metrics_by_type: Dict[MetricType, Dict[str, float]]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        latency = metrics_by_type.get(MetricType.LATENCY, {})
        if latency.get("avg", 0) > 500:
            recommendations.append("Consider implementing response caching")

        error_rate = metrics_by_type.get(MetricType.ERROR_RATE, {})
        if error_rate.get("avg", 0) > 0.05:
            recommendations.append("Error rate is high, review error handling")

        tokens = metrics_by_type.get(MetricType.TOKEN_USAGE, {})
        if tokens.get("total", 0) > 50000:
            recommendations.append("High token usage, consider optimizing prompts")

        return recommendations

    def get_recent_alerts(
        self,
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent alerts.

        Args:
            count: Number of alerts

        Returns:
            List of alerts
        """
        return [
            {
                "severity": sev.name,
                "message": msg,
                "timestamp": ts.isoformat()
            }
            for sev, msg, ts in self._alerts[-count:]
        ]

    def clear_metrics(self):
        """Clear all metrics."""
        self._metrics.clear()
        self._alerts.clear()
        self._start_time = datetime.now()


class OperationTimer:
    """Context manager for timing operations."""

    def __init__(self, monitor: PerformanceMonitor, operation: str):
        """Initialize operation timer.

        Args:
            monitor: Performance monitor
            operation: Operation name
        """
        self.monitor = monitor
        self.operation = operation
        self.start_time = 0.0

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record metric."""
        latency_ms = (time.time() - self.start_time) * 1000
        self.monitor.record_latency(self.operation, latency_ms)

        if exc_type is not None:
            self.monitor.record_error(self.operation, True)
        else:
            self.monitor.record_error(self.operation, False)


def create_performance_monitor(
    max_latency_ms: float = 1000.0,
    max_error_rate: float = 0.1
) -> PerformanceMonitor:
    """Create a performance monitor.

    Args:
        max_latency_ms: Maximum latency threshold
        max_error_rate: Maximum error rate threshold

    Returns:
        PerformanceMonitor instance
    """
    thresholds = PerformanceThresholds(
        max_latency_ms=max_latency_ms,
        max_error_rate=max_error_rate
    )
    return PerformanceMonitor(thresholds)
