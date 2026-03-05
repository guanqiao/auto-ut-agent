"""Performance Dashboard Module.

This module provides advanced performance monitoring capabilities:
- Unified performance dashboard
- Real-time performance analysis
- Performance baseline and anomaly detection
- Resource usage tracking
- Performance report generation

This is part of Phase 3 Week 23-24: Performance Monitoring Enhancement.
"""

import json
import logging
import os
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from pyutagent.core.metrics import MetricsCollector
    from pyutagent.agent.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of performance metrics."""
    EXECUTION = "execution"
    RESOURCE = "resource"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR = "error"
    QUALITY = "quality"
    CACHE = "cache"
    LLM = "llm"


class AlertType(Enum):
    """Types of performance alerts."""
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    TREND_WARNING = "trend_warning"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class Severity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """A single performance metric."""
    name: str
    value: float
    category: MetricCategory
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "unit": self.unit,
        }


@dataclass
class PerformanceAlert:
    """A performance alert."""
    alert_type: AlertType
    severity: Severity
    metric_name: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    metric_name: str
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float
    sample_count: int
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_anomaly(self, value: float, z_threshold: float = 3.0) -> bool:
        """Check if value is an anomaly."""
        if self.std_dev == 0:
            return abs(value - self.mean) > 0.1 * self.mean
        z_score = abs(value - self.mean) / self.std_dev
        return z_score > z_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "mean": self.mean,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class PerformanceReport:
    """Performance analysis report."""
    report_id: str
    start_time: datetime
    end_time: datetime
    metrics_summary: Dict[str, Dict[str, float]]
    alerts: List[PerformanceAlert]
    baselines: Dict[str, PerformanceBaseline]
    recommendations: List[str]
    trends: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "metrics_summary": self.metrics_summary,
            "alerts": [a.to_dict() for a in self.alerts],
            "baselines": {k: v.to_dict() for k, v in self.baselines.items()},
            "recommendations": self.recommendations,
            "trends": self.trends,
            "metadata": self.metadata,
        }


class MetricsStore:
    """Thread-safe metrics storage with time-series support."""
    
    def __init__(self, max_history: int = 10000, time_window_seconds: int = 3600):
        self.max_history = max_history
        self.time_window_seconds = time_window_seconds
        self._lock = threading.RLock()
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
    
    def record(self, metric: PerformanceMetric) -> None:
        """Record a metric."""
        with self._lock:
            self._metrics[metric.name].append(metric)
            self._cleanup_old_metrics()
    
    def record_counter(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
            metric = PerformanceMetric(
                name=name,
                value=self._counters[key],
                category=MetricCategory.THROUGHPUT,
                labels=labels or {},
            )
            self._metrics[name].append(metric)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
            metric = PerformanceMetric(
                name=name,
                value=value,
                category=MetricCategory.RESOURCE,
                labels=labels or {},
            )
            self._metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                category=MetricCategory.LATENCY,
                labels=labels or {},
            )
            self._metrics[name].append(metric)
    
    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics by name."""
        with self._lock:
            metrics = list(self._metrics.get(name, []))
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            return metrics
    
    def get_all_metrics(self, since: Optional[datetime] = None) -> Dict[str, List[PerformanceMetric]]:
        """Get all metrics."""
        with self._lock:
            result = {}
            for name in self._metrics:
                result[name] = self.get_metrics(name, since)
            return result
    
    def get_latest(self, name: str) -> Optional[PerformanceMetric]:
        """Get latest metric by name."""
        with self._lock:
            metrics = self._metrics.get(name, [])
            return metrics[-1] if metrics else None
    
    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        with self._lock:
            key = self._make_key(name, labels)
            return self._counters.get(key, 0.0)
    
    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        with self._lock:
            key = self._make_key(name, labels)
            return self._gauges.get(key, 0.0)
    
    def calculate_statistics(self, name: str, since: Optional[datetime] = None) -> Dict[str, float]:
        """Calculate statistics for a metric."""
        metrics = self.get_metrics(name, since)
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        stats = {
            "count": len(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
        
        if len(values) > 1:
            stats["std_dev"] = statistics.stdev(values)
            sorted_values = sorted(values)
            stats["percentile_95"] = sorted_values[int(len(sorted_values) * 0.95)]
            stats["percentile_99"] = sorted_values[int(len(sorted_values) * 0.99)]
        
        return stats
    
    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for labeled metrics."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than time window."""
        cutoff = datetime.now() - timedelta(seconds=self.time_window_seconds)
        for name in self._metrics:
            while self._metrics[name] and self._metrics[name][0].timestamp < cutoff:
                self._metrics[name].popleft()


class AlertManager:
    """Manages performance alerts."""
    
    def __init__(self, store: MetricsStore):
        self.store = store
        self._lock = threading.RLock()
        self._alerts: List[PerformanceAlert] = []
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._alert_handlers: List[Callable[[PerformanceAlert], None]] = []
    
    def set_threshold(
        self,
        metric_name: str,
        warning_threshold: float,
        error_threshold: float,
        comparison: str = "greater",
    ) -> None:
        """Set alert thresholds for a metric."""
        with self._lock:
            self._thresholds[metric_name] = {
                "warning": warning_threshold,
                "error": error_threshold,
                "comparison": comparison,
            }
    
    def check_thresholds(self) -> List[PerformanceAlert]:
        """Check all thresholds and generate alerts."""
        alerts = []
        
        with self._lock:
            for metric_name, threshold_config in self._thresholds.items():
                latest = self.store.get_latest(metric_name)
                if not latest:
                    continue
                
                value = latest.value
                comparison = threshold_config["comparison"]
                
                warning_threshold = threshold_config["warning"]
                error_threshold = threshold_config["error"]
                
                def compare(v, t):
                    if comparison == "greater":
                        return v > t
                    return v < t
                
                if compare(value, error_threshold):
                    alert = PerformanceAlert(
                        alert_type=AlertType.THRESHOLD_BREACH,
                        severity=Severity.ERROR,
                        metric_name=metric_name,
                        current_value=value,
                        threshold=error_threshold,
                        message=f"{metric_name} ({value:.2f}) exceeded error threshold ({error_threshold})",
                    )
                    alerts.append(alert)
                elif compare(value, warning_threshold):
                    alert = PerformanceAlert(
                        alert_type=AlertType.THRESHOLD_BREACH,
                        severity=Severity.WARNING,
                        metric_name=metric_name,
                        current_value=value,
                        threshold=warning_threshold,
                        message=f"{metric_name} ({value:.2f}) exceeded warning threshold ({warning_threshold})",
                    )
                    alerts.append(alert)
        
        for alert in alerts:
            self._add_alert(alert)
        
        return alerts
    
    def add_handler(self, handler: Callable[[PerformanceAlert], None]) -> None:
        """Add an alert handler."""
        self._alert_handlers.append(handler)
    
    def get_alerts(
        self,
        since: Optional[datetime] = None,
        severity: Optional[Severity] = None,
        unresolved_only: bool = False,
    ) -> List[PerformanceAlert]:
        """Get alerts with optional filtering."""
        with self._lock:
            alerts = self._alerts.copy()
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]
        
        return alerts
    
    def resolve_alert(self, alert: PerformanceAlert) -> None:
        """Mark an alert as resolved."""
        alert.resolved = True
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()
    
    def _add_alert(self, alert: PerformanceAlert) -> None:
        """Add an alert and notify handlers."""
        with self._lock:
            self._alerts.append(alert)
        
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")


class BaselineManager:
    """Manages performance baselines."""
    
    def __init__(self, store: MetricsStore):
        self.store = store
        self._lock = threading.RLock()
        self._baselines: Dict[str, PerformanceBaseline] = {}
        self._baseline_file: Optional[Path] = None
    
    def calculate_baseline(
        self,
        metric_name: str,
        since: Optional[datetime] = None,
    ) -> Optional[PerformanceBaseline]:
        """Calculate baseline for a metric."""
        metrics = self.store.get_metrics(metric_name, since)
        if len(metrics) < 10:
            return None
        
        values = sorted([m.value for m in metrics])
        
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            mean=statistics.mean(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            min_value=min(values),
            max_value=max(values),
            percentile_95=values[int(len(values) * 0.95)],
            percentile_99=values[int(len(values) * 0.99)],
            sample_count=len(values),
        )
        
        with self._lock:
            self._baselines[metric_name] = baseline
        
        return baseline
    
    def get_baseline(self, metric_name: str) -> Optional[PerformanceBaseline]:
        """Get baseline for a metric."""
        with self._lock:
            return self._baselines.get(metric_name)
    
    def check_anomaly(self, metric_name: str, value: float) -> bool:
        """Check if a value is anomalous."""
        baseline = self.get_baseline(metric_name)
        if not baseline:
            return False
        return baseline.is_anomaly(value)
    
    def load_baselines(self, file_path: Path) -> None:
        """Load baselines from file."""
        self._baseline_file = file_path
        if not file_path.exists():
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                for name, baseline_data in data.items():
                    baseline_data_copy = baseline_data.copy()
                    if 'last_updated' in baseline_data_copy:
                        baseline_data_copy['last_updated'] = datetime.fromisoformat(baseline_data_copy['last_updated'])
                    self._baselines[name] = PerformanceBaseline(
                        metric_name=name,
                        mean=baseline_data_copy.get('mean', 0),
                        std_dev=baseline_data_copy.get('std_dev', 0),
                        min_value=baseline_data_copy.get('min_value', 0),
                        max_value=baseline_data_copy.get('max_value', 0),
                        percentile_95=baseline_data_copy.get('percentile_95', 0),
                        percentile_99=baseline_data_copy.get('percentile_99', 0),
                        sample_count=baseline_data_copy.get('sample_count', 0),
                    )
            
            logger.info(f"Loaded {len(self._baselines)} baselines from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")
    
    def save_baselines(self, file_path: Optional[Path] = None) -> None:
        """Save baselines to file."""
        file_path = file_path or self._baseline_file
        if not file_path:
            return
        
        try:
            with self._lock:
                data = {name: b.to_dict() for name, b in self._baselines.items()}
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self._baselines)} baselines to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")


class PerformanceDashboard:
    """Unified performance dashboard."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        max_history: int = 10000,
        time_window_seconds: int = 3600,
        baseline_file: Optional[Path] = None,
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.store = MetricsStore(max_history, time_window_seconds)
        self.alert_manager = AlertManager(self.store)
        self.baseline_manager = BaselineManager(self.store)
        
        if baseline_file:
            self.baseline_manager.load_baselines(baseline_file)
        
        self._initialized = True
        self._start_time = datetime.now()
        
        self._setup_default_thresholds()
    
    @classmethod
    def get_instance(cls) -> "PerformanceDashboard":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.store.clear()
                cls._instance.alert_manager.clear_alerts()
            cls._instance = None
    
    def record_metric(
        self,
        name: str,
        value: float,
        category: MetricCategory = MetricCategory.EXECUTION,
        labels: Optional[Dict[str, str]] = None,
        unit: str = "",
    ) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            category=category,
            labels=labels or {},
            unit=unit,
        )
        self.store.record(metric)
        
        if self.baseline_manager.check_anomaly(name, value):
            alert = PerformanceAlert(
                alert_type=AlertType.ANOMALY_DETECTED,
                severity=Severity.WARNING,
                metric_name=name,
                current_value=value,
                threshold=0,
                message=f"Anomaly detected for {name}: {value:.2f}",
            )
            self.alert_manager._add_alert(alert)
    
    def record_latency(
        self,
        operation: str,
        duration_seconds: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record operation latency."""
        self.record_metric(
            f"{operation}_latency",
            duration_seconds,
            MetricCategory.LATENCY,
            labels,
            unit="seconds",
        )
    
    def record_throughput(
        self,
        operation: str,
        count: int = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record throughput."""
        self.store.record_counter(
            f"{operation}_throughput",
            count,
            labels,
        )
    
    def record_error(
        self,
        operation: str,
        error_type: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record an error."""
        labels = labels or {}
        labels["error_type"] = error_type
        self.store.record_counter(
            f"{operation}_errors",
            1,
            labels,
        )
    
    def record_resource_usage(
        self,
        resource: str,
        usage: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record resource usage."""
        self.store.record_gauge(
            f"{resource}_usage",
            usage,
            labels,
        )
    
    def generate_report(
        self,
        period_minutes: int = 60,
        include_recommendations: bool = True,
    ) -> PerformanceReport:
        """Generate a performance report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=period_minutes)
        
        metrics_summary = {}
        all_metrics = self.store.get_all_metrics(start_time)
        
        for name, metrics in all_metrics.items():
            if metrics:
                values = [m.value for m in metrics]
                metrics_summary[name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                }
        
        alerts = self.alert_manager.get_alerts(since=start_time)
        
        baselines = {}
        for name in all_metrics:
            baseline = self.baseline_manager.get_baseline(name)
            if baseline:
                baselines[name] = baseline
        
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(metrics_summary, alerts)
        
        trends = self._analyze_trends(all_metrics)
        
        return PerformanceReport(
            report_id=f"report_{int(time.time())}",
            start_time=start_time,
            end_time=end_time,
            metrics_summary=metrics_summary,
            alerts=alerts,
            baselines=baselines,
            recommendations=recommendations,
            trends=trends,
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dashboard summary."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        return {
            "uptime_seconds": (now - self._start_time).total_seconds(),
            "total_metrics": sum(len(m) for m in self.store.get_all_metrics().values()),
            "active_alerts": len(self.alert_manager.get_alerts(unresolved_only=True)),
            "baselines_tracked": len(self.baseline_manager._baselines),
            "thresholds_configured": len(self.alert_manager._thresholds),
        }
    
    def _setup_default_thresholds(self) -> None:
        """Setup default alert thresholds."""
        self.alert_manager.set_threshold("llm_call_latency", 5.0, 10.0)
        self.alert_manager.set_threshold("error_rate", 0.05, 0.1)
        self.alert_manager.set_threshold("memory_usage", 0.8, 0.9)
    
    def _generate_recommendations(
        self,
        metrics_summary: Dict[str, Dict[str, float]],
        alerts: List[PerformanceAlert],
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        for name, stats in metrics_summary.items():
            if "latency" in name and stats["mean"] > 1.0:
                recommendations.append(
                    f"Consider optimizing {name} - average latency is {stats['mean']:.2f}s"
                )
            
            if "error" in name and stats["mean"] > 0:
                recommendations.append(
                    f"Investigate errors in {name} - {stats['count']} occurrences"
                )
        
        for alert in alerts:
            if alert.alert_type == AlertType.THRESHOLD_BREACH:
                recommendations.append(
                    f"Address threshold breach: {alert.message}"
                )
        
        return recommendations
    
    def _analyze_trends(
        self,
        metrics: Dict[str, List[PerformanceMetric]],
    ) -> Dict[str, str]:
        """Analyze metric trends."""
        trends = {}
        
        for name, metric_list in metrics.items():
            if len(metric_list) < 10:
                continue
            
            values = [m.value for m in metric_list]
            
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            
            first_mean = statistics.mean(first_half)
            second_mean = statistics.mean(second_half)
            
            if first_mean == 0:
                continue
            
            change = (second_mean - first_mean) / first_mean
            
            if change > 0.1:
                trends[name] = "increasing"
            elif change < -0.1:
                trends[name] = "decreasing"
            else:
                trends[name] = "stable"
        
        return trends


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(
        self,
        dashboard: PerformanceDashboard,
        operation: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.dashboard = dashboard
        self.operation = operation
        self.labels = labels
        self.start_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.error: Optional[Exception] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        self.dashboard.record_latency(
            self.operation,
            self.duration,
            self.labels,
        )
        
        if exc_type:
            self.error = exc_val
            self.dashboard.record_error(
                self.operation,
                exc_type.__name__,
                self.labels,
            )
        
        return False


def get_dashboard() -> PerformanceDashboard:
    """Get the global performance dashboard."""
    return PerformanceDashboard.get_instance()


def reset_dashboard() -> None:
    """Reset the global dashboard."""
    PerformanceDashboard.reset_instance()


def time_operation(
    operation: str,
    labels: Optional[Dict[str, str]] = None,
) -> OperationTimer:
    """Create an operation timer."""
    return OperationTimer(get_dashboard(), operation, labels)
