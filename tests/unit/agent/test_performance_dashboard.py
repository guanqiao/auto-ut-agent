"""Tests for Performance Dashboard.

This module tests the performance monitoring system including:
- Metrics storage and retrieval
- Alert management
- Baseline calculation
- Performance dashboard
"""

import pytest
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from pyutagent.agent.performance_dashboard import (
    MetricCategory,
    AlertType,
    Severity,
    PerformanceMetric,
    PerformanceAlert,
    PerformanceBaseline,
    PerformanceReport,
    MetricsStore,
    AlertManager,
    BaselineManager,
    PerformanceDashboard,
    OperationTimer,
    get_dashboard,
    reset_dashboard,
    time_operation,
)


class TestMetricCategory:
    """Tests for MetricCategory enum."""
    
    def test_category_values(self):
        """Test category values."""
        assert MetricCategory.EXECUTION.value == "execution"
        assert MetricCategory.RESOURCE.value == "resource"
        assert MetricCategory.LATENCY.value == "latency"
        assert MetricCategory.ERROR.value == "error"


class TestAlertType:
    """Tests for AlertType enum."""
    
    def test_alert_type_values(self):
        """Test alert type values."""
        assert AlertType.THRESHOLD_BREACH.value == "threshold_breach"
        assert AlertType.ANOMALY_DETECTED.value == "anomaly_detected"


class TestSeverity:
    """Tests for Severity enum."""
    
    def test_severity_values(self):
        """Test severity values."""
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.ERROR.value == "error"
        assert Severity.CRITICAL.value == "critical"


class TestPerformanceMetric:
    """Tests for PerformanceMetric dataclass."""
    
    def test_metric_creation(self):
        """Test creating a metric."""
        metric = PerformanceMetric(
            name="test_metric",
            value=42.0,
            category=MetricCategory.LATENCY,
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.category == MetricCategory.LATENCY
        assert isinstance(metric.timestamp, datetime)
    
    def test_metric_with_labels(self):
        """Test metric with labels."""
        metric = PerformanceMetric(
            name="test_metric",
            value=10.0,
            category=MetricCategory.EXECUTION,
            labels={"operation": "test"},
        )
        
        assert metric.labels["operation"] == "test"
    
    def test_metric_to_dict(self):
        """Test metric serialization."""
        metric = PerformanceMetric(
            name="test_metric",
            value=10.0,
            category=MetricCategory.LATENCY,
            unit="seconds",
        )
        
        data = metric.to_dict()
        
        assert data["name"] == "test_metric"
        assert data["value"] == 10.0
        assert data["category"] == "latency"
        assert data["unit"] == "seconds"


class TestPerformanceAlert:
    """Tests for PerformanceAlert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = PerformanceAlert(
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=Severity.WARNING,
            metric_name="test_metric",
            current_value=100.0,
            threshold=50.0,
            message="Threshold exceeded",
        )
        
        assert alert.alert_type == AlertType.THRESHOLD_BREACH
        assert alert.severity == Severity.WARNING
        assert alert.resolved is False
    
    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = PerformanceAlert(
            alert_type=AlertType.ANOMALY_DETECTED,
            severity=Severity.ERROR,
            metric_name="test_metric",
            current_value=100.0,
            threshold=0,
            message="Anomaly detected",
        )
        
        data = alert.to_dict()
        
        assert data["alert_type"] == "anomaly_detected"
        assert data["severity"] == "error"
        assert data["resolved"] is False


class TestPerformanceBaseline:
    """Tests for PerformanceBaseline dataclass."""
    
    def test_baseline_creation(self):
        """Test creating a baseline."""
        baseline = PerformanceBaseline(
            metric_name="test_metric",
            mean=50.0,
            std_dev=10.0,
            min_value=30.0,
            max_value=70.0,
            percentile_95=65.0,
            percentile_99=68.0,
            sample_count=100,
        )
        
        assert baseline.metric_name == "test_metric"
        assert baseline.mean == 50.0
        assert baseline.sample_count == 100
    
    def test_is_anomaly_within_range(self):
        """Test anomaly detection within range."""
        baseline = PerformanceBaseline(
            metric_name="test_metric",
            mean=50.0,
            std_dev=10.0,
            min_value=30.0,
            max_value=70.0,
            percentile_95=65.0,
            percentile_99=68.0,
            sample_count=100,
        )
        
        assert baseline.is_anomaly(55.0) is False
        assert baseline.is_anomaly(40.0) is False
    
    def test_is_anomaly_outside_range(self):
        """Test anomaly detection outside range."""
        baseline = PerformanceBaseline(
            metric_name="test_metric",
            mean=50.0,
            std_dev=10.0,
            min_value=30.0,
            max_value=70.0,
            percentile_95=65.0,
            percentile_99=68.0,
            sample_count=100,
        )
        
        assert baseline.is_anomaly(100.0) is True
        assert baseline.is_anomaly(0.0) is True
    
    def test_baseline_to_dict(self):
        """Test baseline serialization."""
        baseline = PerformanceBaseline(
            metric_name="test_metric",
            mean=50.0,
            std_dev=10.0,
            min_value=30.0,
            max_value=70.0,
            percentile_95=65.0,
            percentile_99=68.0,
            sample_count=100,
        )
        
        data = baseline.to_dict()
        
        assert data["metric_name"] == "test_metric"
        assert data["mean"] == 50.0


class TestMetricsStore:
    """Tests for MetricsStore."""
    
    def test_store_creation(self):
        """Test creating a store."""
        store = MetricsStore()
        
        assert store.max_history == 10000
        assert store.time_window_seconds == 3600
    
    def test_record_metric(self):
        """Test recording a metric."""
        store = MetricsStore()
        
        metric = PerformanceMetric(
            name="test_metric",
            value=10.0,
            category=MetricCategory.EXECUTION,
        )
        store.record(metric)
        
        metrics = store.get_metrics("test_metric")
        assert len(metrics) == 1
        assert metrics[0].value == 10.0
    
    def test_record_counter(self):
        """Test recording a counter."""
        store = MetricsStore()
        
        store.record_counter("requests", 1)
        store.record_counter("requests", 2)
        
        assert store.get_counter("requests") == 3
    
    def test_record_gauge(self):
        """Test recording a gauge."""
        store = MetricsStore()
        
        store.record_gauge("memory", 1024.0)
        
        assert store.get_gauge("memory") == 1024.0
    
    def test_record_histogram(self):
        """Test recording a histogram."""
        store = MetricsStore()
        
        store.record_histogram("latency", 0.1)
        store.record_histogram("latency", 0.2)
        store.record_histogram("latency", 0.3)
        
        metrics = store.get_metrics("latency")
        assert len(metrics) == 3
    
    def test_get_latest(self):
        """Test getting latest metric."""
        store = MetricsStore()
        
        store.record_histogram("latency", 0.1)
        store.record_histogram("latency", 0.2)
        
        latest = store.get_latest("latency")
        assert latest.value == 0.2
    
    def test_get_latest_empty(self):
        """Test getting latest from empty store."""
        store = MetricsStore()
        
        latest = store.get_latest("nonexistent")
        assert latest is None
    
    def test_calculate_statistics(self):
        """Test calculating statistics."""
        store = MetricsStore()
        
        for i in range(10):
            store.record_histogram("latency", float(i))
        
        stats = store.calculate_statistics("latency")
        
        assert stats["count"] == 10
        assert stats["min"] == 0.0
        assert stats["max"] == 9.0
        assert "mean" in stats
        assert "std_dev" in stats
    
    def test_calculate_statistics_empty(self):
        """Test calculating statistics for empty store."""
        store = MetricsStore()
        
        stats = store.calculate_statistics("nonexistent")
        
        assert stats == {}
    
    def test_clear(self):
        """Test clearing store."""
        store = MetricsStore()
        
        store.record_counter("requests", 1)
        store.record_gauge("memory", 1024.0)
        store.clear()
        
        assert store.get_counter("requests") == 0
        assert store.get_gauge("memory") == 0.0
    
    def test_labeled_metrics(self):
        """Test labeled metrics."""
        store = MetricsStore()
        
        store.record_counter("requests", 1, labels={"endpoint": "/api"})
        store.record_counter("requests", 1, labels={"endpoint": "/web"})
        
        assert store.get_counter("requests", labels={"endpoint": "/api"}) == 1


class TestAlertManager:
    """Tests for AlertManager."""
    
    def test_set_threshold(self):
        """Test setting thresholds."""
        store = MetricsStore()
        manager = AlertManager(store)
        
        manager.set_threshold("latency", 1.0, 2.0)
        
        assert "latency" in manager._thresholds
        assert manager._thresholds["latency"]["warning"] == 1.0
        assert manager._thresholds["latency"]["error"] == 2.0
    
    def test_check_thresholds_no_breach(self):
        """Test threshold check without breach."""
        store = MetricsStore()
        manager = AlertManager(store)
        
        manager.set_threshold("latency", 1.0, 2.0)
        store.record_gauge("latency", 0.5)
        
        alerts = manager.check_thresholds()
        
        assert len(alerts) == 0
    
    def test_check_thresholds_warning(self):
        """Test threshold check with warning."""
        store = MetricsStore()
        manager = AlertManager(store)
        
        manager.set_threshold("latency", 1.0, 2.0)
        store.record_gauge("latency", 1.5)
        
        alerts = manager.check_thresholds()
        
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.WARNING
    
    def test_check_thresholds_error(self):
        """Test threshold check with error."""
        store = MetricsStore()
        manager = AlertManager(store)
        
        manager.set_threshold("latency", 1.0, 2.0)
        store.record_gauge("latency", 3.0)
        
        alerts = manager.check_thresholds()
        
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.ERROR
    
    def test_add_handler(self):
        """Test adding alert handler."""
        store = MetricsStore()
        manager = AlertManager(store)
        
        handler = MagicMock()
        manager.add_handler(handler)
        
        manager.set_threshold("latency", 1.0, 2.0)
        store.record_gauge("latency", 3.0)
        manager.check_thresholds()
        
        handler.assert_called_once()
    
    def test_get_alerts(self):
        """Test getting alerts."""
        store = MetricsStore()
        manager = AlertManager(store)
        
        alert = PerformanceAlert(
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=Severity.WARNING,
            metric_name="test",
            current_value=1.0,
            threshold=0.5,
            message="Test",
        )
        manager._add_alert(alert)
        
        alerts = manager.get_alerts()
        
        assert len(alerts) == 1
    
    def test_resolve_alert(self):
        """Test resolving alert."""
        store = MetricsStore()
        manager = AlertManager(store)
        
        alert = PerformanceAlert(
            alert_type=AlertType.THRESHOLD_BREACH,
            severity=Severity.WARNING,
            metric_name="test",
            current_value=1.0,
            threshold=0.5,
            message="Test",
        )
        manager._add_alert(alert)
        manager.resolve_alert(alert)
        
        alerts = manager.get_alerts(unresolved_only=True)
        
        assert len(alerts) == 0


class TestBaselineManager:
    """Tests for BaselineManager."""
    
    def test_calculate_baseline(self):
        """Test calculating baseline."""
        store = MetricsStore()
        manager = BaselineManager(store)
        
        for i in range(20):
            store.record_histogram("latency", float(i))
        
        baseline = manager.calculate_baseline("latency")
        
        assert baseline is not None
        assert baseline.metric_name == "latency"
        assert baseline.sample_count == 20
    
    def test_calculate_baseline_insufficient_data(self):
        """Test calculating baseline with insufficient data."""
        store = MetricsStore()
        manager = BaselineManager(store)
        
        for i in range(5):
            store.record_histogram("latency", float(i))
        
        baseline = manager.calculate_baseline("latency")
        
        assert baseline is None
    
    def test_get_baseline(self):
        """Test getting baseline."""
        store = MetricsStore()
        manager = BaselineManager(store)
        
        for i in range(20):
            store.record_histogram("latency", float(i))
        
        manager.calculate_baseline("latency")
        baseline = manager.get_baseline("latency")
        
        assert baseline is not None
    
    def test_check_anomaly(self):
        """Test anomaly checking."""
        store = MetricsStore()
        manager = BaselineManager(store)
        
        for i in range(20):
            store.record_histogram("latency", 10.0 + float(i) * 0.1)
        
        manager.calculate_baseline("latency")
        
        assert manager.check_anomaly("latency", 11.0) is False
        assert manager.check_anomaly("latency", 100.0) is True
    
    def test_save_and_load_baselines(self):
        """Test saving and loading baselines."""
        store = MetricsStore()
        manager = BaselineManager(store)
        
        for i in range(20):
            store.record_histogram("latency", float(i))
        
        manager.calculate_baseline("latency")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            manager.save_baselines(temp_path)
            
            new_manager = BaselineManager(MetricsStore())
            new_manager.load_baselines(temp_path)
            
            baseline = new_manager.get_baseline("latency")
            assert baseline is not None
            assert baseline.sample_count == 20
        finally:
            temp_path.unlink()


class TestPerformanceDashboard:
    """Tests for PerformanceDashboard."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        PerformanceDashboard.reset_instance()
        self.dashboard = PerformanceDashboard()
        yield
        PerformanceDashboard.reset_instance()
    
    def test_singleton(self):
        """Test singleton pattern."""
        dashboard1 = PerformanceDashboard.get_instance()
        dashboard2 = PerformanceDashboard.get_instance()
        
        assert dashboard1 is dashboard2
    
    def test_reset_instance(self):
        """Test resetting singleton."""
        dashboard1 = PerformanceDashboard.get_instance()
        PerformanceDashboard.reset_instance()
        dashboard2 = PerformanceDashboard.get_instance()
        
        assert dashboard1 is not dashboard2
    
    def test_record_metric(self):
        """Test recording a metric."""
        self.dashboard.record_metric(
            "test_metric",
            10.0,
            MetricCategory.LATENCY,
        )
        
        metrics = self.dashboard.store.get_metrics("test_metric")
        assert len(metrics) == 1
    
    def test_record_latency(self):
        """Test recording latency."""
        self.dashboard.record_latency("operation", 0.5)
        
        metrics = self.dashboard.store.get_metrics("operation_latency")
        assert len(metrics) == 1
        assert metrics[0].value == 0.5
    
    def test_record_throughput(self):
        """Test recording throughput."""
        self.dashboard.record_throughput("requests", 10)
        
        counter = self.dashboard.store.get_counter("requests_throughput")
        assert counter == 10
    
    def test_record_error(self):
        """Test recording error."""
        self.dashboard.record_error("operation", "ValueError")
        
        counter = self.dashboard.store.get_counter(
            "operation_errors",
            labels={"error_type": "ValueError"},
        )
        assert counter == 1
    
    def test_record_resource_usage(self):
        """Test recording resource usage."""
        self.dashboard.record_resource_usage("memory", 1024.0)
        
        gauge = self.dashboard.store.get_gauge("memory_usage")
        assert gauge == 1024.0
    
    def test_generate_report(self):
        """Test generating report."""
        for i in range(10):
            self.dashboard.record_latency("operation", 0.1 * (i + 1))
        
        report = self.dashboard.generate_report(period_minutes=60)
        
        assert report.report_id.startswith("report_")
        assert len(report.metrics_summary) > 0
    
    def test_get_summary(self):
        """Test getting summary."""
        self.dashboard.record_metric("test", 1.0)
        
        summary = self.dashboard.get_summary()
        
        assert "uptime_seconds" in summary
        assert "total_metrics" in summary


class TestOperationTimer:
    """Tests for OperationTimer."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        PerformanceDashboard.reset_instance()
        self.dashboard = PerformanceDashboard()
        yield
        PerformanceDashboard.reset_instance()
    
    def test_timer_context_manager(self):
        """Test timer as context manager."""
        with OperationTimer(self.dashboard, "test_op"):
            time.sleep(0.01)
        
        metrics = self.dashboard.store.get_metrics("test_op_latency")
        assert len(metrics) == 1
        assert metrics[0].value >= 0.01
    
    def test_timer_with_error(self):
        """Test timer with error."""
        with pytest.raises(ValueError):
            with OperationTimer(self.dashboard, "test_op"):
                raise ValueError("test error")
        
        counter = self.dashboard.store.get_counter(
            "test_op_errors",
            labels={"error_type": "ValueError"},
        )
        assert counter == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_dashboard(self):
        """Test get_dashboard function."""
        PerformanceDashboard.reset_instance()
        
        dashboard = get_dashboard()
        
        assert isinstance(dashboard, PerformanceDashboard)
        
        PerformanceDashboard.reset_instance()
    
    def test_reset_dashboard(self):
        """Test reset_dashboard function."""
        dashboard1 = get_dashboard()
        reset_dashboard()
        dashboard2 = get_dashboard()
        
        assert dashboard1 is not dashboard2
    
    def test_time_operation(self):
        """Test time_operation function."""
        PerformanceDashboard.reset_instance()
        
        with time_operation("test"):
            time.sleep(0.01)
        
        dashboard = get_dashboard()
        metrics = dashboard.store.get_metrics("test_latency")
        
        assert len(metrics) == 1
        
        PerformanceDashboard.reset_instance()


class TestIntegration:
    """Integration tests for performance dashboard."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        PerformanceDashboard.reset_instance()
        self.dashboard = PerformanceDashboard()
        yield
        PerformanceDashboard.reset_instance()
    
    def test_full_monitoring_workflow(self):
        """Test full monitoring workflow."""
        for i in range(20):
            self.dashboard.record_latency("llm_call", 0.5 + i * 0.1)
            self.dashboard.record_throughput("requests", 1)
        
        baseline = self.dashboard.baseline_manager.calculate_baseline("llm_call_latency")
        assert baseline is not None
        
        self.dashboard.alert_manager.set_threshold("llm_call_latency", 1.0, 2.0)
        
        self.dashboard.record_latency("llm_call", 3.0)
        alerts = self.dashboard.alert_manager.check_thresholds()
        
        assert len(alerts) > 0
        
        report = self.dashboard.generate_report()
        
        assert len(report.metrics_summary) > 0
        assert len(report.alerts) > 0
    
    def test_concurrent_recording(self):
        """Test concurrent metric recording."""
        import threading
        
        def record_metrics():
            for i in range(100):
                self.dashboard.record_metric(
                    f"metric_{threading.current_thread().name}",
                    float(i),
                )
        
        threads = [
            threading.Thread(target=record_metrics, name=f"thread_{i}")
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        total_metrics = sum(
            len(self.dashboard.store.get_metrics(f"metric_thread_{i}"))
            for i in range(5)
        )
        
        assert total_metrics == 500
