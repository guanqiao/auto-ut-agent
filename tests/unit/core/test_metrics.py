"""性能监控和指标测试"""
import pytest
import time
from pyutagent.core.metrics import (
    MetricsCollector,
    PerformanceTracker,
    MetricType,
    Metric
)


class TestMetric:
    """指标测试"""
    
    def test_create_metric(self):
        """测试创建指标"""
        metric = Metric(
            name="test_metric",
            value=100,
            metric_type=MetricType.COUNTER
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 100
        assert metric.metric_type == MetricType.COUNTER
    
    def test_metric_with_timestamp(self):
        """测试带时间戳的指标"""
        metric = Metric(
            name="test_metric",
            value=50,
            metric_type=MetricType.GAUGE
        )
        
        assert metric.timestamp is not None
    
    def test_metric_with_labels(self):
        """测试带标签的指标"""
        metric = Metric(
            name="test_metric",
            value=75,
            metric_type=MetricType.GAUGE,
            labels={"env": "test", "version": "1.0"}
        )
        
        assert metric.labels["env"] == "test"
        assert metric.labels["version"] == "1.0"


class TestMetricsCollector:
    """指标收集器测试"""
    
    def test_create_collector(self):
        """测试创建指标收集器"""
        collector = MetricsCollector()
        assert collector is not None
    
    def test_record_counter(self):
        """测试记录计数器"""
        collector = MetricsCollector()
        
        collector.record_counter("requests_total", 1)
        collector.record_counter("requests_total", 1)
        collector.record_counter("requests_total", 1)
        
        value = collector.get_metric_value("requests_total")
        assert value == 3
    
    def test_record_gauge(self):
        """测试记录仪表盘"""
        collector = MetricsCollector()
        
        collector.record_gauge("active_connections", 10)
        value = collector.get_metric_value("active_connections")
        assert value == 10
        
        collector.record_gauge("active_connections", 5)
        value = collector.get_metric_value("active_connections")
        assert value == 5
    
    def test_record_histogram(self):
        """测试记录直方图"""
        collector = MetricsCollector()
        
        for i in range(100):
            collector.record_histogram("response_time", i * 0.01)
        
        value = collector.get_metric_value("response_time")
        assert value["count"] == 100
        assert "sum" in value
        assert "buckets" in value
    
    def test_get_metric(self):
        """测试获取指标"""
        collector = MetricsCollector()
        
        collector.record_counter("test_counter", 10)
        metric = collector.get_metric("test_counter")
        
        assert metric is not None
        assert metric.name == "test_counter"
        assert metric.value == 10
    
    def test_get_nonexistent_metric(self):
        """测试获取不存在的指标"""
        collector = MetricsCollector()
        
        metric = collector.get_metric("nonexistent")
        
        assert metric is None
    
    def test_list_metrics(self):
        """测试列出所有指标"""
        collector = MetricsCollector()
        
        collector.record_counter("metric1", 1)
        collector.record_gauge("metric2", 2)
        collector.record_histogram("metric3", 3)
        
        metrics = collector.list_metrics()
        
        assert len(metrics) == 3
        assert "metric1" in metrics
        assert "metric2" in metrics
        assert "metric3" in metrics
    
    def test_clear_metrics(self):
        """测试清空指标"""
        collector = MetricsCollector()
        
        collector.record_counter("metric1", 1)
        collector.record_gauge("metric2", 2)
        
        collector.clear()
        
        metrics = collector.list_metrics()
        assert len(metrics) == 0
    
    def test_record_with_labels(self):
        """测试带标签记录指标"""
        collector = MetricsCollector()
        
        collector.record_counter(
            "requests_total",
            1,
            labels={"method": "GET", "path": "/api"}
        )
        collector.record_counter(
            "requests_total",
            1,
            labels={"method": "POST", "path": "/api"}
        )
        
        # 获取特定标签的指标
        metric = collector.get_metric("requests_total", {"method": "GET", "path": "/api"})
        assert metric is not None
        assert metric.value == 1
    
    def test_export_metrics(self):
        """测试导出指标"""
        collector = MetricsCollector()
        
        collector.record_counter("test_counter", 10)
        collector.record_gauge("test_gauge", 50)
        
        exported = collector.export_metrics()
        
        assert "test_counter" in exported
        assert "test_gauge" in exported
        assert exported["test_counter"]["value"] == 10
        assert exported["test_gauge"]["value"] == 50


class TestPerformanceTracker:
    """性能追踪器测试"""
    
    def test_create_tracker(self):
        """测试创建性能追踪器"""
        tracker = PerformanceTracker()
        assert tracker is not None
    
    def test_start_stop_timer(self):
        """测试开始/停止计时器"""
        tracker = PerformanceTracker()
        
        tracker.start_timer("operation")
        time.sleep(0.1)
        elapsed = tracker.stop_timer("operation")
        
        assert elapsed >= 0.1
        assert elapsed < 1.0
    
    def test_multiple_timers(self):
        """测试多个计时器"""
        tracker = PerformanceTracker()
        
        tracker.start_timer("op1")
        tracker.start_timer("op2")
        
        time.sleep(0.05)
        elapsed1 = tracker.stop_timer("op1")
        
        time.sleep(0.05)
        elapsed2 = tracker.stop_timer("op2")
        
        assert elapsed1 < elapsed2
    
    def test_record_execution_time(self):
        """测试记录执行时间"""
        tracker = PerformanceTracker()
        
        @tracker.record_execution_time("test_function")
        def test_function():
            time.sleep(0.05)
            return "result"
        
        result = test_function()
        
        assert result == "result"
        
        # 检查指标是否被记录
        metric = tracker._collector.get_metric("test_function")
        assert metric is not None
    
    def test_get_average_time(self):
        """测试获取平均时间"""
        tracker = PerformanceTracker()
        
        for _ in range(10):
            tracker.start_timer("operation")
            time.sleep(0.01)
            tracker.stop_timer("operation")
        
        avg_time = tracker.get_average_time("operation")
        
        assert avg_time >= 0.01
        assert avg_time < 1.0
    
    def test_get_max_time(self):
        """测试获取最大时间"""
        tracker = PerformanceTracker()
        
        times = [0.01, 0.02, 0.03, 0.04, 0.05]
        for t in times:
            tracker.start_timer("operation")
            time.sleep(t)
            tracker.stop_timer("operation")
        
        max_time = tracker.get_max_time("operation")
        
        assert max_time >= 0.05
    
    def test_get_min_time(self):
        """测试获取最小时间"""
        tracker = PerformanceTracker()
        
        times = [0.05, 0.04, 0.03, 0.02, 0.01]
        for t in times:
            tracker.start_timer("operation")
            time.sleep(t)
            tracker.stop_timer("operation")
        
        min_time = tracker.get_min_time("operation")
        
        assert min_time >= 0.01
    
    def test_get_total_time(self):
        """测试获取总时间"""
        tracker = PerformanceTracker()
        
        for _ in range(10):
            tracker.start_timer("operation")
            time.sleep(0.01)
            tracker.stop_timer("operation")
        
        total_time = tracker.get_total_time("operation")
        
        assert total_time >= 0.1
    
    def test_get_call_count(self):
        """测试获取调用次数"""
        tracker = PerformanceTracker()
        
        for _ in range(5):
            tracker.start_timer("operation")
            time.sleep(0.01)
            tracker.stop_timer("operation")
        
        count = tracker.get_call_count("operation")
        
        assert count == 5
    
    def test_reset_tracker(self):
        """测试重置追踪器"""
        tracker = PerformanceTracker()
        
        for _ in range(5):
            tracker.start_timer("operation")
            time.sleep(0.01)
            tracker.stop_timer("operation")
        
        tracker.reset()
        
        count = tracker.get_call_count("operation")
        assert count == 0
    
    def test_performance_tracking_overhead(self):
        """测试性能追踪开销"""
        tracker = PerformanceTracker()
        
        # 不追踪的执行时间
        start = time.time()
        for _ in range(100):
            pass
        baseline = time.time() - start
        
        # 追踪的执行时间
        start = time.time()
        for _ in range(100):
            tracker.start_timer("loop")
            tracker.stop_timer("loop")
        tracked = time.time() - start
        
        # 追踪开销应该很小
        overhead = tracked - baseline
        assert overhead < 1.0  # 开销应该小于 1 秒


class TestMetricTypes:
    """指标类型测试"""
    
    def test_metric_type_counter(self):
        """测试计数器类型"""
        assert MetricType.COUNTER.value == "counter"
    
    def test_metric_type_gauge(self):
        """测试仪表盘类型"""
        assert MetricType.GAUGE.value == "gauge"
    
    def test_metric_type_histogram(self):
        """测试直方图类型"""
        assert MetricType.HISTOGRAM.value == "histogram"
