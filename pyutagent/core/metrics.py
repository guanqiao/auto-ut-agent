"""性能监控和指标收集"""
import time
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"  # 计数器
    GAUGE = "gauge"  # 仪表盘
    HISTOGRAM = "histogram"  # 直方图


@dataclass
class Metric:
    """指标"""
    name: str
    value: Any
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
    
    def record_counter(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """记录计数器"""
        key = self._make_key(name, labels)
        self._counters[key] += value
        
        self._metrics[key] = Metric(
            name=name,
            value=self._counters[key],
            metric_type=MetricType.COUNTER,
            labels=labels or {}
        )
        
        logger.debug(f"Recorded counter {name}: {value}")
    
    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """记录仪表盘"""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        
        self._metrics[key] = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        )
        
        logger.debug(f"Recorded gauge {name}: {value}")
    
    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """记录直方图"""
        key = self._make_key(name, labels)
        self._histograms[key].append(value)
        
        # 计算统计信息
        values = self._histograms[key]
        histogram_data = {
            "count": len(values),
            "sum": sum(values),
            "avg": statistics.mean(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "buckets": self._create_buckets(values)
        }
        
        self._metrics[key] = Metric(
            name=name,
            value=histogram_data,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {}
        )
        
        logger.debug(f"Recorded histogram {name}: {value}")
    
    def _create_buckets(self, values: List[float]) -> Dict[str, int]:
        """创建直方图桶"""
        if not values:
            return {}
        
        # 创建简单的桶分布
        buckets = {
            "<0.01": 0,
            "0.01-0.05": 0,
            "0.05-0.1": 0,
            "0.1-0.5": 0,
            "0.5-1.0": 0,
            ">1.0": 0
        }
        
        for value in values:
            if value < 0.01:
                buckets["<0.01"] += 1
            elif value < 0.05:
                buckets["0.01-0.05"] += 1
            elif value < 0.1:
                buckets["0.05-0.1"] += 1
            elif value < 0.5:
                buckets["0.1-0.5"] += 1
            elif value < 1.0:
                buckets["0.5-1.0"] += 1
            else:
                buckets[">1.0"] += 1
        
        return buckets
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """创建指标键"""
        if not labels:
            return name
        
        labels_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{labels_str}}}"
    
    def get_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """获取指标"""
        key = self._make_key(name, labels)
        return self._metrics.get(key)
    
    def get_metric_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Any:
        """获取指标值"""
        metric = self.get_metric(name, labels)
        return metric.value if metric else None
    
    def list_metrics(self) -> List[str]:
        """列出所有指标名称"""
        return list(set(m.name for m in self._metrics.values()))
    
    def clear(self) -> None:
        """清空所有指标"""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        logger.debug("Cleared all metrics")
    
    def export_metrics(self) -> Dict[str, Any]:
        """导出指标"""
        result = {}
        
        for name in self.list_metrics():
            metrics = [m for m in self._metrics.values() if m.name == name]
            if metrics:
                result[name] = {
                    "type": metrics[0].metric_type.value,
                    "value": metrics[0].value,
                    "timestamp": metrics[0].timestamp.isoformat(),
                    "labels": metrics[0].labels
                }
        
        return result


class PerformanceTracker:
    """性能追踪器"""
    
    def __init__(self):
        self._collector = MetricsCollector()
        self._timers: Dict[str, float] = {}
        self._execution_times: Dict[str, List[float]] = defaultdict(list)
    
    def start_timer(self, name: str) -> None:
        """开始计时器"""
        self._timers[name] = time.time()
        logger.debug(f"Started timer: {name}")
    
    def stop_timer(self, name: str) -> float:
        """停止计时器"""
        if name not in self._timers:
            logger.warning(f"Timer not started: {name}")
            return 0.0
        
        elapsed = time.time() - self._timers[name]
        del self._timers[name]
        
        # 记录执行时间
        self._execution_times[name].append(elapsed)
        self._collector.record_histogram(name, elapsed)
        
        logger.debug(f"Stopped timer {name}: {elapsed:.6f}s")
        return elapsed
    
    def record_execution_time(self, name: str) -> Callable:
        """装饰器：记录函数执行时间"""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                self.start_timer(name)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.stop_timer(name)
            return wrapper
        return decorator
    
    def get_average_time(self, name: str) -> float:
        """获取平均执行时间"""
        times = self._execution_times.get(name, [])
        return statistics.mean(times) if times else 0.0
    
    def get_max_time(self, name: str) -> float:
        """获取最大执行时间"""
        times = self._execution_times.get(name, [])
        return max(times) if times else 0.0
    
    def get_min_time(self, name: str) -> float:
        """获取最小执行时间"""
        times = self._execution_times.get(name, [])
        return min(times) if times else 0.0
    
    def get_total_time(self, name: str) -> float:
        """获取总执行时间"""
        times = self._execution_times.get(name, [])
        return sum(times) if times else 0.0
    
    def get_call_count(self, name: str) -> int:
        """获取调用次数"""
        return len(self._execution_times.get(name, []))
    
    def reset(self) -> None:
        """重置追踪器"""
        self._timers.clear()
        self._execution_times.clear()
        self._collector.clear()
        logger.debug("Reset performance tracker")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        result = {}
        
        for name in self._execution_times:
            times = self._execution_times[name]
            result[name] = {
                "count": len(times),
                "avg": statistics.mean(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
                "total": sum(times) if times else 0
            }
        
        return result


# 全局指标收集器实例
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def get_metrics() -> Dict[str, Any]:
    """获取所有指标"""
    collector = get_metrics_collector()
    return collector.export_metrics()


def record_counter(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """记录计数器（便捷函数）"""
    collector = get_metrics_collector()
    collector.record_counter(name, value, labels)


def record_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """记录仪表盘（便捷函数）"""
    collector = get_metrics_collector()
    collector.record_gauge(name, value, labels)


def record_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """记录直方图（便捷函数）"""
    collector = get_metrics_collector()
    collector.record_histogram(name, value, labels)
