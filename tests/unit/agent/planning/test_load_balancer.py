"""Unit tests for LoadBalancer module."""

import pytest
from datetime import datetime, timedelta
from typing import List

from pyutagent.agent.planning.load_balancer import (
    LoadBalancer,
    LoadBalancerConfig,
    LoadBalancingStrategy,
    LoadMonitor,
    WorkerStats,
)
from pyutagent.agent.planning.parallel_executor import PriorityTask, TaskStatus


class TestWorkerStats:
    """Test WorkerStats dataclass."""
    
    def test_worker_stats_initialization(self):
        """Test WorkerStats initializes with default values."""
        stats = WorkerStats(worker_id="worker-1")
        
        assert stats.worker_id == "worker-1"
        assert stats.current_tasks == 0
        assert stats.max_capacity == 10
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 0
        assert stats.avg_execution_time == 0.0
        assert stats.success_rate == 1.0
        assert stats.resource_utilization == 0.0
    
    def test_worker_stats_load_calculation(self):
        """Test load property calculation."""
        stats = WorkerStats(worker_id="worker-1", current_tasks=5, max_capacity=10)
        assert stats.load == 0.5
        
        stats_empty = WorkerStats(worker_id="worker-2", current_tasks=0, max_capacity=10)
        assert stats_empty.load == 0.0
        
        stats_full = WorkerStats(worker_id="worker-3", current_tasks=10, max_capacity=10)
        assert stats_full.load == 1.0
    
    def test_worker_stats_load_zero_capacity(self):
        """Test load property with zero capacity."""
        stats = WorkerStats(worker_id="worker-1", max_capacity=0)
        assert stats.load == 0.0
    
    def test_worker_stats_available_capacity(self):
        """Test available property calculation."""
        stats = WorkerStats(worker_id="worker-1", current_tasks=3, max_capacity=10)
        assert stats.available == 7
        
        stats_full = WorkerStats(worker_id="worker-2", current_tasks=10, max_capacity=10)
        assert stats_full.available == 0
    
    def test_worker_stats_update_execution_time(self):
        """Test execution time update using exponential moving average."""
        stats = WorkerStats(worker_id="worker-1")
        
        stats.update_execution_time(100.0)
        expected_first = 0.3 * 100.0 + 0.7 * 0.0
        assert abs(stats.avg_execution_time - expected_first) < 0.01
        
        stats.update_execution_time(200.0)
        expected_second = 0.3 * 200.0 + 0.7 * expected_first
        assert abs(stats.avg_execution_time - expected_second) < 0.01
    
    def test_worker_stats_update_success_rate(self):
        """Test success rate update."""
        stats = WorkerStats(worker_id="worker-1")
        
        stats.completed_tasks = 8
        stats.failed_tasks = 2
        stats.update_success_rate(True)
        
        assert stats.success_rate == 0.8


class TestLoadMonitor:
    """Test LoadMonitor class."""
    
    def test_load_monitor_initialization(self):
        """Test LoadMonitor initializes correctly."""
        monitor = LoadMonitor()
        
        assert monitor.config is not None
        assert monitor._worker_stats == {}
        assert monitor._task_history == []
    
    def test_register_worker(self):
        """Test worker registration."""
        monitor = LoadMonitor()
        monitor.register_worker("worker-1", max_capacity=20)
        
        stats = monitor.get_worker_stats("worker-1")
        assert stats is not None
        assert stats.worker_id == "worker-1"
        assert stats.max_capacity == 20
    
    def test_register_worker_duplicate(self):
        """Test registering same worker twice."""
        monitor = LoadMonitor()
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.register_worker("worker-1", max_capacity=20)
        
        stats = monitor.get_worker_stats("worker-1")
        assert stats.max_capacity == 10
    
    def test_assign_task(self):
        """Test task assignment tracking."""
        monitor = LoadMonitor()
        monitor.register_worker("worker-1", max_capacity=10)
        
        monitor.assign_task("worker-1", "task-1")
        
        stats = monitor.get_worker_stats("worker-1")
        assert stats.current_tasks == 1
        assert stats.last_task_time is not None
    
    def test_assign_task_auto_register(self):
        """Test that assign_task auto-registers worker."""
        monitor = LoadMonitor()
        monitor.assign_task("worker-1", "task-1")
        
        stats = monitor.get_worker_stats("worker-1")
        assert stats is not None
        assert stats.current_tasks == 1
    
    def test_complete_task_success(self):
        """Test successful task completion."""
        monitor = LoadMonitor()
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.assign_task("worker-1", "task-1")
        
        monitor.complete_task("worker-1", "task-1", execution_time=150.0, success=True)
        
        stats = monitor.get_worker_stats("worker-1")
        assert stats.current_tasks == 0
        assert stats.completed_tasks == 1
        assert stats.failed_tasks == 0
        assert stats.avg_execution_time > 0
    
    def test_complete_task_failure(self):
        """Test failed task completion."""
        monitor = LoadMonitor()
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.assign_task("worker-1", "task-1")
        
        monitor.complete_task("worker-1", "task-1", execution_time=150.0, success=False)
        
        stats = monitor.get_worker_stats("worker-1")
        assert stats.current_tasks == 0
        assert stats.completed_tasks == 0
        assert stats.failed_tasks == 1
    
    def test_complete_task_unknown_worker(self):
        """Test completing task for unknown worker."""
        monitor = LoadMonitor()
        monitor.complete_task("unknown-worker", "task-1", 100.0, True)
        
        stats = monitor.get_worker_stats("unknown-worker")
        assert stats is None
    
    def test_get_all_stats(self):
        """Test getting all worker statistics."""
        monitor = LoadMonitor()
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.register_worker("worker-2", max_capacity=20)
        
        all_stats = monitor.get_all_stats()
        
        assert len(all_stats) == 2
        assert "worker-1" in all_stats
        assert "worker-2" in all_stats
    
    def test_is_hotspot(self):
        """Test hotspot detection."""
        config = LoadBalancerConfig(hotspot_threshold=0.8)
        monitor = LoadMonitor(config=config)
        monitor.register_worker("worker-1", max_capacity=10)
        
        assert monitor.is_hotspot("worker-1") is False
        
        for i in range(9):
            monitor.assign_task("worker-1", f"task-{i}")
        
        assert monitor.is_hotspot("worker-1") is True
    
    def test_get_hotspots(self):
        """Test getting all hotspots."""
        config = LoadBalancerConfig(hotspot_threshold=0.8)
        monitor = LoadMonitor(config=config)
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.register_worker("worker-2", max_capacity=10)
        
        for i in range(9):
            monitor.assign_task("worker-1", f"task-{i}")
        
        for i in range(3):
            monitor.assign_task("worker-2", f"task-{i}")
        
        hotspots = monitor.get_hotspots()
        assert len(hotspots) == 1
        assert "worker-1" in hotspots
    
    def test_get_load_trend(self):
        """Test load trend calculation."""
        monitor = LoadMonitor()
        monitor.register_worker("worker-1", max_capacity=10)
        
        trend = monitor.get_load_trend("worker-1")
        assert trend == 0.0
    
    def test_get_stats_summary(self):
        """Test summary statistics."""
        monitor = LoadMonitor()
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.register_worker("worker-2", max_capacity=10)
        
        monitor.assign_task("worker-1", "task-1")
        monitor.assign_task("worker-1", "task-2")
        monitor.assign_task("worker-2", "task-3")
        
        summary = monitor.get_stats_summary()
        
        assert summary["total_workers"] == 2
        assert summary["avg_load"] > 0
        assert "max_load" in summary
        assert "min_load" in summary
        assert "hotspot_count" in summary
    
    def test_get_stats_summary_empty(self):
        """Test summary with no workers."""
        monitor = LoadMonitor()
        summary = monitor.get_stats_summary()
        
        assert summary == {}


class TestLoadBalancer:
    """Test LoadBalancer class."""
    
    def test_load_balancer_initialization(self):
        """Test LoadBalancer initializes correctly."""
        balancer = LoadBalancer()
        
        assert balancer.monitor is not None
        assert balancer.config is not None
        assert balancer._worker_weights == {}
    
    def test_select_worker_round_robin(self):
        """Test round-robin worker selection."""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.ROUND_ROBIN)
        monitor = LoadMonitor(config=config)
        balancer = LoadBalancer(monitor=monitor, config=config)
        
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.register_worker("worker-2", max_capacity=10)
        monitor.register_worker("worker-3", max_capacity=10)
        
        selected = balancer.select_worker()
        assert selected.worker_id == "worker-1"
        assert selected.strategy == LoadBalancingStrategy.ROUND_ROBIN
        
        selected = balancer.select_worker()
        assert selected.worker_id == "worker-2"
        
        selected = balancer.select_worker()
        assert selected.worker_id == "worker-3"
        
        selected = balancer.select_worker()
        assert selected.worker_id == "worker-1"
    
    def test_select_worker_least_connections(self):
        """Test least connections worker selection."""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
        monitor = LoadMonitor(config=config)
        balancer = LoadBalancer(monitor=monitor, config=config)
        
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.register_worker("worker-2", max_capacity=10)
        
        monitor.assign_task("worker-1", "task-1")
        monitor.assign_task("worker-1", "task-2")
        monitor.assign_task("worker-2", "task-3")
        
        selected = balancer.select_worker()
        assert selected.worker_id == "worker-2"
        assert selected.reason == "Least connections selection"
    
    def test_select_worker_weighted_least_connections(self):
        """Test weighted least connections selection."""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS)
        monitor = LoadMonitor(config=config)
        balancer = LoadBalancer(monitor=monitor, config=config)
        
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.register_worker("worker-2", max_capacity=10)
        
        monitor.assign_task("worker-1", "task-1")
        monitor.assign_task("worker-2", "task-2")
        
        monitor.complete_task("worker-1", "task-1", 100.0, success=True)
        monitor.complete_task("worker-2", "task-2", 200.0, success=True)
        
        selected = balancer.select_worker()
        assert selected.strategy == LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS
    
    def test_select_worker_shortest_time(self):
        """Test shortest expected time selection."""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.SHORTEST_EXPECTED_TIME)
        monitor = LoadMonitor(config=config)
        balancer = LoadBalancer(monitor=monitor, config=config)
        
        monitor.register_worker("worker-1", max_capacity=10)
        monitor.register_worker("worker-2", max_capacity=10)
        
        task = PriorityTask(id="task-1", description="Test task")
        
        for i in range(5):
            monitor.complete_task("worker-1", f"task-{i}", 100.0, success=True)
        
        for i in range(5):
            monitor.complete_task("worker-2", f"task-{i+5}", 200.0, success=True)
        
        selected = balancer.select_worker(task=task)
        assert selected.worker_id == "worker-1"
        assert selected.reason == "Shortest expected time selection"
    
    def test_select_worker_no_workers(self):
        """Test selection with no workers raises error."""
        balancer = LoadBalancer()
        
        with pytest.raises(ValueError, match="No available workers"):
            balancer.select_worker()
    
    def test_select_worker_custom_list(self):
        """Test selection with custom worker list."""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.ROUND_ROBIN)
        monitor = LoadMonitor(config=config)
        balancer = LoadBalancer(monitor=monitor, config=config)
        
        monitor.register_worker("worker-1")
        monitor.register_worker("worker-2")
        monitor.register_worker("worker-3")
        
        selected = balancer.select_worker(available_workers=["worker-2", "worker-3"])
        assert selected.worker_id == "worker-2"
    
    def test_record_assignment(self):
        """Test recording task assignment."""
        balancer = LoadBalancer()
        balancer.monitor.register_worker("worker-1", max_capacity=10)
        
        balancer.record_assignment("worker-1", "task-1")
        
        stats = balancer.monitor.get_worker_stats("worker-1")
        assert stats.current_tasks == 1
    
    def test_record_completion(self):
        """Test recording task completion."""
        balancer = LoadBalancer()
        balancer.monitor.register_worker("worker-1", max_capacity=10)
        balancer.record_assignment("worker-1", "task-1")
        
        balancer.record_completion("worker-1", "task-1", execution_time=150.0, success=True)
        
        stats = balancer.monitor.get_worker_stats("worker-1")
        assert stats.current_tasks == 0
        assert stats.completed_tasks == 1
    
    def test_get_balance_stats(self):
        """Test getting balance statistics."""
        config = LoadBalancerConfig(
            strategy=LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS,
            enable_adaptive_weights=True,
            enable_hotspot_detection=True,
        )
        balancer = LoadBalancer(config=config)
        balancer.monitor.register_worker("worker-1", max_capacity=10)
        
        stats = balancer.get_balance_stats()
        
        assert stats["strategy"] == "WEIGHTED_LEAST_CONNECTIONS"
        assert stats["adaptive_weights_enabled"] is True
        assert stats["hotspot_detection_enabled"] is True
        assert "worker_weights" in stats
    
    def test_update_weights(self):
        """Test weight update mechanism."""
        config = LoadBalancerConfig(enable_adaptive_weights=True)
        balancer = LoadBalancer(config=config)
        balancer.monitor.register_worker("worker-1", max_capacity=10)
        
        initial_weight = balancer._worker_weights["worker-1"]
        
        for i in range(5):
            balancer.record_assignment("worker-1", f"task-{i}")
            balancer.record_completion("worker-1", f"task-{i}", 100.0, success=True)
        
        balancer.update_weights()
        
        new_weight = balancer._worker_weights["worker-1"]
        assert new_weight != initial_weight


class TestLoadBalancerConfig:
    """Test LoadBalancerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LoadBalancerConfig()
        
        assert config.strategy == LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS
        assert config.enable_adaptive_weights is True
        assert config.enable_hotspot_detection is True
        assert config.hotspot_threshold == 0.8
        assert config.weight_adjustment_interval == 60.0
        assert config.max_history_size == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LoadBalancerConfig(
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            enable_adaptive_weights=False,
            hotspot_threshold=0.9,
        )
        
        assert config.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert config.enable_adaptive_weights is False
        assert config.hotspot_threshold == 0.9


class TestLoadBalanceDecision:
    """Test LoadBalanceDecision dataclass."""
    
    def test_decision_creation(self):
        """Test creating a load balance decision."""
        from pyutagent.agent.planning.load_balancer import LoadBalanceDecision
        
        decision = LoadBalanceDecision(
            worker_id="worker-1",
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            score=0.5,
            reason="Test selection",
            estimated_wait_time=10.0,
        )
        
        assert decision.worker_id == "worker-1"
        assert decision.strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert decision.score == 0.5
        assert decision.reason == "Test selection"
        assert decision.estimated_wait_time == 10.0


class TestLoadBalancerIntegration:
    """Integration tests for LoadBalancer."""
    
    def test_balancer_distributes_load_evenly(self):
        """Test that load balancer distributes tasks evenly."""
        config = LoadBalancerConfig(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
        monitor = LoadMonitor(config=config)
        balancer = LoadBalancer(monitor=monitor, config=config)
        
        num_workers = 3
        num_tasks = 9
        
        for i in range(num_workers):
            monitor.register_worker(f"worker-{i}", max_capacity=10)
        
        for i in range(num_tasks):
            decision = balancer.select_worker()
            balancer.record_assignment(decision.worker_id, f"task-{i}")
            balancer.record_completion(decision.worker_id, f"task-{i}", 100.0, success=True)
        
        all_stats = monitor.get_all_stats()
        loads = [stats.load for stats in all_stats.values()]
        
        max_load_diff = max(loads) - min(loads)
        assert max_load_diff < 0.2
    
    def test_adaptive_weight_convergence(self):
        """Test that adaptive weights converge over time."""
        config = LoadBalancerConfig(
            strategy=LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS,
            enable_adaptive_weights=True,
        )
        balancer = LoadBalancer(config=config)
        
        balancer.monitor.register_worker("fast-worker", max_capacity=10)
        balancer.monitor.register_worker("slow-worker", max_capacity=10)
        
        for i in range(20):
            balancer.record_assignment("fast-worker", f"task-{i}")
            balancer.record_completion("fast-worker", f"task-{i}", 50.0, success=True)
            
            balancer.record_assignment("slow-worker", f"task-{i+20}")
            balancer.record_completion("slow-worker", f"task-{i+20}", 200.0, success=True)
        
        balancer.update_weights()
        
        fast_weight = balancer._worker_weights["fast-worker"]
        slow_weight = balancer._worker_weights["slow-worker"]
        
        assert fast_weight >= slow_weight
