"""Unit tests for Dependency Tracker module."""

import pytest
from datetime import datetime
from typing import Any, List

from pyutagent.agent.planning.dependency_tracker import (
    DependencyGraph,
    DependencyTracker,
    DependencyNode,
    DependencyChange,
    DependencyChangeType,
    DependencyAnalysisResult,
)


class TestDependencyNode:
    """Tests for DependencyNode."""
    
    def test_node_creation(self):
        """Test creating a dependency node."""
        node = DependencyNode(id="node_1")
        
        assert node.id == "node_1"
        assert node.dependencies == set()
        assert node.dependents == set()
        assert node.status == "pending"
    
    def test_node_with_dependencies(self):
        """Test creating node with dependencies."""
        node = DependencyNode(
            id="node_1",
            dependencies={"dep1", "dep2"},
        )
        
        assert len(node.dependencies) == 2
        assert "dep1" in node.dependencies
        assert "dep2" in node.dependencies
    
    def test_add_dependency(self):
        """Test adding a dependency."""
        node = DependencyNode(id="node_1")
        
        node.add_dependency("dep1")
        
        assert "dep1" in node.dependencies
    
    def test_remove_dependency(self):
        """Test removing a dependency."""
        node = DependencyNode(
            id="node_1",
            dependencies={"dep1", "dep2"},
        )
        
        node.remove_dependency("dep1")
        
        assert "dep1" not in node.dependencies
        assert "dep2" in node.dependencies


class TestDependencyGraph:
    """Tests for DependencyGraph."""
    
    def test_graph_creation(self):
        """Test creating a dependency graph."""
        graph = DependencyGraph()
        
        assert graph is not None
        assert len(graph._nodes) == 0
    
    def test_add_node(self):
        """Test adding a node to the graph."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        
        assert "node_1" in graph._nodes
    
    def test_add_node_with_dependencies(self):
        """Test adding node with dependencies."""
        graph = DependencyGraph()
        
        graph.add_node("node_1", dependencies={"dep1", "dep2"})
        
        node = graph._nodes["node_1"]
        assert len(node.dependencies) == 2
    
    def test_remove_node(self):
        """Test removing a node."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        result = graph.remove_node("node_1")
        
        assert result is True
        assert "node_1" not in graph._nodes
    
    def test_remove_nonexistent_node(self):
        """Test removing a nonexistent node."""
        graph = DependencyGraph()
        
        result = graph.remove_node("nonexistent")
        
        assert result is False
    
    def test_add_dependency(self):
        """Test adding a dependency between nodes."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2")
        result = graph.add_dependency("node_1", "node_2")
        
        assert result is True
        assert "node_2" in graph._nodes["node_1"].dependencies
        assert "node_1" in graph._nodes["node_2"].dependents
    
    def test_remove_dependency(self):
        """Test removing a dependency."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2")
        graph.add_dependency("node_1", "node_2")
        
        result = graph.remove_dependency("node_1", "node_2")
        
        assert result is True
        assert "node_2" not in graph._nodes["node_1"].dependencies
    
    def test_get_dependencies(self):
        """Test getting dependencies of a node."""
        graph = DependencyGraph()
        
        graph.add_node("node_1", dependencies={"dep1", "dep2"})
        
        deps = graph.get_dependencies("node_1")
        
        assert len(deps) == 2
        assert "dep1" in deps
        assert "dep2" in deps
    
    def test_get_dependents(self):
        """Test getting dependents of a node."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        
        dependents = graph.get_dependents("node_1")
        
        assert len(dependents) == 1
        assert "node_2" in dependents
    
    def test_get_ready_nodes(self):
        """Test getting ready nodes."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")  # No dependencies - ready
        graph.add_node("node_2", dependencies={"node_1"})  # Has dependency - not ready
        graph.add_node("node_3")  # No dependencies - ready
        
        ready = graph.get_ready_nodes()
        
        assert len(ready) == 2
        assert "node_1" in ready
        assert "node_3" in ready
        assert "node_2" not in ready
    
    def test_mark_completed(self):
        """Test marking a node as completed."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        
        graph.mark_completed("node_1")
        
        assert graph._nodes["node_1"].status == "completed"
        assert "node_1" in graph._completed
        
        # node_2 should now be ready
        ready = graph.get_ready_nodes()
        assert "node_2" in ready
    
    def test_mark_failed(self):
        """Test marking a node as failed."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.mark_failed("node_1")
        
        assert graph._nodes["node_1"].status == "failed"
        assert "node_1" in graph._failed
    
    def test_has_cycle_no_cycle(self):
        """Test cycle detection - no cycle."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        graph.add_node("node_3", dependencies={"node_2"})
        
        has_cycle = graph.has_cycle()
        
        assert has_cycle is False
    
    def test_has_cycle_with_cycle(self):
        """Test cycle detection - with cycle."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        graph.add_node("node_3", dependencies={"node_2"})
        
        # Create cycle
        graph.add_dependency("node_1", "node_3")
        
        has_cycle = graph.has_cycle()
        
        assert has_cycle is True
    
    def test_topological_sort(self):
        """Test topological sorting."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        graph.add_node("node_3", dependencies={"node_2"})
        
        sorted_nodes = graph.topological_sort()
        
        assert len(sorted_nodes) == 3
        assert sorted_nodes.index("node_1") < sorted_nodes.index("node_2")
        assert sorted_nodes.index("node_2") < sorted_nodes.index("node_3")
    
    def test_topological_sort_with_cycle(self):
        """Test topological sort with cycle raises error."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        graph.add_dependency("node_1", "node_2")  # Create cycle
        
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_sort()
    
    def test_get_critical_path(self):
        """Test getting critical path."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        graph.add_node("node_3", dependencies={"node_2"})
        graph.add_node("node_4", dependencies={"node_1"})  # Shorter path
        
        path, length = graph.get_critical_path()
        
        assert len(path) == 3
        assert length == 3
        assert path[0] == "node_1"
        assert path[-1] == "node_3"
    
    def test_analyze(self):
        """Test dependency graph analysis."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        graph.add_node("node_3")
        
        graph.mark_completed("node_1")
        
        analysis = graph.analyze()
        
        assert analysis.total_nodes == 3
        assert analysis.completed_nodes == 1
        assert analysis.ready_nodes >= 1
    
    def test_get_affected_nodes(self):
        """Test getting affected nodes."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        graph.add_node("node_3", dependencies={"node_2"})
        
        affected = graph.get_affected_nodes("node_1")
        
        assert len(affected) == 2
        assert "node_2" in affected
        assert "node_3" in affected
    
    def test_get_stats(self):
        """Test getting graph statistics."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        
        stats = graph.get_stats()
        
        assert "total_nodes" in stats
        assert stats["total_nodes"] == 2
        assert "has_cycle" in stats
        assert stats["has_cycle"] is False
    
    def test_clear(self):
        """Test clearing the graph."""
        graph = DependencyGraph()
        
        graph.add_node("node_1")
        graph.add_node("node_2", dependencies={"node_1"})
        graph.clear()
        
        assert len(graph._nodes) == 0
        assert len(graph._completed) == 0


class TestDependencyTracker:
    """Tests for DependencyTracker."""
    
    def test_tracker_creation(self):
        """Test creating a dependency tracker."""
        tracker = DependencyTracker()
        
        assert tracker is not None
        assert tracker.graph is not None
    
    def test_tracker_with_graph(self):
        """Test creating tracker with existing graph."""
        graph = DependencyGraph()
        tracker = DependencyTracker(graph=graph)
        
        assert tracker.graph is graph
    
    def test_register_callback(self):
        """Test registering a callback."""
        tracker = DependencyTracker()
        
        callback_called = []
        
        def callback(change):
            callback_called.append(change)
        
        tracker.register_callback(DependencyChangeType.COMPLETED, callback)
        
        assert len(tracker._callbacks[DependencyChangeType.COMPLETED]) == 1
    
    def test_notify_change(self):
        """Test notifying callbacks of a change."""
        tracker = DependencyTracker()
        
        callback_results = []
        
        def callback(change):
            callback_results.append(change)
        
        tracker.register_callback(DependencyChangeType.COMPLETED, callback)
        
        change = DependencyChange(
            node_id="node_1",
            change_type=DependencyChangeType.COMPLETED,
        )
        
        tracker.notify_change(change)
        
        assert len(callback_results) == 1
        assert callback_results[0].node_id == "node_1"
    
    def test_analyze_impact(self):
        """Test analyzing impact of a change."""
        tracker = DependencyTracker()
        
        tracker.graph.add_node("node_1")
        tracker.graph.add_node("node_2", dependencies={"node_1"})
        tracker.graph.add_node("node_3", dependencies={"node_2"})
        
        impact = tracker.analyze_impact("node_1")
        
        assert impact["changed_node"] == "node_1"
        assert len(impact["affected_nodes"]) == 2
        assert impact["affected_count"] == 2
    
    def test_get_incremental_updates(self):
        """Test getting incremental updates."""
        tracker = DependencyTracker()
        
        tracker.graph.add_node("node_1")
        tracker.graph.add_node("node_2", dependencies={"node_1"})
        tracker.graph.add_node("node_3", dependencies={"node_2"})
        
        newly_ready = tracker.get_incremental_updates("node_1")
        
        assert "node_2" in newly_ready
        assert "node_3" not in newly_ready  # Still depends on node_2
    
    def test_get_stats(self):
        """Test getting tracker statistics."""
        tracker = DependencyTracker()
        
        tracker.graph.add_node("node_1")
        tracker.graph.add_node("node_2", dependencies={"node_1"})
        
        def dummy_callback(change):
            pass
        
        tracker.register_callback(DependencyChangeType.COMPLETED, dummy_callback)
        
        stats = tracker.get_stats()
        
        assert "graph_stats" in stats
        assert "callback_counts" in stats
        assert stats["graph_stats"]["total_nodes"] == 2


class TestDependencyChange:
    """Tests for DependencyChange."""
    
    def test_change_creation(self):
        """Test creating a dependency change."""
        change = DependencyChange(
            node_id="node_1",
            change_type=DependencyChangeType.ADDED,
        )
        
        assert change.node_id == "node_1"
        assert change.change_type == DependencyChangeType.ADDED
        assert isinstance(change.timestamp, datetime)


class TestDependencyAnalysisResult:
    """Tests for DependencyAnalysisResult."""
    
    def test_result_creation(self):
        """Test creating an analysis result."""
        result = DependencyAnalysisResult(
            total_nodes=5,
            pending_nodes=3,
            ready_nodes=2,
            blocked_nodes=1,
            completed_nodes=2,
            failed_nodes=0,
        )
        
        assert result.total_nodes == 5
        assert result.pending_nodes == 3
        assert result.ready_nodes == 2


class TestDependencyChangeType:
    """Tests for DependencyChangeType enum."""
    
    def test_change_type_values(self):
        """Test change type enum values."""
        assert DependencyChangeType.ADDED.name == "ADDED"
        assert DependencyChangeType.REMOVED.name == "REMOVED"
        assert DependencyChangeType.UPDATED.name == "UPDATED"
        assert DependencyChangeType.COMPLETED.name == "COMPLETED"
        assert DependencyChangeType.FAILED.name == "FAILED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
