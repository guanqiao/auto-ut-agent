"""Dependency Tracker Module.

Provides dynamic dependency tracking with:
- Runtime dependency graph management
- Incremental dependency resolution
- Dependency change propagation
- Impact analysis

This is part of Phase 1 Week 1-2: Core Engine Enhancement.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DependencyChangeType(Enum):
    """Type of dependency change."""
    ADDED = auto()
    REMOVED = auto()
    UPDATED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class DependencyNode:
    """Node in the dependency graph.
    
    Attributes:
        id: Unique identifier
        dependencies: Set of dependency IDs this node depends on
        dependents: Set of nodes that depend on this node
        status: Node status (pending/running/completed/failed)
        metadata: Additional metadata
        updated_at: Last update timestamp
    """
    id: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_dependency(self, dep_id: str) -> None:
        """Add a dependency."""
        self.dependencies.add(dep_id)
        self.updated_at = datetime.now()
    
    def remove_dependency(self, dep_id: str) -> None:
        """Remove a dependency."""
        self.dependencies.discard(dep_id)
        self.updated_at = datetime.now()


@dataclass
class DependencyChange:
    """Represents a change in the dependency graph.
    
    Attributes:
        node_id: Affected node ID
        change_type: Type of change
        affected_nodes: Set of nodes affected by this change
        timestamp: Change timestamp
        metadata: Additional change metadata
    """
    node_id: str
    change_type: DependencyChangeType
    affected_nodes: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyAnalysisResult:
    """Result of dependency analysis.
    
    Attributes:
        total_nodes: Total number of nodes
        pending_nodes: Number of pending nodes
        ready_nodes: Number of ready nodes (no pending dependencies)
        blocked_nodes: Number of blocked nodes
        completed_nodes: Number of completed nodes
        failed_nodes: Number of failed nodes
        critical_path: List of nodes in critical path
        critical_path_length: Length of critical path
    """
    total_nodes: int
    pending_nodes: int
    ready_nodes: int
    blocked_nodes: int
    completed_nodes: int
    failed_nodes: int
    critical_path: List[str] = field(default_factory=list)
    critical_path_length: int = 0


class DependencyGraph:
    """Dynamic dependency graph for task dependencies.
    
    Provides:
    - Dependency graph construction
    - Topological sorting
    - Cycle detection
    - Ready task identification
    - Incremental updates
    """
    
    def __init__(self):
        """Initialize dependency graph."""
        self._nodes: Dict[str, DependencyNode] = {}
        self._completed: Set[str] = set()
        self._failed: Set[str] = set()
        self._change_history: List[DependencyChange] = []
    
    def add_node(self, node_id: str, dependencies: Optional[Set[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a node to the dependency graph.
        
        Args:
            node_id: Unique node identifier
            dependencies: Set of dependency IDs
            metadata: Additional metadata
        """
        if node_id in self._nodes:
            logger.warning(f"Node {node_id} already exists, updating")
        
        node = DependencyNode(
            id=node_id,
            dependencies=dependencies or set(),
            metadata=metadata or {},
        )
        
        self._nodes[node_id] = node
        
        # Update reverse dependencies
        for dep_id in node.dependencies:
            if dep_id in self._nodes:
                self._nodes[dep_id].dependents.add(node_id)
        
        logger.debug(f"Added node {node_id} with {len(node.dependencies)} dependencies")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the dependency graph.
        
        Args:
            node_id: Node to remove
            
        Returns:
            True if removed successfully
        """
        if node_id not in self._nodes:
            return False
        
        node = self._nodes[node_id]
        
        # Remove from dependents
        for dep_id in node.dependencies:
            if dep_id in self._nodes:
                self._nodes[dep_id].dependents.discard(node_id)
        
        # Remove from dependencies
        for dependent_id in node.dependents:
            if dependent_id in self._nodes:
                self._nodes[dependent_id].dependencies.discard(node_id)
        
        # Remove node
        del self._nodes[node_id]
        self._completed.discard(node_id)
        self._failed.discard(node_id)
        
        logger.debug(f"Removed node {node_id}")
        return True
    
    def add_dependency(self, node_id: str, dependency_id: str) -> bool:
        """Add a dependency between nodes.
        
        Args:
            node_id: Dependent node ID
            dependency_id: Dependency node ID
            
        Returns:
            True if added successfully
        """
        if node_id not in self._nodes:
            logger.error(f"Node {node_id} not found")
            return False
        
        if dependency_id not in self._nodes:
            logger.error(f"Dependency node {dependency_id} not found")
            return False
        
        node = self._nodes[node_id]
        node.add_dependency(dependency_id)
        
        # Update reverse dependency
        self._nodes[dependency_id].dependents.add(node_id)
        
        # Record change
        self._record_change(DependencyChange(
            node_id=node_id,
            change_type=DependencyChangeType.ADDED,
            affected_nodes={dependency_id},
        ))
        
        logger.debug(f"Added dependency: {node_id} -> {dependency_id}")
        return True
    
    def remove_dependency(self, node_id: str, dependency_id: str) -> bool:
        """Remove a dependency between nodes.
        
        Args:
            node_id: Dependent node ID
            dependency_id: Dependency node ID
            
        Returns:
            True if removed successfully
        """
        if node_id not in self._nodes:
            return False
        
        node = self._nodes[node_id]
        node.remove_dependency(dependency_id)
        
        # Update reverse dependency
        if dependency_id in self._nodes:
            self._nodes[dependency_id].dependents.discard(node_id)
        
        # Record change
        self._record_change(DependencyChange(
            node_id=node_id,
            change_type=DependencyChangeType.REMOVED,
            affected_nodes={dependency_id},
        ))
        
        logger.debug(f"Removed dependency: {node_id} -> {dependency_id}")
        return True
    
    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get dependencies of a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Set of dependency IDs
        """
        if node_id not in self._nodes:
            return set()
        return self._nodes[node_id].dependencies.copy()
    
    def get_dependents(self, node_id: str) -> Set[str]:
        """Get nodes that depend on this node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Set of dependent node IDs
        """
        if node_id not in self._nodes:
            return set()
        return self._nodes[node_id].dependents.copy()
    
    def get_ready_nodes(self) -> List[str]:
        """Get nodes that are ready to execute (no pending dependencies).
        
        Returns:
            List of ready node IDs
        """
        ready = []
        
        for node_id, node in self._nodes.items():
            if node.status != "pending":
                continue
            
            # Check if all dependencies are completed
            pending_deps = node.dependencies - self._completed
            if not pending_deps:
                ready.append(node_id)
        
        return ready
    
    def mark_completed(self, node_id: str) -> None:
        """Mark a node as completed.
        
        Args:
            node_id: Node to mark
        """
        if node_id not in self._nodes:
            return
        
        self._nodes[node_id].status = "completed"
        self._completed.add(node_id)
        
        # Record change
        affected = self._nodes[node_id].dependents.copy()
        self._record_change(DependencyChange(
            node_id=node_id,
            change_type=DependencyChangeType.COMPLETED,
            affected_nodes=affected,
        ))
        
        logger.debug(f"Marked node {node_id} as completed")
    
    def mark_failed(self, node_id: str) -> None:
        """Mark a node as failed.
        
        Args:
            node_id: Node to mark
        """
        if node_id not in self._nodes:
            return
        
        self._nodes[node_id].status = "failed"
        self._failed.add(node_id)
        
        # Record change
        affected = self._nodes[node_id].dependents.copy()
        self._record_change(DependencyChange(
            node_id=node_id,
            change_type=DependencyChangeType.FAILED,
            affected_nodes=affected,
        ))
        
        logger.debug(f"Marked node {node_id} as failed")
    
    def has_cycle(self) -> bool:
        """Check if the dependency graph has a cycle.
        
        Returns:
            True if cycle exists
        """
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self._nodes.get(node_id)
            if not node:
                return False
            
            for dep_id in node.dependencies:
                if dep_id not in visited:
                    if dfs(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self._nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        
        return False
    
    def topological_sort(self) -> List[str]:
        """Perform topological sort on the dependency graph.
        
        Returns:
            List of node IDs in topological order
        """
        if self.has_cycle():
            raise ValueError("Graph contains a cycle")
        
        in_degree = {node_id: len(node.dependencies) 
                     for node_id, node in self._nodes.items()}
        
        queue = deque([node_id for node_id, degree in in_degree.items() 
                       if degree == 0])
        
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            node = self._nodes[node_id]
            for dependent_id in node.dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)
        
        if len(result) != len(self._nodes):
            raise ValueError("Graph contains a cycle")
        
        return result
    
    def get_critical_path(self) -> Tuple[List[str], int]:
        """Get the critical path (longest path) in the dependency graph.
        
        Returns:
            Tuple of (critical path nodes, path length)
        """
        if not self._nodes:
            return [], 0
        
        try:
            sorted_nodes = self.topological_sort()
        except ValueError:
            return [], 0
        
        # Calculate longest path to each node
        dist = {node_id: 0 for node_id in self._nodes}
        parent = {node_id: None for node_id in self._nodes}
        
        for node_id in sorted_nodes:
            node = self._nodes[node_id]
            for dependent_id in node.dependents:
                if dist[dependent_id] < dist[node_id] + 1:
                    dist[dependent_id] = dist[node_id] + 1
                    parent[dependent_id] = node_id
        
        # Find node with maximum distance
        max_node = max(dist, key=dist.get)
        max_dist = dist[max_node]
        
        # Reconstruct path
        path = []
        current = max_node
        while current is not None:
            path.append(current)
            current = parent[current]
        
        path.reverse()
        
        return path, max_dist + 1
    
    def analyze(self) -> DependencyAnalysisResult:
        """Analyze the dependency graph.
        
        Returns:
            DependencyAnalysisResult with statistics
        """
        pending = 0
        ready = 0
        blocked = 0
        completed = len(self._completed)
        failed = len(self._failed)
        
        for node in self._nodes.values():
            if node.status == "pending":
                pending += 1
                pending_deps = node.dependencies - self._completed
                if not pending_deps:
                    ready += 1
                else:
                    blocked += 1
        
        critical_path, path_length = self.get_critical_path()
        
        return DependencyAnalysisResult(
            total_nodes=len(self._nodes),
            pending_nodes=pending,
            ready_nodes=ready,
            blocked_nodes=blocked,
            completed_nodes=completed,
            failed_nodes=failed,
            critical_path=critical_path,
            critical_path_length=path_length,
        )
    
    def get_affected_nodes(self, node_id: str) -> Set[str]:
        """Get all nodes affected by a change to this node.
        
        Uses BFS to find all transitive dependents.
        
        Args:
            node_id: Changed node ID
            
        Returns:
            Set of affected node IDs
        """
        affected = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            node = self._nodes.get(current)
            if not node:
                continue
            
            for dependent_id in node.dependents:
                if dependent_id not in affected:
                    affected.add(dependent_id)
                    queue.append(dependent_id)
        
        return affected
    
    def _record_change(self, change: DependencyChange) -> None:
        """Record a dependency change.
        
        Args:
            change: Change to record
        """
        self._change_history.append(change)
        
        # Keep last 1000 changes
        if len(self._change_history) > 1000:
            self._change_history = self._change_history[-1000:]
    
    def get_change_history(self, limit: int = 100) -> List[DependencyChange]:
        """Get recent change history.
        
        Args:
            limit: Maximum number of changes to return
            
        Returns:
            List of recent changes
        """
        return self._change_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dependency graph statistics.
        
        Returns:
            Dictionary with statistics
        """
        analysis = self.analyze()
        
        return {
            "total_nodes": analysis.total_nodes,
            "pending": analysis.pending_nodes,
            "ready": analysis.ready_nodes,
            "blocked": analysis.blocked_nodes,
            "completed": analysis.completed_nodes,
            "failed": analysis.failed_nodes,
            "critical_path_length": analysis.critical_path_length,
            "has_cycle": self.has_cycle(),
            "change_history_size": len(self._change_history),
        }
    
    def clear(self) -> None:
        """Clear the dependency graph."""
        self._nodes.clear()
        self._completed.clear()
        self._failed.clear()
        self._change_history.clear()


class DependencyTracker:
    """Tracker for monitoring and responding to dependency changes.
    
    Provides:
    - Change notification callbacks
    - Incremental dependency resolution
    - Impact analysis
    """
    
    def __init__(self, graph: Optional[DependencyGraph] = None):
        """Initialize DependencyTracker.
        
        Args:
            graph: DependencyGraph to track
        """
        self.graph = graph or DependencyGraph()
        self._callbacks: Dict[DependencyChangeType, List[callable]] = defaultdict(list)
    
    def register_callback(self, change_type: DependencyChangeType, 
                         callback: callable) -> None:
        """Register a callback for dependency changes.
        
        Args:
            change_type: Type of change to listen for
            callback: Callback function to invoke
        """
        self._callbacks[change_type].append(callback)
        logger.debug(f"Registered callback for {change_type.name}")
    
    def notify_change(self, change: DependencyChange) -> None:
        """Notify callbacks of a dependency change.
        
        Args:
            change: Change to notify about
        """
        callbacks = self._callbacks.get(change.change_type, [])
        
        for callback in callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def analyze_impact(self, node_id: str) -> Dict[str, Any]:
        """Analyze the impact of a change to a node.
        
        Args:
            node_id: Changed node ID
            
        Returns:
            Impact analysis result
        """
        affected = self.graph.get_affected_nodes(node_id)
        
        ready_count = 0
        blocked_count = 0
        
        for affected_id in affected:
            ready_nodes = self.graph.get_ready_nodes()
            if affected_id in ready_nodes:
                ready_count += 1
            else:
                blocked_count += 1
        
        return {
            "changed_node": node_id,
            "affected_nodes": affected,
            "affected_count": len(affected),
            "newly_ready": ready_count,
            "still_blocked": blocked_count,
        }
    
    def get_incremental_updates(self, completed_node_id: str) -> List[str]:
        """Get list of nodes that became ready after a completion.
        
        Args:
            completed_node_id: Node that just completed
            
        Returns:
            List of newly ready node IDs
        """
        # Mark as completed
        self.graph.mark_completed(completed_node_id)
        
        # Get affected nodes
        affected = self.graph.get_affected_nodes(completed_node_id)
        
        # Check which became ready
        newly_ready = []
        all_ready = set(self.graph.get_ready_nodes())
        
        for node_id in affected:
            if node_id in all_ready:
                newly_ready.append(node_id)
        
        return newly_ready
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "graph_stats": self.graph.get_stats(),
            "callback_counts": {
                change_type.name: len(callbacks)
                for change_type, callbacks in self._callbacks.items()
            },
        }
