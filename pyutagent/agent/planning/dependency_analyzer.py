"""Dependency Analyzer Module.

Provides advanced dependency analysis for task planning:
- Dependency graph construction
- Topological sorting
- Cycle detection
- Resource conflict detection
- Execution group identification

This is part of Phase 3 Week 17-18: Task Planning Enhancement.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pyutagent.agent.execution.execution_plan import Step, StepType

logger = logging.getLogger(__name__)


@dataclass
class DependencyNode:
    """Node in a dependency graph."""
    step: Step
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    level: int = 0
    resource_type: Optional[str] = None
    estimated_duration: float = 0.0


class DependencyGraph:
    """Graph representing task dependencies."""

    def __init__(self):
        self._nodes: Dict[str, DependencyNode] = {}
        self._edges: List[Tuple[str, str]] = []

    def add_step(self, step: Step) -> None:
        """Add a step to the graph."""
        if step.id not in self._nodes:
            self._nodes[step.id] = DependencyNode(step=step)

    def add_dependency(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge (from_id must complete before to_id)."""
        if from_id in self._nodes and to_id in self._nodes:
            self._nodes[from_id].dependents.add(to_id)
            self._nodes[to_id].dependencies.add(from_id)
            self._edges.append((from_id, to_id))

    def get_dependencies(self, node_id: str) -> Set[str]:
        """Get dependencies of a node."""
        if node_id in self._nodes:
            return self._nodes[node_id].dependencies
        return set()

    def get_dependents(self, node_id: str) -> Set[str]:
        """Get dependents of a node."""
        if node_id in self._nodes:
            return self._nodes[node_id].dependents
        return set()

    def get_node(self, node_id: str) -> Optional[DependencyNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> List[DependencyNode]:
        """Get all nodes."""
        return list(self._nodes.values())

    def topological_sort(self) -> List[str]:
        """Perform topological sort.
        
        Returns:
            List of node IDs in topological order
        """
        in_degree = {node_id: len(node.dependencies) for node_id, node in self._nodes.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for dependent in self._nodes[node_id].dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return result

    def get_execution_groups(self) -> List[List[str]]:
        """Get groups of tasks that can be executed in parallel.
        
        Returns:
            List of groups, where each group can run in parallel
        """
        groups = []
        completed: Set[str] = set()
        remaining = set(self._nodes.keys())
        
        while remaining:
            ready = []
            for node_id in remaining:
                deps = self._nodes[node_id].dependencies
                if deps.issubset(completed):
                    ready.append(node_id)
            
            if not ready:
                break
            
            groups.append(ready)
            completed.update(ready)
            remaining -= set(ready)
        
        return groups

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph.
        
        Returns:
            List of cycles found
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for dependent in self._nodes[node_id].dependents:
                if dependent not in visited:
                    if dfs(dependent):
                        return True
                elif dependent in rec_stack:
                    cycle_start = path.index(dependent)
                    cycles.append(path[cycle_start:] + [dependent])
                    return True
            
            path.pop()
            rec_stack.remove(node_id)
            return False
        
        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)
        
        return cycles

    def get_critical_path(self) -> List[str]:
        """Get the critical path (longest path) in the graph.
        
        Returns:
            List of node IDs on the critical path
        """
        if not self._nodes:
            return []
        
        sorted_ids = self.topological_sort()
        earliest_finish = {}
        predecessor = {}
        
        for node_id in sorted_ids:
            node = self._nodes[node_id]
            deps = node.dependencies
            
            if not deps:
                earliest_finish[node_id] = node.estimated_duration
            else:
                max_dep_finish = max(earliest_finish.get(dep, 0) for dep in deps)
                earliest_finish[node_id] = max_dep_finish + node.estimated_duration
                max_dep = max(deps, key=lambda d: earliest_finish.get(d, 0))
                predecessor[node_id] = max_dep
        
        if not earliest_finish:
            return []
        
        end_node = max(earliest_finish, key=earliest_finish.get)
        
        path = [end_node]
        while end_node in predecessor:
            end_node = predecessor[end_node]
            path.append(end_node)
        
        return list(reversed(path))


class DependencyAnalyzer:
    """Base class for dependency analyzers."""

    def analyze(self, steps: List[Step]) -> DependencyGraph:
        """Analyze dependencies between steps.
        
        Args:
            steps: List of steps to analyze
            
        Returns:
            DependencyGraph with analyzed dependencies
        """
        graph = DependencyGraph()
        
        for step in steps:
            graph.add_step(step)
        
        for step in steps:
            for dep_id in step.dependencies:
                if dep_id in graph._nodes:
                    graph.add_dependency(dep_id, step.id)
        
        return graph


class AdvancedDependencyAnalyzer(DependencyAnalyzer):
    """Advanced dependency analyzer with semantic analysis."""

    def __init__(self):
        super().__init__()
        self._resource_keywords = {
            "file": ["read", "write", "file", "path"],
            "network": ["http", "api", "request", "fetch"],
            "database": ["query", "database", "db", "sql"],
            "llm": ["llm", "generate", "model", "ai"],
        }

    def analyze(self, steps: List[Step]) -> DependencyGraph:
        """Analyze with semantic dependency detection."""
        graph = super().analyze(steps)
        
        self._infer_implicit_dependencies(graph)
        self._assign_resource_types(graph)
        self._estimate_durations(graph)
        
        return graph

    def _infer_implicit_dependencies(self, graph: DependencyGraph) -> None:
        """Infer implicit dependencies based on step types."""
        nodes = graph.get_all_nodes()
        
        for node in nodes:
            if node.step.step_type == StepType.TEST:
                for other in nodes:
                    if other.step.step_type == StepType.ACTION:
                        if other.step.id not in node.dependencies:
                            graph.add_dependency(other.step.id, node.step.id)

    def _assign_resource_types(self, graph: DependencyGraph) -> None:
        """Assign resource types to nodes."""
        for node in graph.get_all_nodes():
            action_lower = node.step.description.lower()
            
            for resource_type, keywords in self._resource_keywords.items():
                if any(kw in action_lower for kw in keywords):
                    node.resource_type = resource_type
                    break

    def _estimate_durations(self, graph: DependencyGraph) -> None:
        """Estimate durations for nodes."""
        base_durations = {
            StepType.ANALYZE: 2.0,
            StepType.PLAN: 1.0,
            StepType.ACTION: 5.0,
            StepType.TEST: 3.0,
        }
        
        for node in graph.get_all_nodes():
            base = base_durations.get(node.step.step_type, 3.0)
            complexity_factor = len(node.step.description) / 100
            node.estimated_duration = base * (1 + complexity_factor * 0.1)

    def detect_resource_conflicts(self, graph: DependencyGraph) -> List[Tuple[str, str, str]]:
        """Detect resource conflicts between parallel tasks.
        
        Returns:
            List of (node1_id, node2_id, resource_type) tuples
        """
        conflicts = []
        groups = graph.get_execution_groups()
        
        for group in groups:
            resources: Dict[str, List[str]] = {}
            
            for node_id in group:
                node = graph.get_node(node_id)
                if node and node.resource_type:
                    if node.resource_type not in resources:
                        resources[node.resource_type] = []
                    resources[node.resource_type].append(node_id)
            
            for resource_type, node_ids in resources.items():
                if len(node_ids) > 1:
                    for i, id1 in enumerate(node_ids):
                        for id2 in node_ids[i + 1:]:
                            conflicts.append((id1, id2, resource_type))
        
        return conflicts

    def optimize_execution_order(
        self,
        graph: DependencyGraph,
        max_parallel: int = 4,
    ) -> List[List[str]]:
        """Optimize execution order considering parallelism and resources.
        
        Args:
            graph: Dependency graph
            max_parallel: Maximum parallel tasks
            
        Returns:
            Optimized execution groups
        """
        groups = graph.get_execution_groups()
        optimized = []
        
        for group in groups:
            if len(group) <= max_parallel:
                optimized.append(group)
            else:
                for i in range(0, len(group), max_parallel):
                    optimized.append(group[i:i + max_parallel])
        
        return optimized
