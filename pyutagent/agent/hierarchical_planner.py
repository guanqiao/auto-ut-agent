"""Hierarchical Task Planner for decomposing complex tasks.

This module provides:
- HierarchicalTaskPlanner: Decompose tasks into subtasks
- TaskTree: Tree structure for task hierarchy
- DependencyGraph: Analyze task dependencies
- ExecutionPlan: Optimized execution order
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class SubtaskType(Enum):
    """Types of subtasks."""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    TESTING = "testing"
    FIXING = "fixing"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    REVIEW = "review"
    CUSTOM = "custom"


class SubtaskStatus(Enum):
    """Status of a subtask."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """A subtask in the task hierarchy."""
    id: str
    name: str
    description: str
    task_type: SubtaskType
    input_data: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5
    estimated_complexity: int = 1
    status: SubtaskStatus = SubtaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        task_type: SubtaskType,
        **kwargs
    ) -> "Subtask":
        """Create a new subtask."""
        return cls(
            id=str(uuid4()),
            name=name,
            description=description,
            task_type=task_type,
            **kwargs
        )


@dataclass
class TaskTree:
    """Tree structure for hierarchical tasks."""
    root_id: str
    task_description: str
    subtasks: Dict[str, Subtask] = field(default_factory=dict)
    parent_map: Dict[str, Optional[str]] = field(default_factory=dict)
    children_map: Dict[str, List[str]] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_subtask(
        self,
        subtask: Subtask,
        parent_id: Optional[str] = None
    ) -> None:
        """Add a subtask to the tree.

        Args:
            subtask: Subtask to add
            parent_id: Optional parent task ID
        """
        self.subtasks[subtask.id] = subtask
        self.parent_map[subtask.id] = parent_id

        if parent_id:
            if parent_id not in self.children_map:
                self.children_map[parent_id] = []
            self.children_map[parent_id].append(subtask.id)

    def get_subtask(self, subtask_id: str) -> Optional[Subtask]:
        """Get a subtask by ID."""
        return self.subtasks.get(subtask_id)

    def get_children(self, subtask_id: str) -> List[Subtask]:
        """Get children of a subtask."""
        child_ids = self.children_map.get(subtask_id, [])
        return [self.subtasks[cid] for cid in child_ids if cid in self.subtasks]

    def get_parent(self, subtask_id: str) -> Optional[Subtask]:
        """Get parent of a subtask."""
        parent_id = self.parent_map.get(subtask_id)
        if parent_id:
            return self.subtasks.get(parent_id)
        return None

    def get_root_subtasks(self) -> List[Subtask]:
        """Get subtasks at the root level (no parent)."""
        return [
            self.subtasks[sid]
            for sid, pid in self.parent_map.items()
            if pid is None or pid == self.root_id
        ]

    def get_all_subtasks(self) -> List[Subtask]:
        """Get all subtasks."""
        return list(self.subtasks.values())

    def get_pending_subtasks(self) -> List[Subtask]:
        """Get all pending subtasks."""
        return [
            s for s in self.subtasks.values()
            if s.status == SubtaskStatus.PENDING
        ]

    def get_ready_subtasks(self) -> List[Subtask]:
        """Get subtasks ready to execute (dependencies met)."""
        ready = []
        for subtask in self.subtasks.values():
            if subtask.status != SubtaskStatus.PENDING:
                continue

            deps_met = all(
                self.subtasks.get(dep_id, Subtask(
                    id="", name="", description="", task_type=SubtaskType.CUSTOM
                )).status == SubtaskStatus.COMPLETED
                for dep_id in subtask.dependencies
            )

            if deps_met:
                ready.append(subtask)

        return ready

    def update_status(self, subtask_id: str, status: SubtaskStatus) -> None:
        """Update subtask status."""
        if subtask_id in self.subtasks:
            self.subtasks[subtask_id].status = status
            if status == SubtaskStatus.RUNNING:
                self.subtasks[subtask_id].started_at = datetime.now().isoformat()
            elif status in [SubtaskStatus.COMPLETED, SubtaskStatus.FAILED]:
                self.subtasks[subtask_id].completed_at = datetime.now().isoformat()

    def is_complete(self) -> bool:
        """Check if all subtasks are complete."""
        return all(
            s.status in [SubtaskStatus.COMPLETED, SubtaskStatus.SKIPPED]
            for s in self.subtasks.values()
        )

    def get_progress(self) -> float:
        """Get completion progress (0.0 - 1.0)."""
        if not self.subtasks:
            return 0.0

        completed = sum(
            1 for s in self.subtasks.values()
            if s.status == SubtaskStatus.COMPLETED
        )
        return completed / len(self.subtasks)


@dataclass
class DependencyEdge:
    """Edge in the dependency graph."""
    from_id: str
    to_id: str
    dependency_type: str = "sequential"


@dataclass
class DependencyGraph:
    """Graph of task dependencies."""
    nodes: Set[str] = field(default_factory=set)
    edges: List[DependencyEdge] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    reverse_adjacency: Dict[str, List[str]] = field(default_factory=dict)

    def add_node(self, node_id: str) -> None:
        """Add a node to the graph."""
        self.nodes.add(node_id)
        if node_id not in self.adjacency:
            self.adjacency[node_id] = []
        if node_id not in self.reverse_adjacency:
            self.reverse_adjacency[node_id] = []

    def add_edge(self, from_id: str, to_id: str, dependency_type: str = "sequential") -> None:
        """Add a dependency edge (from_id must complete before to_id)."""
        self.add_node(from_id)
        self.add_node(to_id)

        edge = DependencyEdge(from_id=from_id, to_id=to_id, dependency_type=dependency_type)
        self.edges.append(edge)

        self.adjacency[from_id].append(to_id)
        self.reverse_adjacency[to_id].append(from_id)

    def get_dependencies(self, node_id: str) -> List[str]:
        """Get nodes that this node depends on."""
        return self.reverse_adjacency.get(node_id, [])

    def get_dependents(self, node_id: str) -> List[str]:
        """Get nodes that depend on this node."""
        return self.adjacency.get(node_id, [])

    def get_roots(self) -> List[str]:
        """Get nodes with no dependencies."""
        return [
            node_id for node_id in self.nodes
            if not self.reverse_adjacency.get(node_id)
        ]

    def get_leaves(self) -> List[str]:
        """Get nodes with no dependents."""
        return [
            node_id for node_id in self.nodes
            if not self.adjacency.get(node_id)
        ]

    def topological_sort(self) -> List[str]:
        """Get topological order of nodes."""
        in_degree = {node: len(self.reverse_adjacency.get(node, [])) for node in self.nodes}

        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for dependent in self.adjacency.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self.nodes):
            logger.warning("[DependencyGraph] Cycle detected in dependencies")

        return result

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node)

        for node in self.nodes:
            if node not in visited:
                dfs(node)

        return cycles


@dataclass
class ExecutionPlan:
    """Optimized execution plan for subtasks."""
    task_tree: TaskTree
    dependency_graph: DependencyGraph
    execution_order: List[str]
    parallel_groups: List[List[str]]
    estimated_duration: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_next_tasks(self, completed: Set[str]) -> List[str]:
        """Get the next tasks to execute based on completed tasks.

        Args:
            completed: Set of completed task IDs

        Returns:
            List of task IDs ready to execute
        """
        ready = []
        for task_id in self.execution_order:
            if task_id in completed:
                continue

            dependencies = self.dependency_graph.get_dependencies(task_id)
            if all(dep in completed for dep in dependencies):
                ready.append(task_id)

        return ready

    def get_parallel_group(self, group_index: int) -> List[str]:
        """Get a parallel execution group."""
        if 0 <= group_index < len(self.parallel_groups):
            return self.parallel_groups[group_index]
        return []


class HierarchicalTaskPlanner:
    """Planner for decomposing complex tasks into hierarchical subtasks.

    Features:
    - LLM-powered task decomposition
    - Dependency analysis
    - Parallel task identification
    - Execution order optimization
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_depth: int = 3,
        max_subtasks: int = 20
    ):
        """Initialize the planner.

        Args:
            llm_client: Optional LLM client for decomposition
            max_depth: Maximum decomposition depth
            max_subtasks: Maximum number of subtasks
        """
        self.llm_client = llm_client
        self.max_depth = max_depth
        self.max_subtasks = max_subtasks

        self._decomposition_history: List[TaskTree] = []

        logger.info(f"[HierarchicalTaskPlanner] Initialized (max_depth={max_depth})")

    async def decompose(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskTree:
        """Decompose a task into subtasks.

        Args:
            task: Task description
            context: Optional execution context

        Returns:
            TaskTree with decomposed subtasks
        """
        context = context or {}
        root_id = str(uuid4())

        tree = TaskTree(
            root_id=root_id,
            task_description=task
        )

        if self.llm_client:
            subtasks = await self._decompose_with_llm(task, context)
        else:
            subtasks = self._decompose_with_rules(task, context)

        for subtask_data in subtasks[:self.max_subtasks]:
            subtask = Subtask.create(
                name=subtask_data.get("name", "Unnamed subtask"),
                description=subtask_data.get("description", ""),
                task_type=SubtaskType(subtask_data.get("type", "custom")),
                input_data=subtask_data.get("input_data", {}),
                dependencies=subtask_data.get("dependencies", []),
                priority=subtask_data.get("priority", 5),
                estimated_complexity=subtask_data.get("complexity", 1)
            )
            tree.add_subtask(subtask, parent_id=root_id)

        self._decomposition_history.append(tree)

        logger.info(f"[HierarchicalTaskPlanner] Decomposed task into {len(tree.subtasks)} subtasks")

        return tree

    async def _decompose_with_llm(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Use LLM to decompose task.

        Args:
            task: Task description
            context: Execution context

        Returns:
            List of subtask definitions
        """
        prompt = f"""Decompose the following task into subtasks:

Task: {task}

Context:
{self._format_context(context)}

Return a JSON array of subtasks, each with:
- name: Short name for the subtask
- description: Detailed description
- type: One of "analysis", "generation", "testing", "fixing", "refactoring", "documentation", "review", "custom"
- input_data: Dict of input parameters
- dependencies: List of indices of subtasks this depends on (0-indexed)
- priority: 1-10 (lower is higher priority)
- complexity: 1-5

Only return the JSON array, no other text."""

        try:
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            content = response.content if hasattr(response, 'content') else str(response)

            import json
            start = content.find('[')
            end = content.rfind(']')

            if start != -1 and end != -1:
                subtasks = json.loads(content[start:end+1])
                return subtasks

        except Exception as e:
            logger.error(f"[HierarchicalTaskPlanner] LLM decomposition failed: {e}")

        return self._decompose_with_rules(task, context)

    def _decompose_with_rules(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Use rules to decompose task.

        Args:
            task: Task description
            context: Execution context

        Returns:
            List of subtask definitions
        """
        task_lower = task.lower()
        subtasks = []

        if any(word in task_lower for word in ["test", "测试"]):
            subtasks = self._create_test_subtasks(task, context)
        elif any(word in task_lower for word in ["fix", "bug", "修复"]):
            subtasks = self._create_fix_subtasks(task, context)
        elif any(word in task_lower for word in ["refactor", "重构"]):
            subtasks = self._create_refactor_subtasks(task, context)
        elif any(word in task_lower for word in ["document", "文档"]):
            subtasks = self._create_doc_subtasks(task, context)
        else:
            subtasks = self._create_generic_subtasks(task, context)

        return subtasks

    def _create_test_subtasks(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create subtasks for testing tasks."""
        return [
            {
                "name": "analyze_target",
                "description": f"Analyze the target code for test generation",
                "type": "analysis",
                "input_data": {"task": task},
                "dependencies": [],
                "priority": 1,
                "complexity": 2
            },
            {
                "name": "generate_tests",
                "description": f"Generate unit tests based on analysis",
                "type": "generation",
                "input_data": {"task": task},
                "dependencies": [0],
                "priority": 2,
                "complexity": 3
            },
            {
                "name": "run_tests",
                "description": f"Run generated tests and check results",
                "type": "testing",
                "input_data": {"task": task},
                "dependencies": [1],
                "priority": 3,
                "complexity": 2
            },
            {
                "name": "fix_failures",
                "description": f"Fix any test failures",
                "type": "fixing",
                "input_data": {"task": task},
                "dependencies": [2],
                "priority": 4,
                "complexity": 3
            }
        ]

    def _create_fix_subtasks(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create subtasks for bug fixing tasks."""
        return [
            {
                "name": "analyze_error",
                "description": f"Analyze the error or bug",
                "type": "analysis",
                "input_data": {"task": task},
                "dependencies": [],
                "priority": 1,
                "complexity": 2
            },
            {
                "name": "locate_issue",
                "description": f"Locate the source of the issue",
                "type": "analysis",
                "input_data": {"task": task},
                "dependencies": [0],
                "priority": 2,
                "complexity": 3
            },
            {
                "name": "apply_fix",
                "description": f"Apply the fix",
                "type": "fixing",
                "input_data": {"task": task},
                "dependencies": [1],
                "priority": 3,
                "complexity": 3
            },
            {
                "name": "verify_fix",
                "description": f"Verify the fix works",
                "type": "testing",
                "input_data": {"task": task},
                "dependencies": [2],
                "priority": 4,
                "complexity": 2
            }
        ]

    def _create_refactor_subtasks(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create subtasks for refactoring tasks."""
        return [
            {
                "name": "analyze_code",
                "description": f"Analyze code structure and identify refactoring opportunities",
                "type": "analysis",
                "input_data": {"task": task},
                "dependencies": [],
                "priority": 1,
                "complexity": 2
            },
            {
                "name": "plan_refactor",
                "description": f"Plan the refactoring steps",
                "type": "analysis",
                "input_data": {"task": task},
                "dependencies": [0],
                "priority": 2,
                "complexity": 2
            },
            {
                "name": "execute_refactor",
                "description": f"Execute the refactoring",
                "type": "refactoring",
                "input_data": {"task": task},
                "dependencies": [1],
                "priority": 3,
                "complexity": 4
            },
            {
                "name": "verify_behavior",
                "description": f"Verify behavior is preserved",
                "type": "testing",
                "input_data": {"task": task},
                "dependencies": [2],
                "priority": 4,
                "complexity": 2
            }
        ]

    def _create_doc_subtasks(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create subtasks for documentation tasks."""
        return [
            {
                "name": "analyze_code",
                "description": f"Analyze code to document",
                "type": "analysis",
                "input_data": {"task": task},
                "dependencies": [],
                "priority": 1,
                "complexity": 2
            },
            {
                "name": "generate_docs",
                "description": f"Generate documentation",
                "type": "documentation",
                "input_data": {"task": task},
                "dependencies": [0],
                "priority": 2,
                "complexity": 3
            },
            {
                "name": "review_docs",
                "description": f"Review generated documentation",
                "type": "review",
                "input_data": {"task": task},
                "dependencies": [1],
                "priority": 3,
                "complexity": 1
            }
        ]

    def _create_generic_subtasks(self, task: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create generic subtasks."""
        return [
            {
                "name": "analyze_task",
                "description": f"Analyze the task requirements",
                "type": "analysis",
                "input_data": {"task": task},
                "dependencies": [],
                "priority": 1,
                "complexity": 2
            },
            {
                "name": "execute_task",
                "description": f"Execute the main task",
                "type": "custom",
                "input_data": {"task": task},
                "dependencies": [0],
                "priority": 2,
                "complexity": 3
            },
            {
                "name": "verify_result",
                "description": f"Verify the result",
                "type": "review",
                "input_data": {"task": task},
                "dependencies": [1],
                "priority": 3,
                "complexity": 1
            }
        ]

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        lines = []
        for key, value in context.items():
            if isinstance(value, str):
                lines.append(f"- {key}: {value[:200]}")
            elif isinstance(value, dict):
                lines.append(f"- {key}: {str(value)[:200]}")
            else:
                lines.append(f"- {key}: {str(value)[:100]}")
        return "\n".join(lines) if lines else "No additional context"

    def analyze_dependencies(self, subtasks: List[Subtask]) -> DependencyGraph:
        """Analyze dependencies between subtasks.

        Args:
            subtasks: List of subtasks

        Returns:
            DependencyGraph
        """
        graph = DependencyGraph()

        subtask_map = {s.id: s for s in subtasks}

        for subtask in subtasks:
            graph.add_node(subtask.id)

            for dep_id in subtask.dependencies:
                if dep_id in subtask_map:
                    graph.add_edge(dep_id, subtask.id)

        return graph

    def identify_parallel_tasks(self, graph: DependencyGraph) -> List[List[str]]:
        """Identify groups of tasks that can run in parallel.

        Args:
            graph: Dependency graph

        Returns:
            List of parallel task groups
        """
        parallel_groups = []
        completed = set()
        remaining = set(graph.nodes)

        while remaining:
            ready = []
            for node in remaining:
                deps = graph.get_dependencies(node)
                if all(dep in completed for dep in deps):
                    ready.append(node)

            if not ready:
                if remaining:
                    ready = [list(remaining)[0]]
                else:
                    break

            parallel_groups.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return parallel_groups

    def optimize_execution_order(self, tree: TaskTree) -> ExecutionPlan:
        """Create an optimized execution plan.

        Args:
            tree: Task tree

        Returns:
            ExecutionPlan
        """
        subtasks = tree.get_all_subtasks()
        graph = self.analyze_dependencies(subtasks)

        execution_order = graph.topological_sort()

        parallel_groups = self.identify_parallel_tasks(graph)

        estimated_duration = sum(
            s.estimated_complexity for s in subtasks
        ) / max(len(parallel_groups), 1)

        plan = ExecutionPlan(
            task_tree=tree,
            dependency_graph=graph,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            estimated_duration=estimated_duration
        )

        logger.info(f"[HierarchicalTaskPlanner] Created execution plan with "
                   f"{len(execution_order)} tasks in {len(parallel_groups)} parallel groups")

        return plan

    def get_decomposition_history(self) -> List[TaskTree]:
        """Get history of decompositions."""
        return self._decomposition_history.copy()


def create_hierarchical_planner(
    llm_client: Optional[Any] = None,
    max_depth: int = 3
) -> HierarchicalTaskPlanner:
    """Create a HierarchicalTaskPlanner.

    Args:
        llm_client: Optional LLM client
        max_depth: Maximum decomposition depth

    Returns:
        HierarchicalTaskPlanner instance
    """
    return HierarchicalTaskPlanner(
        llm_client=llm_client,
        max_depth=max_depth
    )
