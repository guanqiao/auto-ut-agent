"""Dynamic tool orchestration for flexible agent workflows.

This module provides intelligent tool orchestration:
- Automatic tool sequence planning
- Dynamic plan adaptation
- Tool dependency management
- Conditional execution
- Parallel tool execution
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


class ToolState(Enum):
    """States of a tool in execution."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


class PlanState(Enum):
    """States of an execution plan."""
    CREATED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ADAPTING = auto()
    CANCELLED = auto()


@dataclass
class ToolDefinition:
    """Definition of a tool for orchestration."""
    name: str
    description: str
    category: str
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    estimated_duration: float = 1.0
    retry_count: int = 3
    timeout: float = 60.0
    can_run_parallel: bool = True
    required_inputs: List[str] = field(default_factory=list)
    provided_outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A scheduled tool call."""
    tool_name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    state: ToolState = ToolState.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempt: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class ExecutionPlan:
    """A plan for tool execution."""
    id: str
    goal: str
    steps: List[ToolCall]
    state: PlanState = PlanState.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.state == ToolState.COMPLETED)
    
    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.state == ToolState.FAILED)
    
    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        terminal_states = {ToolState.COMPLETED, ToolState.FAILED, ToolState.SKIPPED}
        completed = sum(1 for s in self.steps if s.state in terminal_states)
        return completed / len(self.steps)


@dataclass
class OrchestrationResult:
    """Result of tool orchestration."""
    success: bool
    plan: ExecutionPlan
    results: Dict[str, Any]
    errors: List[Exception] = field(default_factory=list)
    total_time: float = 0.0
    adaptations: int = 0
    message: str = ""


class DependencyGraph:
    """Graph for managing tool dependencies."""
    
    def __init__(self):
        self.nodes: Dict[str, ToolDefinition] = {}
        self.edges: Dict[str, Set[str]] = {}
    
    def add_tool(self, tool: ToolDefinition):
        """Add a tool to the graph."""
        self.nodes[tool.name] = tool
        if tool.name not in self.edges:
            self.edges[tool.name] = set()
    
    def build_dependencies(self):
        """Build dependency edges."""
        for name, tool in self.nodes.items():
            for dep in tool.dependencies:
                if dep in self.nodes:
                    self.edges[name].add(dep)
    
    def get_execution_order(self) -> List[List[str]]:
        """Get tools grouped by execution level (parallelizable groups)."""
        visited: Set[str] = set()
        levels: List[List[str]] = []
        
        remaining = set(self.nodes.keys())
        
        while remaining:
            level = []
            for tool_name in list(remaining):
                deps = self.edges.get(tool_name, set())
                if deps.issubset(visited):
                    level.append(tool_name)
            
            if not level:
                break
            
            levels.append(level)
            visited.update(level)
            remaining -= set(level)
        
        return levels
    
    def get_dependencies(self, tool_name: str) -> Set[str]:
        """Get all dependencies of a tool (transitive)."""
        deps = set()
        to_visit = list(self.edges.get(tool_name, set()))
        
        while to_visit:
            dep = to_visit.pop()
            if dep not in deps:
                deps.add(dep)
                to_visit.extend(self.edges.get(dep, set()))
        
        return deps
    
    def get_dependents(self, tool_name: str) -> Set[str]:
        """Get all tools that depend on this tool."""
        dependents = set()
        for name, deps in self.edges.items():
            if tool_name in deps:
                dependents.add(name)
        return dependents
    
    def has_cycle(self) -> bool:
        """Check if the graph has a cycle."""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for dep in self.edges.get(node, set()):
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False


class ToolOrchestrator:
    """Orchestrates tool execution with intelligent planning.
    
    Features:
    - Automatic tool sequence planning
    - Dynamic plan adaptation
    - Parallel execution support
    - Dependency management
    - Error recovery
    """
    
    def __init__(
        self,
        tools: Optional[Dict[str, Callable]] = None,
        tool_definitions: Optional[Dict[str, ToolDefinition]] = None,
        max_parallel: int = 3,
        adaptation_enabled: bool = True
    ):
        self.tools = tools or {}
        self.tool_definitions = tool_definitions or {}
        self.max_parallel = max_parallel
        self.adaptation_enabled = adaptation_enabled
        
        self._dependency_graph = DependencyGraph()
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._execution_history: List[ExecutionPlan] = []
        
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tool definitions."""
        default_definitions = {
            "parse_code": ToolDefinition(
                name="parse_code",
                description="Parse source code to extract structure",
                category="analysis",
                dependencies=[],
                estimated_duration=0.5,
                can_run_parallel=False,
                provided_outputs=["class_info", "source_code"]
            ),
            "generate_tests": ToolDefinition(
                name="generate_tests",
                description="Generate unit tests using LLM",
                category="generation",
                dependencies=["parse_code"],
                estimated_duration=30.0,
                can_run_parallel=False,
                required_inputs=["class_info"],
                provided_outputs=["test_code", "test_file"]
            ),
            "compile_tests": ToolDefinition(
                name="compile_tests",
                description="Compile generated tests",
                category="build",
                dependencies=["generate_tests"],
                estimated_duration=10.0,
                can_run_parallel=False,
                required_inputs=["test_file"],
                provided_outputs=["compilation_result"]
            ),
            "run_tests": ToolDefinition(
                name="run_tests",
                description="Run compiled tests",
                category="execution",
                dependencies=["compile_tests"],
                estimated_duration=15.0,
                can_run_parallel=False,
                required_inputs=["test_file"],
                provided_outputs=["test_results", "failures"]
            ),
            "analyze_coverage": ToolDefinition(
                name="analyze_coverage",
                description="Analyze test coverage",
                category="analysis",
                dependencies=["run_tests"],
                estimated_duration=5.0,
                can_run_parallel=False,
                required_inputs=["test_results"],
                provided_outputs=["coverage_report", "line_coverage"]
            ),
            "fix_compilation": ToolDefinition(
                name="fix_compilation",
                description="Fix compilation errors",
                category="fix",
                dependencies=["compile_tests"],
                conflicts=["generate_tests"],
                estimated_duration=20.0,
                can_run_parallel=False,
                required_inputs=["test_code", "compilation_errors"],
                provided_outputs=["fixed_code"]
            ),
            "fix_tests": ToolDefinition(
                name="fix_tests",
                description="Fix failing tests",
                category="fix",
                dependencies=["run_tests"],
                conflicts=["generate_tests"],
                estimated_duration=20.0,
                can_run_parallel=False,
                required_inputs=["test_code", "test_failures"],
                provided_outputs=["fixed_code"]
            ),
            "generate_additional_tests": ToolDefinition(
                name="generate_additional_tests",
                description="Generate additional tests for coverage",
                category="generation",
                dependencies=["analyze_coverage"],
                estimated_duration=25.0,
                can_run_parallel=False,
                required_inputs=["class_info", "existing_tests", "uncovered_lines"],
                provided_outputs=["additional_test_code"]
            ),
        }
        
        for name, definition in default_definitions.items():
            if name not in self.tool_definitions:
                self.tool_definitions[name] = definition
        
        self._build_dependency_graph()
    
    def _build_dependency_graph(self):
        """Build the dependency graph from tool definitions."""
        self._dependency_graph = DependencyGraph()
        for tool_def in self.tool_definitions.values():
            self._dependency_graph.add_tool(tool_def)
        self._dependency_graph.build_dependencies()
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        definition: Optional[ToolDefinition] = None
    ):
        """Register a tool with optional definition."""
        self.tools[name] = func
        if definition:
            self.tool_definitions[name] = definition
            self._build_dependency_graph()
    
    def plan_tool_sequence(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """Plan a tool sequence based on goal and context.
        
        Args:
            goal: The goal to achieve
            context: Current context and available data
            constraints: Constraints on execution
            
        Returns:
            ExecutionPlan with scheduled tool calls
        """
        import uuid
        
        context = context or {}
        constraints = constraints or {}
        
        plan_id = str(uuid.uuid4())[:8]
        
        required_tools = self._determine_required_tools(goal, context)
        
        ordered_tools = self._order_tools_by_dependencies(required_tools)
        
        steps = []
        for tool_name in ordered_tools:
            tool_call = ToolCall(
                tool_name=tool_name,
                kwargs=self._prepare_tool_kwargs(tool_name, context),
                state=ToolState.PENDING
            )
            steps.append(tool_call)
        
        plan = ExecutionPlan(
            id=plan_id,
            goal=goal,
            steps=steps,
            state=PlanState.CREATED,
            metadata={
                "context": context,
                "constraints": constraints,
                "created_by": "planner"
            }
        )
        
        self._active_plans[plan_id] = plan
        
        logger.info(
            f"[ToolOrchestrator] Created plan {plan_id} with {len(steps)} steps for goal: {goal}"
        )
        
        return plan
    
    def _determine_required_tools(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Determine which tools are needed for the goal."""
        goal_lower = goal.lower()
        required = []
        
        if "test" in goal_lower or "coverage" in goal_lower:
            required = ["parse_code", "generate_tests", "compile_tests", "run_tests"]
            
            if "coverage" in goal_lower:
                required.append("analyze_coverage")
        
        elif "parse" in goal_lower or "analyze" in goal_lower:
            required = ["parse_code"]
        
        elif "fix" in goal_lower:
            if "compilation" in goal_lower:
                required = ["fix_compilation"]
            elif "test" in goal_lower:
                required = ["fix_tests"]
        
        else:
            required = ["parse_code", "generate_tests", "compile_tests", "run_tests"]
        
        if context.get("has_compilation_errors"):
            if "fix_compilation" not in required:
                required.append("fix_compilation")
        
        if context.get("has_test_failures"):
            if "fix_tests" not in required:
                required.append("fix_tests")
        
        if context.get("coverage_below_target"):
            if "analyze_coverage" not in required:
                required.append("analyze_coverage")
            if "generate_additional_tests" not in required:
                required.append("generate_additional_tests")
        
        return required
    
    def _order_tools_by_dependencies(
        self,
        tool_names: List[str]
    ) -> List[str]:
        """Order tools respecting dependencies."""
        ordered = []
        added = set()
        
        def add_with_deps(tool_name: str):
            if tool_name in added:
                return
            
            if tool_name not in self.tool_definitions:
                logger.warning(f"[ToolOrchestrator] Unknown tool: {tool_name}")
                ordered.append(tool_name)
                added.add(tool_name)
                return
            
            tool_def = self.tool_definitions[tool_name]
            
            for dep in tool_def.dependencies:
                if dep in tool_names:
                    add_with_deps(dep)
            
            ordered.append(tool_name)
            added.add(tool_name)
        
        for tool_name in tool_names:
            add_with_deps(tool_name)
        
        return ordered
    
    def _prepare_tool_kwargs(
        self,
        tool_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare kwargs for a tool call."""
        kwargs = {}
        
        if tool_name not in self.tool_definitions:
            return kwargs
        
        tool_def = self.tool_definitions[tool_name]
        
        for input_name in tool_def.required_inputs:
            if input_name in context:
                kwargs[input_name] = context[input_name]
        
        return kwargs
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> OrchestrationResult:
        """Execute an execution plan.
        
        Args:
            plan: The plan to execute
            on_progress: Optional progress callback
            
        Returns:
            OrchestrationResult with execution results
        """
        plan.state = PlanState.RUNNING
        plan.started_at = datetime.now()
        
        results: Dict[str, Any] = {}
        errors: List[Exception] = []
        adaptations = 0
        
        context = plan.metadata.get("context", {})
        
        logger.info(f"[ToolOrchestrator] Starting execution of plan {plan.id}")
        
        try:
            for i, step in enumerate(plan.steps):
                if plan.state == PlanState.CANCELLED:
                    break
                
                step.state = ToolState.RUNNING
                step.start_time = datetime.now()
                step.attempt += 1
                
                try:
                    tool_func = self.tools.get(step.tool_name)
                    
                    if not tool_func:
                        raise ValueError(f"Tool not found: {step.tool_name}")
                    
                    merged_kwargs = {**step.kwargs}
                    for input_name in self.tool_definitions.get(step.tool_name, ToolDefinition(name="", description="", category="")).required_inputs:
                        if input_name in results:
                            merged_kwargs[input_name] = results[input_name]
                    
                    if asyncio.iscoroutinefunction(tool_func):
                        result = await tool_func(*step.args, **merged_kwargs)
                    else:
                        result = tool_func(*step.args, **merged_kwargs)
                    
                    step.result = result
                    step.state = ToolState.COMPLETED
                    results[step.tool_name] = result
                    
                    if self.tool_definitions.get(step.tool_name):
                        for output_name in self.tool_definitions[step.tool_name].provided_outputs:
                            if isinstance(result, dict) and output_name in result:
                                results[output_name] = result[output_name]
                            else:
                                results[output_name] = result
                    
                    logger.info(
                        f"[ToolOrchestrator] Step {i+1}/{len(plan.steps)} completed: {step.tool_name}"
                    )
                    
                except Exception as e:
                    step.error = e
                    step.state = ToolState.FAILED
                    errors.append(e)
                    
                    logger.warning(
                        f"[ToolOrchestrator] Step {i+1}/{len(plan.steps)} failed: {step.tool_name} - {e}"
                    )
                    
                    if self.adaptation_enabled:
                        adapted = await self._adapt_plan(plan, step, e, results)
                        if adapted:
                            adaptations += 1
                            logger.info(f"[ToolOrchestrator] Plan adapted, continuing execution")
                            continue
                    
                    if self._is_critical_step(step):
                        plan.state = PlanState.FAILED
                        break
                
                finally:
                    step.end_time = datetime.now()
                
                if on_progress:
                    on_progress(plan.progress)
            
            if plan.state == PlanState.RUNNING:
                plan.state = PlanState.COMPLETED
            
        except Exception as e:
            plan.state = PlanState.FAILED
            errors.append(e)
            logger.exception(f"[ToolOrchestrator] Plan execution failed: {e}")
        
        finally:
            plan.completed_at = datetime.now()
            self._execution_history.append(plan)
            self._active_plans.pop(plan.id, None)
        
        success = plan.state == PlanState.COMPLETED and plan.failed_steps == 0
        
        return OrchestrationResult(
            success=success,
            plan=plan,
            results=results,
            errors=errors,
            total_time=plan.duration,
            adaptations=adaptations,
            message=f"Plan {'completed successfully' if success else 'failed'}"
        )
    
    async def _adapt_plan(
        self,
        plan: ExecutionPlan,
        failed_step: ToolCall,
        error: Exception,
        results: Dict[str, Any]
    ) -> bool:
        """Adapt the plan after a step failure.
        
        Args:
            plan: The current plan
            failed_step: The step that failed
            error: The error that occurred
            results: Results from completed steps
            
        Returns:
            True if plan was adapted successfully
        """
        plan.state = PlanState.ADAPTING
        
        tool_name = failed_step.tool_name
        
        if "compilation" in tool_name.lower() or "compile" in str(error).lower():
            fix_step = ToolCall(
                tool_name="fix_compilation",
                kwargs={
                    "test_code": results.get("test_code"),
                    "compilation_errors": str(error)
                },
                state=ToolState.PENDING
            )
            
            insert_idx = plan.steps.index(failed_step) + 1
            plan.steps.insert(insert_idx, fix_step)
            
            logger.info(f"[ToolOrchestrator] Added fix_compilation step to plan")
            return True
        
        if "test" in tool_name.lower() and "fail" in str(error).lower():
            fix_step = ToolCall(
                tool_name="fix_tests",
                kwargs={
                    "test_code": results.get("test_code"),
                    "test_failures": results.get("failures", [])
                },
                state=ToolState.PENDING
            )
            
            insert_idx = plan.steps.index(failed_step) + 1
            plan.steps.insert(insert_idx, fix_step)
            
            logger.info(f"[ToolOrchestrator] Added fix_tests step to plan")
            return True
        
        if failed_step.attempt < 3:
            retry_step = ToolCall(
                tool_name=tool_name,
                args=failed_step.args,
                kwargs=failed_step.kwargs,
                state=ToolState.PENDING,
                attempt=failed_step.attempt
            )
            
            insert_idx = plan.steps.index(failed_step) + 1
            plan.steps.insert(insert_idx, retry_step)
            
            logger.info(f"[ToolOrchestrator] Added retry step for {tool_name}")
            return True
        
        return False
    
    def _is_critical_step(self, step: ToolCall) -> bool:
        """Check if a step is critical for the plan."""
        critical_tools = {"parse_code", "generate_tests"}
        return step.tool_name in critical_tools
    
    async def execute_with_adaptation(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> OrchestrationResult:
        """Plan and execute with automatic adaptation.
        
        Args:
            goal: The goal to achieve
            context: Current context
            on_progress: Progress callback
            
        Returns:
            OrchestrationResult
        """
        plan = self.plan_tool_sequence(goal, context)
        return await self.execute_plan(plan, on_progress)
    
    def cancel_plan(self, plan_id: str) -> bool:
        """Cancel an active plan.
        
        Args:
            plan_id: ID of the plan to cancel
            
        Returns:
            True if plan was cancelled
        """
        plan = self._active_plans.get(plan_id)
        if plan:
            plan.state = PlanState.CANCELLED
            for step in plan.steps:
                if step.state == ToolState.PENDING:
                    step.state = ToolState.CANCELLED
            logger.info(f"[ToolOrchestrator] Cancelled plan {plan_id}")
            return True
        return False
    
    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        """Get a plan by ID."""
        return self._active_plans.get(plan_id)
    
    def get_execution_history(self) -> List[ExecutionPlan]:
        """Get execution history."""
        return self._execution_history.copy()
    
    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each tool."""
        stats: Dict[str, Dict[str, Any]] = {}
        
        for plan in self._execution_history:
            for step in plan.steps:
                if step.tool_name not in stats:
                    stats[step.tool_name] = {
                        "total_calls": 0,
                        "successful": 0,
                        "failed": 0,
                        "total_time": 0.0,
                        "avg_time": 0.0
                    }
                
                stats[step.tool_name]["total_calls"] += 1
                if step.state == ToolState.COMPLETED:
                    stats[step.tool_name]["successful"] += 1
                elif step.state == ToolState.FAILED:
                    stats[step.tool_name]["failed"] += 1
                stats[step.tool_name]["total_time"] += step.duration
        
        for tool_stats in stats.values():
            if tool_stats["total_calls"] > 0:
                tool_stats["avg_time"] = tool_stats["total_time"] / tool_stats["total_calls"]
                tool_stats["success_rate"] = tool_stats["successful"] / tool_stats["total_calls"]
        
        return stats


def create_tool_orchestrator(
    tools: Optional[Dict[str, Callable]] = None,
    max_parallel: int = 3
) -> ToolOrchestrator:
    """Create a ToolOrchestrator instance.
    
    Args:
        tools: Dictionary of tool functions
        max_parallel: Maximum parallel executions
        
    Returns:
        Configured ToolOrchestrator
    """
    return ToolOrchestrator(
        tools=tools,
        max_parallel=max_parallel
    )
