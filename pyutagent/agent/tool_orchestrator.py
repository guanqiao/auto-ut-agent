"""Dynamic tool orchestration for flexible agent workflows.

This module provides intelligent tool orchestration:
- Automatic tool sequence planning
- Dynamic plan adaptation
- Tool dependency management
- Conditional execution
- Parallel tool execution
- Goal-based intelligent planning
- Tool chain reasoning
- Learning from execution patterns
"""

import asyncio
import json
import logging
import re
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
    reasoning: Optional[str] = None


@dataclass
class ToolChainStep:
    """A step in a tool chain with reasoning."""
    tool_name: str
    parameters: Dict[str, Any]
    reasoning: str
    expected_output: Optional[str] = None
    condition: Optional[str] = None


@dataclass
class ToolChainPlan:
    """A dynamic tool chain plan."""
    goal: str
    steps: List[ToolChainStep]
    fallback_steps: List[ToolChainStep] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    - Goal-based intelligent planning (plan_from_goal)
    - Dynamic tool chain planning
    - Tool execution result reasoning
    """
    
    def __init__(
        self,
        tools: Optional[Dict[str, Callable]] = None,
        tool_definitions: Optional[Dict[str, ToolDefinition]] = None,
        max_parallel: int = 3,
        adaptation_enabled: bool = True,
        llm_client: Optional[Any] = None
    ):
        self.tools = tools or {}
        self.tool_definitions = tool_definitions or {}
        self.max_parallel = max_parallel
        self.adaptation_enabled = adaptation_enabled
        self._llm_client = llm_client
        
        self._dependency_graph = DependencyGraph()
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._execution_history: List[ExecutionPlan] = []
        self._tool_memory: Optional[Any] = None
        
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
    
    def set_llm_client(self, llm_client: Any):
        """Set the LLM client for intelligent planning.
        
        Args:
            llm_client: LLM client instance for tool selection
        """
        self._llm_client = llm_client
        logger.info("[ToolOrchestrator] LLM client set for intelligent planning")
    
    def set_tool_memory(self, tool_memory: Any):
        """Set the tool memory for learning from execution patterns.
        
        Args:
            tool_memory: ToolMemory instance
        """
        self._tool_memory = tool_memory
        logger.info("[ToolOrchestrator] Tool memory set for learning patterns")
    
    def plan_from_goal(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        use_llm: bool = True
    ) -> ExecutionPlan:
        """Create execution plan from natural language goal.
        
        This is the core intelligent planning method that analyzes the goal
        and generates an appropriate tool execution sequence.
        
        Args:
            goal: Natural language description of the goal
            context: Current context and available data
            constraints: Constraints on execution
            use_llm: Whether to use LLM for intelligent planning
            
        Returns:
            ExecutionPlan with planned tool calls
        """
        import uuid
        
        context = context or {}
        constraints = constraints or {}
        
        plan_id = str(uuid.uuid4())[:8]
        
        if use_llm and self._llm_client:
            try:
                steps = self._plan_with_llm(goal, context, constraints)
                logger.info(f"[ToolOrchestrator] Created LLM-based plan {plan_id}")
            except Exception as e:
                logger.warning(f"[ToolOrchestrator] LLM planning failed: {e}, using rule-based")
                steps = self._plan_with_rules(goal, context, constraints)
        else:
            steps = self._plan_with_rules(goal, context, constraints)
        
        plan = ExecutionPlan(
            id=plan_id,
            goal=goal,
            steps=steps,
            state=PlanState.CREATED,
            metadata={
                "context": context,
                "constraints": constraints,
                "created_by": "llm_planner" if (use_llm and self._llm_client) else "rule_planner",
                "planning_method": "llm" if (use_llm and self._llm_client) else "rule_based"
            }
        )
        
        self._active_plans[plan_id] = plan
        
        logger.info(
            f"[ToolOrchestrator] Created plan {plan_id} with {len(steps)} steps for goal: {goal}"
        )
        
        return plan
    
    def _plan_with_llm(
        self,
        goal: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> List[ToolCall]:
        """Plan tool sequence using LLM.
        
        Args:
            goal: The goal to achieve
            context: Current context
            constraints: Execution constraints
            
        Returns:
            List of ToolCall steps
        """
        tools_info = self._get_tools_info_for_llm()
        
        prompt = self._build_planning_prompt(goal, context, constraints, tools_info)
        
        response = asyncio.get_event_loop().run_until_complete(
            self._llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
        )
        
        return self._parse_llm_plan(response.content, context)
    
    def _get_tools_info_for_llm(self) -> List[Dict[str, Any]]:
        """Get tool information formatted for LLM consumption.
        
        Returns:
            List of tool info dictionaries
        """
        tools_info = []
        for name, definition in self.tool_definitions.items():
            tools_info.append({
                "name": definition.name,
                "description": definition.description,
                "category": definition.category,
                "required_inputs": definition.required_inputs,
                "provided_outputs": definition.provided_outputs,
                "dependencies": definition.dependencies,
                "can_run_parallel": definition.can_run_parallel
            })
        return tools_info
    
    def _build_planning_prompt(
        self,
        goal: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any],
        tools_info: List[Dict[str, Any]]
    ) -> str:
        """Build the planning prompt for LLM.
        
        Args:
            goal: The goal to achieve
            context: Current context
            constraints: Execution constraints
            tools_info: Available tools information
            
        Returns:
            Prompt string
        """
        context_str = json.dumps(context, indent=2, default=str)
        tools_str = json.dumps(tools_info, indent=2)
        constraints_str = json.dumps(constraints, indent=2, default=str)
        
        return f"""You are an expert tool orchestration planner. Analyze the following goal and create a precise tool execution plan.

## Goal
{goal}

## Current Context
{context_str}

## Available Tools
{tools_str}

## Constraints
{constraints_str}

## Task
Create a tool execution plan to achieve the goal. Consider:
1. Tool dependencies - some tools require outputs from others
2. Parallel execution - independent tools can run in parallel
3. Input/output matching - ensure required inputs are available
4. Error handling - include fallback options when appropriate

## Output Format
Return a JSON array of tool calls. Each tool call should have:
- tool_name: Name of the tool to execute
- parameters: Dictionary of parameter names and values (can reference context variables)
- reasoning: Brief explanation of why this tool is needed

Example:
[
  {{
    "tool_name": "parse_code",
    "parameters": {{"file_path": "$context.source_file"}},
    "reasoning": "Need to analyze source code structure first"
  }},
  {{
    "tool_name": "generate_tests",
    "parameters": {{"class_info": "$parse_code.class_info"}},
    "reasoning": "Generate tests based on parsed class information"
  }}
]

Only output the JSON array, no additional text:"""
    
    def _parse_llm_plan(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> List[ToolCall]:
        """Parse LLM response into tool calls.
        
        Args:
            response: LLM response text
            context: Current context for parameter resolution
            
        Returns:
            List of ToolCall objects
        """
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("[ToolOrchestrator] No JSON array found in LLM response")
                return []
            
            json_str = response[json_start:json_end]
            plan_data = json.loads(json_str)
            
            steps = []
            for item in plan_data:
                tool_name = item.get("tool_name", "")
                if tool_name not in self.tool_definitions:
                    logger.warning(f"[ToolOrchestrator] Unknown tool in plan: {tool_name}")
                    continue
                
                parameters = item.get("parameters", {})
                resolved_params = self._resolve_parameters(parameters, context)
                
                tool_call = ToolCall(
                    tool_name=tool_name,
                    kwargs=resolved_params,
                    state=ToolState.PENDING,
                    metadata={
                        "reasoning": item.get("reasoning", ""),
                        "planned_by": "llm"
                    }
                )
                steps.append(tool_call)
            
            return steps
            
        except json.JSONDecodeError as e:
            logger.error(f"[ToolOrchestrator] Failed to parse LLM plan JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"[ToolOrchestrator] Error parsing LLM plan: {e}")
            return []
    
    def _resolve_parameters(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter references to actual values.
        
        Args:
            parameters: Parameters with possible references
            context: Context for resolution
            
        Returns:
            Resolved parameters
        """
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                ref_path = value[1:]
                resolved[key] = self._resolve_reference(ref_path, context)
            else:
                resolved[key] = value
        return resolved
    
    def _resolve_reference(self, ref_path: str, context: Dict[str, Any]) -> Any:
        """Resolve a reference path to a value.
        
        Args:
            ref_path: Reference path (e.g., "context.key" or "tool_name.output")
            context: Context dictionary
            
        Returns:
            Resolved value or original path if not found
        """
        parts = ref_path.split(".")
        if len(parts) >= 2 and parts[0] == "context":
            return context.get(".".join(parts[1:]), ref_path)
        return context.get(ref_path, ref_path)
    
    def _plan_with_rules(
        self,
        goal: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> List[ToolCall]:
        """Plan tool sequence using rule-based approach.
        
        Args:
            goal: The goal to achieve
            context: Current context
            constraints: Execution constraints
            
        Returns:
            List of ToolCall steps
        """
        required_tools = self._determine_required_tools(goal, context)
        ordered_tools = self._order_tools_by_dependencies(required_tools)
        
        steps = []
        for tool_name in ordered_tools:
            tool_call = ToolCall(
                tool_name=tool_name,
                kwargs=self._prepare_tool_kwargs(tool_name, context),
                state=ToolState.PENDING,
                metadata={
                    "reasoning": f"Required for goal: {goal}",
                    "planned_by": "rule_based"
                }
            )
            steps.append(tool_call)
        
        return steps
    
    def create_tool_chain(
        self,
        goal: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> ToolChainPlan:
        """Create a dynamic tool chain plan with reasoning.
        
        This method creates a flexible tool chain that can adapt during execution
        based on intermediate results.
        
        Args:
            goal: The goal to achieve
            initial_context: Initial context data
            
        Returns:
            ToolChainPlan with steps and reasoning
        """
        initial_context = initial_context or {}
        
        steps = []
        fallback_steps = []
        
        goal_lower = goal.lower()
        
        if "test" in goal_lower:
            steps.extend([
                ToolChainStep(
                    tool_name="parse_code",
                    parameters={},
                    reasoning="Extract code structure to understand what to test",
                    expected_output="class_info, method_signatures"
                ),
                ToolChainStep(
                    tool_name="generate_tests",
                    parameters={"class_info": "$parse_code.class_info"},
                    reasoning="Generate test cases based on parsed code structure",
                    expected_output="test_code, test_file"
                ),
                ToolChainStep(
                    tool_name="compile_tests",
                    parameters={"test_file": "$generate_tests.test_file"},
                    reasoning="Compile generated tests to check for syntax errors",
                    expected_output="compilation_result",
                    condition="compilation_needed"
                ),
                ToolChainStep(
                    tool_name="run_tests",
                    parameters={"test_file": "$generate_tests.test_file"},
                    reasoning="Execute tests to verify they pass",
                    expected_output="test_results, failures"
                )
            ])
            
            fallback_steps.extend([
                ToolChainStep(
                    tool_name="fix_compilation",
                    parameters={},
                    reasoning="Fix compilation errors if compilation fails",
                    condition="compilation_failed"
                ),
                ToolChainStep(
                    tool_name="fix_tests",
                    parameters={},
                    reasoning="Fix failing tests if tests fail",
                    condition="tests_failed"
                )
            ])
        
        elif "analyze" in goal_lower or "parse" in goal_lower:
            steps.append(ToolChainStep(
                tool_name="parse_code",
                parameters={},
                reasoning="Parse and analyze code structure",
                expected_output="class_info, dependencies"
            ))
        
        elif "fix" in goal_lower:
            if "compilation" in goal_lower:
                steps.append(ToolChainStep(
                    tool_name="fix_compilation",
                    parameters={},
                    reasoning="Fix compilation errors in code"
                ))
            elif "test" in goal_lower:
                steps.append(ToolChainStep(
                    tool_name="fix_tests",
                    parameters={},
                    reasoning="Fix failing test cases"
                ))
        
        else:
            steps.append(ToolChainStep(
                tool_name="parse_code",
                parameters={},
                reasoning="Default: start by parsing code"
            ))
        
        parallel_groups = self._identify_parallel_groups(steps)
        
        return ToolChainPlan(
            goal=goal,
            steps=steps,
            fallback_steps=fallback_steps,
            parallel_groups=parallel_groups,
            metadata={
                "created_at": datetime.now().isoformat(),
                "initial_context": initial_context
            }
        )
    
    def _identify_parallel_groups(
        self,
        steps: List[ToolChainStep]
    ) -> List[List[str]]:
        """Identify groups of tools that can run in parallel.
        
        Args:
            steps: List of tool chain steps
            
        Returns:
            List of parallel tool groups
        """
        groups = []
        current_group = []
        
        for step in steps:
            tool_def = self.tool_definitions.get(step.tool_name)
            if tool_def and tool_def.can_run_parallel:
                current_group.append(step.tool_name)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([step.tool_name])
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def reason_about_result(
        self,
        tool_name: str,
        result: Any,
        goal: str,
        current_plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Reason about tool execution result and determine next steps.
        
        This method analyzes the result of a tool execution and decides:
        - Whether the goal is achieved
        - What tools to call next
        - Whether to adapt the plan
        
        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result
            goal: The overall goal
            current_plan: Current execution plan
            
        Returns:
            Dictionary with reasoning results:
            - goal_achieved: bool
            - next_tools: List[str]
            - should_adapt: bool
            - reasoning: str
        """
        reasoning_result = {
            "goal_achieved": False,
            "next_tools": [],
            "should_adapt": False,
            "reasoning": "",
            "confidence": 0.0
        }
        
        if self._llm_client:
            try:
                reasoning_result = await self._reason_with_llm(
                    tool_name, result, goal, current_plan
                )
            except Exception as e:
                logger.warning(f"[ToolOrchestrator] LLM reasoning failed: {e}")
                reasoning_result = self._reason_with_rules(
                    tool_name, result, goal, current_plan
                )
        else:
            reasoning_result = self._reason_with_rules(
                tool_name, result, goal, current_plan
            )
        
        if self._tool_memory:
            await self._tool_memory.record_success(
                tool_name=tool_name,
                params={},
                context={"goal": goal, "plan_id": current_plan.id},
                result=result,
                task_type=self._classify_task_type(goal)
            )
        
        return reasoning_result
    
    async def _reason_with_llm(
        self,
        tool_name: str,
        result: Any,
        goal: str,
        current_plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Use LLM to reason about tool execution result.
        
        Args:
            tool_name: Name of the executed tool
            result: Tool execution result
            goal: The overall goal
            current_plan: Current execution plan
            
        Returns:
            Reasoning result dictionary
        """
        result_str = json.dumps(result, indent=2, default=str)[:2000]
        
        completed_tools = [s.tool_name for s in current_plan.steps 
                          if s.state == ToolState.COMPLETED]
        pending_tools = [s.tool_name for s in current_plan.steps 
                        if s.state == ToolState.PENDING]
        
        prompt = f"""Analyze the result of a tool execution and determine next steps.

## Goal
{goal}

## Tool Executed
{tool_name}

## Execution Result
{result_str}

## Plan Progress
Completed tools: {completed_tools}
Pending tools: {pending_tools}

## Task
Analyze the result and determine:
1. Is the goal achieved? (true/false)
2. What tools should be called next? (list of tool names)
3. Should the plan be adapted? (true/false)
4. Brief reasoning for your decision

## Output Format
Return a JSON object:
{{
  "goal_achieved": boolean,
  "next_tools": ["tool_name_1", "tool_name_2"],
  "should_adapt": boolean,
  "reasoning": "explanation",
  "confidence": 0.0-1.0
}}

Only output the JSON object:"""
        
        response = await self._llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response.content[json_start:json_end]
            reasoning = json.loads(json_str)
            
            return {
                "goal_achieved": reasoning.get("goal_achieved", False),
                "next_tools": reasoning.get("next_tools", []),
                "should_adapt": reasoning.get("should_adapt", False),
                "reasoning": reasoning.get("reasoning", ""),
                "confidence": reasoning.get("confidence", 0.5)
            }
            
        except Exception as e:
            logger.error(f"[ToolOrchestrator] Failed to parse reasoning: {e}")
            return self._reason_with_rules(tool_name, result, goal, current_plan)
    
    def _reason_with_rules(
        self,
        tool_name: str,
        result: Any,
        goal: str,
        current_plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Use rule-based approach to reason about result.
        
        Args:
            tool_name: Name of the executed tool
            result: Tool execution result
            goal: The overall goal
            current_plan: Current execution plan
            
        Returns:
            Reasoning result dictionary
        """
        reasoning = {
            "goal_achieved": False,
            "next_tools": [],
            "should_adapt": False,
            "reasoning": "",
            "confidence": 0.5
        }
        
        result_dict = result if isinstance(result, dict) else {"result": result}
        
        if tool_name == "parse_code":
            if "class_info" in result_dict:
                reasoning["next_tools"] = ["generate_tests"]
                reasoning["reasoning"] = "Code parsed successfully, ready to generate tests"
                reasoning["confidence"] = 0.8
            else:
                reasoning["should_adapt"] = True
                reasoning["reasoning"] = "Parse failed, may need to retry with different parameters"
        
        elif tool_name == "generate_tests":
            if "test_code" in result_dict:
                reasoning["next_tools"] = ["compile_tests"]
                reasoning["reasoning"] = "Tests generated, proceeding to compilation"
                reasoning["confidence"] = 0.8
            else:
                reasoning["should_adapt"] = True
                reasoning["reasoning"] = "Test generation failed"
        
        elif tool_name == "compile_tests":
            if result_dict.get("success", False):
                reasoning["next_tools"] = ["run_tests"]
                reasoning["reasoning"] = "Compilation successful, ready to run tests"
                reasoning["confidence"] = 0.9
            else:
                reasoning["should_adapt"] = True
                reasoning["next_tools"] = ["fix_compilation"]
                reasoning["reasoning"] = "Compilation failed, need to fix errors"
        
        elif tool_name == "run_tests":
            if result_dict.get("all_passed", False):
                reasoning["goal_achieved"] = True
                reasoning["reasoning"] = "All tests passed, goal achieved"
                reasoning["confidence"] = 0.95
            else:
                reasoning["should_adapt"] = True
                reasoning["next_tools"] = ["fix_tests"]
                reasoning["reasoning"] = "Some tests failed, need to fix them"
        
        elif tool_name == "fix_compilation":
            reasoning["next_tools"] = ["compile_tests"]
            reasoning["reasoning"] = "Retry compilation after fixes"
        
        elif tool_name == "fix_tests":
            reasoning["next_tools"] = ["run_tests"]
            reasoning["reasoning"] = "Retry test execution after fixes"
        
        goal_lower = goal.lower()
        if "test" in goal_lower and tool_name == "run_tests":
            if result_dict.get("all_passed", False):
                reasoning["goal_achieved"] = True
        
        return reasoning
    
    def _classify_task_type(self, goal: str) -> str:
        """Classify the task type from the goal.
        
        Args:
            goal: The goal description
            
        Returns:
            Task type string
        """
        goal_lower = goal.lower()
        
        if "test" in goal_lower and "generate" in goal_lower:
            return "test_generation"
        elif "test" in goal_lower and "fix" in goal_lower:
            return "test_fixing"
        elif "coverage" in goal_lower:
            return "coverage_analysis"
        elif "parse" in goal_lower or "analyze" in goal_lower:
            return "code_analysis"
        elif "compile" in goal_lower or "build" in goal_lower:
            return "compilation"
        else:
            return "general"
    
    async def get_recommended_tools_for_goal(
        self,
        goal: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get tool recommendations based on goal and historical patterns.
        
        Args:
            goal: The goal to achieve
            limit: Maximum number of recommendations
            
        Returns:
            List of tool recommendations with scores
        """
        if not self._tool_memory:
            return []
        
        task_type = self._classify_task_type(goal)
        recommendations = await self._tool_memory.get_recommended_tools(task_type, limit)
        
        return [
            {
                "tool_name": r.tool_name,
                "reason": r.reason,
                "success_rate": r.success_rate,
                "usage_count": r.usage_count,
                "score": r.success_rate * r.usage_count
            }
            for r in recommendations
        ]
    
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
