"""Unified Tool Service - 统一工具服务层。

This module provides:
- UnifiedToolService: 统一的工具服务接口
- 整合注册、编排、执行、缓存
- 统一的ToolCall和ToolResult定义
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..tools.tool import Tool, ToolCategory, ToolResult
from ..tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolState(Enum):
    """工具执行状态。"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


class PlanState(Enum):
    """执行计划状态。"""
    CREATED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ADAPTING = auto()
    CANCELLED = auto()


@dataclass
class ToolCall:
    """统一的工具调用定义。"""
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    state: ToolState = ToolState.PENDING
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempt: int = 0
    call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0


@dataclass
class ExecutionPlan:
    """工具执行计划。"""
    id: str
    goal: str
    steps: List[ToolCall]
    state: PlanState = PlanState.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds() * 1000)
        return 0

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
    """编排执行结果。"""
    success: bool
    plan: ExecutionPlan
    results: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    total_time_ms: int = 0
    adaptations: int = 0
    message: str = ""


@dataclass
class ToolServiceConfig:
    """工具服务配置。"""
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    max_parallel: int = 3
    adaptation_enabled: bool = True


class DependencyGraph:
    """工具依赖图。"""

    def __init__(self):
        self.nodes: Dict[str, Set[str]] = {}
        self.tool_info: Dict[str, Dict[str, Any]] = {}

    def add_tool(
        self,
        tool_name: str,
        dependencies: Optional[List[str]] = None,
        provides: Optional[List[str]] = None
    ):
        self.nodes[tool_name] = set(dependencies or [])
        self.tool_info[tool_name] = {
            "dependencies": dependencies or [],
            "provides": provides or []
        }

    def get_execution_order(self, tool_names: List[str]) -> List[str]:
        ordered = []
        added = set()

        def add_with_deps(tool_name: str):
            if tool_name in added:
                return
            if tool_name not in self.nodes:
                ordered.append(tool_name)
                added.add(tool_name)
                return

            for dep in self.nodes.get(tool_name, set()):
                if dep in tool_names:
                    add_with_deps(dep)

            ordered.append(tool_name)
            added.add(tool_name)

        for tool_name in tool_names:
            add_with_deps(tool_name)

        return ordered

    def has_cycle(self) -> bool:
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for dep in self.nodes.get(node, set()):
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


class UnifiedToolService:
    """统一工具服务 - 整合注册、编排、执行、缓存。

    Features:
    - 统一的工具注册接口
    - 智能工具编排
    - 执行结果缓存
    - 重试和超时处理
    - LLM工具选择支持
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        config: Optional[ToolServiceConfig] = None
    ):
        self.registry = registry or ToolRegistry()
        self.config = config or ToolServiceConfig()

        self._tool_functions: Dict[str, Callable] = {}
        self._dependency_graph = DependencyGraph()
        self._cache: Dict[str, tuple] = {}
        self._execution_history: List[ExecutionPlan] = []
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._llm_client: Optional[Any] = None

        logger.info("[UnifiedToolService] Initialized")

    def set_llm_client(self, llm_client: Any):
        self._llm_client = llm_client

    def register_tool(
        self,
        tool: Tool,
        func: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ):
        self.registry.register(tool, tags=tags)

        if func:
            self._tool_functions[tool.definition.name] = func

        self._dependency_graph.add_tool(
            tool.definition.name,
            dependencies=dependencies,
            provides=provides
        )

        logger.info(f"[UnifiedToolService] Registered tool: {tool.definition.name}")

    def register_function(
        self,
        name: str,
        func: Callable,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        provides: Optional[List[str]] = None
    ):
        self._tool_functions[name] = func
        self._dependency_graph.add_tool(
            name,
            dependencies=dependencies,
            provides=provides
        )

        logger.info(f"[UnifiedToolService] Registered function: {name}")

    def unregister_tool(self, tool_name: str) -> bool:
        if tool_name in self._tool_functions:
            del self._tool_functions[tool_name]

        if tool_name in self._dependency_graph.nodes:
            del self._dependency_graph.nodes[tool_name]

        return self.registry.unregister(tool_name)

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        return self.registry.get_or_none(tool_name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        return self.registry.list_tools(category)

    def search_tools(self, query: str) -> List[str]:
        return self.registry.search(query)

    def get_schemas_for_llm(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self.registry.get_schemas(force_refresh)

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        cache_key = self._get_cache_key(tool_name, parameters)

        if self.config.enable_cache and cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl_seconds:
                logger.debug(f"[UnifiedToolService] Cache hit: {tool_name}")
                return cached_result

        tool = self.registry.get_or_none(tool_name)

        if tool is None and tool_name not in self._tool_functions:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}"
            )

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                start_time = datetime.now()

                if tool:
                    result = await tool.execute(**parameters)
                elif tool_name in self._tool_functions:
                    func = self._tool_functions[tool_name]
                    if asyncio.iscoroutinefunction(func):
                        result = await asyncio.wait_for(
                            func(**parameters),
                            timeout=self.config.timeout_seconds
                        )
                    else:
                        result = func(**parameters)
                else:
                    return ToolResult(success=False, error=f"Tool not found: {tool_name}")

                if not isinstance(result, ToolResult):
                    result = ToolResult(success=True, output=result)

                if self.config.enable_cache:
                    self._cache[cache_key] = (result, time.time())

                logger.info(f"[UnifiedToolService] Executed: {tool_name}")
                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.config.timeout_seconds}s"
                logger.warning(f"[UnifiedToolService] Timeout: {tool_name}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"[UnifiedToolService] Error: {tool_name} - {e}")

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))

        return ToolResult(success=False, error=last_error)

    async def execute_tools_parallel(
        self,
        calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        tasks = [
            self.execute_tool(call["tool_name"], call.get("parameters", {}))
            for call in calls
        ]
        return await asyncio.gather(*tasks)

    async def execute_tools_sequence(
        self,
        calls: List[Dict[str, Any]]
    ) -> List[ToolResult]:
        results = []

        for call in calls:
            result = await self.execute_tool(
                call["tool_name"],
                call.get("parameters", {})
            )
            results.append(result)

            if not result.success:
                logger.warning(f"[UnifiedToolService] Tool {call['tool_name']} failed, stopping")
                break

        return results

    def create_plan(
        self,
        goal: str,
        tool_sequence: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        import uuid

        plan_id = str(uuid.uuid4())[:8]

        steps = []
        for tool_def in tool_sequence:
            tool_call = ToolCall(
                tool_name=tool_def["tool_name"],
                parameters=tool_def.get("parameters", {}),
                state=ToolState.PENDING,
                metadata={"reasoning": tool_def.get("reasoning", "")}
            )
            steps.append(tool_call)

        plan = ExecutionPlan(
            id=plan_id,
            goal=goal,
            steps=steps,
            state=PlanState.CREATED,
            metadata={"context": context or {}}
        )

        self._active_plans[plan_id] = plan
        logger.info(f"[UnifiedToolService] Created plan {plan_id} with {len(steps)} steps")

        return plan

    async def plan_from_goal(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        if self._llm_client:
            return await self._plan_with_llm(goal, context or {})
        else:
            return self._plan_heuristic(goal, context or {})

    async def _plan_with_llm(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        import uuid

        tools_info = []
        for tool in self.registry:
            tools_info.append({
                "name": tool.definition.name,
                "description": tool.definition.description,
                "parameters": [p.name for p in tool.definition.parameters]
            })

        prompt = f"""Analyze the goal and select appropriate tools.

Goal: {goal}

Context:
{json.dumps(context, indent=2, default=str)}

Available Tools:
{json.dumps(tools_info, indent=2)}

Return a JSON array of tool selections:
[
  {{"tool_name": "...", "parameters": {{...}}, "reasoning": "..."}}
]
Only output the JSON array:"""

        try:
            response = await self._llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            content = response.content if hasattr(response, 'content') else str(response)
            json_start = content.find('[')
            json_end = content.rfind(']') + 1

            if json_start == -1:
                return self._plan_heuristic(goal, context)

            tool_sequence = json.loads(content[json_start:json_end])

            return self.create_plan(goal, tool_sequence, context)

        except Exception as e:
            logger.error(f"[UnifiedToolService] LLM planning failed: {e}")
            return self._plan_heuristic(goal, context)

    def _plan_heuristic(
        self,
        goal: str,
        context: Dict[str, Any]
    ) -> ExecutionPlan:
        goal_lower = goal.lower()
        tool_sequence = []

        if "read" in goal_lower or "view" in goal_lower:
            file_path = context.get("file_path")
            if file_path:
                tool_sequence.append({
                    "tool_name": "read_file",
                    "parameters": {"file_path": file_path},
                    "reasoning": "Read file content"
                })

        if "write" in goal_lower or "create" in goal_lower:
            file_path = context.get("file_path")
            content = context.get("content")
            if file_path and content:
                tool_sequence.append({
                    "tool_name": "write_file",
                    "parameters": {"file_path": file_path, "content": content},
                    "reasoning": "Write file content"
                })

        if "edit" in goal_lower or "modify" in goal_lower:
            file_path = context.get("file_path")
            search = context.get("search")
            replace = context.get("replace")
            if file_path and search and replace:
                tool_sequence.append({
                    "tool_name": "edit_file",
                    "parameters": {"file_path": file_path, "search": search, "replace": replace},
                    "reasoning": "Edit file content"
                })

        if "search" in goal_lower or "find" in goal_lower:
            pattern = context.get("pattern", ".")
            tool_sequence.append({
                "tool_name": "grep",
                "parameters": {"pattern": pattern},
                "reasoning": "Search for pattern"
            })

        if "run" in goal_lower or "execute" in goal_lower:
            command = context.get("command", "echo 'done'")
            tool_sequence.append({
                "tool_name": "bash",
                "parameters": {"command": command},
                "reasoning": "Execute command"
            })

        return self.create_plan(goal, tool_sequence, context)

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> OrchestrationResult:
        plan.state = PlanState.RUNNING
        plan.started_at = datetime.now()

        results: Dict[str, Any] = {}
        errors: List[str] = []
        adaptations = 0

        context = plan.metadata.get("context", {})

        logger.info(f"[UnifiedToolService] Executing plan {plan.id}")

        try:
            for i, step in enumerate(plan.steps):
                if plan.state == PlanState.CANCELLED:
                    break

                step.state = ToolState.RUNNING
                step.start_time = datetime.now()
                step.attempt += 1

                merged_params = {**step.parameters}
                for key, value in results.items():
                    if key in step.parameters and step.parameters[key] == f"${key}":
                        merged_params[key] = value

                result = await self.execute_tool(step.tool_name, merged_params, context)

                step.result = result
                step.end_time = datetime.now()

                if result.success:
                    step.state = ToolState.COMPLETED
                    results[step.tool_name] = result.output
                else:
                    step.state = ToolState.FAILED
                    step.error = result.error
                    errors.append(f"{step.tool_name}: {result.error}")

                    if self.config.adaptation_enabled:
                        adapted = self._adapt_plan(plan, step, result.error)
                        if adapted:
                            adaptations += 1
                            continue

                    if self._is_critical_step(step):
                        plan.state = PlanState.FAILED
                        break

                if on_progress:
                    on_progress(plan.progress)

            if plan.state == PlanState.RUNNING:
                plan.state = PlanState.COMPLETED

        except Exception as e:
            plan.state = PlanState.FAILED
            errors.append(str(e))
            logger.exception(f"[UnifiedToolService] Plan execution failed: {e}")

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
            total_time_ms=plan.duration_ms,
            adaptations=adaptations,
            message=f"Plan {'completed' if success else 'failed'}"
        )

    def _adapt_plan(
        self,
        plan: ExecutionPlan,
        failed_step: ToolCall,
        error: str
    ) -> bool:
        if failed_step.attempt < 2:
            retry_step = ToolCall(
                tool_name=failed_step.tool_name,
                parameters=failed_step.parameters,
                state=ToolState.PENDING,
                attempt=failed_step.attempt
            )

            insert_idx = plan.steps.index(failed_step) + 1
            plan.steps.insert(insert_idx, retry_step)

            logger.info(f"[UnifiedToolService] Added retry for {failed_step.tool_name}")
            return True

        return False

    def _is_critical_step(self, step: ToolCall) -> bool:
        critical_tools = {"parse_code", "read_file", "generate_tests"}
        return step.tool_name in critical_tools

    async def execute_with_planning(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[float], None]] = None
    ) -> OrchestrationResult:
        plan = await self.plan_from_goal(goal, context)
        return await self.execute_plan(plan, on_progress)

    def cancel_plan(self, plan_id: str) -> bool:
        plan = self._active_plans.get(plan_id)
        if plan:
            plan.state = PlanState.CANCELLED
            for step in plan.steps:
                if step.state == ToolState.PENDING:
                    step.state = ToolState.CANCELLED
            logger.info(f"[UnifiedToolService] Cancelled plan {plan_id}")
            return True
        return False

    def get_plan(self, plan_id: str) -> Optional[ExecutionPlan]:
        return self._active_plans.get(plan_id)

    def get_execution_history(self) -> List[ExecutionPlan]:
        return self._execution_history.copy()

    def _get_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        return hashlib.sha256(f"{tool_name}:{param_str}".encode()).hexdigest()

    def clear_cache(self):
        self._cache.clear()
        logger.info("[UnifiedToolService] Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        total_executions = len(self._execution_history)
        successful = sum(1 for p in self._execution_history if p.state == PlanState.COMPLETED)

        return {
            "total_tools": len(self.registry),
            "total_plans": total_executions,
            "successful_plans": successful,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "active_plans": len(self._active_plans),
            "cache_size": len(self._cache),
            "registry_stats": self.registry.get_stats()
        }


def create_unified_tool_service(
    registry: Optional[ToolRegistry] = None,
    timeout_seconds: int = 60,
    max_retries: int = 3,
    enable_cache: bool = True
) -> UnifiedToolService:
    config = ToolServiceConfig(
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        enable_cache=enable_cache
    )
    return UnifiedToolService(registry=registry, config=config)
