"""Delegating SubAgent with Skill binding and autonomous execution.

This module provides:
- DelegatingSubAgent: SubAgent that can execute Skills and make autonomous decisions
- Task delegation with skill binding
- Progress reporting and result callbacks
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from .subagents import (
    SubAgent,
    SubAgentConfig,
    AgentStatus,
    Task,
    TaskPriority,
    AgentCapability,
)
from .skills import Skill, SkillInput, SkillOutput, SkillRegistry, get_skill_registry

logger = logging.getLogger(__name__)


class DelegationMode(Enum):
    """Modes for task delegation."""
    SKILL_BASED = auto()
    AUTONOMOUS = auto()
    HYBRID = auto()


class ExecutionStrategy(Enum):
    """Strategies for executing delegated tasks."""
    SEQUENTIAL = auto()
    PARALLEL = auto()
    ADAPTIVE = auto()


@dataclass
class DelegationContext:
    """Context for task delegation."""
    parent_agent_id: str
    delegation_mode: DelegationMode = DelegationMode.HYBRID
    execution_strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    timeout: int = 300
    max_retries: int = 3
    share_context: bool = True
    progress_reporting: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DelegationResult:
    """Result of a delegation operation."""
    success: bool
    task_id: str
    agent_id: str
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    skill_used: Optional[str] = None
    iterations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressUpdate:
    """Progress update from SubAgent execution."""
    agent_id: str
    task_id: str
    progress: float
    status: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DelegatingSubAgent(SubAgent):
    """SubAgent with delegation, skill binding, and autonomous execution.

    Extends base SubAgent with:
    - Skill binding and execution
    - LLM-powered autonomous decision making
    - Progress reporting callbacks
    - Result aggregation support
    """

    def __init__(
        self,
        config: SubAgentConfig,
        llm_client: Optional[Any] = None,
        skill_registry: Optional[SkillRegistry] = None,
        tool_service: Optional[Any] = None
    ):
        """Initialize DelegatingSubAgent.

        Args:
            config: Agent configuration
            llm_client: Optional LLM client for autonomous decisions
            skill_registry: Optional skill registry (uses global if not provided)
            tool_service: Optional tool service for execution
        """
        super().__init__(config)
        self.llm_client = llm_client
        self.skill_registry = skill_registry or get_skill_registry()
        self.tool_service = tool_service

        self._bound_skills: Dict[str, Skill] = {}
        self._progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
        self._result_callback: Optional[Callable[[DelegationResult], None]] = None
        self._delegation_context: Optional[DelegationContext] = None

        self._execution_history: List[DelegationResult] = []
        self._skill_success_rates: Dict[str, Dict[str, int]] = {}

        logger.info(f"[DelegatingSubAgent:{self.id}] Initialized with type: {config.agent_type}")

    def bind_skill(self, skill: Skill) -> None:
        """Bind a skill to this agent.

        Args:
            skill: Skill to bind
        """
        self._bound_skills[skill.name] = skill
        logger.info(f"[DelegatingSubAgent:{self.id}] Bound skill: {skill.name}")

    def unbind_skill(self, skill_name: str) -> bool:
        """Unbind a skill from this agent.

        Args:
            skill_name: Name of skill to unbind

        Returns:
            True if skill was unbound
        """
        if skill_name in self._bound_skills:
            del self._bound_skills[skill_name]
            logger.info(f"[DelegatingSubAgent:{self.id}] Unbound skill: {skill_name}")
            return True
        return False

    def get_bound_skills(self) -> List[str]:
        """Get list of bound skill names."""
        return list(self._bound_skills.keys())

    def set_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Set callback for progress updates.

        Args:
            callback: Function to call with progress updates
        """
        self._progress_callback = callback

    def set_result_callback(self, callback: Callable[[DelegationResult], None]) -> None:
        """Set callback for result notifications.

        Args:
            callback: Function to call with results
        """
        self._result_callback = callback

    def set_delegation_context(self, context: DelegationContext) -> None:
        """Set the delegation context.

        Args:
            context: Delegation context from parent agent
        """
        self._delegation_context = context

    async def initialize(self) -> bool:
        """Initialize the SubAgent.

        Returns:
            True if initialization succeeded
        """
        logger.info(f"[DelegatingSubAgent:{self.id}] Initializing")
        self.status = AgentStatus.IDLE
        return True

    async def execute_task(self, task: Task) -> Any:
        """Execute a task with skill binding support.

        Args:
            task: Task to execute

        Returns:
            Task result
        """
        start_time = asyncio.get_event_loop().time()

        try:
            self.status = AgentStatus.BUSY
            self.current_task = task

            await self._report_progress(0.0, "started", f"Starting task: {task.name}")

            result = await self._execute_with_strategy(task)

            execution_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            delegation_result = DelegationResult(
                success=result.get("success", False) if isinstance(result, dict) else True,
                task_id=task.id,
                agent_id=self.id,
                output=result,
                execution_time_ms=execution_time_ms,
                skill_used=result.get("skill_used") if isinstance(result, dict) else None,
                metadata={"task_name": task.name}
            )

            self._execution_history.append(delegation_result)

            if self._result_callback:
                self._result_callback(delegation_result)

            await self._report_progress(1.0, "completed", f"Task completed: {task.name}")

            return result

        except Exception as e:
            logger.exception(f"[DelegatingSubAgent:{self.id}] Task execution failed: {e}")

            execution_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            error_result = DelegationResult(
                success=False,
                task_id=task.id,
                agent_id=self.id,
                error=str(e),
                execution_time_ms=execution_time_ms
            )

            self._execution_history.append(error_result)

            if self._result_callback:
                self._result_callback(error_result)

            await self._report_progress(1.0, "failed", f"Task failed: {str(e)[:100]}")

            raise

        finally:
            self.status = AgentStatus.IDLE
            self.current_task = None

    async def _execute_with_strategy(self, task: Task) -> Any:
        """Execute task using appropriate strategy.

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        if self._delegation_context:
            mode = self._delegation_context.delegation_mode
        else:
            mode = DelegationMode.HYBRID

        if mode == DelegationMode.SKILL_BASED:
            return await self._execute_skill_based(task)
        elif mode == DelegationMode.AUTONOMOUS:
            return await self._execute_autonomous(task)
        else:
            return await self._execute_hybrid(task)

    async def _execute_skill_based(self, task: Task) -> Any:
        """Execute task using bound skills.

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        skill_name = task.input_data.get("skill_name")
        skill = None

        if skill_name:
            skill = self._bound_skills.get(skill_name) or self.skill_registry.get(skill_name)
        else:
            skill = await self._find_best_skill(task)

        if skill:
            await self._report_progress(0.3, "executing", f"Using skill: {skill.name}")
            result = await self.execute_with_skill(task, skill)
            self._update_skill_stats(skill.name, result.get("success", False))
            return result

        return {"success": False, "error": "No suitable skill found for task"}

    async def _execute_autonomous(self, task: Task) -> Any:
        """Execute task autonomously using LLM.

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        if not self.llm_client:
            return {"success": False, "error": "No LLM client available for autonomous execution"}

        await self._report_progress(0.2, "thinking", "Analyzing task autonomously")

        plan = await self._create_execution_plan(task)

        if not plan:
            return {"success": False, "error": "Failed to create execution plan"}

        await self._report_progress(0.4, "executing", "Executing plan")

        results = []
        for i, step in enumerate(plan):
            step_result = await self._execute_step(step, task)
            results.append(step_result)

            progress = 0.4 + (0.5 * (i + 1) / len(plan))
            await self._report_progress(progress, "executing", f"Step {i+1}/{len(plan)} complete")

            if not step_result.get("success", False):
                break

        success = all(r.get("success", False) for r in results)
        return {
            "success": success,
            "results": results,
            "plan_steps": len(plan)
        }

    async def _execute_hybrid(self, task: Task) -> Any:
        """Execute task using hybrid approach.

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        skill = await self._find_best_skill(task)

        if skill:
            skill_result = await self.execute_with_skill(task, skill)
            if skill_result.get("success", False):
                return skill_result

            self._update_skill_stats(skill.name, False)

        if self.llm_client:
            autonomous_result = await self._execute_autonomous(task)
            if autonomous_result.get("success", False):
                return autonomous_result

        return {"success": False, "error": "Both skill-based and autonomous execution failed"}

    async def delegate(
        self,
        task: Task,
        skill: Optional[Skill] = None,
        context: Optional[DelegationContext] = None
    ) -> DelegationResult:
        """Delegate a task to this agent.

        Args:
            task: Task to delegate
            skill: Optional skill to use
            context: Optional delegation context

        Returns:
            DelegationResult with execution details
        """
        if context:
            self.set_delegation_context(context)

        if skill:
            self.bind_skill(skill)

        start_time = asyncio.get_event_loop().time()

        try:
            result = await self.execute_task(task)

            execution_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return DelegationResult(
                success=True if isinstance(result, dict) and result.get("success", True) else True,
                task_id=task.id,
                agent_id=self.id,
                output=result,
                execution_time_ms=execution_time_ms,
                skill_used=skill.name if skill else None
            )

        except Exception as e:
            execution_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return DelegationResult(
                success=False,
                task_id=task.id,
                agent_id=self.id,
                error=str(e),
                execution_time_ms=execution_time_ms
            )

    async def execute_with_skill(self, task: Task, skill: Skill) -> Any:
        """Execute a task using a specific skill.

        Args:
            task: Task to execute
            skill: Skill to use

        Returns:
            Execution result
        """
        logger.info(f"[DelegatingSubAgent:{self.id}] Executing with skill: {skill.name}")

        input_data = SkillInput(
            parameters=task.input_data,
            context={"task_id": task.id, "agent_id": self.id},
            files=task.input_data.get("files", [])
        )

        try:
            output = await skill.execute(input_data)

            return {
                "success": output.success,
                "result": output.result,
                "error": output.error,
                "logs": output.logs,
                "skill_used": skill.name
            }

        except Exception as e:
            logger.error(f"[DelegatingSubAgent:{self.id}] Skill execution failed: {e}")
            return {"success": False, "error": str(e), "skill_used": skill.name}

    async def execute_autonomously(self, task: Task) -> Any:
        """Execute a task autonomously without skill binding.

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        return await self._execute_autonomous(task)

    async def _find_best_skill(self, task: Task) -> Optional[Skill]:
        """Find the best skill for a task.

        Args:
            task: Task to find skill for

        Returns:
            Best matching skill or None
        """
        task_description = f"{task.name} {task.description}"

        bound_matches = []
        for name, skill in self._bound_skills.items():
            score = self._calculate_skill_score(skill, task_description, task.input_data)
            if score > 0:
                bound_matches.append((score, skill))

        if bound_matches:
            bound_matches.sort(key=lambda x: x[0], reverse=True)
            return bound_matches[0][1]

        registry_matches = self.skill_registry.find_by_trigger(task_description)
        if registry_matches:
            return self.skill_registry.get(registry_matches[0])

        return None

    def _calculate_skill_score(
        self,
        skill: Skill,
        task_description: str,
        task_data: Dict[str, Any]
    ) -> float:
        """Calculate how well a skill matches a task.

        Args:
            skill: Skill to score
            task_description: Task description
            task_data: Task input data

        Returns:
            Match score (0.0 - 1.0)
        """
        score = 0.0
        metadata = skill.metadata

        desc_lower = task_description.lower()

        for trigger in metadata.triggers:
            if trigger.lower() in desc_lower:
                score += 0.3
                break

        for tag in metadata.tags:
            if tag.lower() in desc_lower:
                score += 0.1

        required_params = set(metadata.requires)
        if required_params:
            provided_params = set(task_data.keys())
            overlap = len(required_params & provided_params)
            score += 0.3 * (overlap / len(required_params))

        stats = self._skill_success_rates.get(skill.name, {})
        if stats:
            total = stats.get("total", 0)
            successes = stats.get("successes", 0)
            if total > 0:
                success_rate = successes / total
                score += 0.2 * success_rate

        return min(1.0, score)

    async def _create_execution_plan(self, task: Task) -> List[Dict[str, Any]]:
        """Create an execution plan for autonomous execution.

        Args:
            task: Task to plan

        Returns:
            List of execution steps
        """
        if not self.llm_client:
            return []

        prompt = f"""Create an execution plan for the following task:

Task: {task.name}
Description: {task.description}
Input Data: {task.input_data}

Return a JSON array of steps, each with:
- tool_name: name of tool to use
- parameters: dict of parameters
- expected_outcome: what this step should achieve

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
                plan = json.loads(content[start:end+1])
                return plan

        except Exception as e:
            logger.error(f"[DelegatingSubAgent:{self.id}] Failed to create plan: {e}")

        return []

    async def _execute_step(self, step: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Execute a single step in the plan.

        Args:
            step: Step to execute
            task: Parent task

        Returns:
            Step result
        """
        if not self.tool_service:
            return {"success": False, "error": "No tool service available"}

        tool_name = step.get("tool_name")
        parameters = step.get("parameters", {})

        try:
            result = await self.tool_service.execute_tool(tool_name, parameters)
            return {
                "success": getattr(result, 'success', True),
                "result": getattr(result, 'output', result),
                "tool": tool_name
            }
        except Exception as e:
            return {"success": False, "error": str(e), "tool": tool_name}

    async def _report_progress(self, progress: float, status: str, message: str) -> None:
        """Report progress to callback.

        Args:
            progress: Progress value (0.0 - 1.0)
            status: Current status
            message: Progress message
        """
        if self._progress_callback and self.current_task:
            update = ProgressUpdate(
                agent_id=self.id,
                task_id=self.current_task.id,
                progress=progress,
                status=status,
                message=message
            )
            self._progress_callback(update)

    def _update_skill_stats(self, skill_name: str, success: bool) -> None:
        """Update skill success statistics.

        Args:
            skill_name: Name of skill
            success: Whether execution succeeded
        """
        if skill_name not in self._skill_success_rates:
            self._skill_success_rates[skill_name] = {"successes": 0, "failures": 0, "total": 0}

        self._skill_success_rates[skill_name]["total"] += 1
        if success:
            self._skill_success_rates[skill_name]["successes"] += 1
        else:
            self._skill_success_rates[skill_name]["failures"] += 1

    async def cleanup(self):
        """Cleanup resources."""
        logger.info(f"[DelegatingSubAgent:{self.id}] Cleaning up")
        self._bound_skills.clear()
        self._progress_callback = None
        self._result_callback = None
        self.status = AgentStatus.TERMINATED

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.

        Returns:
            Statistics dictionary
        """
        base_stats = super().get_stats()

        base_stats.update({
            "bound_skills": list(self._bound_skills.keys()),
            "skill_success_rates": self._skill_success_rates,
            "delegation_count": len(self._execution_history),
            "successful_delegations": sum(1 for r in self._execution_history if r.success)
        })

        return base_stats


def create_delegating_subagent(
    name: str,
    agent_type: str,
    description: str,
    capabilities: Optional[List[AgentCapability]] = None,
    llm_client: Optional[Any] = None,
    skill_registry: Optional[SkillRegistry] = None,
    tool_service: Optional[Any] = None
) -> DelegatingSubAgent:
    """Create a DelegatingSubAgent.

    Args:
        name: Agent name
        agent_type: Agent type
        description: Agent description
        capabilities: Agent capabilities
        llm_client: Optional LLM client
        skill_registry: Optional skill registry
        tool_service: Optional tool service

    Returns:
        DelegatingSubAgent instance
    """
    config = SubAgentConfig(
        name=name,
        agent_type=agent_type,
        description=description,
        capabilities=capabilities or []
    )

    return DelegatingSubAgent(
        config=config,
        llm_client=llm_client,
        skill_registry=skill_registry,
        tool_service=tool_service
    )
