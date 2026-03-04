"""Delegating Autonomous Loop with SubAgent delegation support.

This module provides:
- DelegatingAutonomousLoop: Enhanced autonomous loop with delegation
- Task decomposition and delegation
- Parallel subtask execution
- Result integration
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from .enhanced_autonomous_loop import (
    EnhancedAutonomousLoop,
    DecisionStrategy,
    ExecutionContext,
    LLMThought,
    LearningEntry
)
from .autonomous_loop import LoopState, Observation, Thought, LoopResult
from .tool_service import AgentToolService
from .hierarchical_planner import (
    HierarchicalTaskPlanner,
    Subtask,
    SubtaskType,
    SubtaskStatus,
    TaskTree,
    ExecutionPlan
)
from .delegation_mixin import AgentDelegationMixin, DelegationOptions
from .result_aggregator import ResultAggregator, AggregationStrategy

logger = logging.getLogger(__name__)


class DelegationDecision(Enum):
    """Decision on whether to delegate."""
    DELEGATE = "delegate"
    EXECUTE_LOCALLY = "execute_locally"
    DELEGATE_PARALLEL = "delegate_parallel"
    SKIP = "skip"


@dataclass
class DelegableSubtask:
    """A subtask identified for potential delegation."""
    subtask: Subtask
    decision: DelegationDecision
    reason: str
    estimated_benefit: float = 0.0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class DelegationPlan:
    """Plan for delegating subtasks."""
    subtasks_to_delegate: List[Subtask] = field(default_factory=list)
    subtasks_to_execute: List[Dict[str, Any]] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    estimated_speedup: float = 1.0


@dataclass
class DelegationResult:
    """Result of delegation execution."""
    success: bool
    total_subtasks: int
    delegated_subtasks: int
    local_subtasks: int
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: int = 0


class DelegatingAutonomousLoop(EnhancedAutonomousLoop):
    """Enhanced autonomous loop with SubAgent delegation.

    Extends EnhancedAutonomousLoop with:
    - Task decomposition for delegation
    - Parallel subtask execution
    - Intelligent delegation decisions
    - Result integration from multiple agents
    """

    def __init__(
        self,
        tool_service: AgentToolService,
        llm_client: Any,
        max_iterations: int = 10,
        confidence_threshold: float = 0.8,
        user_interruptible: bool = True,
        decision_strategy: DecisionStrategy = DecisionStrategy.HYBRID,
        enable_self_correction: bool = True,
        enable_delegation: bool = True,
        max_parallel_delegations: int = 5,
        delegation_threshold: float = 0.6
    ):
        """Initialize DelegatingAutonomousLoop.

        Args:
            tool_service: Tool service for executing tools
            llm_client: LLM client for reasoning
            max_iterations: Maximum iterations before stopping
            confidence_threshold: Confidence threshold for completion
            user_interruptible: Whether user can interrupt
            decision_strategy: Strategy for decisions
            enable_self_correction: Whether to enable self-correction
            enable_delegation: Whether to enable delegation
            max_parallel_delegations: Maximum parallel delegations
            delegation_threshold: Threshold for delegation decision
        """
        super().__init__(
            tool_service=tool_service,
            llm_client=llm_client,
            max_iterations=max_iterations,
            confidence_threshold=confidence_threshold,
            user_interruptible=user_interruptible,
            decision_strategy=decision_strategy,
            enable_self_correction=enable_self_correction
        )

        self.enable_delegation = enable_delegation
        self.max_parallel_delegations = max_parallel_delegations
        self.delegation_threshold = delegation_threshold

        self._task_planner = HierarchicalTaskPlanner(llm_client=llm_client)
        self._delegation_mixin = AgentDelegationMixin()
        self._result_aggregator = ResultAggregator()

        self._delegation_history: List[DelegationResult] = []
        self._active_delegations: Dict[str, Subtask] = {}

        self._delegation_mixin.init_delegation(
            llm_client=llm_client,
            tool_service=tool_service
        )

        logger.info(f"[DelegatingAutonomousLoop] Initialized (delegation_enabled={enable_delegation})")

    async def _think(
        self,
        task: str,
        observation: Observation,
        context: Dict[str, Any]
    ) -> Thought:
        """Think with delegation awareness.

        Args:
            task: Current task
            observation: Current observation
            context: Current context

        Returns:
            Thought with delegation plan
        """
        thought = await super()._think(task, observation, context)

        if self.enable_delegation and thought.confidence < self.delegation_threshold:
            delegation_plan = await self._plan_delegation(task, thought, context)
            if delegation_plan.subtasks_to_delegate:
                thought.plan = self._integrate_delegation_plan(
                    thought.plan, delegation_plan
                )
                thought.reasoning += f" | Delegation planned: {len(delegation_plan.subtasks_to_delegate)} subtasks"

        return thought

    async def _plan_delegation(
        self,
        task: str,
        thought: Thought,
        context: Dict[str, Any]
    ) -> DelegationPlan:
        """Plan task delegation.

        Args:
            task: Current task
            thought: Current thought
            context: Current context

        Returns:
            DelegationPlan
        """
        plan = DelegationPlan()

        try:
            task_tree = await self._task_planner.decompose(task, context)

            for subtask in task_tree.get_all_subtasks():
                decision = self._should_delegate(subtask, context)

                if decision.decision == DelegationDecision.DELEGATE:
                    plan.subtasks_to_delegate.append(subtask)
                elif decision.decision == DelegationDecision.DELEGATE_PARALLEL:
                    plan.subtasks_to_delegate.append(subtask)
                else:
                    plan.subtasks_to_execute.append({
                        "subtask": subtask,
                        "action": self._subtask_to_action(subtask)
                    })

            dependency_graph = self._task_planner.analyze_dependencies(
                task_tree.get_all_subtasks()
            )
            plan.parallel_groups = self._task_planner.identify_parallel_tasks(
                dependency_graph
            )

            if plan.subtasks_to_delegate:
                plan.estimated_speedup = self._estimate_speedup(
                    len(plan.subtasks_to_delegate),
                    len(plan.subtasks_to_execute)
                )

        except Exception as e:
            logger.error(f"[DelegatingAutonomousLoop] Delegation planning failed: {e}")

        return plan

    def _should_delegate(
        self,
        subtask: Subtask,
        context: Dict[str, Any]
    ) -> DelegableSubtask:
        """Determine if a subtask should be delegated.

        Args:
            subtask: Subtask to evaluate
            context: Current context

        Returns:
            DelegableSubtask with decision
        """
        delegable_types = {
            SubtaskType.TESTING,
            SubtaskType.GENERATION,
            SubtaskType.ANALYSIS,
            SubtaskType.DOCUMENTATION
        }

        if subtask.task_type in delegable_types:
            return DelegableSubtask(
                subtask=subtask,
                decision=DelegationDecision.DELEGATE,
                reason=f"Subtask type {subtask.task_type.value} is suitable for delegation",
                estimated_benefit=0.7
            )

        if subtask.estimated_complexity >= 3:
            return DelegableSubtask(
                subtask=subtask,
                decision=DelegationDecision.DELEGATE,
                reason="High complexity task benefits from delegation",
                estimated_benefit=0.5
            )

        if subtask.task_type in {SubtaskType.FIXING, SubtaskType.REVIEW}:
            return DelegableSubtask(
                subtask=subtask,
                decision=DelegationDecision.EXECUTE_LOCALLY,
                reason="Critical tasks should be executed locally for control",
                estimated_benefit=0.3
            )

        return DelegableSubtask(
            subtask=subtask,
            decision=DelegationDecision.EXECUTE_LOCALLY,
            reason="Default to local execution",
            estimated_benefit=0.0
        )

    def _subtask_to_action(self, subtask: Subtask) -> Dict[str, Any]:
        """Convert subtask to action plan.

        Args:
            subtask: Subtask to convert

        Returns:
            Action dictionary
        """
        tool_mapping = {
            SubtaskType.ANALYSIS: "analyze_code",
            SubtaskType.GENERATION: "generate_code",
            SubtaskType.TESTING: "run_tests",
            SubtaskType.FIXING: "fix_errors",
            SubtaskType.REFACTORING: "refactor_code",
            SubtaskType.DOCUMENTATION: "generate_docs",
            SubtaskType.REVIEW: "review_code",
            SubtaskType.CUSTOM: "execute"
        }

        return {
            "tool_name": tool_mapping.get(subtask.task_type, "execute"),
            "parameters": subtask.input_data,
            "expected_outcome": subtask.description
        }

    def _integrate_delegation_plan(
        self,
        original_plan: List[Dict[str, Any]],
        delegation_plan: DelegationPlan
    ) -> List[Dict[str, Any]]:
        """Integrate delegation plan with original plan.

        Args:
            original_plan: Original action plan
            delegation_plan: Delegation plan

        Returns:
            Integrated plan
        """
        integrated = []

        for subtask in delegation_plan.subtasks_to_delegate:
            integrated.append({
                "tool_name": "delegate_subtask",
                "parameters": {
                    "subtask_id": subtask.id,
                    "subtask_name": subtask.name,
                    "subtask_description": subtask.description,
                    "subtask_input": subtask.input_data
                },
                "expected_outcome": f"Delegate {subtask.name} to SubAgent",
                "is_delegation": True
            })

        for action in delegation_plan.subtasks_to_execute:
            integrated.append(action["action"])

        return integrated

    def _estimate_speedup(
        self,
        delegated_count: int,
        local_count: int
    ) -> float:
        """Estimate speedup from delegation.

        Args:
            delegated_count: Number of delegated tasks
            local_count: Number of local tasks

        Returns:
            Estimated speedup factor
        """
        if delegated_count == 0:
            return 1.0

        parallel_factor = min(delegated_count, self.max_parallel_delegations)
        return 1.0 + (parallel_factor * 0.3)

    async def _act(
        self,
        action_plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Execute action with delegation support.

        Args:
            action_plan: Action to execute
            context: Current context

        Returns:
            Action result
        """
        if action_plan.get("is_delegation"):
            return await self._execute_delegation(action_plan, context)

        return await super()._act(action_plan, context)

    async def _execute_delegation(
        self,
        action_plan: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Execute a delegation action.

        Args:
            action_plan: Delegation action
            context: Current context

        Returns:
            Delegation result
        """
        params = action_plan.get("parameters", {})

        subtask = Subtask.create(
            name=params.get("subtask_name", "Unnamed"),
            description=params.get("subtask_description", ""),
            task_type=SubtaskType.CUSTOM,
            input_data=params.get("subtask_input", {})
        )

        self._active_delegations[subtask.id] = subtask

        try:
            options = DelegationOptions(
                timeout=300,
                max_retries=2
            )

            result = await self._delegation_mixin.delegate_subtask(subtask, options)

            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "subtask_id": subtask.id,
                "agent_id": result.agent_id
            }

        finally:
            if subtask.id in self._active_delegations:
                del self._active_delegations[subtask.id]

    async def _identify_delegable_subtasks(
        self,
        thought: Thought
    ) -> List[Subtask]:
        """Identify subtasks that can be delegated.

        Args:
            thought: Current thought

        Returns:
            List of delegable subtasks
        """
        delegable = []

        for action in thought.plan:
            if action.get("is_delegation"):
                params = action.get("parameters", {})
                subtask = Subtask.create(
                    name=params.get("subtask_name", "Unnamed"),
                    description=params.get("subtask_description", ""),
                    task_type=SubtaskType.CUSTOM,
                    input_data=params.get("subtask_input", {})
                )
                delegable.append(subtask)

        return delegable

    async def _delegate_subtasks(
        self,
        subtasks: List[Subtask]
    ) -> List[Any]:
        """Delegate multiple subtasks.

        Args:
            subtasks: Subtasks to delegate

        Returns:
            List of delegation results
        """
        results = await self._delegation_mixin.delegate_parallel(
            subtasks,
            max_concurrent=self.max_parallel_delegations
        )

        return results

    async def _integrate_delegation_results(
        self,
        results: List[Any]
    ) -> Dict[str, Any]:
        """Integrate results from delegations.

        Args:
            results: Delegation results

        Returns:
            Integrated result
        """
        aggregated = await self._result_aggregator.aggregate(
            results,
            strategy=AggregationStrategy.ALL_SUCCESS
        )

        return {
            "success": aggregated.success,
            "merged_result": aggregated.merged_result,
            "total_tasks": aggregated.total_tasks,
            "successful_tasks": aggregated.successful_tasks,
            "inconsistencies": len(aggregated.inconsistencies)
        }

    def get_delegation_stats(self) -> Dict[str, Any]:
        """Get delegation statistics.

        Returns:
            Statistics dictionary
        """
        base_stats = self.get_learning_summary()

        delegation_stats = self._delegation_mixin.get_delegation_stats()

        base_stats.update({
            "delegation_enabled": self.enable_delegation,
            "max_parallel_delegations": self.max_parallel_delegations,
            "active_delegations": len(self._active_delegations),
            "delegation_history_count": len(self._delegation_history),
            **delegation_stats
        })

        return base_stats


def create_delegating_autonomous_loop(
    tool_service: AgentToolService,
    llm_client: Any,
    max_iterations: int = 10,
    enable_delegation: bool = True
) -> DelegatingAutonomousLoop:
    """Create a DelegatingAutonomousLoop.

    Args:
        tool_service: Tool service
        llm_client: LLM client
        max_iterations: Maximum iterations
        enable_delegation: Whether to enable delegation

    Returns:
        DelegatingAutonomousLoop instance
    """
    return DelegatingAutonomousLoop(
        tool_service=tool_service,
        llm_client=llm_client,
        max_iterations=max_iterations,
        enable_delegation=enable_delegation
    )
