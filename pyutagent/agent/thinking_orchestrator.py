"""Thinking Orchestrator for coordinating multiple thinking processes.

This module provides orchestration capabilities for the thinking engine,
enabling complex multi-step reasoning and coordination between different
thinking processes.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable

from ..core.component_registry import SimpleComponent, component
from .thinking_engine import (
    ThinkingEngine,
    ThinkingType,
    ThinkingPhase,
    ThinkingResult,
    ErrorThinkingResult,
    ReasoningStep,
    PredictedIssue,
)

logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Strategies for orchestrating thinking processes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    HIERARCHICAL = "hierarchical"


class OrchestrationState(Enum):
    """States of the orchestration process."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OrchestrationStep:
    """A step in the orchestration plan."""
    step_id: str
    name: str
    thinking_type: ThinkingType
    situation: str
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    result: Optional[ThinkingResult] = None
    state: OrchestrationState = OrchestrationState.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    """A plan for orchestrating multiple thinking processes."""
    plan_id: str
    steps: List[OrchestrationStep]
    strategy: OrchestrationStrategy
    state: OrchestrationState = OrchestrationState.IDLE
    current_step_index: int = 0
    results: Dict[str, ThinkingResult] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def get_step(self, step_id: str) -> Optional[OrchestrationStep]:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_completed_steps(self) -> List[OrchestrationStep]:
        return [s for s in self.steps if s.state == OrchestrationState.COMPLETED]

    def get_pending_steps(self) -> List[OrchestrationStep]:
        return [s for s in self.steps if s.state == OrchestrationState.IDLE]


@dataclass
class OrchestrationResult:
    """Result of an orchestration process."""
    plan_id: str
    success: bool
    results: Dict[str, ThinkingResult]
    final_conclusions: List[str]
    final_recommendations: List[str]
    overall_confidence: float
    duration: float
    steps_completed: int
    steps_total: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "final_conclusions": self.final_conclusions,
            "final_recommendations": self.final_recommendations,
            "overall_confidence": self.overall_confidence,
            "duration": self.duration,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "metadata": self.metadata,
        }


@component(
    component_id="thinking_orchestrator",
    dependencies=["thinking_engine"],
    description="Orchestrates multiple thinking processes for complex reasoning"
)
class ThinkingOrchestrator(SimpleComponent):
    """Orchestrates multiple thinking processes for complex reasoning.

    This orchestrator enables:
    - Multi-step reasoning workflows
    - Conditional thinking execution
    - Parallel thinking for independent analysis
    - Hierarchical thinking for complex problems
    - Result aggregation and synthesis
    """

    def __init__(
        self,
        thinking_engine: Optional[ThinkingEngine] = None,
        default_strategy: OrchestrationStrategy = OrchestrationStrategy.SEQUENTIAL,
        max_parallel_steps: int = 3,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ):
        super().__init__()
        self.thinking_engine = thinking_engine or ThinkingEngine()
        self.default_strategy = default_strategy
        self.max_parallel_steps = max_parallel_steps
        self.progress_callback = progress_callback

        self._plans: Dict[str, OrchestrationPlan] = {}
        self._orchestration_history: List[OrchestrationResult] = []

        logger.info(
            f"[ThinkingOrchestrator] Initialized - "
            f"Strategy: {default_strategy.name}, "
            f"Max parallel: {max_parallel_steps}"
        )

    def create_plan(
        self,
        situation: str,
        context: Dict[str, Any],
        strategy: Optional[OrchestrationStrategy] = None,
    ) -> OrchestrationPlan:
        strategy = strategy or self.default_strategy
        plan_id = str(uuid.uuid4())[:8]

        steps = self._generate_orchestration_steps(situation, context, strategy)

        plan = OrchestrationPlan(
            plan_id=plan_id,
            steps=steps,
            strategy=strategy,
        )

        self._plans[plan_id] = plan
        logger.info(f"[ThinkingOrchestrator] Created plan {plan_id} with {len(steps)} steps")

        return plan

    def _generate_orchestration_steps(
        self,
        situation: str,
        context: Dict[str, Any],
        strategy: OrchestrationStrategy,
    ) -> List[OrchestrationStep]:
        steps = []

        steps.append(OrchestrationStep(
            step_id=f"step_{uuid.uuid4().hex[:6]}",
            name="Initial Perception",
            thinking_type=ThinkingType.ANALYTICAL,
            situation=f"Perceive the situation: {situation[:200]}",
            dependencies=[],
        ))

        steps.append(OrchestrationStep(
            step_id=f"step_{uuid.uuid4().hex[:6]}",
            name="Deep Analysis",
            thinking_type=ThinkingType.CRITICAL,
            situation=f"Analyze critically: {situation[:200]}",
            dependencies=[steps[0].step_id],
        ))

        if context.get("error"):
            steps.append(OrchestrationStep(
                step_id=f"step_{uuid.uuid4().hex[:6]}",
                name="Error Diagnosis",
                thinking_type=ThinkingType.DIAGNOSTIC,
                situation="Diagnose the error and determine root cause",
                dependencies=[steps[-1].step_id],
            ))

        steps.append(OrchestrationStep(
            step_id=f"step_{uuid.uuid4().hex[:6]}",
            name="Strategic Planning",
            thinking_type=ThinkingType.STRATEGIC,
            situation="Plan the best approach forward",
            dependencies=[s.step_id for s in steps],
        ))

        steps.append(OrchestrationStep(
            step_id=f"step_{uuid.uuid4().hex[:6]}",
            name="Final Reflection",
            thinking_type=ThinkingType.REFLECTIVE,
            situation="Reflect on the analysis and finalize conclusions",
            dependencies=[steps[-1].step_id],
        ))

        return steps

    async def execute_plan(
        self,
        plan: OrchestrationPlan,
        context: Dict[str, Any],
    ) -> OrchestrationResult:
        start_time = time.time()
        plan.state = OrchestrationState.EXECUTING
        plan.start_time = datetime.now()

        self._notify_progress("ORCHESTRATING", f"Executing plan {plan.plan_id}")

        try:
            if plan.strategy == OrchestrationStrategy.SEQUENTIAL:
                await self._execute_sequential(plan, context)
            elif plan.strategy == OrchestrationStrategy.PARALLEL:
                await self._execute_parallel(plan, context)
            elif plan.strategy == OrchestrationStrategy.CONDITIONAL:
                await self._execute_conditional(plan, context)
            elif plan.strategy == OrchestrationStrategy.ITERATIVE:
                await self._execute_iterative(plan, context)
            else:
                await self._execute_sequential(plan, context)

            plan.state = OrchestrationState.COMPLETED
            plan.end_time = datetime.now()

        except Exception as e:
            logger.error(f"[ThinkingOrchestrator] Plan execution failed: {e}")
            plan.state = OrchestrationState.FAILED

        duration = time.time() - start_time

        final_conclusions = self._aggregate_conclusions(plan)
        final_recommendations = self._aggregate_recommendations(plan)
        overall_confidence = self._calculate_overall_confidence(plan)

        result = OrchestrationResult(
            plan_id=plan.plan_id,
            success=plan.state == OrchestrationState.COMPLETED,
            results=plan.results,
            final_conclusions=final_conclusions,
            final_recommendations=final_recommendations,
            overall_confidence=overall_confidence,
            duration=duration,
            steps_completed=len(plan.get_completed_steps()),
            steps_total=len(plan.steps),
        )

        self._orchestration_history.append(result)

        self._notify_progress(
            "ORCHESTRATION_COMPLETE",
            f"Completed {result.steps_completed}/{result.steps_total} steps"
        )

        logger.info(
            f"[ThinkingOrchestrator] Plan {plan.plan_id} completed - "
            f"Success: {result.success}, "
            f"Confidence: {overall_confidence:.2f}, "
            f"Duration: {duration:.2f}s"
        )

        return result

    async def _execute_sequential(
        self,
        plan: OrchestrationPlan,
        context: Dict[str, Any],
    ):
        for step in plan.steps:
            if not self._check_dependencies_met(step, plan):
                logger.warning(f"[ThinkingOrchestrator] Dependencies not met for step {step.name}")
                continue

            step.state = OrchestrationState.EXECUTING
            self._notify_progress("THINKING", f"Executing: {step.name}")

            try:
                result = await self.thinking_engine.think(
                    situation=step.situation,
                    context={**context, "previous_results": plan.results},
                    thinking_type=step.thinking_type,
                )

                step.result = result
                step.state = OrchestrationState.COMPLETED
                plan.results[step.step_id] = result

            except Exception as e:
                logger.error(f"[ThinkingOrchestrator] Step {step.name} failed: {e}")
                step.state = OrchestrationState.FAILED

    async def _execute_parallel(
        self,
        plan: OrchestrationPlan,
        context: Dict[str, Any],
    ):
        import asyncio

        independent_steps = [s for s in plan.steps if not s.dependencies]

        batches = []
        for i in range(0, len(independent_steps), self.max_parallel_steps):
            batches.append(independent_steps[i:i + self.max_parallel_steps])

        for batch in batches:
            tasks = []
            for step in batch:
                step.state = OrchestrationState.EXECUTING
                tasks.append(self._execute_step(step, context, plan))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(batch, results):
                if isinstance(result, Exception):
                    step.state = OrchestrationState.FAILED
                else:
                    step.state = OrchestrationState.COMPLETED

        dependent_steps = [s for s in plan.steps if s.dependencies]
        for step in dependent_steps:
            if self._check_dependencies_met(step, plan):
                await self._execute_step(step, context, plan)

    async def _execute_conditional(
        self,
        plan: OrchestrationPlan,
        context: Dict[str, Any],
    ):
        for step in plan.steps:
            if step.condition and not step.condition(plan.results):
                logger.info(f"[ThinkingOrchestrator] Skipping step {step.name} due to condition")
                continue

            if not self._check_dependencies_met(step, plan):
                continue

            await self._execute_step(step, context, plan)

    async def _execute_iterative(
        self,
        plan: OrchestrationPlan,
        context: Dict[str, Any],
    ):
        max_iterations = context.get("max_iterations", 5)
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self._notify_progress("ITERATING", f"Iteration {iteration}/{max_iterations}")

            for step in plan.steps:
                if not self._check_dependencies_met(step, plan):
                    continue

                await self._execute_step(step, context, plan)

            if self._check_completion_criteria(plan, context):
                logger.info(f"[ThinkingOrchestrator] Completion criteria met at iteration {iteration}")
                break

    async def _execute_step(
        self,
        step: OrchestrationStep,
        context: Dict[str, Any],
        plan: OrchestrationPlan,
    ):
        step.state = OrchestrationState.EXECUTING

        try:
            result = await self.thinking_engine.think(
                situation=step.situation,
                context={**context, "previous_results": plan.results},
                thinking_type=step.thinking_type,
            )

            step.result = result
            step.state = OrchestrationState.COMPLETED
            plan.results[step.step_id] = result

        except Exception as e:
            logger.error(f"[ThinkingOrchestrator] Step {step.name} failed: {e}")
            step.state = OrchestrationState.FAILED
            raise

    def _check_dependencies_met(
        self,
        step: OrchestrationStep,
        plan: OrchestrationPlan,
    ) -> bool:
        for dep_id in step.dependencies:
            dep_step = plan.get_step(dep_id)
            if not dep_step or dep_step.state != OrchestrationState.COMPLETED:
                return False
        return True

    def _check_completion_criteria(
        self,
        plan: OrchestrationPlan,
        context: Dict[str, Any],
    ) -> bool:
        if not plan.results:
            return False

        avg_confidence = sum(r.confidence for r in plan.results.values()) / len(plan.results)
        return avg_confidence > 0.8

    def _aggregate_conclusions(self, plan: OrchestrationPlan) -> List[str]:
        conclusions = []
        for result in plan.results.values():
            conclusions.extend(result.conclusions)
        return list(dict.fromkeys(conclusions))

    def _aggregate_recommendations(self, plan: OrchestrationPlan) -> List[str]:
        recommendations = []
        for result in plan.results.values():
            recommendations.extend(result.recommendations)
        return list(dict.fromkeys(recommendations))

    def _calculate_overall_confidence(self, plan: OrchestrationPlan) -> float:
        if not plan.results:
            return 0.0

        confidences = [r.confidence for r in plan.results.values()]
        return sum(confidences) / len(confidences)

    def _notify_progress(self, status: str, message: str):
        if self.progress_callback:
            try:
                self.progress_callback(status, message)
            except Exception as e:
                logger.warning(f"[ThinkingOrchestrator] Progress callback failed: {e}")

    async def orchestrate(
        self,
        situation: str,
        context: Dict[str, Any],
        strategy: Optional[OrchestrationStrategy] = None,
    ) -> OrchestrationResult:
        plan = self.create_plan(situation, context, strategy)
        return await self.execute_plan(plan, context)

    def get_plan(self, plan_id: str) -> Optional[OrchestrationPlan]:
        return self._plans.get(plan_id)

    def get_orchestration_stats(self) -> Dict[str, Any]:
        if not self._orchestration_history:
            return {"total": 0}

        return {
            "total": len(self._orchestration_history),
            "successful": sum(1 for r in self._orchestration_history if r.success),
            "average_confidence": sum(r.overall_confidence for r in self._orchestration_history) / len(self._orchestration_history),
            "average_duration": sum(r.duration for r in self._orchestration_history) / len(self._orchestration_history),
            "average_steps_completed": sum(r.steps_completed for r in self._orchestration_history) / len(self._orchestration_history),
        }

    def get_recent_orchestrations(self, limit: int = 10) -> List[OrchestrationResult]:
        return self._orchestration_history[-limit:]

    def clear_history(self):
        self._plans.clear()
        self._orchestration_history.clear()
        logger.info("[ThinkingOrchestrator] History cleared")
