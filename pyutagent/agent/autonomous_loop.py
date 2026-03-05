"""Autonomous Loop - Observe-Think-Act-Verify-Learn cycle with self-correction.

This module implements a comprehensive autonomous decision loop inspired by
top-tier coding agents like Claude Code and Cursor Agent Mode:

Core Cycle:
1. Observe: Gather current state and environment information
2. Think: Analyze situation and plan next actions
3. Act: Execute tools and actions
4. Verify: Check results against expectations
5. Learn: Update memory and improve future decisions

Features:
- Self-correction on failures
- Risk assessment before actions
- Maximum iteration limits
- User interrupt support
- Comprehensive logging
- Learning from execution history

Example:
    >>> loop = AutonomousLoop(tool_service, llm_client)
    >>> result = await loop.run("Generate tests for UserService.java")
    >>> print(f"Success: {result.success}, Iterations: {result.iterations}")
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from uuid import uuid4

from ..core.protocols import AgentState
from ..tools.tool import ToolResult

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LoopPhase(Enum):
    """Phases of the autonomous loop.
    
    The loop follows a strict sequence:
    IDLE -> OBSERVING -> THINKING -> ACTING -> VERIFYING -> LEARNING -> COMPLETED/FAILED
    """
    IDLE = auto()
    OBSERVING = auto()
    THINKING = auto()
    ACTING = auto()
    VERIFYING = auto()
    LEARNING = auto()
    CORRECTING = auto()  # Self-correction phase
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()
    INTERRUPTED = auto()


class RiskLevel(Enum):
    """Risk levels for actions."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionPriority(Enum):
    """Priority levels for actions."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class LoopState:
    """Current state of the autonomous loop.
    
    Attributes:
        phase: Current phase of the loop
        iteration: Current iteration number
        task: Current task description
        context: Execution context
        timestamp: When this state was recorded
        metadata: Additional state metadata
    """
    phase: LoopPhase
    iteration: int
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "phase": self.phase.name,
            "iteration": self.iteration,
            "task": self.task,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Observation:
    """Observation of the current environment state.
    
    Attributes:
        timestamp: When the observation was made
        state_summary: Human-readable summary of the state
        relevant_data: Key data observed
        tool_results: Results from recent tool executions
        environment_info: System/environment information
        metadata: Additional observation metadata
    """
    timestamp: datetime
    state_summary: str
    relevant_data: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "state_summary": self.state_summary,
            "relevant_data": self.relevant_data,
            "tool_results": self.tool_results,
            "environment_info": self.environment_info,
            "metadata": self.metadata
        }


@dataclass
class Thought:
    """Thought/decision from the thinking phase.
    
    Attributes:
        timestamp: When the thought was generated
        reasoning: Detailed reasoning process
        decision: Final decision made
        confidence: Confidence level (0.0-1.0)
        plan: List of planned actions
        risk_assessment: Risk evaluation
        alternative_approaches: Alternative plans considered
        expected_outcome: Expected result
        metadata: Additional thought metadata
    """
    timestamp: datetime
    reasoning: str
    decision: str
    confidence: float
    plan: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: str = ""
    risk_level: RiskLevel = RiskLevel.NONE
    alternative_approaches: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert thought to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reasoning": self.reasoning,
            "decision": self.decision,
            "confidence": self.confidence,
            "plan": self.plan,
            "risk_assessment": self.risk_assessment,
            "risk_level": self.risk_level.value,
            "alternative_approaches": self.alternative_approaches,
            "expected_outcome": self.expected_outcome,
            "metadata": self.metadata
        }


@dataclass
class Action:
    """Action to be executed.
    
    Attributes:
        tool_name: Name of the tool to execute
        parameters: Tool parameters
        expected_outcome: Expected result
        priority: Action priority
        risk_level: Assessed risk level
        timeout: Execution timeout in seconds
        retry_count: Number of retry attempts
        metadata: Additional action metadata
    """
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    priority: ActionPriority = ActionPriority.MEDIUM
    risk_level: RiskLevel = RiskLevel.NONE
    timeout: int = 60
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "expected_outcome": self.expected_outcome,
            "priority": self.priority.name,
            "risk_level": self.risk_level.value,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "metadata": self.metadata
        }


@dataclass
class Verification:
    """Verification result of an action.
    
    Attributes:
        success: Whether the verification passed
        actual_outcome: What actually happened
        expected_outcome: What was expected
        differences: List of differences found
        confidence: Confidence in the verification
        suggestions: Suggestions for improvement
        metadata: Additional verification metadata
    """
    success: bool
    actual_outcome: str
    expected_outcome: str
    differences: List[str] = field(default_factory=list)
    confidence: float = 1.0
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert verification to dictionary."""
        return {
            "success": self.success,
            "actual_outcome": self.actual_outcome,
            "expected_outcome": self.expected_outcome,
            "differences": self.differences,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "metadata": self.metadata
        }


@dataclass
class LearningEntry:
    """Entry in the learning history.
    
    Attributes:
        timestamp: When the learning occurred
        situation: Description of the situation
        action_taken: What action was taken
        outcome: What was the outcome
        lesson: Lesson learned
        success: Whether the action was successful
        iteration: Which iteration this occurred in
        metadata: Additional learning metadata
    """
    timestamp: datetime
    situation: str
    action_taken: str
    outcome: str
    lesson: str
    success: bool
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert learning entry to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "situation": self.situation,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "lesson": self.lesson,
            "success": self.success,
            "iteration": self.iteration,
            "metadata": self.metadata
        }


@dataclass
class RiskAssessment:
    """Risk assessment for an action or plan.
    
    Attributes:
        level: Overall risk level
        factors: List of risk factors
        mitigation_strategies: Strategies to mitigate risks
        requires_approval: Whether user approval is required
        confidence: Confidence in the assessment
    """
    level: RiskLevel
    factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    requires_approval: bool = False
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert risk assessment to dictionary."""
        return {
            "level": self.level.value,
            "factors": self.factors,
            "mitigation_strategies": self.mitigation_strategies,
            "requires_approval": self.requires_approval,
            "confidence": self.confidence
        }


@dataclass
class LoopMetrics:
    """Metrics for the autonomous loop execution.
    
    Attributes:
        total_iterations: Total number of iterations
        successful_actions: Number of successful actions
        failed_actions: Number of failed actions
        self_corrections: Number of self-corrections performed
        total_execution_time_ms: Total execution time in milliseconds
        average_iteration_time_ms: Average time per iteration
        tool_usage_counts: Count of tool usage
        risk_distribution: Distribution of risk levels encountered
    """
    total_iterations: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    self_corrections: int = 0
    total_execution_time_ms: int = 0
    average_iteration_time_ms: float = 0.0
    tool_usage_counts: Dict[str, int] = field(default_factory=dict)
    risk_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_iterations": self.total_iterations,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "self_corrections": self.self_corrections,
            "total_execution_time_ms": self.total_execution_time_ms,
            "average_iteration_time_ms": self.average_iteration_time_ms,
            "tool_usage_counts": self.tool_usage_counts,
            "risk_distribution": self.risk_distribution
        }


@dataclass
class LoopResult:
    """Result of the autonomous loop execution.
    
    Attributes:
        success: Whether the loop completed successfully
        iterations: Number of iterations executed
        final_state: Final state of the loop
        observations: List of all observations
        thoughts: List of all thoughts
        actions_taken: List of all actions taken
        verifications: List of all verifications
        learnings: List of all learning entries
        metrics: Execution metrics
        error: Error message if failed
        execution_time_ms: Total execution time in milliseconds
        metadata: Additional result metadata
    """
    success: bool
    iterations: int
    final_state: LoopPhase
    observations: List[Observation] = field(default_factory=list)
    thoughts: List[Thought] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    verifications: List[Verification] = field(default_factory=list)
    learnings: List[LearningEntry] = field(default_factory=list)
    metrics: LoopMetrics = field(default_factory=LoopMetrics)
    error: Optional[str] = None
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "iterations": self.iterations,
            "final_state": self.final_state.name,
            "observations": [o.to_dict() for o in self.observations],
            "thoughts": [t.to_dict() for t in self.thoughts],
            "actions_taken": self.actions_taken,
            "verifications": [v.to_dict() for v in self.verifications],
            "learnings": [l.to_dict() for l in self.learnings],
            "metrics": self.metrics.to_dict(),
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the result."""
        status = "✓ Success" if self.success else "✗ Failed"
        summary = f"{status} | Iterations: {self.iterations} | Time: {self.execution_time_ms}ms"
        if self.error:
            summary += f"\nError: {self.error}"
        return summary


@dataclass
class LoopConfig:
    """Configuration for the autonomous loop.
    
    Attributes:
        max_iterations: Maximum number of iterations
        confidence_threshold: Confidence threshold to consider task complete
        user_interruptible: Whether user can interrupt the loop
        enable_self_correction: Whether to enable self-correction
        enable_learning: Whether to enable learning
        enable_risk_assessment: Whether to enable risk assessment
        max_retries_per_action: Maximum retries per action
        action_timeout: Default action timeout in seconds
        pause_between_iterations: Whether to pause between iterations
        log_level: Logging level
    """
    max_iterations: int = 10
    confidence_threshold: float = 0.8
    user_interruptible: bool = True
    enable_self_correction: bool = True
    enable_learning: bool = True
    enable_risk_assessment: bool = True
    max_retries_per_action: int = 3
    action_timeout: int = 60
    pause_between_iterations: bool = False
    pause_duration_ms: int = 100
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "confidence_threshold": self.confidence_threshold,
            "user_interruptible": self.user_interruptible,
            "enable_self_correction": self.enable_self_correction,
            "enable_learning": self.enable_learning,
            "enable_risk_assessment": self.enable_risk_assessment,
            "max_retries_per_action": self.max_retries_per_action,
            "action_timeout": self.action_timeout,
            "pause_between_iterations": self.pause_between_iterations,
            "pause_duration_ms": self.pause_duration_ms,
            "log_level": self.log_level
        }


class AutonomousLoop(ABC):
    """Abstract base class for autonomous decision loops.
    
    Implements the Observe-Think-Act-Verify-Learn cycle with:
    - Self-correction on failures
    - Risk assessment
    - Learning from history
    - User interrupt support
    - Comprehensive logging
    
    Subclasses must implement:
    - _observe(): Gather environment state
    - _think(): Analyze and plan
    - _act(): Execute actions
    - _verify(): Verify results
    
    Example:
        >>> class MyLoop(AutonomousLoop):
        ...     async def _observe(self, task, context):
        ...         return Observation(...)
        ...     async def _think(self, task, observation, context):
        ...         return Thought(...)
        ...     async def _act(self, action, context):
        ...         return result
        ...     async def _verify(self, result, expected, context):
        ...         return Verification(...)
    """
    
    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        result_callback: Optional[Callable[[LoopResult], None]] = None
    ):
        """Initialize the autonomous loop.
        
        Args:
            config: Loop configuration
            progress_callback: Callback for progress updates (phase, progress, message)
            result_callback: Callback for final result
        """
        self.config = config or LoopConfig()
        self.id = str(uuid4())
        
        # State management
        self._state = LoopState(
            phase=LoopPhase.IDLE,
            iteration=0,
            task=""
        )
        self._current_iteration = 0
        
        # Control flags
        self._stop_requested = False
        self._pause_requested = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        
        # Callbacks
        self._progress_callback = progress_callback
        self._result_callback = result_callback
        
        # History tracking
        self._observations: List[Observation] = []
        self._thoughts: List[Thought] = []
        self._actions_taken: List[Dict[str, Any]] = []
        self._verifications: List[Verification] = []
        self._learnings: List[LearningEntry] = []
        self._errors: List[str] = []
        
        # Metrics
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._metrics = LoopMetrics()
        
        # Execution history
        self._execution_history: List[LoopResult] = []
        
        # Set logging level
        logger.setLevel(getattr(logging, self.config.log_level))
        
        logger.info(f"[AutonomousLoop:{self.id}] Initialized with config: {self.config.to_dict()}")
    
    @property
    def state(self) -> LoopState:
        """Get current loop state."""
        return self._state
    
    @property
    def current_iteration(self) -> int:
        """Get current iteration number."""
        return self._current_iteration
    
    @property
    def is_running(self) -> bool:
        """Check if loop is currently running."""
        return self._state.phase not in [
            LoopPhase.IDLE, LoopPhase.COMPLETED, LoopPhase.FAILED, LoopPhase.INTERRUPTED
        ]
    
    def _update_state(self, phase: LoopPhase, **kwargs) -> None:
        """Update loop state.
        
        Args:
            phase: New phase
            **kwargs: Additional state attributes
        """
        old_phase = self._state.phase
        self._state = LoopState(
            phase=phase,
            iteration=self._current_iteration,
            task=self._state.task,
            context=kwargs.get("context", self._state.context),
            metadata=kwargs.get("metadata", {})
        )
        logger.debug(f"[AutonomousLoop:{self.id}] State: {old_phase.name} -> {phase.name}")
    
    def _report_progress(self, phase: str, progress: float, message: str) -> None:
        """Report progress via callback.
        
        Args:
            phase: Current phase
            progress: Progress value (0.0-1.0)
            message: Progress message
        """
        if self._progress_callback:
            try:
                self._progress_callback(phase, progress, message)
            except Exception as e:
                logger.warning(f"[AutonomousLoop:{self.id}] Progress callback failed: {e}")
    
    def _should_continue(self) -> bool:
        """Check if the loop should continue.
        
        Returns:
            True if loop should continue, False otherwise
        """
        if self._stop_requested:
            logger.info(f"[AutonomousLoop:{self.id}] Stop requested, ending loop")
            return False
        
        if self._current_iteration >= self.config.max_iterations:
            logger.info(f"[AutonomousLoop:{self.id}] Max iterations reached")
            return False
        
        return True
    
    async def _check_pause(self) -> None:
        """Check if paused and wait until resumed."""
        if self._pause_requested or not self._pause_event.is_set():
            self._update_state(LoopPhase.PAUSED)
            self._report_progress("paused", 0.0, "Execution paused")
            logger.info(f"[AutonomousLoop:{self.id}] Paused, waiting for resume...")
            await self._pause_event.wait()
            logger.info(f"[AutonomousLoop:{self.id}] Resumed")
    
    def stop(self) -> None:
        """Request the loop to stop."""
        self._stop_requested = True
        logger.info(f"[AutonomousLoop:{self.id}] Stop requested")
    
    def pause(self) -> None:
        """Request the loop to pause."""
        self._pause_requested = True
        self._pause_event.clear()
        logger.info(f"[AutonomousLoop:{self.id}] Pause requested")
    
    def resume(self) -> None:
        """Resume the loop from pause."""
        self._pause_requested = False
        self._pause_event.set()
        logger.info(f"[AutonomousLoop:{self.id}] Resume requested")
    
    def reset(self) -> None:
        """Reset the loop to initial state."""
        self._stop_requested = False
        self._pause_requested = False
        self._pause_event.set()
        self._current_iteration = 0
        self._update_state(LoopPhase.IDLE)
        
        self._observations.clear()
        self._thoughts.clear()
        self._actions_taken.clear()
        self._verifications.clear()
        self._learnings.clear()
        self._errors.clear()
        
        self._metrics = LoopMetrics()
        
        logger.info(f"[AutonomousLoop:{self.id}] Reset to initial state")
    
    @abstractmethod
    async def _observe(self, task: str, context: Dict[str, Any]) -> Observation:
        """Observe the current environment state.
        
        Args:
            task: Current task
            context: Execution context
            
        Returns:
            Observation of current state
        """
        pass
    
    @abstractmethod
    async def _think(
        self,
        task: str,
        observation: Observation,
        context: Dict[str, Any]
    ) -> Thought:
        """Think about the next action.
        
        Args:
            task: Current task
            observation: Current observation
            context: Execution context
            
        Returns:
            Thought with plan
        """
        pass
    
    @abstractmethod
    async def _act(self, action: Action, context: Dict[str, Any]) -> Any:
        """Execute an action.
        
        Args:
            action: Action to execute
            context: Execution context
            
        Returns:
            Action result
        """
        pass
    
    @abstractmethod
    async def _verify(
        self,
        result: Any,
        expected: str,
        context: Dict[str, Any]
    ) -> Verification:
        """Verify the result of an action.
        
        Args:
            result: Action result
            expected: Expected outcome
            context: Execution context
            
        Returns:
            Verification result
        """
        pass
    
    async def _learn(self, entry: LearningEntry) -> None:
        """Learn from an execution.
        
        Args:
            entry: Learning entry
        """
        if self.config.enable_learning:
            self._learnings.append(entry)
            logger.info(f"[AutonomousLoop:{self.id}] Learned: {entry.lesson[:100]}...")
    
    async def _self_correct(
        self,
        error: str,
        context: Dict[str, Any]
    ) -> Optional[Thought]:
        """Self-correct based on an error.
        
        Args:
            error: Error message
            context: Execution context
            
        Returns:
            Corrected thought or None
        """
        if not self.config.enable_self_correction:
            return None
        
        self._errors.append(error)
        self._metrics.self_corrections += 1
        
        logger.info(f"[AutonomousLoop:{self.id}] Self-correcting: {error[:100]}...")
        
        # Create correction thought
        correction = Thought(
            timestamp=datetime.now(),
            reasoning=f"Self-correction triggered by error: {error}",
            decision="retry_with_adjustments",
            confidence=0.5,
            plan=[{"type": "retry", "error_context": error}],
            risk_assessment="Retrying after error",
            risk_level=RiskLevel.MEDIUM
        )
        
        return correction
    
    def _assess_risk(self, action: Action) -> RiskAssessment:
        """Assess risk of an action.
        
        Args:
            action: Action to assess
            
        Returns:
            Risk assessment
        """
        if not self.config.enable_risk_assessment:
            return RiskAssessment(level=RiskLevel.NONE)
        
        factors = []
        mitigation = []
        requires_approval = False
        
        # Assess based on tool name
        high_risk_tools = ["delete", "remove", "drop", "truncate"]
        medium_risk_tools = ["write", "edit", "modify", "update"]
        
        tool_lower = action.tool_name.lower()
        
        if any(risk in tool_lower for risk in high_risk_tools):
            factors.append(f"High-risk tool: {action.tool_name}")
            requires_approval = True
            mitigation.append("Review before execution")
        elif any(risk in tool_lower for risk in medium_risk_tools):
            factors.append(f"Medium-risk tool: {action.tool_name}")
            mitigation.append("Backup before modification")
        
        # Assess based on parameters
        if "force" in str(action.parameters).lower():
            factors.append("Force flag detected")
            requires_approval = True
        
        # Determine level
        if requires_approval or len(factors) > 2:
            level = RiskLevel.HIGH
        elif len(factors) > 0:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.LOW
        
        return RiskAssessment(
            level=level,
            factors=factors,
            mitigation_strategies=mitigation,
            requires_approval=requires_approval
        )
    
    async def _execute_single_action(
        self,
        action: Action,
        context: Dict[str, Any]
    ) -> Tuple[Any, Verification]:
        """Execute a single action with verification.
        
        Args:
            action: Action to execute
            context: Execution context
            
        Returns:
            Tuple of (result, verification)
        """
        # Assess risk
        risk = self._assess_risk(action)
        
        if risk.requires_approval and self.config.user_interruptible:
            logger.warning(f"[AutonomousLoop:{self.id}] High-risk action requires approval: {action.tool_name}")
            # In a real implementation, this would prompt the user
        
        # Execute action
        logger.debug(f"[AutonomousLoop:{self.id}] Executing: {action.tool_name}")
        
        try:
            result = await self._act(action, context)
            
            # Track metrics
            self._metrics.tool_usage_counts[action.tool_name] = \
                self._metrics.tool_usage_counts.get(action.tool_name, 0) + 1
            
            # Verify result
            verification = await self._verify(result, action.expected_outcome, context)
            
            if verification.success:
                self._metrics.successful_actions += 1
            else:
                self._metrics.failed_actions += 1
            
            return result, verification
            
        except Exception as e:
            logger.error(f"[AutonomousLoop:{self.id}] Action failed: {e}")
            self._metrics.failed_actions += 1
            
            error_result = ToolResult(success=False, error=str(e))
            verification = Verification(
                success=False,
                actual_outcome=str(e),
                expected_outcome=action.expected_outcome,
                differences=[str(e)]
            )
            
            return error_result, verification
    
    async def _execute_iteration(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> bool:
        """Execute a single iteration of the loop.
        
        Args:
            task: Current task
            context: Execution context
            
        Returns:
            True if task is complete, False otherwise
        """
        iteration_start = time.time()
        self._current_iteration += 1
        progress = self._current_iteration / self.config.max_iterations
        
        logger.info(f"[AutonomousLoop:{self.id}] Starting iteration {self._current_iteration}")
        
        # Observe
        self._update_state(LoopPhase.OBSERVING)
        self._report_progress("observing", progress, f"Iteration {self._current_iteration}: Observing...")
        
        observation = await self._observe(task, context)
        self._observations.append(observation)
        
        await self._check_pause()
        
        # Think
        self._update_state(LoopPhase.THINKING)
        self._report_progress("thinking", progress, "Analyzing and planning...")
        
        thought = await self._think(task, observation, context)
        self._thoughts.append(thought)
        
        # Check if confidence threshold reached
        if thought.confidence >= self.config.confidence_threshold:
            logger.info(f"[AutonomousLoop:{self.id}] Confidence threshold reached: {thought.confidence}")
            self._report_progress("completed", 1.0, "Confidence threshold reached")
            return True
        
        await self._check_pause()
        
        # Act
        self._update_state(LoopPhase.ACTING)
        
        for action_plan in thought.plan:
            await self._check_pause()
            
            action = Action(
                tool_name=action_plan.get("tool_name", "unknown"),
                parameters=action_plan.get("parameters", {}),
                expected_outcome=action_plan.get("expected_outcome", ""),
                priority=ActionPriority[action_plan.get("priority", "MEDIUM")],
                timeout=action_plan.get("timeout", self.config.action_timeout)
            )
            
            self._report_progress("acting", progress, f"Executing: {action.tool_name}")
            
            result, verification = await self._execute_single_action(action, context)
            
            # Record action
            action_record = {
                "tool_name": action.tool_name,
                "parameters": action.parameters,
                "success": verification.success,
                "result": str(result)[:500] if result else None,
                "iteration": self._current_iteration,
                "timestamp": datetime.now().isoformat()
            }
            self._actions_taken.append(action_record)
            self._verifications.append(verification)
            
            # Verify
            self._update_state(LoopPhase.VERIFYING)
            
            if verification.success:
                # Learn from success
                if self.config.enable_learning:
                    await self._learn(LearningEntry(
                        timestamp=datetime.now(),
                        situation=observation.state_summary[:200],
                        action_taken=action.tool_name,
                        outcome=verification.actual_outcome[:200],
                        lesson=f"Action {action.tool_name} succeeded",
                        success=True,
                        iteration=self._current_iteration
                    ))
                
                # Check if task is complete
                if self._check_completion(task, result, context):
                    return True
            else:
                # Learn from failure
                if self.config.enable_learning:
                    await self._learn(LearningEntry(
                        timestamp=datetime.now(),
                        situation=observation.state_summary[:200],
                        action_taken=action.tool_name,
                        outcome=verification.actual_outcome[:200],
                        lesson=f"Action {action.tool_name} failed: {verification.differences}",
                        success=False,
                        iteration=self._current_iteration
                    ))
                
                # Self-correct if enabled
                if self.config.enable_self_correction:
                    error_msg = verification.differences[0] if verification.differences else "Unknown error"
                    correction = await self._self_correct(error_msg, context)
                    if correction:
                        self._thoughts.append(correction)
        
        # Learn
        self._update_state(LoopPhase.LEARNING)
        self._report_progress("learning", progress, "Updating knowledge...")
        
        # Update metrics
        iteration_time = (time.time() - iteration_start) * 1000
        self._metrics.total_iterations = self._current_iteration
        self._metrics.total_execution_time_ms += int(iteration_time)
        self._metrics.average_iteration_time_ms = \
            self._metrics.total_execution_time_ms / self._current_iteration
        
        # Pause if configured
        if self.config.pause_between_iterations:
            await asyncio.sleep(self.config.pause_duration_ms / 1000)
        
        return False
    
    def _check_completion(self, task: str, result: Any, context: Dict[str, Any]) -> bool:
        """Check if the task is complete.
        
        Args:
            task: Current task
            result: Last action result
            context: Execution context
            
        Returns:
            True if task is complete
        """
        # Default implementation - can be overridden
        if isinstance(result, ToolResult):
            return result.success
        return False
    
    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> LoopResult:
        """Run the autonomous loop.
        
        Args:
            task: Task to accomplish
            context: Optional execution context
            
        Returns:
            LoopResult with execution details
        """
        context = context or {}
        self._state.task = task
        self._start_time = time.time()
        
        logger.info(f"[AutonomousLoop:{self.id}] Starting task: {task}")
        self._report_progress("starting", 0.0, f"Starting: {task[:50]}...")
        
        try:
            while self._should_continue():
                is_complete = await self._execute_iteration(task, context)
                
                if is_complete:
                    self._update_state(LoopPhase.COMPLETED)
                    logger.info(f"[AutonomousLoop:{self.id}] Task completed successfully")
                    break
            
            # Determine final state
            if self._stop_requested:
                final_state = LoopPhase.INTERRUPTED
                success = False
                error = "Loop interrupted by user"
            elif self._current_iteration >= self.config.max_iterations:
                final_state = LoopPhase.COMPLETED if self._state.phase == LoopPhase.COMPLETED else LoopPhase.FAILED
                success = final_state == LoopPhase.COMPLETED
                error = None if success else "Max iterations reached without completion"
            else:
                final_state = self._state.phase
                success = final_state == LoopPhase.COMPLETED
                error = None
            
            result = LoopResult(
                success=success,
                iterations=self._current_iteration,
                final_state=final_state,
                observations=self._observations.copy(),
                thoughts=self._thoughts.copy(),
                actions_taken=self._actions_taken.copy(),
                verifications=self._verifications.copy(),
                learnings=self._learnings.copy(),
                metrics=self._metrics,
                error=error
            )
            
        except Exception as e:
            logger.exception(f"[AutonomousLoop:{self.id}] Loop failed: {e}")
            self._update_state(LoopPhase.FAILED)
            
            result = LoopResult(
                success=False,
                iterations=self._current_iteration,
                final_state=LoopPhase.FAILED,
                observations=self._observations.copy(),
                thoughts=self._thoughts.copy(),
                actions_taken=self._actions_taken.copy(),
                verifications=self._verifications.copy(),
                learnings=self._learnings.copy(),
                metrics=self._metrics,
                error=str(e)
            )
        
        finally:
            self._end_time = time.time()
            if self._start_time:
                result.execution_time_ms = int((self._end_time - self._start_time) * 1000)
            
            self._execution_history.append(result)
            
            if self._result_callback:
                try:
                    self._result_callback(result)
                except Exception as e:
                    logger.warning(f"[AutonomousLoop:{self.id}] Result callback failed: {e}")
            
            self._report_progress("finished", 1.0, result.get_summary())
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loop statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "id": self.id,
            "state": self._state.to_dict(),
            "current_iteration": self._current_iteration,
            "max_iterations": self.config.max_iterations,
            "observations_count": len(self._observations),
            "thoughts_count": len(self._thoughts),
            "actions_count": len(self._actions_taken),
            "verifications_count": len(self._verifications),
            "learnings_count": len(self._learnings),
            "errors_count": len(self._errors),
            "execution_count": len(self._execution_history),
            "metrics": self._metrics.to_dict(),
            "config": self.config.to_dict()
        }
    
    def get_execution_history(self) -> List[LoopResult]:
        """Get execution history.
        
        Returns:
            List of past loop results
        """
        return self._execution_history.copy()


class DefaultAutonomousLoop(AutonomousLoop):
    """Default implementation of AutonomousLoop.
    
    This implementation provides basic functionality that can be used
    directly or extended for specific use cases.
    """
    
    def __init__(
        self,
        tool_service: Any,
        llm_client: Optional[Any] = None,
        config: Optional[LoopConfig] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
        result_callback: Optional[Callable[[LoopResult], None]] = None
    ):
        """Initialize default autonomous loop.
        
        Args:
            tool_service: Service for executing tools
            llm_client: Optional LLM client for thinking
            config: Loop configuration
            progress_callback: Progress callback
            result_callback: Result callback
        """
        super().__init__(config, progress_callback, result_callback)
        self.tool_service = tool_service
        self.llm_client = llm_client
    
    async def _observe(self, task: str, context: Dict[str, Any]) -> Observation:
        """Observe current state.
        
        Args:
            task: Current task
            context: Execution context
            
        Returns:
            Observation
        """
        logger.debug(f"[AutonomousLoop:{self.id}] Observing - iteration {self._current_iteration}")
        
        # Build state summary
        state_summary = f"Iteration {self._current_iteration}: Working on '{task[:50]}...'"
        
        # Get recent tool results
        tool_results = []
        for action in self._actions_taken[-3:]:
            tool_results.append({
                "tool": action.get("tool_name"),
                "success": action.get("success"),
                "preview": str(action.get("result", ""))[:200]
            })
        
        # Get environment info
        environment_info = {
            "iteration": self._current_iteration,
            "max_iterations": self.config.max_iterations,
            "previous_actions": len(self._actions_taken),
            "previous_errors": len(self._errors)
        }
        
        return Observation(
            timestamp=datetime.now(),
            state_summary=state_summary,
            relevant_data=context.copy(),
            tool_results=tool_results,
            environment_info=environment_info
        )
    
    async def _think(
        self,
        task: str,
        observation: Observation,
        context: Dict[str, Any]
    ) -> Thought:
        """Think about next action.
        
        Args:
            task: Current task
            observation: Current observation
            context: Execution context
            
        Returns:
            Thought with plan
        """
        logger.debug(f"[AutonomousLoop:{self.id}] Thinking - iteration {self._current_iteration}")
        
        # If LLM client available, use it for thinking
        if self.llm_client:
            return await self._think_with_llm(task, observation, context)
        
        # Otherwise use rule-based thinking
        return await self._think_rule_based(task, observation, context)
    
    async def _think_with_llm(
        self,
        task: str,
        observation: Observation,
        context: Dict[str, Any]
    ) -> Thought:
        """Use LLM for thinking.
        
        Args:
            task: Current task
            observation: Current observation
            context: Execution context
            
        Returns:
            Thought from LLM
        """
        try:
            # Build prompt
            prompt = self._build_thinking_prompt(task, observation, context)
            
            # Get LLM response
            response = await self.llm_client.generate(prompt, temperature=0.3)
            
            # Parse response
            return self._parse_llm_thought(response)
            
        except Exception as e:
            logger.error(f"[AutonomousLoop:{self.id}] LLM thinking failed: {e}")
            # Fallback to rule-based
            return await self._think_rule_based(task, observation, context)
    
    def _build_thinking_prompt(
        self,
        task: str,
        observation: Observation,
        context: Dict[str, Any]
    ) -> str:
        """Build thinking prompt for LLM.
        
        Args:
            task: Current task
            observation: Current observation
            context: Execution context
            
        Returns:
            Prompt string
        """
        # Get available tools
        available_tools = []
        if hasattr(self.tool_service, 'list_available_tools'):
            available_tools = self.tool_service.list_available_tools()
        
        # Build action history
        action_history = "\n".join([
            f"- {a.get('tool_name')}: {'Success' if a.get('success') else 'Failed'}"
            for a in self._actions_taken[-5:]
        ]) or "No previous actions"
        
        prompt = f"""You are an autonomous coding agent. Analyze the current situation and decide the next action.

## Task
{task}

## Progress
- Iteration: {self._current_iteration}/{self.config.max_iterations}
- State: {observation.state_summary}
- Previous Actions:
{action_history}

## Available Tools
{', '.join(available_tools)}

## Instructions
1. Analyze the task and progress so far
2. Consider what went wrong in previous attempts (if any)
3. Choose the most appropriate next action
4. Provide your reasoning and confidence level (0.0-1.0)

## Response Format
Return JSON with:
{{
    "reasoning": "Your detailed analysis and reasoning",
    "decision": "Brief description of what you decided to do",
    "confidence": 0.8,
    "plan": [
        {{
            "tool_name": "name_of_tool",
            "parameters": {{"param": "value"}},
            "expected_outcome": "What you expect to achieve"
        }}
    ],
    "risk_assessment": "Potential risks or issues",
    "alternative_approaches": ["Alternative if this fails"]
}}

## Your Analysis:"""
        
        return prompt
    
    def _parse_llm_thought(self, content: str) -> Thought:
        """Parse LLM response into Thought.
        
        Args:
            content: LLM response content
            
        Returns:
            Parsed Thought
        """
        try:
            # Extract JSON
            start = content.find('{')
            end = content.rfind('}')
            
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                data = json.loads(json_str)
                
                return Thought(
                    timestamp=datetime.now(),
                    reasoning=data.get("reasoning", "No reasoning provided"),
                    decision=data.get("decision", "No decision made"),
                    confidence=float(data.get("confidence", 0.5)),
                    plan=data.get("plan", []),
                    risk_assessment=data.get("risk_assessment", ""),
                    alternative_approaches=data.get("alternative_approaches", [])
                )
        except json.JSONDecodeError as e:
            logger.warning(f"[AutonomousLoop:{self.id}] Failed to parse LLM response: {e}")
        
        # Return default if parsing fails
        return Thought(
            timestamp=datetime.now(),
            reasoning="Failed to parse LLM response",
            decision="Use default approach",
            confidence=0.3,
            plan=[],
            risk_assessment="Parsing error",
            alternative_approaches=["Retry with different prompt"]
        )
    
    async def _think_rule_based(
        self,
        task: str,
        observation: Observation,
        context: Dict[str, Any]
    ) -> Thought:
        """Rule-based thinking.
        
        Args:
            task: Current task
            observation: Current observation
            context: Execution context
            
        Returns:
            Thought from rules
        """
        task_lower = task.lower()
        
        # Check for specific task patterns
        if any(word in task_lower for word in ["test", "testing", "coverage"]):
            plan = [{
                "tool_name": "glob",
                "parameters": {"pattern": "**/*.java"},
                "expected_outcome": "Find Java source files"
            }]
            reasoning = "Task involves testing, using test generation workflow"
            confidence = 0.7
        elif any(word in task_lower for word in ["fix", "bug", "error"]):
            plan = [{
                "tool_name": "git_status",
                "parameters": {},
                "expected_outcome": "Check repository status"
            }]
            reasoning = "Task involves debugging, using diagnostic workflow"
            confidence = 0.6
        elif any(word in task_lower for word in ["refactor", "clean", "improve"]):
            plan = [{
                "tool_name": "glob",
                "parameters": {"pattern": "**/*.py"},
                "expected_outcome": "Find Python files to refactor"
            }]
            reasoning = "Task involves refactoring, using code improvement workflow"
            confidence = 0.7
        else:
            plan = [{
                "tool_name": "git_status",
                "parameters": {},
                "expected_outcome": "Understand current repository state"
            }]
            reasoning = "Using generic exploration workflow"
            confidence = 0.5
        
        return Thought(
            timestamp=datetime.now(),
            reasoning=reasoning,
            decision=f"Execute {len(plan)} actions based on rules",
            confidence=confidence,
            plan=plan
        )
    
    async def _act(self, action: Action, context: Dict[str, Any]) -> Any:
        """Execute an action.
        
        Args:
            action: Action to execute
            context: Execution context
            
        Returns:
            Action result
        """
        logger.debug(f"[AutonomousLoop:{self.id}] Acting - executing {action.tool_name}")
        
        try:
            if hasattr(self.tool_service, 'execute_tool'):
                result = await self.tool_service.execute_tool(
                    action.tool_name,
                    action.parameters
                )
                return result
            else:
                # Fallback if tool_service doesn't have execute_tool
                return ToolResult(
                    success=False,
                    error=f"Tool service does not support execute_tool: {action.tool_name}"
                )
        except Exception as e:
            logger.error(f"[AutonomousLoop:{self.id}] Action failed: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _verify(
        self,
        result: Any,
        expected: str,
        context: Dict[str, Any]
    ) -> Verification:
        """Verify action result.
        
        Args:
            result: Action result
            expected: Expected outcome
            context: Execution context
            
        Returns:
            Verification result
        """
        if isinstance(result, ToolResult):
            success = result.success
            actual = result.output if result.success else result.error
        else:
            success = True
            actual = str(result)
        
        differences = []
        if not success:
            differences.append(f"Expected success but got failure: {actual}")
        
        return Verification(
            success=success,
            actual_outcome=str(actual)[:500] if actual else "",
            expected_outcome=expected,
            differences=differences
        )


def create_autonomous_loop(
    tool_service: Any,
    llm_client: Optional[Any] = None,
    max_iterations: int = 10,
    confidence_threshold: float = 0.8,
    enable_self_correction: bool = True,
    enable_learning: bool = True,
    **kwargs
) -> DefaultAutonomousLoop:
    """Create a DefaultAutonomousLoop instance.
    
    Args:
        tool_service: Service for executing tools
        llm_client: Optional LLM client for thinking
        max_iterations: Maximum iterations
        confidence_threshold: Confidence threshold
        enable_self_correction: Whether to enable self-correction
        enable_learning: Whether to enable learning
        **kwargs: Additional config parameters
        
    Returns:
        DefaultAutonomousLoop instance
    """
    config = LoopConfig(
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold,
        enable_self_correction=enable_self_correction,
        enable_learning=enable_learning,
        **kwargs
    )
    
    return DefaultAutonomousLoop(
        tool_service=tool_service,
        llm_client=llm_client,
        config=config
    )


# Type alias for backward compatibility
LoopStateEnum = LoopPhase

__all__ = [
    # Enums
    "LoopPhase",
    "RiskLevel",
    "ActionPriority",
    "LoopStateEnum",
    
    # Data classes
    "LoopState",
    "Observation",
    "Thought",
    "Action",
    "Verification",
    "LearningEntry",
    "RiskAssessment",
    "LoopMetrics",
    "LoopResult",
    "LoopConfig",
    
    # Classes
    "AutonomousLoop",
    "DefaultAutonomousLoop",
    
    # Functions
    "create_autonomous_loop",
]
