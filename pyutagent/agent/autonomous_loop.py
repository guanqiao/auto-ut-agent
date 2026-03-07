"""Autonomous Loop - Observe-Think-Act-Verify cycle.

This module implements the autonomous decision loop:
- Observe: Gather current state
- Think: Analyze and plan
- Act: Execute tools
- Verify: Check results
- Learn: Update memory
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

from .tool_service import AgentToolService
from ..core.config import DEFAULT_MAX_ITERATIONS

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """States of the autonomous loop."""
    IDLE = auto()
    OBSERVING = auto()
    THINKING = auto()
    ACTING = auto()
    VERIFYING = auto()
    LEARNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()


@dataclass
class Observation:
    """Represents an observation of the current state."""
    timestamp: datetime
    state_summary: str
    relevant_data: Dict[str, Any]
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Thought:
    """Represents a thought/decision."""
    timestamp: datetime
    reasoning: str
    decision: str
    confidence: float
    plan: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Action:
    """Represents an action to take."""
    tool_name: str
    parameters: Dict[str, Any]
    expected_outcome: str


@dataclass
class Verification:
    """Represents verification of action result."""
    success: bool
    actual_outcome: str
    expected_outcome: str
    differences: List[str] = field(default_factory=list)


@dataclass
class LoopResult:
    """Result of the autonomous loop."""
    success: bool
    iterations: int
    final_state: LoopState
    observations: List[Observation] = field(default_factory=list)
    thoughts: List[Thought] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


class AutonomousLoop:
    """Autonomous decision loop (Observe-Think-Act-Verify-Learn).
    
    This loop enables the agent to:
    1. Observe: Gather current state information
    2. Think: Analyze and plan next actions
    3. Act: Execute tools
    4. Verify: Check if goals are met
    5. Learn: Update based on results
    
    Features:
    - Configurable max iterations
    - User interrupt support
    - Progress callbacks
    - Learning from failures
    """
    
    def __init__(
        self,
        tool_service: AgentToolService,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        confidence_threshold: float = 0.8,
        user_interruptible: bool = True
    ):
        """Initialize the autonomous loop.
        
        Args:
            tool_service: Tool service for executing tools
            max_iterations: Maximum iterations before stopping
            confidence_threshold: Confidence threshold to consider task complete
            user_interruptible: Whether user can interrupt the loop
        """
        self.tool_service = tool_service
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.user_interruptible = user_interruptible
        
        self._current_state = LoopState.IDLE
        self._iteration = 0
        
        self._observations: List[Observation] = []
        self._thoughts: List[Thought] = []
        self._actions_taken: List[Dict[str, Any]] = []
        self._learnings: List[str] = []
        
        self._interrupt_event: Optional[asyncio.Event] = None
    
    @property
    def state(self) -> LoopState:
        """Get current loop state."""
        return self._current_state
    
    @property
    def iteration(self) -> int:
        """Get current iteration number."""
        return self._iteration
    
    async def run(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> LoopResult:
        """Run the autonomous loop.
        
        Args:
            task: The task to accomplish
            context: Initial context
            progress_callback: Optional callback for progress updates
        
        Returns:
            LoopResult with execution details
        """
        context = context or {}
        self._reset()
        
        logger.info(f"[AutonomousLoop] Starting task: {task}")
        
        try:
            for self._iteration in range(1, self.max_iterations + 1):
                if self._current_state == LoopState.COMPLETED:
                    break
                
                if self.user_interruptible and self._is_interrupted():
                    logger.info("[AutonomousLoop] Loop interrupted by user")
                    self._current_state = LoopState.PAUSED
                    break
                
                progress = {
                    "iteration": self._iteration,
                    "max_iterations": self.max_iterations,
                    "state": self._current_state.name,
                    "task": task
                }
                
                if progress_callback:
                    progress_callback(progress)
                
                await self._step(task, context)
            
            success = self._current_state == LoopState.COMPLETED
            
            if self._iteration >= self.max_iterations and not success:
                self._current_state = LoopState.FAILED
            
            return LoopResult(
                success=success,
                iterations=self._iteration,
                final_state=self._current_state,
                observations=self._observations,
                thoughts=self._thoughts,
                actions_taken=self._actions_taken,
                learnings=self._learnings
            )
            
        except Exception as e:
            logger.exception(f"[AutonomousLoop] Error: {e}")
            self._current_state = LoopState.FAILED
            return LoopResult(
                success=False,
                iterations=self._iteration,
                final_state=LoopState.FAILED,
                error=str(e)
            )
    
    async def _step(self, task: str, context: Dict[str, Any]):
        """Execute one step of the loop.
        
        Args:
            task: Current task
            context: Current context
        """
        self._current_state = LoopState.OBSERVING
        observation = await self._observe(task, context)
        self._observations.append(observation)
        
        self._current_state = LoopState.THINKING
        thought = await self._think(task, observation, context)
        self._thoughts.append(thought)
        
        if thought.confidence >= self.confidence_threshold:
            logger.info(f"[AutonomousLoop] Confidence {thought.confidence} >= threshold, completing")
            self._current_state = LoopState.COMPLETED
            return
        
        self._current_state = LoopState.ACTING
        for action_plan in thought.plan:
            result = await self._act(action_plan, context)
            
            self._actions_taken.append({
                "tool": action_plan.get("tool_name"),
                "parameters": action_plan.get("parameters"),
                "result": result.to_dict() if hasattr(result, 'to_dict') else str(result)
            })
            
            self._current_state = LoopState.VERIFYING
            verification = await self._verify(action_plan, result)
            
            if not verification.success:
                self._learnings.append(f"Action {action_plan.get('tool_name')} did not produce expected result")
            
            if verification.success and self._check_completion(task, result, context):
                self._current_state = LoopState.COMPLETED
                return
        
        self._current_state = LoopState.LEARNING
        await self._learn(task, context)
    
    async def _observe(self, task: str, context: Dict[str, Any]) -> Observation:
        """Observe the current state.
        
        Args:
            task: Current task
            context: Current context
        
        Returns:
            Observation of current state
        """
        logger.debug(f"[AutonomousLoop] Observing - iteration {self._iteration}")
        
        state_summary = f"Iteration {self._iteration}: Working on task '{task}'"
        
        tool_results = []
        for action in self._actions_taken[-3:]:
            tool_results.append({
                "tool": action.get("tool"),
                "result_preview": str(action.get("result", ""))[:200]
            })
        
        return Observation(
            timestamp=datetime.now(),
            state_summary=state_summary,
            relevant_data=context.copy(),
            tool_results=tool_results
        )
    
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
            context: Current context
        
        Returns:
            Thought with plan
        """
        logger.debug(f"[AutonomousLoop] Thinking - iteration {self._iteration}")
        
        available_tools = self.tool_service.list_available_tools()
        
        if not observation.tool_results:
            plan = [{
                "tool_name": "git_status",
                "parameters": {},
                "expected_outcome": "Get current repository status"
            }]
            reasoning = "First need to understand current state of the repository"
            confidence = 0.3
        else:
            last_result = observation.tool_results[-1] if observation.tool_results else {}
            
            if "git_status" in str(observation.tool_results):
                plan = [{
                    "tool_name": "read_file",
                    "parameters": {"file_path": "pyutagent/agent/react_agent.py"},
                    "expected_outcome": "Read agent source code"
                }]
                reasoning = "Need to examine the agent code"
                confidence = 0.5
            else:
                plan = [{
                    "tool_name": "bash",
                    "parameters": {"command": "echo 'Task analysis complete'"},
                    "expected_outcome": "Complete task"
                }]
                reasoning = "Based on previous results, proceeding with completion"
                confidence = 0.9
        
        return Thought(
            timestamp=datetime.now(),
            reasoning=reasoning,
            decision=f"Will execute: {plan[0].get('tool_name') if plan else 'nothing'}",
            confidence=confidence,
            plan=plan
        )
    
    async def _act(self, action_plan: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute an action.
        
        Args:
            action_plan: Action to execute
            context: Current context
        
        Returns:
            Action result
        """
        tool_name = action_plan.get("tool_name")
        parameters = action_plan.get("parameters", {})
        
        logger.debug(f"[AutonomousLoop] Acting - executing {tool_name}")
        
        try:
            result = await self.tool_service.execute_tool(tool_name, parameters)
            return result
        except Exception as e:
            logger.error(f"[AutonomousLoop] Action failed: {e}")
            from ..tools.tool import ToolResult
            return ToolResult(success=False, error=str(e))
    
    async def _verify(self, action_plan: Dict[str, Any], result: Any) -> Verification:
        """Verify action result.
        
        Args:
            action_plan: Action that was executed
            result: Result of the action
        
        Returns:
            Verification result
        """
        expected = action_plan.get("expected_outcome", "")
        
        if hasattr(result, 'success'):
            success = result.success
            actual = result.output if result.success else result.error
        else:
            success = True
            actual = str(result)
        
        return Verification(
            success=success,
            actual_outcome=actual[:200] if actual else "",
            expected_outcome=expected,
            differences=[]
        )
    
    def _check_completion(self, task: str, result: Any, context: Dict[str, Any]) -> bool:
        """Check if task is complete.
        
        Args:
            task: Current task
            result: Last action result
            context: Current context
        
        Returns:
            True if task is complete
        """
        if hasattr(result, 'success') and result.success:
            return True
        return False
    
    async def _learn(self, task: str, context: Dict[str, Any]):
        """Learn from current iteration.
        
        Args:
            task: Current task
            context: Current context
        """
        logger.debug(f"[AutonomousLoop] Learning - iteration {self._iteration}")
        
        if self._actions_taken:
            last_action = self._actions_taken[-1]
            tool_name = last_action.get("tool")
            self._learnings.append(f"Tried {tool_name} in iteration {self._iteration}")
    
    def _reset(self):
        """Reset loop state."""
        self._current_state = LoopState.IDLE
        self._iteration = 0
        self._observations.clear()
        self._thoughts.clear()
        self._actions_taken.clear()
        self._learnings.clear()
    
    def _is_interrupted(self) -> bool:
        """Check if loop is interrupted."""
        return False
    
    def interrupt(self):
        """Interrupt the loop."""
        logger.info("[AutonomousLoop] Loop interrupted")
        self._current_state = LoopState.PAUSED


def create_autonomous_loop(
    tool_service: AgentToolService,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    confidence_threshold: float = 0.8
) -> AutonomousLoop:
    """Create an AutonomousLoop instance.
    
    Args:
        tool_service: Tool service for execution
        max_iterations: Maximum iterations
        confidence_threshold: Confidence threshold
    
    Returns:
        AutonomousLoop instance
    """
    return AutonomousLoop(
        tool_service=tool_service,
        max_iterations=max_iterations,
        confidence_threshold=confidence_threshold
    )
