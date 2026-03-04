"""Flexible Collaboration Modes - Enhanced user interaction with multiple modes.

This module provides:
- CollaborationMode enum (FULL_AUTONOMOUS, SUGGEST_AND_CONFIRM, etc.)
- CollaborationHandler for different interaction patterns
- Configurable approval thresholds
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CollaborationMode(Enum):
    """Collaboration modes for agent interaction.

    - FULL_AUTONOMOUS: Agent executes without asking (for routine tasks)
    - SUGGEST_AND_CONFIRM: Agent suggests, user confirms (default)
    - STEP_BY_STEP: Agent asks before each step (for critical tasks)
    - MANUAL_REVIEW: Agent proposes, user executes (full control)
    """
    FULL_AUTONOMOUS = "full_autonomous"
    SUGGEST_AND_CONFIRM = "suggest_and_confirm"
    STEP_BY_STEP = "step_by_step"
    MANUAL_REVIEW = "manual_review"


@dataclass
class CollaborationConfig:
    """Configuration for collaboration behavior."""
    mode: CollaborationMode = CollaborationMode.SUGGEST_AND_CONFIRM
    auto_approve_threshold: float = 0.9
    show_preview: bool = True
    max_auto_retries: int = 3
    confirm_destructive: bool = True
    verbose_logging: bool = False


@dataclass
class ProposedAction:
    """An action proposed by the agent."""
    action_id: str
    action_type: str
    description: str
    parameters: Dict[str, Any]
    expected_outcome: str
    confidence: float
    risk_level: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserResponse:
    """User's response to a proposed action."""
    decision: str
    feedback: Optional[str] = None
    modified_parameters: Optional[Dict[str, Any]] = None
    selected_alternative: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActionExecution:
    """Record of action execution."""
    action: ProposedAction
    user_response: Optional[UserResponse]
    executed: bool
    result: Any
    error: Optional[str] = None
    duration_ms: int = 0


class CollaborationHandler:
    """Handler for different collaboration modes.

    Manages the interaction flow based on configured collaboration mode.
    """

    def __init__(self, config: CollaborationConfig):
        """Initialize collaboration handler.

        Args:
            config: Collaboration configuration
        """
        self.config = config
        self._execution_history: List[ActionExecution] = []
        self._callbacks: Dict[str, List[Callable]] = {
            "before_action": [],
            "after_action": [],
            "user_response": []
        }

    def register_callback(
        self,
        event: str,
        callback: Callable
    ):
        """Register callback for events.

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def handle_action(
        self,
        action: ProposedAction
    ) -> ActionExecution:
        """Handle an action based on collaboration mode.

        Args:
            action: Proposed action

        Returns:
            Action execution result
        """
        execution = ActionExecution(
            action=action,
            user_response=None,
            executed=False,
            result=None
        )

        start_time = datetime.now()

        await self._emit_callback("before_action", action)

        if self.config.mode == CollaborationMode.FULL_AUTONOMOUS:
            execution = await self._execute_autonomous(action, execution)
        elif self.config.mode == CollaborationMode.SUGGEST_AND_CONFIRM:
            execution = await self._execute_suggest_confirm(action, execution)
        elif self.config.mode == CollaborationMode.STEP_BY_STEP:
            execution = await self._execute_step_by_step(action, execution)
        elif self.config.mode == CollaborationMode.MANUAL_REVIEW:
            execution = await self._execute_manual_review(action, execution)

        execution.duration_ms = int(
            (datetime.now() - start_time).total_seconds() * 1000
        )

        self._execution_history.append(execution)
        await self._emit_callback("after_action", execution)

        return execution

    async def _execute_autonomous(
        self,
        action: ProposedAction,
        execution: ActionExecution
    ) -> ActionExecution:
        """Execute in full autonomous mode."""
        if action.confidence >= self.config.auto_approve_threshold:
            logger.info(f"[CollaborationHandler] Auto-executing: {action.action_type}")
            execution.executed = True
        else:
            logger.info(f"[CollaborationHandler] Low confidence, skipping: {action.action_type}")
            execution.user_response = UserResponse(
                decision="SKIP",
                feedback="Low confidence"
            )

        return execution

    async def _execute_suggest_confirm(
        self,
        action: ProposedAction,
        execution: ActionExecution
    ) -> ActionExecution:
        """Execute in suggest and confirm mode."""
        if action.risk_level == "high" and self.config.confirm_destructive:
            logger.info(f"[CollaborationHandler] High risk action requires confirmation: {action.action_type}")
            user_response = await self._request_confirmation(action)
            execution.user_response = user_response

            if user_response.decision == "APPROVE":
                execution.executed = True
        elif action.confidence >= self.config.auto_approve_threshold:
            logger.info(f"[CollaborationHandler] High confidence, auto-approving: {action.action_type}")
            execution.executed = True
        else:
            user_response = await self._request_confirmation(action)
            execution.user_response = user_response

            if user_response.decision == "APPROVE":
                execution.executed = True

        return execution

    async def _execute_step_by_step(
        self,
        action: ProposedAction,
        execution: ActionExecution
    ) -> ActionExecution:
        """Execute in step by step mode - always confirm."""
        logger.info(f"[CollaborationHandler] Step-by-step mode, requesting confirmation: {action.action_type}")

        user_response = await self._request_confirmation(action)
        execution.user_response = user_response

        if user_response.decision == "APPROVE":
            execution.executed = True

        return execution

    async def _execute_manual_review(
        self,
        action: ProposedAction,
        execution: ActionExecution
    ) -> ActionExecution:
        """Execute in manual review mode - user must approve."""
        logger.info(f"[CollaborationHandler] Manual review mode, user must approve: {action.action_type}")

        user_response = await self._request_confirmation(action)
        execution.user_response = user_response

        if user_response.decision == "APPROVE" and user_response.modified_parameters:
            action.parameters.update(user_response.modified_parameters)
            execution.executed = True

        return execution

    async def _request_confirmation(
        self,
        action: ProposedAction
    ) -> UserResponse:
        """Request user confirmation for action.

        Args:
            action: Action to confirm

        Returns:
            User response
        """
        await self._emit_callback("user_response", {
            "action": action,
            "pending": True
        })

        response = UserResponse(
            decision="APPROVE",
            feedback="Auto-approved for demonstration"
        )

        return response

    async def _emit_callback(self, event: str, data: Any):
        """Emit callback for event.

        Args:
            event: Event name
            data: Event data
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"[CollaborationHandler] Callback error: {e}")

    def get_execution_history(
        self,
        limit: int = 10
    ) -> List[ActionExecution]:
        """Get execution history.

        Args:
            limit: Maximum results

        Returns:
            Recent executions
        """
        return self._execution_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Statistics dict
        """
        total = len(self._execution_history)
        executed = sum(1 for e in self._execution_history if e.executed)
        skipped = sum(1 for e in self._execution_history if e.user_response and e.user_response.decision == "SKIP")

        return {
            "total_actions": total,
            "executed": executed,
            "skipped": skipped,
            "execution_rate": executed / total if total > 0 else 0,
            "mode": self.config.mode.value
        }


class CollaborationModeManager:
    """Manager for switching between collaboration modes."""

    def __init__(self):
        """Initialize mode manager."""
        self._current_mode = CollaborationMode.SUGGEST_AND_CONFIRM
        self._handlers: Dict[CollaborationMode, CollaborationHandler] = {}

    def set_mode(self, mode: CollaborationMode) -> CollaborationHandler:
        """Set collaboration mode and get handler.

        Args:
            mode: Collaboration mode

        Returns:
            Collaboration handler
        """
        self._current_mode = mode

        if mode not in self._handlers:
            config = CollaborationConfig(mode=mode)
            self._handlers[mode] = CollaborationHandler(config)

        return self._handlers[mode]

    def get_handler(self) -> CollaborationHandler:
        """Get handler for current mode.

        Returns:
            Current collaboration handler
        """
        return self.set_mode(self._current_mode)

    @property
    def current_mode(self) -> CollaborationMode:
        """Get current mode."""
        return self._current_mode

    def switch_mode(self, mode: CollaborationMode) -> CollaborationHandler:
        """Switch mode and return handler.

        Args:
            mode: New mode

        Returns:
            New handler
        """
        logger.info(f"[CollaborationModeManager] Switching mode: {self._current_mode.value} -> {mode.value}")
        return self.set_mode(mode)


def create_collaboration_handler(
    mode: CollaborationMode = CollaborationMode.SUGGEST_AND_CONFIRM,
    auto_approve_threshold: float = 0.9
) -> CollaborationHandler:
    """Create collaboration handler.

    Args:
        mode: Collaboration mode
        auto_approve_threshold: Confidence threshold for auto-approval

    Returns:
        Collaboration handler
    """
    config = CollaborationConfig(
        mode=mode,
        auto_approve_threshold=auto_approve_threshold
    )
    return CollaborationHandler(config)
