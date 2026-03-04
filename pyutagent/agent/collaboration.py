"""User Collaboration - Flexible human-agent interaction modes.

This module provides multiple collaboration modes:
- Full Autonomous: Agent works independently
- Suggest and Confirm: Agent suggests, user confirms
- Step by Step: User approves each step
- Manual Review: User reviews all changes
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class CollaborationMode(Enum):
    """Collaboration mode between user and agent."""
    FULL_AUTONOMOUS = auto()           # Agent works independently
    SUGGEST_AND_CONFIRM = auto()       # Agent suggests, user confirms
    STEP_BY_STEP = auto()              # User approves each step
    MANUAL_REVIEW = auto()             # User reviews all changes


class UserResponse(Enum):
    """User response to agent action."""
    APPROVE = auto()                   # Approve and continue
    REJECT = auto()                    # Reject and stop
    MODIFY = auto()                    # Modify and continue
    SKIP = auto()                      # Skip this action
    ASK_QUESTION = auto()              # Ask for clarification


@dataclass
class ProposedAction:
    """An action proposed by the agent."""
    action_id: str
    description: str
    action_type: str
    details: Dict[str, Any]
    impact: str  # Description of impact
    can_undo: bool = True
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ActionResult:
    """Result of an action execution."""
    action_id: str
    success: bool
    message: str
    output: Any = None
    requires_review: bool = False
    changes_made: List[str] = field(default_factory=list)


@dataclass
class UserDecision:
    """User's decision on a proposed action."""
    response: UserResponse
    feedback: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContentPreview:
    """Preview of content for user review."""
    title: str
    content_type: str  # code, diff, text, etc.
    original_content: Optional[str]
    proposed_content: str
    highlights: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class UserInteractionHandler:
    """Handler for user interactions in different collaboration modes.

    Features:
    - Multiple collaboration modes
    - Action proposal and confirmation
    - Content preview and review
    - Question asking
    - Decision tracking
    """

    def __init__(
        self,
        mode: CollaborationMode = CollaborationMode.SUGGEST_AND_CONFIRM,
        auto_approve_threshold: float = 0.9,
        callback: Optional[Callable] = None
    ):
        """Initialize user interaction handler.

        Args:
            mode: Collaboration mode
            auto_approve_threshold: Confidence threshold for auto-approval
            callback: Callback function for user interactions
        """
        self.mode = mode
        self.auto_approve_threshold = auto_approve_threshold
        self.callback = callback
        self._decision_history: List[Dict[str, Any]] = []

        logger.info(f"[UserInteractionHandler] Initialized with mode: {mode.name}")

    async def propose_action(
        self,
        action: ProposedAction,
        confidence: float = 0.5
    ) -> UserDecision:
        """Propose an action to the user.

        Args:
            action: Action to propose
            confidence: Agent's confidence in the action

        Returns:
            User's decision
        """
        # In full autonomous mode, auto-approve if confidence is high
        if self.mode == CollaborationMode.FULL_AUTONOMOUS:
            if confidence >= self.auto_approve_threshold:
                logger.info(f"[UserInteractionHandler] Auto-approved action: {action.action_id}")
                return UserDecision(response=UserResponse.APPROVE)

        # In suggest and confirm mode, ask for confirmation
        if self.mode == CollaborationMode.SUGGEST_AND_CONFIRM:
            return await self._ask_for_confirmation(action, confidence)

        # In step by step mode, always ask
        if self.mode == CollaborationMode.STEP_BY_STEP:
            return await self._ask_for_confirmation(action, confidence, detailed=True)

        # In manual review mode, queue for review
        if self.mode == CollaborationMode.MANUAL_REVIEW:
            return UserDecision(
                response=UserResponse.APPROVE,
                feedback="Queued for manual review"
            )

        # Default: ask for confirmation
        return await self._ask_for_confirmation(action, confidence)

    async def _ask_for_confirmation(
        self,
        action: ProposedAction,
        confidence: float,
        detailed: bool = False
    ) -> UserDecision:
        """Ask user for confirmation.

        Args:
            action: Action to confirm
            confidence: Confidence level
            detailed: Whether to show detailed info

        Returns:
            User decision
        """
        # Build confirmation message
        message = self._build_confirmation_message(action, confidence, detailed)

        # If callback is provided, use it
        if self.callback:
            try:
                response = await self.callback("confirm_action", {
                    "action": action,
                    "message": message,
                    "confidence": confidence
                })
                return self._parse_user_response(response)
            except Exception as e:
                logger.error(f"[UserInteractionHandler] Callback failed: {e}")

        # Default: auto-approve (for testing/non-interactive mode)
        logger.info(f"[UserInteractionHandler] No callback, auto-approving: {action.action_id}")
        return UserDecision(response=UserResponse.APPROVE)

    def _build_confirmation_message(
        self,
        action: ProposedAction,
        confidence: float,
        detailed: bool
    ) -> str:
        """Build confirmation message.

        Args:
            action: Action to confirm
            confidence: Confidence level
            detailed: Whether to include details

        Returns:
            Confirmation message
        """
        lines = [
            f"Proposed Action: {action.description}",
            f"Type: {action.action_type}",
            f"Confidence: {confidence:.0%}",
            f"Impact: {action.impact}",
        ]

        if detailed:
            lines.append(f"Details: {action.details}")
            if action.alternatives:
                lines.append(f"Alternatives: {len(action.alternatives)} available")

        lines.append("Approve? (yes/no/modify/skip)")

        return "\n".join(lines)

    def _parse_user_response(self, response: Any) -> UserDecision:
        """Parse user response.

        Args:
            response: Raw user response

        Returns:
            Parsed decision
        """
        if isinstance(response, UserDecision):
            return response

        if isinstance(response, str):
            response_lower = response.lower().strip()

            if response_lower in ("yes", "y", "approve", "ok"):
                return UserDecision(response=UserResponse.APPROVE)
            elif response_lower in ("no", "n", "reject", "cancel"):
                return UserDecision(response=UserResponse.REJECT)
            elif response_lower in ("modify", "m", "change"):
                return UserDecision(response=UserResponse.MODIFY)
            elif response_lower in ("skip", "s", "next"):
                return UserDecision(response=UserResponse.SKIP)
            elif "?" in response_lower:
                return UserDecision(
                    response=UserResponse.ASK_QUESTION,
                    feedback=response
                )

        # Default to approve
        return UserDecision(response=UserResponse.APPROVE)

    async def show_preview(
        self,
        preview: ContentPreview,
        wait_for_feedback: bool = True
    ) -> Optional[UserDecision]:
        """Show content preview to user.

        Args:
            preview: Content preview
            wait_for_feedback: Whether to wait for user feedback

        Returns:
            User decision if waiting for feedback
        """
        # Build preview message
        message = self._build_preview_message(preview)

        # If callback is provided, use it
        if self.callback:
            try:
                await self.callback("show_preview", {
                    "preview": preview,
                    "message": message
                })
            except Exception as e:
                logger.error(f"[UserInteractionHandler] Preview callback failed: {e}")

        if wait_for_feedback and self.mode != CollaborationMode.FULL_AUTONOMOUS:
            # Ask for approval
            if self.callback:
                response = await self.callback("ask_approval", {
                    "message": "Approve these changes?"
                })
                return self._parse_user_response(response)

        return None

    def _build_preview_message(self, preview: ContentPreview) -> str:
        """Build preview message.

        Args:
            preview: Content preview

        Returns:
            Preview message
        """
        lines = [
            f"Preview: {preview.title}",
            f"Type: {preview.content_type}",
        ]

        if preview.warnings:
            lines.append("Warnings:")
            for warning in preview.warnings:
                lines.append(f"  - {warning}")

        if preview.highlights:
            lines.append("Highlights:")
            for highlight in preview.highlights:
                lines.append(f"  - {highlight}")

        lines.append("\nProposed Content:")
        lines.append(preview.proposed_content[:500])  # Truncate for display

        if len(preview.proposed_content) > 500:
            lines.append("... (truncated)")

        return "\n".join(lines)

    async def ask_question(
        self,
        question: str,
        options: Optional[List[str]] = None,
        allow_free_text: bool = True
    ) -> str:
        """Ask user a question.

        Args:
            question: Question to ask
            options: Optional predefined options
            allow_free_text: Whether to allow free text response

        Returns:
            User's answer
        """
        # Build question message
        message = question
        if options:
            message += "\nOptions:\n"
            for i, option in enumerate(options, 1):
                message += f"  {i}. {option}\n"

        if allow_free_text:
            message += "\n(Or type your own answer)"

        # If callback is provided, use it
        if self.callback:
            try:
                response = await self.callback("ask_question", {
                    "question": question,
                    "options": options,
                    "message": message
                })
                return str(response)
            except Exception as e:
                logger.error(f"[UserInteractionHandler] Question callback failed: {e}")

        # Default: return first option or empty string
        return options[0] if options else ""

    async def report_progress(
        self,
        message: str,
        progress: Optional[float] = None
    ):
        """Report progress to user.

        Args:
            message: Progress message
            progress: Optional progress percentage (0-1)
        """
        if self.callback:
            try:
                await self.callback("report_progress", {
                    "message": message,
                    "progress": progress
                })
            except Exception as e:
                logger.error(f"[UserInteractionHandler] Progress callback failed: {e}")

        logger.info(f"[UserInteractionHandler] Progress: {message}")

    async def report_completion(
        self,
        task: str,
        results: Dict[str, Any],
        success: bool = True
    ):
        """Report task completion to user.

        Args:
            task: Task description
            results: Task results
            success: Whether task succeeded
        """
        status = "completed successfully" if success else "failed"
        message = f"Task {status}: {task}"

        if self.callback:
            try:
                await self.callback("report_completion", {
                    "task": task,
                    "results": results,
                    "success": success,
                    "message": message
                })
            except Exception as e:
                logger.error(f"[UserInteractionHandler] Completion callback failed: {e}")

        logger.info(f"[UserInteractionHandler] {message}")

    def record_decision(
        self,
        action_id: str,
        decision: UserDecision,
        action_result: Optional[ActionResult] = None
    ):
        """Record a decision for learning.

        Args:
            action_id: Action ID
            decision: User decision
            action_result: Optional action result
        """
        self._decision_history.append({
            "action_id": action_id,
            "decision": decision.response.name,
            "feedback": decision.feedback,
            "timestamp": decision.timestamp.isoformat(),
            "result": action_result.message if action_result else None
        })

    def get_decision_history(
        self,
        action_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get decision history.

        Args:
            action_type: Optional filter by action type

        Returns:
            List of decisions
        """
        if action_type:
            return [d for d in self._decision_history if d.get("action_type") == action_type]
        return self._decision_history.copy()

    def set_mode(self, mode: CollaborationMode):
        """Change collaboration mode.

        Args:
            mode: New collaboration mode
        """
        self.mode = mode
        logger.info(f"[UserInteractionHandler] Mode changed to: {mode.name}")


class CollaborationManager:
    """Manager for collaboration sessions.

    Manages the interaction flow between agent and user.
    """

    def __init__(
        self,
        default_mode: CollaborationMode = CollaborationMode.SUGGEST_AND_CONFIRM
    ):
        """Initialize collaboration manager.

        Args:
            default_mode: Default collaboration mode
        """
        self.default_mode = default_mode
        self._handlers: Dict[str, UserInteractionHandler] = {}
        self._session_history: List[Dict[str, Any]] = []

    def create_session(
        self,
        session_id: str,
        mode: Optional[CollaborationMode] = None,
        callback: Optional[Callable] = None
    ) -> UserInteractionHandler:
        """Create a new collaboration session.

        Args:
            session_id: Session identifier
            mode: Collaboration mode
            callback: Interaction callback

        Returns:
            User interaction handler
        """
        handler = UserInteractionHandler(
            mode=mode or self.default_mode,
            callback=callback
        )
        self._handlers[session_id] = handler

        logger.info(f"[CollaborationManager] Created session: {session_id}")
        return handler

    def get_session(self, session_id: str) -> Optional[UserInteractionHandler]:
        """Get an existing session.

        Args:
            session_id: Session identifier

        Returns:
            User interaction handler or None
        """
        return self._handlers.get(session_id)

    def end_session(self, session_id: str):
        """End a collaboration session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._handlers:
            handler = self._handlers[session_id]
            self._session_history.append({
                "session_id": session_id,
                "decisions": handler.get_decision_history(),
                "ended_at": datetime.now().isoformat()
            })
            del self._handlers[session_id]
            logger.info(f"[CollaborationManager] Ended session: {session_id}")

    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get all session history.

        Returns:
            List of session records
        """
        return self._session_history.copy()


def create_collaboration_handler(
    mode: str = "suggest_and_confirm",
    callback: Optional[Callable] = None
) -> UserInteractionHandler:
    """Create a collaboration handler.

    Args:
        mode: Collaboration mode name
        callback: Interaction callback

    Returns:
        User interaction handler
    """
    mode_map = {
        "full_autonomous": CollaborationMode.FULL_AUTONOMOUS,
        "suggest_and_confirm": CollaborationMode.SUGGEST_AND_CONFIRM,
        "step_by_step": CollaborationMode.STEP_BY_STEP,
        "manual_review": CollaborationMode.MANUAL_REVIEW
    }

    collaboration_mode = mode_map.get(mode.lower(), CollaborationMode.SUGGEST_AND_CONFIRM)

    return UserInteractionHandler(
        mode=collaboration_mode,
        callback=callback
    )
