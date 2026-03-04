"""Safety module for coding agent.

This module provides:
- SafetyPolicy: Security policy configuration
- SafetyValidator: Validate actions for safety
- UserInterventionHandler: Handle user interventions
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result types."""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    WARNING = "warning"
    REQUIRES_CONFIRMATION = "requires_confirmation"


class UserDecision(Enum):
    """User decision types."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SKIP = "skip"


class InterruptType(Enum):
    """User interrupt types."""
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    CONFIRM = "confirm"
    MODIFY = "modify"


@dataclass
class SafetyPolicy:
    """Security policy for agent actions."""
    max_iterations: int = 50
    max_file_edits: int = 100
    max_command_executions: int = 50
    max_concurrent_tasks: int = 5
    allow_destructive: bool = False
    allow_network: bool = False
    allow_shell: bool = True
    allowed_commands: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=list)
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    file_extensions: List[str] = field(default_factory=list)
    require_confirmation_for: List[str] = field(default_factory=list)
    timeout_seconds: int = 300


@dataclass
class ValidationContext:
    """Context for validation."""
    action_type: str
    parameters: Dict[str, Any]
    project_path: Optional[str] = None
    user_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResponse:
    """Response from validation."""
    result: ValidationResult
    message: str
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ProposedAction:
    """An action proposed by the agent."""
    action_id: str
    action_type: str
    description: str
    parameters: Dict[str, Any]
    risk_level: str = "low"
    affected_files: List[str] = field(default_factory=list)
    estimated_impact: str = ""


@dataclass
class UserResponse:
    """User's response to a proposed action."""
    decision: UserDecision
    feedback: str = ""
    modified_parameters: Optional[Dict[str, Any]] = None


@dataclass
class UserInterrupt:
    """User interrupt signal."""
    interrupt_type: InterruptType
    message: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class SafetyValidator:
    """Validator for agent actions."""

    DESTRUCTIVE_PATTERNS = [
        r"rm\s+-rf",
        r"del\s+/[pqrs]",
        r"format",
        r"drop\s+table",
        r"truncate",
    ]

    DANGEROUS_COMMANDS = [
        "rm -rf",
        "del /s /q",
        "mkfs",
        "dd if=",
        ":(){ :|:& };:",
    ]

    def __init__(self, policy: Optional[SafetyPolicy] = None):
        """Initialize safety validator.

        Args:
            policy: Safety policy (uses default if not provided)
        """
        self.policy = policy or SafetyPolicy()

    def validate_action(
        self,
        action: ProposedAction,
        context: Optional[ValidationContext] = None
    ) -> ValidationResponse:
        """Validate an action against the policy.

        Args:
            action: Proposed action
            context: Validation context

        Returns:
            ValidationResponse
        """
        if action.action_type == "file_edit":
            return self.validate_file_edit(action, context)
        elif action.action_type == "command":
            return self.validate_command(action, context)
        elif action.action_type == "network":
            return self.validate_network(action, context)
        else:
            return ValidationResponse(
                result=ValidationResult.ALLOWED,
                message="Action type not recognized, allowing by default"
            )

    def validate_file_edit(
        self,
        action: ProposedAction,
        context: Optional[ValidationContext] = None
    ) -> ValidationResponse:
        """Validate file edit action.

        Args:
            action: Proposed action
            context: Validation context

        Returns:
            ValidationResponse
        """
        warnings = []
        blocked = False

        for file_path in action.affected_files:
            if self._is_blocked_path(file_path):
                blocked = True
                return ValidationResponse(
                    result=ValidationResult.BLOCKED,
                    message=f"Path is blocked: {file_path}",
                    reason="Path matches blocked pattern"
                )

            if self._is_destructive_change(action):
                warnings.append(f"Potentially destructive change: {file_path}")
                if not self.policy.allow_destructive:
                    blocked = True

        if blocked:
            return ValidationResponse(
                result=ValidationResult.BLOCKED,
                message="Destructive changes are not allowed",
                warnings=warnings
            )

        if warnings and self._requires_confirmation(action):
            return ValidationResponse(
                result=ValidationResult.REQUIRES_CONFIRMATION,
                message="Action requires confirmation",
                warnings=warnings
            )

        return ValidationResponse(
            result=ValidationResult.ALLOWED,
            message="File edit allowed",
            warnings=warnings
        )

    def validate_command(
        self,
        action: ProposedAction,
        context: Optional[ValidationContext] = None
    ) -> ValidationResponse:
        """Validate command execution.

        Args:
            action: Proposed action
            context: Validation context

        Returns:
            ValidationResponse
        """
        command = action.parameters.get("command", "")

        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in command:
                return ValidationResponse(
                    result=ValidationResult.BLOCKED,
                    message=f"Dangerous command blocked: {dangerous}",
                    reason="Command matches dangerous pattern"
                )

        for pattern in self.DESTRUCTIVE_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                if not self.policy.allow_destructive:
                    return ValidationResponse(
                        result=ValidationResult.BLOCKED,
                        message="Destructive command pattern detected",
                        reason=f"Pattern: {pattern}"
                    )

        if self.policy.allowed_commands:
            allowed = any(cmd in command for cmd in self.policy.allowed_commands)
            if not allowed:
                return ValidationResponse(
                    result=ValidationResult.BLOCKED,
                    message="Command not in allowed list"
                )

        blocked = any(cmd in command for cmd in self.policy.blocked_commands)
        if blocked:
            return ValidationResponse(
                result=ValidationResult.BLOCKED,
                message="Command is blocked"
            )

        return ValidationResponse(
            result=ValidationResult.ALLOWED,
            message="Command allowed"
        )

    def validate_network(
        self,
        action: ProposedAction,
        context: Optional[ValidationContext] = None
    ) -> ValidationResponse:
        """Validate network access.

        Args:
            action: Proposed action
            context: Validation context

        Returns:
            ValidationResponse
        """
        if not self.policy.allow_network:
            return ValidationResponse(
                result=ValidationResult.BLOCKED,
                message="Network access is disabled"
            )

        return ValidationResponse(
            result=ValidationResult.ALLOWED,
            message="Network access allowed"
        )

    def _is_blocked_path(self, file_path: str) -> bool:
        """Check if path is blocked.

        Args:
            file_path: File path to check

        Returns:
            True if blocked
        """
        for blocked in self.policy.blocked_paths:
            if blocked in file_path:
                return True
        return False

    def _is_destructive_change(self, action: ProposedAction) -> bool:
        """Check if change is destructive.

        Args:
            action: Proposed action

        Returns:
            True if destructive
        """
        new_content = action.parameters.get("new_content", "")
        old_content = action.parameters.get("old_content", "")

        if len(new_content) < len(old_content) * 0.5:
            return True

        return False

    def _requires_confirmation(self, action: ProposedAction) -> bool:
        """Check if action requires confirmation.

        Args:
            action: Proposed action

        Returns:
            True if confirmation required
        """
        return action.action_type in self.policy.require_confirmation_for


class UserInterventionHandler:
    """Handler for user interventions."""

    def __init__(self):
        """Initialize user intervention handler."""
        self._pending_confirmations: Dict[str, ProposedAction] = {}
        self._interrupt_event: Optional[asyncio.Event] = None

    async def request_confirmation(
        self,
        action: ProposedAction,
        timeout: int = 60
    ) -> UserResponse:
        """Request user confirmation for an action.

        Args:
            action: Proposed action
            timeout: Timeout in seconds

        Returns:
            UserResponse
        """
        self._pending_confirmations[action.action_id] = action

        logger.info(f"[UserInterventionHandler] Requesting confirmation for: {action.action_id}")

        return UserResponse(
            decision=UserDecision.APPROVE,
            feedback="Auto-approved (implement UI callback)"
        )

    async def handle_interrupt(
        self,
        interrupt: UserInterrupt
    ) -> bool:
        """Handle user interrupt.

        Args:
            interrupt: User interrupt

        Returns:
            True if handled successfully
        """
        logger.info(f"[UserInterventionHandler] Handling interrupt: {interrupt.interrupt_type.value}")

        if interrupt.interrupt_type == InterruptType.PAUSE:
            return True
        elif interrupt.interrupt_type == InterruptType.RESUME:
            return True
        elif interrupt.interrupt_type == InterruptType.STOP:
            return True

        return False

    async def suggest_alternatives(
        self,
        action: ProposedAction
    ) -> List[ProposedAction]:
        """Suggest alternative actions.

        Args:
            action: Original proposed action

        Returns:
            List of alternative actions
        """
        alternatives = []

        if action.action_type == "file_edit":
            alt = ProposedAction(
                action_id=f"{action.action_id}_alt1",
                action_type="file_edit",
                description=f"Alternative: {action.description}",
                parameters=action.parameters.copy(),
                risk_level="low"
            )
            alternatives.append(alt)

        return alternatives

    def get_pending_confirmation(self, action_id: str) -> Optional[ProposedAction]:
        """Get pending confirmation by ID.

        Args:
            action_id: Action ID

        Returns:
            ProposedAction or None
        """
        return self._pending_confirmations.get(action_id)

    def clear_pending(self, action_id: str = None):
        """Clear pending confirmations.

        Args:
            action_id: Specific ID to clear, or None for all
        """
        if action_id:
            self._pending_confirmations.pop(action_id, None)
        else:
            self._pending_confirmations.clear()


def create_default_policy() -> SafetyPolicy:
    """Create default safety policy.

    Returns:
        SafetyPolicy with defaults
    """
    return SafetyPolicy(
        max_iterations=50,
        max_file_edits=100,
        max_command_executions=50,
        allow_destructive=False,
        allow_network=False,
        allow_shell=True,
        blocked_paths=[
            "/etc",
            "~/.ssh",
            "~/.aws",
            "**/secrets/**",
            "**/credentials/**",
        ],
        require_confirmation_for=["file_edit", "command", "delete"],
    )
