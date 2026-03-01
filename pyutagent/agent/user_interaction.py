"""User interaction for interactive error fixing.

This module provides user interaction capabilities:
- Interactive question handling
- User feedback collection
- Guided error resolution
- Progress updates
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Types of user interactions."""
    QUESTION = auto()
    CHOICE = auto()
    CONFIRMATION = auto()
    INPUT = auto()
    PROGRESS = auto()
    ERROR_REPORT = auto()
    SUGGESTION = auto()


class UserResponse(Enum):
    """User response types."""
    APPROVED = auto()
    REJECTED = auto()
    MODIFIED = auto()
    DEFERRED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class InteractionRequest:
    """A request for user interaction."""
    id: str
    interaction_type: InteractionType
    title: str
    message: str
    options: List[str] = field(default_factory=list)
    default_option: Optional[str] = None
    timeout_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class InteractionResponse:
    """A response from user interaction."""
    request_id: str
    response_type: UserResponse
    selected_option: Optional[str] = None
    user_input: Optional[str] = None
    modifications: Optional[str] = None
    feedback: Optional[str] = None
    responded_at: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorContext:
    """Context for an error requiring user interaction."""
    error_type: str
    error_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    attempted_fixes: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class GuidedFix:
    """A guided fix with user interaction."""
    success: bool
    fixed_code: Optional[str] = None
    user_feedback: Optional[str] = None
    iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class UserInteractionHandler:
    """Handles user interactions for error fixing.
    
    Features:
    - Interactive questions
    - Choice selection
    - Code review
    - Feedback collection
    """
    
    def __init__(
        self,
        timeout_seconds: int = 300,
        max_retries: int = 3
    ):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        self._pending_requests: Dict[str, InteractionRequest] = {}
        self._response_handlers: Dict[str, Callable] = {}
        self._interaction_history: List[InteractionResponse] = []
        
        self._callbacks: Dict[str, Callable] = {}
    
    def register_callback(
        self,
        event_type: str,
        callback: Callable
    ):
        """Register a callback for an event type.
        
        Args:
            event_type: Type of event ('request', 'response', 'timeout')
            callback: Callback function
        """
        self._callbacks[event_type] = callback
    
    async def request_user_help(
        self,
        error_context: ErrorContext,
        suggestions: List[str]
    ) -> InteractionResponse:
        """Request user help for an error.
        
        Args:
            error_context: Context of the error
            suggestions: List of suggested solutions
            
        Returns:
            User's response
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        options = suggestions + ["Provide custom fix", "Skip this error", "Abort"]
        
        request = InteractionRequest(
            id=request_id,
            interaction_type=InteractionType.CHOICE,
            title=f"Error: {error_context.error_type}",
            message=self._format_error_message(error_context),
            options=options,
            default_option=suggestions[0] if suggestions else "Skip this error",
            timeout_seconds=self.timeout_seconds,
            metadata={
                "error_context": error_context.__dict__,
                "suggestions": suggestions
            }
        )
        
        self._pending_requests[request_id] = request
        
        if 'request' in self._callbacks:
            await self._invoke_callback('request', request)
        
        response = await self._wait_for_response(request_id)
        
        return response
    
    def _format_error_message(self, context: ErrorContext) -> str:
        """Format error message for display."""
        parts = [f"**Error**: {context.error_message}"]
        
        if context.file_path:
            location = context.file_path
            if context.line_number:
                location += f":{context.line_number}"
            parts.append(f"**Location**: {location}")
        
        if context.code_snippet:
            parts.append(f"**Code**:\n```\n{context.code_snippet}\n```")
        
        if context.attempted_fixes:
            parts.append(f"**Attempted fixes**: {', '.join(context.attempted_fixes)}")
        
        return "\n\n".join(parts)
    
    async def _wait_for_response(
        self,
        request_id: str
    ) -> InteractionResponse:
        """Wait for user response."""
        request = self._pending_requests.get(request_id)
        
        if not request:
            return InteractionResponse(
                request_id=request_id,
                response_type=UserResponse.CANCELLED
            )
        
        try:
            while True:
                if request_id in self._response_handlers:
                    response = await self._response_handlers[request_id]()
                    self._interaction_history.append(response)
                    return response
                
                await asyncio.sleep(0.5)
                
        except asyncio.TimeoutError:
            return InteractionResponse(
                request_id=request_id,
                response_type=UserResponse.TIMEOUT
            )
        finally:
            self._pending_requests.pop(request_id, None)
    
    async def _invoke_callback(self, event_type: str, *args):
        """Invoke a registered callback."""
        callback = self._callbacks.get(event_type)
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
    
    def submit_response(
        self,
        request_id: str,
        response_type: UserResponse,
        selected_option: Optional[str] = None,
        user_input: Optional[str] = None,
        modifications: Optional[str] = None,
        feedback: Optional[str] = None
    ):
        """Submit a response to a pending request.
        
        Args:
            request_id: ID of the request
            response_type: Type of response
            selected_option: Selected option (for CHOICE type)
            user_input: User input (for INPUT type)
            modifications: Code modifications
            feedback: User feedback
        """
        response = InteractionResponse(
            request_id=request_id,
            response_type=response_type,
            selected_option=selected_option,
            user_input=user_input,
            modifications=modifications,
            feedback=feedback
        )
        
        async def response_handler():
            return response
        
        self._response_handlers[request_id] = response_handler
    
    async def ask_question(
        self,
        question: str,
        options: Optional[List[str]] = None,
        default: Optional[str] = None
    ) -> InteractionResponse:
        """Ask the user a question.
        
        Args:
            question: Question to ask
            options: Optional list of choices
            default: Default option
            
        Returns:
            User's response
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        interaction_type = InteractionType.CHOICE if options else InteractionType.QUESTION
        
        request = InteractionRequest(
            id=request_id,
            interaction_type=interaction_type,
            title="Question",
            message=question,
            options=options or [],
            default_option=default,
            timeout_seconds=self.timeout_seconds
        )
        
        self._pending_requests[request_id] = request
        
        if 'request' in self._callbacks:
            await self._invoke_callback('request', request)
        
        return await self._wait_for_response(request_id)
    
    async def request_code_review(
        self,
        original_code: str,
        modified_code: str,
        reason: str
    ) -> InteractionResponse:
        """Request code review from user.
        
        Args:
            original_code: Original code
            modified_code: Modified code
            reason: Reason for modification
            
        Returns:
            User's response
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        options = ["Approve", "Reject", "Modify"]
        
        request = InteractionRequest(
            id=request_id,
            interaction_type=InteractionType.CHOICE,
            title="Code Review Request",
            message=f"**Reason**: {reason}\n\n**Changes**:\n```diff\n{self._create_diff(original_code, modified_code)}\n```",
            options=options,
            default_option="Approve",
            timeout_seconds=self.timeout_seconds,
            metadata={
                "original_code": original_code,
                "modified_code": modified_code
            }
        )
        
        self._pending_requests[request_id] = request
        
        if 'request' in self._callbacks:
            await self._invoke_callback('request', request)
        
        return await self._wait_for_response(request_id)
    
    def _create_diff(self, original: str, modified: str) -> str:
        """Create a simple diff representation."""
        import difflib
        
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile='original',
            tofile='modified',
            lineterm=''
        )
        
        return ''.join(diff)
    
    async def collect_feedback(
        self,
        context: str,
        questions: List[str]
    ) -> Dict[str, str]:
        """Collect user feedback.
        
        Args:
            context: Context for feedback
            questions: List of questions
            
        Returns:
            Dictionary of question -> answer
        """
        feedback = {}
        
        for question in questions:
            response = await self.ask_question(question)
            if response.user_input:
                feedback[question] = response.user_input
            elif response.selected_option:
                feedback[question] = response.selected_option
        
        return feedback
    
    def get_pending_requests(self) -> List[InteractionRequest]:
        """Get all pending requests."""
        return list(self._pending_requests.values())
    
    def get_interaction_history(self) -> List[InteractionResponse]:
        """Get interaction history."""
        return self._interaction_history.copy()


class InteractiveFixer:
    """Provides interactive error fixing with user guidance.
    
    Features:
    - Guided error resolution
    - Step-by-step fixes
    - User confirmation
    - Rollback support
    """
    
    def __init__(
        self,
        interaction_handler: UserInteractionHandler,
        max_iterations: int = 5
    ):
        self.handler = interaction_handler
        self.max_iterations = max_iterations
    
    async def interactive_fix(
        self,
        code: str,
        error: str,
        error_location: Optional[tuple] = None,
        auto_suggestions: Optional[List[str]] = None
    ) -> GuidedFix:
        """Perform interactive fix with user guidance.
        
        Args:
            code: Code with error
            error: Error message
            error_location: (line, column) of error
            auto_suggestions: Automatic suggestions
            
        Returns:
            GuidedFix result
        """
        current_code = code
        iterations = 0
        attempted_fixes = []
        
        while iterations < self.max_iterations:
            iterations += 1
            
            context = ErrorContext(
                error_type="Compilation" if "compile" in error.lower() else "Runtime",
                error_message=error,
                line_number=error_location[0] if error_location else None,
                code_snippet=self._extract_snippet(current_code, error_location),
                attempted_fixes=attempted_fixes,
                suggestions=auto_suggestions or []
            )
            
            suggestions = auto_suggestions or self._generate_suggestions(error, current_code)
            
            response = await self.handler.request_user_help(context, suggestions)
            
            if response.response_type == UserResponse.APPROVED:
                if response.selected_option in suggestions:
                    fixed_code = await self._apply_suggestion(
                        current_code,
                        response.selected_option,
                        error_location
                    )
                    if fixed_code:
                        return GuidedFix(
                            success=True,
                            fixed_code=fixed_code,
                            user_feedback=response.feedback,
                            iterations=iterations
                        )
            
            elif response.response_type == UserResponse.MODIFIED:
                if response.modifications:
                    current_code = response.modifications
                    attempted_fixes.append("User modification")
                    continue
            
            elif response.response_type in (UserResponse.CANCELLED, UserResponse.TIMEOUT):
                return GuidedFix(
                    success=False,
                    user_feedback=response.feedback or "User cancelled",
                    iterations=iterations
                )
            
            elif response.selected_option == "Skip this error":
                return GuidedFix(
                    success=False,
                    user_feedback="Skipped by user",
                    iterations=iterations
                )
            
            elif response.selected_option == "Abort":
                return GuidedFix(
                    success=False,
                    user_feedback="Aborted by user",
                    iterations=iterations
                )
            
            elif response.selected_option == "Provide custom fix":
                if response.user_input:
                    current_code = response.user_input
                    attempted_fixes.append("Custom fix")
                    continue
        
        return GuidedFix(
            success=False,
            fixed_code=current_code,
            user_feedback="Max iterations reached",
            iterations=iterations
        )
    
    def _extract_snippet(
        self,
        code: str,
        location: Optional[tuple],
        context_lines: int = 5
    ) -> str:
        """Extract code snippet around error location."""
        if not location:
            return code[:500] if len(code) > 500 else code
        
        lines = code.split('\n')
        line_no = location[0]
        
        start = max(0, line_no - context_lines)
        end = min(len(lines), line_no + context_lines + 1)
        
        return '\n'.join(lines[start:end])
    
    def _generate_suggestions(
        self,
        error: str,
        code: str
    ) -> List[str]:
        """Generate automatic suggestions for the error."""
        suggestions = []
        
        if "cannot find symbol" in error.lower():
            suggestions.append("Add missing import statement")
            suggestions.append("Check for typos in class/method name")
        
        elif "missing semicolon" in error.lower():
            suggestions.append("Add missing semicolon")
        
        elif "nullpointer" in error.lower():
            suggestions.append("Add null check")
            suggestions.append("Initialize the variable")
        
        elif "assertion" in error.lower():
            suggestions.append("Update assertion expected value")
            suggestions.append("Fix the actual value logic")
        
        else:
            suggestions.append("Regenerate the code")
            suggestions.append("Try alternative approach")
        
        return suggestions[:4]
    
    async def _apply_suggestion(
        self,
        code: str,
        suggestion: str,
        location: Optional[tuple]
    ) -> Optional[str]:
        """Apply a suggestion to the code."""
        return code


def create_user_interaction_handler(
    timeout: int = 300,
    max_retries: int = 3
) -> UserInteractionHandler:
    """Create a UserInteractionHandler instance.
    
    Args:
        timeout: Timeout for user responses
        max_retries: Maximum retry attempts
        
    Returns:
        Configured UserInteractionHandler
    """
    return UserInteractionHandler(
        timeout_seconds=timeout,
        max_retries=max_retries
    )
