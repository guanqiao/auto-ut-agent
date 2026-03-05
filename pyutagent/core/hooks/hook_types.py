"""Hook Type Definitions.

This module defines all hook types for the agent lifecycle.
"""

from enum import Enum, auto


class HookType(Enum):
    """Types of hooks available in the agent lifecycle.
    
    Hook execution order:
    1. User Interaction Hooks
    2. Agent Lifecycle Hooks
    3. Tool Execution Hooks
    4. File Operation Hooks
    5. Plan Management Hooks
    6. Error Handling Hooks
    """
    
    USER_PROMPT_SUBMIT = auto()
    USER_RESPONSE_RECEIVE = auto()
    
    PRE_AGENT_START = auto()
    POST_AGENT_STOP = auto()
    
    PRE_TASK = auto()
    POST_TASK = auto()
    TASK_SUCCESS = auto()
    TASK_FAILURE = auto()
    
    PRE_SUBTASK = auto()
    POST_SUBTASK = auto()
    
    PRE_TOOL_USE = auto()
    POST_TOOL_USE = auto()
    TOOL_ERROR = auto()
    
    PRE_FILE_READ = auto()
    POST_FILE_READ = auto()
    PRE_FILE_WRITE = auto()
    POST_FILE_WRITE = auto()
    PRE_FILE_DELETE = auto()
    POST_FILE_DELETE = auto()
    
    ON_PLAN_CREATED = auto()
    ON_PLAN_ADJUSTED = auto()
    ON_STEP_START = auto()
    ON_STEP_COMPLETE = auto()
    ON_STEP_FAIL = auto()
    
    ON_ERROR = auto()
    ON_RECOVERY = auto()
    ON_RETRY = auto()
    
    ON_LLM_CALL = auto()
    ON_LLM_RESPONSE = auto()
    ON_LLM_ERROR = auto()
    
    ON_COMPILE_START = auto()
    ON_COMPILE_SUCCESS = auto()
    ON_COMPILE_ERROR = auto()
    
    ON_TEST_START = auto()
    ON_TEST_SUCCESS = auto()
    ON_TEST_FAILURE = auto()
    
    ON_COVERAGE_UPDATE = auto()
    
    @property
    def category(self) -> str:
        """Get the category of this hook type."""
        if self in (
            HookType.USER_PROMPT_SUBMIT,
            HookType.USER_RESPONSE_RECEIVE,
        ):
            return "user_interaction"
        
        if self in (
            HookType.PRE_AGENT_START,
            HookType.POST_AGENT_STOP,
            HookType.PRE_TASK,
            HookType.POST_TASK,
            HookType.TASK_SUCCESS,
            HookType.TASK_FAILURE,
            HookType.PRE_SUBTASK,
            HookType.POST_SUBTASK,
        ):
            return "agent_lifecycle"
        
        if self in (
            HookType.PRE_TOOL_USE,
            HookType.POST_TOOL_USE,
            HookType.TOOL_ERROR,
        ):
            return "tool_execution"
        
        if self in (
            HookType.PRE_FILE_READ,
            HookType.POST_FILE_READ,
            HookType.PRE_FILE_WRITE,
            HookType.POST_FILE_WRITE,
            HookType.PRE_FILE_DELETE,
            HookType.POST_FILE_DELETE,
        ):
            return "file_operations"
        
        if self in (
            HookType.ON_PLAN_CREATED,
            HookType.ON_PLAN_ADJUSTED,
            HookType.ON_STEP_START,
            HookType.ON_STEP_COMPLETE,
            HookType.ON_STEP_FAIL,
        ):
            return "plan_management"
        
        if self in (
            HookType.ON_ERROR,
            HookType.ON_RECOVERY,
            HookType.ON_RETRY,
        ):
            return "error_handling"
        
        if self in (
            HookType.ON_LLM_CALL,
            HookType.ON_LLM_RESPONSE,
            HookType.ON_LLM_ERROR,
        ):
            return "llm_interaction"
        
        if self in (
            HookType.ON_COMPILE_START,
            HookType.ON_COMPILE_SUCCESS,
            HookType.ON_COMPILE_ERROR,
        ):
            return "compilation"
        
        if self in (
            HookType.ON_TEST_START,
            HookType.ON_TEST_SUCCESS,
            HookType.ON_TEST_FAILURE,
        ):
            return "testing"
        
        return "other"
    
    @property
    def is_pre_hook(self) -> bool:
        """Check if this is a pre-event hook."""
        return self.name.startswith("PRE_")
    
    @property
    def is_post_hook(self) -> bool:
        """Check if this is a post-event hook."""
        return self.name.startswith("POST_")
    
    @property
    def is_error_hook(self) -> bool:
        """Check if this is an error-related hook."""
        return "ERROR" in self.name or "FAILURE" in self.name
