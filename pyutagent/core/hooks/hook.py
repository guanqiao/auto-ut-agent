"""Hook Definition and Context.

This module provides:
- HookContext: Context passed to hook handlers
- HookResult: Result returned by hook handlers
- Hook: Hook definition
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
import asyncio
import logging

from .hook_types import HookType

logger = logging.getLogger(__name__)


@dataclass
class HookContext:
    """Context passed to a hook handler.
    
    Contains all relevant information about the event that triggered the hook.
    """
    
    hook_type: HookType
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from data."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in data."""
        self.data[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a value from metadata."""
        return self.metadata.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hook_type": self.hook_type.name,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HookResult:
    """Result returned by a hook handler.
    
    Controls the flow of hook execution and allows modifying context.
    """
    
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    should_abort: bool = False
    modified_context: Optional[Dict[str, Any]] = None
    message: str = ""
    error: Optional[str] = None
    
    @classmethod
    def ok(cls, data: Optional[Dict[str, Any]] = None) -> "HookResult":
        """Create a successful result."""
        return cls(success=True, data=data or {})
    
    @classmethod
    def abort(cls, reason: str = "") -> "HookResult":
        """Create an abort result."""
        return cls(success=False, should_abort=True, message=reason)
    
    @classmethod
    def modify(cls, context: Dict[str, Any]) -> "HookResult":
        """Create a result that modifies the context."""
        return cls(success=True, modified_context=context)
    
    @classmethod
    def error(cls, error: str) -> "HookResult":
        """Create an error result."""
        return cls(success=False, error=error)


class Hook:
    """A hook that can be registered and executed.
    
    Hooks are callbacks that are executed at specific points in the agent lifecycle.
    They can:
    - Observe and log events
    - Modify data before/after operations
    - Abort operations
    - Trigger side effects
    """
    
    def __init__(
        self,
        name: str,
        hook_type: HookType,
        handler: Callable[[HookContext], HookResult],
        priority: int = 0,
        condition: Optional[Callable[[HookContext], bool]] = None,
        description: str = "",
        enabled: bool = True,
    ):
        """Initialize a hook.
        
        Args:
            name: Unique name for this hook
            hook_type: Type of hook (when it should be triggered)
            handler: Function to call when hook is triggered
            priority: Priority for execution order (higher = earlier)
            condition: Optional condition function to check before execution
            description: Human-readable description
            enabled: Whether the hook is enabled
        """
        self.name = name
        self.hook_type = hook_type
        self.handler = handler
        self.priority = priority
        self.condition = condition
        self.description = description
        self.enabled = enabled
        
        self._call_count = 0
        self._last_execution_time: Optional[datetime] = None
        self._errors: List[str] = []
    
    async def execute(self, context: HookContext) -> HookResult:
        """Execute the hook.
        
        Args:
            context: Hook context with event data
            
        Returns:
            HookResult from the handler
        """
        if not self.enabled:
            return HookResult.ok()
        
        if self.condition and not self.condition(context):
            return HookResult.ok()
        
        self._call_count += 1
        self._last_execution_time = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(self.handler):
                result = await self.handler(context)
            else:
                result = self.handler(context)
            
            if not isinstance(result, HookResult):
                result = HookResult.ok(data=result if isinstance(result, dict) else {})
            
            return result
            
        except Exception as e:
            error_msg = f"Hook '{self.name}' failed: {e}"
            logger.error(error_msg)
            self._errors.append(error_msg)
            return HookResult.error(error_msg)
    
    def enable(self) -> None:
        """Enable the hook."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the hook."""
        self.enabled = False
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get hook statistics."""
        return {
            "name": self.name,
            "hook_type": self.hook_type.name,
            "enabled": self.enabled,
            "call_count": self._call_count,
            "last_execution": self._last_execution_time.isoformat() if self._last_execution_time else None,
            "error_count": len(self._errors),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Hook(name={self.name!r}, type={self.hook_type.name}, priority={self.priority})"


def hook(
    hook_type: HookType,
    priority: int = 0,
    condition: Optional[Callable[[HookContext], bool]] = None,
    name: Optional[str] = None,
):
    """Decorator to create a hook from a function.
    
    Usage:
        @hook(HookType.PRE_FILE_WRITE, priority=10)
        def my_hook(context: HookContext) -> HookResult:
            # Do something
            return HookResult.ok()
    
    Args:
        hook_type: Type of hook
        priority: Priority for execution order
        condition: Optional condition function
        name: Optional hook name (defaults to function name)
    
    Returns:
        Decorated function that is also a Hook
    """
    def decorator(func: Callable[[HookContext], HookResult]) -> Hook:
        hook_name = name or func.__name__
        return Hook(
            name=hook_name,
            hook_type=hook_type,
            handler=func,
            priority=priority,
            condition=condition,
            description=func.__doc__ or "",
        )
    
    return decorator
