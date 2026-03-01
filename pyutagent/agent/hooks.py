"""Hooks mechanism for lifecycle automation.

This module provides:
- Hook: Lifecycle event handlers
- HookRegistry: Management of hooks
- HookExecutor: Execute hooks at specific points
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class HookEvent(Enum):
    """Lifecycle events."""
    STARTUP = auto()
    SHUTDOWN = auto()
    BEFORE_TASK = auto()
    AFTER_TASK = auto()
    TASK_SUCCESS = auto()
    TASK_FAILURE = auto()
    BEFORE_TOOL_CALL = auto()
    AFTER_TOOL_CALL = auto()
    BEFORE_LLM_CALL = auto()
    AFTER_LLM_CALL = auto()
    ERROR = auto()
    USER_INPUT = auto()
    AGENT_MESSAGE = auto()


class HookPriority(Enum):
    """Hook execution priority."""
    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


@dataclass
class Hook:
    """A hook for lifecycle events."""
    id: str
    name: str
    event: HookEvent
    handler: Callable[..., Awaitable[Any]]
    priority: HookPriority = HookPriority.NORMAL
    enabled: bool = True
    timeout: int = 30
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())


@dataclass
class HookContext:
    """Context passed to hooks."""
    event: HookEvent
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"


@dataclass
class HookResult:
    """Result of hook execution."""
    hook_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class HookRegistry:
    """Registry for managing hooks."""

    def __init__(self):
        """Initialize hook registry."""
        self._hooks: Dict[HookEvent, List[Hook]] = {event: [] for event in HookEvent}
        self._global_hooks: List[Hook] = []
        logger.debug("[HookRegistry] Initialized")

    def register(
        self,
        name: str,
        event: HookEvent,
        handler: Callable[..., Awaitable[Any]],
        priority: HookPriority = HookPriority.NORMAL,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Hook:
        """Register a hook.

        Args:
            name: Hook name
            event: Event to hook
            handler: Async handler function
            priority: Execution priority
            description: Hook description
            metadata: Additional metadata

        Returns:
            Registered hook
        """
        hook = Hook(
            id=str(uuid4()),
            name=name,
            event=event,
            handler=handler,
            priority=priority,
            description=description,
            metadata=metadata or {}
        )

        self._hooks[event].append(hook)
        self._hooks[event].sort(key=lambda h: h.priority.value, reverse=True)

        logger.info(f"[HookRegistry] Registered hook: {name} for {event.name}")
        return hook

    def register_global(
        self,
        name: str,
        handler: Callable[..., Awaitable[Any]],
        priority: HookPriority = HookPriority.NORMAL,
        description: str = ""
    ) -> Hook:
        """Register a global hook that runs for all events.

        Args:
            name: Hook name
            handler: Async handler function
            priority: Execution priority
            description: Hook description

        Returns:
            Registered hook
        """
        hook = Hook(
            id=str(uuid4()),
            name=name,
            event=None,
            handler=handler,
            priority=priority,
            description=description
        )

        self._global_hooks.append(hook)
        self._global_hooks.sort(key=lambda h: h.priority.value, reverse=True)

        logger.info(f"[HookRegistry] Registered global hook: {name}")
        return hook

    def unregister(self, hook_id: str) -> bool:
        """Unregister a hook.

        Args:
            hook_id: Hook ID

        Returns:
            True if hook was unregistered
        """
        for event_hooks in self._hooks.values():
            for i, hook in enumerate(event_hooks):
                if hook.id == hook_id:
                    event_hooks.pop(i)
                    logger.info(f"[HookRegistry] Unregistered hook: {hook.name}")
                    return True

        for i, hook in enumerate(self._global_hooks):
            if hook.id == hook_id:
                self._global_hooks.pop(i)
                logger.info(f"[HookRegistry] Unregistered global hook: {hook.name}")
                return True

        return False

    def get_hooks(self, event: HookEvent) -> List[Hook]:
        """Get hooks for an event.

        Args:
            event: Event

        Returns:
            List of hooks
        """
        return [h for h in self._hooks.get(event, []) if h.enabled]

    def get_global_hooks(self) -> List[Hook]:
        """Get all global hooks.

        Returns:
            List of global hooks
        """
        return [h for h in self._global_hooks if h.enabled]

    def enable_hook(self, hook_id: str) -> bool:
        """Enable a hook.

        Args:
            hook_id: Hook ID

        Returns:
            True if hook was enabled
        """
        hook = self._find_hook(hook_id)
        if hook:
            hook.enabled = True
            return True
        return False

    def disable_hook(self, hook_id: str) -> bool:
        """Disable a hook.

        Args:
            hook_id: Hook ID

        Returns:
            True if hook was disabled
        """
        hook = self._find_hook(hook_id)
        if hook:
            hook.enabled = False
            return True
        return False

    def _find_hook(self, hook_id: str) -> Optional[Hook]:
        """Find a hook by ID."""
        for event_hooks in self._hooks.values():
            for hook in event_hooks:
                if hook.id == hook_id:
                    return hook

        for hook in self._global_hooks:
            if hook.id == hook_id:
                return hook

        return None


class HookExecutor:
    """Executor for hooks."""

    def __init__(self, registry: Optional[HookRegistry] = None):
        """Initialize hook executor.

        Args:
            registry: Hook registry
        """
        self.registry = registry or HookRegistry()
        self._execution_history: List[HookResult] = []
        logger.debug("[HookExecutor] Initialized")

    async def execute(
        self,
        event: HookEvent,
        context: Optional[HookContext] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> List[HookResult]:
        """Execute hooks for an event.

        Args:
            event: Event to execute
            context: Execution context
            data: Additional data

        Returns:
            List of hook results
        """
        if context is None:
            context = HookContext(event=event, data=data or {})
        elif data:
            context.data.update(data)

        results = []

        for hook in self.registry.get_global_hooks():
            result = await self._execute_hook(hook, context)
            results.append(result)

        for hook in self.registry.get_hooks(event):
            result = await self._execute_hook(hook, context)
            results.append(result)

        self._execution_history.extend(results)
        return results

    async def _execute_hook(self, hook: Hook, context: HookContext) -> HookResult:
        """Execute a single hook.

        Args:
            hook: Hook to execute
            context: Execution context

        Returns:
            Hook result
        """
        start_time = datetime.now()

        try:
            if asyncio.iscoroutinefunction(hook.handler):
                result = await asyncio.wait_for(
                    hook.handler(context),
                    timeout=hook.timeout
                )
            else:
                result = hook.handler(context)

            duration = (datetime.now() - start_time).total_seconds() * 1000

            logger.debug(f"[HookExecutor] Hook {hook.name} executed in {duration:.2f}ms")

            return HookResult(
                hook_id=hook.id,
                success=True,
                output=result,
                duration_ms=duration
            )

        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.warning(f"[HookExecutor] Hook {hook.name} timed out after {duration:.2f}ms")

            return HookResult(
                hook_id=hook.id,
                success=False,
                error=f"Hook timed out after {hook.timeout}s",
                duration_ms=duration
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.exception(f"[HookExecutor] Hook {hook.name} failed: {e}")

            return HookResult(
                hook_id=hook.id,
                success=False,
                error=str(e),
                duration_ms=duration
            )

    def get_history(self) -> List[HookResult]:
        """Get execution history.

        Returns:
            List of hook results
        """
        return self._execution_history.copy()

    def clear_history(self):
        """Clear execution history."""
        self._execution_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Statistics dictionary
        """
        total = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.success)
        failed = total - successful

        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "avg_duration_ms": sum(r.duration_ms for r in self._execution_history) / total if total > 0 else 0
        }


_global_registry: Optional[HookRegistry] = None
_global_executor: Optional[HookExecutor] = None


def get_hook_registry() -> HookRegistry:
    """Get global hook registry.

    Returns:
        Global HookRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = HookRegistry()
    return _global_registry


def get_hook_executor() -> HookExecutor:
    """Get global hook executor.

    Returns:
        Global HookExecutor
    """
    global _global_executor
    if _global_executor is None:
        _global_executor = HookExecutor(get_hook_registry())
    return _global_executor


def register_hook(
    name: str,
    event: HookEvent,
    handler: Callable[..., Awaitable[Any]],
    priority: HookPriority = HookPriority.NORMAL,
    description: str = ""
) -> Hook:
    """Register a hook in global registry.

    Args:
        name: Hook name
        event: Event to hook
        handler: Async handler function
        priority: Execution priority
        description: Hook description

    Returns:
        Registered hook
    """
    return get_hook_registry().register(name, event, handler, priority, description)


async def trigger_hook(event: HookEvent, data: Optional[Dict[str, Any]] = None) -> List[HookResult]:
    """Trigger hooks for an event.

    Args:
        event: Event to trigger
        data: Event data

    Returns:
        List of hook results
    """
    return await get_hook_executor().execute(event, data=data)


def create_startup_hook(name: str, handler: Callable[..., Awaitable[Any]]) -> Hook:
    """Create a startup hook.

    Args:
        name: Hook name
        handler: Handler function

    Returns:
        Registered hook
    """
    return register_hook(name, HookEvent.STARTUP, handler, HookPriority.HIGH, "Startup handler")


def create_shutdown_hook(name: str, handler: Callable[..., Awaitable[Any]]) -> Hook:
    """Create a shutdown hook.

    Args:
        name: Hook name
        handler: Handler function

    Returns:
        Registered hook
    """
    return register_hook(name, HookEvent.SHUTDOWN, handler, HookPriority.HIGH, "Shutdown handler")


def create_error_hook(name: str, handler: Callable[..., Awaitable[Any]]) -> Hook:
    """Create an error hook.

    Args:
        name: Hook name
        handler: Handler function

    Returns:
        Registered hook
    """
    return register_hook(name, HookEvent.ERROR, handler, HookPriority.CRITICAL, "Error handler")
