"""Hook Manager for High-Level Hook Management.

This module provides:
- HookManager: High-level hook management
- Built-in hooks
- Global hook manager instance
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from .hook_types import HookType
from .hook import Hook, HookContext, HookResult, hook
from .registry import HookRegistry

logger = logging.getLogger(__name__)


class HookManager:
    """High-level hook management.
    
    Features:
    - Register built-in hooks
    - Trigger hooks easily
    - Manage hook lifecycle
    - Integration with agent
    """
    
    def __init__(self):
        """Initialize hook manager."""
        self.registry = HookRegistry()
        self._builtin_hooks_registered = False
    
    def register_builtin_hooks(self) -> None:
        """Register all built-in hooks."""
        if self._builtin_hooks_registered:
            return
        
        self._register_logging_hooks()
        self._register_file_operation_hooks()
        self._register_error_handling_hooks()
        
        self._builtin_hooks_registered = True
        logger.info("Built-in hooks registered")
    
    def _register_logging_hooks(self) -> None:
        """Register logging-related hooks."""
        @hook(HookType.PRE_TOOL_USE, priority=-100, name="log_tool_use")
        def log_tool_use(context: HookContext) -> HookResult:
            tool_name = context.get("tool_name", "unknown")
            logger.info(f"[Hook] Tool use: {tool_name}")
            return HookResult.ok()
        
        @hook(HookType.POST_TOOL_USE, priority=-100, name="log_tool_result")
        def log_tool_result(context: HookContext) -> HookResult:
            tool_name = context.get("tool_name", "unknown")
            success = context.get("success", False)
            logger.info(f"[Hook] Tool result: {tool_name} - {'success' if success else 'failed'}")
            return HookResult.ok()
        
        @hook(HookType.ON_ERROR, priority=100, name="log_error")
        def log_error(context: HookContext) -> HookResult:
            error = context.get("error", "unknown error")
            logger.error(f"[Hook] Error occurred: {error}")
            return HookResult.ok()
        
        self.registry.register(log_tool_use)
        self.registry.register(log_tool_result)
        self.registry.register(log_error)
    
    def _register_file_operation_hooks(self) -> None:
        """Register file operation hooks."""
        @hook(HookType.PRE_FILE_WRITE, priority=50, name="validate_file_path")
        def validate_file_path(context: HookContext) -> HookResult:
            file_path = context.get("file_path")
            if not file_path:
                return HookResult.abort("No file path provided")
            
            path = Path(file_path)
            if not path.is_absolute():
                logger.warning(f"Relative file path: {file_path}")
            
            return HookResult.ok()
        
        @hook(HookType.POST_FILE_WRITE, priority=10, name="log_file_write")
        def log_file_write(context: HookContext) -> HookResult:
            file_path = context.get("file_path", "unknown")
            content_length = context.get("content_length", 0)
            logger.info(f"[Hook] File written: {file_path} ({content_length} bytes)")
            return HookResult.ok()
        
        self.registry.register(validate_file_path)
        self.registry.register(log_file_write)
    
    def _register_error_handling_hooks(self) -> None:
        """Register error handling hooks."""
        @hook(HookType.ON_RECOVERY, priority=50, name="log_recovery")
        def log_recovery(context: HookContext) -> HookResult:
            error = context.get("error", "unknown")
            recovery_action = context.get("recovery_action", "unknown")
            logger.info(f"[Hook] Recovery: {recovery_action} for error: {error}")
            return HookResult.ok()
        
        @hook(HookType.ON_RETRY, priority=50, name="log_retry")
        def log_retry(context: HookContext) -> HookResult:
            step_name = context.get("step_name", "unknown")
            retry_count = context.get("retry_count", 0)
            max_retries = context.get("max_retries", 0)
            logger.info(f"[Hook] Retry {retry_count}/{max_retries} for: {step_name}")
            return HookResult.ok()
        
        self.registry.register(log_recovery)
        self.registry.register(log_retry)
    
    def register(self, hook: Hook) -> None:
        """Register a hook.
        
        Args:
            hook: Hook to register
        """
        self.registry.register(hook)
    
    def unregister(self, hook_name: str) -> bool:
        """Unregister a hook.
        
        Args:
            hook_name: Name of hook to unregister
            
        Returns:
            True if hook was removed
        """
        return self.registry.unregister(hook_name)
    
    async def trigger(
        self,
        hook_type: HookType,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HookResult:
        """Trigger hooks of a specific type.
        
        Args:
            hook_type: Type of hooks to trigger
            data: Data to pass to hooks
            metadata: Metadata to pass to hooks
            
        Returns:
            Combined HookResult
        """
        context = HookContext(
            hook_type=hook_type,
            data=data or {},
            metadata=metadata or {},
        )
        
        return await self.registry.execute_hooks(hook_type, context)
    
    def get_hook(self, hook_name: str) -> Optional[Hook]:
        """Get a hook by name."""
        return self.registry.get_hook(hook_name)
    
    def get_hooks(self, hook_type: HookType) -> List[Hook]:
        """Get all hooks of a type."""
        return self.registry.get_hooks(hook_type)
    
    def enable_hook(self, hook_name: str) -> bool:
        """Enable a hook."""
        return self.registry.enable_hook(hook_name)
    
    def disable_hook(self, hook_name: str) -> bool:
        """Disable a hook."""
        return self.registry.disable_hook(hook_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hook statistics."""
        return self.registry.get_stats()
    
    def clear(self) -> None:
        """Clear all hooks."""
        self.registry.clear()
        self._builtin_hooks_registered = False


_global_hook_manager: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """Get the global hook manager instance.
    
    Returns:
        Global HookManager instance
    """
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
        _global_hook_manager.register_builtin_hooks()
    return _global_hook_manager


async def trigger_hook(
    hook_type: HookType,
    data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> HookResult:
    """Trigger hooks using the global hook manager.
    
    This is a convenience function for triggering hooks without
    directly accessing the hook manager.
    
    Args:
        hook_type: Type of hooks to trigger
        data: Data to pass to hooks
        metadata: Metadata to pass to hooks
        
    Returns:
        Combined HookResult
    """
    return await get_hook_manager().trigger(hook_type, data, metadata)
