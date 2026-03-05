"""Hook Registry for Managing Hooks.

This module provides:
- HookRegistry: Central registry for all hooks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import logging

from .hook_types import HookType
from .hook import Hook, HookContext, HookResult

logger = logging.getLogger(__name__)


class HookRegistry:
    """Central registry for managing hooks.
    
    Features:
    - Register/unregister hooks
    - Execute hooks by type
    - Priority-based execution
    - Error handling
    """
    
    def __init__(self):
        """Initialize the hook registry."""
        self._hooks: Dict[HookType, List[Hook]] = {
            hook_type: [] for hook_type in HookType
        }
        self._hooks_by_name: Dict[str, Hook] = {}
        self._disabled_hook_types: Set[HookType] = set()
    
    def register(self, hook: Hook) -> None:
        """Register a hook.
        
        Args:
            hook: Hook to register
        """
        if hook.name in self._hooks_by_name:
            logger.warning(f"Hook '{hook.name}' already registered, replacing")
            self.unregister(hook.name)
        
        self._hooks[hook.hook_type].append(hook)
        self._hooks[hook.hook_type].sort(key=lambda h: h.priority, reverse=True)
        self._hooks_by_name[hook.name] = hook
        
        logger.debug(f"Registered hook: {hook.name} for {hook.hook_type.name}")
    
    def unregister(self, hook_name: str) -> bool:
        """Unregister a hook by name.
        
        Args:
            hook_name: Name of hook to unregister
            
        Returns:
            True if hook was found and removed
        """
        hook = self._hooks_by_name.pop(hook_name, None)
        if hook:
            self._hooks[hook.hook_type] = [
                h for h in self._hooks[hook.hook_type] if h.name != hook_name
            ]
            logger.debug(f"Unregistered hook: {hook_name}")
            return True
        return False
    
    def get_hook(self, hook_name: str) -> Optional[Hook]:
        """Get a hook by name.
        
        Args:
            hook_name: Name of hook
            
        Returns:
            Hook or None if not found
        """
        return self._hooks_by_name.get(hook_name)
    
    def get_hooks(self, hook_type: HookType) -> List[Hook]:
        """Get all hooks for a type.
        
        Args:
            hook_type: Type of hooks to get
            
        Returns:
            List of hooks (sorted by priority)
        """
        return self._hooks[hook_type].copy()
    
    def enable_hook(self, hook_name: str) -> bool:
        """Enable a hook.
        
        Args:
            hook_name: Name of hook to enable
            
        Returns:
            True if hook was found
        """
        hook = self._hooks_by_name.get(hook_name)
        if hook:
            hook.enable()
            return True
        return False
    
    def disable_hook(self, hook_name: str) -> bool:
        """Disable a hook.
        
        Args:
            hook_name: Name of hook to disable
            
        Returns:
            True if hook was found
        """
        hook = self._hooks_by_name.get(hook_name)
        if hook:
            hook.disable()
            return True
        return False
    
    def enable_hook_type(self, hook_type: HookType) -> None:
        """Enable all hooks of a type.
        
        Args:
            hook_type: Type of hooks to enable
        """
        self._disabled_hook_types.discard(hook_type)
        for hook in self._hooks[hook_type]:
            hook.enable()
    
    def disable_hook_type(self, hook_type: HookType) -> None:
        """Disable all hooks of a type.
        
        Args:
            hook_type: Type of hooks to disable
        """
        self._disabled_hook_types.add(hook_type)
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        context: HookContext,
    ) -> HookResult:
        """Execute all hooks of a type.
        
        Hooks are executed in priority order (highest first).
        Execution stops if a hook returns should_abort=True.
        
        Args:
            hook_type: Type of hooks to execute
            context: Context to pass to hooks
            
        Returns:
            Combined HookResult
        """
        if hook_type in self._disabled_hook_types:
            return HookResult.ok()
        
        hooks = self._hooks[hook_type]
        combined_data: Dict[str, Any] = {}
        
        for hook in hooks:
            try:
                result = await hook.execute(context)
                
                combined_data.update(result.data)
                
                if result.modified_context:
                    context.data.update(result.modified_context)
                
                if result.should_abort:
                    logger.info(f"Hook '{hook.name}' requested abort: {result.message}")
                    return HookResult(
                        success=result.success,
                        data=combined_data,
                        should_abort=True,
                        message=result.message,
                    )
                
            except Exception as e:
                logger.error(f"Hook '{hook.name}' raised exception: {e}")
                combined_data[f"error_{hook.name}"] = str(e)
        
        return HookResult(success=True, data=combined_data)
    
    def clear(self) -> None:
        """Clear all registered hooks."""
        self._hooks = {hook_type: [] for hook_type in HookType}
        self._hooks_by_name.clear()
        self._disabled_hook_types.clear()
        logger.info("All hooks cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about registered hooks.
        
        Returns:
            Dictionary with hook statistics
        """
        stats = {
            "total_hooks": len(self._hooks_by_name),
            "hooks_by_type": {},
            "hooks_by_category": {},
            "disabled_types": [t.name for t in self._disabled_hook_types],
        }
        
        for hook_type, hooks in self._hooks.items():
            stats["hooks_by_type"][hook_type.name] = len(hooks)
            
            category = hook_type.category
            if category not in stats["hooks_by_category"]:
                stats["hooks_by_category"][category] = 0
            stats["hooks_by_category"][category] += len(hooks)
        
        return stats
    
    def list_hooks(self) -> List[Dict[str, Any]]:
        """List all registered hooks with their stats.
        
        Returns:
            List of hook information dictionaries
        """
        return [hook.stats for hook in self._hooks_by_name.values()]


from typing import Any
