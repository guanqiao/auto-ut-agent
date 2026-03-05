"""Hooks System for Agent Lifecycle Events.

This module provides:
- HookType: Types of hooks
- Hook: Hook definition
- HookRegistry: Hook registration and execution
- HookManager: High-level hook management
"""

from .hook_types import HookType
from .hook import Hook, HookContext, HookResult
from .registry import HookRegistry
from .manager import HookManager, get_hook_manager, trigger_hook

__all__ = [
    "HookType",
    "Hook",
    "HookContext",
    "HookResult",
    "HookRegistry",
    "HookManager",
    "get_hook_manager",
    "trigger_hook",
]
