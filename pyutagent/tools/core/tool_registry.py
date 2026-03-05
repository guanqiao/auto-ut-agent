"""Tool Registry for Managing Tools.

This module provides:
- ToolRegistry: Central registry for tools
- Global registry instance
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable
import importlib

from .tool_base import ToolBase, ToolCategory, ToolMeta
from .tool_result import ToolResult
from .tool_context import ToolContext

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for managing tools.
    
    Features:
    - Register/unregister tools
    - Discover tools by category
    - Get tool schemas for LLM
    - Execute tools by name
    - Tool discovery from modules
    """
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Type[ToolBase]] = {}
        self._instances: Dict[str, ToolBase] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            cat: [] for cat in ToolCategory
        }
        self._aliases: Dict[str, str] = {}
    
    def register(
        self,
        tool_class: Type[ToolBase],
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a tool class.
        
        Args:
            tool_class: Tool class to register
            name: Optional name override
            aliases: Optional aliases for the tool
        """
        tool_name = name or tool_class.name
        
        if not tool_name:
            raise ValueError("Tool must have a name")
        
        if tool_name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool_name}")
        
        self._tools[tool_name] = tool_class
        self._categories[tool_class.category].append(tool_name)
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = tool_name
        
        logger.debug(f"Registered tool: {tool_name} ({tool_class.category.value})")
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool.
        
        Args:
            name: Name of tool to unregister
            
        Returns:
            True if tool was removed
        """
        if name not in self._tools:
            return False
        
        tool_class = self._tools.pop(name)
        self._categories[tool_class.category].remove(name)
        self._instances.pop(name, None)
        
        aliases_to_remove = [
            alias for alias, target in self._aliases.items()
            if target == name
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        logger.debug(f"Unregistered tool: {name}")
        return True
    
    def get(self, name: str) -> Optional[Type[ToolBase]]:
        """Get a tool class by name.
        
        Args:
            name: Tool name or alias
            
        Returns:
            Tool class or None
        """
        actual_name = self._aliases.get(name, name)
        return self._tools.get(actual_name)
    
    def get_instance(
        self,
        name: str,
        context: Optional[ToolContext] = None,
    ) -> Optional[ToolBase]:
        """Get or create a tool instance.
        
        Args:
            name: Tool name
            context: Optional execution context
            
        Returns:
            Tool instance or None
        """
        actual_name = self._aliases.get(name, name)
        
        if actual_name not in self._tools:
            return None
        
        tool_class = self._tools[actual_name]
        
        instance = tool_class(context=context)
        return instance
    
    def has(self, name: str) -> bool:
        """Check if a tool is registered.
        
        Args:
            name: Tool name or alias
            
        Returns:
            True if tool exists
        """
        actual_name = self._aliases.get(name, name)
        return actual_name in self._tools
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List all registered tools.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        if category:
            return self._categories[category].copy()
        return list(self._tools.keys())
    
    def get_schemas(self, category: Optional[ToolCategory] = None) -> List[Dict[str, Any]]:
        """Get schemas for LLM function calling.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool schemas
        """
        tools = self.list_tools(category)
        schemas = []
        
        for name in tools:
            tool_class = self._tools.get(name)
            if tool_class:
                instance = tool_class()
                schemas.append(instance.get_schema())
        
        return schemas
    
    def get_metadata(self, name: str) -> Optional[ToolMeta]:
        """Get metadata for a tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool metadata or None
        """
        tool_class = self.get(name)
        if tool_class:
            instance = tool_class()
            return instance.metadata
        return None
    
    def get_all_metadata(self) -> Dict[str, ToolMeta]:
        """Get metadata for all tools.
        
        Returns:
            Dictionary of tool name to metadata
        """
        result = {}
        for name, tool_class in self._tools.items():
            instance = tool_class()
            result[name] = instance.metadata
        return result
    
    async def execute(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """Execute a tool by name.
        
        Args:
            name: Tool name
            params: Tool parameters
            context: Execution context
            timeout: Optional timeout
            
        Returns:
            ToolResult from execution
        """
        instance = self.get_instance(name, context)
        
        if instance is None:
            return ToolResult.fail(
                error=f"Tool not found: {name}",
                code="TOOL_NOT_FOUND",
            )
        
        return await instance.run(params, context, timeout)
    
    def discover_tools(self, module_path: str) -> int:
        """Discover and register tools from a module.
        
        Looks for classes that:
        - Inherit from ToolBase
        - Have a non-empty name attribute
        
        Args:
            module_path: Python module path
            
        Returns:
            Number of tools registered
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            return 0
        
        count = 0
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            if (
                isinstance(attr, type) and
                issubclass(attr, ToolBase) and
                attr is not ToolBase and
                attr.name
            ):
                self.register(attr)
                count += 1
        
        logger.info(f"Discovered {count} tools from {module_path}")
        return count
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._instances.clear()
        self._categories = {cat: [] for cat in ToolCategory}
        self._aliases.clear()
        logger.info("All tools cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_tools": len(self._tools),
            "tools_by_category": {
                cat.value: len(tools)
                for cat, tools in self._categories.items()
            },
            "aliases": len(self._aliases),
        }
    
    def __contains__(self, name: str) -> bool:
        """Check if tool is registered using 'in' operator."""
        return self.has(name)
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)


_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance.
    
    Returns:
        Global ToolRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def register_tool(
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[ToolBase]], Type[ToolBase]]:
    """Decorator to register a tool class.
    
    Usage:
        @register_tool()
        class MyTool(ToolBase):
            name = "my_tool"
            ...
    
    Args:
        name: Optional name override
        aliases: Optional aliases
        
    Returns:
        Decorator function
    """
    def decorator(tool_class: Type[ToolBase]) -> Type[ToolBase]:
        get_tool_registry().register(tool_class, name, aliases)
        return tool_class
    
    return decorator
