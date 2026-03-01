"""Tool Registry for managing available tools.

This module provides:
- Registry for registering and managing tools
- Category-based organization
- Schema generation for LLM function calling
- Tool lookup and discovery
"""

import logging
from typing import Any, Dict, List, Optional, Set, Type
from dataclasses import dataclass, field

from .tool import Tool, ToolDefinition, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class RegistryConfig:
    """Configuration for tool registry."""
    enable_caching: bool = True
    max_cache_size: int = 100
    enable_auto_discovery: bool = False
    strict_validation: bool = True


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found."""
    pass


class ToolValidationError(Exception):
    """Raised when tool validation fails."""
    pass


class ToolRegistry:
    """Registry for managing tools.

    Features:
    - Register/unregister tools
    - Category-based organization
    - Schema generation for LLM
    - Tool discovery and lookup
    - Dependency injection support
    """

    def __init__(self, config: Optional[RegistryConfig] = None):
        """Initialize tool registry.

        Args:
            config: Optional registry configuration
        """
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
        self._tags: Dict[str, List[str]] = {}
        self._aliases: Dict[str, str] = {}
        self._config = config or RegistryConfig()
        self._schemas_cache: Optional[List[Dict[str, Any]]] = None

        logger.debug("[ToolRegistry] Registry initialized")

    def register(
        self,
        tool: Tool,
        alias: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
            alias: Optional alias for the tool
            tags: Optional tags for organization
        """
        tool_name = tool.definition.name

        if tool_name in self._tools:
            logger.warning(f"[ToolRegistry] Tool already registered: {tool_name}, replacing")

        self._tools[tool_name] = tool

        category = tool.definition.category
        if tool_name not in self._categories[category]:
            self._categories[category].append(tool_name)

        if tags:
            for tag in tags:
                if tag not in self._tags:
                    self._tags[tag] = []
                if tool_name not in self._tags[tag]:
                    self._tags[tag].append(tool_name)

        if alias:
            self._aliases[alias] = tool_name

        self._invalidate_cache()
        logger.info(f"[ToolRegistry] Registered tool: {tool_name}, category: {category.name}")

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool.

        Args:
            tool_name: Name of tool to unregister

        Returns:
            True if tool was unregistered
        """
        if tool_name not in self._tools:
            logger.warning(f"[ToolRegistry] Tool not found for unregister: {tool_name}")
            return False

        tool = self._tools.pop(tool_name)

        category = tool.definition.category
        if tool_name in self._categories[category]:
            self._categories[category].remove(tool_name)

        for tag_tools in self._tags.values():
            if tool_name in tag_tools:
                tag_tools.remove(tool_name)

        if tool_name in self._aliases:
            del self._aliases[tool_name]

        self._invalidate_cache()
        logger.info(f"[ToolRegistry] Unregistered tool: {tool_name}")
        return True

    def get(self, tool_name: str) -> Tool:
        """Get a tool by name.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool instance

        Raises:
            ToolNotFoundError: If tool not found
        """
        if tool_name in self._aliases:
            tool_name = self._aliases[tool_name]

        if tool_name not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {tool_name}")

        return self._tools[tool_name]

    def get_or_none(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name, returning None if not found.

        Args:
            tool_name: Name of tool to retrieve

        Returns:
            Tool instance or None
        """
        try:
            return self.get(tool_name)
        except ToolNotFoundError:
            return None

    def has(self, tool_name: str) -> bool:
        """Check if a tool is registered.

        Args:
            tool_name: Name of tool to check

        Returns:
            True if tool is registered
        """
        return tool_name in self._tools or tool_name in self._aliases

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List all registered tool names.

        Args:
            category: Optional category filter

        Returns:
            List of tool names
        """
        if category:
            return self._categories.get(category, []).copy()
        return list(self._tools.keys())

    def list_by_tag(self, tag: str) -> List[str]:
        """List tools by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of tool names with the tag
        """
        return self._tags.get(tag, []).copy()

    def get_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category.

        Args:
            category: Category to filter by

        Returns:
            List of Tool instances
        """
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions.

        Returns:
            List of ToolDefinition
        """
        return [tool.definition for tool in self._tools.values()]

    def get_schemas(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get OpenAI Function Calling schemas for all tools.

        Args:
            force_refresh: Force refresh cached schemas

        Returns:
            List of tool schemas in OpenAI format
        """
        if self._schemas_cache and not force_refresh:
            return self._schemas_cache

        schemas = []
        for tool in self._tools.values():
            schemas.append(tool.get_schema())

        if self._config.enable_caching:
            self._schemas_cache = schemas

        logger.debug(f"[ToolRegistry] Generated {len(schemas)} tool schemas")
        return schemas

    def get_schema_json(self, force_refresh: bool = False) -> str:
        """Get schemas as JSON string.

        Args:
            force_refresh: Force refresh cached schemas

        Returns:
            JSON string of schemas
        """
        import json
        return json.dumps(self.get_schemas(force_refresh), indent=2)

    def search(self, query: str) -> List[str]:
        """Search tools by name or description.

        Args:
            query: Search query

        Returns:
            List of matching tool names
        """
        query_lower = query.lower()
        results = []

        for name, tool in self._tools.items():
            if query_lower in name.lower():
                results.append(name)
            elif query_lower in tool.definition.description.lower():
                results.append(name)

        return results

    def validate(self) -> Dict[str, List[str]]:
        """Validate all registered tools.

        Returns:
            Dictionary with 'valid' and 'invalid' tool lists
        """
        valid = []
        invalid = []

        for name, tool in self._tools.items():
            try:
                defn = tool.definition
                if not defn.name or not defn.description:
                    invalid.append(name)
                else:
                    valid.append(name)
            except Exception as e:
                logger.error(f"[ToolRegistry] Validation failed for {name}: {e}")
                invalid.append(name)

        return {"valid": valid, "invalid": invalid}

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry stats
        """
        return {
            "total_tools": len(self._tools),
            "by_category": {
                cat.name: len(tools)
                for cat, tools in self._categories.items()
            },
            "total_tags": len(self._tags),
            "total_aliases": len(self._aliases),
            "cache_enabled": self._config.enable_caching,
            "cache_populated": self._schemas_cache is not None
        }

    def clear(self):
        """Clear all registered tools."""
        self._tools.clear()
        for category in self._categories:
            self._categories[category].clear()
        self._tags.clear()
        self._aliases.clear()
        self._invalidate_cache()
        logger.info("[ToolRegistry] Registry cleared")

    def _invalidate_cache(self):
        """Invalidate cached schemas."""
        self._schemas_cache = None

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return self.has(tool_name)

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, categories={len(self._categories)})"


_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get global tool registry instance.

    Returns:
        Global ToolRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        logger.debug("[ToolRegistry] Created global registry instance")
    return _global_registry


def set_registry(registry: ToolRegistry):
    """Set global tool registry.

    Args:
        registry: Registry to use as global instance
    """
    global _global_registry
    _global_registry = registry
    logger.debug("[ToolRegistry] Set global registry instance")


def register_tool(tool: Tool, tags: Optional[List[str]] = None):
    """Register a tool in the global registry.

    Args:
        tool: Tool to register
        tags: Optional tags
    """
    get_registry().register(tool, tags=tags)


def get_tool(tool_name: str) -> Tool:
    """Get a tool from the global registry.

    Args:
        tool_name: Name of tool

    Returns:
        Tool instance
    """
    return get_registry().get(tool_name)


def list_tools(category: Optional[ToolCategory] = None) -> List[str]:
    """List tools in global registry.

    Args:
        category: Optional category filter

    Returns:
        List of tool names
    """
    return get_registry().list_tools(category)
