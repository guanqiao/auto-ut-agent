"""Tool system for LLM-driven agent.

This module provides:
- Base Tool class with structured definitions
- Tool Registry for managing available tools
- OpenAI Function Calling compatible schema generation
- Standard tools: Read, Write, Edit, Glob, Grep, Bash
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Type
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools."""
    FILE = auto()
    SEARCH = auto()
    COMMAND = auto()
    ANALYSIS = auto()
    PROJECT = auto()
    TEST = auto()
    CUSTOM = auto()


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """Structured definition of a tool for LLM."""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI Function Calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            param_dict = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                param_dict["enum"] = param.enum
            if param.default is not None:
                param_dict["default"] = param.default

            properties[param.name] = param_dict

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.name,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum
                }
                for p in self.parameters
            ],
            "examples": self.examples,
            "tags": self.tags
        }


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata
        }


class Tool(ABC):
    """Base class for all tools.

    Each tool must implement:
    - definition: ToolDefinition with name, description, parameters
    - execute: Async method that performs the actual operation
    """

    def __init__(self):
        self._cached_schema: Optional[Dict[str, Any]] = None

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Get tool definition."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with execution results
        """
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get cached OpenAI schema."""
        if self._cached_schema is None:
            self._cached_schema = self.definition.to_openai_schema()
        return self._cached_schema

    def get_name(self) -> str:
        """Get tool name."""
        return self.definition.name

    def get_description(self) -> str:
        """Get tool description."""
        return self.definition.description

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters against definition.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        for param_def in self.definition.parameters:
            if param_def.required and param_def.name not in params:
                return False, f"Missing required parameter: {param_def.name}"

            if param_def.name in params:
                value = params[param_def.name]
                expected_type = param_def.type

                if expected_type == "string" and not isinstance(value, str):
                    return False, f"Parameter {param_def.name} must be string"
                if expected_type == "integer" and not isinstance(value, int):
                    return False, f"Parameter {param_def.name} must be integer"
                if expected_type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter {param_def.name} must be number"
                if expected_type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter {param_def.name} must be boolean"
                if expected_type == "array" and not isinstance(value, list):
                    return False, f"Parameter {param_def.name} must be array"
                if expected_type == "object" and not isinstance(value, dict):
                    return False, f"Parameter {param_def.name} must be object"

                if param_def.enum and value not in param_def.enum:
                    return False, f"Parameter {param_def.name} must be one of: {param_def.enum}"

        return True, None


class ToolExecutor:
    """Executes tools with error handling and logging."""

    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []

    async def execute_tool(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute a tool with error handling.

        Args:
            tool: Tool to execute
            params: Parameters for tool
            context: Optional execution context

        Returns:
            ToolResult
        """
        tool_name = tool.get_name()
        logger.info(f"[ToolExecutor] Executing tool: {tool_name}")
        logger.debug(f"[ToolExecutor] Parameters: {json.dumps(params, indent=2)}")

        is_valid, error_msg = tool.validate_params(params)
        if not is_valid:
            logger.error(f"[ToolExecutor] Parameter validation failed: {error_msg}")
            return ToolResult(
                success=False,
                error=f"Parameter validation failed: {error_msg}"
            )

        try:
            result = await tool.execute(**params)

            self.execution_history.append({
                "tool": tool_name,
                "params": params,
                "context": context or {},
                "success": result.success,
                "output_type": type(result.output).__name__ if result.output else None,
                "error": result.error
            })

            logger.info(f"[ToolExecutor] Tool {tool_name} executed - Success: {result.success}")
            return result

        except Exception as e:
            logger.exception(f"[ToolExecutor] Tool execution failed: {tool_name} - {e}")
            self.execution_history.append({
                "tool": tool_name,
                "params": params,
                "context": context or {},
                "success": False,
                "error": str(e)
            })
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )

    def get_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history.copy()

    def clear_history(self):
        """Clear execution history."""
        self.execution_history.clear()


def create_tool_parameter(
    name: str,
    param_type: str,
    description: str,
    required: bool = True,
    default: Any = None,
    enum: Optional[List[str]] = None
) -> ToolParameter:
    """Helper to create tool parameters.

    Args:
        name: Parameter name
        param_type: Type (string, integer, number, boolean, array, object)
        description: Parameter description
        required: Whether parameter is required
        default: Default value
        enum: Optional list of allowed values

    Returns:
        ToolParameter
    """
    return ToolParameter(
        name=name,
        type=param_type,
        description=description,
        required=required,
        default=default,
        enum=enum
    )
