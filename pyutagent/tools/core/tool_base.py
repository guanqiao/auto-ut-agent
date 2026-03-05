"""Tool Base Class and Metadata.

This module provides:
- ToolMeta: Metadata for tools
- ToolBase: Abstract base class for all tools
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Type, ClassVar, TYPE_CHECKING
import json

from .tool_result import ToolResult
from .tool_context import ToolContext

if TYPE_CHECKING:
    from ..external.mcp_adapter import MCPAdapter
    from ..external.aider_adapter import AiderAdapter

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for tools."""
    FILE = "file"
    BUILD = "build"
    TEST = "test"
    CODE = "code"
    GIT = "git"
    SEARCH = "search"
    EXTERNAL = "external"
    UTILITY = "utility"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        
        if self.enum:
            schema["enum"] = self.enum
        
        return schema


@dataclass
class ToolMeta:
    """Metadata for a tool.
    
    Includes:
    - Name and description
    - Parameters schema
    - Category
    - Tags
    - Examples
    """
    
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    category: ToolCategory = ToolCategory.UTILITY
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""
    requires_confirmation: bool = False
    dangerous: bool = False
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON Schema for parameters.
        
        Returns:
            JSON Schema dictionary
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "enum": p.enum,
                }
                for p in self.parameters
            ],
            "category": self.category.value,
            "tags": self.tags,
            "examples": self.examples,
            "version": self.version,
            "author": self.author,
            "requires_confirmation": self.requires_confirmation,
            "dangerous": self.dangerous,
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Get OpenAI function calling schema.
        
        Returns:
            Schema for LLM function calling
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameter_schema(),
        }


class ToolBase(ABC):
    """Abstract base class for all tools.
    
    All tools must:
    - Define metadata (name, description, parameters)
    - Implement execute() method
    - Optionally implement validate_parameters()
    
    Features:
    - Automatic parameter validation
    - Timeout support
    - Retry support
    - Hooks integration
    - Logging
    """
    
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[ToolCategory] = ToolCategory.UTILITY
    parameters: ClassVar[List[ToolParameter]] = []
    tags: ClassVar[List[str]] = []
    
    _registry: ClassVar[Optional["ToolRegistry"]] = None
    
    def __init__(self, context: Optional[ToolContext] = None):
        """Initialize tool.
        
        Args:
            context: Optional execution context
        """
        self._context = context
        self._logger = logging.getLogger(f"tool.{self.name}")
        self._metadata: Optional[ToolMeta] = None
    
    @property
    def context(self) -> Optional[ToolContext]:
        """Get execution context."""
        return self._context
    
    @context.setter
    def context(self, value: ToolContext) -> None:
        """Set execution context."""
        self._context = value
    
    @property
    def metadata(self) -> ToolMeta:
        """Get tool metadata."""
        if self._metadata is None:
            self._metadata = ToolMeta(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
                category=self.category,
                tags=self.tags,
            )
        return self._metadata
    
    def get_schema(self) -> Dict[str, Any]:
        """Get OpenAI function calling schema."""
        return self.metadata.get_schema()
    
    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Validate parameters against schema.
        
        Args:
            params: Parameters to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Missing required parameter: {param.name}")
                continue
            
            value = params.get(param.name, param.default)
            
            if value is not None and param.enum:
                if value not in param.enum:
                    errors.append(
                        f"Invalid value for {param.name}: {value}. "
                        f"Must be one of: {param.enum}"
                    )
        
        return errors
    
    @abstractmethod
    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute the tool.
        
        Args:
            params: Tool parameters
            context: Execution context (uses instance context if None)
            
        Returns:
            ToolResult with execution result
        """
        pass
    
    async def run(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """Run the tool with validation and timeout.
        
        Args:
            params: Tool parameters
            context: Execution context
            timeout: Optional timeout override
            
        Returns:
            ToolResult with execution result
        """
        ctx = context or self._context
        
        validation_errors = self.validate_parameters(params)
        if validation_errors:
            return ToolResult.fail(
                error="; ".join(validation_errors),
                code="VALIDATION_ERROR",
            )
        
        actual_timeout = timeout or (ctx.timeout if ctx else 60.0)
        
        start_time = datetime.now()
        
        try:
            if actual_timeout:
                result = await asyncio.wait_for(
                    self.execute(params, ctx),
                    timeout=actual_timeout,
                )
            else:
                result = await self.execute(params, ctx)
            
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result.duration_ms = duration_ms
            
            return result
            
        except asyncio.TimeoutError:
            return ToolResult.timeout(actual_timeout)
            
        except asyncio.CancelledError:
            return ToolResult.cancel("Execution was cancelled")
            
        except Exception as e:
            self._logger.exception(f"Tool {self.name} failed: {e}")
            return ToolResult.fail(
                error=str(e),
                code="EXECUTION_ERROR",
                details={"exception_type": type(e).__name__},
            )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Tool({self.name}, category={self.category.value})"


def tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.UTILITY,
    parameters: Optional[List[ToolParameter]] = None,
    tags: Optional[List[str]] = None,
):
    """Decorator to create a simple tool from a function.
    
    Usage:
        @tool("echo", "Echo a message", parameters=[
            ToolParameter("message", "string", "Message to echo")
        ])
        async def echo(params, context):
            return ToolResult.ok(output=params["message"])
    
    Args:
        name: Tool name
        description: Tool description
        category: Tool category
        parameters: Tool parameters
        tags: Tool tags
        
    Returns:
        Decorated function as a Tool
    """
    def decorator(func: Callable) -> type:
        """Create tool class from function."""
        _params = parameters or []
        _tags = tags or []
        _name = name
        _description = description
        _category = category
        
        class FunctionTool(ToolBase):
            name = _name
            description = _description
            category = _category
            parameters = _params
            tags = _tags
            
            async def execute(self, params, context=None):
                if asyncio.iscoroutinefunction(func):
                    return await func(params, context)
                return func(params, context)
        
        FunctionTool.__name__ = f"{_name.title()}Tool"
        return FunctionTool
    
    return decorator
