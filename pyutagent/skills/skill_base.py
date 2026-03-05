"""Skill Base Class and Metadata.

This module provides:
- SkillMeta: Metadata for skills
- SkillBase: Abstract base class for all skills
- SkillResult: Result from skill execution
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, ClassVar, Type

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Categories for skills."""
    BUILD = "build"
    TEST = "test"
    CODE = "code"
    GIT = "git"
    SEARCH = "search"
    ANALYSIS = "analysis"
    UTILITY = "utility"


@dataclass
class SkillExample:
    """Example of skill usage."""
    task: str
    description: str
    expected_result: str = ""
    code_example: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "description": self.description,
            "expected_result": self.expected_result,
            "code_example": self.code_example,
        }


@dataclass
class SkillMeta:
    """Metadata for a skill.
    
    Includes:
    - Name and description
    - Required tools
    - Usage instructions
    - Examples
    - Best practices
    """
    
    name: str
    description: str
    required_tools: List[str] = field(default_factory=list)
    category: SkillCategory = SkillCategory.UTILITY
    instructions: str = ""
    examples: List[SkillExample] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "required_tools": self.required_tools,
            "category": self.category.value,
            "instructions": self.instructions,
            "examples": [e.to_dict() for e in self.examples],
            "best_practices": self.best_practices,
            "common_mistakes": self.common_mistakes,
            "prerequisites": self.prerequisites,
            "version": self.version,
        }
    
    def get_prompt_context(self) -> str:
        """Get context string for LLM prompts.
        
        Returns:
            Formatted context string
        """
        lines = [f"# Skill: {self.name}"]
        lines.append(f"\n{self.description}")
        
        if self.prerequisites:
            lines.append("\n## Prerequisites")
            for prereq in self.prerequisites:
                lines.append(f"- {prereq}")
        
        if self.required_tools:
            lines.append("\n## Required Tools")
            for tool in self.required_tools:
                lines.append(f"- {tool}")
        
        if self.instructions:
            lines.append(f"\n## Instructions\n{self.instructions}")
        
        if self.examples:
            lines.append("\n## Examples")
            for example in self.examples:
                lines.append(f"\n### {example.task}")
                lines.append(example.description)
                if example.code_example:
                    lines.append(f"\n```\n{example.code_example}\n```")
        
        if self.best_practices:
            lines.append("\n## Best Practices")
            for practice in self.best_practices:
                lines.append(f"- {practice}")
        
        if self.common_mistakes:
            lines.append("\n## Common Mistakes to Avoid")
            for mistake in self.common_mistakes:
                lines.append(f"- {mistake}")
        
        return "\n".join(lines)


@dataclass
class SkillResult:
    """Result from skill execution."""
    
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    duration_ms: int = 0
    
    @classmethod
    def ok(cls, message: str = "", data: Dict[str, Any] = None, artifacts: List[str] = None) -> "SkillResult":
        """Create a successful result."""
        return cls(
            success=True,
            message=message,
            data=data or {},
            artifacts=artifacts or [],
        )
    
    @classmethod
    def fail(cls, message: str, data: Dict[str, Any] = None) -> "SkillResult":
        """Create a failed result."""
        return cls(
            success=False,
            message=message,
            data=data or {},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "artifacts": self.artifacts,
            "duration_ms": self.duration_ms,
        }


class SkillBase(ABC):
    """Abstract base class for all skills.
    
    Skills are high-level capabilities that combine multiple tools
    to accomplish complex tasks. They provide:
    - Usage instructions for LLMs
    - Best practices and examples
    - Orchestration of multiple tools
    - Error handling and recovery
    """
    
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[SkillCategory] = SkillCategory.UTILITY
    required_tools: ClassVar[List[str]] = []
    
    _registry: ClassVar[Optional["SkillRegistry"]] = None
    
    def __init__(self):
        """Initialize skill."""
        self._logger = logging.getLogger(f"skill.{self.name}")
        self._metadata: Optional[SkillMeta] = None
    
    @property
    def metadata(self) -> SkillMeta:
        """Get skill metadata."""
        if self._metadata is None:
            self._metadata = SkillMeta(
                name=self.name,
                description=self.description,
                required_tools=self.required_tools,
                category=self.category,
                instructions=self.get_instructions(),
                examples=self.get_examples(),
                best_practices=self.get_best_practices(),
                common_mistakes=self.get_common_mistakes(),
                prerequisites=self.get_prerequisites(),
            )
        return self._metadata
    
    def get_instructions(self) -> str:
        """Get usage instructions.
        
        Override this method to provide detailed instructions.
        
        Returns:
            Instructions string
        """
        return ""
    
    def get_examples(self) -> List[SkillExample]:
        """Get usage examples.
        
        Override this method to provide examples.
        
        Returns:
            List of examples
        """
        return []
    
    def get_best_practices(self) -> List[str]:
        """Get best practices.
        
        Override this method to provide best practices.
        
        Returns:
            List of best practices
        """
        return []
    
    def get_common_mistakes(self) -> List[str]:
        """Get common mistakes to avoid.
        
        Override this method to provide common mistakes.
        
        Returns:
            List of common mistakes
        """
        return []
    
    def get_prerequisites(self) -> List[str]:
        """Get prerequisites.
        
        Override this method to provide prerequisites.
        
        Returns:
            List of prerequisites
        """
        return []
    
    @abstractmethod
    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Execute the skill.
        
        Args:
            task: Task description
            context: Execution context
            tools: Available tools
            
        Returns:
            SkillResult with execution result
        """
        pass
    
    async def run(
        self,
        task: str,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Run the skill with timing.
        
        Args:
            task: Task description
            context: Execution context
            tools: Available tools
            
        Returns:
            SkillResult with execution result
        """
        start_time = datetime.now()
        
        try:
            result = await self.execute(task, context, tools)
            result.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return result
            
        except Exception as e:
            self._logger.exception(f"Skill {self.name} failed: {e}")
            return SkillResult.fail(
                message=f"Skill execution failed: {e}",
                data={"error": str(e), "exception_type": type(e).__name__},
            )
    
    def get_prompt_context(self) -> str:
        """Get context string for LLM prompts."""
        return self.metadata.get_prompt_context()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Skill({self.name}, category={self.category.value})"


def skill(
    name: str,
    description: str,
    category: SkillCategory = SkillCategory.UTILITY,
    required_tools: Optional[List[str]] = None,
):
    """Decorator to create a simple skill from a function.
    
    Usage:
        @skill("echo", "Echo a message")
        async def echo_skill(task, context, tools):
            return SkillResult.ok(message=task)
    
    Args:
        name: Skill name
        description: Skill description
        category: Skill category
        required_tools: Required tools
        
    Returns:
        Decorated function as a Skill
    """
    import asyncio
    
    def decorator(func: Callable) -> type:
        """Create skill class from function."""
        _name = name
        _description = description
        _category = category
        _required_tools = required_tools or []
        
        class FunctionSkill(SkillBase):
            name = _name
            description = _description
            category = _category
            required_tools = _required_tools
            
            async def execute(self, task, context, tools):
                if asyncio.iscoroutinefunction(func):
                    return await func(task, context, tools)
                return func(task, context, tools)
        
        FunctionSkill.__name__ = f"{_name.title()}Skill"
        return FunctionSkill
    
    return decorator
