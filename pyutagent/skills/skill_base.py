"""Skill Base Class and Metadata.

This module provides:
- SkillMetadata: Metadata for skills with version management
- SkillContext: Execution context for skills
- SkillInput/SkillOutput: Typed input/output definitions
- SkillBase: Abstract base class for all skills
- SkillResult: Result from skill execution

Design inspired by Claude Code Skills system:
- Progressive disclosure (三级系统)
- Doc-as-Power (文档即能力)
- Community shareable (社区可共享)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, ClassVar, Type, Union
import json

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
    UT_GENERATION = "ut_generation"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"


class SkillLevel(Enum):
    """Skill complexity levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class SkillParameter:
    """Definition of a skill parameter."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None
    example: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "required": self.required,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.enum:
            result["enum"] = self.enum
        if self.example:
            result["example"] = self.example
        return result


@dataclass
class SkillInput:
    """Input specification for a skill.

    Defines what parameters the skill accepts.
    """
    parameters: List[SkillParameter] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
        }

    def validate(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate input parameters.

        Args:
            inputs: Input parameters to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        for param in self.parameters:
            if param.required and param.name not in inputs:
                errors.append(f"Missing required parameter: {param.name}")
                continue

            value = inputs.get(param.name, param.default)
            if value is not None and param.enum:
                if value not in param.enum:
                    errors.append(
                        f"Invalid value for {param.name}: {value}. "
                        f"Must be one of: {param.enum}"
                    )
        return errors


@dataclass
class SkillOutput:
    """Output specification for a skill.

    Defines what the skill returns.
    """
    type: str = "object"
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "description": self.description,
            "properties": self.properties,
        }


@dataclass
class SkillExample:
    """Example of skill usage."""
    task: str
    description: str
    expected_result: str = ""
    code_example: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task": self.task,
            "description": self.description,
            "expected_result": self.expected_result,
            "code_example": self.code_example,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


@dataclass
class SkillVersion:
    """Version information for a skill."""
    major: int = 1
    minor: int = 0
    patch: int = 0

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> "SkillVersion":
        """Parse version from string."""
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
        )

    def is_compatible_with(self, other: "SkillVersion") -> bool:
        """Check if versions are compatible (same major version)."""
        return self.major == other.major


@dataclass
class SkillMetadata:
    """Metadata for a skill.

    Includes:
    - Name and description
    - Version information
    - Required tools
    - Usage instructions
    - Examples
    - Best practices
    - Author and tags
    """

    name: str
    description: str
    version: SkillVersion = field(default_factory=lambda: SkillVersion(1, 0, 0))
    required_tools: List[str] = field(default_factory=list)
    category: SkillCategory = SkillCategory.UTILITY
    level: SkillLevel = SkillLevel.INTERMEDIATE
    instructions: str = ""
    examples: List[SkillExample] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = ""
    homepage: str = ""
    input_spec: Optional[SkillInput] = None
    output_spec: Optional[SkillOutput] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": str(self.version),
            "required_tools": self.required_tools,
            "category": self.category.value,
            "level": self.level.value,
            "instructions": self.instructions,
            "examples": [e.to_dict() for e in self.examples],
            "best_practices": self.best_practices,
            "common_mistakes": self.common_mistakes,
            "prerequisites": self.prerequisites,
            "tags": self.tags,
            "author": self.author,
            "homepage": self.homepage,
            "input_spec": self.input_spec.to_dict() if self.input_spec else None,
            "output_spec": self.output_spec.to_dict() if self.output_spec else None,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_prompt_context(self, detail_level: str = "full") -> str:
        """Get context string for LLM prompts.

        Args:
            detail_level: Level of detail (brief, standard, full)

        Returns:
            Formatted context string
        """
        if detail_level == "brief":
            return self._get_brief_context()
        elif detail_level == "standard":
            return self._get_standard_context()
        else:
            return self._get_full_context()

    def _get_brief_context(self) -> str:
        """Get brief context (name and description only)."""
        return f"- {self.name}: {self.description}"

    def _get_standard_context(self) -> str:
        """Get standard context (brief + instructions)."""
        lines = [f"# Skill: {self.name} (v{self.version})"]
        lines.append(f"\n{self.description}")

        if self.tags:
            lines.append(f"\nTags: {', '.join(self.tags)}")

        if self.instructions:
            lines.append(f"\n## Instructions\n{self.instructions}")

        return "\n".join(lines)

    def _get_full_context(self) -> str:
        """Get full context (all metadata)."""
        lines = [f"# Skill: {self.name} (v{self.version})"]
        lines.append(f"\n{self.description}")

        if self.tags:
            lines.append(f"\nTags: {', '.join(self.tags)}")

        if self.author:
            lines.append(f"Author: {self.author}")

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

        if self.input_spec and self.input_spec.parameters:
            lines.append("\n## Input Parameters")
            for param in self.input_spec.parameters:
                req = "(required)" if param.required else "(optional)"
                lines.append(f"- `{param.name}` {req}: {param.description}")
                if param.example:
                    lines.append(f"  Example: `{param.example}`")

        if self.examples:
            lines.append("\n## Examples")
            for example in self.examples:
                lines.append(f"\n### {example.task}")
                lines.append(example.description)
                if example.inputs:
                    lines.append(f"\nInputs: {json.dumps(example.inputs, indent=2)}")
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
class SkillContext:
    """Execution context for skills.

    Provides:
    - Project information
    - Working directory
    - Configuration
    - Memory/knowledge access
    - Tool access
    """

    project_path: Optional[Path] = None
    working_dir: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    environment_vars: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_path": str(self.project_path) if self.project_path else None,
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "config": self.config,
            "memory": self.memory,
            "session_id": self.session_id,
            "user_preferences": self.user_preferences,
            "environment_vars": self.environment_vars,
        }

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        return self.tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is available."""
        return name in self.tools


@dataclass
class SkillResult:
    """Result from skill execution."""

    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        message: str = "",
        data: Dict[str, Any] = None,
        artifacts: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> "SkillResult":
        """Create a successful result."""
        return cls(
            success=True,
            message=message,
            data=data or {},
            artifacts=artifacts or [],
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        message: str,
        data: Dict[str, Any] = None,
        error_code: str = "",
    ) -> "SkillResult":
        """Create a failed result."""
        return cls(
            success=False,
            message=message,
            data=data or {},
            metadata={"error_code": error_code} if error_code else {},
        )

    @classmethod
    def partial(
        cls,
        message: str,
        data: Dict[str, Any],
        completed_steps: int,
        total_steps: int,
    ) -> "SkillResult":
        """Create a partial success result."""
        return cls(
            success=True,
            message=message,
            data=data,
            metadata={
                "partial": True,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "artifacts": self.artifacts,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    def is_partial(self) -> bool:
        """Check if this is a partial result."""
        return self.metadata.get("partial", False)


class SkillBase(ABC):
    """Abstract base class for all skills.

    Skills are high-level capabilities that combine multiple tools
    to accomplish complex tasks. They provide:
    - Usage instructions for LLMs
    - Best practices and examples
    - Orchestration of multiple tools
    - Error handling and recovery

    Design inspired by Claude Code Skills:
    - Progressive disclosure (三级系统)
    - Doc-as-Power (文档即能力)
    - Community shareable
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[SkillCategory] = SkillCategory.UTILITY
    level: ClassVar[SkillLevel] = SkillLevel.INTERMEDIATE
    required_tools: ClassVar[List[str]] = []
    version: ClassVar[str] = "1.0.0"
    author: ClassVar[str] = ""
    homepage: ClassVar[str] = ""
    tags: ClassVar[List[str]] = []

    _registry: ClassVar[Optional["SkillRegistry"]] = None

    def __init__(self):
        """Initialize skill."""
        self._logger = logging.getLogger(f"skill.{self.name}")
        self._metadata: Optional[SkillMetadata] = None

    @property
    def metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        if self._metadata is None:
            self._metadata = SkillMetadata(
                name=self.name,
                description=self.description,
                version=SkillVersion.from_string(self.version),
                required_tools=self.required_tools,
                category=self.category,
                level=self.level,
                instructions=self.get_instructions(),
                examples=self.get_examples(),
                best_practices=self.get_best_practices(),
                common_mistakes=self.get_common_mistakes(),
                prerequisites=self.get_prerequisites(),
                tags=self.tags,
                author=self.author,
                homepage=self.homepage,
                input_spec=self.get_input_spec(),
                output_spec=self.get_output_spec(),
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

    def get_input_spec(self) -> Optional[SkillInput]:
        """Get input specification.

        Override this method to define input parameters.

        Returns:
            SkillInput specification or None
        """
        return None

    def get_output_spec(self) -> Optional[SkillOutput]:
        """Get output specification.

        Override this method to define output format.

        Returns:
            SkillOutput specification or None
        """
        return None

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate input parameters.

        Args:
            inputs: Input parameters

        Returns:
            List of validation errors (empty if valid)
        """
        input_spec = self.get_input_spec()
        if input_spec:
            return input_spec.validate(inputs)
        return []

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: SkillContext,
        inputs: Dict[str, Any],
    ) -> SkillResult:
        """Execute the skill.

        Args:
            task: Task description
            context: Execution context
            inputs: Input parameters

        Returns:
            SkillResult with execution result
        """
        pass

    async def run(
        self,
        task: str,
        context: SkillContext,
        inputs: Dict[str, Any] = None,
    ) -> SkillResult:
        """Run the skill with timing and validation.

        Args:
            task: Task description
            context: Execution context
            inputs: Input parameters

        Returns:
            SkillResult with execution result
        """
        inputs = inputs or {}
        start_time = datetime.now()

        # Validate inputs
        validation_errors = self.validate_inputs(inputs)
        if validation_errors:
            return SkillResult.fail(
                message=f"Input validation failed: {'; '.join(validation_errors)}",
                error_code="VALIDATION_ERROR",
            )

        try:
            result = await self.execute(task, context, inputs)
            result.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return result

        except Exception as e:
            self._logger.exception(f"Skill {self.name} failed: {e}")
            return SkillResult.fail(
                message=f"Skill execution failed: {e}",
                data={"error": str(e), "exception_type": type(e).__name__},
                error_code="EXECUTION_ERROR",
            )

    def get_prompt_context(self, detail_level: str = "full") -> str:
        """Get context string for LLM prompts."""
        return self.metadata.get_prompt_context(detail_level)

    def export_to_file(self, path: Union[str, Path]) -> bool:
        """Export skill metadata to a file.

        Args:
            path: Path to export to

        Returns:
            True if successful
        """
        try:
            path = Path(path)
            path.write_text(self.metadata.to_json(), encoding="utf-8")
            return True
        except Exception as e:
            self._logger.error(f"Failed to export skill: {e}")
            return False

    def __repr__(self) -> str:
        """String representation."""
        return f"Skill({self.name}, v{self.version}, category={self.category.value})"


def skill(
    name: str,
    description: str,
    category: SkillCategory = SkillCategory.UTILITY,
    required_tools: Optional[List[str]] = None,
    version: str = "1.0.0",
):
    """Decorator to create a simple skill from a function.

    Usage:
        @skill("echo", "Echo a message")
        async def echo_skill(task, context, inputs):
            return SkillResult.ok(message=inputs.get("message", ""))

    Args:
        name: Skill name
        description: Skill description
        category: Skill category
        required_tools: Required tools
        version: Skill version

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
        _version = version

        class FunctionSkill(SkillBase):
            name = _name
            description = _description
            category = _category
            required_tools = _required_tools
            version = _version

            async def execute(self, task, context, inputs):
                if asyncio.iscoroutinefunction(func):
                    return await func(task, context, inputs)
                return func(task, context, inputs)

        FunctionSkill.__name__ = f"{_name.title()}Skill"
        return FunctionSkill

    return decorator
