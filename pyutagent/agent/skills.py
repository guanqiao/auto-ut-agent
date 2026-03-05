"""Skills framework for reusable skill libraries.

This module provides:
- Skill: Reusable skill definition
- SkillRegistry: Management of available skills
- SkillLoader: Load skills from files/directories
- Built-in skills for common scenarios
"""

import asyncio
import importlib
import importlib.util
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
from enum import Enum, auto

logger = logging.getLogger(__name__)


class SkillCategory(Enum):
    """Categories of skills."""
    CODE_GENERATION = auto()
    CODE_REVIEW = auto()
    DEBUGGING = auto()
    REFACTORING = auto()
    TESTING = auto()
    DOCUMENTATION = auto()
    DEPLOYMENT = auto()
    CODE_EXPLANATION = auto()
    CUSTOM = auto()


@dataclass
class SkillMetadata:
    """Metadata for a skill."""
    name: str
    description: str
    category: SkillCategory
    version: str = "1.0.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    
    triggers: List[str] = field(default_factory=list)
    tool_usage_guide: str = ""
    best_practices: List[str] = field(default_factory=list)
    error_handling: List[str] = field(default_factory=list)
    requires_tools: List[str] = field(default_factory=list)
    estimated_duration: Optional[str] = None


@dataclass
class SkillStep:
    """A single step in skill execution."""
    step_id: str
    description: str
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation: Optional[str] = None
    rollback: Optional[str] = None


@dataclass
class SkillExample:
    """Example of skill usage."""
    input_params: Dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    description: str = ""


@dataclass
class SkillInput:
    """Input for skill execution."""
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    files: List[str] = field(default_factory=list)


@dataclass
class SkillOutput:
    """Output from skill execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """Base class for skills.

    A skill encapsulates a reusable capability that can be invoked
    with specific parameters and context.
    """

    def __init__(self):
        self._metadata = self._create_metadata()
        self._handlers: Dict[str, Callable] = {}

    @abstractmethod
    def _create_metadata(self) -> SkillMetadata:
        """Create skill metadata."""
        pass

    @abstractmethod
    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute the skill.

        Args:
            input_data: Skill input

        Returns:
            Skill output
        """
        pass

    @property
    def metadata(self) -> SkillMetadata:
        """Get skill metadata."""
        return self._metadata

    @property
    def name(self) -> str:
        """Get skill name."""
        return self._metadata.name

    def register_handler(self, event: str, handler: Callable):
        """Register an event handler.

        Args:
            event: Event name
            handler: Handler function
        """
        self._handlers[event] = handler
        logger.debug(f"[Skill] Registered handler for event: {event}")

    async def trigger(self, event: str, **kwargs):
        """Trigger an event handler.

        Args:
            event: Event name
            **kwargs: Handler arguments
        """
        handler = self._handlers.get(event)
        if handler:
            if asyncio.iscoroutinefunction(handler):
                await handler(**kwargs)
            else:
                handler(**kwargs)


class SkillRegistry:
    """Registry for managing skills."""

    def __init__(self):
        """Initialize skill registry."""
        self._skills: Dict[str, Skill] = {}
        self._categories: Dict[SkillCategory, List[str]] = {
            cat: [] for cat in SkillCategory
        }
        logger.debug("[SkillRegistry] Initialized")

    def register(self, skill: Skill):
        """Register a skill.

        Args:
            skill: Skill to register
        """
        name = skill.name
        self._skills[name] = skill

        category = skill.metadata.category
        if name not in self._categories[category]:
            self._categories[category].append(name)

        logger.info(f"[SkillRegistry] Registered skill: {name} ({category.name})")

    def unregister(self, name: str) -> bool:
        """Unregister a skill.

        Args:
            name: Skill name

        Returns:
            True if skill was unregistered
        """
        if name not in self._skills:
            return False

        skill = self._skills.pop(name)
        category = skill.metadata.category
        if name in self._categories[category]:
            self._categories[category].remove(name)

        logger.info(f"[SkillRegistry] Unregistered skill: {name}")
        return True

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name.

        Args:
            name: Skill name

        Returns:
            Skill or None
        """
        return self._skills.get(name)

    def list_skills(self, category: Optional[SkillCategory] = None) -> List[str]:
        """List all skill names.

        Args:
            category: Optional category filter

        Returns:
            List of skill names
        """
        if category:
            return self._categories.get(category, []).copy()
        return list(self._skills.keys())

    def list_by_tag(self, tag: str) -> List[str]:
        """List skills by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of skill names
        """
        results = []
        for name, skill in self._skills.items():
            if tag in skill.metadata.tags:
                results.append(name)
        return results

    def search(self, query: str) -> List[str]:
        """Search skills by name or description.

        Args:
            query: Search query

        Returns:
            List of matching skill names
        """
        query_lower = query.lower()
        results = []

        for name, skill in self._skills.items():
            if query_lower in name.lower():
                results.append(name)
            elif query_lower in skill.metadata.description.lower():
                results.append(name)

        return results

    def find_by_trigger(self, query: str) -> List[str]:
        """Find skills by trigger keywords.

        Args:
            query: Query to match against triggers

        Returns:
            List of matching skill names
        """
        query_lower = query.lower()
        results = []

        for name, skill in self._skills.items():
            triggers = skill.metadata.triggers
            if not triggers:
                continue
            for trigger in triggers:
                if query_lower in trigger.lower():
                    results.append(name)
                    break

        return results

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed skill information.

        Args:
            name: Skill name

        Returns:
            Dictionary with skill information or None
        """
        skill = self._skills.get(name)
        if not skill:
            return None

        metadata = skill.metadata
        return {
            "name": metadata.name,
            "description": metadata.description,
            "category": metadata.category.name,
            "version": metadata.version,
            "author": metadata.author,
            "tags": metadata.tags,
            "triggers": metadata.triggers,
            "tool_usage_guide": metadata.tool_usage_guide,
            "best_practices": metadata.best_practices,
            "error_handling": metadata.error_handling,
            "requires_tools": metadata.requires_tools,
            "estimated_duration": metadata.estimated_duration,
            "examples": metadata.examples,
        }


class SkillLoader:
    """Loader for skills from files and directories."""

    def __init__(self, registry: Optional[SkillRegistry] = None):
        """Initialize skill loader.

        Args:
            registry: Optional registry to load skills into
        """
        self.registry = registry or SkillRegistry()

    def load_from_file(self, file_path: str) -> Optional[Skill]:
        """Load a skill from a Python file.

        Args:
            file_path: Path to skill file

        Returns:
            Loaded skill or None
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"[SkillLoader] File not found: {file_path}")
            return None

        try:
            spec = importlib.util.spec_from_file_location("skill", file_path)
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, Skill) and attr != Skill:
                    skill = attr()
                    logger.info(f"[SkillLoader] Loaded skill from {file_path}: {skill.name}")
                    return skill

        except Exception as e:
            logger.error(f"[SkillLoader] Failed to load skill from {file_path}: {e}")

        return None

    def load_from_directory(self, dir_path: str, recursive: bool = True):
        """Load all skills from a directory.

        Args:
            dir_path: Directory path
            recursive: Whether to search recursively
        """
        path = Path(dir_path)
        if not path.is_dir():
            logger.error(f"[SkillLoader] Not a directory: {dir_path}")
            return

        pattern = "**/*.py" if recursive else "*.py"

        for file_path in path.glob(pattern):
            if file_path.stem.startswith("_"):
                continue

            skill = self.load_from_file(str(file_path))
            if skill:
                self.registry.register(skill)

        self.load_from_config_directory(dir_path)

    def load_from_config_directory(self, dir_path: str):
        """Load skills from JSON/YAML config files.

        Args:
            dir_path: Directory containing skill config files
        """
        path = Path(dir_path)
        if not path.is_dir():
            return

        for file_path in path.glob("*.json"):
            try:
                self._load_skill_from_json(file_path)
            except Exception as e:
                logger.warning(f"[SkillLoader] Failed to load skill from {file_path}: {e}")

        for file_path in path.glob("*.yaml"):
            try:
                self._load_skill_from_yaml(file_path)
            except Exception as e:
                logger.warning(f"[SkillLoader] Failed to load skill from {file_path}: {e}")

    def _load_skill_from_json(self, file_path: Path) -> None:
        """Load a skill from JSON config file.

        Args:
            file_path: Path to JSON config file
        """
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self._register_skill_from_config(config)

    def _load_skill_from_yaml(self, file_path: Path) -> None:
        """Load a skill from YAML config file.

        Args:
            file_path: Path to YAML config file
        """
        try:
            import yaml
        except ImportError:
            logger.warning("[SkillLoader] PyYAML not installed, skipping YAML files")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self._register_skill_from_config(config)

    def _register_skill_from_config(self, config: Dict[str, Any]) -> None:
        """Register a skill from config dictionary.

        Args:
            config: Skill configuration dictionary
        """
        name = config.get("name")
        if not name:
            logger.warning("[SkillLoader] Skill config missing 'name' field")
            return

        category_str = config.get("category", "CUSTOM")
        try:
            category = SkillCategory[category_str.upper()]
        except KeyError:
            category = SkillCategory.CUSTOM

        metadata = SkillMetadata(
            name=name,
            description=config.get("description", ""),
            category=category,
            version=config.get("version", "1.0.0"),
            author=config.get("author"),
            tags=config.get("tags", []),
            examples=config.get("examples", []),
            requires=config.get("requires", []),
            triggers=config.get("triggers", []),
            tool_usage_guide=config.get("tool_usage_guide", ""),
            best_practices=config.get("best_practices", []),
            error_handling=config.get("error_handling", []),
            requires_tools=config.get("requires_tools", []),
            estimated_duration=config.get("estimated_duration"),
        )

        class_config = {
            "name": metadata.name,
            "description": metadata.description,
            "category": metadata.category,
            "metadata": metadata,
            "config": config,
        }

        skill = type(
            f"{metadata.name.title().replace('_', '')}Skill",
            (Skill,),
            {
                "_create_metadata": lambda self: metadata,
                "_skill_config": class_config,
                "execute": lambda self, input_data: self._execute_from_config(input_data),
            }
        )()

        self.registry.register(skill)
        logger.info(f"[SkillLoader] Loaded skill from config: {name}")

    def _execute_from_config(self, input_data: SkillInput) -> SkillOutput:
        """Execute a config-based skill.

        Args:
            input_data: Skill input

        Returns:
            Skill output
        """
        config = getattr(self, "_skill_config", {}).get("config", {})

        return SkillOutput(
            success=True,
            result={"executed_from_config": True, "config": config},
            logs=[f"Executed skill: {self.name}"]
        )

    def load_builtin_skills(self):
        """Load built-in skills."""
        from .builtin_skills import (
            GenerateUnitTestSkill,
            FixCompilationErrorSkill,
            AnalyzeCodeSkill,
            RefactorCodeSkill,
            GenerateDocSkill,
            ExplainCodeSkill,
            DebugTestSkill,
        )

        for skill_class in [
            GenerateUnitTestSkill,
            FixCompilationErrorSkill,
            AnalyzeCodeSkill,
            RefactorCodeSkill,
            GenerateDocSkill,
            ExplainCodeSkill,
            DebugTestSkill,
        ]:
            skill = skill_class()
            self.registry.register(skill)

        logger.info("[SkillLoader] Loaded built-in skills")


class EnhancedSkillExecutor:
    """Enhanced skill executor with validation, retry and rollback support."""

    def __init__(
        self,
        registry: SkillRegistry,
        tool_registry: Optional[Any] = None,
        max_retries: int = 3,
        default_timeout: int = 300
    ):
        """Initialize enhanced skill executor.

        Args:
            registry: Skill registry
            tool_registry: Optional tool registry for skill execution
            max_retries: Maximum retry attempts
            default_timeout: Default execution timeout in seconds
        """
        self.registry = registry
        self.tool_registry = tool_registry
        self.max_retries = max_retries
        self.default_timeout = default_timeout
        self._execution_history: List[Dict[str, Any]] = []

    async def execute_skill(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SkillOutput:
        """Execute a skill with validation and retry.

        Args:
            skill_name: Name of skill to execute
            parameters: Skill parameters
            context: Execution context

        Returns:
            SkillOutput with execution result
        """
        skill = self.registry.get(skill_name)
        if not skill:
            return SkillOutput(
                success=False,
                error=f"Skill not found: {skill_name}"
            )

        if not await self.validate_execution(skill, parameters, context):
            return SkillOutput(
                success=False,
                error="Validation failed for skill execution"
            )

        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                input_data = SkillInput(
                    parameters=parameters,
                    context=context or {},
                    files=parameters.get("files", [])
                )

                output = await skill.execute(input_data)
                self._record_execution(skill_name, parameters, output, retry_count)
                return output

            except Exception as e:
                last_error = e
                retry_count += 1
                logger.warning(
                    f"[EnhancedSkillExecutor] Retry {retry_count}/{self.max_retries} "
                    f"for skill {skill_name}: {e}"
                )

                if retry_count <= self.max_retries:
                    await asyncio.sleep(2 ** retry_count)

        rollback_success = await self.rollback_on_failure(skill, parameters, context, last_error)

        error_output = SkillOutput(
            success=False,
            error=f"Skill execution failed after {self.max_retries} retries: {last_error}",
            logs=[f"Rollback {'succeeded' if rollback_success else 'failed'}"]
        )
        self._record_execution(skill_name, parameters, error_output, retry_count)
        return error_output

    async def validate_execution(
        self,
        skill: Skill,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Validate skill execution conditions.

        Args:
            skill: Skill to validate
            parameters: Skill parameters
            context: Execution context

        Returns:
            True if execution is valid
        """
        metadata = skill.metadata

        if metadata.requires_tools and self.tool_registry:
            available_tools = [t.name for t in self.tool_registry.list_tools()]
            for required_tool in metadata.requires_tools:
                if required_tool not in available_tools:
                    logger.error(
                        f"[EnhancedSkillExecutor] Required tool not available: {required_tool}"
                    )
                    return False

        return True

    async def rollback_on_failure(
        self,
        skill: Skill,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: Optional[Exception]
    ) -> bool:
        """Attempt rollback on skill failure.

        Args:
            skill: Skill that failed
            parameters: Parameters passed to skill
            context: Execution context
            error: Exception that caused failure

        Returns:
            True if rollback succeeded
        """
        logger.info(f"[EnhancedSkillExecutor] Attempting rollback for skill: {skill.name}")

        if context and "rollback_action" in context:
            try:
                rollback_action = context["rollback_action"]
                if asyncio.iscoroutinefunction(rollback_action):
                    await rollback_action(parameters, context)
                else:
                    rollback_action(parameters, context)
                return True
            except Exception as e:
                logger.error(f"[EnhancedSkillExecutor] Rollback failed: {e}")
                return False

        return False

    def _record_execution(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        output: SkillOutput,
        retry_count: int
    ) -> None:
        """Record skill execution for history.

        Args:
            skill_name: Name of executed skill
            parameters: Parameters used
            output: Execution output
            retry_count: Number of retries
        """
        self._execution_history.append({
            "skill_name": skill_name,
            "parameters": parameters,
            "success": output.success,
            "error": output.error,
            "retry_count": retry_count,
            "timestamp": asyncio.get_event_loop().time()
        })

        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-500:]

    def get_execution_history(
        self,
        skill_name: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get execution history.

        Args:
            skill_name: Optional filter by skill name
            limit: Maximum number of records

        Returns:
            List of execution records
        """
        history = self._execution_history
        if skill_name:
            history = [h for h in history if h["skill_name"] == skill_name]
        return history[-limit:]


_global_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get global skill registry.

    Returns:
        Global SkillRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def register_skill(skill: Skill):
    """Register a skill in global registry.

    Args:
        skill: Skill to register
    """
    get_skill_registry().register(skill)


def get_skill(name: str) -> Optional[Skill]:
    """Get a skill from global registry.

    Args:
        name: Skill name

    Returns:
        Skill or None
    """
    return get_skill_registry().get(name)


async def execute_skill(name: str, **kwargs) -> SkillOutput:
    """Execute a skill from global registry.

    Args:
        name: Skill name
        **kwargs: Skill parameters

    Returns:
        Skill output
    """
    skill = get_skill(name)
    if not skill:
        return SkillOutput(
            success=False,
            error=f"Skill not found: {name}"
        )

    input_data = SkillInput(parameters=kwargs)
    return await skill.execute(input_data)
