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

    def load_builtin_skills(self):
        """Load built-in skills."""
        from .builtin_skills import (
            GenerateUnitTestSkill,
            FixCompilationErrorSkill,
            AnalyzeCodeSkill,
        )

        for skill_class in [GenerateUnitTestSkill, FixCompilationErrorSkill, AnalyzeCodeSkill]:
            skill = skill_class()
            self.registry.register(skill)

        logger.info("[SkillLoader] Loaded built-in skills")


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
