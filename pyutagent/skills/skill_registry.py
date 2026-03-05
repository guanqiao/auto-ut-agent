"""Skill Registry for Managing Skills.

This module provides:
- SkillRegistry: Central registry for skills with version management
- SkillDiscovery: Discovery and search capabilities
- SkillPackage: Skill packaging for community sharing
- Global registry instance

Design inspired by Claude Code Skills system:
- Community shareable packages
- Version compatibility checking
- Progressive discovery
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable, Set, Tuple
from dataclasses import dataclass, field
import fnmatch

from .skill_base import (
    SkillBase,
    SkillCategory,
    SkillLevel,
    SkillMetadata,
    SkillResult,
    SkillContext,
    SkillVersion,
)

logger = logging.getLogger(__name__)


@dataclass
class SkillPackage:
    """Skill package for community sharing.

    Represents a shareable skill package that can be:
    - Exported to a file
    - Imported from a file
    - Shared with the community
    """

    metadata: SkillMetadata
    skill_class_name: str
    skill_module: str
    dependencies: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    readme: str = ""
    license: str = "MIT"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "skill_class_name": self.skill_class_name,
            "skill_module": self.skill_module,
            "dependencies": self.dependencies,
            "files": self.files,
            "readme": self.readme,
            "license": self.license,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPackage":
        """Create from dictionary."""
        metadata = SkillMetadata(
            name=data["metadata"]["name"],
            description=data["metadata"]["description"],
            version=SkillVersion.from_string(data["metadata"].get("version", "1.0.0")),
            required_tools=data["metadata"].get("required_tools", []),
            category=SkillCategory(data["metadata"].get("category", "utility")),
            level=SkillLevel(data["metadata"].get("level", "intermediate")),
            instructions=data["metadata"].get("instructions", ""),
            best_practices=data["metadata"].get("best_practices", []),
            common_mistakes=data["metadata"].get("common_mistakes", []),
            prerequisites=data["metadata"].get("prerequisites", []),
            tags=data["metadata"].get("tags", []),
            author=data["metadata"].get("author", ""),
            homepage=data["metadata"].get("homepage", ""),
        )

        return cls(
            metadata=metadata,
            skill_class_name=data["skill_class_name"],
            skill_module=data["skill_module"],
            dependencies=data.get("dependencies", []),
            files=data.get("files", []),
            readme=data.get("readme", ""),
            license=data.get("license", "MIT"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SkillPackage":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class SkillInfo:
    """Information about a registered skill."""

    name: str
    skill_class: Type[SkillBase]
    version: SkillVersion
    category: SkillCategory
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    is_builtin: bool = True
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": str(self.version),
            "category": self.category.value,
            "aliases": self.aliases,
            "tags": self.tags,
            "is_builtin": self.is_builtin,
            "source": self.source,
        }


class SkillRegistry:
    """Central registry for managing skills.

    Features:
    - Register/unregister skills
    - Discover skills by category, tags, or search
    - Version management and compatibility checking
    - Get skill instructions for LLM
    - Execute skills by name
    - Import/export skill packages
    """

    def __init__(self):
        """Initialize skill registry."""
        self._skills: Dict[str, SkillInfo] = {}
        self._instances: Dict[str, SkillBase] = {}
        self._categories: Dict[SkillCategory, List[str]] = {
            cat: [] for cat in SkillCategory
        }
        self._aliases: Dict[str, str] = {}
        self._tags: Dict[str, List[str]] = {}
        self._packages: Dict[str, SkillPackage] = {}

    def register(
        self,
        skill_class: Type[SkillBase],
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        is_builtin: bool = True,
        source: str = "",
    ) -> None:
        """Register a skill class.

        Args:
            skill_class: Skill class to register
            name: Optional name override
            aliases: Optional aliases for the skill
            tags: Optional tags for the skill
            is_builtin: Whether this is a built-in skill
            source: Source of the skill (module path, file path, etc.)
        """
        skill_name = name or skill_class.name

        if not skill_name:
            raise ValueError("Skill must have a name")

        # Create instance to get metadata
        try:
            instance = skill_class()
            version = instance.metadata.version
            category = instance.metadata.category
        except Exception as e:
            logger.warning(f"Could not instantiate skill {skill_name}: {e}")
            version = SkillVersion.from_string(skill_class.version)
            category = skill_class.category

        # Check for existing skill with same name
        if skill_name in self._skills:
            existing = self._skills[skill_name]
            if existing.version.major == version.major:
                logger.warning(
                    f"Overwriting existing skill: {skill_name} "
                    f"(v{existing.version} -> v{version})"
                )
            else:
                logger.error(
                    f"Cannot overwrite skill {skill_name} with incompatible version: "
                    f"v{existing.version} vs v{version}"
                )
                raise ValueError(
                    f"Incompatible version for skill {skill_name}: "
                    f"v{existing.version} vs v{version}"
                )

        # Create skill info
        skill_info = SkillInfo(
            name=skill_name,
            skill_class=skill_class,
            version=version,
            category=category,
            aliases=aliases or [],
            tags=tags or list(skill_class.tags),
            is_builtin=is_builtin,
            source=source or skill_class.__module__,
        )

        # Register skill
        self._skills[skill_name] = skill_info
        self._categories[category].append(skill_name)

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = skill_name

        # Register tags
        for tag in skill_info.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            self._tags[tag].append(skill_name)

        logger.debug(f"Registered skill: {skill_name} ({category.value}, v{version})")

    def unregister(self, name: str) -> bool:
        """Unregister a skill.

        Args:
            name: Name of skill to unregister

        Returns:
            True if skill was removed
        """
        actual_name = self._aliases.get(name, name)

        if actual_name not in self._skills:
            return False

        skill_info = self._skills.pop(actual_name)
        self._categories[skill_info.category].remove(actual_name)
        self._instances.pop(actual_name, None)

        # Remove aliases
        aliases_to_remove = [
            alias for alias, target in self._aliases.items() if target == actual_name
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]

        # Remove from tags
        for tag in skill_info.tags:
            if tag in self._tags and actual_name in self._tags[tag]:
                self._tags[tag].remove(actual_name)

        logger.debug(f"Unregistered skill: {actual_name}")
        return True

    def get(self, name: str) -> Optional[Type[SkillBase]]:
        """Get a skill class by name.

        Args:
            name: Skill name or alias

        Returns:
            Skill class or None
        """
        actual_name = self._aliases.get(name, name)
        skill_info = self._skills.get(actual_name)
        return skill_info.skill_class if skill_info else None

    def get_info(self, name: str) -> Optional[SkillInfo]:
        """Get skill information.

        Args:
            name: Skill name or alias

        Returns:
            SkillInfo or None
        """
        actual_name = self._aliases.get(name, name)
        return self._skills.get(actual_name)

    def get_instance(self, name: str) -> Optional[SkillBase]:
        """Get or create a skill instance.

        Args:
            name: Skill name

        Returns:
            Skill instance or None
        """
        actual_name = self._aliases.get(name, name)

        if actual_name not in self._skills:
            return None

        if actual_name not in self._instances:
            skill_class = self._skills[actual_name].skill_class
            self._instances[actual_name] = skill_class()

        return self._instances[actual_name]

    def has(self, name: str) -> bool:
        """Check if a skill is registered."""
        actual_name = self._aliases.get(name, name)
        return actual_name in self._skills

    def list_skills(
        self,
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
        level: Optional[SkillLevel] = None,
    ) -> List[str]:
        """List all registered skills.

        Args:
            category: Optional category filter
            tags: Optional tags filter (all must match)
            level: Optional skill level filter

        Returns:
            List of skill names
        """
        if category:
            result = self._categories[category].copy()
        else:
            result = list(self._skills.keys())

        if tags:
            # Filter by tags (intersection)
            tag_skills: Set[str] = set()
            for tag in tags:
                if tag in self._tags:
                    if not tag_skills:
                        tag_skills = set(self._tags[tag])
                    else:
                        tag_skills &= set(self._tags[tag])
            result = [name for name in result if name in tag_skills]

        if level:
            result = [
                name
                for name in result
                if self.get_instance(name)
                and self.get_instance(name).metadata.level == level
            ]

        return result

    def search(self, query: str) -> List[Tuple[str, float]]:
        """Search skills by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of (skill_name, relevance_score) tuples
        """
        query_lower = query.lower()
        results = []

        for name, skill_info in self._skills.items():
            score = 0.0

            # Name match
            if query_lower in name.lower():
                score += 1.0

            # Get instance for more info
            instance = self.get_instance(name)
            if instance:
                metadata = instance.metadata

                # Description match
                if query_lower in metadata.description.lower():
                    score += 0.5

                # Instructions match
                if query_lower in metadata.instructions.lower():
                    score += 0.3

                # Tags match
                for tag in metadata.tags:
                    if query_lower in tag.lower():
                        score += 0.4

            if score > 0:
                results.append((name, score))

        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def discover_by_pattern(self, pattern: str) -> List[str]:
        """Discover skills by glob pattern.

        Args:
            pattern: Glob pattern (e.g., "test_*", "*generation*")

        Returns:
            List of matching skill names
        """
        return [name for name in self._skills if fnmatch.fnmatch(name, pattern)]

    def get_instructions(
        self,
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
        detail_level: str = "full",
    ) -> str:
        """Get all skill instructions for LLM context.

        Args:
            category: Optional category filter
            tags: Optional tags filter
            detail_level: Level of detail (brief, standard, full)

        Returns:
            Combined instructions string
        """
        skills = self.list_skills(category=category, tags=tags)
        instructions = []

        for name in skills:
            instance = self.get_instance(name)
            if instance:
                instructions.append(instance.get_prompt_context(detail_level))

        return "\n\n---\n\n".join(instructions)

    def get_metadata(self, name: str) -> Optional[SkillMetadata]:
        """Get metadata for a skill."""
        instance = self.get_instance(name)
        return instance.metadata if instance else None

    def get_all_metadata(self) -> Dict[str, SkillMetadata]:
        """Get metadata for all skills."""
        result = {}
        for name in self._skills:
            instance = self.get_instance(name)
            if instance:
                result[name] = instance.metadata
        return result

    async def execute(
        self,
        name: str,
        task: str,
        context: SkillContext,
        inputs: Dict[str, Any] = None,
    ) -> SkillResult:
        """Execute a skill by name.

        Args:
            name: Skill name
            task: Task description
            context: Execution context
            inputs: Input parameters

        Returns:
            SkillResult from execution
        """
        instance = self.get_instance(name)

        if instance is None:
            return SkillResult.fail(
                message=f"Skill not found: {name}",
                error_code="SKILL_NOT_FOUND",
            )

        return await instance.run(task, context, inputs or {})

    def check_version_compatibility(
        self, name: str, required_version: str
    ) -> Tuple[bool, str]:
        """Check if a skill version is compatible.

        Args:
            name: Skill name
            required_version: Required version string

        Returns:
            Tuple of (is_compatible, message)
        """
        skill_info = self.get_info(name)
        if not skill_info:
            return False, f"Skill not found: {name}"

        try:
            required = SkillVersion.from_string(required_version)
            current = skill_info.version

            if current.major != required.major:
                return (
                    False,
                    f"Incompatible major version: required v{required}, found v{current}",
                )

            if current.minor < required.minor:
                return (
                    False,
                    f"Minor version too old: required v{required}, found v{current}",
                )

            return True, f"Compatible: v{current} satisfies v{required}"

        except Exception as e:
            return False, f"Invalid version format: {e}"

    def export_skill(self, name: str) -> Optional[SkillPackage]:
        """Export a skill as a package.

        Args:
            name: Skill name

        Returns:
            SkillPackage or None if not found
        """
        instance = self.get_instance(name)
        if not instance:
            return None

        skill_info = self.get_info(name)
        if not skill_info:
            return None

        return SkillPackage(
            metadata=instance.metadata,
            skill_class_name=skill_info.skill_class.__name__,
            skill_module=skill_info.skill_class.__module__,
        )

    def import_skill(self, package: SkillPackage) -> bool:
        """Import a skill from a package.

        Args:
            package: Skill package to import

        Returns:
            True if successful
        """
        try:
            # Store package info
            self._packages[package.metadata.name] = package
            logger.info(f"Imported skill package: {package.metadata.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to import skill package: {e}")
            return False

    def load_from_directory(self, directory: Path) -> int:
        """Load all skill packages from a directory.

        Args:
            directory: Directory containing skill package files

        Returns:
            Number of skills loaded
        """
        count = 0
        if not directory.exists():
            logger.warning(f"Skill directory not found: {directory}")
            return 0

        for file_path in directory.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                package = SkillPackage.from_dict(data)
                if self.import_skill(package):
                    count += 1
            except Exception as e:
                logger.error(f"Failed to load skill from {file_path}: {e}")

        logger.info(f"Loaded {count} skills from {directory}")
        return count

    def clear(self) -> None:
        """Clear all registered skills."""
        self._skills.clear()
        self._instances.clear()
        self._categories = {cat: [] for cat in SkillCategory}
        self._aliases.clear()
        self._tags.clear()
        self._packages.clear()
        logger.info("All skills cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_skills": len(self._skills),
            "skills_by_category": {
                cat.value: len(skills) for cat, skills in self._categories.items()
            },
            "aliases": len(self._aliases),
            "tags": len(self._tags),
            "packages": len(self._packages),
        }

    def __contains__(self, name: str) -> bool:
        """Check if skill is registered using 'in' operator."""
        return self.has(name)

    def __len__(self) -> int:
        """Get number of registered skills."""
        return len(self._skills)

    def __iter__(self):
        """Iterate over registered skill names."""
        return iter(self._skills.keys())


_global_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def reset_skill_registry() -> None:
    """Reset the global skill registry."""
    global _global_registry
    _global_registry = SkillRegistry()


def register_skill(
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    is_builtin: bool = True,
    source: str = "",
) -> Callable[[Type[SkillBase]], Type[SkillBase]]:
    """Decorator to register a skill class.

    Usage:
        @register_skill()
        class MySkill(SkillBase):
            name = "my_skill"
            ...

    Args:
        name: Optional name override
        aliases: Optional aliases
        tags: Optional tags
        is_builtin: Whether this is a built-in skill
        source: Source of the skill

    Returns:
        Decorator function
    """

    def decorator(skill_class: Type[SkillBase]) -> Type[SkillBase]:
        get_skill_registry().register(
            skill_class, name, aliases, tags, is_builtin, source
        )
        return skill_class

    return decorator


def discover_skills(
    category: Optional[SkillCategory] = None,
    tags: Optional[List[str]] = None,
    query: Optional[str] = None,
) -> List[str]:
    """Discover skills with filters.

    Args:
        category: Optional category filter
        tags: Optional tags filter
        query: Optional search query

    Returns:
        List of skill names
    """
    registry = get_skill_registry()

    if query:
        results = registry.search(query)
        return [name for name, _ in results]

    return registry.list_skills(category=category, tags=tags)
