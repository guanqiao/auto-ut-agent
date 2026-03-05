"""Skill Registry for Managing Skills.

This module provides:
- SkillRegistry: Central registry for skills
- Global registry instance
"""

import logging
from typing import Dict, List, Optional, Any, Type, Callable

from .skill_base import SkillBase, SkillCategory, SkillMeta, SkillResult

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central registry for managing skills.
    
    Features:
    - Register/unregister skills
    - Discover skills by category
    - Get skill instructions for LLM
    - Execute skills by name
    """
    
    def __init__(self):
        """Initialize skill registry."""
        self._skills: Dict[str, Type[SkillBase]] = {}
        self._instances: Dict[str, SkillBase] = {}
        self._categories: Dict[SkillCategory, List[str]] = {
            cat: [] for cat in SkillCategory
        }
        self._aliases: Dict[str, str] = {}
    
    def register(
        self,
        skill_class: Type[SkillBase],
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a skill class.
        
        Args:
            skill_class: Skill class to register
            name: Optional name override
            aliases: Optional aliases for the skill
        """
        skill_name = name or skill_class.name
        
        if not skill_name:
            raise ValueError("Skill must have a name")
        
        if skill_name in self._skills:
            logger.warning(f"Overwriting existing skill: {skill_name}")
        
        self._skills[skill_name] = skill_class
        self._categories[skill_class.category].append(skill_name)
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = skill_name
        
        logger.debug(f"Registered skill: {skill_name} ({skill_class.category.value})")
    
    def unregister(self, name: str) -> bool:
        """Unregister a skill.
        
        Args:
            name: Name of skill to unregister
            
        Returns:
            True if skill was removed
        """
        if name not in self._skills:
            return False
        
        skill_class = self._skills.pop(name)
        self._categories[skill_class.category].remove(name)
        self._instances.pop(name, None)
        
        aliases_to_remove = [
            alias for alias, target in self._aliases.items()
            if target == name
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        logger.debug(f"Unregistered skill: {name}")
        return True
    
    def get(self, name: str) -> Optional[Type[SkillBase]]:
        """Get a skill class by name.
        
        Args:
            name: Skill name or alias
            
        Returns:
            Skill class or None
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
            self._instances[actual_name] = self._skills[actual_name]()
        
        return self._instances[actual_name]
    
    def has(self, name: str) -> bool:
        """Check if a skill is registered."""
        actual_name = self._aliases.get(name, name)
        return actual_name in self._skills
    
    def list_skills(self, category: Optional[SkillCategory] = None) -> List[str]:
        """List all registered skills.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of skill names
        """
        if category:
            return self._categories[category].copy()
        return list(self._skills.keys())
    
    def get_instructions(self, category: Optional[SkillCategory] = None) -> str:
        """Get all skill instructions for LLM context.
        
        Args:
            category: Optional category filter
            
        Returns:
            Combined instructions string
        """
        skills = self.list_skills(category)
        instructions = []
        
        for name in skills:
            instance = self.get_instance(name)
            if instance:
                instructions.append(instance.get_prompt_context())
        
        return "\n\n---\n\n".join(instructions)
    
    def get_metadata(self, name: str) -> Optional[SkillMeta]:
        """Get metadata for a skill."""
        instance = self.get_instance(name)
        return instance.metadata if instance else None
    
    def get_all_metadata(self) -> Dict[str, SkillMeta]:
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
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Execute a skill by name.
        
        Args:
            name: Skill name
            task: Task description
            context: Execution context
            tools: Available tools
            
        Returns:
            SkillResult from execution
        """
        instance = self.get_instance(name)
        
        if instance is None:
            return SkillResult.fail(
                message=f"Skill not found: {name}",
                data={"error": "SKILL_NOT_FOUND"},
            )
        
        return await instance.run(task, context, tools)
    
    def clear(self) -> None:
        """Clear all registered skills."""
        self._skills.clear()
        self._instances.clear()
        self._categories = {cat: [] for cat in SkillCategory}
        self._aliases.clear()
        logger.info("All skills cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_skills": len(self._skills),
            "skills_by_category": {
                cat.value: len(skills)
                for cat, skills in self._categories.items()
            },
            "aliases": len(self._aliases),
        }
    
    def __contains__(self, name: str) -> bool:
        """Check if skill is registered using 'in' operator."""
        return self.has(name)
    
    def __len__(self) -> int:
        """Get number of registered skills."""
        return len(self._skills)


_global_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global skill registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def register_skill(
    name: Optional[str] = None,
    aliases: Optional[List[str]] = None,
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
        
    Returns:
        Decorator function
    """
    def decorator(skill_class: Type[SkillBase]) -> Type[SkillBase]:
        get_skill_registry().register(skill_class, name, aliases)
        return skill_class
    
    return decorator
