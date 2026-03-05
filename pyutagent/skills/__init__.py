"""Skills System - Tool Usage Instructions.

This module provides:
- SkillBase: Base class for skills
- SkillRegistry: Registry for skill management
- Core skills for common operations
"""

from .skill_base import SkillBase, SkillMeta, SkillResult, SkillCategory, SkillExample
from .skill_registry import SkillRegistry, get_skill_registry, register_skill

__all__ = [
    "SkillBase",
    "SkillMeta",
    "SkillResult",
    "SkillCategory",
    "SkillExample",
    "SkillRegistry",
    "get_skill_registry",
    "register_skill",
]
