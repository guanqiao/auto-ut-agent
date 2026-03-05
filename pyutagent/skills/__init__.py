"""Skills System - Reusable Domain Knowledge Packages.

This module provides a comprehensive skill system inspired by Claude Code Skills:
- SkillBase: Base class for all skills
- SkillRegistry: Central registry for skill management
- SkillMetadata: Rich metadata for skills
- SkillContext: Execution context for skills
- SkillPackage: Community shareable skill packages

Core Features:
- Skill encapsulation with domain knowledge
- Version management and compatibility checking
- Progressive disclosure (三级系统: brief/standard/full)
- Community sharing via SkillPackage
- Discovery and search capabilities

Example Usage:
    # Define a skill
    class MySkill(SkillBase):
        name = "my_skill"
        description = "Does something useful"
        category = SkillCategory.UTILITY

        async def execute(self, task, context, inputs):
            # Implementation
            return SkillResult.ok(message="Success")

    # Register and use
    registry = get_skill_registry()
    registry.register(MySkill)

    # Execute
    result = await registry.execute("my_skill", "task", context, inputs)

    # Export for sharing
    package = registry.export_skill("my_skill")
    with open("my_skill.json", "w") as f:
        f.write(package.to_json())
"""

from .skill_base import (
    SkillBase,
    SkillCategory,
    SkillLevel,
    SkillMetadata,
    SkillContext,
    SkillInput,
    SkillOutput,
    SkillParameter,
    SkillExample,
    SkillResult,
    SkillVersion,
    skill,
)
from .skill_registry import (
    SkillRegistry,
    SkillPackage,
    SkillInfo,
    get_skill_registry,
    reset_skill_registry,
    register_skill,
    discover_skills,
)

# Import UT generation skills
from .ut_generation_skill import (
    UTGenerationSkill,
    UTImprovementSkill,
    UTFixSkill,
    UTGenerationConfig,
)

__all__ = [
    # Base classes
    "SkillBase",
    "SkillCategory",
    "SkillLevel",
    "SkillMetadata",
    "SkillContext",
    "SkillInput",
    "SkillOutput",
    "SkillParameter",
    "SkillExample",
    "SkillResult",
    "SkillVersion",
    "skill",
    # Registry
    "SkillRegistry",
    "SkillPackage",
    "SkillInfo",
    "get_skill_registry",
    "reset_skill_registry",
    "register_skill",
    "discover_skills",
    # UT Generation Skills
    "UTGenerationSkill",
    "UTImprovementSkill",
    "UTFixSkill",
    "UTGenerationConfig",
]

__version__ = "1.0.0"
