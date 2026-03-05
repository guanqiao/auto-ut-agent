"""Unit tests for SkillRegistry."""

import pytest
from unittest.mock import AsyncMock, patch

from pyutagent.skills.skill_base import (
    SkillBase,
    SkillCategory,
    SkillResult,
    SkillExample,
)
from pyutagent.skills.skill_registry import (
    SkillRegistry,
    get_skill_registry,
    register_skill,
)


class SimpleSkill(SkillBase):
    """Simple skill for testing."""

    name = "simple_skill"
    description = "A simple test skill"
    category = SkillCategory.UTILITY
    required_tools = []

    async def execute(self, task, context, tools):
        return SkillResult.ok(message=f"Simple: {task}")


class BuildSkill(SkillBase):
    """Build skill for testing."""

    name = "build_skill"
    description = "A build skill"
    category = SkillCategory.BUILD
    required_tools = ["maven"]

    async def execute(self, task, context, tools):
        return SkillResult.ok(message=f"Build: {task}")


class SkillForTesting(SkillBase):
    """Test skill for testing."""

    name = "skill_for_testing"
    description = "A skill for testing"
    category = SkillCategory.TEST
    required_tools = ["junit"]

    async def execute(self, task, context, tools):
        return SkillResult.ok(message=f"Test: {task}")


class TestSkillRegistry:
    """Tests for SkillRegistry class."""

    def test_create_registry(self):
        """Test creating a skill registry."""
        registry = SkillRegistry()
        
        assert len(registry) == 0
        assert registry.list_skills() == []

    def test_register_skill(self):
        """Test registering a skill."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        assert len(registry) == 1
        assert registry.has("simple_skill")
        assert registry.get("simple_skill") == SimpleSkill

    def test_register_skill_with_name_override(self):
        """Test registering skill with name override."""
        registry = SkillRegistry()
        registry.register(SimpleSkill, name="custom_name")
        
        assert registry.has("custom_name")
        assert not registry.has("simple_skill")

    def test_register_skill_with_aliases(self):
        """Test registering skill with aliases."""
        registry = SkillRegistry()
        registry.register(SimpleSkill, aliases=["simple", "basic"])
        
        assert registry.has("simple_skill")
        assert registry.has("simple")
        assert registry.has("basic")

    def test_register_skill_overwrites(self):
        """Test that registering overwrites existing skill."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        class NewSimpleSkill(SkillBase):
            name = "simple_skill"
            description = "New simple skill"
            async def execute(self, task, context, tools):
                return SkillResult.ok()
        
        registry.register(NewSimpleSkill)
        
        assert registry.get("simple_skill") == NewSimpleSkill

    def test_register_skill_without_name_raises(self):
        """Test that registering skill without name raises error."""
        registry = SkillRegistry()
        
        class NamelessSkill(SkillBase):
            name = ""
            description = "No name"
            async def execute(self, task, context, tools):
                return SkillResult.ok()
        
        with pytest.raises(ValueError, match="must have a name"):
            registry.register(NamelessSkill)

    def test_unregister_skill(self):
        """Test unregistering a skill."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        result = registry.unregister("simple_skill")
        
        assert result is True
        assert len(registry) == 0
        assert not registry.has("simple_skill")

    def test_unregister_nonexistent_skill(self):
        """Test unregistering non-existent skill returns False."""
        registry = SkillRegistry()
        
        result = registry.unregister("nonexistent")
        
        assert result is False

    def test_unregister_removes_aliases(self):
        """Test that unregistering removes aliases."""
        registry = SkillRegistry()
        registry.register(SimpleSkill, aliases=["alias1", "alias2"])
        
        registry.unregister("simple_skill")
        
        assert not registry.has("alias1")
        assert not registry.has("alias2")

    def test_get_skill(self):
        """Test getting a skill class."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        skill_class = registry.get("simple_skill")
        
        assert skill_class == SimpleSkill

    def test_get_skill_by_alias(self):
        """Test getting a skill by alias."""
        registry = SkillRegistry()
        registry.register(SimpleSkill, aliases=["simple"])
        
        skill_class = registry.get("simple")
        
        assert skill_class == SimpleSkill

    def test_get_nonexistent_skill(self):
        """Test getting non-existent skill returns None."""
        registry = SkillRegistry()
        
        assert registry.get("nonexistent") is None

    def test_get_instance(self):
        """Test getting a skill instance."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        instance = registry.get_instance("simple_skill")
        
        assert isinstance(instance, SimpleSkill)

    def test_get_instance_caches(self):
        """Test that get_instance caches instances."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        instance1 = registry.get_instance("simple_skill")
        instance2 = registry.get_instance("simple_skill")
        
        assert instance1 is instance2

    def test_get_instance_nonexistent(self):
        """Test getting instance of non-existent skill."""
        registry = SkillRegistry()
        
        assert registry.get_instance("nonexistent") is None

    def test_has_skill(self):
        """Test checking if skill exists."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        assert registry.has("simple_skill") is True
        assert registry.has("nonexistent") is False

    def test_list_skills(self):
        """Test listing all skills."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        registry.register(BuildSkill)
        registry.register(SkillForTesting)
        
        skills = registry.list_skills()
        
        assert len(skills) == 3
        assert "simple_skill" in skills
        assert "build_skill" in skills
        assert "skill_for_testing" in skills

    def test_list_skills_by_category(self):
        """Test listing skills by category."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        registry.register(BuildSkill)
        registry.register(SkillForTesting)
        
        build_skills = registry.list_skills(SkillCategory.BUILD)
        test_skills = registry.list_skills(SkillCategory.TEST)
        utility_skills = registry.list_skills(SkillCategory.UTILITY)
        
        assert build_skills == ["build_skill"]
        assert test_skills == ["skill_for_testing"]
        assert utility_skills == ["simple_skill"]

    def test_get_instructions(self):
        """Test getting combined instructions."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        registry.register(BuildSkill)
        
        instructions = registry.get_instructions()
        
        assert "simple_skill" in instructions
        assert "build_skill" in instructions

    def test_get_instructions_by_category(self):
        """Test getting instructions by category."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        registry.register(BuildSkill)
        
        instructions = registry.get_instructions(SkillCategory.BUILD)
        
        assert "build_skill" in instructions
        assert "simple_skill" not in instructions

    def test_get_metadata(self):
        """Test getting skill metadata."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        meta = registry.get_metadata("simple_skill")
        
        assert meta is not None
        assert meta.name == "simple_skill"
        assert meta.description == "A simple test skill"

    def test_get_metadata_nonexistent(self):
        """Test getting metadata for non-existent skill."""
        registry = SkillRegistry()
        
        assert registry.get_metadata("nonexistent") is None

    def test_get_all_metadata(self):
        """Test getting all skill metadata."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        registry.register(BuildSkill)
        
        all_meta = registry.get_all_metadata()
        
        assert len(all_meta) == 2
        assert "simple_skill" in all_meta
        assert "build_skill" in all_meta

    @pytest.mark.asyncio
    async def test_execute_skill(self):
        """Test executing a skill."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        result = await registry.execute(
            "simple_skill",
            task="test task",
            context={},
            tools={},
        )
        
        assert result.success is True
        assert result.message == "Simple: test task"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_skill(self):
        """Test executing non-existent skill."""
        registry = SkillRegistry()
        
        result = await registry.execute(
            "nonexistent",
            task="test",
            context={},
            tools={},
        )
        
        assert result.success is False
        assert "not found" in result.message.lower()

    def test_clear(self):
        """Test clearing all skills."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        registry.register(BuildSkill)
        
        registry.clear()
        
        assert len(registry) == 0
        assert registry.list_skills() == []

    def test_get_stats(self):
        """Test getting registry statistics."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        registry.register(BuildSkill)
        registry.register(SkillForTesting)
        
        stats = registry.get_stats()
        
        assert stats["total_skills"] == 3
        assert stats["aliases"] == 0
        assert "skills_by_category" in stats

    def test_contains_operator(self):
        """Test 'in' operator for checking skill existence."""
        registry = SkillRegistry()
        registry.register(SimpleSkill)
        
        assert "simple_skill" in registry
        assert "nonexistent" not in registry

    def test_len_operator(self):
        """Test len() operator."""
        registry = SkillRegistry()
        
        assert len(registry) == 0
        
        registry.register(SimpleSkill)
        assert len(registry) == 1
        
        registry.register(BuildSkill)
        assert len(registry) == 2


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_skill_registry_singleton(self):
        """Test that get_skill_registry returns singleton."""
        registry1 = get_skill_registry()
        registry2 = get_skill_registry()
        
        assert registry1 is registry2

    def test_register_skill_decorator(self):
        """Test @register_skill decorator."""
        registry = get_skill_registry()
        initial_count = len(registry)
        
        @register_skill()
        class DecoratedSkill(SkillBase):
            name = "decorated_skill"
            description = "Decorated skill"
            async def execute(self, task, context, tools):
                return SkillResult.ok()
        
        assert "decorated_skill" in registry
        
        registry.unregister("decorated_skill")

    def test_register_skill_decorator_with_name(self):
        """Test @register_skill decorator with name override."""
        registry = get_skill_registry()
        
        @register_skill(name="custom_decorated")
        class NamedDecoratedSkill(SkillBase):
            name = "named_decorated"
            description = "Named decorated skill"
            async def execute(self, task, context, tools):
                return SkillResult.ok()
        
        assert "custom_decorated" in registry
        
        registry.unregister("custom_decorated")

    def test_register_skill_decorator_with_aliases(self):
        """Test @register_skill decorator with aliases."""
        registry = get_skill_registry()
        
        @register_skill(aliases=["alias_dec"])
        class AliasedDecoratedSkill(SkillBase):
            name = "aliased_decorated"
            description = "Aliased decorated skill"
            async def execute(self, task, context, tools):
                return SkillResult.ok()
        
        assert "aliased_decorated" in registry
        assert "alias_dec" in registry
        
        registry.unregister("aliased_decorated")
