"""Unit tests for skill_registry module."""

import pytest
from pathlib import Path
from typing import Dict, Any

from pyutagent.skills.skill_registry import (
    SkillRegistry,
    SkillPackage,
    SkillInfo,
    get_skill_registry,
    reset_skill_registry,
    register_skill,
    discover_skills,
)
from pyutagent.skills.skill_base import (
    SkillBase,
    SkillCategory,
    SkillLevel,
    SkillResult,
    SkillContext,
    SkillVersion,
)


class MockSkill(SkillBase):
    """Mock skill for testing."""
    name = "mock_skill"
    description = "A mock skill for testing"
    category = SkillCategory.TEST
    version = "1.0.0"
    tags = ["mock", "test"]

    async def execute(self, task, context, inputs):
        return SkillResult.ok(message="Mock executed")


class AnotherMockSkill(SkillBase):
    """Another mock skill for testing."""
    name = "another_mock"
    description = "Another mock skill"
    category = SkillCategory.CODE
    version = "2.0.0"
    tags = ["mock", "code"]

    async def execute(self, task, context, inputs):
        return SkillResult.ok(message="Another mock executed")


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_skill_registry()
        self.registry = get_skill_registry()

    def test_register_skill(self):
        """Test registering a skill."""
        self.registry.register(MockSkill)

        assert self.registry.has("mock_skill")
        assert "mock_skill" in self.registry

    def test_register_with_aliases(self):
        """Test registering a skill with aliases."""
        self.registry.register(MockSkill, aliases=["mock", "ms"])

        assert self.registry.has("mock")
        assert self.registry.has("ms")
        assert self.registry.get("mock") == MockSkill

    def test_register_with_tags(self):
        """Test registering a skill with tags."""
        self.registry.register(MockSkill, tags=["extra", "tags"])

        skills = self.registry.list_skills(tags=["extra"])
        assert "mock_skill" in skills

    def test_unregister_skill(self):
        """Test unregistering a skill."""
        self.registry.register(MockSkill)
        assert self.registry.has("mock_skill")

        result = self.registry.unregister("mock_skill")
        assert result is True
        assert not self.registry.has("mock_skill")

    def test_get_skill(self):
        """Test getting a skill class."""
        self.registry.register(MockSkill)

        skill_class = self.registry.get("mock_skill")
        assert skill_class == MockSkill

    def test_get_instance(self):
        """Test getting a skill instance."""
        self.registry.register(MockSkill)

        instance1 = self.registry.get_instance("mock_skill")
        instance2 = self.registry.get_instance("mock_skill")

        assert isinstance(instance1, MockSkill)
        assert instance1 is instance2  # Should return same instance

    def test_get_info(self):
        """Test getting skill info."""
        self.registry.register(MockSkill, aliases=["mock"])

        info = self.registry.get_info("mock_skill")
        assert isinstance(info, SkillInfo)
        assert info.name == "mock_skill"
        assert info.category == SkillCategory.TEST
        assert "mock" in info.aliases

    def test_list_skills(self):
        """Test listing skills."""
        self.registry.register(MockSkill)
        self.registry.register(AnotherMockSkill)

        all_skills = self.registry.list_skills()
        assert "mock_skill" in all_skills
        assert "another_mock" in all_skills

        test_skills = self.registry.list_skills(category=SkillCategory.TEST)
        assert "mock_skill" in test_skills
        assert "another_mock" not in test_skills

    def test_list_skills_by_tags(self):
        """Test listing skills by tags."""
        self.registry.register(MockSkill)
        self.registry.register(AnotherMockSkill)

        mock_skills = self.registry.list_skills(tags=["mock"])
        assert "mock_skill" in mock_skills
        assert "another_mock" in mock_skills

        test_skills = self.registry.list_skills(tags=["test"])
        assert "mock_skill" in test_skills
        assert "another_mock" not in test_skills

    def test_search(self):
        """Test searching skills."""
        self.registry.register(MockSkill)
        self.registry.register(AnotherMockSkill)

        results = self.registry.search("mock")
        assert len(results) == 2

        results = self.registry.search("another")
        assert len(results) == 1
        assert results[0][0] == "another_mock"

    def test_discover_by_pattern(self):
        """Test discovering skills by pattern."""
        self.registry.register(MockSkill)
        self.registry.register(AnotherMockSkill)

        results = self.registry.discover_by_pattern("mock_*")
        assert "mock_skill" in results

        results = self.registry.discover_by_pattern("*mock*")
        assert "mock_skill" in results
        assert "another_mock" in results

    def test_get_instructions(self):
        """Test getting skill instructions."""
        self.registry.register(MockSkill)

        instructions = self.registry.get_instructions()
        assert "mock_skill" in instructions
        assert "A mock skill for testing" in instructions

    def test_get_metadata(self):
        """Test getting skill metadata."""
        self.registry.register(MockSkill)

        metadata = self.registry.get_metadata("mock_skill")
        assert metadata is not None
        assert metadata.name == "mock_skill"

    def test_get_all_metadata(self):
        """Test getting all skill metadata."""
        self.registry.register(MockSkill)
        self.registry.register(AnotherMockSkill)

        all_metadata = self.registry.get_all_metadata()
        assert "mock_skill" in all_metadata
        assert "another_mock" in all_metadata

    @pytest.mark.asyncio
    async def test_execute_skill(self):
        """Test executing a skill."""
        self.registry.register(MockSkill)

        context = SkillContext()
        result = await self.registry.execute("mock_skill", "test task", context, {})

        assert result.success is True
        assert result.message == "Mock executed"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_skill(self):
        """Test executing a non-existent skill."""
        context = SkillContext()
        result = await self.registry.execute("nonexistent", "task", context, {})

        assert result.success is False
        assert result.metadata.get("error_code") == "SKILL_NOT_FOUND"

    def test_check_version_compatibility(self):
        """Test version compatibility checking."""
        self.registry.register(MockSkill)  # version 1.0.0

        # Compatible version (same major, lower or equal minor)
        compatible, message = self.registry.check_version_compatibility("mock_skill", "1.0.0")
        assert compatible is True

        # Note: 0.5.0 has different major version (0 vs 1), so it's incompatible
        compatible, message = self.registry.check_version_compatibility("mock_skill", "0.5.0")
        assert compatible is False

        # Incompatible version (higher minor required)
        compatible, message = self.registry.check_version_compatibility("mock_skill", "1.5.0")
        assert compatible is False

        # Incompatible version (different major)
        compatible, message = self.registry.check_version_compatibility("mock_skill", "2.0.0")
        assert compatible is False

    def test_export_skill(self):
        """Test exporting a skill."""
        self.registry.register(MockSkill)

        package = self.registry.export_skill("mock_skill")
        assert isinstance(package, SkillPackage)
        assert package.metadata.name == "mock_skill"
        assert package.skill_class_name == "MockSkill"

    def test_import_skill(self):
        """Test importing a skill package."""
        package = SkillPackage(
            metadata=MockSkill().metadata,
            skill_class_name="MockSkill",
            skill_module="test_module",
        )

        result = self.registry.import_skill(package)
        assert result is True

    def test_get_stats(self):
        """Test getting registry statistics."""
        self.registry.register(MockSkill, aliases=["mock"])
        self.registry.register(AnotherMockSkill)

        stats = self.registry.get_stats()
        assert stats["total_skills"] == 2
        assert stats["aliases"] == 1

    def test_clear(self):
        """Test clearing registry."""
        self.registry.register(MockSkill)
        assert len(self.registry) == 1

        self.registry.clear()
        assert len(self.registry) == 0
        assert not self.registry.has("mock_skill")

    def test_len(self):
        """Test registry length."""
        assert len(self.registry) == 0

        self.registry.register(MockSkill)
        assert len(self.registry) == 1

    def test_iter(self):
        """Test iterating over registry."""
        self.registry.register(MockSkill)
        self.registry.register(AnotherMockSkill)

        skills = list(self.registry)
        assert "mock_skill" in skills
        assert "another_mock" in skills


class TestSkillPackage:
    """Tests for SkillPackage."""

    def test_to_dict(self):
        """Test package to dict conversion."""
        package = SkillPackage(
            metadata=MockSkill().metadata,
            skill_class_name="MockSkill",
            skill_module="test_module",
            dependencies=["dep1", "dep2"],
            readme="Test readme",
        )

        result = package.to_dict()
        assert result["skill_class_name"] == "MockSkill"
        assert result["skill_module"] == "test_module"
        assert result["dependencies"] == ["dep1", "dep2"]
        assert result["readme"] == "Test readme"

    def test_to_json(self):
        """Test package to JSON conversion."""
        package = SkillPackage(
            metadata=MockSkill().metadata,
            skill_class_name="MockSkill",
            skill_module="test_module",
        )

        json_str = package.to_json()
        assert "MockSkill" in json_str
        assert "mock_skill" in json_str

    def test_from_dict(self):
        """Test creating package from dict."""
        data = {
            "metadata": {
                "name": "test_skill",
                "description": "Test description",
                "version": "1.0.0",
                "category": "test",
                "level": "intermediate",
            },
            "skill_class_name": "TestSkill",
            "skill_module": "test_module",
            "dependencies": [],
            "files": [],
        }

        package = SkillPackage.from_dict(data)
        assert package.metadata.name == "test_skill"
        assert package.skill_class_name == "TestSkill"

    def test_from_json(self):
        """Test creating package from JSON."""
        json_str = '''{
            "metadata": {
                "name": "test_skill",
                "description": "Test description",
                "version": "1.0.0",
                "category": "test",
                "level": "intermediate"
            },
            "skill_class_name": "TestSkill",
            "skill_module": "test_module"
        }'''

        package = SkillPackage.from_json(json_str)
        assert package.metadata.name == "test_skill"


class TestRegisterSkillDecorator:
    """Tests for @register_skill decorator."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_skill_registry()

    def test_register_skill_decorator(self):
        """Test register_skill decorator."""

        @register_skill(aliases=["decorated"], tags=["decorator"])
        class DecoratedSkill(SkillBase):
            name = "decorated_skill"
            description = "A decorated skill"
            category = SkillCategory.UTILITY

            async def execute(self, task, context, inputs):
                return SkillResult.ok()

        registry = get_skill_registry()
        assert registry.has("decorated_skill")
        assert registry.has("decorated")


class TestDiscoverSkills:
    """Tests for discover_skills function."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_skill_registry()
        self.registry = get_skill_registry()
        self.registry.register(MockSkill)
        self.registry.register(AnotherMockSkill)

    def test_discover_by_category(self):
        """Test discovering skills by category."""
        results = discover_skills(category=SkillCategory.TEST)
        assert "mock_skill" in results
        assert "another_mock" not in results

    def test_discover_by_tags(self):
        """Test discovering skills by tags."""
        results = discover_skills(tags=["code"])
        assert "another_mock" in results
        assert "mock_skill" not in results

    def test_discover_by_query(self):
        """Test discovering skills by query."""
        results = discover_skills(query="another")
        assert "another_mock" in results
        assert "mock_skill" not in results
