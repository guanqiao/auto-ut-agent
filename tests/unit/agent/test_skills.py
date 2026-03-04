"""Tests for Skills framework."""

import pytest
from pyutagent.agent.skills import (
    SkillCategory,
    SkillMetadata,
    SkillInput,
    SkillOutput,
    Skill,
    SkillRegistry,
    SkillLoader,
    EnhancedSkillExecutor,
)


class TestSkillMetadata:
    """Test SkillMetadata dataclass."""

    def test_basic_metadata(self):
        """Test basic metadata creation."""
        metadata = SkillMetadata(
            name="test_skill",
            description="A test skill",
            category=SkillCategory.TESTING,
        )
        assert metadata.name == "test_skill"
        assert metadata.description == "A test skill"
        assert metadata.category == SkillCategory.TESTING

    def test_enhanced_metadata(self):
        """Test enhanced metadata with new fields."""
        metadata = SkillMetadata(
            name="generate_unit_test",
            description="Generate unit tests",
            category=SkillCategory.TESTING,
            triggers=["测试", "generate test"],
            tool_usage_guide="1. Parse the class\n2. Generate tests",
            best_practices=["Test all public methods"],
            error_handling=["Handle compilation errors"],
            requires_tools=["java_parser", "maven_tools"],
            estimated_duration="2-5 minutes",
        )
        assert len(metadata.triggers) == 2
        assert "java_parser" in metadata.requires_tools
        assert metadata.estimated_duration == "2-5 minutes"


class TestSkillRegistry:
    """Test SkillRegistry."""

    def test_registry_init(self):
        """Test registry initialization."""
        registry = SkillRegistry()
        assert len(registry.list_skills()) == 0

    def test_register_and_get(self):
        """Test skill registration and retrieval."""

        class TestSkill(Skill):
            def _create_metadata(self):
                return SkillMetadata(
                    name="test_skill",
                    description="Test skill",
                    category=SkillCategory.CUSTOM,
                )

            async def execute(self, input_data):
                return SkillOutput(success=True, result={"test": True})

        registry = SkillRegistry()
        skill = TestSkill()
        registry.register(skill)

        assert "test_skill" in registry.list_skills()
        assert registry.get("test_skill") is not None
        assert registry.get("test_skill").name == "test_skill"

    def test_find_by_trigger(self):
        """Test finding skills by trigger."""

        class TriggerSkill(Skill):
            def _create_metadata(self):
                return SkillMetadata(
                    name="trigger_skill",
                    description="A skill with triggers",
                    category=SkillCategory.TESTING,
                    triggers=["测试", "generate test", "单元测试"],
                )

            async def execute(self, input_data):
                return SkillOutput(success=True)

        registry = SkillRegistry()
        skill = TriggerSkill()
        registry.register(skill)

        results = registry.find_by_trigger("生成测试")
        assert "trigger_skill" in results

        results = registry.find_by_trigger("unit test")
        assert "trigger_skill" in results

    def test_get_skill_info(self):
        """Test getting skill info."""

        class InfoSkill(Skill):
            def _create_metadata(self):
                return SkillMetadata(
                    name="info_skill",
                    description="Skill with full info",
                    category=SkillCategory.TESTING,
                    version="1.0.0",
                    author="Test Author",
                    tags=["test", "info"],
                    triggers=["info"],
                    tool_usage_guide="Test guide",
                    best_practices=["Practice 1"],
                    error_handling=["Error 1"],
                    requires_tools=["tool1"],
                    estimated_duration="1 minute",
                )

            async def execute(self, input_data):
                return SkillOutput(success=True)

        registry = SkillRegistry()
        skill = InfoSkill()
        registry.register(skill)

        info = registry.get_skill_info("info_skill")
        assert info is not None
        assert info["name"] == "info_skill"
        assert info["version"] == "1.0.0"
        assert info["author"] == "Test Author"
        assert len(info["triggers"]) == 1
        assert info["tool_usage_guide"] == "Test guide"


class TestSkillLoader:
    """Test SkillLoader."""

    def test_load_builtin_skills(self):
        """Test loading built-in skills."""
        loader = SkillLoader()
        loader.load_builtin_skills()

        skills = loader.registry.list_skills()
        assert "generate_unit_test" in skills
        assert "fix_compilation_error" in skills
        assert "analyze_code" in skills
        assert "refactor_code" in skills
        assert "generate_doc" in skills
        assert "explain_code" in skills
        assert "debug_test" in skills

    def test_builtin_skill_has_triggers(self):
        """Test that built-in skills have triggers."""
        loader = SkillLoader()
        loader.load_builtin_skills()

        generate_test = loader.registry.get("generate_unit_test")
        assert generate_test is not None
        assert len(generate_test.metadata.triggers) > 0
        assert len(generate_test.metadata.best_practices) > 0
        assert len(generate_test.metadata.error_handling) > 0


class TestEnhancedSkillExecutor:
    """Test EnhancedSkillExecutor."""

    @pytest.mark.asyncio
    async def test_execute_skill_success(self):
        """Test successful skill execution."""

        class SuccessSkill(Skill):
            def _create_metadata(self):
                return SkillMetadata(
                    name="success_skill",
                    description="Always succeeds",
                    category=SkillCategory.CUSTOM,
                )

            async def execute(self, input_data):
                return SkillOutput(success=True, result={"value": 42})

        registry = SkillRegistry()
        skill = SuccessSkill()
        registry.register(skill)

        executor = EnhancedSkillExecutor(registry)
        output = await executor.execute_skill("success_skill", {})

        assert output.success is True
        assert output.result["value"] == 42

    @pytest.mark.asyncio
    async def test_execute_skill_not_found(self):
        """Test execution of non-existent skill."""
        registry = SkillRegistry()
        executor = EnhancedSkillExecutor(registry)

        output = await executor.execute_skill("non_existent_skill", {})

        assert output.success is False
        assert "not found" in output.error.lower()

    @pytest.mark.asyncio
    async def test_validate_execution(self):
        """Test execution validation."""

        class ValidatedSkill(Skill):
            def _create_metadata(self):
                return SkillMetadata(
                    name="validated_skill",
                    description="Skill requiring tools",
                    category=SkillCategory.CUSTOM,
                    requires_tools=["required_tool"],
                )

            async def execute(self, input_data):
                return SkillOutput(success=True)

        registry = SkillRegistry()
        skill = ValidatedSkill()
        registry.register(skill)

        executor = EnhancedSkillExecutor(registry, tool_registry=None)
        is_valid = await executor.validate_execution(skill, {}, {})

        assert is_valid is True

    @pytest.mark.asyncio
    async def test_execution_history(self):
        """Test execution history tracking."""

        class HistorySkill(Skill):
            def _create_metadata(self):
                return SkillMetadata(
                    name="history_skill",
                    description="Track history",
                    category=SkillCategory.CUSTOM,
                )

            async def execute(self, input_data):
                return SkillOutput(success=True)

        registry = SkillRegistry()
        skill = HistorySkill()
        registry.register(skill)

        executor = EnhancedSkillExecutor(registry)
        await executor.execute_skill("history_skill", {"param": "value"})

        history = executor.get_execution_history("history_skill")
        assert len(history) == 1
        assert history[0]["skill_name"] == "history_skill"
        assert history[0]["parameters"]["param"] == "value"


class TestBuiltinSkills:
    """Test built-in skills metadata."""

    def test_generate_unit_test_triggers(self):
        """Test generate_unit_test skill has Chinese triggers."""
        loader = SkillLoader()
        loader.load_builtin_skills()

        skill = loader.registry.get("generate_unit_test")
        assert skill is not None
        assert "生成测试" in skill.metadata.triggers
        assert "测试" in skill.metadata.triggers
        assert len(skill.metadata.best_practices) > 0

    def test_fix_compilation_error_triggers(self):
        """Test fix_compilation_error skill has triggers."""
        loader = SkillLoader()
        loader.load_builtin_skills()

        skill = loader.registry.get("fix_compilation_error")
        assert skill is not None
        assert "编译错误" in skill.metadata.triggers
        assert len(skill.metadata.tool_usage_guide) > 0

    def test_all_skills_have_estimated_duration(self):
        """Test all built-in skills have estimated duration."""
        loader = SkillLoader()
        loader.load_builtin_skills()

        for skill_name in loader.registry.list_skills():
            skill = loader.registry.get(skill_name)
            assert skill.metadata.estimated_duration is not None, f"{skill_name} missing estimated_duration"
