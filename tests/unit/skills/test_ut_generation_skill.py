"""Unit tests for ut_generation_skill module."""

import pytest
from pathlib import Path
from typing import Dict, Any

from pyutagent.skills.ut_generation_skill import (
    UTGenerationSkill,
    UTImprovementSkill,
    UTFixSkill,
    UTGenerationConfig,
)
from pyutagent.skills.skill_base import (
    SkillBase,
    SkillCategory,
    SkillLevel,
    SkillResult,
    SkillContext,
    SkillInput,
    SkillOutput,
)


class TestUTGenerationConfig:
    """Tests for UTGenerationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = UTGenerationConfig(
            target_file="src/main/java/Test.java",
            project_path="/project",
        )

        assert config.target_file == "src/main/java/Test.java"
        assert config.project_path == "/project"
        assert config.test_framework == "junit5"
        assert config.mock_framework == "mockito"
        assert config.coverage_target == 0.8
        assert config.generate_mocks is True
        assert config.include_edge_cases is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = UTGenerationConfig(
            target_file="Test.java",
            project_path="/project",
            test_framework="testng",
            mock_framework="easymock",
            coverage_target=0.9,
            generate_mocks=False,
            include_edge_cases=False,
        )

        assert config.test_framework == "testng"
        assert config.mock_framework == "easymock"
        assert config.coverage_target == 0.9
        assert config.generate_mocks is False
        assert config.include_edge_cases is False


class TestUTGenerationSkill:
    """Tests for UTGenerationSkill."""

    def test_skill_attributes(self):
        """Test skill class attributes."""
        assert UTGenerationSkill.name == "ut_generation"
        assert UTGenerationSkill.category == SkillCategory.UT_GENERATION
        assert UTGenerationSkill.level == SkillLevel.INTERMEDIATE
        assert "java_parser" in UTGenerationSkill.required_tools
        assert "llm_client" in UTGenerationSkill.required_tools
        assert UTGenerationSkill.version == "1.0.0"
        assert "testing" in UTGenerationSkill.tags

    def test_get_input_spec(self):
        """Test input specification."""
        skill = UTGenerationSkill()
        input_spec = skill.get_input_spec()

        assert isinstance(input_spec, SkillInput)
        assert len(input_spec.parameters) > 0

        # Check required parameter
        target_file_param = next(p for p in input_spec.parameters if p.name == "target_file")
        assert target_file_param.required is True
        assert target_file_param.type == "string"

        # Check optional parameter with default
        test_framework_param = next(p for p in input_spec.parameters if p.name == "test_framework")
        assert test_framework_param.required is False
        assert test_framework_param.default == "junit5"
        assert "junit5" in test_framework_param.enum

    def test_get_output_spec(self):
        """Test output specification."""
        skill = UTGenerationSkill()
        output_spec = skill.get_output_spec()

        assert isinstance(output_spec, SkillOutput)
        assert "test_file" in output_spec.properties
        assert "test_code" in output_spec.properties
        assert "coverage" in output_spec.properties
        assert "test_count" in output_spec.properties

    def test_get_instructions(self):
        """Test instructions are provided."""
        skill = UTGenerationSkill()
        instructions = skill.get_instructions()

        assert len(instructions) > 0
        assert "Workflow" in instructions
        assert "AAA Pattern" in instructions

    def test_get_examples(self):
        """Test examples are provided."""
        skill = UTGenerationSkill()
        examples = skill.get_examples()

        assert len(examples) > 0
        assert all(hasattr(e, "task") for e in examples)
        assert all(hasattr(e, "inputs") for e in examples)

    def test_get_best_practices(self):
        """Test best practices are provided."""
        skill = UTGenerationSkill()
        practices = skill.get_best_practices()

        assert len(practices) > 0
        assert any("AAA" in p for p in practices)
        assert any("test names" in p.lower() for p in practices)

    def test_get_common_mistakes(self):
        """Test common mistakes are provided."""
        skill = UTGenerationSkill()
        mistakes = skill.get_common_mistakes()

        assert len(mistakes) > 0

    def test_get_prerequisites(self):
        """Test prerequisites are provided."""
        skill = UTGenerationSkill()
        prerequisites = skill.get_prerequisites()

        assert len(prerequisites) > 0
        assert any("Maven" in p or "Gradle" in p for p in prerequisites)

    def test_metadata(self):
        """Test skill metadata."""
        skill = UTGenerationSkill()
        metadata = skill.metadata

        assert metadata.name == "ut_generation"
        assert metadata.category == SkillCategory.UT_GENERATION
        assert metadata.input_spec is not None
        assert metadata.output_spec is not None

    @pytest.mark.asyncio
    async def test_execute_missing_tools(self):
        """Test execution with missing tools."""
        skill = UTGenerationSkill()
        context = SkillContext(tools={})  # No tools

        result = await skill.execute(
            "Generate tests",
            context,
            {"target_file": "Test.java"},
        )

        assert result.success is False
        assert result.metadata.get("error_code") == "MISSING_TOOLS"

    @pytest.mark.asyncio
    async def test_execute_missing_target_file(self):
        """Test execution without target file."""
        skill = UTGenerationSkill()
        context = SkillContext(tools={
            "java_parser": {},
            "llm_client": {},
            "maven_tools": {},
            "file_tools": {},
            "coverage_analyzer": {},
        })

        result = await skill.execute(
            "Generate tests",
            context,
            {},  # Missing target_file
        )

        assert result.success is False
        assert "VALIDATION_ERROR" in result.metadata.get("error_code", "")

    def test_basic_parse_java(self):
        """Test basic Java parsing."""
        skill = UTGenerationSkill()

        java_code = """
package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
"""

        result = skill._basic_parse_java(java_code)

        assert result["name"] == "Calculator"
        assert result["package"] == "com.example"
        assert len(result["methods"]) >= 2

    def test_extract_java_code(self):
        """Test extracting Java code from markdown."""
        skill = UTGenerationSkill()

        markdown = """
Here's the test code:

```java
public class Test {
    @Test
    void test() {}
}
```

Some other text.
"""

        result = skill._extract_java_code(markdown)
        assert "public class Test" in result
        assert "@Test" in result

    def test_extract_java_code_without_markdown(self):
        """Test extracting Java code without markdown."""
        skill = UTGenerationSkill()

        code = "public class Test { }"
        result = skill._extract_java_code(code)

        assert result == code

    def test_get_test_file_path(self):
        """Test test file path generation."""
        skill = UTGenerationSkill()
        config = UTGenerationConfig(
            target_file="src/main/java/com/example/Service.java",
            project_path="/project",
        )
        class_info = {
            "name": "Service",
            "package": "com.example",
        }

        path = skill._get_test_file_path(config, class_info)

        assert "ServiceTest.java" in path
        assert "src/test/java" in path
        assert "com/example" in path

    def test_count_test_methods(self):
        """Test counting test methods."""
        skill = UTGenerationSkill()

        test_code = """
@Test
void test1() {}

@Test
@DisplayName("Test 2")
void test2() {}

void notATest() {}
"""

        count = skill._count_test_methods(test_code)
        assert count == 2

    def test_generate_test_template(self):
        """Test test template generation."""
        skill = UTGenerationSkill()
        config = UTGenerationConfig(
            target_file="Test.java",
            project_path="/project",
        )
        class_info = {
            "name": "Calculator",
            "package": "com.example",
            "methods": [
                {"name": "add"},
                {"name": "subtract"},
            ],
        }

        template = skill._generate_test_template(config, class_info)

        assert "CalculatorTest" in template
        assert "@Test" in template
        assert "@BeforeEach" in template
        assert "testAdd" in template or "testSubtract" in template


class TestUTImprovementSkill:
    """Tests for UTImprovementSkill."""

    def test_skill_attributes(self):
        """Test skill class attributes."""
        assert UTImprovementSkill.name == "ut_improvement"
        assert UTImprovementSkill.category == SkillCategory.UT_GENERATION
        assert UTImprovementSkill.level == SkillLevel.ADVANCED

    @pytest.mark.asyncio
    async def test_execute_missing_test_file(self):
        """Test execution without test file."""
        skill = UTImprovementSkill()
        context = SkillContext()

        result = await skill.execute(
            "Improve tests",
            context,
            {},  # Missing test_file
        )

        assert result.success is False
        assert "VALIDATION_ERROR" in result.metadata.get("error_code", "")

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        skill = UTImprovementSkill()
        context = SkillContext()

        result = await skill.execute(
            "Improve tests",
            context,
            {"test_file": "Test.java", "coverage_target": 0.9},
        )

        assert result.success is True


class TestUTFixSkill:
    """Tests for UTFixSkill."""

    def test_skill_attributes(self):
        """Test skill class attributes."""
        assert UTFixSkill.name == "ut_fix"
        assert UTFixSkill.category == SkillCategory.UT_GENERATION
        assert UTFixSkill.level == SkillLevel.ADVANCED

    @pytest.mark.asyncio
    async def test_execute_missing_test_file(self):
        """Test execution without test file."""
        skill = UTFixSkill()
        context = SkillContext()

        result = await skill.execute(
            "Fix tests",
            context,
            {},  # Missing test_file
        )

        assert result.success is False
        assert "VALIDATION_ERROR" in result.metadata.get("error_code", "")

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        skill = UTFixSkill()
        context = SkillContext()

        result = await skill.execute(
            "Fix tests",
            context,
            {"test_file": "Test.java"},
        )

        assert result.success is True
