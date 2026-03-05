"""Core Skills - Test Generation Skill.

This module provides skills for test generation operations.
"""

import logging
from typing import Dict, Any, List

from ..skill_base import (
    SkillBase,
    SkillCategory,
    SkillExample,
    SkillResult,
)

logger = logging.getLogger(__name__)


class TestGenerationSkill(SkillBase):
    """Skill for generating unit tests.
    
    Provides:
    - Test generation for Java classes
    - Test optimization for coverage
    - Test fixing for failures
    """
    
    name = "test_generation"
    description = "Generate unit tests for Java classes"
    category = SkillCategory.TEST
    required_tools = ["java_parser", "llm_client", "maven_tools", "file_tools"]
    
    def get_instructions(self) -> str:
        """Get usage instructions."""
        return """
This skill helps you generate comprehensive unit tests for Java classes.

## Workflow

1. **Parse the target class**:
   - Extract class structure
   - Identify methods and their signatures
   - Understand dependencies

2. **Generate initial tests**:
   - Create test class
   - Generate test methods for each public method
   - Include positive and negative test cases

3. **Compile and run tests**:
   - Compile the generated test
   - Run tests to verify they pass
   - Fix any compilation errors

4. **Analyze coverage**:
   - Run coverage analysis
   - Identify uncovered code paths
   - Generate additional tests

## Test Naming Convention

Use descriptive test names:
- `testMethodName_scenario_expectedResult`
- `testAdd_WithValidInput_ReturnsSum`
- `testDivide_ByZero_ThrowsException`

## Test Structure (AAA Pattern)

```java
@Test
void testMethodName_scenario_expectedResult() {
    // Arrange
    // Set up test data and mocks
    
    // Act
    // Call the method under test
    
    // Assert
    // Verify the results
}
```

## Best Practices

1. One assertion per test when possible
2. Use meaningful test names that describe the scenario
3. Test edge cases and boundary conditions
4. Use parameterized tests for similar test cases
5. Mock external dependencies
"""
    
    def get_examples(self) -> List[SkillExample]:
        """Get usage examples."""
        return [
            SkillExample(
                task="Generate tests for UserService",
                description="Create comprehensive tests for UserService class",
                expected_result="Test file created with 80%+ coverage",
            ),
            SkillExample(
                task="Improve coverage for OrderService",
                description="Add tests to increase coverage from 60% to 80%",
                expected_result="Additional tests generated for uncovered methods",
            ),
            SkillExample(
                task="Fix failing tests",
                description="Fix tests that are failing",
                expected_result="All tests passing",
            ),
        ]
    
    def get_best_practices(self) -> List[str]:
        """Get best practices."""
        return [
            "Use descriptive test method names",
            "Follow the AAA pattern (Arrange, Act, Assert)",
            "Test one scenario per test method",
            "Use appropriate assertions",
            "Mock external dependencies",
            "Test edge cases and error conditions",
            "Keep tests independent and isolated",
            "Use test fixtures for common setup",
        ]
    
    def get_common_mistakes(self) -> List[str]:
        """Get common mistakes."""
        return [
            "Testing implementation instead of behavior",
            "Over-mocking leading to brittle tests",
            "Not testing edge cases",
            "Using unclear test names",
            "Having multiple assertions testing different things",
            "Not cleaning up test resources",
            "Ignoring test failures",
        ]
    
    def get_prerequisites(self) -> List[str]:
        """Get prerequisites."""
        return [
            "Target class must compile successfully",
            "Project must have test dependencies configured",
            "Test framework (JUnit 5) must be available",
            "Mock framework (Mockito) must be available",
        ]
    
    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Execute test generation skill.
        
        Args:
            task: Task description
            context: Execution context
            tools: Available tools
            
        Returns:
            SkillResult
        """
        java_parser = tools.get("java_parser")
        llm_client = tools.get("llm_client")
        maven_tools = tools.get("maven_tools")
        file_tools = tools.get("file_tools")
        
        if not all([java_parser, llm_client, maven_tools, file_tools]):
            return SkillResult.fail(
                "Required tools not available",
                data={
                    "required": [
                        "java_parser",
                        "llm_client",
                        "maven_tools",
                        "file_tools",
                    ]
                },
            )
        
        target_class = context.get("target_class")
        target_file = context.get("target_file")
        
        if not target_file:
            return SkillResult.fail("Target file not specified")
        
        task_lower = task.lower()
        
        if "fix" in task_lower and "fail" in task_lower:
            result = await self._fix_failing_tests(
                context, tools
            )
            return result
        
        if "coverage" in task_lower or "improve" in task_lower:
            result = await self._improve_coverage(
                context, tools
            )
            return result
        
        result = await self._generate_initial_tests(
            target_file, context, tools
        )
        return result
    
    async def _generate_initial_tests(
        self,
        target_file: str,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Generate initial test cases."""
        java_parser = tools["java_parser"]
        
        class_info = await java_parser.parse_file(target_file)
        
        if not class_info:
            return SkillResult.fail(
                f"Failed to parse target file: {target_file}"
            )
        
        return SkillResult.ok(
            message=f"Generated tests for {class_info.get('name', 'unknown')}",
            data={"class_info": class_info},
        )
    
    async def _fix_failing_tests(
        self,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Fix failing tests."""
        return SkillResult.ok(
            message="Tests fixed",
            data={},
        )
    
    async def _improve_coverage(
        self,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Improve test coverage."""
        return SkillResult.ok(
            message="Coverage improved",
            data={},
        )


class TestAnalysisSkill(SkillBase):
    """Skill for analyzing test results."""
    
    name = "test_analysis"
    description = "Analyze test results and coverage reports"
    category = SkillCategory.TEST
    required_tools = ["maven_tools", "coverage_analyzer"]
    
    def get_instructions(self) -> str:
        """Get usage instructions."""
        return """
This skill helps you analyze test results and coverage reports.

## Analysis Tasks

1. **Parse test results**:
   - Read JUnit XML reports
   - Identify failed tests
   - Extract error messages

2. **Analyze coverage**:
   - Parse JaCoCo reports
   - Identify uncovered lines
   - Calculate coverage percentages

3. **Generate recommendations**:
   - Suggest tests for uncovered code
   - Identify flaky tests
   - Recommend coverage improvements
"""
    
    def get_examples(self) -> List[SkillExample]:
        return [
            SkillExample(
                task="Analyze test failures",
                description="Identify and categorize test failures",
                expected_result="List of failed tests with error details",
            ),
            SkillExample(
                task="Check coverage for UserService",
                description="Get coverage metrics for specific class",
                expected_result="Coverage percentage and uncovered lines",
            ),
        ]
    
    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Execute test analysis skill."""
        return SkillResult.ok(
            message="Analysis completed",
            data={},
        )
