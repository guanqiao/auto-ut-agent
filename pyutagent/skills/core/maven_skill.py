"""Core Skills - Maven Build Skill.

This module provides skills for Maven build operations.
"""

import logging
from typing import Dict, Any, List

from ..skill_base import (
    SkillBase,
    SkillCategory,
    SkillExample,
    SkillResult,
)
from ..skill_registry import register_skill

logger = logging.getLogger(__name__)


@register_skill()
class MavenBuildSkill(SkillBase):
    """Skill for Maven build operations.
    
    Provides:
    - Building projects
    - Running tests
    - Generating coverage reports
    - Managing dependencies
    """
    
    name = "maven_build"
    description = "Build Java projects using Maven"
    category = SkillCategory.BUILD
    required_tools = ["maven_tools", "file_tools"]
    
    def get_instructions(self) -> str:
        """Get usage instructions."""
        return """
This skill helps you build and test Java projects using Maven.

## Basic Commands

1. **Build the project**:
   ```
   mvn compile
   ```
   Compiles the source code.

2. **Run all tests**:
   ```
   mvn test
   ```
   Runs all unit tests.

3. **Run specific test class**:
   ```
   mvn test -Dtest=ClassName
   ```
   Runs tests for a specific class.

4. **Generate coverage report**:
   ```
   mvn jacoco:report
   ```
   Generates code coverage report.

5. **Clean build**:
   ```
   mvn clean install
   ```
   Cleans and builds the entire project.

## Common Options

- `-DskipTests`: Skip test execution
- `-P <profile>`: Use a specific profile
- `-Dmaven.test.skip=true`: Skip test compilation and execution
- `-pl <module>`: Build only specific module
- `-am`: Also build required dependencies

## Best Practices

1. Always run `mvn compile` before running tests
2. Use `-DskipTests` only when necessary
3. Check for compilation errors before running tests
4. Review test output in `target/surefire-reports/`
"""
    
    def get_examples(self) -> List[SkillExample]:
        """Get usage examples."""
        return [
            SkillExample(
                task="Build the project",
                description="Compile all source code",
                code_example="mvn compile",
                expected_result="BUILD SUCCESS",
            ),
            SkillExample(
                task="Run all tests",
                description="Execute all unit tests",
                code_example="mvn test",
                expected_result="Tests run: X, Failures: 0, Errors: 0",
            ),
            SkillExample(
                task="Run specific test",
                description="Run tests for a single class",
                code_example="mvn test -Dtest=UserServiceTest",
                expected_result="Tests for UserServiceTest completed",
            ),
            SkillExample(
                task="Generate coverage report",
                description="Create JaCoCo coverage report",
                code_example="mvn jacoco:report",
                expected_result="Report generated at target/site/jacoco/",
            ),
        ]
    
    def get_best_practices(self) -> List[str]:
        """Get best practices."""
        return [
            "Always compile before running tests",
            "Use specific test execution for faster feedback",
            "Check coverage reports after test runs",
            "Keep dependencies up to date",
            "Use Maven profiles for different environments",
        ]
    
    def get_common_mistakes(self) -> List[str]:
        """Get common mistakes."""
        return [
            "Running tests without compiling first",
            "Forgetting to add test dependencies",
            "Not checking test output for failures",
            "Using outdated plugins",
            "Ignoring compilation warnings",
        ]
    
    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Execute Maven build skill.
        
        Args:
            task: Task description
            context: Execution context
            tools: Available tools
            
        Returns:
            SkillResult
        """
        maven_tools = tools.get("maven_tools")
        
        if not maven_tools:
            return SkillResult.fail(
                "Maven tools not available",
                data={"required": "maven_tools"},
            )
        
        task_lower = task.lower()
        
        if "test" in task_lower and "coverage" not in task_lower:
            if "specific" in task_lower or "class" in task_lower:
                test_class = context.get("test_class")
                if test_class:
                    result = await maven_tools.run_test(test_class)
                    return SkillResult.ok(
                        message=f"Tests executed for {test_class}",
                        data={"test_class": test_class, "result": result},
                    )
            
            result = await maven_tools.run_all_tests()
            return SkillResult.ok(
                message="All tests executed",
                data={"result": result},
            )
        
        if "coverage" in task_lower:
            result = await maven_tools.generate_coverage()
            return SkillResult.ok(
                message="Coverage report generated",
                data={"result": result},
                artifacts=["target/site/jacoco/index.html"],
            )
        
        if "clean" in task_lower:
            result = await maven_tools.clean()
            return SkillResult.ok(
                message="Project cleaned",
                data={"result": result},
            )
        
        result = await maven_tools.compile()
        return SkillResult.ok(
            message="Project compiled",
            data={"result": result},
        )


@register_skill()
class MavenDependencySkill(SkillBase):
    """Skill for Maven dependency management."""
    
    name = "maven_dependency"
    description = "Manage Maven project dependencies"
    category = SkillCategory.BUILD
    required_tools = ["maven_tools"]
    
    def get_instructions(self) -> str:
        """Get usage instructions."""
        return """
This skill helps you manage Maven project dependencies.

## Common Tasks

1. **Resolve dependencies**:
   ```
   mvn dependency:resolve
   ```
   Downloads and resolves all dependencies.

2. **Show dependency tree**:
   ```
   mvn dependency:tree
   ```
   Displays the dependency hierarchy.

3. **Check for updates**:
   ```
   mvn versions:display-dependency-updates
   ```
   Shows available dependency updates.

4. **Add dependency**:
   Add to pom.xml:
   ```xml
   <dependency>
       <groupId>group</groupId>
       <artifactId>artifact</artifactId>
       <version>version</version>
   </dependency>
   ```

## Dependency Scopes

- `compile`: Default, available in all phases
- `test`: Only for test compilation and execution
- `provided`: Not packaged with the application
- `runtime`: Not needed for compilation
- `system`: Uses local JAR file
"""
    
    def get_examples(self) -> List[SkillExample]:
        return [
            SkillExample(
                task="Resolve dependencies",
                description="Download all project dependencies",
                code_example="mvn dependency:resolve",
                expected_result="Dependencies resolved successfully",
            ),
            SkillExample(
                task="Show dependency tree",
                description="Display dependency hierarchy",
                code_example="mvn dependency:tree",
                expected_result="Dependency tree displayed",
            ),
        ]
    
    async def execute(
        self,
        task: str,
        context: Dict[str, Any],
        tools: Dict[str, Any],
    ) -> SkillResult:
        """Execute dependency management skill."""
        maven_tools = tools.get("maven_tools")
        
        if not maven_tools:
            return SkillResult.fail("Maven tools not available")
        
        task_lower = task.lower()
        
        if "resolve" in task_lower:
            result = await maven_tools.resolve_dependencies()
            return SkillResult.ok(
                message="Dependencies resolved",
                data={"result": result},
            )
        
        if "tree" in task_lower:
            result = await maven_tools.show_dependency_tree()
            return SkillResult.ok(
                message="Dependency tree displayed",
                data={"result": result},
            )
        
        return SkillResult.fail(f"Unknown dependency task: {task}")
