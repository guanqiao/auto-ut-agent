"""Project Configuration System - PYUT.md

This module provides a project-level configuration system similar to CLAUDE.md.
PYUT.md files allow users to persist project understanding and Agent preferences.

Example PYUT.md:
```markdown
# Project Configuration

## Build
- Tool: Maven
- Java Version: 17
- Build Command: mvn clean install -DskipTests

## Testing
- Framework: TestNG
- Target Coverage: 80%
- Test Directory: src/test/java

## Agent Preferences
- Enable Multi-Agent: true
- Preferred Strategies: [boundary, mutation]
- Max Iterations: 10

## Code Style
- Naming Convention: camelCase
- Max Line Length: 120

## Dependencies
- Spring Boot: 3.0.0
- Lombok: enabled
```
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class BuildTool(Enum):
    """Supported build tools."""
    MAVEN = "maven"
    GRADLE = "gradle"
    UNKNOWN = "unknown"


class TestFramework(Enum):
    """Supported test frameworks."""
    JUNIT4 = "junit4"
    JUNIT5 = "junit5"
    TESTNG = "testng"
    UNKNOWN = "unknown"


class MockFramework(Enum):
    """Supported mock frameworks."""
    MOCKITO = "mockito"
    EASYMOCK = "easymock"
    JMOCK = "jmock"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass
class BuildConfig:
    """Build configuration."""
    tool: BuildTool = BuildTool.MAVEN
    java_version: str = "17"
    build_command: str = "mvn clean install -DskipTests"
    test_command: str = "mvn test"
    compile_command: str = "mvn compile"


@dataclass
class TestConfig:
    """Testing configuration."""
    framework: TestFramework = TestFramework.JUNIT5
    target_coverage: float = 0.8
    test_directory: str = "src/test/java"
    test_suffix: str = "Test"
    generate_integration_tests: bool = False
    mock_framework: str = "mockito"


@dataclass
class AgentPreferences:
    """Agent behavior preferences."""
    enable_multi_agent: bool = False
    enable_error_prediction: bool = True
    enable_self_reflection: bool = True
    enable_pattern_library: bool = True
    enable_chain_of_thought: bool = True
    max_iterations: int = 10
    preferred_strategies: List[str] = field(default_factory=lambda: ["boundary", "positive"])
    timeout_per_file: int = 300
    parallel_workers: int = 1


@dataclass
class CodeStyle:
    """Code style preferences."""
    naming_convention: str = "camelCase"
    max_line_length: int = 120
    indent_size: int = 4
    use_lombok: bool = False
    generate_javadoc: bool = False


@dataclass
class DependencyInfo:
    """Dependency information."""
    name: str
    version: str = ""
    enabled: bool = True


@dataclass
class ProjectContext:
    """Project context information for agent operations.
    
    This class provides a simplified view of project information
    that can be easily passed to agents and other components.
    """
    name: str = ""
    description: str = ""
    language: str = "java"
    build_tool: str = "maven"
    java_version: str = "17"
    test_framework: str = "junit5"
    mock_framework: str = "mockito"
    project_root: Path = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "build_tool": self.build_tool,
            "java_version": self.java_version,
            "test_framework": self.test_framework,
            "mock_framework": self.mock_framework,
            "project_root": str(self.project_root) if self.project_root else None,
        }


@dataclass
class ProjectConfig:
    """Complete project configuration from PYUT.md."""
    project_name: str = ""
    project_root: Path = None
    
    build: BuildConfig = field(default_factory=BuildConfig)
    testing: TestConfig = field(default_factory=TestConfig)
    agent: AgentPreferences = field(default_factory=AgentPreferences)
    code_style: CodeStyle = field(default_factory=CodeStyle)
    dependencies: Dict[str, DependencyInfo] = field(default_factory=dict)
    
    custom_instructions: List[str] = field(default_factory=list)
    ignore_patterns: List[str] = field(default_factory=list)
    
    def to_context(self) -> ProjectContext:
        """Convert to ProjectContext."""
        return ProjectContext(
            name=self.project_name,
            language="java",
            build_tool=self.build.tool.value if self.build.tool else "maven",
            java_version=self.build.java_version,
            test_framework=self.testing.framework.value if self.testing.framework else "junit5",
            mock_framework=self.testing.mock_framework,
            project_root=self.project_root,
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], project_root: Path = None) -> "ProjectConfig":
        """Create ProjectConfig from dictionary."""
        config = cls(project_root=project_root)
        
        if "build" in data:
            build_data = data["build"]
            config.build = BuildConfig(
                tool=BuildTool(build_data.get("tool", "maven").lower()),
                java_version=build_data.get("java_version", "17"),
                build_command=build_data.get("build_command", "mvn clean install -DskipTests"),
                test_command=build_data.get("test_command", "mvn test"),
                compile_command=build_data.get("compile_command", "mvn compile"),
            )
        
        if "testing" in data:
            test_data = data["testing"]
            config.testing = TestConfig(
                framework=TestFramework(test_data.get("framework", "junit5").lower()),
                target_coverage=test_data.get("target_coverage", 0.8),
                test_directory=test_data.get("test_directory", "src/test/java"),
                test_suffix=test_data.get("test_suffix", "Test"),
                generate_integration_tests=test_data.get("generate_integration_tests", False),
                mock_framework=test_data.get("mock_framework", "mockito"),
            )
        
        if "agent" in data:
            agent_data = data["agent"]
            config.agent = AgentPreferences(
                enable_multi_agent=agent_data.get("enable_multi_agent", False),
                enable_error_prediction=agent_data.get("enable_error_prediction", True),
                enable_self_reflection=agent_data.get("enable_self_reflection", True),
                enable_pattern_library=agent_data.get("enable_pattern_library", True),
                enable_chain_of_thought=agent_data.get("enable_chain_of_thought", True),
                max_iterations=agent_data.get("max_iterations", 10),
                preferred_strategies=agent_data.get("preferred_strategies", ["boundary", "positive"]),
                timeout_per_file=agent_data.get("timeout_per_file", 300),
                parallel_workers=agent_data.get("parallel_workers", 1),
            )
        
        if "code_style" in data:
            style_data = data["code_style"]
            config.code_style = CodeStyle(
                naming_convention=style_data.get("naming_convention", "camelCase"),
                max_line_length=style_data.get("max_line_length", 120),
                indent_size=style_data.get("indent_size", 4),
                use_lombok=style_data.get("use_lombok", False),
                generate_javadoc=style_data.get("generate_javadoc", False),
            )
        
        if "dependencies" in data:
            for name, dep_data in data["dependencies"].items():
                if isinstance(dep_data, dict):
                    config.dependencies[name] = DependencyInfo(
                        name=name,
                        version=dep_data.get("version", ""),
                        enabled=dep_data.get("enabled", True),
                    )
                elif isinstance(dep_data, str):
                    config.dependencies[name] = DependencyInfo(
                        name=name,
                        version=dep_data,
                    )
        
        if "custom_instructions" in data:
            config.custom_instructions = data["custom_instructions"]
        
        if "ignore_patterns" in data:
            config.ignore_patterns = data["ignore_patterns"]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_name": self.project_name,
            "build": {
                "tool": self.build.tool.value,
                "java_version": self.build.java_version,
                "build_command": self.build.build_command,
                "test_command": self.build.test_command,
                "compile_command": self.build.compile_command,
            },
            "testing": {
                "framework": self.testing.framework.value,
                "target_coverage": self.testing.target_coverage,
                "test_directory": self.testing.test_directory,
                "test_suffix": self.testing.test_suffix,
                "generate_integration_tests": self.testing.generate_integration_tests,
                "mock_framework": self.testing.mock_framework,
            },
            "agent": {
                "enable_multi_agent": self.agent.enable_multi_agent,
                "enable_error_prediction": self.agent.enable_error_prediction,
                "enable_self_reflection": self.agent.enable_self_reflection,
                "enable_pattern_library": self.agent.enable_pattern_library,
                "enable_chain_of_thought": self.agent.enable_chain_of_thought,
                "max_iterations": self.agent.max_iterations,
                "preferred_strategies": self.agent.preferred_strategies,
                "timeout_per_file": self.agent.timeout_per_file,
                "parallel_workers": self.agent.parallel_workers,
            },
            "code_style": {
                "naming_convention": self.code_style.naming_convention,
                "max_line_length": self.code_style.max_line_length,
                "indent_size": self.code_style.indent_size,
                "use_lombok": self.code_style.use_lombok,
                "generate_javadoc": self.code_style.generate_javadoc,
            },
            "dependencies": {
                name: {"version": dep.version, "enabled": dep.enabled}
                for name, dep in self.dependencies.items()
            },
            "custom_instructions": self.custom_instructions,
            "ignore_patterns": self.ignore_patterns,
        }


class PYUTMdParser:
    """Parser for PYUT.md files."""
    
    SECTION_PATTERN = re.compile(r'^##\s+(.+)$', re.MULTILINE)
    LIST_ITEM_PATTERN = re.compile(r'^-\s+(.+)$', re.MULTILINE)
    KEY_VALUE_PATTERN = re.compile(r'^-\s+(\w+):\s*(.+)$', re.MULTILINE)
    ARRAY_PATTERN = re.compile(r'\[(.+)\]')
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse PYUT.md content into a dictionary.
        
        Args:
            content: PYUT.md file content
            
        Returns:
            Parsed configuration dictionary
        """
        result: Dict[str, Any] = {}
        
        sections = self._split_sections(content)
        
        for section_name, section_content in sections.items():
            section_name_lower = section_name.lower().replace(" ", "_")
            
            if section_name_lower == "build":
                result["build"] = self._parse_build_section(section_content)
            elif section_name_lower == "testing":
                result["testing"] = self._parse_testing_section(section_content)
            elif section_name_lower == "agent_preferences":
                result["agent"] = self._parse_agent_section(section_content)
            elif section_name_lower == "code_style":
                result["code_style"] = self._parse_style_section(section_content)
            elif section_name_lower == "dependencies":
                result["dependencies"] = self._parse_dependencies_section(section_content)
            elif section_name_lower == "custom_instructions":
                result["custom_instructions"] = self._parse_list_section(section_content)
            elif section_name_lower == "ignore":
                result["ignore_patterns"] = self._parse_list_section(section_content)
            else:
                result[section_name_lower] = self._parse_generic_section(section_content)
        
        return result
    
    def _split_sections(self, content: str) -> Dict[str, str]:
        """Split content into sections by ## headers."""
        sections: Dict[str, str] = {}
        
        lines = content.split('\n')
        current_section = "header"
        current_content: List[str] = []
        
        for line in lines:
            if line.startswith('## '):
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _parse_build_section(self, content: str) -> Dict[str, Any]:
        """Parse Build section."""
        result: Dict[str, Any] = {}
        
        for match in self.KEY_VALUE_PATTERN.finditer(content):
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip()
            result[key] = value
        
        return result
    
    def _parse_testing_section(self, content: str) -> Dict[str, Any]:
        """Parse Testing section."""
        result: Dict[str, Any] = {}
        
        for match in self.KEY_VALUE_PATTERN.finditer(content):
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip()
            
            if key == "target_coverage":
                value = float(value.rstrip('%')) / 100 if '%' in value else float(value)
            elif key == "framework":
                value = value.lower()
            
            result[key] = value
        
        return result
    
    def _parse_agent_section(self, content: str) -> Dict[str, Any]:
        """Parse Agent Preferences section."""
        result: Dict[str, Any] = {}
        
        for match in self.KEY_VALUE_PATTERN.finditer(content):
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip()
            
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif self.ARRAY_PATTERN.match(value):
                array_match = self.ARRAY_PATTERN.match(value)
                if array_match:
                    items = array_match.group(1).split(',')
                    value = [item.strip().strip('"\'') for item in items]
            elif value.isdigit():
                value = int(value)
            
            result[key] = value
        
        return result
    
    def _parse_style_section(self, content: str) -> Dict[str, Any]:
        """Parse Code Style section."""
        result: Dict[str, Any] = {}
        
        for match in self.KEY_VALUE_PATTERN.finditer(content):
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip()
            
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            
            result[key] = value
        
        return result
    
    def _parse_dependencies_section(self, content: str) -> Dict[str, Any]:
        """Parse Dependencies section."""
        result: Dict[str, Any] = {}
        
        for match in self.KEY_VALUE_PATTERN.finditer(content):
            key = match.group(1)
            value = match.group(2).strip()
            
            if value.lower() == "enabled":
                result[key] = {"version": "", "enabled": True}
            elif value.lower() == "disabled":
                result[key] = {"version": "", "enabled": False}
            else:
                result[key] = {"version": value, "enabled": True}
        
        return result
    
    def _parse_list_section(self, content: str) -> List[str]:
        """Parse a simple list section."""
        result: List[str] = []
        
        for match in self.LIST_ITEM_PATTERN.finditer(content):
            result.append(match.group(1).strip())
        
        return result
    
    def _parse_generic_section(self, content: str) -> Dict[str, Any]:
        """Parse a generic section."""
        result: Dict[str, Any] = {}
        
        for match in self.KEY_VALUE_PATTERN.finditer(content):
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip()
            result[key] = value
        
        list_items = self._parse_list_section(content)
        if list_items and not result:
            return {"items": list_items}
        
        return result


class ProjectConfigLoader:
    """Loader for project configuration."""
    
    CONFIG_FILENAMES = ["PYUT.md", ".pyut.md", "pyut.md"]
    
    def __init__(self):
        self.parser = PYUTMdParser()
        self.logger = logging.getLogger(__name__)
    
    def find_config_file(self, project_path: Path) -> Optional[Path]:
        """Find PYUT.md file in project directory.
        
        Args:
            project_path: Project root directory
            
        Returns:
            Path to config file or None
        """
        for filename in self.CONFIG_FILENAMES:
            config_path = project_path / filename
            if config_path.exists():
                return config_path
        
        return None
    
    def load(self, project_path: Path) -> ProjectConfig:
        """Load project configuration.
        
        Args:
            project_path: Project root directory
            
        Returns:
            ProjectConfig instance
        """
        config_path = self.find_config_file(project_path)
        
        if config_path:
            self.logger.info(f"[ProjectConfig] Loading config from: {config_path}")
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                data = self.parser.parse(content)
                config = ProjectConfig.from_dict(data, project_path)
                config.project_name = project_path.name
                
                self.logger.info(f"[ProjectConfig] Loaded configuration for: {config.project_name}")
                return config
                
            except Exception as e:
                self.logger.warning(f"[ProjectConfig] Failed to load config: {e}")
        
        self.logger.info("[ProjectConfig] No config file found, using defaults")
        return ProjectConfig(project_root=project_path, project_name=project_path.name)
    
    def create_template(self, output_path: Path) -> None:
        """Create a template PYUT.md file.
        
        Args:
            output_path: Path to create the template
        """
        template = '''# Project Configuration

This file configures the PyUT Agent for this project.

## Build
- Tool: Maven
- Java Version: 17
- Build Command: mvn clean install -DskipTests
- Test Command: mvn test
- Compile Command: mvn compile

## Testing
- Framework: junit5
- Target Coverage: 80%
- Test Directory: src/test/java
- Test Suffix: Test
- Generate Integration Tests: false
- Mock Framework: mockito

## Agent Preferences
- Enable Multi-Agent: false
- Enable Error Prediction: true
- Enable Self Reflection: true
- Enable Pattern Library: true
- Enable Chain Of Thought: true
- Max Iterations: 10
- Preferred Strategies: [boundary, positive, negative]
- Timeout Per File: 300
- Parallel Workers: 1

## Code Style
- Naming Convention: camelCase
- Max Line Length: 120
- Indent Size: 4
- Use Lombok: false
- Generate Javadoc: false

## Dependencies
- Spring Boot: 3.0.0
- Lombok: enabled

## Custom Instructions
- Always use @DisplayName annotations
- Generate tests for edge cases
- Use AssertJ assertions

## Ignore
- **/generated/**
- **/target/**
- **/*_Test.java
'''
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)
        
        self.logger.info(f"[ProjectConfig] Created template at: {output_path}")


_global_loader: Optional[ProjectConfigLoader] = None


def get_project_config_loader() -> ProjectConfigLoader:
    """Get global project config loader."""
    global _global_loader
    if _global_loader is None:
        _global_loader = ProjectConfigLoader()
    return _global_loader


def load_project_config(project_path: Path) -> ProjectConfig:
    """Load project configuration.
    
    Args:
        project_path: Project root directory
        
    Returns:
        ProjectConfig instance
    """
    return get_project_config_loader().load(project_path)


def create_config_template(output_path: Path) -> None:
    """Create a template PYUT.md file.
    
    Args:
        output_path: Path to create the template
    """
    get_project_config_loader().create_template(output_path)


@dataclass
class ProjectContext:
    """Project context for test generation."""
    project_path: Path = None
    project_name: str = ""
    build_config: BuildConfig = field(default_factory=BuildConfig)
    test_config: TestConfig = field(default_factory=TestConfig)
    agent_preferences: AgentPreferences = field(default_factory=AgentPreferences)
    code_style: CodeStyle = field(default_factory=CodeStyle)
    dependencies: Dict[str, DependencyInfo] = field(default_factory=dict)
    custom_instructions: List[str] = field(default_factory=list)
    
    @classmethod
    def from_project_config(cls, config: ProjectConfig) -> "ProjectContext":
        """Create ProjectContext from ProjectConfig."""
        return cls(
            project_path=config.project_root,
            project_name=config.project_name,
            build_config=config.build,
            test_config=config.testing,
            agent_preferences=config.agent,
            code_style=config.code_style,
            dependencies=config.dependencies,
            custom_instructions=config.custom_instructions
        )


class ProjectConfigManager:
    """Manager for project configuration."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self._config: Optional[ProjectConfig] = None
        self._context: Optional[ProjectContext] = None
    
    def load_config(self) -> ProjectConfig:
        """Load project configuration."""
        if self._config is None:
            self._config = load_project_config(self.project_path)
        return self._config
    
    def load_context(self) -> Optional[ProjectContext]:
        """Load project context."""
        if self._context is None:
            config = self.load_config()
            self._context = ProjectContext.from_project_config(config)
        return self._context
    
    def get_test_framework(self) -> TestFramework:
        """Get test framework."""
        config = self.load_config()
        return config.testing.framework
    
    def get_mock_framework(self) -> MockFramework:
        """Get mock framework."""
        config = self.load_config()
        mock_name = config.testing.mock_framework.lower()
        try:
            return MockFramework(mock_name)
        except ValueError:
            return MockFramework.MOCKITO
    
    def get_build_command(self) -> str:
        """Get build command."""
        config = self.load_config()
        return config.build.build_command
    
    def get_test_command(self) -> str:
        """Get test command."""
        config = self.load_config()
        return config.build.test_command
