"""Project Configuration Management.

This module provides project-level configuration similar to CLAUDE.md:
- ProjectContext: Project information and preferences
- BuildCommands: Build tool commands
- TestPreferences: Test generation preferences
- ProjectConfigManager: Load/save/manage project configuration
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

logger = logging.getLogger(__name__)


@dataclass
class BuildCommands:
    """Build tool commands configuration."""
    
    build: str = "mvn compile"
    test: str = "mvn test"
    test_single: str = "mvn test -Dtest={test_class}"
    coverage: str = "mvn jacoco:report"
    clean: str = "mvn clean"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "build": self.build,
            "test": self.test,
            "test_single": self.test_single,
            "coverage": self.coverage,
            "clean": self.clean,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "BuildCommands":
        """Create from dictionary."""
        return cls(
            build=data.get("build", "mvn compile"),
            test=data.get("test", "mvn test"),
            test_single=data.get("test_single", "mvn test -Dtest={test_class}"),
            coverage=data.get("coverage", "mvn jacoco:report"),
            clean=data.get("clean", "mvn clean"),
        )


@dataclass
class TestPreferences:
    """Test generation preferences."""
    
    framework: str = "junit5"
    mock_framework: str = "mockito"
    target_coverage: float = 0.8
    include_positive_cases: bool = True
    include_negative_cases: bool = True
    include_edge_cases: bool = True
    use_descriptive_names: bool = True
    add_javadoc: bool = True
    max_methods_per_test: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "framework": self.framework,
            "mock_framework": self.mock_framework,
            "target_coverage": self.target_coverage,
            "include_positive_cases": self.include_positive_cases,
            "include_negative_cases": self.include_negative_cases,
            "include_edge_cases": self.include_edge_cases,
            "use_descriptive_names": self.use_descriptive_names,
            "add_javadoc": self.add_javadoc,
            "max_methods_per_test": self.max_methods_per_test,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestPreferences":
        """Create from dictionary."""
        return cls(
            framework=data.get("framework", "junit5"),
            mock_framework=data.get("mock_framework", "mockito"),
            target_coverage=data.get("target_coverage", 0.8),
            include_positive_cases=data.get("include_positive_cases", True),
            include_negative_cases=data.get("include_negative_cases", True),
            include_edge_cases=data.get("include_edge_cases", True),
            use_descriptive_names=data.get("use_descriptive_names", True),
            add_javadoc=data.get("add_javadoc", True),
            max_methods_per_test=data.get("max_methods_per_test", 5),
        )


@dataclass
class ProjectContext:
    """Project-level configuration context.
    
    Similar to CLAUDE.md, this stores project-specific information
    that helps the agent understand and work with the codebase.
    """
    
    name: str = ""
    language: str = "java"
    build_tool: str = "maven"
    java_version: str = "17"
    
    architecture: str = ""
    key_modules: List[str] = field(default_factory=list)
    coding_standards: List[str] = field(default_factory=list)
    
    build_commands: BuildCommands = field(default_factory=BuildCommands)
    test_preferences: TestPreferences = field(default_factory=TestPreferences)
    
    custom_instructions: str = ""
    custom_hooks: Dict[str, List[str]] = field(default_factory=dict)
    
    source_path: str = "src/main/java"
    test_path: str = "src/test/java"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "language": self.language,
            "build_tool": self.build_tool,
            "java_version": self.java_version,
            "architecture": self.architecture,
            "key_modules": self.key_modules,
            "coding_standards": self.coding_standards,
            "build_commands": self.build_commands.to_dict(),
            "test_preferences": self.test_preferences.to_dict(),
            "custom_instructions": self.custom_instructions,
            "custom_hooks": self.custom_hooks,
            "source_path": self.source_path,
            "test_path": self.test_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectContext":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            language=data.get("language", "java"),
            build_tool=data.get("build_tool", "maven"),
            java_version=data.get("java_version", "17"),
            architecture=data.get("architecture", ""),
            key_modules=data.get("key_modules", []),
            coding_standards=data.get("coding_standards", []),
            build_commands=BuildCommands.from_dict(data.get("build_commands", {})),
            test_preferences=TestPreferences.from_dict(data.get("test_preferences", {})),
            custom_instructions=data.get("custom_instructions", ""),
            custom_hooks=data.get("custom_hooks", {}),
            source_path=data.get("source_path", "src/main/java"),
            test_path=data.get("test_path", "src/test/java"),
        )
    
    def get_prompt_context(self) -> str:
        """Get context string for LLM prompts.
        
        Returns:
            Formatted context string
        """
        lines = []
        
        if self.name:
            lines.append(f"Project: {self.name}")
        
        lines.append(f"Language: {self.language}")
        lines.append(f"Build Tool: {self.build_tool}")
        lines.append(f"Java Version: {self.java_version}")
        
        if self.architecture:
            lines.append(f"\nArchitecture: {self.architecture}")
        
        if self.key_modules:
            lines.append(f"\nKey Modules: {', '.join(self.key_modules)}")
        
        if self.coding_standards:
            lines.append("\nCoding Standards:")
            for standard in self.coding_standards:
                lines.append(f"- {standard}")
        
        if self.custom_instructions:
            lines.append(f"\nCustom Instructions:\n{self.custom_instructions}")
        
        return "\n".join(lines)


class ProjectConfigManager:
    """Manages project-level configuration.
    
    Features:
    - Load/save PYUT.md configuration
    - Auto-detect project settings
    - Merge with user-level configuration
    - Validate configuration
    """
    
    CONFIG_FILENAME = "PYUT.md"
    CONFIG_JSON = ".pyutagent/config.json"
    
    def __init__(self, project_root: Path):
        """Initialize project config manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self._context: Optional[ProjectContext] = None
    
    @property
    def config_path(self) -> Path:
        """Get the PYUT.md config file path."""
        return self.project_root / self.CONFIG_FILENAME
    
    @property
    def json_config_path(self) -> Path:
        """Get the JSON config file path."""
        return self.project_root / self.CONFIG_JSON
    
    def exists(self) -> bool:
        """Check if project config exists.
        
        Returns:
            True if PYUT.md or config.json exists
        """
        return self.config_path.exists() or self.json_config_path.exists()
    
    def load(self) -> ProjectContext:
        """Load project configuration.
        
        Priority:
        1. PYUT.md (markdown format)
        2. .pyutagent/config.json (JSON format)
        3. Auto-detect from project
        
        Returns:
            ProjectContext instance
        """
        if self._context:
            return self._context
        
        if self.config_path.exists():
            self._context = self._load_markdown_config()
        elif self.json_config_path.exists():
            self._context = self._load_json_config()
        else:
            self._context = self._auto_detect()
        
        return self._context
    
    def _load_markdown_config(self) -> ProjectContext:
        """Load configuration from PYUT.md file.
        
        Returns:
            ProjectContext from markdown
        """
        content = self.config_path.read_text(encoding="utf-8")
        return self._parse_markdown(content)
    
    def _parse_markdown(self, content: str) -> ProjectContext:
        """Parse PYUT.md markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            ProjectContext instance
        """
        context = ProjectContext()
        
        context.name = self._extract_project_name(content)
        
        build_tool = self._extract_field(content, "Build Tool")
        if build_tool:
            context.build_tool = build_tool.lower()
        
        java_version = self._extract_field(content, "Java Version")
        if java_version:
            context.java_version = java_version
        
        test_framework = self._extract_field(content, "Test Framework")
        if test_framework:
            context.test_preferences.framework = test_framework.lower()
        
        mock_framework = self._extract_field(content, "Mock Framework")
        if mock_framework:
            context.test_preferences.mock_framework = mock_framework.lower()
        
        context.architecture = self._extract_section(content, "Architecture")
        
        key_modules = self._extract_list(content, "Key Modules")
        if key_modules:
            context.key_modules = key_modules
        
        coding_standards = self._extract_list(content, "Coding Standards")
        if coding_standards:
            context.coding_standards = coding_standards
        
        build_commands = self._extract_code_block(content, "Build Commands")
        if build_commands:
            context.build_commands = self._parse_build_commands(build_commands)
        
        context.custom_instructions = self._extract_section(content, "Custom Instructions")
        
        return context
    
    def _extract_project_name(self, content: str) -> str:
        """Extract project name from markdown.
        
        Args:
            content: Markdown content
            
        Returns:
            Project name or empty string
        """
        match = re.search(r"^#\s+(.+?)(?:\s+Configuration)?$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        match = re.search(r"^\*\*Name\*\*:\s*(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        return self.project_root.name
    
    def _extract_field(self, content: str, field_name: str) -> Optional[str]:
        """Extract a field value from markdown.
        
        Args:
            content: Markdown content
            field_name: Name of the field
            
        Returns:
            Field value or None
        """
        patterns = [
            rf"^\*\*{field_name}\*\*:\s*(.+)$",
            rf"^- \*\*{field_name}\*\*:\s*(.+)$",
            rf"^- {field_name}:\s*(.+)$",
            rf"^{field_name}:\s*(.+)$",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a section from markdown.
        
        Args:
            content: Markdown content
            section_name: Name of the section
            
        Returns:
            Section content or empty string
        """
        pattern = rf"^##\s+{section_name}\s*\n(.*?)(?=^##\s|\Z)"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _extract_list(self, content: str, list_name: str) -> List[str]:
        """Extract a list from markdown.
        
        Args:
            content: Markdown content
            list_name: Name of the list
            
        Returns:
            List of items
        """
        section = self._extract_section(content, list_name)
        if not section:
            return []
        
        items = []
        for line in section.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].strip())
        
        return items
    
    def _extract_code_block(self, content: str, block_name: str) -> Optional[str]:
        """Extract a code block from markdown.
        
        Args:
            content: Markdown content
            block_name: Name of the code block
            
        Returns:
            Code block content or None
        """
        pattern = rf"###\s+{block_name}\s*\n```bash\n(.*?)```"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    def _parse_build_commands(self, content: str) -> BuildCommands:
        """Parse build commands from markdown.
        
        Args:
            content: Build commands content
            
        Returns:
            BuildCommands instance
        """
        commands = BuildCommands()
        
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                continue
            
            if "compile" in line or "install" in line or "build" in line.lower():
                if "test" not in line.lower() or "skipTests" in line:
                    commands.build = line
                    continue
            
            if "mvn test" in line or "gradle test" in line:
                if "-Dtest=" in line:
                    commands.test_single = line
                else:
                    commands.test = line
                continue
            
            if "jacoco" in line or "coverage" in line.lower():
                commands.coverage = line
                continue
            
            if "mvn clean" in line or "gradle clean" in line:
                commands.clean = line
                continue
        
        return commands
    
    def _load_json_config(self) -> ProjectContext:
        """Load configuration from JSON file.
        
        Returns:
            ProjectContext from JSON
        """
        with open(self.json_config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return ProjectContext.from_dict(data)
    
    def _auto_detect(self) -> ProjectContext:
        """Auto-detect project settings.
        
        Returns:
            ProjectContext with detected settings
        """
        context = ProjectContext(name=self.project_root.name)
        
        context.build_tool = self._detect_build_tool()
        context.java_version = self._detect_java_version()
        context.test_preferences.framework = self._detect_test_framework()
        context.test_preferences.mock_framework = self._detect_mock_framework()
        
        context.source_path = self._detect_source_path()
        context.test_path = self._detect_test_path()
        
        return context
    
    def _detect_build_tool(self) -> str:
        """Detect build tool from project files.
        
        Returns:
            Build tool name (maven, gradle, or unknown)
        """
        if (self.project_root / "pom.xml").exists():
            return "maven"
        elif (self.project_root / "build.gradle").exists() or (self.project_root / "build.gradle.kts").exists():
            return "gradle"
        
        return "maven"
    
    def _detect_java_version(self) -> str:
        """Detect Java version from project configuration.
        
        Returns:
            Java version string
        """
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            content = pom_path.read_text(encoding="utf-8")
            match = re.search(r"<java\.version>(\d+)</java\.version>", content)
            if match:
                return match.group(1)
            
            match = re.search(r"<source>(\d+)</source>", content)
            if match:
                return match.group(1)
        
        gradle_path = self.project_root / "build.gradle"
        if gradle_path.exists():
            content = gradle_path.read_text(encoding="utf-8")
            match = re.search(r"sourceCompatibility\s*=\s*['\"]?(\d+)", content)
            if match:
                return match.group(1)
        
        return "17"
    
    def _detect_test_framework(self) -> str:
        """Detect test framework from project dependencies.
        
        Returns:
            Test framework name
        """
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            content = pom_path.read_text(encoding="utf-8").lower()
            if "junit-jupiter" in content or "junit5" in content:
                return "junit5"
            if "testng" in content:
                return "testng"
            if "junit" in content:
                return "junit4"
        
        return "junit5"
    
    def _detect_mock_framework(self) -> str:
        """Detect mock framework from project dependencies.
        
        Returns:
            Mock framework name
        """
        pom_path = self.project_root / "pom.xml"
        if pom_path.exists():
            content = pom_path.read_text(encoding="utf-8").lower()
            if "mockito" in content:
                return "mockito"
            if "easymock" in content:
                return "easymock"
            if "powermock" in content:
                return "powermock"
        
        return "mockito"
    
    def _detect_source_path(self) -> str:
        """Detect source code path.
        
        Returns:
            Source path
        """
        common_paths = [
            "src/main/java",
            "src/java",
            "src",
        ]
        
        for path in common_paths:
            if (self.project_root / path).exists():
                return path
        
        return "src/main/java"
    
    def _detect_test_path(self) -> str:
        """Detect test code path.
        
        Returns:
            Test path
        """
        common_paths = [
            "src/test/java",
            "src/test",
            "test",
        ]
        
        for path in common_paths:
            if (self.project_root / path).exists():
                return path
        
        return "src/test/java"
    
    def save(self, context: Optional[ProjectContext] = None) -> None:
        """Save project configuration.
        
        Args:
            context: ProjectContext to save (uses loaded context if None)
        """
        if context:
            self._context = context
        
        if not self._context:
            return
        
        self.json_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.json_config_path, "w", encoding="utf-8") as f:
            json.dump(self._context.to_dict(), f, indent=2)
        
        logger.info(f"Saved project config to {self.json_config_path}")
    
    def init_config(self, force: bool = False) -> bool:
        """Initialize project configuration file.
        
        Creates a PYUT.md file with auto-detected settings.
        
        Args:
            force: Overwrite existing config
            
        Returns:
            True if config was created
        """
        if self.exists() and not force:
            logger.info("Project config already exists")
            return False
        
        context = self._auto_detect()
        
        content = self._generate_markdown(context)
        
        self.config_path.write_text(content, encoding="utf-8")
        
        self._context = context
        
        logger.info(f"Created project config at {self.config_path}")
        return True
    
    def _generate_markdown(self, context: ProjectContext) -> str:
        """Generate PYUT.md content.
        
        Args:
            context: ProjectContext to use
            
        Returns:
            Markdown content
        """
        return f"""# {context.name or 'Project'} Configuration

## Project Information

- **Name**: {context.name}
- **Language**: {context.language}
- **Build Tool**: {context.build_tool}
- **Java Version**: {context.java_version}
- **Test Framework**: {context.test_preferences.framework}
- **Mock Framework**: {context.test_preferences.mock_framework}

## Architecture

{context.architecture or 'Standard Maven project structure with layered architecture.'}

## Key Modules

{self._format_list(context.key_modules or ['service', 'repository', 'controller', 'util'])}

## Build Commands

```bash
# Build the project
{context.build_commands.build}

# Run all tests
{context.build_commands.test}

# Run single test class
{context.build_commands.test_single}

# Generate coverage report
{context.build_commands.coverage}
```

## Coding Standards

{self._format_list(context.coding_standards or [
    'Follow existing code style in the project',
    'Use meaningful variable and method names',
    'Add Javadoc for public APIs',
    'Keep methods focused and small',
    'Write unit tests for new functionality',
])}

## Test Generation Preferences

- **Framework**: {context.test_preferences.framework}
- **Mock Framework**: {context.test_preferences.mock_framework}
- **Target Coverage**: {context.test_preferences.target_coverage:.0%}
- **Include Positive Cases**: {context.test_preferences.include_positive_cases}
- **Include Negative Cases**: {context.test_preferences.include_negative_cases}

## Custom Instructions

{context.custom_instructions or 'When working with this codebase, prioritize readability over cleverness. Ask clarifying questions when requirements are ambiguous.'}
"""
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list for markdown.
        
        Args:
            items: List items
            
        Returns:
            Formatted string
        """
        return "\n".join(f"- {item}" for item in items)
    
    def get_context(self) -> ProjectContext:
        """Get the current project context.
        
        Returns:
            ProjectContext instance
        """
        return self._context or self.load()
    
    def update(self, **kwargs: Any) -> None:
        """Update project context fields.
        
        Args:
            **kwargs: Field values to update
        """
        context = self.get_context()
        
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        self.save(context)
