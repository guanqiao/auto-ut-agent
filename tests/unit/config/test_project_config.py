"""Tests for Project Configuration."""

import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile
import os

from pyutagent.config.project_config import (
    ProjectContext,
    BuildCommands,
    TestPreferences,
    ProjectConfigManager,
)


class TestBuildCommands:
    """Tests for BuildCommands."""
    
    def test_default_values(self):
        """Test default build commands."""
        commands = BuildCommands()
        
        assert commands.build == "mvn compile"
        assert commands.test == "mvn test"
        assert "test" in commands.test_single
        assert "jacoco" in commands.coverage
    
    def test_to_dict(self):
        """Test serialization."""
        commands = BuildCommands(
            build="gradle build",
            test="gradle test",
        )
        
        result = commands.to_dict()
        
        assert result["build"] == "gradle build"
        assert result["test"] == "gradle test"
    
    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "build": "mvn compile -DskipTests",
            "test": "mvn test -Pintegration",
        }
        
        commands = BuildCommands.from_dict(data)
        
        assert commands.build == "mvn compile -DskipTests"
        assert commands.test == "mvn test -Pintegration"


class TestTestPreferences:
    """Tests for TestPreferences."""
    
    def test_default_values(self):
        """Test default test preferences."""
        prefs = TestPreferences()
        
        assert prefs.framework == "junit5"
        assert prefs.mock_framework == "mockito"
        assert prefs.target_coverage == 0.8
    
    def test_to_dict(self):
        """Test serialization."""
        prefs = TestPreferences(
            framework="testng",
            target_coverage=0.9,
        )
        
        result = prefs.to_dict()
        
        assert result["framework"] == "testng"
        assert result["target_coverage"] == 0.9
    
    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "framework": "junit4",
            "target_coverage": 0.75,
            "include_edge_cases": False,
        }
        
        prefs = TestPreferences.from_dict(data)
        
        assert prefs.framework == "junit4"
        assert prefs.target_coverage == 0.75
        assert prefs.include_edge_cases is False


class TestProjectContext:
    """Tests for ProjectContext."""
    
    def test_default_values(self):
        """Test default project context."""
        context = ProjectContext()
        
        assert context.name == ""
        assert context.language == "java"
        assert context.build_tool == "maven"
        assert context.java_version == "17"
    
    def test_to_dict(self):
        """Test serialization."""
        context = ProjectContext(
            name="test-project",
            build_tool="gradle",
            java_version="21",
        )
        
        result = context.to_dict()
        
        assert result["name"] == "test-project"
        assert result["build_tool"] == "gradle"
        assert result["java_version"] == "21"
    
    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "name": "my-project",
            "build_tool": "maven",
            "java_version": "11",
            "key_modules": ["service", "repository"],
        }
        
        context = ProjectContext.from_dict(data)
        
        assert context.name == "my-project"
        assert context.java_version == "11"
        assert context.key_modules == ["service", "repository"]
    
    def test_get_prompt_context(self):
        """Test prompt context generation."""
        context = ProjectContext(
            name="TestProject",
            build_tool="maven",
            java_version="17",
            architecture="Layered architecture",
            key_modules=["service", "repository"],
        )
        
        prompt = context.get_prompt_context()
        
        assert "TestProject" in prompt
        assert "maven" in prompt
        assert "17" in prompt
        assert "Layered architecture" in prompt


class TestProjectConfigManager:
    """Tests for ProjectConfigManager."""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project directory."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()
        return project_dir
    
    @pytest.fixture
    def manager(self, temp_project):
        """Create a project config manager."""
        return ProjectConfigManager(temp_project)
    
    def test_init(self, manager, temp_project):
        """Test manager initialization."""
        assert manager.project_root == temp_project
        assert manager.config_path == temp_project / "PYUT.md"
    
    def test_exists_false(self, manager):
        """Test exists returns False when no config."""
        assert manager.exists() is False
    
    def test_exists_true_markdown(self, manager):
        """Test exists returns True with PYUT.md."""
        manager.config_path.write_text("# Test Project")
        assert manager.exists() is True
    
    def test_exists_true_json(self, manager):
        """Test exists returns True with config.json."""
        manager.json_config_path.parent.mkdir(parents=True, exist_ok=True)
        manager.json_config_path.write_text("{}")
        assert manager.exists() is True
    
    def test_auto_detect_maven(self, manager, temp_project):
        """Test auto-detection of Maven project."""
        (temp_project / "pom.xml").write_text("<project></project>")
        
        context = manager._auto_detect()
        
        assert context.build_tool == "maven"
    
    def test_auto_detect_gradle(self, manager, temp_project):
        """Test auto-detection of Gradle project."""
        (temp_project / "build.gradle").write_text("plugins { id 'java' }")
        
        context = manager._auto_detect()
        
        assert context.build_tool == "gradle"
    
    def test_auto_detect_java_version(self, manager, temp_project):
        """Test auto-detection of Java version."""
        pom_content = """
        <project>
            <properties>
                <java.version>11</java.version>
            </properties>
        </project>
        """
        (temp_project / "pom.xml").write_text(pom_content)
        
        context = manager._auto_detect()
        
        assert context.java_version == "11"
    
    def test_auto_detect_test_framework_junit5(self, manager, temp_project):
        """Test auto-detection of JUnit 5."""
        pom_content = """
        <project>
            <dependency>
                <groupId>org.junit.jupiter</groupId>
                <artifactId>junit-jupiter</artifactId>
            </dependency>
        </project>
        """
        (temp_project / "pom.xml").write_text(pom_content)
        
        context = manager._auto_detect()
        
        assert context.test_preferences.framework == "junit5"
    
    def test_auto_detect_test_framework_testng(self, manager, temp_project):
        """Test auto-detection of TestNG."""
        pom_content = """
        <project>
            <dependency>
                <groupId>org.testng</groupId>
                <artifactId>testng</artifactId>
            </dependency>
        </project>
        """
        (temp_project / "pom.xml").write_text(pom_content)
        
        context = manager._auto_detect()
        
        assert context.test_preferences.framework == "testng"
    
    def test_load_auto_detect(self, manager, temp_project):
        """Test loading with auto-detection."""
        (temp_project / "pom.xml").write_text("<project></project>")
        
        context = manager.load()
        
        assert context is not None
        assert context.build_tool == "maven"
    
    def test_load_json_config(self, manager, temp_project):
        """Test loading from JSON config."""
        manager.json_config_path.parent.mkdir(parents=True, exist_ok=True)
        manager.json_config_path.write_text('{"name": "json-project", "build_tool": "gradle"}')
        
        context = manager.load()
        
        assert context.name == "json-project"
        assert context.build_tool == "gradle"
    
    def test_load_markdown_config(self, manager, temp_project):
        """Test loading from PYUT.md config."""
        markdown_content = """# TestProject Configuration

## Project Information

- **Name**: TestProject
- **Build Tool**: gradle
- **Java Version**: 21
- **Test Framework**: testng

## Architecture

Microservices architecture with REST APIs.

## Key Modules

- api-gateway
- user-service
- order-service
"""
        manager.config_path.write_text(markdown_content)
        
        context = manager.load()
        
        assert context.name == "TestProject"
        assert context.build_tool == "gradle"
        assert context.java_version == "21"
        assert context.test_preferences.framework == "testng"
        assert "Microservices" in context.architecture
        assert "api-gateway" in context.key_modules
    
    def test_save_json(self, manager, temp_project):
        """Test saving to JSON config."""
        context = ProjectContext(
            name="save-test",
            build_tool="maven",
        )
        
        manager.save(context)
        
        assert manager.json_config_path.exists()
        
        loaded = manager._load_json_config()
        assert loaded.name == "save-test"
    
    def test_init_config(self, manager, temp_project):
        """Test initializing config file."""
        (temp_project / "pom.xml").write_text("<project></project>")
        
        result = manager.init_config()
        
        assert result is True
        assert manager.config_path.exists()
    
    def test_init_config_already_exists(self, manager, temp_project):
        """Test init config when already exists."""
        manager.config_path.write_text("# Existing")
        
        result = manager.init_config(force=False)
        
        assert result is False
    
    def test_init_config_force(self, manager, temp_project):
        """Test init config with force overwrite."""
        manager.config_path.write_text("# Existing")
        (temp_project / "pom.xml").write_text("<project></project>")
        
        result = manager.init_config(force=True)
        
        assert result is True
    
    def test_update(self, manager, temp_project):
        """Test updating context fields."""
        (temp_project / "pom.xml").write_text("<project></project>")
        manager.load()
        
        manager.update(name="updated-name", java_version="11")
        
        assert manager._context.name == "updated-name"
        assert manager._context.java_version == "11"
    
    def test_parse_build_commands(self, manager):
        """Test parsing build commands from markdown."""
        content = """
# Build the project
mvn clean install -DskipTests

# Run all tests
mvn test

# Run single test class
mvn test -Dtest=MyTest

# Generate coverage report
mvn jacoco:report

# Clean
mvn clean
"""
        commands = manager._parse_build_commands(content)
        
        assert "mvn clean install" in commands.build
        assert commands.test == "mvn test"
        assert "MyTest" in commands.test_single
        assert "jacoco" in commands.coverage
        assert commands.clean == "mvn clean"
    
    def test_detect_source_path(self, manager, temp_project):
        """Test detecting source path."""
        src_path = temp_project / "src" / "main" / "java"
        src_path.mkdir(parents=True)
        
        path = manager._detect_source_path()
        
        assert path == "src/main/java"
    
    def test_detect_test_path(self, manager, temp_project):
        """Test detecting test path."""
        test_path = temp_project / "src" / "test" / "java"
        test_path.mkdir(parents=True)
        
        path = manager._detect_test_path()
        
        assert path == "src/test/java"
