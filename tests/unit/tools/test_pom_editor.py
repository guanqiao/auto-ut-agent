"""Unit tests for PomEditor."""

import pytest
import shutil
import tempfile
from pathlib import Path

from pyutagent.tools.pom_editor import PomEditor


@pytest.fixture
def temp_project():
    """Create a temporary Maven project for testing."""
    temp_dir = tempfile.mkdtemp()
    project_path = Path(temp_dir) / "test_project"
    project_path.mkdir()
    
    pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
    
    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>
    
    <dependencies>
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-api</artifactId>
            <version>2.0.9</version>
        </dependency>
    </dependencies>
</project>
"""
    
    pom_path = project_path / "pom.xml"
    pom_path.write_text(pom_content, encoding='utf-8')
    
    yield project_path
    
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPomEditor:
    """Test cases for PomEditor."""
    
    def test_init(self, temp_project):
        """Test PomEditor initialization."""
        editor = PomEditor(str(temp_project))
        
        assert editor.project_path == temp_project
        assert editor.pom_path == temp_project / "pom.xml"
        assert editor.pom_path.exists()
    
    def test_read_pom(self, temp_project):
        """Test reading pom.xml."""
        editor = PomEditor(str(temp_project))
        
        content = editor.read_pom()
        
        assert "com.example" in content
        assert "test-project" in content
        assert "slf4j-api" in content
    
    def test_backup_pom(self, temp_project):
        """Test backing up pom.xml."""
        editor = PomEditor(str(temp_project))
        
        backup_path = editor.backup_pom(label="test")
        
        assert Path(backup_path).exists()
        assert "pom_" in backup_path
        assert "_test.xml" in backup_path
    
    def test_has_dependency_true(self, temp_project):
        """Test checking if dependency exists (true case)."""
        editor = PomEditor(str(temp_project))
        
        has_dep = editor.has_dependency("org.slf4j", "slf4j-api")
        
        assert has_dep is True
    
    def test_has_dependency_false(self, temp_project):
        """Test checking if dependency exists (false case)."""
        editor = PomEditor(str(temp_project))
        
        has_dep = editor.has_dependency("org.junit.jupiter", "junit-jupiter")
        
        assert has_dep is False
    
    def test_add_dependency(self, temp_project):
        """Test adding a dependency."""
        editor = PomEditor(str(temp_project))
        
        dependency = {
            "group_id": "org.junit.jupiter",
            "artifact_id": "junit-jupiter",
            "version": "5.10.0",
            "scope": "test"
        }
        
        success, message = editor.add_dependency(dependency, backup=False)
        
        assert success is True
        assert "Successfully added" in message
        assert editor.has_dependency("org.junit.jupiter", "junit-jupiter")
    
    def test_add_dependency_already_exists(self, temp_project):
        """Test adding a dependency that already exists."""
        editor = PomEditor(str(temp_project))
        
        dependency = {
            "group_id": "org.slf4j",
            "artifact_id": "slf4j-api",
            "version": "2.0.9"
        }
        
        success, message = editor.add_dependency(dependency, backup=False)
        
        assert success is False
        assert "already exists" in message
    
    def test_add_dependency_missing_required_field(self, temp_project):
        """Test adding a dependency with missing required field."""
        editor = PomEditor(str(temp_project))
        
        dependency = {
            "group_id": "org.junit.jupiter",
            "artifact_id": "junit-jupiter"
        }
        
        success, message = editor.add_dependency(dependency, backup=False)
        
        assert success is False
        assert "Missing required key" in message
    
    def test_add_dependencies_batch(self, temp_project):
        """Test adding multiple dependencies."""
        editor = PomEditor(str(temp_project))
        
        dependencies = [
            {
                "group_id": "org.junit.jupiter",
                "artifact_id": "junit-jupiter",
                "version": "5.10.0",
                "scope": "test"
            },
            {
                "group_id": "org.mockito",
                "artifact_id": "mockito-core",
                "version": "5.8.0",
                "scope": "test"
            }
        ]
        
        success, messages = editor.add_dependencies(dependencies, backup=False)
        
        assert success is True
        assert editor.has_dependency("org.junit.jupiter", "junit-jupiter")
        assert editor.has_dependency("org.mockito", "mockito-core")
    
    def test_remove_dependency(self, temp_project):
        """Test removing a dependency."""
        editor = PomEditor(str(temp_project))
        
        dependency = {
            "group_id": "org.junit.jupiter",
            "artifact_id": "junit-jupiter",
            "version": "5.10.0",
            "scope": "test"
        }
        editor.add_dependency(dependency, backup=False)
        
        success, message = editor.remove_dependency(
            "org.junit.jupiter", 
            "junit-jupiter",
            backup=False
        )
        
        assert success is True
        assert "Successfully removed" in message
        assert not editor.has_dependency("org.junit.jupiter", "junit-jupiter")
    
    def test_get_dependencies(self, temp_project):
        """Test getting all dependencies."""
        editor = PomEditor(str(temp_project))
        
        deps = editor.get_dependencies()
        
        assert len(deps) >= 1
        assert any(d.get('groupId') == 'org.slf4j' for d in deps)
    
    def test_validate_pom_valid(self, temp_project):
        """Test validating a valid pom.xml."""
        editor = PomEditor(str(temp_project))
        
        is_valid, errors = editor.validate_pom()
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_format_dependency_xml(self, temp_project):
        """Test formatting a dependency as XML."""
        editor = PomEditor(str(temp_project))
        
        dependency = {
            "group_id": "org.junit.jupiter",
            "artifact_id": "junit-jupiter",
            "version": "5.10.0",
            "scope": "test"
        }
        
        xml_str = editor.format_dependency_xml(dependency)
        
        assert "<dependency>" in xml_str
        assert "<groupId>org.junit.jupiter</groupId>" in xml_str
        assert "<artifactId>junit-jupiter</artifactId>" in xml_str
        assert "<version>5.10.0</version>" in xml_str
        assert "<scope>test</scope>" in xml_str
    
    def test_list_backups(self, temp_project):
        """Test listing backups."""
        editor = PomEditor(str(temp_project))
        
        editor.backup_pom(label="test1")
        editor.backup_pom(label="test2")
        
        backups = editor.list_backups()
        
        assert len(backups) >= 2
    
    def test_cleanup_old_backups(self, temp_project):
        """Test cleaning up old backups."""
        editor = PomEditor(str(temp_project))
        
        for i in range(15):
            editor.backup_pom(label=f"test{i}")
        
        removed_count = editor.cleanup_old_backups(keep_count=5)
        
        backups = editor.list_backups()
        assert len(backups) == 5
        assert removed_count == 10
    
    def test_restore_pom(self, temp_project):
        """Test restoring pom.xml from backup."""
        editor = PomEditor(str(temp_project))
        
        backup_path = editor.backup_pom(label="before_change")
        
        dependency = {
            "group_id": "org.junit.jupiter",
            "artifact_id": "junit-jupiter",
            "version": "5.10.0",
            "scope": "test"
        }
        editor.add_dependency(dependency, backup=False)
        
        success = editor.restore_pom(backup_path)
        
        assert success is True
        assert not editor.has_dependency("org.junit.jupiter", "junit-jupiter")
    
    def test_restore_pom_invalid_path(self, temp_project):
        """Test restoring pom.xml from invalid path."""
        editor = PomEditor(str(temp_project))
        
        success = editor.restore_pom("/invalid/path/backup.xml")
        
        assert success is False
