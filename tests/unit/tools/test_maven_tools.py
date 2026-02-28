"""Tests for Maven tools."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestMavenRunner:
    """Test suite for MavenRunner."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary Maven project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            # Create pom.xml
            pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
</project>
"""
            (project_path / "pom.xml").write_text(pom_content)
            yield str(project_path)
    
    def test_is_maven_project(self, temp_project):
        """Test detecting Maven project."""
        from pyutagent.tools.maven_tools import MavenRunner
        
        runner = MavenRunner(temp_project)
        assert runner.is_maven_project() is True
    
    def test_is_not_maven_project(self, tmp_path):
        """Test non-Maven project detection."""
        from pyutagent.tools.maven_tools import MavenRunner
        
        runner = MavenRunner(str(tmp_path))
        assert runner.is_maven_project() is False
    
    @patch('subprocess.run')
    def test_run_tests(self, mock_run, temp_project):
        """Test running Maven tests."""
        from pyutagent.tools.maven_tools import MavenRunner
        
        mock_run.return_value = Mock(returncode=0, stdout="Tests run: 5", stderr="")
        
        runner = MavenRunner(temp_project)
        result = runner.run_tests()
        
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "mvn" in args
        assert "test" in args
    
    @patch('subprocess.run')
    def test_run_tests_failure(self, mock_run, temp_project):
        """Test handling test failures."""
        from pyutagent.tools.maven_tools import MavenRunner
        
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Build failed")
        
        runner = MavenRunner(temp_project)
        result = runner.run_tests()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_generate_coverage(self, mock_run, temp_project):
        """Test generating JaCoCo coverage report."""
        from pyutagent.tools.maven_tools import MavenRunner
        
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        runner = MavenRunner(temp_project)
        result = runner.generate_coverage()
        
        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "jacoco:report" in args


class TestCoverageAnalyzer:
    """Test suite for CoverageAnalyzer."""
    
    @pytest.fixture
    def temp_project_with_coverage(self):
        """Create a temporary project with JaCoCo report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)
            
            # Create JaCoCo report directory
            jacoco_dir = project_path / "target" / "site" / "jacoco"
            jacoco_dir.mkdir(parents=True)
            
            # Create sample JaCoCo XML
            jacoco_xml = """<?xml version="1.0" encoding="UTF-8"?>
<report name="test">
    <sessioninfo id="test-session" start="1234567890" dump="1234567891"/>
    <counter type="INSTRUCTION" missed="100" covered="900"/>
    <counter type="BRANCH" missed="20" covered="80"/>
    <counter type="LINE" missed="50" covered="450"/>
    <counter type="COMPLEXITY" missed="30" covered="70"/>
    <counter type="METHOD" missed="10" covered="40"/>
    <counter type="CLASS" missed="2" covered="8"/>
    <package name="com/example">
        <class name="com/example/UserService">
            <method name="getUser" desc="(I)Lcom/example/User;" line="10">
                <counter type="INSTRUCTION" missed="0" covered="10"/>
                <counter type="LINE" missed="0" covered="3"/>
            </method>
        </class>
        <sourcefile name="UserService.java">
            <line nr="10" mi="0" ci="3"/>
            <line nr="11" mi="0" ci="2"/>
            <line nr="12" mi="1" ci="0"/>
        </sourcefile>
    </package>
</report>
"""
            (jacoco_dir / "jacoco.xml").write_text(jacoco_xml)
            
            yield str(project_path)
