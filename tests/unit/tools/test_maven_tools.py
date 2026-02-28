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
    
    def test_parse_jacoco_report(self, temp_project_with_coverage):
        """Test parsing JaCoCo XML report."""
        from pyutagent.tools.maven_tools import CoverageAnalyzer
        
        analyzer = CoverageAnalyzer(temp_project_with_coverage)
        report = analyzer.parse_report()
        
        assert report is not None
        assert report.line_coverage == 0.9  # 450 / 500
        assert report.branch_coverage == 0.8  # 80 / 100
        assert report.instruction_coverage == 0.9  # 900 / 1000
    
    def test_get_uncovered_lines(self, temp_project_with_coverage):
        """Test getting uncovered line numbers."""
        from pyutagent.tools.maven_tools import CoverageAnalyzer
        
        analyzer = CoverageAnalyzer(temp_project_with_coverage)
        
        # Try different path formats
        uncovered = analyzer.get_uncovered_lines("UserService.java")
        
        # Line 12 has mi="1" (missed)
        assert len(uncovered) >= 1
        assert 12 in uncovered
    
    def test_get_file_coverage(self, temp_project_with_coverage):
        """Test getting coverage for specific file."""
        from pyutagent.tools.maven_tools import CoverageAnalyzer
        
        analyzer = CoverageAnalyzer(temp_project_with_coverage)
        file_coverage = analyzer.get_file_coverage("UserService.java")
        
        assert file_coverage is not None
        assert file_coverage.line_coverage > 0


class TestProjectScanner:
    """Test suite for ProjectScanner."""
    
    def test_scan_java_files(self, tmp_path):
        """Test scanning for Java files."""
        from pyutagent.tools.maven_tools import ProjectScanner
        
        # Create some Java files
        src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
        src_dir.mkdir(parents=True)
        (src_dir / "UserService.java").write_text("public class UserService {}")
        (src_dir / "OrderService.java").write_text("public class OrderService {}")
        
        scanner = ProjectScanner(str(tmp_path))
        java_files = scanner.scan_java_files()
        
        assert len(java_files) == 2
        assert any("UserService.java" in f for f in java_files)
        assert any("OrderService.java" in f for f in java_files)
    
    def test_scan_test_files(self, tmp_path):
        """Test scanning for test files."""
        from pyutagent.tools.maven_tools import ProjectScanner
        
        # Create test files
        test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
        test_dir.mkdir(parents=True)
        (test_dir / "UserServiceTest.java").write_text("public class UserServiceTest {}")
        
        scanner = ProjectScanner(str(tmp_path))
        test_files = scanner.scan_test_files()
        
        assert len(test_files) == 1
        assert "UserServiceTest.java" in test_files[0]
    
    def test_get_source_directories(self, tmp_path):
        """Test getting source directories."""
        from pyutagent.tools.maven_tools import ProjectScanner
        
        # Create Maven structure
        (tmp_path / "src" / "main" / "java").mkdir(parents=True)
        (tmp_path / "src" / "test" / "java").mkdir(parents=True)
        
        scanner = ProjectScanner(str(tmp_path))
        src_dirs = scanner.get_source_directories()
        
        assert len(src_dirs) == 2
