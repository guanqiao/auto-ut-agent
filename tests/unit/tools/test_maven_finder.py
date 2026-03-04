"""Tests for Maven executable finder."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pyutagent.tools.maven_tools import (
    find_maven_executable,
    _check_maven_bin,
    _find_maven_windows,
    _find_maven_unix
)


class TestMavenFinder:
    """Test Maven executable finder functions."""
    
    def test_find_maven_in_path(self):
        """Test finding mvn in PATH."""
        with patch('shutil.which', return_value='/usr/bin/mvn'):
            result = find_maven_executable()
            assert result == '/usr/bin/mvn'
    
    def test_find_maven_via_m3_home(self):
        """Test finding mvn via M3_HOME environment variable."""
        with patch('shutil.which', return_value=None):
            with patch.dict(os.environ, {'M3_HOME': '/opt/maven'}):
                with patch('pyutagent.tools.maven_tools.Path.exists', return_value=True):
                    result = find_maven_executable()
                    assert result is not None
    
    def test_check_maven_bin_windows(self):
        """Test checking Maven bin directory on Windows."""
        with patch('pyutagent.tools.maven_tools.platform.system', return_value='Windows'):
            with patch('pyutagent.tools.maven_tools.Path.exists', side_effect=[True, False]):
                result = _check_maven_bin('/opt/maven')
                assert result is not None
                assert result.endswith('mvn.cmd')
    
    def test_check_maven_bin_unix(self):
        """Test checking Maven bin directory on Unix."""
        with patch('pyutagent.tools.maven_tools.platform.system', return_value='Linux'):
            with patch('pyutagent.tools.maven_tools.Path.exists', return_value=True):
                result = _check_maven_bin('/opt/maven')
                assert result is not None
                assert result.endswith('mvn')
    
    def test_find_maven_windows_program_files(self):
        """Test finding Maven in Windows Program Files."""
        with patch('pyutagent.tools.maven_tools.platform.system', return_value='Windows'):
            with patch.dict(os.environ, {
                'ProgramFiles': 'C:\\Program Files',
                'ProgramFiles(x86)': 'C:\\Program Files (x86)',
                'ProgramData': 'C:\\ProgramData',
                'USERPROFILE': 'C:\\Users\\Test'
            }):
                with patch('pyutagent.tools.maven_tools.Path.exists', side_effect=[
                    False, False,  # mvn.cmd checks
                    False, False,  # mvn.bat checks
                    True,  # Found in Chocolatey
                    False, False, False, False, False, False
                ]):
                    result = _find_maven_windows()
                    assert result is not None
    
    def test_find_maven_unix_opt(self):
        """Test finding Maven in Unix /opt directory."""
        with patch('pyutagent.tools.maven_tools.platform.system', return_value='Linux'):
            with patch('pyutagent.tools.maven_tools.Path.exists', side_effect=[
                False, False, True, False, False, False
            ]):
                result = _find_maven_unix()
                assert result is not None
                assert 'opt/maven/bin' in result or 'opt\\maven\\bin' in result
    
    def test_maven_not_found(self):
        """Test when Maven is not found anywhere."""
        with patch('shutil.which', return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                with patch('pyutagent.tools.maven_tools.platform.system', return_value='Linux'):
                    with patch('pyutagent.tools.maven_tools.Path.exists', return_value=False):
                        result = find_maven_executable()
                        assert result is None


class TestMavenRunnerWithFinder:
    """Test MavenRunner integration with finder."""
    
    def test_maven_runner_uses_finder(self, tmp_path):
        """Test that MavenRunner uses the finder."""
        from pyutagent.tools.maven_tools import MavenRunner
        
        project_path = tmp_path / "test_project"
        project_path.mkdir()
        (project_path / "pom.xml").touch()
        
        runner = MavenRunner(str(project_path))
        
        with patch('pyutagent.tools.maven_tools.find_maven_executable', return_value='/usr/bin/mvn'):
            mvn_path = runner._get_maven_executable()
            assert mvn_path == '/usr/bin/mvn'
    
    def test_maven_runner_caches_executable(self, tmp_path):
        """Test that MavenRunner caches the executable path."""
        from pyutagent.tools.maven_tools import MavenRunner
        
        project_path = tmp_path / "test_project"
        project_path.mkdir()
        (project_path / "pom.xml").touch()
        
        runner = MavenRunner(str(project_path))
        
        with patch('pyutagent.tools.maven_tools.find_maven_executable', return_value='/usr/bin/mvn') as mock_finder:
            runner._get_maven_executable()
            runner._get_maven_executable()
            
            assert mock_finder.call_count == 1
