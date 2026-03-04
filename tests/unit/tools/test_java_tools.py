"""Tests for Java tools module."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pyutagent.tools.java_tools import (
    find_java_executable,
    find_javac_executable,
    find_java_home,
    get_java_info,
    get_configured_java_paths,
    JavaInfo,
)


class TestFindJavaExecutable:
    """Tests for find_java_executable function."""

    @patch("pyutagent.tools.java_tools.shutil.which")
    @patch("pyutagent.tools.java_tools._get_java_home_from_env")
    def test_find_java_from_env(self, mock_env, mock_which):
        """Test finding Java from JAVA_HOME environment variable."""
        mock_env.return_value = "/path/to/java/home"
        mock_which.return_value = None
        
        with patch("pyutagent.tools.java_tools._check_java_bin") as mock_check:
            mock_check.return_value = "/path/to/java/home/bin/java"
            result = find_java_executable()
            assert result == "/path/to/java/home/bin/java"

    @patch("pyutagent.tools.java_tools.shutil.which")
    @patch("pyutagent.tools.java_tools._get_java_home_from_env")
    def test_find_java_from_path(self, mock_env, mock_which):
        """Test finding Java from PATH."""
        mock_env.return_value = None
        mock_which.return_value = "/usr/bin/java"
        
        result = find_java_executable()
        assert result == "/usr/bin/java"

    @patch("pyutagent.tools.java_tools.shutil.which")
    @patch("pyutagent.tools.java_tools._get_java_home_from_env")
    @patch("pyutagent.tools.java_tools.platform.system")
    def test_find_java_not_found(self, mock_platform, mock_env, mock_which):
        """Test when Java is not found."""
        mock_platform.return_value = "Linux"
        mock_env.return_value = None
        mock_which.return_value = None
        
        with patch("pyutagent.tools.java_tools._find_java_unix") as mock_unix:
            mock_unix.return_value = None
            result = find_java_executable()
            assert result is None


class TestFindJavacExecutable:
    """Tests for find_javac_executable function."""

    @patch("pyutagent.tools.java_tools.shutil.which")
    @patch("pyutagent.tools.java_tools._get_java_home_from_env")
    def test_find_javac_from_env(self, mock_env, mock_which):
        """Test finding javac from JAVA_HOME environment variable."""
        mock_env.return_value = "/path/to/java/home"
        mock_which.return_value = None
        
        with patch("pyutagent.tools.java_tools._check_javac_bin") as mock_check:
            mock_check.return_value = "/path/to/java/home/bin/javac"
            result = find_javac_executable()
            assert result == "/path/to/java/home/bin/javac"


class TestFindJavaHome:
    """Tests for find_java_home function."""

    @patch("pyutagent.tools.java_tools._get_java_home_from_env")
    def test_find_java_home_from_env(self, mock_env):
        """Test finding JAVA_HOME from environment."""
        mock_env.return_value = "/path/to/java/home"
        
        with patch.object(Path, "exists", return_value=True):
            result = find_java_home()
            assert result == "/path/to/java/home"


class TestGetJavaInfo:
    """Tests for get_java_info function."""

    @patch("pyutagent.tools.java_tools.find_java_executable")
    @patch("pyutagent.tools.java_tools.find_java_home")
    @patch("pyutagent.tools.java_tools.find_javac_executable")
    @patch("pyutagent.tools.java_tools._get_java_version_and_vendor")
    def test_get_java_info_success(
        self, mock_version, mock_javac, mock_home, mock_java
    ):
        """Test getting Java info successfully."""
        mock_java.return_value = "/usr/bin/java"
        mock_home.return_value = "/usr/lib/jvm/java-11"
        mock_javac.return_value = "/usr/bin/javac"
        mock_version.return_value = ("11.0.11", "OpenJDK")
        
        result = get_java_info()
        
        assert result is not None
        assert result.java_home == "/usr/lib/jvm/java-11"
        assert result.java_version == "11.0.11"
        assert result.vendor == "OpenJDK"
        assert result.java_path == "/usr/bin/java"
        assert result.javac_path == "/usr/bin/javac"
        assert result.is_jdk is True

    @patch("pyutagent.tools.java_tools.find_java_executable")
    def test_get_java_info_not_found(self, mock_java):
        """Test when Java is not found."""
        mock_java.return_value = None
        
        result = get_java_info()
        assert result is None


class TestGetConfiguredJavaPaths:
    """Tests for get_configured_java_paths function."""

    @patch("pyutagent.tools.java_tools.find_java_executable")
    @patch("pyutagent.tools.java_tools.find_javac_executable")
    def test_configured_path_takes_priority(
        self, mock_javac, mock_java
    ):
        """Test that configured path takes priority."""
        mock_settings = MagicMock(
            jdk=MagicMock(java_home="/configured/java/home")
        )
        
        with patch("pyutagent.core.config.get_settings", return_value=mock_settings):
            with patch.object(Path, "exists", return_value=True):
                with patch("pyutagent.tools.java_tools._check_java_bin") as mock_check_java:
                    with patch("pyutagent.tools.java_tools._check_javac_bin") as mock_check_javac:
                        mock_check_java.return_value = "/configured/java/home/bin/java"
                        mock_check_javac.return_value = "/configured/java/home/bin/javac"
                        
                        java_path, javac_path = get_configured_java_paths()
                        
                        assert java_path == "/configured/java/home/bin/java"
                        assert javac_path == "/configured/java/home/bin/javac"

    @patch("pyutagent.tools.java_tools.find_java_executable")
    @patch("pyutagent.tools.java_tools.find_javac_executable")
    def test_fallback_to_auto_detect(
        self, mock_javac, mock_java
    ):
        """Test fallback to auto-detect when no config."""
        mock_settings = MagicMock(
            jdk=MagicMock(java_home="")
        )
        
        with patch("pyutagent.core.config.get_settings", return_value=mock_settings):
            mock_java.return_value = "/auto/detected/java"
            mock_javac.return_value = "/auto/detected/javac"
            
            java_path, javac_path = get_configured_java_paths()
            
            assert java_path == "/auto/detected/java"
            assert javac_path == "/auto/detected/javac"


class TestJavaInfo:
    """Tests for JavaInfo dataclass."""

    def test_java_info_creation(self):
        """Test creating JavaInfo instance."""
        info = JavaInfo(
            java_home="/usr/lib/jvm/java-11",
            java_version="11.0.11",
            vendor="OpenJDK",
            java_path="/usr/bin/java",
            javac_path="/usr/bin/javac",
            is_jdk=True
        )
        
        assert info.java_home == "/usr/lib/jvm/java-11"
        assert info.java_version == "11.0.11"
        assert info.vendor == "OpenJDK"
        assert info.java_path == "/usr/bin/java"
        assert info.javac_path == "/usr/bin/javac"
        assert info.is_jdk is True

    def test_java_info_jre_only(self):
        """Test JavaInfo for JRE (no javac)."""
        info = JavaInfo(
            java_home="/usr/lib/jvm/jre",
            java_version="11.0.11",
            vendor="OpenJDK",
            java_path="/usr/bin/java",
            javac_path=None,
            is_jdk=False
        )
        
        assert info.is_jdk is False
        assert info.javac_path is None
