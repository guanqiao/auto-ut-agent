"""Tests for scan command."""

import os
import tempfile
from pathlib import Path
from click.testing import CliRunner
import pytest


class TestScanCommand:
    """Test scan command functionality."""

    def test_scan_help(self):
        """Test scan command shows help."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['scan', '--help'])

        assert result.exit_code == 0
        assert 'Scan a Maven project' in result.output
        assert 'PROJECT_PATH' in result.output

    def test_scan_nonexistent_project(self):
        """Test scan with non-existent project path."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['scan', '/nonexistent/path'])

        assert result.exit_code != 0
        assert 'does not exist' in result.output or 'Error' in result.output

    def test_scan_valid_maven_project(self):
        """Test scan with valid Maven project."""
        from pyutagent.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock Maven project structure
            pom_path = Path(tmpdir) / 'pom.xml'
            pom_path.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>test</groupId>
    <artifactId>test</artifactId>
    <version>1.0</version>
</project>''')

            src_dir = Path(tmpdir) / 'src' / 'main' / 'java' / 'com' / 'example'
            src_dir.mkdir(parents=True)
            (src_dir / 'MyClass.java').write_text('''
package com.example;
public class MyClass {
    public int add(int a, int b) {
        return a + b;
    }
}
''')

            runner = CliRunner()
            result = runner.invoke(cli, ['scan', tmpdir])

            assert result.exit_code == 0
            assert 'MyClass.java' in result.output or 'com.example' in result.output

    def test_scan_with_tree_option(self):
        """Test scan with --tree option."""
        from pyutagent.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            pom_path = Path(tmpdir) / 'pom.xml'
            pom_path.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>test</groupId>
    <artifactId>test</artifactId>
    <version>1.0</version>
</project>''')

            runner = CliRunner()
            result = runner.invoke(cli, ['scan', tmpdir, '--tree'])

            assert result.exit_code == 0
