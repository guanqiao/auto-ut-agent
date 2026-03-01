"""Tests for CLI main module."""

import pytest
from click.testing import CliRunner


class TestCLIMain:
    """Test CLI main entry point."""

    def test_cli_help(self):
        """Test CLI shows help message."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'PyUT Agent CLI' in result.output
        assert 'scan' in result.output
        assert 'generate' in result.output
        assert 'config' in result.output

    def test_cli_version(self):
        """Test CLI shows version."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert '0.1.0' in result.output

    def test_cli_no_args_shows_help(self):
        """Test CLI with no args shows help."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli)

        assert result.exit_code == 0
        assert 'Usage:' in result.output
