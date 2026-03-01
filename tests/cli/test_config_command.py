"""Tests for config command."""

import tempfile
from pathlib import Path
from click.testing import CliRunner
import pytest
from unittest.mock import patch, MagicMock


class TestConfigCommand:
    """Test config command functionality."""

    def test_config_help(self):
        """Test config command shows help."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['config', '--help'])

        assert result.exit_code == 0
        assert 'Manage configuration' in result.output

    def test_config_llm_help(self):
        """Test config llm subcommand shows help."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'llm', '--help'])

        assert result.exit_code == 0
        assert 'Manage LLM configurations' in result.output

    @patch('pyutagent.config.load_llm_config')
    def test_config_llm_list_empty(self, mock_load):
        """Test config llm list with no configurations."""
        from pyutagent.cli.main import cli

        mock_collection = MagicMock()
        mock_collection.configs = []
        mock_load.return_value = mock_collection

        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'llm', 'list'])

        assert result.exit_code == 0
        assert 'No LLM configurations' in result.output or 'default' in result.output

    @patch('pyutagent.config.save_llm_config')
    @patch('pyutagent.config.load_llm_config')
    @patch('pyutagent.llm.config.LLMConfig')
    def test_config_llm_add(self, mock_config_class, mock_load, mock_save):
        """Test config llm add command."""
        from pyutagent.cli.main import cli

        mock_collection = MagicMock()
        mock_collection.configs = []
        mock_load.return_value = mock_collection

        mock_config = MagicMock()
        mock_config.id = 'test-id-1234'
        mock_config_class.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(cli, [
            'config', 'llm', 'add',
            '--name', 'gpt4',
            '--provider', 'openai',
            '--model', 'gpt-4',
            '--api-key', 'test-key'
        ])

        assert result.exit_code == 0
        mock_collection.add_config.assert_called_once()
        mock_save.assert_called_once()

    @patch('pyutagent.config.save_llm_config')
    @patch('pyutagent.config.load_llm_config')
    def test_config_llm_set_default(self, mock_load, mock_save):
        """Test config llm set-default command."""
        from pyutagent.cli.main import cli

        mock_config = MagicMock()
        mock_config.id = 'test-config-id'
        mock_config.name = 'gpt4'

        mock_collection = MagicMock()
        mock_collection.configs = [mock_config]
        mock_load.return_value = mock_collection

        runner = CliRunner()
        result = runner.invoke(cli, [
            'config', 'llm', 'set-default', 'test-config'
        ])

        assert result.exit_code == 0
        mock_collection.set_default_config.assert_called_once_with('test-config-id')
        mock_save.assert_called_once()

    def test_config_aider_help(self):
        """Test config aider subcommand shows help."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'aider', '--help'])

        assert result.exit_code == 0
        assert 'Aider configuration' in result.output or 'architect' in result.output

    @patch('pyutagent.config.load_llm_config')
    def test_config_show(self, mock_load):
        """Test config show command."""
        from pyutagent.cli.main import cli

        mock_collection = MagicMock()
        mock_collection.configs = []
        mock_load.return_value = mock_collection

        runner = CliRunner()
        result = runner.invoke(cli, ['config', 'show'])

        assert result.exit_code == 0
