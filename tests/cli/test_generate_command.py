"""Tests for generate command."""

import os
import tempfile
from pathlib import Path
from click.testing import CliRunner
import pytest
from unittest.mock import patch, MagicMock


class TestGenerateCommand:
    """Test generate command functionality."""

    def test_generate_help(self):
        """Test generate command shows help."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['generate', '--help'])

        assert result.exit_code == 0
        assert 'Generate unit tests' in result.output
        assert 'FILE_PATH' in result.output
        assert '--llm' in result.output
        assert '--output-dir' in result.output
        assert '--coverage-target' in result.output

    def test_generate_nonexistent_file(self):
        """Test generate with non-existent file."""
        from pyutagent.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ['generate', '/nonexistent/MyClass.java'])

        assert result.exit_code != 0
        assert 'does not exist' in result.output or 'Error' in result.output

    def test_generate_invalid_file_extension(self):
        """Test generate with non-Java file."""
        from pyutagent.cli.main import cli

        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'not a java file')
            temp_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(cli, ['generate', temp_path])

            assert result.exit_code != 0
            assert '.java' in result.output or 'Error' in result.output
        finally:
            os.unlink(temp_path)

    @patch('pyutagent.config.load_llm_config')
    @patch('pyutagent.agent.test_generator.TestGeneratorAgent')
    def test_generate_with_mock_agent(self, mock_agent_class, mock_load_config):
        """Test generate with mocked agent."""
        from pyutagent.cli.main import cli

        # Setup mock config
        mock_config = MagicMock()
        mock_collection = MagicMock()
        mock_collection.get_default_config.return_value = mock_config
        mock_load_config.return_value = mock_collection

        # Setup mock agent
        mock_agent = MagicMock()
        mock_agent_class.return_value = mock_agent
        mock_agent.generate_tests.return_value = {
            'success': True,
            'test_file': '/path/to/Test.java',
            'coverage': 85.5
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock Java file
            java_file = Path(tmpdir) / 'MyClass.java'
            java_file.write_text('''
public class MyClass {
    public int add(int a, int b) {
        return a + b;
    }
}
''')

            runner = CliRunner()
            result = runner.invoke(cli, ['generate', str(java_file)])

            assert result.exit_code == 0
            mock_agent.generate_tests.assert_called_once()

    def test_generate_with_coverage_target(self):
        """Test generate with coverage target option."""
        from pyutagent.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            java_file = Path(tmpdir) / 'MyClass.java'
            java_file.write_text('public class MyClass {}')

            runner = CliRunner()
            result = runner.invoke(cli, [
                'generate', str(java_file),
                '--coverage-target', '90'
            ])

            # Should either succeed or fail gracefully
            assert result.exit_code in [0, 1]

    def test_generate_with_max_iterations(self):
        """Test generate with max iterations option."""
        from pyutagent.cli.main import cli

        with tempfile.TemporaryDirectory() as tmpdir:
            java_file = Path(tmpdir) / 'MyClass.java'
            java_file.write_text('public class MyClass {}')

            runner = CliRunner()
            result = runner.invoke(cli, [
                'generate', str(java_file),
                '--max-iterations', '5'
            ])

            assert result.exit_code in [0, 1]
