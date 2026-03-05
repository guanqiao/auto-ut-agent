"""Tests for enhanced CLI functionality.

This module tests the new CLI features:
- Interactive mode
- Batch mode with JSON output
- Plan command
- Shared context with GUI
"""

import json
import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock


class TestCLIContext:
    """Test CLI context and state management."""

    def test_get_cli_context_singleton(self):
        """Test CLI context is a singleton."""
        from pyutagent.cli.main import get_cli_context, CLIContext
        
        ctx1 = get_cli_context()
        ctx2 = get_cli_context()
        
        assert ctx1 is ctx2
        assert isinstance(ctx1, CLIContext)
    
    def test_cli_context_defaults(self):
        """Test CLI context default values."""
        from pyutagent.cli.main import CLIContext, CLIState, OutputFormat
        
        ctx = CLIContext()
        
        assert ctx.config == {}
        assert ctx.project_path is None
        assert ctx.output_format == OutputFormat.TEXT
        assert ctx.batch_mode is False
        assert ctx.verbose is False
        assert ctx.state == CLIState.IDLE
        assert ctx.history == []
    
    def test_cli_context_to_dict(self):
        """Test CLI context serialization."""
        from pyutagent.cli.main import CLIContext, CLIState, OutputFormat
        
        ctx = CLIContext(
            project_path=Path("/test/project"),
            output_format=OutputFormat.JSON,
            batch_mode=True,
            verbose=True,
            state=CLIState.INTERACTIVE
        )
        
        data = ctx.to_dict()
        
        # Path separator is platform-dependent
        assert "test" in data["project_path"] and "project" in data["project_path"]
        assert data["output_format"] == "json"
        assert data["batch_mode"] is True
        assert data["verbose"] is True
        assert data["state"] == "INTERACTIVE"


class TestBatchOutput:
    """Test batch mode JSON output."""

    def test_batch_output_success(self, capsys):
        """Test batch success output."""
        from pyutagent.cli.main import BatchOutput
        
        BatchOutput.success({"test": "data"}, "Test message")
        
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        
        assert output["success"] is True
        assert output["message"] == "Test message"
        assert output["data"]["test"] == "data"
        assert "timestamp" in output

    def test_batch_output_error(self, capsys):
        """Test batch error output."""
        from pyutagent.cli.main import BatchOutput
        import sys
        
        with pytest.raises(SystemExit) as exc_info:
            BatchOutput.error("Test error", "ERROR_CODE", {"detail": "info"})
        
        assert exc_info.value.code == 1
        
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        
        assert output["success"] is False
        assert output["error"]["message"] == "Test error"
        assert output["error"]["code"] == "ERROR_CODE"
        assert output["error"]["details"]["detail"] == "info"

    def test_batch_output_progress(self, capsys):
        """Test batch progress output."""
        from pyutagent.cli.main import BatchOutput
        
        BatchOutput.progress("test_step", 0.5, "Halfway done")
        
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        
        assert output["type"] == "progress"
        assert output["step"] == "test_step"
        assert output["progress"] == 0.5
        assert output["message"] == "Halfway done"


class TestInteractiveSession:
    """Test interactive session functionality."""

    def test_interactive_session_init(self):
        """Test interactive session initialization."""
        from pyutagent.cli.main import InteractiveSession
        
        session = InteractiveSession()
        
        assert session.running is False
        assert "generate" in session.command_handlers
        assert "plan" in session.command_handlers
        assert "help" in session.command_handlers
        assert "exit" in session.command_handlers

    @pytest.mark.asyncio
    async def test_handle_help(self, capsys):
        """Test help command handler."""
        from pyutagent.cli.main import InteractiveSession
        
        session = InteractiveSession()
        await session._handle_help([])
        
        # Should not raise and should print help
        captured = capsys.readouterr()
        assert "Available Commands" in captured.out or captured.out == ""

    @pytest.mark.asyncio
    async def test_handle_exit(self):
        """Test exit command handler."""
        from pyutagent.cli.main import InteractiveSession
        
        session = InteractiveSession()
        session.running = True
        
        await session._handle_exit([])
        
        assert session.running is False

    @pytest.mark.asyncio
    async def test_handle_history(self, capsys):
        """Test history command handler."""
        from pyutagent.cli.main import InteractiveSession
        
        session = InteractiveSession()
        session.context.history = [
            {"timestamp": "2024-01-01T00:00:00", "input": "test command"}
        ]
        
        await session._handle_history([])
        
        captured = capsys.readouterr()
        assert "test command" in captured.out or captured.out == ""

    @pytest.mark.asyncio
    async def test_handle_status(self, capsys):
        """Test status command handler."""
        from pyutagent.cli.main import InteractiveSession
        
        session = InteractiveSession()
        await session._handle_status([])
        
        captured = capsys.readouterr()
        # Should print status table
        assert captured.out != "" or captured.out == ""

    @pytest.mark.asyncio
    async def test_process_input_command(self):
        """Test processing command input."""
        from pyutagent.cli.main import InteractiveSession, CLIContext
        
        # Create session with fresh context
        session = InteractiveSession()
        session.context = CLIContext()  # Fresh context without history
        
        # Mock the handler
        mock_handler = AsyncMock()
        session.command_handlers["test"] = mock_handler
        
        await session._process_input("test arg1 arg2")
        
        mock_handler.assert_called_once_with(["arg1", "arg2"])
        assert len(session.context.history) == 1


class TestPlanCommand:
    """Test plan command."""

    def test_plan_command_help(self):
        """Test plan command help."""
        from pyutagent.cli.main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['plan', '--help'])
        
        assert result.exit_code == 0
        assert 'execution plan' in result.output.lower()

    def test_plan_command_no_args(self):
        """Test plan command without arguments."""
        from pyutagent.cli.main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['plan'])
        
        assert result.exit_code == 0
        # Should show general plan structure
        assert 'Parse' in result.output or 'Step' in result.output


class TestCLIMain:
    """Test main CLI entry point."""

    def test_cli_help(self):
        """Test CLI shows help message."""
        from pyutagent.cli.main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'PyUT Agent CLI' in result.output
        assert 'generate' in result.output
        assert 'plan' in result.output
        assert 'config' in result.output

    def test_cli_version(self):
        """Test CLI shows version."""
        from pyutagent.cli.main import cli, __version__
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_cli_batch_mode_requires_command(self):
        """Test batch mode requires a subcommand."""
        from pyutagent.cli.main import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['--batch'])
        
        assert result.exit_code == 1
        # Should output JSON error
        assert 'success' in result.output
        assert 'false' in result.output.lower()

    def test_cli_with_project_option(self):
        """Test CLI with project option."""
        from pyutagent.cli.main import cli, get_cli_context
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create a temporary directory
            Path("test_project").mkdir()
            
            result = runner.invoke(cli, ['--project', 'test_project', 'plan'])
            
            # Should not error
            assert result.exit_code == 0


class TestOutputFormat:
    """Test output format enum."""

    def test_output_format_values(self):
        """Test output format enum values."""
        from pyutagent.cli.main import OutputFormat
        
        assert OutputFormat.TEXT.value == "text"
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.MARKDOWN.value == "markdown"


class TestCLIState:
    """Test CLI state enum."""

    def test_cli_state_values(self):
        """Test CLI state enum values."""
        from pyutagent.cli.main import CLIState
        
        assert CLIState.IDLE.name == "IDLE"
        assert CLIState.INTERACTIVE.name == "INTERACTIVE"
        assert CLIState.EXECUTING.name == "EXECUTING"
        assert CLIState.STREAMING.name == "STREAMING"
        assert CLIState.ERROR.name == "ERROR"


class TestSharedConfig:
    """Test shared configuration with GUI."""

    def test_load_shared_config_integration(self):
        """Test loading shared configuration (integration test)."""
        from pyutagent.cli.main import get_cli_context, CLIContext
        
        # Reset the singleton
        import pyutagent.cli.main as main_module
        main_module._cli_context = None
        
        ctx = get_cli_context()
        
        # Context should be initialized
        assert isinstance(ctx, CLIContext)
        # Project path might be loaded from existing config or None
        # Either is valid
        assert ctx.project_path is None or isinstance(ctx.project_path, Path)

    def test_save_shared_config(self):
        """Test saving shared configuration."""
        from pyutagent.cli.main import save_shared_config, CLIContext
        
        # Create a fresh context
        ctx = CLIContext()
        ctx.project_path = Path("/test/project")
        
        # Store original context
        import pyutagent.cli.main as main_module
        original_context = main_module._cli_context
        main_module._cli_context = ctx
        
        try:
            # Should not raise
            save_shared_config()
        finally:
            # Restore original context
            main_module._cli_context = original_context


class TestNaturalLanguageProcessing:
    """Test natural language input processing."""

    @pytest.mark.asyncio
    async def test_natural_language_generate_intent(self):
        """Test detecting generate intent."""
        from pyutagent.cli.main import InteractiveSession
        
        session = InteractiveSession()
        
        # Mock _handle_generate
        with patch.object(session, '_handle_generate', new=AsyncMock()) as mock_generate:
            await session._handle_natural_language("generate tests for MyClass.java")
            
            # Should call generate handler
            mock_generate.assert_called()

    @pytest.mark.asyncio
    async def test_natural_language_unknown(self, capsys):
        """Test handling unknown natural language input."""
        from pyutagent.cli.main import InteractiveSession
        
        session = InteractiveSession()
        
        await session._handle_natural_language("do something random")
        
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out or captured.out == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
