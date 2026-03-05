"""Enhanced CLI main entry point for PyUT Agent.

This module provides a comprehensive CLI interface with:
- Interactive mode with natural language input and streaming output
- Command mode (generate, plan, config)
- Batch processing mode with JSON output
- Shared state with GUI (config, project history, generated code)

Usage:
    $ pyutagent-cli                          # Interactive mode
    $ pyutagent-cli generate <file.java>     # Generate tests
    $ pyutagent-cli plan <file.java>         # View execution plan
    $ pyutagent-cli config llm list          # Manage LLM configs
    $ pyutagent-cli --batch generate ...     # Batch mode (JSON output)

Reference:
    - Qoder CLI design patterns
    - Claude Code interactive interface
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.status import Status
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from . import __version__
from .commands.scan import scan_command
from .commands.generate import generate_command
from .commands.generate_all import generate_all_command
from .commands.config import config_group
from .commands.history import history_group
from .commands.hooks import hooks_group
from .commands.skills import skills_group
from .commands.project import project_group
from .commands.task import task_command, task_types_command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


class OutputFormat(Enum):
    """Output format for CLI."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class CLIState(Enum):
    """CLI application state."""
    IDLE = auto()
    INTERACTIVE = auto()
    EXECUTING = auto()
    STREAMING = auto()
    ERROR = auto()


@dataclass
class CLIContext:
    """Shared CLI context object.
    
    This context is shared between CLI and GUI, ensuring consistent
    configuration, project history, and generated code.
    
    Attributes:
        config: Application configuration
        project_path: Current project path
        output_format: Output format (text/json/markdown)
        batch_mode: Whether running in batch mode
        verbose: Verbose output flag
        state: Current CLI state
        history: Command history
    """
    config: Dict[str, Any] = field(default_factory=dict)
    project_path: Optional[Path] = None
    output_format: OutputFormat = OutputFormat.TEXT
    batch_mode: bool = False
    verbose: bool = False
    state: CLIState = CLIState.IDLE
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "project_path": str(self.project_path) if self.project_path else None,
            "output_format": self.output_format.value,
            "batch_mode": self.batch_mode,
            "verbose": self.verbose,
            "state": self.state.name,
            "history_count": len(self.history),
        }


# Global context instance (shared with GUI)
_cli_context: Optional[CLIContext] = None


def get_cli_context() -> CLIContext:
    """Get or create global CLI context.
    
    Returns:
        CLIContext instance shared between CLI and GUI
    """
    global _cli_context
    if _cli_context is None:
        _cli_context = CLIContext()
        # Load shared configuration from GUI
        _load_shared_config()
    return _cli_context


def _load_shared_config() -> None:
    """Load configuration shared with GUI."""
    global _cli_context
    if _cli_context is None:
        return
    
    try:
        from pyutagent.core.config import load_app_config, load_app_state
        
        # Load app config
        settings = load_app_config()
        _cli_context.config["settings"] = settings.model_dump(mode="json")
        
        # Load app state (project history)
        app_state = load_app_state()
        if app_state.last_project_path:
            _cli_context.project_path = Path(app_state.last_project_path)
        
        logger.info(f"[CLI] Loaded shared config from GUI")
    except Exception as e:
        logger.warning(f"[CLI] Failed to load shared config: {e}")


def save_shared_config() -> None:
    """Save configuration to be shared with GUI."""
    ctx = get_cli_context()
    
    try:
        from pyutagent.core.config import save_app_state, AppState, ProjectHistory
        
        # Save project history
        if ctx.project_path:
            app_state = AppState()
            app_state.add_project(str(ctx.project_path))
            save_app_state(app_state)
        
        logger.info(f"[CLI] Saved shared config for GUI")
    except Exception as e:
        logger.warning(f"[CLI] Failed to save shared config: {e}")


# =============================================================================
# Interactive Mode
# =============================================================================

class InteractiveSession:
    """Interactive CLI session with natural language support.
    
    Features:
    - Natural language command parsing
    - Real-time streaming output
    - Context-aware responses
    - Command history
    """
    
    def __init__(self):
        self.context = get_cli_context()
        self.running = False
        self.command_handlers: Dict[str, Callable] = {
            "generate": self._handle_generate,
            "gen": self._handle_generate,
            "plan": self._handle_plan,
            "config": self._handle_config,
            "help": self._handle_help,
            "exit": self._handle_exit,
            "quit": self._handle_exit,
            "history": self._handle_history,
            "status": self._handle_status,
        }
    
    async def start(self) -> None:
        """Start interactive session."""
        self.running = True
        self.context.state = CLIState.INTERACTIVE
        
        console.print(Panel.fit(
            "[bold blue]PyUT Agent CLI[/bold blue] - Interactive Mode\n"
            "[dim]Type 'help' for commands, 'exit' to quit[/dim]",
            title=f"v{__version__}",
            border_style="blue"
        ))
        
        # Show current project if available
        if self.context.project_path:
            console.print(f"[dim]Current project: {self.context.project_path}[/dim]\n")
        
        while self.running:
            try:
                # Get user input
                user_input = Prompt.ask("[bold green]>>>[/bold green]").strip()
                
                if not user_input:
                    continue
                
                # Parse and execute command
                await self._process_input(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                if self.context.verbose:
                    console.print_exception()
        
        console.print("[dim]Goodbye![/dim]")
    
    async def _process_input(self, user_input: str) -> None:
        """Process user input."""
        # Record in history
        self.context.history.append({
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
        })
        
        # Parse command
        parts = user_input.split()
        command = parts[0].lower()
        args = parts[1:]
        
        # Check for direct command
        if command in self.command_handlers:
            await self.command_handlers[command](args)
        else:
            # Treat as natural language query
            await self._handle_natural_language(user_input)
    
    async def _handle_natural_language(self, query: str) -> None:
        """Handle natural language query with streaming output."""
        console.print(f"[dim]Processing: {query}[/dim]")
        
        # Simple intent detection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["generate", "test", "create"]):
            # Extract file path if present
            words = query.split()
            for word in words:
                if word.endswith(".java"):
                    await self._handle_generate([word])
                    return
            
            console.print("[yellow]Please specify a Java file to generate tests for[/yellow]")
            console.print("Example: 'generate tests for MyClass.java' or 'gen src/main/java/MyClass.java'")
            
        elif any(word in query_lower for word in ["plan", "show plan", "what will"]):
            console.print("[dim]Execution plan feature - specify a file to see the plan[/dim]")
            
        elif any(word in query_lower for word in ["config", "setting", "setup"]):
            await self._handle_config([])
            
        else:
            console.print(f"[yellow]Unknown command: {query}[/yellow]")
            console.print("Type 'help' for available commands")
    
    async def _handle_generate(self, args: List[str]) -> None:
        """Handle generate command in interactive mode."""
        if not args:
            console.print("[yellow]Usage: generate <file.java>[/yellow]")
            return
        
        file_path = Path(args[0])
        if not file_path.exists():
            # Try relative to project path
            if self.context.project_path:
                file_path = self.context.project_path / file_path
        
        if not file_path.exists():
            console.print(f"[red]File not found: {args[0]}[/red]")
            return
        
        await self._stream_generate(file_path)
    
    async def _stream_generate(self, file_path: Path) -> None:
        """Generate tests with streaming output."""
        self.context.state = CLIState.STREAMING
        
        try:
            from pyutagent.agent.streaming import StreamingTestGenerator, StreamingConfig
            from pyutagent.core.config import load_llm_config
            from pyutagent.llm.client import LLMClient
            from pyutagent.memory.working_memory import WorkingMemory
            
            # Load configuration
            config_collection = load_llm_config()
            llm_config = config_collection.get_default_config()
            
            if not llm_config:
                console.print("[red]No LLM configuration found. Run 'config llm add' first.[/red]")
                return
            
            llm_client = LLMClient.from_config(llm_config)
            
            # Setup working memory
            working_memory = WorkingMemory(
                target_coverage=0.8,
                max_iterations=10,
                current_file=str(file_path)
            )
            
            # Create streaming generator
            stream_config = StreamingConfig(
                enable_preview=True,
                preview_interval=0.5
            )
            generator = StreamingTestGenerator(llm_client, stream_config)
            
            # Create prompt
            from pyutagent.agent.prompts import create_test_generation_prompt
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            prompt = create_test_generation_prompt(source_code, str(file_path))
            
            # Stream generation with live display
            accumulated_code = []
            detected_methods = []
            
            with Status("[bold green]Generating tests...", spinner="dots") as status:
                async for chunk_data in generator.generate_tests_with_preview(prompt):
                    preview = chunk_data["preview"]
                    
                    # Update status
                    if chunk_data["detected_methods"]:
                        detected_methods = chunk_data["detected_methods"]
                        status.update(f"[bold green]Generated {len(detected_methods)} test methods...")
                    
                    accumulated_code.append(preview.partial_code)
            
            # Show result
            result = generator.get_result()
            
            if result and result.success:
                console.print(f"\n[green]✓ Tests generated successfully![/green]")
                console.print(f"  Methods: {len(detected_methods)}")
                console.print(f"  Tokens: {result.total_tokens}")
                console.print(f"  Time: {result.total_time:.2f}s")
                
                # Show preview of generated code
                code = generator.extract_java_code()
                if code:
                    console.print("\n[bold]Preview:[/bold]")
                    syntax = Syntax(code[:500] + "..." if len(code) > 500 else code, 
                                   "java", theme="monokai", line_numbers=True)
                    console.print(syntax)
            else:
                console.print(f"\n[red]✗ Generation failed[/red]")
                if result and result.error:
                    console.print(f"Error: {result.error}")
                    
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            if self.context.verbose:
                console.print_exception()
        finally:
            self.context.state = CLIState.INTERACTIVE
    
    async def _handle_plan(self, args: List[str]) -> None:
        """Handle plan command in interactive mode."""
        if not args:
            # Show general execution plan structure
            table = Table(title="Test Generation Execution Plan")
            table.add_column("Step", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Dependencies", style="yellow")
            
            steps = [
                ("1. Parse", "Parse target file to extract class info", "-"),
                ("2. Generate", "Generate initial test cases", "Parse"),
                ("3. Compile", "Compile generated tests", "Generate"),
                ("4. Test", "Execute tests and collect results", "Compile"),
                ("5. Analyze", "Analyze test coverage", "Test"),
                ("6. Optimize", "Generate additional tests for uncovered code", "Analyze"),
            ]
            
            for step, desc, deps in steps:
                table.add_row(step, desc, deps)
            
            console.print(table)
            console.print("\n[dim]To see a specific plan, run: plan <file.java>[/dim]")
        else:
            # Show plan for specific file
            file_path = Path(args[0])
            console.print(f"[dim]Execution plan for: {file_path.name}[/dim]")
            
            from pyutagent.agent.execution.execution_plan import ExecutionPlan
            
            plan = ExecutionPlan.create_test_generation_plan(str(file_path))
            
            table = Table(title=f"Plan: {plan.name}")
            table.add_column("Step", style="cyan")
            table.add_column("Type", style="blue")
            table.add_column("Description", style="green")
            table.add_column("Status", style="yellow")
            
            for step in plan.steps:
                table.add_row(
                    step.name,
                    step.step_type.value,
                    step.description,
                    step.status.name
                )
            
            console.print(table)
    
    async def _handle_config(self, args: List[str]) -> None:
        """Handle config command in interactive mode."""
        if not args:
            # Show current config summary
            from pyutagent.core.config import load_llm_config, load_app_config
            
            settings = load_app_config()
            config_collection = load_llm_config()
            
            console.print("\n[bold]Current Configuration:[/bold]\n")
            
            # LLM Config
            console.print("[bold cyan]LLM Configuration:[/bold cyan]")
            default = config_collection.get_default_config()
            if default:
                console.print(f"  Default: {default.get_display_name()}")
                console.print(f"  Provider: {default.provider}")
                console.print(f"  Model: {default.model}")
            else:
                console.print("  [yellow]No LLM configuration set[/yellow]")
            
            # Coverage settings
            console.print("\n[bold cyan]Coverage Settings:[/bold cyan]")
            console.print(f"  Target: {settings.coverage.target_coverage:.1%}")
            console.print(f"  Max iterations: {settings.coverage.max_iterations}")
            
            console.print("\n[dim]Use 'config llm' to manage LLM settings[/dim]")
        else:
            # Delegate to config command
            console.print(f"[dim]Use: pyutagent-cli config {' '.join(args)}[/dim]")
    
    async def _handle_help(self, args: List[str]) -> None:
        """Handle help command."""
        help_text = """
[bold]Available Commands:[/bold]

  [green]generate <file>[/green]     Generate unit tests for a Java file
  [green]gen <file>[/green]          Short alias for generate
  [green]plan [file][/green]          Show execution plan
  [green]config[/green]              Show current configuration
  [green]history[/green]             Show command history
  [green]status[/green]              Show current status
  [green]help[/green]                Show this help message
  [green]exit/quit[/green]           Exit interactive mode

[bold]Natural Language:[/bold]
  You can also type natural language queries:
  - "generate tests for MyClass.java"
  - "create unit tests for src/main/java/MyClass.java"
  - "show me the plan for MyClass.java"

[bold]Examples:[/bold]
  >>> generate src/main/java/com/example/UserService.java
  >>> plan
  >>> config
  >>> exit
        """
        console.print(Markdown(help_text))
    
    async def _handle_exit(self, args: List[str]) -> None:
        """Handle exit command."""
        self.running = False
    
    async def _handle_history(self, args: List[str]) -> None:
        """Handle history command."""
        if not self.context.history:
            console.print("[dim]No command history[/dim]")
            return
        
        console.print("\n[bold]Command History:[/bold]\n")
        for i, entry in enumerate(self.context.history[-20:], 1):  # Show last 20
            time = entry["timestamp"][:19]  # Trim to seconds
            console.print(f"  {i}. [{time}] {entry['input']}")
    
    async def _handle_status(self, args: List[str]) -> None:
        """Handle status command."""
        status_table = Table(title="CLI Status")
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="green")
        
        ctx = self.context
        status_table.add_row("State", ctx.state.name)
        status_table.add_row("Project", str(ctx.project_path) if ctx.project_path else "Not set")
        status_table.add_row("Output Format", ctx.output_format.value)
        status_table.add_row("Batch Mode", str(ctx.batch_mode))
        status_table.add_row("Verbose", str(ctx.verbose))
        status_table.add_row("History Count", str(len(ctx.history)))
        
        console.print(status_table)


# =============================================================================
# Batch Mode Output
# =============================================================================

class BatchOutput:
    """JSON output formatter for batch mode.
    
    Provides structured JSON output suitable for CI/CD integration.
    """
    
    @staticmethod
    def success(data: Dict[str, Any], message: str = "") -> None:
        """Output successful result as JSON."""
        output = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "data": data
        }
        print(json.dumps(output, indent=2, default=str))
    
    @staticmethod
    def error(message: str, code: str = "", details: Optional[Dict] = None) -> None:
        """Output error as JSON."""
        output = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": {
                "message": message,
                "code": code,
                "details": details or {}
            }
        }
        print(json.dumps(output, indent=2, default=str))
        sys.exit(1)
    
    @staticmethod
    def progress(step: str, progress: float, message: str = "") -> None:
        """Output progress update as JSON."""
        output = {
            "type": "progress",
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "progress": progress,
            "message": message
        }
        print(json.dumps(output, default=str))
        sys.stdout.flush()


# =============================================================================
# Main CLI Group
# =============================================================================

@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="pyutagent-cli")
@click.option('--batch', is_flag=True, help='Batch mode (JSON output)')
@click.option('--project', '-p', type=click.Path(), help='Project path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--format', 'output_format', 
              type=click.Choice(['text', 'json', 'markdown']),
              default='text', help='Output format')
@click.option('--interactive', '-i', is_flag=True, help='Force interactive mode')
@click.pass_context
def cli(ctx: click.Context, batch: bool, project: Optional[str], 
        verbose: bool, output_format: str, interactive: bool):
    """PyUT Agent CLI - AI-powered Java Unit Test Generator.
    
    Also supports general programming tasks via the Universal Agent.
    
    \b
    Usage Examples:
        pyutagent-cli                              # Interactive mode
        pyutagent-cli generate MyClass.java        # Generate tests
        pyutagent-cli plan MyClass.java            # View execution plan
        pyutagent-cli --batch generate ...         # Batch mode (JSON)
    
    \b
    For more help: https://github.com/pyutagent/docs
    """
    # Initialize context
    cli_ctx = get_cli_context()
    cli_ctx.batch_mode = batch
    cli_ctx.verbose = verbose
    cli_ctx.output_format = OutputFormat(output_format)
    
    if project:
        cli_ctx.project_path = Path(project)
        save_shared_config()
    
    # Store context in click context
    ctx.ensure_object(dict)
    ctx.obj['cli_context'] = cli_ctx
    
    # Handle interactive mode (no subcommand)
    if ctx.invoked_subcommand is None:
        if batch:
            # Batch mode requires a subcommand
            BatchOutput.error(
                "Batch mode requires a subcommand",
                "MISSING_COMMAND",
                {"available_commands": ["generate", "plan", "config", "scan"]}
            )
        else:
            # Start interactive mode
            session = InteractiveSession()
            asyncio.run(session.start())


# =============================================================================
# Command: Plan
# =============================================================================

@cli.command(name='plan')
@click.argument('file_path', type=click.Path(exists=True), required=False)
@click.option('--json-output', is_flag=True, help='Output as JSON')
@click.pass_context
def plan_command(ctx: click.Context, file_path: Optional[str], json_output: bool):
    """View execution plan for test generation.
    
    Shows the step-by-step plan that will be executed to generate tests.
    If no file is specified, shows the general plan structure.
    
    \b
    Examples:
        pyutagent-cli plan                          # Show general plan
        pyutagent-cli plan MyClass.java             # Plan for specific file
        pyutagent-cli plan MyClass.java --json-output  # JSON output
    """
    cli_ctx = ctx.obj['cli_context']
    
    if cli_ctx.batch_mode or json_output:
        # JSON output
        if file_path:
            from pyutagent.agent.execution.execution_plan import ExecutionPlan
            
            plan = ExecutionPlan.create_test_generation_plan(file_path)
            BatchOutput.success({
                "plan": plan.to_dict(),
                "file": file_path
            })
        else:
            BatchOutput.success({
                "plan_template": {
                    "steps": [
                        {"id": "parse", "name": "Parse Target", "type": "parse"},
                        {"id": "generate", "name": "Generate Tests", "type": "generate"},
                        {"id": "compile", "name": "Compile Tests", "type": "compile"},
                        {"id": "test", "name": "Run Tests", "type": "test"},
                        {"id": "analyze", "name": "Analyze Coverage", "type": "analyze"},
                        {"id": "optimize", "name": "Optimize Coverage", "type": "optimize"},
                    ]
                }
            })
    else:
        # Rich text output
        if file_path:
            from pyutagent.agent.execution.execution_plan import ExecutionPlan
            
            plan = ExecutionPlan.create_test_generation_plan(file_path)
            
            console.print(Panel(
                f"[bold]{plan.name}[/bold]\n"
                f"[dim]{plan.description}[/dim]",
                title="Execution Plan",
                border_style="blue"
            ))
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim")
            table.add_column("Step")
            table.add_column("Type", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Dependencies", style="yellow")
            
            for i, step in enumerate(plan.steps, 1):
                deps = ", ".join(step.dependencies) if step.dependencies else "-"
                table.add_row(
                    str(i),
                    step.name,
                    step.step_type.value,
                    step.description,
                    deps
                )
            
            console.print(table)
            console.print(f"\n[dim]Total steps: {len(plan.steps)}[/dim]")
        else:
            console.print(Panel(
                "[bold]Test Generation Execution Plan[/bold]",
                border_style="blue"
            ))
            
            table = Table()
            table.add_column("Step", style="cyan")
            table.add_column("Description", style="green")
            table.add_column("Purpose")
            
            table.add_row(
                "1. Parse",
                "Parse target file",
                "Extract class structure, methods, dependencies"
            )
            table.add_row(
                "2. Generate",
                "Generate initial tests",
                "Create test cases based on analysis"
            )
            table.add_row(
                "3. Compile",
                "Compile tests",
                "Verify generated code compiles"
            )
            table.add_row(
                "4. Test",
                "Run tests",
                "Execute tests and collect results"
            )
            table.add_row(
                "5. Analyze",
                "Analyze coverage",
                "Check code coverage metrics"
            )
            table.add_row(
                "6. Optimize",
                "Optimize coverage",
                "Generate additional tests for uncovered code"
            )
            
            console.print(table)


# =============================================================================
# Register existing commands
# =============================================================================

cli.add_command(scan_command)
cli.add_command(generate_command)
cli.add_command(generate_all_command)
cli.add_command(config_group, name='config')
cli.add_command(history_group, name='history')
cli.add_command(hooks_group, name='hooks')
cli.add_command(skills_group, name='skills')
cli.add_command(project_group, name='project')
cli.add_command(task_command, name='task')
cli.add_command(task_types_command, name='task-types')


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
