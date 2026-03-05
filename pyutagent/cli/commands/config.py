"""Config command for CLI."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group(name='config')
def config_group():
    """Manage configuration."""
    pass


@config_group.group(name='llm')
def llm_group():
    """Manage LLM configurations."""
    pass


@llm_group.command(name='list')
def llm_list():
    """List all LLM configurations."""
    try:
        from pyutagent.core.config import load_llm_config

        collection = load_llm_config()

        if not collection.configs:
            console.print("[yellow]No LLM configurations found[/yellow]")
            return

        table = Table(title="LLM Configurations")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="blue")
        table.add_column("Default", style="yellow")

        for config in collection.configs:
            is_default = "✓" if config.id == collection.default_config_id else ""
            table.add_row(
                config.id[:8],
                config.name or "-",
                str(config.provider),
                config.model,
                is_default
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@llm_group.command(name='add')
@click.option('--name', required=True, help='Configuration name')
@click.option('--provider', required=True,
              type=click.Choice(['openai', 'azure', 'anthropic', 'deepseek', 'ollama', 'custom']),
              help='LLM provider')
@click.option('--model', required=True, help='Model name')
@click.option('--api-key', help='API key')
@click.option('--endpoint', help='API endpoint URL')
@click.option('--timeout', type=int, default=60, help='Request timeout in seconds')
def llm_add(name: str, provider: str, model: str, api_key: str, endpoint: str, timeout: int):
    """Add a new LLM configuration."""
    try:
        from pyutagent.core.config import load_llm_config, save_llm_config
        from pyutagent.llm.config import LLMConfig, LLMProvider

        collection = load_llm_config()

        config = LLMConfig(
            name=name,
            provider=LLMProvider(provider),
            model=model,
            api_key=api_key or "",
            endpoint=endpoint or "",
            timeout=timeout
        )

        collection.add_config(config)
        save_llm_config(collection)

        console.print(f"[green]✓ Added LLM configuration '{name}' (ID: {config.id[:8]})[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@llm_group.command(name='set-default')
@click.argument('config_id')
def llm_set_default(config_id: str):
    """Set the default LLM configuration by ID."""
    try:
        from pyutagent.core.config import load_llm_config, save_llm_config

        collection = load_llm_config()

        found_config = None
        for config in collection.configs:
            if config.id.startswith(config_id):
                found_config = config
                break

        if not found_config:
            console.print(f"[red]Error: Configuration with ID '{config_id}' not found[/red]")
            raise click.Abort()

        collection.set_default_config(found_config.id)
        save_llm_config(collection)

        console.print(f"[green]✓ Set '{found_config.name or found_config.id[:8]}' as default configuration[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@llm_group.command(name='test')
@click.argument('config_id')
def llm_test(config_id: str):
    """Test an LLM configuration by ID."""
    try:
        from pyutagent.core.config import load_llm_config
        from pyutagent.llm.client import LLMClient

        collection = load_llm_config()

        if config_id == 'default':
            config = collection.get_default_config()
        else:
            config = None
            for c in collection.configs:
                if c.id.startswith(config_id):
                    config = c
                    break

        if not config:
            console.print(f"[red]Error: Configuration '{config_id}' not found[/red]")
            raise click.Abort()

        console.print(f"Testing connection to {config.provider}...")

        client = LLMClient(config)
        success = client.test_connection()

        if success:
            console.print(f"[green]✓ Connection successful![/green]")
        else:
            console.print(f"[red]✗ Connection failed[/red]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@config_group.command(name='show')
def config_show():
    """Show current configuration."""
    try:
        from pyutagent.core.config import load_llm_config

        collection = load_llm_config()

        console.print("[bold]Configuration Directory:[/bold]")
        config_dir = Path.home() / '.pyutagent'
        console.print(f"  {config_dir}")
        console.print()

        console.print("[bold]Default LLM Configuration:[/bold]")
        default = collection.get_default_config()
        if default:
            console.print(f"  ID: {default.id[:8]}")
            console.print(f"  Name: {default.name or '-'}")
            console.print(f"  Provider: {default.provider}")
            console.print(f"  Model: {default.model}")
        else:
            console.print("  [yellow]No default configuration set[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@config_group.group(name='aider')
def aider_group():
    """Manage Aider configuration."""
    pass


@aider_group.command(name='show')
def aider_show():
    """Show Aider configuration."""
    try:
        from pyutagent.core.config import load_aider_config

        config = load_aider_config()

        console.print("[bold]Aider Configuration:[/bold]")
        console.print(f"  Max attempts: {config.max_attempts}")
        console.print(f"  Enable fallback: {config.enable_fallback}")
        console.print(f"  Enable circuit breaker: {config.enable_circuit_breaker}")
        console.print(f"  Timeout: {config.timeout_seconds}s")
        console.print()
        console.print(f"  Use Architect/Editor: {config.use_architect_editor}")
        if config.use_architect_editor:
            console.print(f"    Architect model ID: {config.architect_model_id or 'Not set'}")
            console.print(f"    Editor model ID: {config.editor_model_id or 'Not set'}")
            console.print(f"    Mode: {config.architect_mode.value}")
        console.print()
        console.print(f"  Enable multi-file: {config.enable_multi_file}")
        if config.enable_multi_file:
            console.print(f"    Max files per edit: {config.max_files_per_edit}")
        console.print()
        console.print(f"  Auto detect format: {config.auto_detect_format}")
        console.print(f"  Preferred format: {config.preferred_format or 'Auto'}")
        console.print()
        console.print(f"  Track costs: {config.track_costs}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@config_group.group(name='maven')
def maven_group():
    """Manage Maven configuration."""
    pass


@maven_group.command(name='show')
def maven_show():
    """Show Maven configuration."""
    try:
        from pyutagent.core.config import get_settings

        settings = get_settings()

        console.print("[bold]Maven Configuration:[/bold]")
        if settings.maven.maven_path:
            console.print(f"  Maven path: {settings.maven.maven_path}")
        else:
            console.print("  Maven path: [yellow]Auto-detect[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@maven_group.command(name='set')
@click.option('--path', required=True, help='Path to Maven executable')
def maven_set(path: str):
    """Set Maven executable path."""
    try:
        from pyutagent.core.config import get_settings, save_app_config
        from pathlib import Path

        maven_path = Path(path)
        if not maven_path.exists():
            console.print(f"[red]Error: Maven executable not found at {path}[/red]")
            raise click.Abort()

        settings = get_settings()
        settings.maven.maven_path = str(maven_path)
        save_app_config(settings)

        console.print(f"[green]✓ Maven path set to: {maven_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@config_group.group(name='jdk')
def jdk_group():
    """Manage JDK configuration."""
    pass


@jdk_group.command(name='show')
def jdk_show():
    """Show JDK configuration."""
    try:
        from pyutagent.core.config import get_settings

        settings = get_settings()

        console.print("[bold]JDK Configuration:[/bold]")
        if settings.jdk.java_home:
            console.print(f"  JAVA_HOME: {settings.jdk.java_home}")
        else:
            console.print("  JAVA_HOME: [yellow]Auto-detect[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@jdk_group.command(name='set')
@click.option('--path', required=True, help='Path to JDK home directory')
def jdk_set(path: str):
    """Set JAVA_HOME path."""
    try:
        from pyutagent.core.config import get_settings, save_app_config
        from pathlib import Path

        java_home = Path(path)
        if not java_home.exists():
            console.print(f"[red]Error: JDK directory not found at {path}[/red]")
            raise click.Abort()

        settings = get_settings()
        settings.jdk.java_home = str(java_home)
        save_app_config(settings)

        console.print(f"[green]✓ JAVA_HOME set to: {java_home}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@config_group.group(name='coverage')
def coverage_group():
    """Manage coverage configuration."""
    pass


@coverage_group.command(name='show')
def coverage_show():
    """Show coverage configuration."""
    try:
        from pyutagent.core.config import get_settings

        settings = get_settings()

        console.print("[bold]Coverage Configuration:[/bold]")
        console.print(f"  Target coverage: {settings.coverage.target_coverage:.1%}")
        console.print(f"  Min coverage: {settings.coverage.min_coverage:.1%}")
        console.print(f"  Max iterations: {settings.coverage.max_iterations}")
        console.print(f"  Max compilation attempts: {settings.coverage.max_compilation_attempts}")
        console.print(f"  Max test attempts: {settings.coverage.max_test_attempts}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@coverage_group.command(name='set')
@click.option('--target', type=float, help='Target coverage (0.0-1.0)')
@click.option('--min', type=float, help='Minimum coverage (0.0-1.0)')
@click.option('--max-iterations', type=int, help='Maximum iterations')
@click.option('--max-compilation-attempts', type=int, help='Maximum compilation attempts')
@click.option('--max-test-attempts', type=int, help='Maximum test attempts')
def coverage_set(
    target: float,
    min: float,
    max_iterations: int,
    max_compilation_attempts: int,
    max_test_attempts: int
):
    """Set coverage configuration."""
    try:
        from pyutagent.core.config import get_settings, save_app_config

        settings = get_settings()

        if target is not None:
            if not 0.0 <= target <= 1.0:
                console.print("[red]Error: Target coverage must be between 0.0 and 1.0[/red]")
                raise click.Abort()
            settings.coverage.target_coverage = target
            console.print(f"[green]✓ Target coverage set to: {target:.1%}[/green]")

        if min is not None:
            if not 0.0 <= min <= 1.0:
                console.print("[red]Error: Min coverage must be between 0.0 and 1.0[/red]")
                raise click.Abort()
            settings.coverage.min_coverage = min
            console.print(f"[green]✓ Min coverage set to: {min:.1%}[/green]")

        if max_iterations is not None:
            if max_iterations < 1:
                console.print("[red]Error: Max iterations must be at least 1[/red]")
                raise click.Abort()
            settings.coverage.max_iterations = max_iterations
            console.print(f"[green]✓ Max iterations set to: {max_iterations}[/green]")

        if max_compilation_attempts is not None:
            if max_compilation_attempts < 1:
                console.print("[red]Error: Max compilation attempts must be at least 1[/red]")
                raise click.Abort()
            settings.coverage.max_compilation_attempts = max_compilation_attempts
            console.print(f"[green]✓ Max compilation attempts set to: {max_compilation_attempts}[/green]")

        if max_test_attempts is not None:
            if max_test_attempts < 1:
                console.print("[red]Error: Max test attempts must be at least 1[/red]")
                raise click.Abort()
            settings.coverage.max_test_attempts = max_test_attempts
            console.print(f"[green]✓ Max test attempts set to: {max_test_attempts}[/green]")

        save_app_config(settings)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
