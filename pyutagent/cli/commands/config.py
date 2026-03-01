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
        from pyutagent.config import load_llm_config

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
        from pyutagent.config import load_llm_config, save_llm_config
        from pyutagent.llm.config import LLMConfig, LLMProvider

        collection = load_llm_config()

        # Create new config
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
        from pyutagent.config import load_llm_config, save_llm_config

        collection = load_llm_config()

        # Find config by ID (can be partial match)
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
        from pyutagent.config import load_llm_config
        from pyutagent.llm.client import LLMClient

        collection = load_llm_config()

        if config_id == 'default':
            config = collection.get_default_config()
        else:
            # Find config by ID (can be partial match)
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
        from pyutagent.config import load_llm_config

        collection = load_llm_config()

        console.print("[bold]Configuration Directory:[/bold]")
        # Config directory is typically ~/.pyutagent
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
    console.print("[yellow]Aider configuration management coming soon[/yellow]")
