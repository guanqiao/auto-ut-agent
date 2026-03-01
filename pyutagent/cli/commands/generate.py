"""Generate command for CLI."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.command(name='generate')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--llm', default='default', help='LLM configuration to use')
@click.option('--output-dir', type=click.Path(), help='Output directory for test files')
@click.option('--coverage-target', type=int, default=80, help='Target coverage percentage')
@click.option('--max-iterations', type=int, default=10, help='Maximum iterations')
@click.option('--watch', is_flag=True, help='Watch progress in real-time')
def generate_command(
    file_path: str,
    llm: str,
    output_dir: str,
    coverage_target: int,
    max_iterations: int,
    watch: bool
):
    """Generate unit tests for a Java file."""
    file_path = Path(file_path)

    # Validate file extension
    if file_path.suffix != '.java':
        console.print(f"[red]Error: {file_path} is not a Java file[/red]")
        raise click.Abort()

    console.print(f"[blue]Generating tests for {file_path.name}...[/blue]")
    console.print(f"  LLM: {llm}")
    console.print(f"  Coverage target: {coverage_target}%")
    console.print(f"  Max iterations: {max_iterations}")
    console.print()

    try:
        # Import here to avoid slow startup
        from pyutagent.agent.test_generator import TestGeneratorAgent
        from pyutagent.config import load_llm_config

        # Load LLM configuration
        config_collection = load_llm_config()
        if llm == 'default':
            llm_config = config_collection.get_default_config()
        else:
            # Find config by name or ID
            llm_config = None
            for config in config_collection.configs:
                if config.name == llm or config.id.startswith(llm):
                    llm_config = config
                    break

        if not llm_config:
            console.print(f"[red]Error: LLM configuration '{llm}' not found[/red]")
            raise click.Abort()

        # Create agent and generate tests
        agent = TestGeneratorAgent(llm_config=llm_config)

        if watch:
            # Show progress with rich
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating tests...", total=None)

                result = agent.generate_tests(
                    target_file=str(file_path),
                    output_dir=output_dir,
                    coverage_target=coverage_target,
                    max_iterations=max_iterations
                )
        else:
            result = agent.generate_tests(
                target_file=str(file_path),
                output_dir=output_dir,
                coverage_target=coverage_target,
                max_iterations=max_iterations
            )

        if result.get('success'):
            console.print(f"[green]✓ Tests generated successfully![/green]")
            if result.get('test_file'):
                console.print(f"  Test file: {result['test_file']}")
            if result.get('coverage'):
                console.print(f"  Coverage: {result['coverage']:.1f}%")
        else:
            console.print(f"[red]✗ Test generation failed[/red]")
            if result.get('error'):
                console.print(f"  Error: {result['error']}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
