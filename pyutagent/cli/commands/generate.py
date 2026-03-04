"""Generate command for CLI."""

import asyncio
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
@click.option('-i', '--incremental', is_flag=True, help='Enable incremental mode (preserve existing passing tests)')
@click.option('--skip-analysis', is_flag=True, help='Skip running existing tests, just analyze file content')
def generate_command(
    file_path: str,
    llm: str,
    output_dir: str,
    coverage_target: int,
    max_iterations: int,
    watch: bool,
    incremental: bool,
    skip_analysis: bool
):
    """Generate unit tests for a Java file."""
    file_path = Path(file_path)

    if file_path.suffix != '.java':
        console.print(f"[red]Error: {file_path} is not a Java file[/red]")
        raise click.Abort()

    console.print(f"[blue]Generating tests for {file_path.name}...[/blue]")
    console.print(f"  LLM: {llm}")
    console.print(f"  Coverage target: {coverage_target}%")
    console.print(f"  Max iterations: {max_iterations}")
    if incremental:
        console.print(f"  [green]Incremental mode: ENABLED[/green] (will preserve existing passing tests)")
    console.print()

    try:
        from pyutagent.agent.react_agent import ReActAgent
        from pyutagent.memory.working_memory import WorkingMemory
        from pyutagent.config import load_llm_config
        from pyutagent.llm.client import LLMClient
        
        config_collection = load_llm_config()
        if llm == 'default':
            llm_config = config_collection.get_default_config()
        else:
            llm_config = None
            for config in config_collection.configs:
                if config.name == llm or config.id.startswith(llm):
                    llm_config = config
                    break

        if not llm_config:
            console.print(f"[red]Error: LLM configuration '{llm}' not found[/red]")
            raise click.Abort()

        llm_client = LLMClient.from_config(llm_config)
        
        working_memory = WorkingMemory(
            target_coverage=coverage_target / 100.0,
            max_iterations=max_iterations,
            current_file=str(file_path)
        )
        
        project_path = file_path.parent
        while project_path.parent and not (project_path / 'pom.xml').exists():
            project_path = project_path.parent
        
        agent = ReActAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=str(project_path),
            incremental_mode=incremental,
            skip_test_analysis=skip_analysis,
        )

        if watch:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Generating tests...", total=None)

                result = asyncio.run(agent.generate_tests(str(file_path)))
        else:
            result = asyncio.run(agent.generate_tests(str(file_path)))

        if result.success:
            console.print(f"[green]✓ Tests generated successfully![/green]")
            if result.test_file:
                console.print(f"  Test file: {result.test_file}")
            if result.coverage:
                console.print(f"  Coverage: {result.coverage:.1f}%")
            if incremental and hasattr(result, 'preserved_count'):
                console.print(f"  Preserved tests: {result.preserved_count}")
        else:
            console.print(f"[red]✗ Test generation failed[/red]")
            if result.message:
                console.print(f"  Error: {result.message}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
