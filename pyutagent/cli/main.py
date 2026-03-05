"""Main CLI entry point for PyUT Agent."""

import click
from rich.console import Console

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

console = Console()


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="pyutagent-cli")
@click.pass_context
def cli(ctx):
    """PyUT Agent CLI - AI-powered Java Unit Test Generator.
    
    Also supports general programming tasks via the Universal Agent.
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register commands
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


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
