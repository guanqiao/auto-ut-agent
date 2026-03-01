"""Main CLI entry point for PyUT Agent."""

import click
from rich.console import Console

from . import __version__
from .commands.scan import scan_command
from .commands.generate import generate_command
from .commands.config import config_group

console = Console()


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="pyutagent-cli")
@click.pass_context
def cli(ctx):
    """PyUT Agent CLI - AI-powered Java Unit Test Generator."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register commands
cli.add_command(scan_command)
cli.add_command(generate_command)
cli.add_command(config_group, name='config')


def main():
    """CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
