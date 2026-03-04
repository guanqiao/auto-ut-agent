"""History command for CLI - manage project history."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group(name='history')
def history_group():
    """Manage project history."""
    pass


@history_group.command(name='list')
def history_list():
    """List recent projects."""
    try:
        from pyutagent.core.config import load_app_state

        app_state = load_app_state()

        if not app_state.recent_projects:
            console.print("[yellow]No recent projects found[/yellow]")
            return

        table = Table(title="Recent Projects")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Project", style="green")
        table.add_column("Path", style="blue")
        table.add_column("Last Opened", style="yellow")

        for i, project in enumerate(app_state.recent_projects[:10], 1):
            path = Path(project.path)
            if not path.exists():
                status = "[red]✗ Not found[/red]"
            else:
                status = ""

            table.add_row(
                str(i),
                f"{project.name} {status}",
                str(project.path),
                project.last_opened[:19] if project.last_opened else "-"
            )

        console.print(table)

        if app_state.last_project_path:
            console.print()
            console.print(f"[bold]Last project:[/bold] {app_state.last_project_path}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@history_group.command(name='clear')
def history_clear():
    """Clear project history."""
    try:
        from pyutagent.core.config import load_app_state, save_app_state

        app_state = load_app_state()

        if not app_state.recent_projects:
            console.print("[yellow]No projects in history[/yellow]")
            return

        count = len(app_state.recent_projects)

        if click.confirm(f"Clear {count} project(s) from history?"):
            app_state.recent_projects.clear()
            app_state.last_project_path = None
            save_app_state(app_state)

            console.print(f"[green]✓ Cleared {count} project(s) from history[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@history_group.command(name='remove')
@click.argument('project_path', type=click.Path(exists=False))
def history_remove(project_path: str):
    """Remove a project from history."""
    try:
        from pyutagent.core.config import load_app_state, save_app_state

        app_state = load_app_state()

        if app_state.remove_project(project_path):
            save_app_state(app_state)
            console.print(f"[green]✓ Removed project from history: {project_path}[/green]")
        else:
            console.print(f"[yellow]Project not found in history: {project_path}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
