"""Project CLI commands for managing PYUT.md configuration."""

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


@click.group(name='project')
def project_group():
    """Manage project configuration via PYUT.md.
    
    PYUT.md files allow you to persist project understanding and
    Agent preferences, similar to CLAUDE.md.
    """
    pass


@project_group.command(name='init')
@click.option('--path', type=click.Path(), default='.', help='Project directory')
@click.option('--force', is_flag=True, help='Overwrite existing PYUT.md')
def init_project(path: str, force: bool):
    """Initialize a new PYUT.md configuration file.
    
    Example:
        pyutagent-cli project init
        pyutagent-cli project init --path /path/to/project
    """
    from pyutagent.core.project_config import create_config_template
    
    project_path = Path(path).resolve()
    config_file = project_path / "PYUT.md"
    
    if config_file.exists() and not force:
        console.print(f"[yellow]PYUT.md already exists at {config_file}[/yellow]")
        console.print("Use --force to overwrite")
        return
    
    create_config_template(config_file)
    
    console.print(Panel(
        f"[green]✓ Created PYUT.md[/green]\n\n"
        f"Location: [cyan]{config_file}[/cyan]\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"1. Edit PYUT.md to customize your project settings\n"
        f"2. Run 'pyutagent-cli project show' to view configuration",
        title="Project Initialized"
    ))


@project_group.command(name='show')
@click.option('--path', type=click.Path(), default='.', help='Project directory')
@click.option('--format', type=click.Choice(['text', 'json']), default='text', help='Output format')
def show_config(path: str, format: str):
    """Show current project configuration.
    
    Example:
        pyutagent-cli project show
        pyutagent-cli project show --format json
    """
    from pyutagent.core.project_config import load_project_config
    
    project_path = Path(path).resolve()
    config = load_project_config(project_path)
    
    if format == 'json':
        console.print_json(data=config.to_dict())
        return
    
    console.print(Panel(
        f"[bold cyan]Project: {config.project_name}[/bold cyan]\n"
        f"Root: {config.project_root}",
        title="Project Configuration"
    ))
    
    table = Table(title="Build Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Tool", config.build.tool.value)
    table.add_row("Java Version", config.build.java_version)
    table.add_row("Build Command", config.build.build_command)
    table.add_row("Test Command", config.build.test_command)
    console.print(table)
    
    table = Table(title="Testing Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Framework", config.testing.framework.value)
    table.add_row("Target Coverage", f"{config.testing.target_coverage:.0%}")
    table.add_row("Test Directory", config.testing.test_directory)
    table.add_row("Mock Framework", config.testing.mock_framework)
    console.print(table)
    
    table = Table(title="Agent Preferences")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Multi-Agent", str(config.agent.enable_multi_agent))
    table.add_row("Error Prediction", str(config.agent.enable_error_prediction))
    table.add_row("Self-Reflection", str(config.agent.enable_self_reflection))
    table.add_row("Max Iterations", str(config.agent.max_iterations))
    table.add_row("Preferred Strategies", ", ".join(config.agent.preferred_strategies))
    console.print(table)
    
    if config.dependencies:
        table = Table(title="Dependencies")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Enabled", style="yellow")
        for name, dep in config.dependencies.items():
            table.add_row(name, dep.version or "-", str(dep.enabled))
        console.print(table)
    
    if config.custom_instructions:
        console.print("\n[bold]Custom Instructions:[/bold]")
        for instruction in config.custom_instructions:
            console.print(f"  • {instruction}")
    
    if config.ignore_patterns:
        console.print("\n[bold]Ignore Patterns:[/bold]")
        for pattern in config.ignore_patterns:
            console.print(f"  • {pattern}")


@project_group.command(name='validate')
@click.option('--path', type=click.Path(), default='.', help='Project directory')
def validate_config(path: str):
    """Validate PYUT.md configuration.
    
    Example:
        pyutagent-cli project validate
    """
    from pyutagent.core.project_config import ProjectConfigLoader
    
    project_path = Path(path).resolve()
    loader = ProjectConfigLoader()
    
    config_file = loader.find_config_file(project_path)
    
    if not config_file:
        console.print("[yellow]No PYUT.md file found[/yellow]")
        console.print("Run 'pyutagent-cli project init' to create one")
        return
    
    console.print(f"[cyan]Validating: {config_file}[/cyan]\n")
    
    try:
        config = loader.load(project_path)
        
        issues = []
        
        if config.testing.target_coverage < 0 or config.testing.target_coverage > 1:
            issues.append("target_coverage should be between 0 and 1")
        
        if config.agent.max_iterations < 1:
            issues.append("max_iterations should be at least 1")
        
        if config.agent.timeout_per_file < 60:
            issues.append("timeout_per_file is very low (< 60s)")
        
        if issues:
            console.print("[yellow]⚠ Configuration issues found:[/yellow]")
            for issue in issues:
                console.print(f"  • {issue}")
        else:
            console.print("[green]✓ Configuration is valid[/green]")
        
        console.print(f"\n[bold]Configuration summary:[/bold]")
        console.print(f"  Build Tool: {config.build.tool.value}")
        console.print(f"  Test Framework: {config.testing.framework.value}")
        console.print(f"  Target Coverage: {config.testing.target_coverage:.0%}")
        
    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")


@project_group.command(name='set')
@click.argument('key')
@click.argument('value')
@click.option('--path', type=click.Path(), default='.', help='Project directory')
def set_config(key: str, value: str, path: str):
    """Set a configuration value.
    
    Example:
        pyutagent-cli project set testing.target_coverage 0.9
        pyutagent-cli project set agent.max_iterations 15
    """
    from pyutagent.core.project_config import load_project_config, ProjectConfigLoader
    
    project_path = Path(path).resolve()
    config = load_project_config(project_path)
    
    keys = key.split('.')
    if len(keys) < 2:
        console.print("[red]Key must be in format: section.setting[/red]")
        console.print("Example: testing.target_coverage")
        return
    
    section = keys[0]
    setting = '_'.join(keys[1:])
    
    try:
        if section == "testing":
            if setting == "target_coverage":
                val = float(value.rstrip('%'))
                if '%' in value:
                    val = val / 100
                config.testing.target_coverage = val
            elif setting == "framework":
                from pyutagent.core.project_config import TestFramework
                config.testing.framework = TestFramework(value.lower())
            elif hasattr(config.testing, setting):
                setattr(config.testing, setting, value)
        
        elif section == "agent":
            if value.lower() == "true":
                val = True
            elif value.lower() == "false":
                val = False
            elif value.isdigit():
                val = int(value)
            else:
                val = value
            
            if hasattr(config.agent, setting):
                setattr(config.agent, setting, val)
        
        elif section == "build":
            if hasattr(config.build, setting):
                setattr(config.build, setting, value)
        
        else:
            console.print(f"[red]Unknown section: {section}[/red]")
            return
        
        config_file = project_path / "PYUT.md"
        if not config_file.exists():
            loader = ProjectConfigLoader()
            loader.create_template(config_file)
        
        console.print(f"[green]✓ Set {key} = {value}[/green]")
        console.print("[yellow]Note: PYUT.md file needs to be manually updated[/yellow]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to set {key}: {e}[/red]")


@project_group.command(name='list')
@click.option('--path', type=click.Path(), default='.', help='Project directory')
def list_files(path: str):
    """List Java files in the project.
    
    Example:
        pyutagent-cli project list
    """
    from pyutagent.core.project_config import load_project_config
    
    project_path = Path(path).resolve()
    config = load_project_config(project_path)
    
    src_main_java = project_path / config.build.tool.value / "src" / "main" / "java"
    if not src_main_java.exists():
        src_main_java = project_path / "src" / "main" / "java"
    
    if not src_main_java.exists():
        console.print("[yellow]No src/main/java directory found[/yellow]")
        return
    
    java_files = list(src_main_java.rglob("*.java"))
    
    if not java_files:
        console.print("[yellow]No Java files found[/yellow]")
        return
    
    table = Table(title=f"Java Files ({len(java_files)})")
    table.add_column("#", style="dim")
    table.add_column("File", style="cyan")
    table.add_column("Package", style="green")
    
    for i, java_file in enumerate(java_files[:50], 1):
        rel_path = java_file.relative_to(src_main_java)
        package = str(rel_path.parent).replace('/', '.').replace('\\', '.')
        if package == '.':
            package = "(default)"
        table.add_row(str(i), java_file.name, package)
    
    console.print(table)
    
    if len(java_files) > 50:
        console.print(f"\n[yellow]... and {len(java_files) - 50} more files[/yellow]")


@project_group.command(name='info')
@click.option('--path', type=click.Path(), default='.', help='Project directory')
def project_info(path: str):
    """Show project information and statistics.
    
    Example:
        pyutagent-cli project info
    """
    from pyutagent.core.project_config import load_project_config
    
    project_path = Path(path).resolve()
    config = load_project_config(project_path)
    
    pom_file = project_path / "pom.xml"
    gradle_file = project_path / "build.gradle"
    
    console.print(Panel(
        f"[bold cyan]{config.project_name}[/bold cyan]\n"
        f"Path: {config.project_root}",
        title="Project Information"
    ))
    
    stats = {
        "Build System": "Maven" if pom_file.exists() else ("Gradle" if gradle_file.exists() else "Unknown"),
        "Config File": "PYUT.md exists" if (project_path / "PYUT.md").exists() else "No PYUT.md",
    }
    
    src_main = project_path / "src" / "main" / "java"
    src_test = project_path / "src" / "test" / "java"
    
    if src_main.exists():
        java_files = list(src_main.rglob("*.java"))
        stats["Source Files"] = str(len(java_files))
    
    if src_test.exists():
        test_files = list(src_test.rglob("*.java"))
        stats["Test Files"] = str(len(test_files))
    
    table = Table(title="Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for metric, value in stats.items():
        table.add_row(metric, value)
    
    console.print(table)
