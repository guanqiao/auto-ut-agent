"""Scan command for CLI."""

import os
from pathlib import Path

import click
from rich.console import Console
from rich.tree import Tree

console = Console()


@click.command(name='scan')
@click.argument('project_path', type=click.Path(exists=True))
@click.option('--tree', is_flag=True, help='Display as tree structure')
def scan_command(project_path: str, tree: bool):
    """Scan a Maven project and list Java files."""
    project_path = Path(project_path)

    # Check if it's a valid Maven project
    pom_file = project_path / 'pom.xml'
    if not pom_file.exists():
        console.print(f"[red]Error: {project_path} is not a valid Maven project (pom.xml not found)[/red]")
        raise click.Abort()

    # Find Java source directory
    src_dir = project_path / 'src' / 'main' / 'java'
    if not src_dir.exists():
        console.print(f"[yellow]Warning: Java source directory not found at {src_dir}[/yellow]")
        return

    # Find all Java files
    java_files = list(src_dir.rglob('*.java'))

    if not java_files:
        console.print(f"[yellow]No Java files found in {src_dir}[/yellow]")
        return

    console.print(f"[green]Found {len(java_files)} Java files in {project_path.name}[/green]")
    console.print()

    if tree:
        # Display as tree
        root_tree = Tree(f"📁 {project_path.name}")
        src_tree = root_tree.add("src/main/java")

        # Build tree structure
        for java_file in sorted(java_files):
            rel_path = java_file.relative_to(src_dir)
            parts = rel_path.parts

            current = src_tree
            for part in parts[:-1]:
                # Find or create subdirectory
                found = False
                for child in current.children:
                    if hasattr(child, 'label') and str(part) in str(child.label):
                        current = child
                        found = True
                        break
                if not found:
                    current = current.add(f"📁 {part}")

            current.add(f"📄 {parts[-1]}")

        console.print(root_tree)
    else:
        # Display as list
        for java_file in sorted(java_files):
            rel_path = java_file.relative_to(src_dir)
            console.print(f"  📄 {rel_path}")
