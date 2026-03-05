"""Skills CLI commands for managing reusable skill libraries."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


@click.group(name='skills')
def skills_group():
    """Manage reusable skill libraries for Agent automation.
    
    Skills are reusable capabilities that can be invoked with specific
    parameters and context. They encapsulate best practices for common tasks.
    """
    pass


@skills_group.command(name='list')
@click.option('--category', type=str, help='Filter by category')
@click.option('--tag', type=str, help='Filter by tag')
def list_skills(category: Optional[str], tag: Optional[str]):
    """List all registered skills."""
    from pyutagent.agent.skills import get_skill_registry, SkillCategory
    
    registry = get_skill_registry()
    
    if category:
        try:
            cat = SkillCategory[category.upper()]
            skills = registry.list_skills(category=cat)
        except KeyError:
            console.print(f"[red]Invalid category: {category}[/red]")
            console.print("Valid categories: " + ", ".join(c.name for c in SkillCategory))
            return
    elif tag:
        skills = registry.list_by_tag(tag)
    else:
        skills = registry.list_skills()
    
    if not skills:
        console.print("[yellow]No skills found[/yellow]")
        return
    
    table = Table(title="Registered Skills")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Tags", style="magenta")
    table.add_column("Description", style="white")
    
    for skill_name in skills:
        info = registry.get_skill_info(skill_name)
        if info:
            table.add_row(
                skill_name,
                info.get('category', 'UNKNOWN'),
                info.get('version', '1.0.0'),
                ', '.join(info.get('tags', [])[:3]),
                info.get('description', '')[:50] + "..." if len(info.get('description', '')) > 50 else info.get('description', '')
            )
    
    console.print(table)


@skills_group.command(name='info')
@click.argument('skill_name')
def skill_info(skill_name: str):
    """Show detailed information about a skill."""
    from pyutagent.agent.skills import get_skill_registry
    
    registry = get_skill_registry()
    info = registry.get_skill_info(skill_name)
    
    if not info:
        console.print(f"[red]Skill not found: {skill_name}[/red]")
        return
    
    console.print(Panel(
        f"[bold cyan]{info['name']}[/bold cyan] v{info['version']}\n\n"
        f"[bold]Category:[/bold] {info['category']}\n"
        f"[bold]Author:[/bold] {info.get('author', 'Unknown') or 'Unknown'}\n\n"
        f"[bold]Description:[/bold]\n{info['description']}\n",
        title="Skill Information"
    ))
    
    if info.get('tags'):
        console.print(f"\n[bold]Tags:[/bold] {', '.join(info['tags'])}")
    
    if info.get('triggers'):
        console.print(f"\n[bold]Triggers:[/bold] {', '.join(info['triggers'])}")
    
    if info.get('requires_tools'):
        console.print(f"\n[bold]Required Tools:[/bold] {', '.join(info['requires_tools'])}")
    
    if info.get('best_practices'):
        console.print("\n[bold]Best Practices:[/bold]")
        for practice in info['best_practices']:
            console.print(f"  • {practice}")
    
    if info.get('examples'):
        console.print("\n[bold]Examples:[/bold]")
        for i, example in enumerate(info['examples'][:3], 1):
            console.print(f"  {i}. {example.get('description', 'Example')}")


@skills_group.command(name='categories')
def list_categories():
    """List all skill categories."""
    from pyutagent.agent.skills import SkillCategory
    
    table = Table(title="Skill Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Description", style="white")
    
    category_descriptions = {
        SkillCategory.CODE_GENERATION: "Code generation and synthesis",
        SkillCategory.CODE_REVIEW: "Code review and quality analysis",
        SkillCategory.DEBUGGING: "Debugging and error fixing",
        SkillCategory.REFACTORING: "Code refactoring and optimization",
        SkillCategory.TESTING: "Test generation and execution",
        SkillCategory.DOCUMENTATION: "Documentation generation",
        SkillCategory.DEPLOYMENT: "Deployment and DevOps tasks",
        SkillCategory.CODE_EXPLANATION: "Code explanation and analysis",
        SkillCategory.CUSTOM: "Custom user-defined skills",
    }
    
    for category, desc in category_descriptions.items():
        table.add_row(category.name, desc)
    
    console.print(table)


@skills_group.command(name='search')
@click.argument('query')
def search_skills(query: str):
    """Search skills by name or description."""
    from pyutagent.agent.skills import get_skill_registry
    
    registry = get_skill_registry()
    results = registry.search(query)
    
    if not results:
        console.print(f"[yellow]No skills found matching: {query}[/yellow]")
        return
    
    console.print(f"[green]Found {len(results)} skill(s):[/green]\n")
    
    for skill_name in results:
        info = registry.get_skill_info(skill_name)
        if info:
            console.print(f"  [cyan]{skill_name}[/cyan]: {info['description'][:60]}...")


@skills_group.command(name='execute')
@click.argument('skill_name')
@click.option('--params', type=str, help='JSON parameters for skill execution')
@click.option('--param-file', type=click.Path(exists=True), help='JSON file with parameters')
def execute_skill(skill_name: str, params: Optional[str], param_file: Optional[str]):
    """Execute a skill with given parameters.
    
    Example:
        pyutagent-cli skills execute generate_unit_test --params '{"file": "MyClass.java"}'
    """
    from pyutagent.agent.skills import get_skill_registry, SkillInput
    
    registry = get_skill_registry()
    skill = registry.get(skill_name)
    
    if not skill:
        console.print(f"[red]Skill not found: {skill_name}[/red]")
        return
    
    parameters = {}
    
    if param_file:
        with open(param_file, 'r', encoding='utf-8') as f:
            parameters = json.load(f)
    elif params:
        try:
            parameters = json.loads(params)
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON parameters[/red]")
            return
    
    console.print(f"[cyan]Executing skill: {skill_name}...[/cyan]")
    
    async def run_skill():
        input_data = SkillInput(parameters=parameters)
        return await skill.execute(input_data)
    
    try:
        result = asyncio.run(run_skill())
        
        if result.success:
            console.print(Panel(
                f"[green]✓ Skill executed successfully[/green]\n\n"
                f"[bold]Result:[/bold]\n{json.dumps(result.result, indent=2, default=str) if result.result else 'No output'}",
                title="Execution Result"
            ))
        else:
            console.print(Panel(
                f"[red]✗ Skill execution failed[/red]\n\n"
                f"[bold]Error:[/bold] {result.error}",
                title="Execution Failed"
            ))
        
        if result.logs:
            console.print("\n[bold]Logs:[/bold]")
            for log in result.logs:
                console.print(f"  {log}")
                
    except Exception as e:
        console.print(f"[red]✗ Execution error: {e}[/red]")


@skills_group.command(name='load')
@click.argument('path', type=click.Path(exists=True))
def load_skills(path: str):
    """Load skills from a file or directory.
    
    Example:
        pyutagent-cli skills load ./my_skills/
    """
    from pyutagent.agent.skills import get_skill_registry, SkillLoader
    
    registry = get_skill_registry()
    loader = SkillLoader(registry)
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        skill = loader.load_from_file(path)
        if skill:
            console.print(f"[green]✓ Loaded skill: {skill.name}[/green]")
        else:
            console.print(f"[red]✗ Failed to load skill from: {path}[/red]")
    elif path_obj.is_dir():
        before = len(registry.list_skills())
        loader.load_from_directory(path)
        after = len(registry.list_skills())
        console.print(f"[green]✓ Loaded {after - before} skill(s) from directory[/green]")
    else:
        console.print(f"[red]✗ Invalid path: {path}[/red]")


@skills_group.command(name='load-builtin')
def load_builtin():
    """Load built-in skills."""
    from pyutagent.agent.skills import get_skill_registry, SkillLoader
    
    registry = get_skill_registry()
    loader = SkillLoader(registry)
    
    before = len(registry.list_skills())
    loader.load_builtin_skills()
    after = len(registry.list_skills())
    
    console.print(f"[green]✓ Loaded {after - before} built-in skill(s)[/green]")
    
    console.print("\n[bold]Built-in Skills:[/bold]")
    for name in registry.list_skills():
        info = registry.get_skill_info(name)
        if info:
            console.print(f"  [cyan]{name}[/cyan]: {info['description'][:50]}...")


@skills_group.command(name='history')
@click.option('--skill', type=str, help='Filter by skill name')
@click.option('--limit', type=int, default=10, help='Number of records to show')
def execution_history(skill: Optional[str], limit: int):
    """Show skill execution history."""
    from pyutagent.agent.skills import get_skill_registry, EnhancedSkillExecutor
    
    registry = get_skill_registry()
    executor = EnhancedSkillExecutor(registry)
    
    history = executor.get_execution_history(skill_name=skill, limit=limit)
    
    if not history:
        console.print("[yellow]No execution history[/yellow]")
        return
    
    table = Table(title="Skill Execution History")
    table.add_column("Skill", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Retries", style="yellow")
    table.add_column("Error", style="red")
    
    for record in history:
        table.add_row(
            record.get('skill_name', 'Unknown'),
            "✓ Success" if record.get('success') else "✗ Failed",
            str(record.get('retry_count', 0)),
            record.get('error', '')[:40] if record.get('error') else "-"
        )
    
    console.print(table)


@skills_group.command(name='create')
@click.argument('skill_name')
@click.option('--category', type=str, default='CUSTOM', help='Skill category')
@click.option('--description', type=str, default='', help='Skill description')
@click.option('--output', type=click.Path(), help='Output file path')
def create_skill(skill_name: str, category: str, description: str, output: Optional[str]):
    """Create a new skill template.
    
    Example:
        pyutagent-cli skills create my_custom_skill --category TESTING --output ./my_skill.py
    """
    template = f'''"""Custom skill: {skill_name}"""

import asyncio
from typing import Any, Dict
from pyutagent.agent.skills import (
    Skill, SkillMetadata, SkillInput, SkillOutput, SkillCategory
)


class {skill_name.title().replace("_", "")}Skill(Skill):
    """Skill: {skill_name}
    
    {description or 'Description of what this skill does.'}
    """
    
    def _create_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="{skill_name}",
            description="{description or 'Description of what this skill does.'}",
            category=SkillCategory.{category.upper()},
            version="1.0.0",
            author="Your Name",
            tags=["custom", "{category.lower()}"],
            triggers=["{skill_name}", "trigger keyword"],
            best_practices=[
                "Best practice 1",
                "Best practice 2",
            ],
            requires_tools=[],
        )
    
    async def execute(self, input_data: SkillInput) -> SkillOutput:
        """Execute the skill.
        
        Args:
            input_data: Skill input with parameters and context
            
        Returns:
            SkillOutput with execution result
        """
        parameters = input_data.parameters
        context = input_data.context
        
        # TODO: Implement skill logic here
        result = {{
            "status": "success",
            "message": "Skill executed successfully",
            "parameters": parameters,
        }}
        
        return SkillOutput(
            success=True,
            result=result,
            logs=[f"Executed skill: {{self.name}}"],
            metadata={{"execution_time": "0.1s"}}
        )


# Register the skill
def register():
    from pyutagent.agent.skills import register_skill
    register_skill({skill_name.title().replace("_", "")}Skill())


if __name__ == "__main__":
    register()
'''
    
    output_path = Path(output) if output else Path(f"./{skill_name}_skill.py")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    console.print(Panel(
        f"[green]✓ Skill template created[/green]\n\n"
        f"File: [cyan]{output_path}[/cyan]\n"
        f"Name: [green]{skill_name}[/green]\n"
        f"Category: [yellow]{category}[/yellow]\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"1. Edit the file to implement your skill logic\n"
        f"2. Load with: pyutagent-cli skills load {output_path}",
        title="Skill Created"
    ))
