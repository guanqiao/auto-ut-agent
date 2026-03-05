"""Task CLI command for executing various programming tasks."""

import asyncio
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.command(name='task')
@click.argument('task_description', required=False)
@click.option('--project', type=click.Path(exists=True), default='.', help='Project directory')
@click.option('--mode', type=click.Choice(['autonomous', 'interactive', 'planning']), default='interactive', help='Agent mode')
@click.option('--llm', default='default', help='LLM configuration to use')
@click.option('--plan-only', is_flag=True, help='Only create plan, do not execute')
@click.option('--type', 'task_type', type=str, help='Force task type (ut_generation, refactoring, bug_fix, etc.)')
def task_command(
    task_description: Optional[str],
    project: str,
    mode: str,
    llm: str,
    plan_only: bool,
    task_type: Optional[str]
):
    """Execute a programming task using the Universal Agent.
    
    The Universal Agent can handle various types of programming tasks:
    - UT Generation: Generate unit tests
    - Code Refactoring: Refactor code for better quality
    - Bug Fixing: Fix bugs in the codebase
    - Feature Addition: Add new features
    - Code Review: Review code quality
    - Documentation: Generate documentation
    - Code Explanation: Explain code behavior
    - Test Debugging: Debug failing tests
    - Performance Optimization: Optimize performance
    - Security Audit: Perform security analysis
    
    Examples:
        pyutagent-cli task "Generate unit tests for UserService.java"
        pyutagent-cli task "Refactor OrderService to use dependency injection"
        pyutagent-cli task "Fix the null pointer exception in PaymentHandler"
        pyutagent-cli task --plan-only "Add caching to the API layer"
    """
    if not task_description:
        console.print("[yellow]No task description provided.[/yellow]")
        console.print("\n[bold]Example usage:[/bold]")
        console.print("  pyutagent-cli task \"Generate unit tests for UserService.java\"")
        console.print("  pyutagent-cli task \"Refactor OrderService to use dependency injection\"")
        console.print("  pyutagent-cli task \"Fix the null pointer exception in PaymentHandler\"")
        return
    
    project_path = Path(project).resolve()
    
    console.print(Panel(
        f"[bold cyan]Universal Agent Task[/bold cyan]\n\n"
        f"Project: {project_path.name}\n"
        f"Mode: {mode}\n"
        f"Task: {task_description[:100]}{'...' if len(task_description) > 100 else ''}",
        title="Task Configuration"
    ))
    
    try:
        from pyutagent.agent.universal_agent import UniversalCodingAgent, AgentMode
        from pyutagent.agent.task_understanding import TaskType
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
        
        working_memory = WorkingMemory()
        
        agent_mode = {
            'autonomous': AgentMode.AUTONOMOUS,
            'interactive': AgentMode.INTERACTIVE,
            'planning': AgentMode.PLANNING_ONLY,
        }.get(mode, AgentMode.INTERACTIVE)
        
        if plan_only:
            agent_mode = AgentMode.PLANNING_ONLY
        
        agent = UniversalCodingAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=str(project_path),
            mode=agent_mode,
        )
        
        context = {}
        if task_type:
            try:
                context['forced_task_type'] = TaskType(task_type.lower())
            except ValueError:
                console.print(f"[yellow]Warning: Unknown task type '{task_type}'[/yellow]")
        
        async def run_task():
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing task...", total=None)
                
                result = await agent.handle_request(task_description, context)
                
                return result
        
        result = asyncio.run(run_task())
        
        console.print("\n" + "="*60)
        
        if result.success:
            console.print(f"[green]✓ Task completed successfully![/green]")
        else:
            console.print(f"[red]✗ Task failed[/red]")
        
        console.print(f"\n[bold]Task Type:[/bold] {result.task_type.value}")
        console.print(f"[bold]Message:[/bold] {result.message}")
        
        if result.understanding:
            understanding = result.understanding
            console.print(f"\n[bold]Task Understanding:[/bold]")
            console.print(f"  Type: {understanding.task_type.value}")
            console.print(f"  Priority: {understanding.priority.value}")
            console.print(f"  Complexity: {understanding.complexity.value}")
            if understanding.target_files:
                console.print(f"  Target Files: {', '.join(understanding.target_files[:5])}")
        
        if result.plan:
            plan = result.plan
            console.print(f"\n[bold]Execution Plan:[/bold]")
            console.print(f"  Total Steps: {len(plan.subtasks)}")
            
            table = Table(title="Plan Steps")
            table.add_column("#", style="dim")
            table.add_column("Type", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Status", style="green")
            
            for i, subtask in enumerate(plan.subtasks, 1):
                table.add_row(
                    str(i),
                    subtask.type.value,
                    subtask.description[:50] + "..." if len(subtask.description) > 50 else subtask.description,
                    subtask.status.value
                )
            
            console.print(table)
        
        if result.artifacts:
            console.print(f"\n[bold]Artifacts:[/bold]")
            for key, value in result.artifacts.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                console.print(f"  {key}: {value}")
        
        if result.metrics:
            console.print(f"\n[bold]Metrics:[/bold]")
            for key, value in result.metrics.items():
                console.print(f"  {key}: {value}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()


@click.command(name='task-types')
def task_types_command():
    """List all supported task types."""
    from pyutagent.agent.task_understanding import TaskType, TaskPriority, TaskComplexity
    
    console.print("[bold cyan]Supported Task Types[/bold cyan]\n")
    
    task_descriptions = {
        TaskType.UT_GENERATION: "Generate unit tests for source code",
        TaskType.REFACTORING: "Refactor code for better quality and maintainability",
        TaskType.BUG_FIX: "Fix bugs in the codebase",
        TaskType.FEATURE_ADDITION: "Add new features to existing code",
        TaskType.CODE_REVIEW: "Review code quality and suggest improvements",
        TaskType.DOCUMENTATION: "Generate documentation for code",
        TaskType.CODE_EXPLANATION: "Explain code behavior and logic",
        TaskType.TEST_DEBUGGING: "Debug failing tests",
        TaskType.PERFORMANCE_OPTIMIZATION: "Optimize code performance",
        TaskType.SECURITY_AUDIT: "Perform security analysis and fixes",
        TaskType.UNKNOWN: "Unknown or unclassified task type",
    }
    
    table = Table()
    table.add_column("Task Type", style="cyan")
    table.add_column("Description", style="white")
    
    for task_type, description in task_descriptions.items():
        table.add_row(task_type.value, description)
    
    console.print(table)
    
    console.print("\n[bold cyan]Task Priorities[/bold cyan]\n")
    
    priority_descriptions = {
        TaskPriority.CRITICAL: "Critical priority - must be done immediately",
        TaskPriority.HIGH: "High priority - should be done soon",
        TaskPriority.MEDIUM: "Medium priority - normal scheduling",
        TaskPriority.LOW: "Low priority - can be deferred",
    }
    
    table = Table()
    table.add_column("Priority", style="yellow")
    table.add_column("Description", style="white")
    
    for priority, description in priority_descriptions.items():
        table.add_row(priority.value, description)
    
    console.print(table)
    
    console.print("\n[bold cyan]Task Complexity Levels[/bold cyan]\n")
    
    complexity_descriptions = {
        TaskComplexity.SIMPLE: "Simple task - single file, basic changes",
        TaskComplexity.MODERATE: "Moderate task - multiple files, some complexity",
        TaskComplexity.COMPLEX: "Complex task - architectural changes, many files",
        TaskComplexity.VERY_COMPLEX: "Very complex task - major refactoring, system-wide changes",
    }
    
    table = Table()
    table.add_column("Complexity", style="magenta")
    table.add_column("Description", style="white")
    
    for complexity, description in complexity_descriptions.items():
        table.add_row(complexity.value, description)
    
    console.print(table)
