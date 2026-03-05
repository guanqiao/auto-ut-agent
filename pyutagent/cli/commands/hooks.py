"""Hooks CLI commands for managing lifecycle hooks."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group(name='hooks')
def hooks_group():
    """Manage lifecycle hooks for Agent automation.
    
    Hooks allow you to customize Agent behavior at specific lifecycle events
    such as startup, shutdown, before/after tasks, etc.
    """
    pass


@hooks_group.command(name='list')
@click.option('--event', type=str, help='Filter by event type')
@click.option('--all', 'show_all', is_flag=True, help='Show all hooks including disabled')
def list_hooks(event: Optional[str], show_all: bool):
    """List all registered hooks."""
    from pyutagent.agent.hooks import get_hook_registry, HookEvent
    
    registry = get_hook_registry()
    
    table = Table(title="Registered Hooks")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name", style="green")
    table.add_column("Event", style="yellow")
    table.add_column("Priority", style="magenta")
    table.add_column("Status", style="blue")
    table.add_column("Description", style="white")
    
    hooks_found = False
    
    for hook_event in HookEvent:
        if event and event.lower() not in hook_event.name.lower():
            continue
            
        hooks = registry._hooks.get(hook_event, [])
        for hook in hooks:
            if not show_all and not hook.enabled:
                continue
            hooks_found = True
            table.add_row(
                hook.id[:8],
                hook.name,
                hook_event.name,
                hook.priority.name,
                "✓ Enabled" if hook.enabled else "✗ Disabled",
                hook.description[:50] + "..." if len(hook.description) > 50 else hook.description
            )
    
    global_hooks = registry.get_global_hooks()
    for hook in global_hooks:
        if not show_all and not hook.enabled:
            continue
        hooks_found = True
        table.add_row(
            hook.id[:8],
            hook.name,
            "GLOBAL",
            hook.priority.name,
            "✓ Enabled" if hook.enabled else "✗ Disabled",
            hook.description[:50] + "..." if len(hook.description) > 50 else hook.description
        )
    
    if hooks_found:
        console.print(table)
    else:
        console.print("[yellow]No hooks found[/yellow]")
        if not show_all:
            console.print("Use --all to show disabled hooks")


@hooks_group.command(name='events')
def list_events():
    """List all available hook events."""
    from pyutagent.agent.hooks import HookEvent, HookPriority
    
    table = Table(title="Available Hook Events")
    table.add_column("Event", style="cyan")
    table.add_column("Description", style="white")
    
    event_descriptions = {
        HookEvent.STARTUP: "Agent startup - runs once when agent initializes",
        HookEvent.SHUTDOWN: "Agent shutdown - runs once before agent terminates",
        HookEvent.BEFORE_TASK: "Before task execution - runs before each task",
        HookEvent.AFTER_TASK: "After task execution - runs after each task",
        HookEvent.TASK_SUCCESS: "Task success - runs when a task succeeds",
        HookEvent.TASK_FAILURE: "Task failure - runs when a task fails",
        HookEvent.BEFORE_TOOL_CALL: "Before tool call - runs before each tool invocation",
        HookEvent.AFTER_TOOL_CALL: "After tool call - runs after each tool invocation",
        HookEvent.BEFORE_LLM_CALL: "Before LLM call - runs before each LLM request",
        HookEvent.AFTER_LLM_CALL: "After LLM call - runs after each LLM response",
        HookEvent.ERROR: "Error handling - runs when an error occurs",
        HookEvent.USER_INPUT: "User input - runs when user provides input",
        HookEvent.AGENT_MESSAGE: "Agent message - runs when agent sends a message",
    }
    
    for event, desc in event_descriptions.items():
        table.add_row(event.name, desc)
    
    console.print(table)
    
    console.print("\n[bold]Priorities:[/bold]")
    for priority in HookPriority:
        console.print(f"  [magenta]{priority.name}[/magenta]: {priority.value}")


@hooks_group.command(name='enable')
@click.argument('hook_id')
def enable_hook(hook_id: str):
    """Enable a hook by ID."""
    from pyutagent.agent.hooks import get_hook_registry
    
    registry = get_hook_registry()
    if registry.enable_hook(hook_id):
        console.print(f"[green]✓ Hook {hook_id} enabled[/green]")
    else:
        console.print(f"[red]✗ Hook {hook_id} not found[/red]")


@hooks_group.command(name='disable')
@click.argument('hook_id')
def disable_hook(hook_id: str):
    """Disable a hook by ID."""
    from pyutagent.agent.hooks import get_hook_registry
    
    registry = get_hook_registry()
    if registry.disable_hook(hook_id):
        console.print(f"[green]✓ Hook {hook_id} disabled[/green]")
    else:
        console.print(f"[red]✗ Hook {hook_id} not found[/red]")


@hooks_group.command(name='stats')
def hook_stats():
    """Show hook execution statistics."""
    from pyutagent.agent.hooks import get_hook_executor
    
    executor = get_hook_executor()
    stats = executor.get_stats()
    
    console.print(Panel(
        f"[bold]Hook Execution Statistics[/bold]\n\n"
        f"Total Executions: [cyan]{stats['total_executions']}[/cyan]\n"
        f"Successful: [green]{stats['successful']}[/green]\n"
        f"Failed: [red]{stats['failed']}[/red]\n"
        f"Success Rate: [yellow]{stats['success_rate']:.1%}[/yellow]\n"
        f"Avg Duration: [magenta]{stats['avg_duration_ms']:.2f}ms[/magenta]",
        title="Statistics"
    ))


@hooks_group.command(name='history')
@click.option('--limit', type=int, default=10, help='Number of records to show')
def hook_history(limit: int):
    """Show hook execution history."""
    from pyutagent.agent.hooks import get_hook_executor
    
    executor = get_hook_executor()
    history = executor.get_history()[-limit:]
    
    if not history:
        console.print("[yellow]No execution history[/yellow]")
        return
    
    table = Table(title="Recent Hook Executions")
    table.add_column("Hook ID", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Duration", style="magenta")
    table.add_column("Error", style="red")
    
    for result in reversed(history):
        table.add_row(
            result.hook_id[:8],
            "✓ Success" if result.success else "✗ Failed",
            f"{result.duration_ms:.2f}ms",
            result.error[:30] if result.error else "-"
        )
    
    console.print(table)


@hooks_group.command(name='add')
@click.option('--name', required=True, help='Hook name')
@click.option('--event', required=True, type=str, help='Event to hook (e.g., BEFORE_TASK)')
@click.option('--script', type=click.Path(exists=True), help='Python script to execute')
@click.option('--priority', type=str, default='NORMAL', help='Priority (LOW, NORMAL, HIGH, CRITICAL)')
@click.option('--description', type=str, default='', help='Hook description')
def add_hook(name: str, event: str, script: Optional[str], priority: str, description: str):
    """Add a new hook from a Python script.
    
    Example:
        pyutagent-cli hooks add --name my_hook --event BEFORE_TASK --script ./my_hook.py
    """
    from pyutagent.agent.hooks import (
        get_hook_registry, HookEvent, HookPriority,
        register_hook
    )
    
    try:
        hook_event = HookEvent[event.upper()]
    except KeyError:
        console.print(f"[red]✗ Invalid event: {event}[/red]")
        console.print("Run 'pyutagent-cli hooks events' to see available events")
        return
    
    try:
        hook_priority = HookPriority[priority.upper()]
    except KeyError:
        console.print(f"[red]✗ Invalid priority: {priority}[/red]")
        console.print("Valid priorities: LOW, NORMAL, HIGH, CRITICAL")
        return
    
    if script:
        import importlib.util
        
        try:
            spec = importlib.util.spec_from_file_location("hook_module", script)
            if not spec or not spec.loader:
                raise ValueError(f"Could not load script: {script}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'handler'):
                raise ValueError("Script must define an async 'handler' function")
            
            handler = module.handler
        except Exception as e:
            console.print(f"[red]✗ Failed to load script: {e}[/red]")
            return
    else:
        async def default_handler(context):
            console.print(f"[yellow]Hook '{name}' triggered for {event}[/yellow]")
            return {"status": "executed"}
        
        handler = default_handler
    
    hook = register_hook(
        name=name,
        event=hook_event,
        handler=handler,
        priority=hook_priority,
        description=description
    )
    
    console.print(Panel(
        f"[green]✓ Hook registered successfully[/green]\n\n"
        f"ID: [cyan]{hook.id}[/cyan]\n"
        f"Name: [green]{name}[/green]\n"
        f"Event: [yellow]{hook_event.name}[/yellow]\n"
        f"Priority: [magenta]{hook_priority.name}[/magenta]",
        title="Hook Added"
    ))


@hooks_group.command(name='remove')
@click.argument('hook_id')
def remove_hook(hook_id: str):
    """Remove a hook by ID."""
    from pyutagent.agent.hooks import get_hook_registry
    
    registry = get_hook_registry()
    if registry.unregister(hook_id):
        console.print(f"[green]✓ Hook {hook_id} removed[/green]")
    else:
        console.print(f"[red]✗ Hook {hook_id} not found[/red]")


@hooks_group.command(name='clear-history')
def clear_history():
    """Clear hook execution history."""
    from pyutagent.agent.hooks import get_hook_executor
    
    executor = get_hook_executor()
    executor.clear_history()
    console.print("[green]✓ Hook execution history cleared[/green]")
