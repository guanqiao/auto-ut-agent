"""Generate-all command for CLI - batch test generation for all Java files."""

import time
from pathlib import Path
from typing import List

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

console = Console()


@click.command(name='generate-all')
@click.argument('project_path', type=click.Path(exists=True))
@click.option('--llm', default='default', help='LLM configuration to use')
@click.option('--output-dir', type=click.Path(), help='Output directory for test files')
@click.option('--coverage-target', type=int, default=80, help='Target coverage percentage')
@click.option('--max-iterations', type=int, default=10, help='Maximum iterations per file')
@click.option('-p', '--parallel', type=int, default=1, help='Number of parallel workers (0 for unlimited)')
@click.option('--timeout', type=int, default=300, help='Timeout per file in seconds')
@click.option('--continue-on-error', is_flag=True, default=True, help='Continue on errors')
@click.option('--stop-on-error', is_flag=True, help='Stop on first error')
@click.option('--defer-compilation', is_flag=True, default=False, help='Defer compilation until all files are generated')
@click.option('--compile-only-at-end', is_flag=True, default=False, help='Only compile once after all files are generated (implies --defer-compilation)')
@click.option('-i', '--incremental', is_flag=True, default=False, help='Enable incremental mode (preserve existing passing tests)')
@click.option('--skip-analysis', is_flag=True, default=False, help='Skip running existing tests, just analyze file content')
@click.option('--basic', is_flag=True, help='Use basic ReActAgent instead of EnhancedAgent')
@click.option('--enable-error-prediction', is_flag=True, default=True, show_default=True, help='Enable error prediction')
@click.option('--enable-self-reflection', is_flag=True, default=True, show_default=True, help='Enable self-reflection')
@click.option('--enable-pattern-library', is_flag=True, default=True, show_default=True, help='Enable pattern library')
def generate_all_command(
    project_path: str,
    llm: str,
    output_dir: str,
    coverage_target: int,
    max_iterations: int,
    parallel: int,
    timeout: int,
    continue_on_error: bool,
    stop_on_error: bool,
    defer_compilation: bool,
    compile_only_at_end: bool,
    incremental: bool,
    skip_analysis: bool,
    basic: bool,
    enable_error_prediction: bool,
    enable_self_reflection: bool,
    enable_pattern_library: bool
):
    """Generate unit tests for all Java files in a Maven project.
    
    This command scans the project for Java files and generates tests
    for each one, with support for parallel execution and error recovery.
    """
    project_path = Path(project_path)
    
    pom_file = project_path / 'pom.xml'
    if not pom_file.exists():
        console.print(f"[red]Error: {project_path} is not a valid Maven project (pom.xml not found)[/red]")
        raise click.Abort()
    
    settings = None
    try:
        from pyutagent.core.config import get_settings
        settings = get_settings()
        src_dir = project_path / settings.project_paths.src_main_java
    except Exception:
        src_dir = project_path / 'src' / 'main' / 'java'
    
    if not src_dir.exists():
        console.print(f"[yellow]Warning: Java source directory not found at {src_dir}[/yellow]")
        return
    
    java_files = list(src_dir.rglob('*.java'))
    
    if not java_files:
        console.print(f"[yellow]No Java files found in {src_dir}[/yellow]")
        return
    
    if stop_on_error:
        continue_on_error = False
    
    # compile_only_at_end implies defer_compilation
    if compile_only_at_end:
        defer_compilation = True
    
    console.print(Panel(
        f"[bold blue]Batch Test Generation[/bold blue]\n\n"
        f"Project: {project_path.name}\n"
        f"Java files: {len(java_files)}\n"
        f"Parallel workers: {parallel if parallel > 0 else 'unlimited'}\n"
        f"Coverage target: {coverage_target}%\n"
        f"Timeout per file: {timeout}s\n"
        f"Continue on error: {continue_on_error}\n"
        f"Agent mode: {'Basic (ReActAgent)' if basic else 'Enhanced (EnhancedAgent)'}\n"
        f"Defer compilation: {defer_compilation}" + 
        (f"\n[green]✓ Two-phase mode enabled (generate all → compile all)[/green]" if defer_compilation else "") +
        (f"\n[green]✓ Incremental mode: ENABLED[/green] (preserve existing passing tests)" if incremental else "") +
        (f"\n[green]✓ Error prediction enabled[/green]" if enable_error_prediction and not basic else "") +
        (f"\n[green]✓ Self-reflection enabled[/green]" if enable_self_reflection and not basic else "") +
        (f"\n[green]✓ Pattern library enabled[/green]" if enable_pattern_library and not basic else ""),
        title="Configuration"
    ))
    
    try:
        from pyutagent.services.batch_generator import BatchGenerator, BatchConfig
        from pyutagent.core.config import load_llm_config
        
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
        
        from pyutagent.llm.client import LLMClient
        llm_client = LLMClient.from_config(llm_config)
        
        batch_config = BatchConfig(
            parallel_workers=parallel,
            timeout_per_file=timeout,
            continue_on_error=continue_on_error,
            coverage_target=coverage_target,
            max_iterations=max_iterations,
            defer_compilation=defer_compilation,
            compile_only_at_end=compile_only_at_end,
            incremental_mode=incremental,
            skip_test_analysis=skip_analysis,
            use_enhanced_agent=not basic,
            enable_error_prediction=enable_error_prediction,
            enable_self_reflection=enable_self_reflection,
            enable_pattern_library=enable_pattern_library,
        )
        
        file_paths = [str(f.relative_to(project_path)) for f in java_files]
        
        progress_table = Table(show_header=True, header_style="bold")
        progress_table.add_column("File", style="cyan", width=30)
        progress_table.add_column("Status", width=12)
        progress_table.add_column("Coverage", width=10)
        progress_table.add_column("Iters", width=6)
        progress_table.add_column("Time", width=8)
        progress_table.add_column("Error/Info", width=30)
        
        file_status = {fp: {"status": "⏳ Pending", "coverage": "-", "iters": "-", "time": "-", "error": ""} for fp in file_paths}
        
        def update_progress(batch_progress):
            pass
        
        generator = BatchGenerator(
            llm_client=llm_client,
            project_path=str(project_path),
            config=batch_config,
            progress_callback=update_progress
        )
        
        console.print("\n[bold]Starting batch generation...[/bold]\n")
        
        start_time = time.time()
        result = generator.generate_all_sync(file_paths)
        total_time = time.time() - start_time
        
        result_table = Table(show_header=True, header_style="bold magenta")
        result_table.add_column("File", style="cyan", width=30)
        result_table.add_column("Status", width=12)
        result_table.add_column("Coverage", width=10)
        result_table.add_column("Iters", width=6)
        result_table.add_column("Time", width=8)
        result_table.add_column("Error", width=30)
        
        for file_result in result.results:
            file_name = Path(file_result.file_path).name
            if len(file_name) > 28:
                file_name = file_name[:25] + "..."
            
            if file_result.success:
                status = "[green]✓ Done[/green]"
                coverage_source = getattr(file_result, 'coverage_source', 'jacoco')
                if coverage_source == 'llm_estimated':
                    coverage = f"{file_result.coverage:.1f}%*"
                else:
                    coverage = f"{file_result.coverage:.1f}%"
                error = ""
            else:
                status = "[red]✗ Failed[/red]"
                coverage = "-"
                error = file_result.error[:28] + "..." if file_result.error and len(file_result.error) > 28 else (file_result.error or "")
            
            result_table.add_row(
                file_name,
                status,
                coverage,
                str(file_result.iterations) if file_result.iterations > 0 else "-",
                f"{file_result.duration:.1f}s",
                error
            )
        
        console.print(result_table)
        
        llm_estimated_count = sum(
            1 for r in result.results 
            if r.success and getattr(r, 'coverage_source', 'jacoco') == 'llm_estimated'
        )
        if llm_estimated_count > 0:
            console.print("[dim]* Coverage marked with * is LLM estimated (JaCoCo unavailable)[/dim]")
        
        summary_style = "green" if result.success_count == result.total_files else ("yellow" if result.success_count > 0 else "red")
        
        console.print()
        
        # Show compilation results if defer_compilation was enabled
        if result.compilation_result:
            compile_style = "green" if result.compilation_result.success else "red"
            console.print(Panel(
                f"[bold]Batch Compilation Results[/bold]\n\n"
                f"Status: [{compile_style}]{'✓ Success' if result.compilation_result.success else '✗ Failed'}[/{compile_style}]\n"
                f"Compiled files: {result.compilation_result.compiled_files}\n"
                f"Failed files: {result.compilation_result.failed_files}\n"
                f"Compilation time: {result.compilation_result.duration:.1f}s" +
                (f"\n\n[red]Errors:[/red]\n" + "\n".join(result.compilation_result.errors[:5]) if result.compilation_result.errors else ""),
                title="Phase 2: Compilation",
                border_style=compile_style
            ))
            console.print()
        
        console.print(Panel(
            f"[bold]Summary[/bold]\n\n"
            f"Total files: {result.total_files}\n"
            f"[green]Successful: {result.success_count}[/green]\n"
            f"[red]Failed: {result.failed_count}[/red]\n"
            f"Success rate: [{summary_style}]{result.success_rate:.1f}%[/{summary_style}]\n"
            f"Total time: {total_time:.1f}s",
            title="Batch Generation Complete",
            border_style=summary_style
        ))
        
        if result.failed_count > 0:
            console.print("\n[yellow]Failed files:[/yellow]")
            for file_result in result.results:
                if not file_result.success:
                    console.print(f"  [red]✗[/red] {file_result.file_path}")
                    if file_result.error:
                        console.print(f"    Error: {file_result.error}")
        
        # Show compilation errors if defer_compilation was enabled
        if result.compilation_result and not result.compilation_result.success:
            console.print("\n[red]Compilation errors:[/red]")
            for error in result.compilation_result.errors[:10]:
                console.print(f"  [red]✗[/red] {error}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
