#!/usr/bin/env python3
"""Migration check script - 检查代码库中使用的废弃接口.

This script scans the codebase for deprecated imports and provides
recommendations for migration to unified interfaces.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class MigrationStatus(Enum):
    """Migration status for deprecated imports."""
    DEPRECATED = "deprecated"
    REMOVED = "removed"
    RECOMMENDED = "recommended"


@dataclass
class MigrationRule:
    """Rule for migration from old to new interface."""
    old_module: str
    old_name: str
    new_module: str
    new_name: str
    status: MigrationStatus
    note: str = ""


# Migration rules mapping
MIGRATION_RULES: List[MigrationRule] = [
    # Message Bus
    MigrationRule(
        "pyutagent.core.event_bus", "EventBus",
        "pyutagent.core.messaging", "UnifiedMessageBus",
        MigrationStatus.DEPRECATED,
        "Use UnifiedMessageBus for all messaging needs"
    ),
    MigrationRule(
        "pyutagent.core.event_bus", "AsyncEventBus",
        "pyutagent.core.messaging", "UnifiedMessageBus",
        MigrationStatus.DEPRECATED,
        "UnifiedMessageBus supports both sync and async operations"
    ),
    MigrationRule(
        "pyutagent.core.message_bus", "MessageBus",
        "pyutagent.core.messaging", "UnifiedMessageBus",
        MigrationStatus.DEPRECATED,
        "Use UnifiedMessageBus for unified messaging"
    ),
    MigrationRule(
        "pyutagent.agent.multi_agent.message_bus", "MessageBus",
        "pyutagent.core.messaging", "UnifiedMessageBus",
        MigrationStatus.DEPRECATED,
        "Use UnifiedMessageBus for agent communication"
    ),
    
    # Agent Base Classes
    MigrationRule(
        "pyutagent.agent.base_agent", "BaseAgent",
        "pyutagent.agent.unified_agent_base", "UnifiedAgentBase",
        MigrationStatus.DEPRECATED,
        "UnifiedAgentBase provides all agent functionality"
    ),
    MigrationRule(
        "pyutagent.agent.subagent_base", "SubAgent",
        "pyutagent.agent.unified_agent_base", "UnifiedAgentBase",
        MigrationStatus.DEPRECATED,
        "Use UnifiedAgentBase with AgentConfig for subagents"
    ),
    
    # Autonomous Loops
    MigrationRule(
        "pyutagent.agent.autonomous_loop", "AutonomousLoop",
        "pyutagent.agent.unified_autonomous_loop", "UnifiedAutonomousLoop",
        MigrationStatus.DEPRECATED,
        "UnifiedAutonomousLoop with LoopConfig supports all features"
    ),
    MigrationRule(
        "pyutagent.agent.enhanced_autonomous_loop", "EnhancedAutonomousLoop",
        "pyutagent.agent.unified_autonomous_loop", "UnifiedAutonomousLoop",
        MigrationStatus.DEPRECATED,
        "Use UnifiedAutonomousLoop with LoopFeature flags"
    ),
    MigrationRule(
        "pyutagent.agent.llm_driven_autonomous_loop", "LLMDrivenAutonomousLoop",
        "pyutagent.agent.unified_autonomous_loop", "UnifiedAutonomousLoop",
        MigrationStatus.DEPRECATED,
        "Enable LoopFeature.LLM_REASONING in LoopConfig"
    ),
    MigrationRule(
        "pyutagent.agent.delegating_autonomous_loop", "DelegatingAutonomousLoop",
        "pyutagent.agent.unified_autonomous_loop", "UnifiedAutonomousLoop",
        MigrationStatus.DEPRECATED,
        "Enable LoopFeature.DELEGATION in LoopConfig"
    ),
    
    # Executors
    MigrationRule(
        "pyutagent.agent.batch_executor", "BatchExecutor",
        "pyutagent.agent.execution.executor", "StepExecutor",
        MigrationStatus.DEPRECATED,
        "StepExecutor supports batch execution with parallel=True"
    ),
    MigrationRule(
        "pyutagent.agent.parallel_executor", "ParallelExecutor",
        "pyutagent.agent.execution.executor", "StepExecutor",
        MigrationStatus.DEPRECATED,
        "Use StepExecutor with parallel=True for parallel execution"
    ),
    
    # Configuration
    MigrationRule(
        "pyutagent.config.project_config", "ProjectConfigManager",
        "pyutagent.core.project_config", "ProjectConfigManager",
        MigrationStatus.DEPRECATED,
        "Use core.project_config for unified configuration"
    ),
    
    # Managers
    MigrationRule(
        "pyutagent.agent.utils.state_manager", "StateManager",
        "pyutagent.agent.core.agent_state", "StateManager",
        MigrationStatus.DEPRECATED,
        "Use agent.core.agent_state for state management"
    ),
    MigrationRule(
        "pyutagent.agent.context_manager", "ContextManager",
        "pyutagent.agent.unified_context_manager", "UnifiedContextManager",
        MigrationStatus.DEPRECATED,
        "Use UnifiedContextManager for context management"
    ),
]


class ImportFinder(ast.NodeVisitor):
    """AST visitor to find imports in Python files."""
    
    def __init__(self):
        self.imports: List[Tuple[str, str, int]] = []  # (module, name, line)
    
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append((alias.name, alias.name, node.lineno))
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            for alias in node.names:
                self.imports.append((node.module, alias.name, node.lineno))
        self.generic_visit(node)


def find_deprecated_imports(file_path: Path) -> List[Tuple[str, str, int, MigrationRule]]:
    """Find deprecated imports in a Python file.
    
    Args:
        file_path: Path to Python file
        
    Returns:
        List of (module, name, line, rule) tuples
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        finder = ImportFinder()
        finder.visit(tree)
        
        deprecated = []
        for module, name, line in finder.imports:
            for rule in MIGRATION_RULES:
                if module == rule.old_module and name == rule.old_name:
                    deprecated.append((module, name, line, rule))
        
        return deprecated
    except SyntaxError:
        return []
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def scan_directory(directory: Path) -> Dict[Path, List[Tuple[str, str, int, MigrationRule]]]:
    """Scan directory for deprecated imports.
    
    Args:
        directory: Directory to scan
        
    Returns:
        Dictionary mapping file paths to lists of deprecated imports
    """
    results = {}
    
    for py_file in directory.rglob("*.py"):
        # Skip certain directories
        if any(part.startswith(".") for part in py_file.parts):
            continue
        if "__pycache__" in py_file.parts:
            continue
        if "venv" in py_file.parts or "env" in py_file.parts:
            continue
        
        deprecated = find_deprecated_imports(py_file)
        if deprecated:
            results[py_file] = deprecated
    
    return results


def print_report(results: Dict[Path, List[Tuple[str, str, int, MigrationRule]]], base_path: Path):
    """Print migration report.
    
    Args:
        results: Scan results
        base_path: Base path for relative paths
    """
    if not results:
        print("[OK] No deprecated imports found!")
        return
    
    print("\n" + "=" * 80)
    print("MIGRATION REPORT - Deprecated Imports Found")
    print("=" * 80)
    
    total_files = len(results)
    total_issues = sum(len(imports) for imports in results.values())
    
    print(f"\n[Summary]")
    print(f"   Files with deprecated imports: {total_files}")
    print(f"   Total deprecated imports: {total_issues}")
    
    # Group by file
    print(f"\n[By File]")
    print("-" * 80)
    
    for file_path, imports in sorted(results.items()):
        rel_path = file_path.relative_to(base_path)
        print(f"\n{rel_path}")
        
        for module, name, line, rule in imports:
            print(f"  Line {line:4d}: {name} from {module}")
            print(f"           -> Use: {rule.new_name} from {rule.new_module}")
            if rule.note:
                print(f"           Note: {rule.note}")
    
    # Group by deprecated module
    print(f"\n[By Deprecated Module]")
    print("-" * 80)
    
    module_counts: Dict[str, int] = {}
    for imports in results.values():
        for module, name, line, rule in imports:
            key = f"{rule.old_module}.{rule.old_name}"
            module_counts[key] = module_counts.get(key, 0) + 1
    
    for module_name, count in sorted(module_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:3d} usages: {module_name}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Prioritize migrating high-usage deprecated imports")
    print("2. Use the migration guide: docs/migration_guide.md")
    print("3. Run tests after each migration to ensure functionality")
    print("4. Consider using the unified interfaces for new code")
    print("=" * 80)


def main():
    """Main entry point."""
    # Determine base path
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent  # Go up from scripts/ to project root
    
    # Scan pyutagent directory
    target_dir = base_path / "pyutagent"
    
    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)
    
    print(f"Scanning {target_dir} for deprecated imports...")
    
    results = scan_directory(target_dir)
    print_report(results, base_path)
    
    # Exit with error code if deprecated imports found
    if results:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
