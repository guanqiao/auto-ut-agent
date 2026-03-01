"""Multi-file batch editing support for Aider integration.

This module provides support for editing multiple files with:
- Dependency analysis between files
- Topological sorting for correct edit order
- Batch validation and rollback support
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)


@dataclass
class FileNode:
    """Represents a file in the dependency graph."""
    path: str
    content: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    edit_content: Optional[str] = None


@dataclass
class MultiFileEditResult:
    """Result from multi-file editing operation."""
    success: bool
    edited_files: List[str] = field(default_factory=list)
    failed_files: List[Tuple[str, str]] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    rollback_applied: bool = False
    error_message: Optional[str] = None


class DependencyAnalyzer:
    """Analyzes dependencies between Java files."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.file_nodes: Dict[str, FileNode] = {}

    def analyze_files(self, file_paths: List[str]) -> Dict[str, FileNode]:
        """Analyze dependencies for a list of files.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            Dictionary mapping file paths to FileNode objects
        """
        # First pass: create nodes and extract imports
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                content = path.read_text(encoding='utf-8')
                node = FileNode(
                    path=str(path.absolute()),
                    content=content,
                    dependencies=self._extract_imports(content)
                )
                self.file_nodes[str(path.absolute())] = node

        # Second pass: build dependency graph
        self._build_dependency_graph()

        return self.file_nodes

    def _extract_imports(self, content: str) -> Set[str]:
        """Extract import statements from Java code."""
        imports = set()

        # Match import statements
        import_pattern = re.compile(r'^import\s+(?:static\s+)?([^;]+);', re.MULTILINE)

        for match in import_pattern.finditer(content):
            import_path = match.group(1).strip()

            # Convert import to potential file path
            # e.g., com.example.MyClass -> com/example/MyClass.java
            file_path = import_path.replace('.', '/') + '.java'
            imports.add(file_path)

            # Also add without the last part (for inner classes)
            if '.' in import_path:
                parent_path = import_path.rsplit('.', 1)[0].replace('.', '/') + '.java'
                imports.add(parent_path)

        return imports

    def _build_dependency_graph(self):
        """Build the dependency graph between files."""
        # Map class names to file paths
        class_to_file: Dict[str, str] = {}

        for path, node in self.file_nodes.items():
            # Extract package and class name
            package = self._extract_package(node.content)
            class_name = self._extract_class_name(node.content)

            if package and class_name:
                full_class = f"{package}.{class_name}"
                class_to_file[full_class] = path
                class_to_file[class_name] = path  # Also map simple name

        # Resolve dependencies
        for path, node in self.file_nodes.items():
            resolved_deps = set()

            for dep in node.dependencies:
                # Try to find matching file
                for file_path in self.file_nodes:
                    if dep in file_path or file_path.endswith(dep):
                        resolved_deps.add(file_path)
                        break

            node.dependencies = resolved_deps

        # Build reverse dependencies (dependents)
        for path, node in self.file_nodes.items():
            for dep_path in node.dependencies:
                if dep_path in self.file_nodes:
                    self.file_nodes[dep_path].dependents.add(path)

    def _extract_package(self, content: str) -> Optional[str]:
        """Extract package declaration from Java code."""
        match = re.search(r'^package\s+([^;]+);', content, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _extract_class_name(self, content: str) -> Optional[str]:
        """Extract class name from Java code."""
        # Match class, interface, enum declarations
        patterns = [
            r'public\s+class\s+(\w+)',
            r'class\s+(\w+)',
            r'public\s+interface\s+(\w+)',
            r'interface\s+(\w+)',
            r'public\s+enum\s+(\w+)',
            r'enum\s+(\w+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)

        return None

    def topological_sort(self) -> List[str]:
        """Sort files in dependency order (dependencies first).

        Uses Kahn's algorithm for topological sorting.

        Returns:
            List of file paths in dependency order
        """
        # Calculate in-degrees
        in_degree: Dict[str, int] = {}
        for path, node in self.file_nodes.items():
            in_degree[path] = len(node.dependencies)

        # Start with files that have no dependencies
        queue = deque([path for path, degree in in_degree.items() if degree == 0])
        sorted_files = []

        while queue:
            current = queue.popleft()
            sorted_files.append(current)

            # Reduce in-degree for dependents
            for dependent_path in self.file_nodes[current].dependents:
                if dependent_path in in_degree:
                    in_degree[dependent_path] -= 1
                    if in_degree[dependent_path] == 0:
                        queue.append(dependent_path)

        # Check for cycles
        if len(sorted_files) != len(self.file_nodes):
            # Handle cycle: add remaining files in any order
            remaining = set(self.file_nodes.keys()) - set(sorted_files)
            sorted_files.extend(sorted(remaining))

        return sorted_files

    def get_related_files(self, file_path: str, depth: int = 1) -> Set[str]:
        """Get files related to a given file.

        Args:
            file_path: The starting file path
            depth: How many levels of relationships to follow

        Returns:
            Set of related file paths
        """
        related = set()
        to_process = {file_path}
        current_depth = 0

        while to_process and current_depth < depth:
            next_level = set()

            for path in to_process:
                if path in self.file_nodes:
                    node = self.file_nodes[path]
                    # Add dependencies and dependents
                    next_level.update(node.dependencies)
                    next_level.update(node.dependents)

            related.update(next_level)
            to_process = next_level - related
            current_depth += 1

        return related


class MultiFileEditor:
    """Handles editing multiple files with dependency management."""

    def __init__(
        self,
        llm_client,
        project_path: str,
        max_concurrent: int = 3
    ):
        """Initialize multi-file editor.

        Args:
            llm_client: LLM client for generating edits
            project_path: Root path of the project
            max_concurrent: Maximum concurrent edits
        """
        self.llm_client = llm_client
        self.project_path = project_path
        self.max_concurrent = max_concurrent
        self.dependency_analyzer = DependencyAnalyzer(project_path)
        self._backup: Dict[str, str] = {}  # File path -> original content

    async def edit_files(
        self,
        edits_by_file: Dict[str, str],
        validate_each: bool = True,
        rollback_on_failure: bool = True
    ) -> MultiFileEditResult:
        """Edit multiple files with dependency-aware ordering.

        Args:
            edits_by_file: Dictionary mapping file paths to edit instructions
            validate_each: Whether to validate each edit
            rollback_on_failure: Whether to rollback on failure

        Returns:
            MultiFileEditResult with edit status
        """
        result = MultiFileEditResult(success=True)

        # Analyze dependencies
        file_paths = list(edits_by_file.keys())
        self.dependency_analyzer.analyze_files(file_paths)
        sorted_files = self.dependency_analyzer.topological_sort()

        # Create backup
        self._create_backup(file_paths)

        try:
            # Process files in dependency order with semaphore for concurrency
            semaphore = asyncio.Semaphore(self.max_concurrent)

            tasks = []
            for file_path in sorted_files:
                if file_path in edits_by_file:
                    instruction = edits_by_file[file_path]
                    task = self._edit_single_file(
                        semaphore, file_path, instruction, validate_each
                    )
                    tasks.append((file_path, task))

            # Execute edits
            for file_path, task in tasks:
                try:
                    success, error = await task
                    if success:
                        result.edited_files.append(file_path)
                    else:
                        result.failed_files.append((file_path, error))
                        if rollback_on_failure:
                            result.rollback_applied = True
                            self._rollback()
                            result.success = False
                            result.error_message = f"Edit failed for {file_path}: {error}"
                            return result
                except Exception as e:
                    result.failed_files.append((file_path, str(e)))
                    if rollback_on_failure:
                        result.rollback_applied = True
                        self._rollback()
                        result.success = False
                        result.error_message = f"Exception editing {file_path}: {e}"
                        return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if rollback_on_failure:
                result.rollback_applied = True
                self._rollback()

        return result

    async def _edit_single_file(
        self,
        semaphore: asyncio.Semaphore,
        file_path: str,
        instruction: str,
        validate: bool
    ) -> Tuple[bool, str]:
        """Edit a single file.

        Args:
            semaphore: Concurrency semaphore
            file_path: Path to the file
            instruction: Edit instruction
            validate: Whether to validate the edit

        Returns:
            Tuple of (success, error_message)
        """
        async with semaphore:
            try:
                # Read current content
                path = Path(file_path)
                if not path.exists():
                    return False, f"File not found: {file_path}"

                original_content = path.read_text(encoding='utf-8')

                # Generate edit using LLM
                edit_prompt = self._build_edit_prompt(file_path, original_content, instruction)

                response = await self.llm_client.complete(
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": edit_prompt}
                    ],
                    temperature=0.1
                )

                # Parse and apply edit
                new_content = self._apply_edit(original_content, response)

                if validate:
                    is_valid, error = self._validate_edit(file_path, original_content, new_content)
                    if not is_valid:
                        return False, error

                # Write the file
                path.write_text(new_content, encoding='utf-8')

                return True, ""

            except Exception as e:
                return False, str(e)

    def _build_edit_prompt(
        self,
        file_path: str,
        original_content: str,
        instruction: str
    ) -> str:
        """Build prompt for editing a file."""
        return f"""Please edit the following Java file according to the instruction.

## File
{file_path}

## Instruction
{instruction}

## Current Content

```java
{original_content}
```

Please provide the complete updated file content using SEARCH/REPLACE format or provide the full file content.
"""

    def _get_system_prompt(self) -> str:
        """Get system prompt for file editing."""
        return """You are a precise code editor. Edit the provided Java file according to the instruction.

Use SEARCH/REPLACE format for changes:

<<<<<<< SEARCH
exact content to find
=======
new content to replace
>>>>>>> REPLACE

Or provide the complete updated file content.

Rules:
- Make minimal, focused changes
- Ensure the code remains syntactically correct
- Preserve existing formatting and style
"""

    def _apply_edit(self, original_content: str, edit_response: str) -> str:
        """Apply edit to content."""
        # Try to parse SEARCH/REPLACE blocks
        from .edit_formats import edit_format_registry, EditFormat

        edits, format_used = edit_format_registry.auto_detect_and_parse(edit_response)

        if not edits:
            # If no edits found, return the response as-is (assuming it's the full file)
            return edit_response.strip()

        # Apply edits
        content = original_content
        for edit in edits:
            if edit.original:
                # Replace specific content
                content = content.replace(edit.original, edit.modified, 1)
            else:
                # Replace entire content
                content = edit.modified

        return content

    def _validate_edit(
        self,
        file_path: str,
        original_content: str,
        new_content: str
    ) -> Tuple[bool, str]:
        """Validate an edit.

        Args:
            file_path: Path to the file
            original_content: Original file content
            new_content: New file content

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for empty content
        if not new_content.strip():
            return False, "Edit resulted in empty file"

        # Check for basic Java syntax indicators
        if 'class ' not in new_content and 'interface ' not in new_content:
            # Might be a partial edit, which is okay
            pass

        # Check that we're not losing too much content unexpectedly
        original_lines = len(original_content.split('\n'))
        new_lines = len(new_content.split('\n'))

        if new_lines < original_lines * 0.5:
            return False, f"Edit removed too many lines ({original_lines} -> {new_lines})"

        return True, ""

    def _create_backup(self, file_paths: List[str]):
        """Create backup of files before editing."""
        self._backup.clear()

        for file_path in file_paths:
            path = Path(file_path)
            if path.exists():
                self._backup[file_path] = path.read_text(encoding='utf-8')

    def _rollback(self):
        """Rollback to backup state."""
        for file_path, content in self._backup.items():
            try:
                Path(file_path).write_text(content, encoding='utf-8')
            except Exception as e:
                logger.warning(f"Failed to rollback {file_path}: {e}")

    def get_edit_plan(self, file_paths: List[str]) -> Dict[str, Any]:
        """Get edit plan showing dependency order.

        Args:
            file_paths: List of files to edit

        Returns:
            Dictionary with dependency information
        """
        nodes = self.dependency_analyzer.analyze_files(file_paths)
        sorted_files = self.dependency_analyzer.topological_sort()

        return {
            'total_files': len(file_paths),
            'edit_order': sorted_files,
            'dependencies': {
                path: list(node.dependencies)
                for path, node in nodes.items()
            },
            'dependents': {
                path: list(node.dependents)
                for path, node in nodes.items()
            }
        }
