"""Standard tool implementations.

This module provides:
- ReadTool: Read file contents
- WriteTool: Write content to files
- EditTool: Search/Replace style editing
- GlobTool: File pattern matching
- GrepTool: Code search
- BashTool: Command execution
"""

import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tool import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    create_tool_parameter
)

logger = logging.getLogger(__name__)


class ReadTool(Tool):
    """Tool for reading file contents.

    Supports:
    - Full file reading
    - Line range reading
    - Encoding detection
    """

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Read tool.

        Args:
            base_path: Base path for relative file paths
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._definition = ToolDefinition(
            name="read_file",
            description="Read the contents of a file from the file system. "
                        "Returns the file content as text. Use this to view existing code.",
            category=ToolCategory.FILE,
            parameters=[
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="Path to the file to read. Can be absolute or relative to the working directory.",
                    required=True
                ),
                create_tool_parameter(
                    name="offset",
                    param_type="integer",
                    description="Line number to start reading from (1-based). Default: 1",
                    required=False,
                    default=1
                ),
                create_tool_parameter(
                    name="limit",
                    param_type="integer",
                    description="Maximum number of lines to read. Default: all lines",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="show_line_numbers",
                    param_type="boolean",
                    description="Whether to show line numbers in the output. Default: false",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {"file_path": "src/main/java/App.java"},
                    "description": "Read entire App.java file"
                },
                {
                    "params": {"file_path": "src/main/java/App.java", "offset": 10, "limit": 50},
                    "description": "Read lines 10-60 of App.java"
                }
            ],
            tags=["file", "read", "view"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute file reading."""
        file_path = kwargs.get("file_path")
        offset = kwargs.get("offset", 1)
        limit = kwargs.get("limit")
        show_line_numbers = kwargs.get("show_line_numbers", False)

        if not file_path:
            return ToolResult(success=False, error="file_path is required")

        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}"
                )

            if not path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Path is not a file: {path}"
                )

            content = path.read_text(encoding="utf-8")
            lines = content.split("\n")

            if offset > 1:
                lines = lines[offset - 1:]

            if limit:
                lines = lines[:limit]

            output_lines = []
            for i, line in enumerate(lines, start=offset):
                if show_line_numbers:
                    output_lines.append(f"{i:6d}\t{line}")
                else:
                    output_lines.append(line)

            result_content = "\n".join(output_lines)

            logger.info(f"[ReadTool] Read file: {path}, lines: {len(output_lines)}")

            return ToolResult(
                success=True,
                output=result_content,
                metadata={
                    "file_path": str(path),
                    "total_lines": len(content.split("\n")),
                    "read_lines": len(output_lines),
                    "offset": offset,
                    "limit": limit
                }
            )

        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {file_path}")
        except Exception as e:
            logger.exception(f"[ReadTool] Failed to read file: {e}")
            return ToolResult(success=False, error=f"Failed to read file: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to base path."""
        path = Path(file_path)
        if not path.is_absolute() and self._base_path:
            path = self._base_path / path
        return path.resolve()


class WriteTool(Tool):
    """Tool for writing content to files.

    Supports:
    - Full file creation
    - Directory creation
    - Backup before overwrite
    """

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Write tool.

        Args:
            base_path: Base path for relative file paths
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._definition = ToolDefinition(
            name="write_file",
            description="Write content to a file. Creates the file or overwrites existing content. "
                        "Can also create intermediate directories.",
            category=ToolCategory.FILE,
            parameters=[
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="Path to the file to write",
                    required=True
                ),
                create_tool_parameter(
                    name="content",
                    param_type="string",
                    description="Content to write to the file",
                    required=True
                ),
                create_tool_parameter(
                    name="create_dirs",
                    param_type="boolean",
                    description="Create intermediate directories if they don't exist. Default: true",
                    required=False,
                    default=True
                ),
                create_tool_parameter(
                    name="backup",
                    param_type="boolean",
                    description="Create backup of existing file with .bak extension. Default: false",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {
                        "file_path": "src/main/java/com/example/NewClass.java",
                        "content": "package com.example;\n\npublic class NewClass {}\n"
                    },
                    "description": "Create a new Java class file"
                }
            ],
            tags=["file", "write", "create", "edit"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute file writing."""
        file_path = kwargs.get("file_path")
        content = kwargs.get("content")
        create_dirs = kwargs.get("create_dirs", True)
        backup = kwargs.get("backup", False)

        if not file_path:
            return ToolResult(success=False, error="file_path is required")
        if content is None:
            return ToolResult(success=False, error="content is required")

        try:
            path = self._resolve_path(file_path)

            if path.exists() and backup:
                backup_path = path.with_suffix(path.suffix + ".bak")
                path.rename(backup_path)
                logger.info(f"[WriteTool] Created backup: {backup_path}")

            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(content, encoding="utf-8")

            logger.info(f"[WriteTool] Wrote file: {path}, size: {len(content)} bytes")

            return ToolResult(
                success=True,
                output=f"File written successfully: {path}",
                metadata={
                    "file_path": str(path),
                    "size": len(content),
                    "lines": len(content.split("\n"))
                }
            )

        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {file_path}")
        except Exception as e:
            logger.exception(f"[WriteTool] Failed to write file: {e}")
            return ToolResult(success=False, error=f"Failed to write file: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to base path."""
        path = Path(file_path)
        if not path.is_absolute() and self._base_path:
            path = self._base_path / path
        return path.resolve()


class EditTool(Tool):
    """Tool for precise Search/Replace style editing.

    This is similar to Aider's editing approach - find a specific
    section of code and replace it with new content.
    """

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Edit tool.

        Args:
            base_path: Base path for relative file paths
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._definition = ToolDefinition(
            name="edit_file",
            description="Make targeted edits to a file using Search/Replace. "
                        "Find the \"search\" code block and replace it with \"replace\" block. "
                        "This enables precise, minimal edits without rewriting entire files.",
            category=ToolCategory.FILE,
            parameters=[
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="Path to the file to edit",
                    required=True
                ),
                create_tool_parameter(
                    name="search",
                    param_type="string",
                    description="The exact code block to find in the file. Must match exactly including indentation.",
                    required=True
                ),
                create_tool_parameter(
                    name="replace",
                    param_type="string",
                    description="The code to replace the search block with",
                    required=True
                ),
                create_tool_parameter(
                    name="global_replace",
                    param_type="boolean",
                    description="Replace all occurrences of search pattern. Default: false",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="regex",
                    param_type="boolean",
                    description="Treat search as regular expression. Default: false",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {
                        "file_path": "src/main/java/App.java",
                        "search": "public class App {\n    private String name;\n}",
                        "replace": "public class App {\n    private String name;\n    private int age;\n}"
                    },
                    "description": "Add a new field to a class"
                }
            ],
            tags=["file", "edit", "modify", "replace"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute file editing."""
        file_path = kwargs.get("file_path")
        search = kwargs.get("search")
        replace = kwargs.get("replace")
        global_replace = kwargs.get("global_replace", False)
        regex = kwargs.get("regex", False)

        if not file_path:
            return ToolResult(success=False, error="file_path is required")
        if not search:
            return ToolResult(success=False, error="search is required")
        if replace is None:
            return ToolResult(success=False, error="replace is required")

        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}"
                )

            original_content = path.read_text(encoding="utf-8")
            content = original_content

            if regex:
                if global_replace:
                    content = re.sub(search, replace, content, flags=re.MULTILINE)
                else:
                    content = re.sub(search, replace, content, flags=re.MULTILINE, count=1)
            else:
                if global_replace:
                    content = content.replace(search, replace)
                else:
                    content = content.replace(search, replace, 1)

            if content == original_content:
                return ToolResult(
                    success=False,
                    error="Search pattern not found in file. Check indentation and content carefully."
                )

            path.write_text(content, encoding="utf-8")

            replacements_made = original_content.count(search) if not regex else 1

            logger.info(f"[EditTool] Edited file: {path}, replacements: {replacements_made}")

            return ToolResult(
                success=True,
                output=f"File edited successfully: {path}",
                metadata={
                    "file_path": str(path),
                    "replacements_made": replacements_made,
                    "original_size": len(original_content),
                    "new_size": len(content)
                }
            )

        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied: {file_path}")
        except Exception as e:
            logger.exception(f"[EditTool] Failed to edit file: {e}")
            return ToolResult(success=False, error=f"Failed to edit file: {str(e)}")

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to base path."""
        path = Path(file_path)
        if not path.is_absolute() and self._base_path:
            path = self._base_path / path
        return path.resolve()


class GlobTool(Tool):
    """Tool for finding files by pattern."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Glob tool.

        Args:
            base_path: Base path for glob patterns
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._definition = ToolDefinition(
            name="glob",
            description="Find files matching a glob pattern. "
                        "Supports wildcards like **/*.java for recursive search.",
            category=ToolCategory.SEARCH,
            parameters=[
                create_tool_parameter(
                    name="pattern",
                    param_type="string",
                    description="Glob pattern to match files. Examples: **/*.java, src/**/*.py",
                    required=True
                ),
                create_tool_parameter(
                    name="path",
                    param_type="string",
                    description="Base directory to search in. Default: current directory",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="recursive",
                    param_type="boolean",
                    description="Search recursively. Default: true for patterns with **",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="include_hidden",
                    param_type="boolean",
                    description="Include hidden files (starting with .). Default: false",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {"pattern": "**/*.java"},
                    "description": "Find all Java files recursively"
                },
                {
                    "params": {"pattern": "src/**/*.Test.java"},
                    "description": "Find test files in src directory"
                }
            ],
            tags=["search", "find", "glob", "file"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute glob pattern matching."""
        pattern = kwargs.get("pattern")
        search_path = kwargs.get("path")
        recursive = kwargs.get("recursive")
        include_hidden = kwargs.get("include_hidden", False)

        if not pattern:
            return ToolResult(success=False, error="pattern is required")

        try:
            if search_path:
                base = Path(search_path)
            elif self._base_path:
                base = self._base_path
            else:
                base = Path.cwd()

            if recursive is None:
                recursive = "**" in pattern

            if not recursive:
                pattern_parts = pattern.split("/")
                if len(pattern_parts) == 1:
                    pattern_parts = pattern_parts[0].split("\\")
                pattern = pattern_parts[-1]

            matches = list(base.glob(pattern))

            if not include_hidden:
                matches = [m for m in matches if not any(
                    part.startswith(".") for part in m.parts
                )]

            paths = [str(m.relative_to(base)) for m in matches if m.is_file()]

            logger.info(f"[GlobTool] Pattern: {pattern}, matches: {len(paths)}")

            return ToolResult(
                success=True,
                output=paths,
                metadata={
                    "pattern": pattern,
                    "base_path": str(base),
                    "total_matches": len(paths)
                }
            )

        except Exception as e:
            logger.exception(f"[GlobTool] Failed to glob: {e}")
            return ToolResult(success=False, error=f"Glob failed: {str(e)}")


class GrepTool(Tool):
    """Tool for searching file contents."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Grep tool.

        Args:
            base_path: Base path for grep search
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._definition = ToolDefinition(
            name="grep",
            description="Search for text patterns in files. "
                        "Returns matching lines with file path and line number.",
            category=ToolCategory.SEARCH,
            parameters=[
                create_tool_parameter(
                    name="pattern",
                    param_type="string",
                    description="Text pattern or regex to search for",
                    required=True
                ),
                create_tool_parameter(
                    name="path",
                    param_type="string",
                    description="Directory or file path to search in. Default: current directory",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="glob",
                    param_type="string",
                    description="File glob pattern to filter files (e.g., *.java)",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="regex",
                    param_type="boolean",
                    description="Treat pattern as regular expression. Default: true",
                    required=False,
                    default=True
                ),
                create_tool_parameter(
                    name="ignore_case",
                    param_type="boolean",
                    description="Case insensitive search. Default: false",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="max_results",
                    param_type="integer",
                    description="Maximum number of results to return. Default: 100",
                    required=False,
                    default=100
                )
            ],
            examples=[
                {
                    "params": {"pattern": "class.*Test", "glob": "*.java"},
                    "description": "Find test class definitions in Java files"
                },
                {
                    "params": {"pattern": "TODO", "ignore_case": True},
                    "description": "Find TODO comments"
                }
            ],
            tags=["search", "grep", "find", "regex"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute grep search."""
        pattern = kwargs.get("pattern")
        search_path = kwargs.get("path")
        file_glob = kwargs.get("glob")
        regex = kwargs.get("regex", True)
        ignore_case = kwargs.get("ignore_case", False)
        max_results = kwargs.get("max_results", 100)

        if not pattern:
            return ToolResult(success=False, error="pattern is required")

        try:
            if search_path:
                base = Path(search_path)
            elif self._base_path:
                base = self._base_path
            else:
                base = Path.cwd()

            if not base.exists():
                return ToolResult(success=False, error=f"Path not found: {base}")

            flags = 0
            if ignore_case:
                flags |= re.IGNORECASE

            if regex:
                compiled_pattern = re.compile(pattern, flags)
            else:
                compiled_pattern = re.compile(re.escape(pattern), flags)

            results = []
            files_to_search = []

            if base.is_file():
                files_to_search = [base]
            elif file_glob:
                files_to_search = list(base.glob(file_glob))
            else:
                if base.is_dir():
                    files_to_search = [f for f in base.rglob("*") if f.is_file()]

            for file_path in files_to_search:
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                        lines = content.split("\n")

                        for line_num, line in enumerate(lines, start=1):
                            if compiled_pattern.search(line):
                                results.append({
                                    "file": str(file_path.relative_to(base) if base.is_dir() else file_path),
                                    "line": line_num,
                                    "content": line.strip()
                                })

                                if len(results) >= max_results:
                                    break
                    except Exception:
                        continue

                if len(results) >= max_results:
                    break

            logger.info(f"[GrepTool] Pattern: {pattern}, results: {len(results)}")

            return ToolResult(
                success=True,
                output=results,
                metadata={
                    "pattern": pattern,
                    "base_path": str(base),
                    "total_matches": len(results),
                    "max_results": max_results,
                    "truncated": len(results) >= max_results
                }
            )

        except Exception as e:
            logger.exception(f"[GrepTool] Failed to grep: {e}")
            return ToolResult(success=False, error=f"Grep failed: {str(e)}")


class BashTool(Tool):
    """Tool for executing shell commands."""

    def __init__(
        self,
        base_path: Optional[str] = None,
        allowed_commands: Optional[List[str]] = None,
        timeout: int = 60
    ):
        """Initialize Bash tool.

        Args:
            base_path: Working directory for commands
            allowed_commands: List of allowed command prefixes (empty = all allowed)
            timeout: Command timeout in seconds
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else None
        self._allowed_commands = allowed_commands or []
        self._timeout = timeout

        self._definition = ToolDefinition(
            name="bash",
            description="Execute shell commands. Use for running builds, tests, "
                        "or other command-line operations.",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="command",
                    param_type="string",
                    description="Command to execute",
                    required=True
                ),
                create_tool_parameter(
                    name="timeout",
                    param_type="integer",
                    description="Timeout in seconds. Default: 60",
                    required=False,
                    default=60
                ),
                create_tool_parameter(
                    name="env",
                    param_type="object",
                    description="Environment variables to set",
                    required=False,
                    default=None
                )
            ],
            examples=[
                {
                    "params": {"command": "mvn test -Dtest=AppTest"},
                    "description": "Run specific test with Maven"
                },
                {
                    "params": {"command": "ls -la"},
                    "description": "List files in current directory"
                }
            ],
            tags=["command", "bash", "shell", "execute", "run"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute bash command."""
        command = kwargs.get("command")
        timeout = kwargs.get("timeout", self._timeout)
        env = kwargs.get("env")

        if not command:
            return ToolResult(success=False, error="command is required")

        if self._allowed_commands:
            allowed = any(command.startswith(allowed) for allowed in self._allowed_commands)
            if not allowed:
                return ToolResult(
                    success=False,
                    error=f"Command not allowed. Allowed prefixes: {self._allowed_commands}"
                )

        try:
            cwd = str(self._base_path) if self._base_path else None

            import platform
            shell = platform.system() == "Windows"
            cmd = command if shell else ["bash", "-c", command]

            process = await asyncio.create_subprocess_exec(
                *cmd if not shell else [command],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {timeout} seconds",
                    metadata={"timeout": True}
                )

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            logger.info(f"[BashTool] Command: {command}, exit code: {process.returncode}")

            return ToolResult(
                success=process.returncode == 0,
                output=stdout_str,
                error=stderr_str if process.returncode != 0 else None,
                metadata={
                    "command": command,
                    "exit_code": process.returncode,
                    "timeout": False
                }
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=f"Command not found: {command.split()[0]}"
            )
        except Exception as e:
            logger.exception(f"[BashTool] Command failed: {e}")
            return ToolResult(success=False, error=f"Command execution failed: {str(e)}")
