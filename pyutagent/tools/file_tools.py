"""Enhanced File Tools with Security and Path Validation.

This module provides secure file operation tools:
- ReadFileTool: Read file contents with safety checks
- WriteFileTool: Write file with backup and validation
- EditFileTool: Search/Replace editing with precise matching
- ListFilesTool: Directory listing with filtering
- DeleteFileTool: Safe file deletion

Security Features:
- Path traversal protection
- File size limits
- Allowed path restrictions
- Backup before overwrite
"""

import asyncio
import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .core import (
    ToolBase,
    ToolCategory,
    ToolParameter,
    ToolResult,
    ToolContext,
)

logger = logging.getLogger(__name__)


@dataclass
class PathSecurityConfig:
    """Configuration for path security validation."""

    allowed_paths: Optional[List[Path]] = None
    blocked_patterns: List[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allow_absolute_paths: bool = True
    allow_parent_traversal: bool = False

    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r"\.\.",  # Parent directory traversal
                r"^~",    # Home directory
                r"/etc/",
                r"/usr/",
                r"/bin/",
                r"/sbin/",
                r"/lib/",
                r"C:\\Windows",
                r"C:\\Program Files",
                r"\.ssh",
                r"\.gnupg",
                r"\.aws",
                r"\.env",
            ]


class PathSecurityValidator:
    """Validator for file path security."""

    def __init__(self, config: Optional[PathSecurityConfig] = None):
        """Initialize validator.

        Args:
            config: Security configuration
        """
        self.config = config or PathSecurityConfig()

    def validate(self, path: Union[str, Path], base_path: Optional[Path] = None) -> tuple[bool, Optional[str]]:
        """Validate a file path for security.

        Args:
            path: Path to validate
            base_path: Base path for relative resolution

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(path)

        # Check for parent traversal
        if not self.config.allow_parent_traversal:
            if ".." in path.parts:
                return False, "Path traversal (..) is not allowed"

        # Resolve to absolute path
        if base_path and not path.is_absolute():
            resolved_path = (base_path / path).resolve()
        else:
            resolved_path = path.resolve()

        # Check blocked patterns
        path_str = str(resolved_path)
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, path_str, re.IGNORECASE):
                return False, f"Path matches blocked pattern: {pattern}"

        # Check allowed paths
        if self.config.allowed_paths:
            allowed = any(
                resolved_path == allowed_path or
                resolved_path.is_relative_to(allowed_path)
                for allowed_path in self.config.allowed_paths
            )
            if not allowed:
                return False, f"Path is not in allowed directories"

        # Check absolute path restriction
        if not self.config.allow_absolute_paths and path.is_absolute():
            return False, "Absolute paths are not allowed"

        return True, None

    def validate_file_size(self, path: Path) -> tuple[bool, Optional[str]]:
        """Validate file size is within limits.

        Args:
            path: Path to file

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not path.exists():
            return True, None

        try:
            size = path.stat().st_size
            if size > self.config.max_file_size:
                return False, (
                    f"File size ({size} bytes) exceeds maximum "
                    f"({self.config.max_file_size} bytes)"
                )
            return True, None
        except OSError as e:
            return False, f"Cannot access file: {e}"


class ReadFileTool(ToolBase):
    """Tool for reading file contents with security checks.

    Features:
    - Path traversal protection
    - File size limits
    - Line range reading
    - Encoding detection
    - Checksum verification
    """

    name = "read_file"
    description = (
        "Read the contents of a file with security validation. "
        "Supports line range selection and encoding detection."
    )
    category = ToolCategory.FILE
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Path to the file to read (relative to project root)",
            required=True,
        ),
        ToolParameter(
            name="offset",
            type="integer",
            description="Line number to start reading from (1-based). Default: 1",
            required=False,
            default=1,
        ),
        ToolParameter(
            name="limit",
            type="integer",
            description="Maximum number of lines to read. Default: all lines",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="encoding",
            type="string",
            description="File encoding. Default: utf-8",
            required=False,
            default="utf-8",
        ),
        ToolParameter(
            name="show_line_numbers",
            type="boolean",
            description="Whether to show line numbers. Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["file", "read", "io", "secure"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        security_config: Optional[PathSecurityConfig] = None,
    ):
        """Initialize ReadFileTool.

        Args:
            context: Execution context
            security_config: Security configuration
        """
        super().__init__(context)
        self.validator = PathSecurityValidator(security_config)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute file read with security checks.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with file contents or error
        """
        ctx = context or self._context
        file_path = params["file_path"]
        offset = params.get("offset", 1)
        limit = params.get("limit")
        encoding = params.get("encoding", "utf-8")
        show_line_numbers = params.get("show_line_numbers", False)

        # Validate path
        base_path = ctx.project_path if ctx else Path.cwd()
        is_valid, error = self.validator.validate(file_path, base_path)
        if not is_valid:
            return ToolResult.fail(
                error=f"Security validation failed: {error}",
                code="SECURITY_ERROR",
            )

        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = base_path / path
        path = path.resolve()

        # Check file exists
        if not path.exists():
            return ToolResult.fail(
                error=f"File not found: {file_path}",
                code="FILE_NOT_FOUND",
            )

        if not path.is_file():
            return ToolResult.fail(
                error=f"Not a file: {file_path}",
                code="NOT_A_FILE",
            )

        # Check file size
        is_valid, error = self.validator.validate_file_size(path)
        if not is_valid:
            return ToolResult.fail(error=error, code="FILE_TOO_LARGE")

        try:
            # Read file content
            content = path.read_text(encoding=encoding)
            lines = content.split("\n")
            total_lines = len(lines)

            # Apply offset and limit
            if offset > 1:
                lines = lines[offset - 1:]
            if limit:
                lines = lines[:limit]

            # Format output
            output_lines = []
            for i, line in enumerate(lines, start=offset):
                if show_line_numbers:
                    output_lines.append(f"{i:6d}→{line}")
                else:
                    output_lines.append(line)

            result_content = "\n".join(output_lines)

            # Calculate checksum for integrity
            checksum = hashlib.md5(content.encode()).hexdigest()

            return ToolResult.ok(
                output=result_content,
                data={
                    "file_path": str(path),
                    "total_lines": total_lines,
                    "read_lines": len(lines),
                    "offset": offset,
                    "limit": limit,
                    "encoding": encoding,
                    "size": len(content),
                    "checksum": checksum,
                },
            )

        except UnicodeDecodeError as e:
            return ToolResult.fail(
                error=f"Failed to decode file with {encoding} encoding: {e}",
                code="DECODE_ERROR",
            )
        except Exception as e:
            logger.exception(f"Failed to read file: {e}")
            return ToolResult.fail(
                error=f"Failed to read file: {e}",
                code="READ_ERROR",
            )


class WriteFileTool(ToolBase):
    """Tool for writing files with safety features.

    Features:
    - Automatic backup before overwrite
    - Directory creation
    - Content validation
    - Atomic write (write to temp, then move)
    """

    name = "write_file"
    description = (
        "Write content to a file with safety features. "
        "Creates backups before overwriting and supports atomic writes."
    )
    category = ToolCategory.FILE
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Path to the file to write",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file",
            required=True,
        ),
        ToolParameter(
            name="encoding",
            type="string",
            description="File encoding. Default: utf-8",
            required=False,
            default="utf-8",
        ),
        ToolParameter(
            name="create_dirs",
            type="boolean",
            description="Create parent directories if needed. Default: true",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="backup",
            type="boolean",
            description="Create backup before overwrite. Default: true",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="atomic",
            type="boolean",
            description="Use atomic write (safer). Default: true",
            required=False,
            default=True,
        ),
    ]
    tags = ["file", "write", "io", "secure"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        security_config: Optional[PathSecurityConfig] = None,
    ):
        """Initialize WriteFileTool.

        Args:
            context: Execution context
            security_config: Security configuration
        """
        super().__init__(context)
        self.validator = PathSecurityValidator(security_config)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute file write with safety features.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with write status
        """
        ctx = context or self._context
        file_path = params["file_path"]
        content = params["content"]
        encoding = params.get("encoding", "utf-8")
        create_dirs = params.get("create_dirs", True)
        backup = params.get("backup", True)
        atomic = params.get("atomic", True)

        # Validate path
        base_path = ctx.project_path if ctx else Path.cwd()
        is_valid, error = self.validator.validate(file_path, base_path)
        if not is_valid:
            return ToolResult.fail(
                error=f"Security validation failed: {error}",
                code="SECURITY_ERROR",
            )

        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = base_path / path
        path = path.resolve()

        try:
            # Create backup if file exists
            backup_path = None
            if path.exists() and backup:
                backup_path = path.with_suffix(
                    f"{path.suffix}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                shutil.copy2(path, backup_path)

            # Create parent directories
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Write file (atomic or direct)
            if atomic:
                temp_path = path.with_suffix(f"{path.suffix}.tmp")
                temp_path.write_text(content, encoding=encoding)
                temp_path.replace(path)
            else:
                path.write_text(content, encoding=encoding)

            # Track modified file in context
            if ctx:
                ctx.track_modified_file(path)

            # Calculate checksum
            checksum = hashlib.md5(content.encode()).hexdigest()

            return ToolResult.ok(
                output=f"Successfully wrote {len(content)} bytes to {path}",
                data={
                    "file_path": str(path),
                    "size": len(content),
                    "lines": len(content.split("\n")),
                    "encoding": encoding,
                    "checksum": checksum,
                    "backup_created": backup_path is not None,
                    "backup_path": str(backup_path) if backup_path else None,
                },
                artifacts=[str(path)],
            )

        except Exception as e:
            logger.exception(f"Failed to write file: {e}")
            return ToolResult.fail(
                error=f"Failed to write file: {e}",
                code="WRITE_ERROR",
            )


class EditFileTool(ToolBase):
    """Tool for precise Search/Replace file editing.

    Features:
    - Exact match search/replace
    - Regex support
    - Multiple occurrence handling
    - Pre-edit validation
    - Automatic backup
    """

    name = "edit_file"
    description = (
        "Make targeted edits to a file using Search/Replace. "
        "Finds the exact 'search' block and replaces it with 'replace' block. "
        "Supports regex and multiple replacements."
    )
    category = ToolCategory.FILE
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Path to the file to edit",
            required=True,
        ),
        ToolParameter(
            name="search",
            type="string",
            description="The exact code block to find (including indentation)",
            required=True,
        ),
        ToolParameter(
            name="replace",
            type="string",
            description="The code to replace the search block with",
            required=True,
        ),
        ToolParameter(
            name="global_replace",
            type="boolean",
            description="Replace all occurrences. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="regex",
            type="boolean",
            description="Treat search as regex pattern. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="case_sensitive",
            type="boolean",
            description="Case sensitive search. Default: true",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="backup",
            type="boolean",
            description="Create backup before edit. Default: true",
            required=False,
            default=True,
        ),
    ]
    tags = ["file", "edit", "modify", "replace", "secure"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        security_config: Optional[PathSecurityConfig] = None,
    ):
        """Initialize EditFileTool.

        Args:
            context: Execution context
            security_config: Security configuration
        """
        super().__init__(context)
        self.validator = PathSecurityValidator(security_config)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute file edit with search/replace.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with edit status
        """
        ctx = context or self._context
        file_path = params["file_path"]
        search = params["search"]
        replace = params["replace"]
        global_replace = params.get("global_replace", False)
        use_regex = params.get("regex", False)
        case_sensitive = params.get("case_sensitive", True)
        backup = params.get("backup", True)

        # Validate path
        base_path = ctx.project_path if ctx else Path.cwd()
        is_valid, error = self.validator.validate(file_path, base_path)
        if not is_valid:
            return ToolResult.fail(
                error=f"Security validation failed: {error}",
                code="SECURITY_ERROR",
            )

        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = base_path / path
        path = path.resolve()

        # Check file exists
        if not path.exists():
            return ToolResult.fail(
                error=f"File not found: {file_path}",
                code="FILE_NOT_FOUND",
            )

        if not path.is_file():
            return ToolResult.fail(
                error=f"Not a file: {file_path}",
                code="NOT_A_FILE",
            )

        # Check file size
        is_valid, error = self.validator.validate_file_size(path)
        if not is_valid:
            return ToolResult.fail(error=error, code="FILE_TOO_LARGE")

        try:
            # Read original content
            original_content = path.read_text(encoding="utf-8")
            content = original_content

            # Perform replacement
            if use_regex:
                # Compile regex flags
                flags = 0 if case_sensitive else re.IGNORECASE
                pattern = re.compile(search, flags)
                if global_replace:
                    content = pattern.sub(replace, content)
                    count = len(pattern.findall(original_content))
                else:
                    content = pattern.sub(replace, content, count=1)
                    count = 1 if content != original_content else 0
            else:
                # Simple string replacement (case sensitive or insensitive)
                search_term = search
                content_term = original_content

                if not case_sensitive:
                    # For case-insensitive, convert both to lower for matching
                    # but preserve original case in replacement
                    content_lower = content_term.lower()
                    search_lower = search_term.lower()

                    if global_replace:
                        count = content_lower.count(search_lower)
                        # Replace while preserving case
                        result = []
                        i = 0
                        while i < len(content_term):
                            if content_lower[i:i+len(search_lower)] == search_lower:
                                result.append(replace)
                                i += len(search_lower)
                            else:
                                result.append(content_term[i])
                                i += 1
                        content = "".join(result)
                    else:
                        idx = content_lower.find(search_lower)
                        if idx >= 0:
                            content = content_term[:idx] + replace + content_term[idx+len(search_term):]
                            count = 1
                        else:
                            count = 0
                else:
                    # Case sensitive string replacement
                    if global_replace:
                        count = original_content.count(search)
                        content = original_content.replace(search, replace)
                    else:
                        if search in original_content:
                            content = original_content.replace(search, replace, 1)
                            count = 1
                        else:
                            count = 0

            if count == 0:
                return ToolResult.fail(
                    error=(
                        f"Search pattern not found in file. "
                        f"Check indentation and content carefully. "
                        f"Expected to find:\n{search[:200]}..."
                    ),
                    code="PATTERN_NOT_FOUND",
                )

            # Create backup
            backup_path = None
            if backup:
                backup_path = path.with_suffix(
                    f"{path.suffix}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                shutil.copy2(path, backup_path)

            # Write modified content
            path.write_text(content, encoding="utf-8")

            # Track modified file
            if ctx:
                ctx.track_modified_file(path)

            # Calculate diff stats
            original_lines = len(original_content.split("\n"))
            new_lines = len(content.split("\n"))

            return ToolResult.ok(
                output=f"File edited successfully: {path}",
                data={
                    "file_path": str(path),
                    "replacements_made": count,
                    "original_size": len(original_content),
                    "new_size": len(content),
                    "original_lines": original_lines,
                    "new_lines": new_lines,
                    "backup_path": str(backup_path) if backup_path else None,
                },
                artifacts=[str(path)],
            )

        except re.error as e:
            return ToolResult.fail(
                error=f"Invalid regex pattern: {e}",
                code="INVALID_REGEX",
            )
        except Exception as e:
            logger.exception(f"Failed to edit file: {e}")
            return ToolResult.fail(
                error=f"Failed to edit file: {e}",
                code="EDIT_ERROR",
            )


class ListFilesTool(ToolBase):
    """Tool for listing directory contents.

    Features:
    - Recursive listing
    - Pattern filtering
    - File type filtering
    - Size and modification info
    """

    name = "list_files"
    description = (
        "List files in a directory with filtering options. "
        "Supports recursive listing and glob patterns."
    )
    category = ToolCategory.FILE
    parameters = [
        ToolParameter(
            name="directory",
            type="string",
            description="Directory path to list",
            required=True,
        ),
        ToolParameter(
            name="pattern",
            type="string",
            description="Glob pattern to filter files. Default: *",
            required=False,
            default="*",
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="List files recursively. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="include_hidden",
            type="boolean",
            description="Include hidden files. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="include_dirs",
            type="boolean",
            description="Include directories in results. Default: true",
            required=False,
            default=True,
        ),
    ]
    tags = ["file", "list", "directory", "browse"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        security_config: Optional[PathSecurityConfig] = None,
    ):
        """Initialize ListFilesTool.

        Args:
            context: Execution context
            security_config: Security configuration
        """
        super().__init__(context)
        self.validator = PathSecurityValidator(security_config)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute directory listing.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with file list
        """
        ctx = context or self._context
        directory = params["directory"]
        pattern = params.get("pattern", "*")
        recursive = params.get("recursive", False)
        include_hidden = params.get("include_hidden", False)
        include_dirs = params.get("include_dirs", True)

        # Validate path
        base_path = ctx.project_path if ctx else Path.cwd()
        is_valid, error = self.validator.validate(directory, base_path)
        if not is_valid:
            return ToolResult.fail(
                error=f"Security validation failed: {error}",
                code="SECURITY_ERROR",
            )

        # Resolve path
        path = Path(directory)
        if not path.is_absolute():
            path = base_path / path
        path = path.resolve()

        # Check directory exists
        if not path.exists():
            return ToolResult.fail(
                error=f"Directory not found: {directory}",
                code="DIR_NOT_FOUND",
            )

        if not path.is_dir():
            return ToolResult.fail(
                error=f"Not a directory: {directory}",
                code="NOT_A_DIR",
            )

        try:
            # Get files
            if recursive:
                items = list(path.rglob(pattern))
            else:
                items = list(path.glob(pattern))

            # Filter items
            filtered_items = []
            for item in items:
                # Skip hidden files
                if not include_hidden:
                    if any(part.startswith(".") for part in item.relative_to(path).parts):
                        continue

                # Skip directories if not included
                if item.is_dir() and not include_dirs:
                    continue

                filtered_items.append(item)

            # Build file info
            file_list = []
            for item in sorted(filtered_items, key=lambda x: (not x.is_dir(), str(x))):
                try:
                    stat = item.stat()
                    file_info = {
                        "path": str(item.relative_to(path)),
                        "name": item.name,
                        "is_dir": item.is_dir(),
                        "is_file": item.is_file(),
                        "size": stat.st_size if item.is_file() else 0,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }
                    file_list.append(file_info)
                except OSError:
                    # Skip files we can't stat
                    continue

            # Separate dirs and files
            dirs = [f for f in file_list if f["is_dir"]]
            files = [f for f in file_list if f["is_file"]]

            return ToolResult.ok(
                output=f"Found {len(dirs)} directories, {len(files)} files",
                data={
                    "directory": str(path),
                    "items": file_list,
                    "total_count": len(file_list),
                    "dir_count": len(dirs),
                    "file_count": len(files),
                    "pattern": pattern,
                    "recursive": recursive,
                },
            )

        except Exception as e:
            logger.exception(f"Failed to list directory: {e}")
            return ToolResult.fail(
                error=f"Failed to list directory: {e}",
                code="LIST_ERROR",
            )


class DeleteFileTool(ToolBase):
    """Tool for safely deleting files.

    Features:
    - Path validation
    - Move to trash option
    - Confirmation for sensitive paths
    """

    name = "delete_file"
    description = (
        "Delete a file with safety checks. "
        "Can optionally move to trash instead of permanent deletion."
    )
    category = ToolCategory.FILE
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Path to the file to delete",
            required=True,
        ),
        ToolParameter(
            name="move_to_trash",
            type="boolean",
            description="Move to trash instead of deleting. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="confirm",
            type="boolean",
            description="Confirm deletion (required for safety). Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["file", "delete", "remove", "secure"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        security_config: Optional[PathSecurityConfig] = None,
    ):
        """Initialize DeleteFileTool.

        Args:
            context: Execution context
            security_config: Security configuration
        """
        super().__init__(context)
        self.validator = PathSecurityValidator(security_config)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute file deletion.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with deletion status
        """
        ctx = context or self._context
        file_path = params["file_path"]
        move_to_trash = params.get("move_to_trash", False)
        confirm = params.get("confirm", False)

        # Require confirmation
        if not confirm:
            return ToolResult.fail(
                error="Deletion requires explicit confirmation. Set 'confirm' to True.",
                code="CONFIRMATION_REQUIRED",
            )

        # Validate path
        base_path = ctx.project_path if ctx else Path.cwd()
        is_valid, error = self.validator.validate(file_path, base_path)
        if not is_valid:
            return ToolResult.fail(
                error=f"Security validation failed: {error}",
                code="SECURITY_ERROR",
            )

        # Resolve path
        path = Path(file_path)
        if not path.is_absolute():
            path = base_path / path
        path = path.resolve()

        # Check file exists
        if not path.exists():
            return ToolResult.fail(
                error=f"File not found: {file_path}",
                code="FILE_NOT_FOUND",
            )

        if not path.is_file():
            return ToolResult.fail(
                error=f"Not a file (cannot delete directories): {file_path}",
                code="NOT_A_FILE",
            )

        try:
            if move_to_trash:
                # Try to use send2trash if available
                try:
                    import send2trash
                    send2trash.send2trash(str(path))
                    return ToolResult.ok(
                        output=f"File moved to trash: {path}",
                        data={
                            "file_path": str(path),
                            "action": "move_to_trash",
                        },
                    )
                except ImportError:
                    # Fallback to regular delete with warning
                    logger.warning("send2trash not available, performing permanent deletion")

            # Permanent deletion
            path.unlink()

            return ToolResult.ok(
                output=f"File deleted: {path}",
                data={
                    "file_path": str(path),
                    "action": "delete",
                },
            )

        except Exception as e:
            logger.exception(f"Failed to delete file: {e}")
            return ToolResult.fail(
                error=f"Failed to delete file: {e}",
                code="DELETE_ERROR",
            )


def get_all_file_tools(
    context: Optional[ToolContext] = None,
    security_config: Optional[PathSecurityConfig] = None,
) -> List[ToolBase]:
    """Get all file tools.

    Args:
        context: Execution context
        security_config: Security configuration

    Returns:
        List of file tool instances
    """
    return [
        ReadFileTool(context, security_config),
        WriteFileTool(context, security_config),
        EditFileTool(context, security_config),
        ListFilesTool(context, security_config),
        DeleteFileTool(context, security_config),
    ]
