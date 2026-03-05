"""File Tools using Tool Abstraction Layer.

This module provides file operation tools:
- ReadFileTool: Read file contents
- WriteFileTool: Write file contents
- DeleteFileTool: Delete files
- ListFilesTool: List directory contents
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core import (
    ToolBase,
    ToolCategory,
    ToolParameter,
    ToolResult,
    ToolContext,
)

logger = logging.getLogger(__name__)


class ReadFileTool(ToolBase):
    """Tool for reading file contents."""
    
    name = "read_file"
    description = "Read the contents of a file"
    category = ToolCategory.FILE
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Path to the file to read",
            required=True,
        ),
        ToolParameter(
            name="encoding",
            type="string",
            description="File encoding (default: utf-8)",
            required=False,
            default="utf-8",
        ),
    ]
    tags = ["file", "read", "io"]
    
    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute file read.
        
        Args:
            params: Tool parameters
            context: Execution context
            
        Returns:
            ToolResult with file contents
        """
        file_path = Path(params["file_path"])
        encoding = params.get("encoding", "utf-8")
        
        ctx = context or self._context
        
        if ctx:
            file_path = ctx.resolve_path(file_path)
        
        if not file_path.exists():
            return ToolResult.fail(
                error=f"File not found: {file_path}",
                code="FILE_NOT_FOUND",
            )
        
        if not file_path.is_file():
            return ToolResult.fail(
                error=f"Not a file: {file_path}",
                code="NOT_A_FILE",
            )
        
        try:
            content = file_path.read_text(encoding=encoding)
            
            return ToolResult.ok(
                output=content,
                data={
                    "file_path": str(file_path),
                    "size": len(content),
                    "encoding": encoding,
                },
            )
            
        except UnicodeDecodeError as e:
            return ToolResult.fail(
                error=f"Failed to decode file: {e}",
                code="DECODE_ERROR",
            )
        except Exception as e:
            return ToolResult.fail(
                error=f"Failed to read file: {e}",
                code="READ_ERROR",
            )


class WriteFileTool(ToolBase):
    """Tool for writing file contents."""
    
    name = "write_file"
    description = "Write content to a file"
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
            description="File encoding (default: utf-8)",
            required=False,
            default="utf-8",
        ),
        ToolParameter(
            name="create_dirs",
            type="boolean",
            description="Create parent directories if needed",
            required=False,
            default=True,
        ),
    ]
    tags = ["file", "write", "io"]
    
    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute file write.
        
        Args:
            params: Tool parameters
            context: Execution context
            
        Returns:
            ToolResult with write status
        """
        file_path = Path(params["file_path"])
        content = params["content"]
        encoding = params.get("encoding", "utf-8")
        create_dirs = params.get("create_dirs", True)
        
        ctx = context or self._context
        
        if ctx:
            file_path = ctx.resolve_path(file_path)
        
        try:
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_path.write_text(content, encoding=encoding)
            
            if ctx:
                ctx.track_modified_file(file_path)
            
            return ToolResult.ok(
                output=f"Successfully wrote {len(content)} bytes to {file_path}",
                data={
                    "file_path": str(file_path),
                    "size": len(content),
                },
                artifacts=[str(file_path)],
            )
            
        except Exception as e:
            return ToolResult.fail(
                error=f"Failed to write file: {e}",
                code="WRITE_ERROR",
            )


class DeleteFileTool(ToolBase):
    """Tool for deleting files."""
    
    name = "delete_file"
    description = "Delete a file"
    category = ToolCategory.FILE
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Path to the file to delete",
            required=True,
        ),
    ]
    tags = ["file", "delete", "io"]
    
    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute file delete.
        
        Args:
            params: Tool parameters
            context: Execution context
            
        Returns:
            ToolResult with delete status
        """
        file_path = Path(params["file_path"])
        
        ctx = context or self._context
        
        if ctx:
            file_path = ctx.resolve_path(file_path)
        
        if not file_path.exists():
            return ToolResult.fail(
                error=f"File not found: {file_path}",
                code="FILE_NOT_FOUND",
            )
        
        try:
            file_path.unlink()
            
            return ToolResult.ok(
                output=f"Successfully deleted {file_path}",
                data={"file_path": str(file_path)},
            )
            
        except Exception as e:
            return ToolResult.fail(
                error=f"Failed to delete file: {e}",
                code="DELETE_ERROR",
            )


class ListFilesTool(ToolBase):
    """Tool for listing directory contents."""
    
    name = "list_files"
    description = "List files in a directory"
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
            description="Glob pattern to filter files",
            required=False,
            default="*",
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="List files recursively",
            required=False,
            default=False,
        ),
    ]
    tags = ["file", "list", "directory"]
    
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
        directory = Path(params["directory"])
        pattern = params.get("pattern", "*")
        recursive = params.get("recursive", False)
        
        ctx = context or self._context
        
        if ctx:
            directory = ctx.resolve_path(directory)
        
        if not directory.exists():
            return ToolResult.fail(
                error=f"Directory not found: {directory}",
                code="DIR_NOT_FOUND",
            )
        
        if not directory.is_dir():
            return ToolResult.fail(
                error=f"Not a directory: {directory}",
                code="NOT_A_DIR",
            )
        
        try:
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
            
            file_list = [
                {
                    "path": str(f.relative_to(directory)),
                    "is_dir": f.is_dir(),
                    "size": f.stat().st_size if f.is_file() else 0,
                }
                for f in files
            ]
            
            return ToolResult.ok(
                output=f"Found {len(file_list)} items",
                data={
                    "directory": str(directory),
                    "files": file_list,
                    "count": len(file_list),
                },
            )
            
        except Exception as e:
            return ToolResult.fail(
                error=f"Failed to list directory: {e}",
                code="LIST_ERROR",
            )
