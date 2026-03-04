"""Git tool implementations.

This module provides:
- GitStatusTool: Check repository status
- GitDiffTool: View file changes
- GitCommitTool: Commit changes
- GitBranchTool: Branch management
- GitLogTool: View commit history
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tool import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    create_tool_parameter
)

logger = logging.getLogger(__name__)


class GitStatusTool(Tool):
    """Git status - Check repository status."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Git status tool.

        Args:
            base_path: Base path for git repository
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._definition = ToolDefinition(
            name="git_status",
            description="Check the current status of the Git repository. "
                        "Shows modified files, staged changes, and untracked files.",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="path",
                    param_type="string",
                    description="Path to the git repository. Default: current directory",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="short",
                    param_type="boolean",
                    description="Show short format output. Default: false",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "Check git status of current directory"
                },
                {
                    "params": {"short": True},
                    "description": "Get short format status"
                }
            ],
            tags=["git", "status", "repository"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute git status."""
        path = kwargs.get("path")
        short = kwargs.get("short", False)

        try:
            cwd = Path(path) if path else self._base_path
            
            # Build command
            cmd = ["git", "status"]
            if short:
                cmd.append("--short")
            else:
                cmd.append("--porcelain")

            # Execute git status
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd)
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Unknown error"
                return ToolResult(
                    success=False,
                    error=f"Git status failed: {error_msg}"
                )

            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            
            # Parse status output
            status_info = self._parse_status(output, short)

            logger.info(f"[GitStatusTool] Status checked in {cwd}")

            return ToolResult(
                success=True,
                output=status_info,
                metadata={
                    "path": str(cwd),
                    "short_format": short,
                    "has_changes": len(status_info.get("modified", [])) > 0 or 
                                   len(status_info.get("staged", [])) > 0 or
                                   len(status_info.get("untracked", [])) > 0
                }
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git not found. Please ensure git is installed and in PATH."
            )
        except Exception as e:
            logger.exception(f"[GitStatusTool] Failed: {e}")
            return ToolResult(success=False, error=f"Git status failed: {str(e)}")

    def _parse_status(self, output: str, short: bool) -> Dict[str, Any]:
        """Parse git status output."""
        status_info = {
            "modified": [],
            "staged": [],
            "untracked": [],
            "deleted": [],
            "renamed": [],
            "raw_output": output
        }

        if not output.strip():
            return status_info

        lines = output.strip().split("\n")
        
        for line in lines:
            if not line.strip():
                continue

            if short:
                # Short format: XY filename
                if len(line) >= 3:
                    x, y = line[0], line[1]
                    filename = line[3:]
                    
                    if x in "MADRC" or y in "MADRC":
                        if x != " " and x != "?":
                            status_info["staged"].append(filename)
                        if y == "M":
                            status_info["modified"].append(filename)
                        elif y == "D":
                            status_info["deleted"].append(filename)
                        elif y == "?":
                            status_info["untracked"].append(filename)
            else:
                # Porcelain format: XY filename
                if len(line) >= 3:
                    x, y = line[0], line[1]
                    filename = line[3:]
                    
                    if x == "?" and y == "?":
                        status_info["untracked"].append(filename)
                    elif x != " ":
                        status_info["staged"].append(filename)
                    elif y == "M":
                        status_info["modified"].append(filename)
                    elif y == "D":
                        status_info["deleted"].append(filename)

        return status_info


class GitDiffTool(Tool):
    """Git diff - View file changes."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Git diff tool.

        Args:
            base_path: Base path for git repository
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._definition = ToolDefinition(
            name="git_diff",
            description="View changes in files. Shows differences between working directory "
                        "and staging area, or between commits.",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="Specific file to diff. Default: all changed files",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="staged",
                    param_type="boolean",
                    description="Show staged changes (cached). Default: false",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="commit",
                    param_type="string",
                    description="Compare against specific commit. Default: HEAD",
                    required=False,
                    default=None
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "Show all unstaged changes"
                },
                {
                    "params": {"staged": True},
                    "description": "Show staged changes"
                },
                {
                    "params": {"file_path": "src/main/java/App.java"},
                    "description": "Show changes in specific file"
                }
            ],
            tags=["git", "diff", "changes"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute git diff."""
        file_path = kwargs.get("file_path")
        staged = kwargs.get("staged", False)
        commit = kwargs.get("commit")

        try:
            # Build command
            cmd = ["git", "diff"]
            
            if staged:
                cmd.append("--cached")
            
            if commit:
                cmd.append(commit)
            
            if file_path:
                cmd.append(file_path)

            # Execute git diff
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._base_path)
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Unknown error"
                return ToolResult(
                    success=False,
                    error=f"Git diff failed: {error_msg}"
                )

            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            # Parse diff stats
            stats = self._parse_diff_stats(output)

            logger.info(f"[GitDiffTool] Diff generated")

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "staged": staged,
                    "commit": commit,
                    "file_path": file_path,
                    "has_changes": len(output) > 0,
                    "stats": stats
                }
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git not found. Please ensure git is installed and in PATH."
            )
        except Exception as e:
            logger.exception(f"[GitDiffTool] Failed: {e}")
            return ToolResult(success=False, error=f"Git diff failed: {str(e)}")

    def _parse_diff_stats(self, diff_output: str) -> Dict[str, Any]:
        """Parse diff statistics."""
        stats = {
            "files_changed": 0,
            "insertions": 0,
            "deletions": 0
        }

        for line in diff_output.split("\n"):
            if line.startswith("diff --git"):
                stats["files_changed"] += 1
            elif line.startswith("+") and not line.startswith("+++"):
                stats["insertions"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                stats["deletions"] += 1

        return stats


class GitCommitTool(Tool):
    """Git commit - Commit changes."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Git commit tool.

        Args:
            base_path: Base path for git repository
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._definition = ToolDefinition(
            name="git_commit",
            description="Commit changes to the Git repository. "
                        "Creates a new commit with the specified message.",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="message",
                    param_type="string",
                    description="Commit message",
                    required=True
                ),
                create_tool_parameter(
                    name="add_all",
                    param_type="boolean",
                    description="Stage all changes before commit. Default: false",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="amend",
                    param_type="boolean",
                    description="Amend previous commit. Default: false",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {"message": "Add new feature"},
                    "description": "Commit staged changes"
                },
                {
                    "params": {"message": "Fix bug", "add_all": True},
                    "description": "Stage all and commit"
                }
            ],
            tags=["git", "commit", "version-control"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute git commit."""
        message = kwargs.get("message")
        add_all = kwargs.get("add_all", False)
        amend = kwargs.get("amend", False)

        if not message:
            return ToolResult(success=False, error="Commit message is required")

        try:
            # Stage all if requested
            if add_all:
                add_process = await asyncio.create_subprocess_exec(
                    "git", "add", "-A",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self._base_path)
                )
                await add_process.communicate()

            # Build commit command
            cmd = ["git", "commit", "-m", message]
            
            if amend:
                cmd.append("--amend")

            # Execute git commit
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._base_path)
            )

            stdout, stderr = await process.communicate()

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            if process.returncode != 0:
                return ToolResult(
                    success=False,
                    error=f"Git commit failed: {stderr_str}"
                )

            # Extract commit hash
            commit_hash = self._extract_commit_hash(stdout_str)

            logger.info(f"[GitCommitTool] Committed: {message[:50]}...")

            return ToolResult(
                success=True,
                output=stdout_str,
                metadata={
                    "message": message,
                    "commit_hash": commit_hash,
                    "amend": amend
                }
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git not found. Please ensure git is installed and in PATH."
            )
        except Exception as e:
            logger.exception(f"[GitCommitTool] Failed: {e}")
            return ToolResult(success=False, error=f"Git commit failed: {str(e)}")

    def _extract_commit_hash(self, output: str) -> Optional[str]:
        """Extract commit hash from git output."""
        for line in output.split("\n"):
            if line.startswith("[") and "]" in line:
                # Format: [branch hash] message
                parts = line.split("]")[0].split(" ")
                if len(parts) >= 2:
                    return parts[-1]
        return None


class GitBranchTool(Tool):
    """Git branch - Branch management."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Git branch tool.

        Args:
            base_path: Base path for git repository
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._definition = ToolDefinition(
            name="git_branch",
            description="Manage Git branches. List, create, or delete branches.",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="action",
                    param_type="string",
                    description="Action: list, create, delete, switch. Default: list",
                    required=False,
                    default="list"
                ),
                create_tool_parameter(
                    name="branch_name",
                    param_type="string",
                    description="Branch name (for create/delete/switch)",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="from_branch",
                    param_type="string",
                    description="Base branch for creation. Default: current",
                    required=False,
                    default=None
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "List all branches"
                },
                {
                    "params": {"action": "create", "branch_name": "feature-x"},
                    "description": "Create new branch"
                },
                {
                    "params": {"action": "switch", "branch_name": "main"},
                    "description": "Switch to branch"
                }
            ],
            tags=["git", "branch", "version-control"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute git branch command."""
        action = kwargs.get("action", "list")
        branch_name = kwargs.get("branch_name")
        from_branch = kwargs.get("from_branch")

        try:
            if action == "list":
                return await self._list_branches()
            elif action == "create":
                if not branch_name:
                    return ToolResult(success=False, error="Branch name required for create")
                return await self._create_branch(branch_name, from_branch)
            elif action == "delete":
                if not branch_name:
                    return ToolResult(success=False, error="Branch name required for delete")
                return await self._delete_branch(branch_name)
            elif action == "switch":
                if not branch_name:
                    return ToolResult(success=False, error="Branch name required for switch")
                return await self._switch_branch(branch_name)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git not found. Please ensure git is installed and in PATH."
            )
        except Exception as e:
            logger.exception(f"[GitBranchTool] Failed: {e}")
            return ToolResult(success=False, error=f"Git branch failed: {str(e)}")

    async def _list_branches(self) -> ToolResult:
        """List all branches."""
        process = await asyncio.create_subprocess_exec(
            "git", "branch", "-a",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._base_path)
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Unknown error"
            return ToolResult(success=False, error=f"Failed to list branches: {error_msg}")

        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        
        # Parse branches
        branches = []
        current_branch = None
        
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("*"):
                current_branch = line[1:].strip()
                branches.append(current_branch)
            else:
                branches.append(line)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "branches": branches,
                "current_branch": current_branch,
                "count": len(branches)
            }
        )

    async def _create_branch(self, branch_name: str, from_branch: Optional[str]) -> ToolResult:
        """Create a new branch."""
        cmd = ["git", "checkout", "-b", branch_name]
        
        if from_branch:
            cmd.append(from_branch)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._base_path)
        )

        stdout, stderr = await process.communicate()

        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        if process.returncode != 0:
            return ToolResult(success=False, error=f"Failed to create branch: {stderr_str}")

        logger.info(f"[GitBranchTool] Created branch: {branch_name}")

        return ToolResult(
            success=True,
            output=stdout_str,
            metadata={
                "action": "create",
                "branch_name": branch_name,
                "from_branch": from_branch
            }
        )

    async def _delete_branch(self, branch_name: str) -> ToolResult:
        """Delete a branch."""
        process = await asyncio.create_subprocess_exec(
            "git", "branch", "-d", branch_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._base_path)
        )

        stdout, stderr = await process.communicate()

        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        if process.returncode != 0:
            # Try force delete
            process = await asyncio.create_subprocess_exec(
                "git", "branch", "-D", branch_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._base_path)
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                return ToolResult(success=False, error=f"Failed to delete branch: {stderr_str}")

        logger.info(f"[GitBranchTool] Deleted branch: {branch_name}")

        return ToolResult(
            success=True,
            output=stdout_str,
            metadata={
                "action": "delete",
                "branch_name": branch_name
            }
        )

    async def _switch_branch(self, branch_name: str) -> ToolResult:
        """Switch to a branch."""
        process = await asyncio.create_subprocess_exec(
            "git", "checkout", branch_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._base_path)
        )

        stdout, stderr = await process.communicate()

        stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        if process.returncode != 0:
            return ToolResult(success=False, error=f"Failed to switch branch: {stderr_str}")

        logger.info(f"[GitBranchTool] Switched to branch: {branch_name}")

        return ToolResult(
            success=True,
            output=stdout_str,
            metadata={
                "action": "switch",
                "branch_name": branch_name
            }
        )


class GitLogTool(Tool):
    """Git log - View commit history."""

    def __init__(self, base_path: Optional[str] = None):
        """Initialize Git log tool.

        Args:
            base_path: Base path for git repository
        """
        super().__init__()
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._definition = ToolDefinition(
            name="git_log",
            description="View commit history. Shows recent commits with messages and hashes.",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="max_count",
                    param_type="integer",
                    description="Maximum number of commits to show. Default: 10",
                    required=False,
                    default=10
                ),
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="Show commits for specific file only",
                    required=False,
                    default=None
                ),
                create_tool_parameter(
                    name="oneline",
                    param_type="boolean",
                    description="Show one line per commit. Default: true",
                    required=False,
                    default=True
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "Show last 10 commits"
                },
                {
                    "params": {"max_count": 5, "file_path": "src/main.java"},
                    "description": "Show last 5 commits for specific file"
                }
            ],
            tags=["git", "log", "history"]
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, **kwargs) -> ToolResult:
        """Execute git log."""
        max_count = kwargs.get("max_count", 10)
        file_path = kwargs.get("file_path")
        oneline = kwargs.get("oneline", True)

        try:
            # Build command
            cmd = ["git", "log"]
            
            if oneline:
                cmd.append("--oneline")
            
            cmd.extend(["-n", str(max_count)])
            
            if file_path:
                cmd.append(file_path)

            # Execute git log
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._base_path)
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace") if stderr else "Unknown error"
                return ToolResult(
                    success=False,
                    error=f"Git log failed: {error_msg}"
                )

            output = stdout.decode("utf-8", errors="replace") if stdout else ""

            # Parse commits
            commits = self._parse_commits(output, oneline)

            logger.info(f"[GitLogTool] Retrieved {len(commits)} commits")

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "commits": commits,
                    "count": len(commits),
                    "file_path": file_path
                }
            )

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git not found. Please ensure git is installed and in PATH."
            )
        except Exception as e:
            logger.exception(f"[GitLogTool] Failed: {e}")
            return ToolResult(success=False, error=f"Git log failed: {str(e)}")

    def _parse_commits(self, output: str, oneline: bool) -> List[Dict[str, str]]:
        """Parse commit log output."""
        commits = []
        
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            
            if oneline:
                # Format: hash message
                parts = line.split(" ", 1)
                if len(parts) >= 2:
                    commits.append({
                        "hash": parts[0],
                        "message": parts[1]
                    })
            else:
                # Full format parsing would be more complex
                commits.append({"raw": line})

        return commits
