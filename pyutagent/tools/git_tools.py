"""Enhanced Git Tools using Tool Abstraction Layer.

This module provides comprehensive Git operation tools:
- GitStatusTool: Check repository status
- GitDiffTool: View file changes
- GitCommitTool: Commit changes
- GitBranchTool: Branch management
- GitAddTool: Stage files
- GitLogTool: View commit history
- GitCloneTool: Clone repositories
- GitPullTool: Pull changes
- GitPushTool: Push changes

Features:
- Async execution
- Detailed output parsing
- Error handling
- Security validation
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .core import (
    ToolBase,
    ToolCategory,
    ToolParameter,
    ToolResult,
    ToolContext,
)

logger = logging.getLogger(__name__)


@dataclass
class GitFileStatus:
    """Represents the status of a file in git."""

    path: str
    index_status: str  # Staged changes
    worktree_status: str  # Unstaged changes
    original_path: Optional[str] = None  # For renamed files

    @property
    def is_staged(self) -> bool:
        """Check if file has staged changes."""
        return self.index_status != " " and self.index_status != "?"

    @property
    def is_modified(self) -> bool:
        """Check if file has unstaged modifications."""
        return self.worktree_status == "M"

    @property
    def is_untracked(self) -> bool:
        """Check if file is untracked."""
        return self.index_status == "?" and self.worktree_status == "?"

    @property
    def is_deleted(self) -> bool:
        """Check if file is deleted."""
        return self.worktree_status == "D" or self.index_status == "D"


class GitExecutor:
    """Helper class for executing git commands."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize Git executor.

        Args:
            base_path: Base path for git repository
        """
        self.base_path = base_path or Path.cwd()

    async def execute(
        self,
        args: List[str],
        cwd: Optional[Path] = None,
        timeout: float = 60.0,
    ) -> Tuple[int, str, str]:
        """Execute a git command.

        Args:
            args: Git command arguments
            cwd: Working directory
            timeout: Command timeout

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        cmd = ["git"] + args
        working_dir = cwd or self.base_path

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir),
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            return process.returncode or 0, stdout_str, stderr_str

        except asyncio.TimeoutError:
            return -1, "", f"Command timed out after {timeout}s"
        except FileNotFoundError:
            return -1, "", "Git not found. Please ensure git is installed and in PATH."
        except Exception as e:
            return -1, "", str(e)

    def is_git_repo(self, path: Optional[Path] = None) -> bool:
        """Check if path is a git repository.

        Args:
            path: Path to check

        Returns:
            True if path is a git repository
        """
        check_path = path or self.base_path
        git_dir = check_path / ".git"
        return git_dir.exists() or (check_path / ".git").is_dir()


class GitStatusTool(ToolBase):
    """Tool for checking git repository status.

    Shows:
    - Modified files
    - Staged changes
    - Untracked files
    - Branch information
    """

    name = "git_status"
    description = (
        "Check the current status of the Git repository. "
        "Shows modified files, staged changes, untracked files, and branch information."
    )
    category = ToolCategory.GIT
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the git repository. Default: current directory",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="short",
            type="boolean",
            description="Show short format output. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="porcelain",
            type="boolean",
            description="Use porcelain format (machine-readable). Default: true",
            required=False,
            default=True,
        ),
    ]
    tags = ["git", "status", "repository", "vcs"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize GitStatusTool.

        Args:
            context: Execution context
            base_path: Base path for git repository
        """
        super().__init__(context)
        self.executor = GitExecutor(base_path)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute git status.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with status information
        """
        ctx = context or self._context
        path = params.get("path")
        short = params.get("short", False)
        porcelain = params.get("porcelain", True)

        # Resolve path
        if path:
            repo_path = Path(path)
        elif ctx:
            repo_path = ctx.project_path
        else:
            repo_path = self.executor.base_path

        # Check if git repo
        if not self.executor.is_git_repo(repo_path):
            return ToolResult.fail(
                error=f"Not a git repository: {repo_path}",
                code="NOT_A_REPO",
            )

        # Build command
        cmd = ["status"]
        if porcelain:
            cmd.append("--porcelain")
        if short:
            cmd.append("--short")
        cmd.append("-b")  # Show branch info

        # Execute
        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode != 0:
            return ToolResult.fail(
                error=f"Git status failed: {stderr}",
                code="GIT_ERROR",
            )

        # Parse status
        status_info = self._parse_status(stdout)
        status_info["path"] = str(repo_path)

        # Get branch info
        branch_info = await self._get_branch_info(repo_path)
        status_info.update(branch_info)

        # Format output
        output_lines = [f"On branch {status_info.get('branch', 'unknown')}"]

        if status_info["staged"]:
            output_lines.append("\nChanges to be committed:")
            for f in status_info["staged"]:
                output_lines.append(f"  {f['status']}: {f['path']}")

        if status_info["modified"]:
            output_lines.append("\nChanges not staged for commit:")
            for f in status_info["modified"]:
                output_lines.append(f"  modified: {f['path']}")

        if status_info["untracked"]:
            output_lines.append("\nUntracked files:")
            for f in status_info["untracked"]:
                output_lines.append(f"  {f['path']}")

        if not any([status_info["staged"], status_info["modified"], status_info["untracked"]]):
            output_lines.append("\nnothing to commit, working tree clean")

        return ToolResult.ok(
            output="\n".join(output_lines),
            data=status_info,
        )

    def _parse_status(self, output: str) -> Dict[str, Any]:
        """Parse git status output.

        Args:
            output: Git status output

        Returns:
            Parsed status information
        """
        result = {
            "staged": [],
            "modified": [],
            "untracked": [],
            "deleted": [],
            "renamed": [],
            "conflicted": [],
            "branch": None,
            "ahead": 0,
            "behind": 0,
        }

        if not output.strip():
            return result

        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            # Parse branch info line (## branch...upstream [ahead X, behind Y])
            if line.startswith("##"):
                result["branch"] = self._parse_branch_line(line)
                ahead, behind = self._parse_ahead_behind(line)
                result["ahead"] = ahead
                result["behind"] = behind
                continue

            # Parse file status (XY filename or XY filename -> new_filename)
            if len(line) >= 3:
                x, y = line[0], line[1]
                rest = line[3:]

                # Handle renamed files
                if " -> " in rest:
                    parts = rest.split(" -> ")
                    file_path = parts[1]
                    original_path = parts[0]
                else:
                    file_path = rest
                    original_path = None

                file_info = {
                    "path": file_path,
                    "original_path": original_path,
                    "index_status": x,
                    "worktree_status": y,
                }

                # Categorize file
                if x == "?" and y == "?":
                    result["untracked"].append(file_info)
                elif x == "U" or y == "U" or x == "A" and y == "A" or x == "D" and y == "D":
                    result["conflicted"].append(file_info)
                elif x != " ":
                    result["staged"].append(file_info)
                elif y == "M":
                    result["modified"].append(file_info)
                elif y == "D":
                    result["deleted"].append(file_info)

        return result

    def _parse_branch_line(self, line: str) -> str:
        """Parse branch name from status line."""
        # Format: ## branch...upstream or ## HEAD (no branch)
        match = re.match(r"## (.+?)(?:\.\.\.|$)", line)
        if match:
            branch = match.group(1)
            if branch.startswith("HEAD"):
                return "HEAD (detached)"
            return branch
        return "unknown"

    def _parse_ahead_behind(self, line: str) -> Tuple[int, int]:
        """Parse ahead/behind counts from branch line."""
        ahead = behind = 0

        # ahead X
        ahead_match = re.search(r"ahead (\d+)", line)
        if ahead_match:
            ahead = int(ahead_match.group(1))

        # behind Y
        behind_match = re.search(r"behind (\d+)", line)
        if behind_match:
            behind = int(behind_match.group(1))

        return ahead, behind

    async def _get_branch_info(self, repo_path: Path) -> Dict[str, Any]:
        """Get additional branch information."""
        result = {}

        # Get current branch
        returncode, stdout, _ = await self.executor.execute(
            ["rev-parse", "--abbrev-ref", "HEAD"],
            repo_path,
        )
        if returncode == 0:
            result["current_branch"] = stdout.strip()

        # Get last commit
        returncode, stdout, _ = await self.executor.execute(
            ["log", "-1", "--format=%H|%s|%ci"],
            repo_path,
        )
        if returncode == 0:
            parts = stdout.strip().split("|", 2)
            if len(parts) >= 3:
                result["last_commit"] = {
                    "hash": parts[0],
                    "message": parts[1],
                    "date": parts[2],
                }

        return result


class GitDiffTool(ToolBase):
    """Tool for viewing git diff output.

    Shows:
    - Unstaged changes
    - Staged changes
    - Changes between commits
    """

    name = "git_diff"
    description = (
        "View changes in files. Shows differences between working directory "
        "and staging area, or between commits."
    )
    category = ToolCategory.GIT
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="Specific file to diff. Default: all changed files",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="staged",
            type="boolean",
            description="Show staged changes (cached). Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="commit",
            type="string",
            description="Compare against specific commit. Default: HEAD",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="commit2",
            type="string",
            description="Second commit for comparison (creates range)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="stat",
            type="boolean",
            description="Show diffstat only. Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["git", "diff", "changes", "vcs"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize GitDiffTool.

        Args:
            context: Execution context
            base_path: Base path for git repository
        """
        super().__init__(context)
        self.executor = GitExecutor(base_path)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute git diff.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with diff output
        """
        ctx = context or self._context
        file_path = params.get("file_path")
        staged = params.get("staged", False)
        commit = params.get("commit")
        commit2 = params.get("commit2")
        stat = params.get("stat", False)

        # Resolve path
        if ctx:
            repo_path = ctx.project_path
        else:
            repo_path = self.executor.base_path

        # Check if git repo
        if not self.executor.is_git_repo(repo_path):
            return ToolResult.fail(
                error=f"Not a git repository: {repo_path}",
                code="NOT_A_REPO",
            )

        # Build command
        cmd = ["diff"]

        if stat:
            cmd.append("--stat")

        if staged:
            cmd.append("--cached")

        if commit and commit2:
            cmd.extend([commit, commit2])
        elif commit:
            cmd.append(commit)

        if file_path:
            cmd.append("--")
            cmd.append(file_path)

        # Execute
        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode not in [0, 1]:  # 1 means differences found
            return ToolResult.fail(
                error=f"Git diff failed: {stderr}",
                code="GIT_ERROR",
            )

        # Parse diff stats
        stats = self._parse_diff_stats(stdout) if not stat else {}

        return ToolResult.ok(
            output=stdout or "No differences found",
            data={
                "staged": staged,
                "commit": commit,
                "commit2": commit2,
                "file_path": file_path,
                "has_changes": len(stdout) > 0,
                "stats": stats,
            },
        )

    def _parse_diff_stats(self, diff_output: str) -> Dict[str, Any]:
        """Parse diff statistics.

        Args:
            diff_output: Git diff output

        Returns:
            Diff statistics
        """
        stats = {
            "files_changed": 0,
            "insertions": 0,
            "deletions": 0,
        }

        for line in diff_output.split("\n"):
            if line.startswith("diff --git"):
                stats["files_changed"] += 1
            elif line.startswith("+") and not line.startswith("+++"):
                stats["insertions"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                stats["deletions"] += 1

        return stats


class GitAddTool(ToolBase):
    """Tool for staging files in git.

    Supports:
    - Staging specific files
    - Staging all changes
    - Interactive staging
    """

    name = "git_add"
    description = (
        "Stage files for commit. Adds files to the staging area "
        "to be included in the next commit."
    )
    category = ToolCategory.GIT
    parameters = [
        ToolParameter(
            name="files",
            type="array",
            description="List of file paths to stage. Use ['.'] for all files",
            required=True,
        ),
        ToolParameter(
            name="all",
            type="boolean",
            description="Stage all changes including deletions. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="update",
            type="boolean",
            description="Stage only tracked files. Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["git", "add", "stage", "vcs"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize GitAddTool.

        Args:
            context: Execution context
            base_path: Base path for git repository
        """
        super().__init__(context)
        self.executor = GitExecutor(base_path)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute git add.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with add status
        """
        ctx = context or self._context
        files = params.get("files", [])
        add_all = params.get("all", False)
        update = params.get("update", False)

        # Resolve path
        if ctx:
            repo_path = ctx.project_path
        else:
            repo_path = self.executor.base_path

        # Check if git repo
        if not self.executor.is_git_repo(repo_path):
            return ToolResult.fail(
                error=f"Not a git repository: {repo_path}",
                code="NOT_A_REPO",
            )

        # Build command
        cmd = ["add"]

        if add_all:
            cmd.append("-A")
        elif update:
            cmd.append("-u")
        elif files:
            cmd.extend(files)
        else:
            return ToolResult.fail(
                error="No files specified for staging",
                code="NO_FILES",
            )

        # Execute
        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode != 0:
            return ToolResult.fail(
                error=f"Git add failed: {stderr}",
                code="GIT_ERROR",
            )

        return ToolResult.ok(
            output=f"Successfully staged: {', '.join(files) if files else 'all changes'}",
            data={
                "files": files,
                "all": add_all,
                "update": update,
            },
        )


class GitCommitTool(ToolBase):
    """Tool for committing changes in git.

    Features:
    - Commit with message
    - Stage and commit in one action
    - Amend previous commit
    """

    name = "git_commit"
    description = (
        "Commit changes to the Git repository. "
        "Creates a new commit with the specified message."
    )
    category = ToolCategory.GIT
    parameters = [
        ToolParameter(
            name="message",
            type="string",
            description="Commit message",
            required=True,
        ),
        ToolParameter(
            name="add_all",
            type="boolean",
            description="Stage all changes before commit. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="amend",
            type="boolean",
            description="Amend previous commit. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="no_edit",
            type="boolean",
            description="Use previous message when amending. Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["git", "commit", "vcs"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize GitCommitTool.

        Args:
            context: Execution context
            base_path: Base path for git repository
        """
        super().__init__(context)
        self.executor = GitExecutor(base_path)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute git commit.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with commit status
        """
        ctx = context or self._context
        message = params.get("message")
        add_all = params.get("add_all", False)
        amend = params.get("amend", False)
        no_edit = params.get("no_edit", False)

        if not message and not (amend and no_edit):
            return ToolResult.fail(
                error="Commit message is required",
                code="NO_MESSAGE",
            )

        # Resolve path
        if ctx:
            repo_path = ctx.project_path
        else:
            repo_path = self.executor.base_path

        # Check if git repo
        if not self.executor.is_git_repo(repo_path):
            return ToolResult.fail(
                error=f"Not a git repository: {repo_path}",
                code="NOT_A_REPO",
            )

        # Stage all if requested
        if add_all:
            await self.executor.execute(["add", "-A"], repo_path)

        # Build commit command
        cmd = ["commit"]

        if amend:
            cmd.append("--amend")
            if no_edit:
                cmd.append("--no-edit")
            elif message:
                cmd.extend(["-m", message])
        else:
            cmd.extend(["-m", message])

        # Execute
        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode != 0:
            return ToolResult.fail(
                error=f"Git commit failed: {stderr}",
                code="GIT_ERROR",
            )

        # Extract commit hash
        commit_hash = self._extract_commit_hash(stdout)

        return ToolResult.ok(
            output=stdout or f"Committed: {message[:50]}...",
            data={
                "message": message,
                "commit_hash": commit_hash,
                "amend": amend,
            },
        )

    def _extract_commit_hash(self, output: str) -> Optional[str]:
        """Extract commit hash from git output."""
        for line in output.split("\n"):
            if line.startswith("[") and "]" in line:
                # Format: [branch hash] message
                parts = line.split("]")[0].split(" ")
                if len(parts) >= 2:
                    return parts[-1]
        return None


class GitBranchTool(ToolBase):
    """Tool for git branch management.

    Supports:
    - List branches
    - Create branches
    - Delete branches
    - Switch branches
    """

    name = "git_branch"
    description = (
        "Manage Git branches. List, create, delete, or switch branches."
    )
    category = ToolCategory.GIT
    parameters = [
        ToolParameter(
            name="action",
            type="string",
            description="Action: list, create, delete, switch. Default: list",
            required=False,
            default="list",
            enum=["list", "create", "delete", "switch"],
        ),
        ToolParameter(
            name="branch_name",
            type="string",
            description="Branch name (for create/delete/switch)",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="from_branch",
            type="string",
            description="Base branch for creation. Default: current",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="force",
            type="boolean",
            description="Force delete or switch. Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["git", "branch", "vcs"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize GitBranchTool.

        Args:
            context: Execution context
            base_path: Base path for git repository
        """
        super().__init__(context)
        self.executor = GitExecutor(base_path)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute git branch command.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with branch operation result
        """
        ctx = context or self._context
        action = params.get("action", "list")
        branch_name = params.get("branch_name")
        from_branch = params.get("from_branch")
        force = params.get("force", False)

        # Resolve path
        if ctx:
            repo_path = ctx.project_path
        else:
            repo_path = self.executor.base_path

        # Check if git repo
        if not self.executor.is_git_repo(repo_path):
            return ToolResult.fail(
                error=f"Not a git repository: {repo_path}",
                code="NOT_A_REPO",
            )

        # Route to specific action
        if action == "list":
            return await self._list_branches(repo_path)
        elif action == "create":
            if not branch_name:
                return ToolResult.fail(
                    error="Branch name required for create",
                    code="NO_BRANCH_NAME",
                )
            return await self._create_branch(repo_path, branch_name, from_branch)
        elif action == "delete":
            if not branch_name:
                return ToolResult.fail(
                    error="Branch name required for delete",
                    code="NO_BRANCH_NAME",
                )
            return await self._delete_branch(repo_path, branch_name, force)
        elif action == "switch":
            if not branch_name:
                return ToolResult.fail(
                    error="Branch name required for switch",
                    code="NO_BRANCH_NAME",
                )
            return await self._switch_branch(repo_path, branch_name, force)
        else:
            return ToolResult.fail(
                error=f"Unknown action: {action}",
                code="UNKNOWN_ACTION",
            )

    async def _list_branches(self, repo_path: Path) -> ToolResult:
        """List all branches."""
        returncode, stdout, stderr = await self.executor.execute(
            ["branch", "-a", "-vv"],
            repo_path,
        )

        if returncode != 0:
            return ToolResult.fail(
                error=f"Failed to list branches: {stderr}",
                code="GIT_ERROR",
            )

        # Parse branches
        branches = []
        current_branch = None

        for line in stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("*"):
                current_branch = line[1:].strip().split()[0]
                branches.append({
                    "name": current_branch,
                    "current": True,
                })
            else:
                branch_name = line.split()[0]
                branches.append({
                    "name": branch_name,
                    "current": False,
                })

        return ToolResult.ok(
            output=stdout,
            data={
                "branches": branches,
                "current_branch": current_branch,
                "count": len(branches),
            },
        )

    async def _create_branch(
        self,
        repo_path: Path,
        branch_name: str,
        from_branch: Optional[str],
    ) -> ToolResult:
        """Create a new branch."""
        cmd = ["checkout", "-b", branch_name]

        if from_branch:
            cmd.append(from_branch)

        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode != 0:
            return ToolResult.fail(
                error=f"Failed to create branch: {stderr}",
                code="GIT_ERROR",
            )

        return ToolResult.ok(
            output=f"Created and switched to branch: {branch_name}",
            data={
                "action": "create",
                "branch_name": branch_name,
                "from_branch": from_branch,
            },
        )

    async def _delete_branch(
        self,
        repo_path: Path,
        branch_name: str,
        force: bool,
    ) -> ToolResult:
        """Delete a branch."""
        flag = "-D" if force else "-d"

        returncode, stdout, stderr = await self.executor.execute(
            ["branch", flag, branch_name],
            repo_path,
        )

        if returncode != 0:
            return ToolResult.fail(
                error=f"Failed to delete branch: {stderr}",
                code="GIT_ERROR",
            )

        return ToolResult.ok(
            output=f"Deleted branch: {branch_name}",
            data={
                "action": "delete",
                "branch_name": branch_name,
                "force": force,
            },
        )

    async def _switch_branch(
        self,
        repo_path: Path,
        branch_name: str,
        force: bool,
    ) -> ToolResult:
        """Switch to a branch."""
        cmd = ["checkout"]
        if force:
            cmd.append("-f")
        cmd.append(branch_name)

        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode != 0:
            return ToolResult.fail(
                error=f"Failed to switch branch: {stderr}",
                code="GIT_ERROR",
            )

        return ToolResult.ok(
            output=f"Switched to branch: {branch_name}",
            data={
                "action": "switch",
                "branch_name": branch_name,
            },
        )


class GitLogTool(ToolBase):
    """Tool for viewing git commit history.

    Features:
    - View commit log
    - Filter by file
    - Custom formatting
    """

    name = "git_log"
    description = (
        "View commit history. Shows recent commits with messages and hashes."
    )
    category = ToolCategory.GIT
    parameters = [
        ToolParameter(
            name="max_count",
            type="integer",
            description="Maximum number of commits to show. Default: 10",
            required=False,
            default=10,
        ),
        ToolParameter(
            name="file_path",
            type="string",
            description="Show commits for specific file only",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="oneline",
            type="boolean",
            description="Show one line per commit. Default: true",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="author",
            type="string",
            description="Filter by author",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="since",
            type="string",
            description="Show commits since date (e.g., '1 week ago')",
            required=False,
            default=None,
        ),
    ]
    tags = ["git", "log", "history", "vcs"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize GitLogTool.

        Args:
            context: Execution context
            base_path: Base path for git repository
        """
        super().__init__(context)
        self.executor = GitExecutor(base_path)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute git log.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with commit history
        """
        ctx = context or self._context
        max_count = params.get("max_count", 10)
        file_path = params.get("file_path")
        oneline = params.get("oneline", True)
        author = params.get("author")
        since = params.get("since")

        # Resolve path
        if ctx:
            repo_path = ctx.project_path
        else:
            repo_path = self.executor.base_path

        # Check if git repo
        if not self.executor.is_git_repo(repo_path):
            return ToolResult.fail(
                error=f"Not a git repository: {repo_path}",
                code="NOT_A_REPO",
            )

        # Build command
        cmd = ["log"]

        if oneline:
            cmd.append("--oneline")

        cmd.extend(["-n", str(max_count)])

        if author:
            cmd.extend(["--author", author])

        if since:
            cmd.extend(["--since", since])

        # Use custom format for better parsing
        if not oneline:
            cmd.extend(["--format=%H|%an|%ae|%ad|%s"])

        if file_path:
            cmd.append("--")
            cmd.append(file_path)

        # Execute
        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode != 0:
            return ToolResult.fail(
                error=f"Git log failed: {stderr}",
                code="GIT_ERROR",
            )

        # Parse commits
        commits = self._parse_commits(stdout, oneline)

        return ToolResult.ok(
            output=stdout or "No commits found",
            data={
                "commits": commits,
                "count": len(commits),
                "file_path": file_path,
            },
        )

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
                        "short_hash": parts[0][:7],
                        "message": parts[1],
                    })
            else:
                # Format: hash|author|email|date|subject
                parts = line.split("|", 4)
                if len(parts) >= 5:
                    commits.append({
                        "hash": parts[0],
                        "short_hash": parts[0][:7],
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "message": parts[4],
                    })

        return commits


class GitPullTool(ToolBase):
    """Tool for pulling changes from remote.

    Features:
    - Pull from default remote
    - Pull specific branch
    - Rebase option
    """

    name = "git_pull"
    description = (
        "Pull changes from remote repository. "
        "Fetches and merges changes from the remote."
    )
    category = ToolCategory.GIT
    parameters = [
        ToolParameter(
            name="remote",
            type="string",
            description="Remote name. Default: origin",
            required=False,
            default="origin",
        ),
        ToolParameter(
            name="branch",
            type="string",
            description="Branch to pull. Default: current branch",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="rebase",
            type="boolean",
            description="Rebase instead of merge. Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["git", "pull", "remote", "vcs"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize GitPullTool.

        Args:
            context: Execution context
            base_path: Base path for git repository
        """
        super().__init__(context)
        self.executor = GitExecutor(base_path)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute git pull.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with pull status
        """
        ctx = context or self._context
        remote = params.get("remote", "origin")
        branch = params.get("branch")
        rebase = params.get("rebase", False)

        # Resolve path
        if ctx:
            repo_path = ctx.project_path
        else:
            repo_path = self.executor.base_path

        # Check if git repo
        if not self.executor.is_git_repo(repo_path):
            return ToolResult.fail(
                error=f"Not a git repository: {repo_path}",
                code="NOT_A_REPO",
            )

        # Build command
        cmd = ["pull"]

        if rebase:
            cmd.append("--rebase")

        cmd.append(remote)

        if branch:
            cmd.append(branch)

        # Execute
        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode != 0:
            return ToolResult.fail(
                error=f"Git pull failed: {stderr}",
                code="GIT_ERROR",
            )

        return ToolResult.ok(
            output=stdout or "Pull completed successfully",
            data={
                "remote": remote,
                "branch": branch,
                "rebase": rebase,
            },
        )


class GitPushTool(ToolBase):
    """Tool for pushing changes to remote.

    Features:
    - Push to default remote
    - Push specific branch
    - Force push option
    """

    name = "git_push"
    description = (
        "Push changes to remote repository. "
        "Uploads local commits to the remote."
    )
    category = ToolCategory.GIT
    parameters = [
        ToolParameter(
            name="remote",
            type="string",
            description="Remote name. Default: origin",
            required=False,
            default="origin",
        ),
        ToolParameter(
            name="branch",
            type="string",
            description="Branch to push. Default: current branch",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="force",
            type="boolean",
            description="Force push. Default: false",
            required=False,
            default=False,
        ),
        ToolParameter(
            name="set_upstream",
            type="boolean",
            description="Set upstream tracking. Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["git", "push", "remote", "vcs"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        base_path: Optional[Path] = None,
    ):
        """Initialize GitPushTool.

        Args:
            context: Execution context
            base_path: Base path for git repository
        """
        super().__init__(context)
        self.executor = GitExecutor(base_path)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute git push.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with push status
        """
        ctx = context or self._context
        remote = params.get("remote", "origin")
        branch = params.get("branch")
        force = params.get("force", False)
        set_upstream = params.get("set_upstream", False)

        # Resolve path
        if ctx:
            repo_path = ctx.project_path
        else:
            repo_path = self.executor.base_path

        # Check if git repo
        if not self.executor.is_git_repo(repo_path):
            return ToolResult.fail(
                error=f"Not a git repository: {repo_path}",
                code="NOT_A_REPO",
            )

        # Build command
        cmd = ["push"]

        if force:
            cmd.append("--force")

        if set_upstream:
            cmd.append("--set-upstream")

        cmd.append(remote)

        if branch:
            cmd.append(branch)

        # Execute
        returncode, stdout, stderr = await self.executor.execute(cmd, repo_path)

        if returncode != 0:
            return ToolResult.fail(
                error=f"Git push failed: {stderr}",
                code="GIT_ERROR",
            )

        return ToolResult.ok(
            output=stdout or stderr or "Push completed successfully",
            data={
                "remote": remote,
                "branch": branch,
                "force": force,
            },
        )


def get_all_git_tools(
    context: Optional[ToolContext] = None,
    base_path: Optional[Path] = None,
) -> List[ToolBase]:
    """Get all Git tools.

    Args:
        context: Execution context
        base_path: Base path for git repository

    Returns:
        List of Git tool instances
    """
    return [
        GitStatusTool(context, base_path),
        GitDiffTool(context, base_path),
        GitAddTool(context, base_path),
        GitCommitTool(context, base_path),
        GitBranchTool(context, base_path),
        GitLogTool(context, base_path),
        GitPullTool(context, base_path),
        GitPushTool(context, base_path),
    ]
