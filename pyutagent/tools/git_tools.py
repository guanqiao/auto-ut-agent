"""Git tools for version control operations.

This module provides Git-related tools:
- GitStatusTool: Check repository status
- GitDiffTool: View file changes
- GitCommitTool: Commit changes
- GitBranchTool: Branch management
- GitLogTool: View commit history
- GitStackTool: Manage git stash
"""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tool import (
    Tool,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    create_tool_parameter,
)
from .standard_tools import BaseFileTool

logger = logging.getLogger(__name__)


class GitTool(BaseFileTool):
    """Base class for Git tools."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._git_path = self._find_git()
    
    def _find_git(self) -> str:
        """Find git executable."""
        return "git"
    
    async def _run_git_command(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        timeout: int = 30
    ) -> ToolResult:
        """Run a git command.
        
        Args:
            args: Git command arguments
            cwd: Working directory
            timeout: Command timeout
        
        Returns:
            ToolResult with command output
        """
        cwd = cwd or str(self._base_path) if self._base_path else None
        
        if not cwd or not Path(cwd).exists():
            return ToolResult(
                success=False,
                error=f"Working directory not found: {cwd}"
            )
        
        try:
            cmd = [self._git_path] + args
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
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
                    error=f"Git command timed out after {timeout} seconds"
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""
            
            if process.returncode == 0:
                return ToolResult(
                    success=True,
                    output=stdout_str,
                    metadata={
                        "command": " ".join(cmd),
                        "exit_code": process.returncode
                    }
                )
            else:
                return ToolResult(
                    success=False,
                    output=stdout_str,
                    error=stderr_str or f"Git command failed with code {process.returncode}",
                    metadata={
                        "command": " ".join(cmd),
                        "exit_code": process.returncode
                    }
                )
        
        except FileNotFoundError:
            return ToolResult(
                success=False,
                error="Git executable not found"
            )
        except Exception as e:
            logger.exception(f"[GitTool] Command failed: {e}")
            return ToolResult(
                success=False,
                error=f"Git command failed: {str(e)}"
            )


class GitStatusTool(GitTool):
    """Tool for checking git repository status."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="git_status",
            description="查看Git仓库的当前状态，包括已修改、已暂存、未跟踪的文件。",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="short",
                    param_type="boolean",
                    description="使用简洁格式输出",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="porcelain",
                    param_type="boolean",
                    description="使用机器可读的简洁格式",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "查看完整git状态"
                },
                {
                    "params": {"short": True},
                    "description": "使用简洁格式查看状态"
                }
            ],
            tags=["git", "version_control", "status"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute git status."""
        short = kwargs.get("short", False)
        porcelain = kwargs.get("porcelain", False)
        
        args = ["status"]
        
        if porcelain:
            args.append("--porcelain")
        elif short:
            args.append("--short")
        
        return await self._run_git_command(args)


class GitDiffTool(GitTool):
    """Tool for viewing file changes."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="git_diff",
            description="查看文件的更改内容，可以是工作区与索引、索引与HEAD、或两个提交之间的差异。",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="要查看差异的文件路径",
                    required=False
                ),
                create_tool_parameter(
                    name="staged",
                    param_type="boolean",
                    description="仅显示已暂存的更改",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="cached",
                    param_type="boolean",
                    description="仅显示已暂存的更改（staged的别名）",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="commit",
                    param_type="string",
                    description="比较的提交ID（与HEAD或其他提交）",
                    required=False
                ),
                create_tool_parameter(
                    name="stat",
                    param_type="boolean",
                    description="显示统计信息而非完整差异",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="no_color",
                    param_type="boolean",
                    description="不使用颜色输出",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "查看所有未暂存的更改"
                },
                {
                    "params": {"staged": True},
                    "description": "查看已暂存的更改"
                },
                {
                    "params": {"file_path": "src/main/java/App.java"},
                    "description": "查看特定文件的更改"
                }
            ],
            tags=["git", "version_control", "diff", "changes"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute git diff."""
        file_path = kwargs.get("file_path")
        staged = kwargs.get("staged", False)
        cached = kwargs.get("cached", False)
        commit = kwargs.get("commit")
        stat = kwargs.get("stat", False)
        no_color = kwargs.get("no_color", False)
        
        args = ["diff"]
        
        if cached or staged:
            args.append("--cached")
        if stat:
            args.append("--stat")
        if no_color:
            args.append("--no-color")
        
        if commit:
            args.append(commit)
        
        if file_path:
            args.append("--")
            args.append(file_path)
        
        return await self._run_git_command(args)


class GitCommitTool(GitTool):
    """Tool for committing changes."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="git_commit",
            description="将暂存的更改提交到Git仓库。",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="message",
                    param_type="string",
                    description="提交信息",
                    required=True
                ),
                create_tool_parameter(
                    name="add_all",
                    param_type="boolean",
                    description="暂存所有修改的文件",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="amend",
                    param_type="boolean",
                    description="修改最后一次提交",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="no_edit",
                    param_type="boolean",
                    description="不打开编辑器",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {"message": "Add new feature"},
                    "description": "提交暂存的更改"
                },
                {
                    "params": {"message": "Update", "add_all": True},
                    "description": "暂存所有修改并提交"
                }
            ],
            tags=["git", "version_control", "commit"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute git commit."""
        message = kwargs.get("message")
        add_all = kwargs.get("add_all", False)
        amend = kwargs.get("amend", False)
        no_edit = kwargs.get("no_edit", False)
        
        if not message:
            return ToolResult(
                success=False,
                error="Commit message is required"
            )
        
        if add_all:
            add_result = await self._run_git_command(["add", "-A"])
            if not add_result.success:
                return add_result
        
        args = ["commit"]
        
        if no_edit:
            args.append("--no-edit")
        
        if amend:
            args.append("--amend")
        
        args.extend(["-m", message])
        
        return await self._run_git_command(args)


class GitBranchTool(GitTool):
    """Tool for branch management."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="git_branch",
            description="列出、创建或删除Git分支。",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="action",
                    param_type="string",
                    description="操作类型: list(列出), create(创建), delete(删除), rename(重命名)",
                    required=False,
                    default="list"
                ),
                create_tool_parameter(
                    name="branch_name",
                    param_type="string",
                    description="分支名称（创建/删除/重命名时使用）",
                    required=False
                ),
                create_tool_parameter(
                    name="start_point",
                    param_type="string",
                    description="新分支的起始点（创建时使用）",
                    required=False
                ),
                create_tool_parameter(
                    name="force",
                    param_type="boolean",
                    description="强制执行操作",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="all",
                    param_type="boolean",
                    description="列出所有分支（包括远程）",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "列出所有本地分支"
                },
                {
                    "params": {"action": "create", "branch_name": "feature/new-feature"},
                    "description": "创建新分支"
                },
                {
                    "params": {"action": "delete", "branch_name": "old-feature", "force": True},
                    "description": "强制删除分支"
                }
            ],
            tags=["git", "version_control", "branch"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute git branch."""
        action = kwargs.get("action", "list")
        branch_name = kwargs.get("branch_name")
        start_point = kwargs.get("start_point")
        force = kwargs.get("force", False)
        all_branches = kwargs.get("all", False)
        
        args = ["branch"]
        
        if all_branches:
            args.append("-a")
        
        if action == "list":
            if all_branches:
                args = ["branch", "-a"]
            return await self._run_git_command(args)
        
        elif action == "create":
            if not branch_name:
                return ToolResult(
                    success=False,
                    error="branch_name is required for create action"
                )
            
            args.append(branch_name)
            
            if start_point:
                args.append(start_point)
            
            return await self._run_git_command(args)
        
        elif action == "delete":
            if not branch_name:
                return ToolResult(
                    success=False,
                    error="branch_name is required for delete action"
                )
            
            if force:
                args.append("-D")
            else:
                args.append("-d")
            
            args.append(branch_name)
            
            return await self._run_git_command(args)
        
        elif action == "rename":
            if not branch_name:
                return ToolResult(
                    success=False,
                    error="branch_name is required for rename action"
                )
            
            new_name = kwargs.get("new_name")
            if not new_name:
                return ToolResult(
                    success=False,
                    error="new_name is required for rename action"
                )
            
            args.extend(["-m", branch_name, new_name])
            
            return await self._run_git_command(args)
        
        else:
            return ToolResult(
                success=False,
                error=f"Unknown action: {action}"
            )


class GitLogTool(GitTool):
    """Tool for viewing commit history."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="git_log",
            description="查看Git提交历史。",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="max_count",
                    param_type="integer",
                    description="限制显示的提交数量",
                    required=False,
                    default=10
                ),
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="查看特定文件的提交历史",
                    required=False
                ),
                create_tool_parameter(
                    name="oneline",
                    param_type="boolean",
                    description="使用简洁格式显示",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="author",
                    param_type="string",
                    description="按作者过滤",
                    required=False
                ),
                create_tool_parameter(
                    name="since",
                    param_type="string",
                    description="显示自指定日期以来的提交",
                    required=False
                ),
                create_tool_parameter(
                    name="graph",
                    param_type="boolean",
                    description="显示分支合并图",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "查看最近10条提交"
                },
                {
                    "params": {"max_count": 5, "oneline": True},
                    "description": "查看最近5条简洁格式提交"
                }
            ],
            tags=["git", "version_control", "log", "history"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute git log."""
        max_count = kwargs.get("max_count", 10)
        file_path = kwargs.get("file_path")
        oneline = kwargs.get("oneline", False)
        author = kwargs.get("author")
        since = kwargs.get("since")
        graph = kwargs.get("graph", False)
        
        args = ["log"]
        
        args.append(f"--max-count={max_count}")
        
        if oneline:
            args.append("--oneline")
        
        if graph:
            args.append("--graph")
        
        if author:
            args.extend(["--author", author])
        
        if since:
            args.extend(["--since", since])
        
        if file_path:
            args.append("--")
            args.append(file_path)
        
        return await self._run_git_command(args)


class GitAddTool(GitTool):
    """Tool for staging files."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="git_add",
            description="将文件添加到Git暂存区。",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="file_path",
                    param_type="string",
                    description="要暂存的文件路径（.表示所有文件）",
                    required=False,
                    default="."
                ),
                create_tool_parameter(
                    name="patch",
                    param_type="boolean",
                    description="交互式暂存",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="all",
                    param_type="boolean",
                    description="暂存所有修改（包括未跟踪）",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {"file_path": "."},
                    "description": "暂存所有修改"
                },
                {
                    "params": {"file_path": "src/main/java/App.java"},
                    "description": "暂存特定文件"
                }
            ],
            tags=["git", "version_control", "add", "stage"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute git add."""
        file_path = kwargs.get("file_path", ".")
        patch = kwargs.get("patch", False)
        all_files = kwargs.get("all", False)
        
        args = ["add"]
        
        if patch:
            args.append("-p")
        elif all_files:
            args.append("-A")
        
        args.append(file_path)
        
        return await self._run_git_command(args)


class GitPushTool(GitTool):
    """Tool for pushing to remote."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="git_push",
            description="将本地分支推送到远程仓库。",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="remote",
                    param_type="string",
                    description="远程仓库名称",
                    required=False,
                    default="origin"
                ),
                create_tool_parameter(
                    name="branch",
                    param_type="string",
                    description="要推送的分支名",
                    required=False
                ),
                create_tool_parameter(
                    name="force",
                    param_type="boolean",
                    description="强制推送",
                    required=False,
                    default=False
                ),
                create_tool_parameter(
                    name="set_upstream",
                    param_type="boolean",
                    description="设置上游分支",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "推送到默认远程"
                },
                {
                    "params": {"remote": "origin", "branch": "main", "set_upstream": True},
                    "description": "推送并设置上游"
                }
            ],
            tags=["git", "version_control", "push", "remote"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute git push."""
        remote = kwargs.get("remote", "origin")
        branch = kwargs.get("branch")
        force = kwargs.get("force", False)
        set_upstream = kwargs.get("set_upstream", False)
        
        args = ["push"]
        
        if force:
            args.append("--force")
        
        if set_upstream:
            args.append("-u")
        
        if remote:
            args.append(remote)
        
        if branch:
            args.append(branch)
        
        return await self._run_git_command(args)


class GitPullTool(GitTool):
    """Tool for pulling from remote."""
    
    def __init__(self, base_path: Optional[str] = None):
        super().__init__(base_path)
        self._definition = ToolDefinition(
            name="git_pull",
            description="从远程仓库拉取并合并更改。",
            category=ToolCategory.COMMAND,
            parameters=[
                create_tool_parameter(
                    name="remote",
                    param_type="string",
                    description="远程仓库名称",
                    required=False,
                    default="origin"
                ),
                create_tool_parameter(
                    name="branch",
                    param_type="string",
                    description="要拉取的分支名",
                    required=False
                ),
                create_tool_parameter(
                    name="rebase",
                    param_type="boolean",
                    description="使用rebase代替merge",
                    required=False,
                    default=False
                )
            ],
            examples=[
                {
                    "params": {},
                    "description": "从默认远程拉取"
                },
                {
                    "params": {"remote": "origin", "branch": "main"},
                    "description": "拉取特定分支"
                }
            ],
            tags=["git", "version_control", "pull", "remote"]
        )
    
    @property
    def definition(self) -> ToolDefinition:
        return self._definition
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute git pull."""
        remote = kwargs.get("remote", "origin")
        branch = kwargs.get("branch")
        rebase = kwargs.get("rebase", False)
        
        args = ["pull"]
        
        if rebase:
            args.append("--rebase")
        
        if remote:
            args.append(remote)
        
        if branch:
            args.append(branch)
        
        return await self._run_git_command(args)


def get_all_git_tools(base_path: Optional[str] = None) -> List[Tool]:
    """Get all Git tools.
    
    Args:
        base_path: Base path for git operations
    
    Returns:
        List of Git tool instances
    """
    return [
        GitStatusTool(base_path),
        GitDiffTool(base_path),
        GitCommitTool(base_path),
        GitBranchTool(base_path),
        GitLogTool(base_path),
        GitAddTool(base_path),
        GitPushTool(base_path),
        GitPullTool(base_path),
    ]
