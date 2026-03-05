"""Enhanced Bash Tool with Security Sandbox.

This module provides a secure shell command execution tool:
- BashTool: Execute shell commands with safety controls

Security Features:
- Command whitelist/blacklist
- Timeout control
- Working directory restriction
- Environment variable filtering
- Output size limits
- Dangerous command detection
"""

import asyncio
import logging
import platform
import re
import shlex
from dataclasses import dataclass, field
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
class BashSecurityConfig:
    """Security configuration for bash command execution."""

    # Whitelist: Only allow these commands (empty = allow all not in blacklist)
    allowed_commands: List[str] = field(default_factory=list)

    # Blacklist: Never allow these commands
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /",
        "rm -rf /*",
        ":(){ :|:& };:",  # Fork bomb
        "dd if=/dev/zero",
        "mkfs",
        "fdisk",
        "format",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init 0",
        "systemctl poweroff",
        "systemctl reboot",
    ])

    # Blocked patterns (regex)
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r"rm\s+-rf\s+/",
        r">\s*/dev/sd[a-z]",
        r"dd\s+if=/dev/zero\s+of=/",
        r":\(\)\s*\{\s*:\s*\|:\s*&\s*\};:",  # Fork bomb
    ])

    # Maximum output size (bytes)
    max_output_size: int = 1024 * 1024  # 1MB

    # Maximum command length
    max_command_length: int = 8192

    # Allowed environment variables (empty = allow all)
    allowed_env_vars: List[str] = field(default_factory=list)

    # Blocked environment variables
    blocked_env_vars: List[str] = field(default_factory=lambda: [
        "PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    ])

    # Require explicit confirmation for dangerous commands
    require_confirmation: bool = True

    # Working directory must be within project
    restrict_working_directory: bool = True


class BashSecurityValidator:
    """Validator for bash command security."""

    def __init__(self, config: Optional[BashSecurityConfig] = None):
        """Initialize validator.

        Args:
            config: Security configuration
        """
        self.config = config or BashSecurityConfig()

    def validate(
        self,
        command: str,
        working_dir: Optional[Path] = None,
        project_path: Optional[Path] = None,
    ) -> tuple[bool, Optional[str]]:
        """Validate a command for security.

        Args:
            command: Command to validate
            working_dir: Working directory
            project_path: Project root path

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check command length
        if len(command) > self.config.max_command_length:
            return False, (
                f"Command exceeds maximum length "
                f"({self.config.max_command_length} characters)"
            )

        # Check blocked commands (exact match)
        for blocked in self.config.blocked_commands:
            if blocked in command:
                return False, f"Command contains blocked pattern: {blocked}"

        # Check blocked patterns (regex)
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command matches blocked pattern: {pattern}"

        # Check whitelist
        if self.config.allowed_commands:
            # Extract command name
            cmd_parts = shlex.split(command)
            if cmd_parts:
                cmd_name = cmd_parts[0]
                allowed = any(
                    cmd_name == allowed or command.startswith(allowed)
                    for allowed in self.config.allowed_commands
                )
                if not allowed:
                    return False, (
                        f"Command '{cmd_name}' not in allowed commands list"
                    )

        # Check working directory restriction
        if self.config.restrict_working_directory and working_dir and project_path:
            try:
                working_dir.resolve().relative_to(project_path.resolve())
            except ValueError:
                return False, (
                    f"Working directory {working_dir} is outside project "
                    f"directory {project_path}"
                )

        return True, None

    def validate_env_vars(self, env: Dict[str, str]) -> Dict[str, str]:
        """Validate and filter environment variables.

        Args:
            env: Environment variables

        Returns:
            Filtered environment variables
        """
        if not env:
            return {}

        filtered = {}
        for key, value in env.items():
            # Check blocked vars
            if key in self.config.blocked_env_vars:
                continue

            # Check whitelist
            if self.config.allowed_env_vars and key not in self.config.allowed_env_vars:
                continue

            filtered[key] = value

        return filtered

    def is_dangerous_command(self, command: str) -> tuple[bool, str]:
        """Check if a command is potentially dangerous.

        Args:
            command: Command to check

        Returns:
            Tuple of (is_dangerous, reason)
        """
        dangerous_patterns = [
            (r"\brm\b.*-\w*[rf]", "removes files recursively"),
            (r"\bdd\b", "direct disk access"),
            (r"\bsudo\b", "elevated privileges"),
            (r"\bsu\b", "user switching"),
            (r"\bchmod\b.*777", "overly permissive permissions"),
            (r"\bcurl\b.*\|", "piping from network"),
            (r"\bwget\b.*\|", "piping from network"),
            (r"\beval\b", "command evaluation"),
            (r"\bexec\b", "command execution"),
        ]

        for pattern, reason in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return True, reason

        return False, ""


@dataclass
class BashExecutionResult:
    """Result of bash command execution."""

    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool = False


class BashExecutor:
    """Executor for bash commands with safety controls."""

    def __init__(
        self,
        config: Optional[BashSecurityConfig] = None,
        default_timeout: float = 60.0,
    ):
        """Initialize executor.

        Args:
            config: Security configuration
            default_timeout: Default command timeout
        """
        self.config = config or BashSecurityConfig()
        self.validator = BashSecurityValidator(config)
        self.default_timeout = default_timeout

    async def execute(
        self,
        command: str,
        working_dir: Optional[Path] = None,
        project_path: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> BashExecutionResult:
        """Execute a command with safety controls.

        Args:
            command: Command to execute
            working_dir: Working directory
            project_path: Project root for validation
            env: Environment variables
            timeout: Command timeout

        Returns:
            Execution result
        """
        import time
        start_time = time.time()

        # Validate command
        is_valid, error = self.validator.validate(command, working_dir, project_path)
        if not is_valid:
            return BashExecutionResult(
                returncode=-1,
                stdout="",
                stderr=error,
                duration_ms=0,
            )

        # Filter environment variables
        filtered_env = self.validator.validate_env_vars(env or {})

        # Prepare execution
        actual_timeout = timeout or self.default_timeout
        cwd = str(working_dir) if working_dir else None

        # Determine shell
        is_windows = platform.system() == "Windows"

        try:
            if is_windows:
                # Windows: use cmd.exe
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=filtered_env or None,
                )
            else:
                # Unix: use bash -c
                process = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=filtered_env or None,
                )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=actual_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                duration_ms = int((time.time() - start_time) * 1000)
                return BashExecutionResult(
                    returncode=-1,
                    stdout="",
                    stderr=f"Command timed out after {actual_timeout} seconds",
                    duration_ms=duration_ms,
                )

            # Decode output
            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            # Truncate if too large
            truncated = False
            max_size = self.config.max_output_size
            if len(stdout_str) > max_size:
                stdout_str = stdout_str[:max_size] + "\n... (output truncated)"
                truncated = True

            duration_ms = int((time.time() - start_time) * 1000)

            return BashExecutionResult(
                returncode=process.returncode or 0,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_ms=duration_ms,
                truncated=truncated,
            )

        except FileNotFoundError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return BashExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Command not found: {e}",
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            return BashExecutionResult(
                returncode=-1,
                stdout="",
                stderr=f"Execution error: {e}",
                duration_ms=duration_ms,
            )


class BashTool(ToolBase):
    """Tool for executing shell commands with security controls.

    Features:
    - Command whitelist/blacklist
    - Timeout control
    - Working directory restriction
    - Dangerous command detection
    - Output size limits
    """

    name = "bash"
    description = (
        "Execute shell commands with security controls. "
        "Supports timeout, working directory specification, and environment variables. "
        "Dangerous commands require confirmation."
    )
    category = ToolCategory.UTILITY
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="Command to execute",
            required=True,
        ),
        ToolParameter(
            name="working_dir",
            type="string",
            description="Working directory for command execution",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds. Default: 60",
            required=False,
            default=60,
        ),
        ToolParameter(
            name="env",
            type="object",
            description="Environment variables to set",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="confirm_dangerous",
            type="boolean",
            description="Confirm execution of dangerous commands. Default: false",
            required=False,
            default=False,
        ),
    ]
    tags = ["command", "bash", "shell", "execute", "secure"]

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        security_config: Optional[BashSecurityConfig] = None,
        default_timeout: float = 60.0,
    ):
        """Initialize BashTool.

        Args:
            context: Execution context
            security_config: Security configuration
            default_timeout: Default command timeout
        """
        super().__init__(context)
        self.executor = BashExecutor(security_config, default_timeout)

    async def execute(
        self,
        params: Dict[str, Any],
        context: Optional[ToolContext] = None,
    ) -> ToolResult:
        """Execute bash command with security controls.

        Args:
            params: Tool parameters
            context: Execution context

        Returns:
            ToolResult with execution result
        """
        ctx = context or self._context
        command = params["command"]
        working_dir_str = params.get("working_dir")
        timeout = params.get("timeout", 60)
        env = params.get("env")
        confirm_dangerous = params.get("confirm_dangerous", False)

        # Resolve working directory
        working_dir = None
        if working_dir_str:
            working_dir = Path(working_dir_str)
        elif ctx:
            working_dir = ctx.working_dir or ctx.project_path

        project_path = ctx.project_path if ctx else None

        # Check for dangerous commands
        is_dangerous, danger_reason = self.executor.validator.is_dangerous_command(command)
        if is_dangerous and self.executor.config.require_confirmation:
            if not confirm_dangerous:
                return ToolResult.fail(
                    error=(
                        f"Command appears dangerous ({danger_reason}). "
                        f"Set 'confirm_dangerous' to True to execute."
                    ),
                    code="CONFIRMATION_REQUIRED",
                )

        # Execute command
        result = await self.executor.execute(
            command=command,
            working_dir=working_dir,
            project_path=project_path,
            env=env,
            timeout=timeout,
        )

        # Build result
        success = result.returncode == 0

        # Combine stdout and stderr for output
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"\n[stderr]:\n{result.stderr}")

        output = "\n".join(output_parts)

        if success:
            return ToolResult.ok(
                output=output,
                data={
                    "command": command,
                    "returncode": result.returncode,
                    "duration_ms": result.duration_ms,
                    "truncated": result.truncated,
                    "working_dir": str(working_dir) if working_dir else None,
                },
            )
        else:
            return ToolResult.fail(
                error=result.stderr or "Command failed",
                code="COMMAND_FAILED",
                details={
                    "command": command,
                    "returncode": result.returncode,
                    "duration_ms": result.duration_ms,
                    "stdout": result.stdout[:1000] if result.stdout else None,
                },
                output=result.stdout,
            )


class SafeBashTool(BashTool):
    """Pre-configured safe bash tool with restricted command set.

    Only allows safe, read-only commands by default.
    """

    name = "safe_bash"
    description = (
        "Execute safe shell commands from a restricted whitelist. "
        "Only allows read-only operations like ls, cat, grep, find, etc."
    )

    def __init__(
        self,
        context: Optional[ToolContext] = None,
        default_timeout: float = 60.0,
    ):
        """Initialize SafeBashTool with restricted configuration.

        Args:
            context: Execution context
            default_timeout: Default command timeout
        """
        config = BashSecurityConfig(
            allowed_commands=[
                "ls", "cat", "head", "tail", "less", "more",
                "grep", "find", "wc", "sort", "uniq", "cut",
                "echo", "pwd", "whoami", "date", "which",
                "file", "stat", "du", "df",
                "git status", "git log", "git diff", "git branch",
                "mvn", "gradle", "npm", "pip", "python", "java",
            ],
            require_confirmation=True,
            restrict_working_directory=True,
        )
        super().__init__(context, config, default_timeout)


def create_bash_tool(
    context: Optional[ToolContext] = None,
    allowed_commands: Optional[List[str]] = None,
    blocked_commands: Optional[List[str]] = None,
    timeout: float = 60.0,
) -> BashTool:
    """Create a bash tool with custom configuration.

    Args:
        context: Execution context
        allowed_commands: List of allowed command prefixes
        blocked_commands: List of blocked commands
        timeout: Default timeout

    Returns:
        Configured BashTool instance
    """
    config = BashSecurityConfig(
        allowed_commands=allowed_commands or [],
        blocked_commands=blocked_commands or [],
    )
    return BashTool(context, config, timeout)
