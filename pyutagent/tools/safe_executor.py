"""Safe Tool Executor - Enhanced security and controllability for tool execution.

This module provides comprehensive security controls for tool execution, inspired by
Cursor's security protection mechanisms. It includes:

- Multi-layered security protection
- Configurable security policies
- User confirmation mechanisms
- Audit logging
- File deletion protection
- Workspace boundary protection
- Privacy mode support

Example:
    >>> from pyutagent.tools.safe_executor import SafeToolExecutor, ToolSecurityLevel
    >>> executor = SafeToolExecutor(workspace_path="/path/to/workspace")
    >>> result = await executor.execute("write_file", {"path": "test.txt", "content": "hello"})
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Protocol
from collections import deque

logger = logging.getLogger(__name__)


class ToolSecurityLevel(Enum):
    """Security levels for tools.
    
    Each level represents different risk categories:
    - SAFE: Read-only operations, no risk
    - NORMAL: Low-risk operations with minimal side effects
    - CAUTION: Operations that modify state but are generally safe
    - DANGEROUS: Operations that can cause significant changes or data loss
    """
    SAFE = 1
    NORMAL = 2
    CAUTION = 3
    DANGEROUS = 4


class ConfirmationResult(Enum):
    """Result of a user confirmation request."""
    APPROVED = auto()
    DENIED = auto()
    SKIPPED = auto()
    TIMEOUT = auto()


class PrivacyMode(Enum):
    """Privacy mode settings."""
    OFF = "off"  # Normal operation
    LOCAL_ONLY = "local_only"  # No network access
    STRICT = "strict"  # Maximum restrictions


class ExecutionStatus(Enum):
    """Status of tool execution."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SecurityPolicy:
    """Security policy configuration for tool execution.
    
    Attributes:
        security_level: Base security level for the tool
        require_confirmation: Whether user confirmation is required
        allowed_users: Set of users allowed to execute this tool
        blocked: Whether the tool is completely blocked
        max_daily_executions: Maximum number of executions per day
        allowed_paths: List of allowed path patterns
        blocked_paths: List of blocked path patterns
        require_reason: Whether to require a reason for execution
    """
    security_level: ToolSecurityLevel = ToolSecurityLevel.NORMAL
    require_confirmation: bool = False
    allowed_users: Optional[Set[str]] = None
    blocked: bool = False
    max_daily_executions: Optional[int] = None
    allowed_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    require_reason: bool = False


@dataclass
class ExecutionLog:
    """Log entry for tool execution.
    
    Attributes:
        timestamp: When the execution occurred
        tool_name: Name of the tool
        parameters: Tool parameters (sanitized)
        user: User who executed the tool
        security_level: Security level of the tool
        confirmed: Whether execution was confirmed
        result: Execution result status
        execution_time_ms: Time taken to execute in milliseconds
        session_id: Session identifier
        reason: Reason provided for execution
    """
    timestamp: datetime
    tool_name: str
    parameters: Dict[str, Any]
    user: str
    security_level: ToolSecurityLevel
    confirmed: bool
    result: str
    execution_time_ms: Optional[float] = None
    session_id: Optional[str] = None
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "user": self.user,
            "security_level": self.security_level.name,
            "confirmed": self.confirmed,
            "result": self.result,
            "execution_time_ms": self.execution_time_ms,
            "session_id": self.session_id,
            "reason": self.reason,
        }


@dataclass
class DeletionProtectionConfig:
    """Configuration for file deletion protection.
    
    Attributes:
        enabled: Whether deletion protection is enabled
        require_confirmation: Whether to require confirmation before deletion
        whitelist_patterns: Patterns for paths that can be deleted without confirmation
        blacklist_patterns: Patterns for paths that cannot be deleted
        max_files_without_confirmation: Maximum files that can be deleted at once without confirmation
        backup_before_delete: Whether to backup files before deletion
        backup_retention_days: How long to keep backups
    """
    enabled: bool = True
    require_confirmation: bool = True
    whitelist_patterns: List[str] = field(default_factory=list)
    blacklist_patterns: List[str] = field(default_factory=lambda: [
        ".git/**",
        "**/important/**",
        "**/critical/**",
        "**/.env*",
        "**/secrets/**",
        "**/credentials/**",
    ])
    max_files_without_confirmation: int = 1
    backup_before_delete: bool = True
    backup_retention_days: int = 7


@dataclass
class WorkspaceProtectionConfig:
    """Configuration for workspace boundary protection.
    
    Attributes:
        enabled: Whether workspace protection is enabled
        allowed_workspaces: List of allowed workspace paths
        allow_subdirectories: Whether to allow operations in subdirectories
        allow_parent_access: Whether to allow access to parent directories
        allow_symlinks: Whether to follow symbolic links
        max_path_depth: Maximum directory depth allowed
    """
    enabled: bool = True
    allowed_workspaces: List[str] = field(default_factory=list)
    allow_subdirectories: bool = True
    allow_parent_access: bool = False
    allow_symlinks: bool = False
    max_path_depth: int = 20


@dataclass
class AuditConfig:
    """Configuration for audit logging.
    
    Attributes:
        enabled: Whether audit logging is enabled
        log_file_path: Path to audit log file
        max_log_size_mb: Maximum log file size in MB
        max_log_files: Maximum number of log files to keep
        log_retention_days: Number of days to retain logs
        log_sensitive_operations_only: Only log sensitive operations
        include_parameters: Whether to include parameters in logs
    """
    enabled: bool = True
    log_file_path: Optional[str] = None
    max_log_size_mb: int = 100
    max_log_files: int = 10
    log_retention_days: int = 90
    log_sensitive_operations_only: bool = False
    include_parameters: bool = True


class ConfirmationCallback(Protocol):
    """Protocol for confirmation callback functions."""
    
    async def __call__(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        security_level: ToolSecurityLevel,
        reason: Optional[str] = None
    ) -> ConfirmationResult:
        """Request user confirmation.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            security_level: Security level of the tool
            reason: Optional reason for execution
            
        Returns:
            ConfirmationResult indicating user decision
        """
        ...


class ToolSecurityManager:
    """Manages tool security policies and permissions.
    
    This class provides comprehensive security management for tool execution,
    including security level assignment, permission checking, and policy enforcement.
    
    Example:
        >>> manager = ToolSecurityManager()
        >>> manager.set_security_level("bash", ToolSecurityLevel.DANGEROUS)
        >>> manager.block_tool("dangerous_command")
    """
    
    DEFAULT_SECURITY_CONFIG: Dict[str, ToolSecurityLevel] = {
        # File operations
        "read_file": ToolSecurityLevel.SAFE,
        "write_file": ToolSecurityLevel.CAUTION,
        "edit_file": ToolSecurityLevel.CAUTION,
        "delete_file": ToolSecurityLevel.DANGEROUS,
        "move_file": ToolSecurityLevel.CAUTION,
        "copy_file": ToolSecurityLevel.NORMAL,
        "glob": ToolSecurityLevel.SAFE,
        "list_dir": ToolSecurityLevel.SAFE,
        
        # Search operations
        "grep": ToolSecurityLevel.SAFE,
        "search_files": ToolSecurityLevel.SAFE,
        "semantic_grep": ToolSecurityLevel.SAFE,
        
        # Shell operations
        "bash": ToolSecurityLevel.DANGEROUS,
        "shell": ToolSecurityLevel.DANGEROUS,
        "run_command": ToolSecurityLevel.DANGEROUS,
        
        # Git operations
        "git_status": ToolSecurityLevel.SAFE,
        "git_diff": ToolSecurityLevel.SAFE,
        "git_log": ToolSecurityLevel.SAFE,
        "git_add": ToolSecurityLevel.CAUTION,
        "git_commit": ToolSecurityLevel.DANGEROUS,
        "git_push": ToolSecurityLevel.DANGEROUS,
        "git_pull": ToolSecurityLevel.DANGEROUS,
        "git_branch": ToolSecurityLevel.NORMAL,
        "git_checkout": ToolSecurityLevel.CAUTION,
        "git_reset": ToolSecurityLevel.DANGEROUS,
        "git_stash": ToolSecurityLevel.CAUTION,
        
        # Build operations
        "maven_build": ToolSecurityLevel.CAUTION,
        "gradle_build": ToolSecurityLevel.CAUTION,
        "npm_install": ToolSecurityLevel.CAUTION,
        
        # Network operations
        "http_request": ToolSecurityLevel.CAUTION,
        "download_file": ToolSecurityLevel.CAUTION,
        "upload_file": ToolSecurityLevel.DANGEROUS,
    }
    
    def __init__(
        self,
        privacy_mode: PrivacyMode = PrivacyMode.OFF,
        deletion_config: Optional[DeletionProtectionConfig] = None,
        workspace_config: Optional[WorkspaceProtectionConfig] = None,
        audit_config: Optional[AuditConfig] = None,
    ):
        """Initialize the security manager.
        
        Args:
            privacy_mode: Privacy mode setting
            deletion_config: Configuration for deletion protection
            workspace_config: Configuration for workspace protection
            audit_config: Configuration for audit logging
        """
        self._security_config: Dict[str, ToolSecurityLevel] = self.DEFAULT_SECURITY_CONFIG.copy()
        self._custom_policies: Dict[str, SecurityPolicy] = {}
        self._execution_logs: deque = deque(maxlen=10000)
        self._confirmation_callback: Optional[ConfirmationCallback] = None
        self._dangerous_tools: Set[str] = set()
        self._daily_execution_counts: Dict[str, Dict[str, int]] = {}
        
        self._privacy_mode = privacy_mode
        self._deletion_config = deletion_config or DeletionProtectionConfig()
        self._workspace_config = workspace_config or WorkspaceProtectionConfig()
        self._audit_config = audit_config or AuditConfig()
        
        # Initialize dangerous tools set
        for tool, level in self._security_config.items():
            if level == ToolSecurityLevel.DANGEROUS:
                self._dangerous_tools.add(tool)
        
        logger.info(f"[ToolSecurityManager] Initialized with privacy_mode={privacy_mode.value}")
    
    def set_security_level(self, tool_name: str, level: ToolSecurityLevel) -> None:
        """Set security level for a tool.
        
        Args:
            tool_name: Name of the tool
            level: Security level to assign
        """
        self._security_config[tool_name] = level
        
        if level == ToolSecurityLevel.DANGEROUS:
            self._dangerous_tools.add(tool_name)
        else:
            self._dangerous_tools.discard(tool_name)
        
        logger.info(f"[ToolSecurityManager] Set {tool_name} to {level.name}")
    
    def get_security_level(self, tool_name: str) -> ToolSecurityLevel:
        """Get security level for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Security level (defaults to NORMAL if not configured)
        """
        return self._security_config.get(tool_name, ToolSecurityLevel.NORMAL)
    
    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if tool requires user confirmation.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if confirmation is required
        """
        # Check custom policy first
        policy = self._custom_policies.get(tool_name)
        if policy:
            return policy.require_confirmation
        
        # Use default based on security level
        level = self.get_security_level(tool_name)
        return level in (ToolSecurityLevel.CAUTION, ToolSecurityLevel.DANGEROUS)
    
    def is_blocked(self, tool_name: str, user: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check if tool is blocked.
        
        Args:
            tool_name: Name of the tool
            user: Optional user to check permissions for
            
        Returns:
            Tuple of (is_blocked, reason)
        """
        # Check custom policy
        policy = self._custom_policies.get(tool_name)
        if policy:
            if policy.blocked:
                return True, f"Tool {tool_name} is blocked by security policy"
            
            if policy.allowed_users and user and user not in policy.allowed_users:
                return True, f"User {user} is not allowed to use {tool_name}"
            
            # Check daily execution limit
            if policy.max_daily_executions is not None:
                today = datetime.now().strftime("%Y-%m-%d")
                count = self._daily_execution_counts.get(tool_name, {}).get(today, 0)
                if count >= policy.max_daily_executions:
                    return True, f"Daily execution limit reached for {tool_name}"
        
        # Check privacy mode restrictions
        if self._privacy_mode == PrivacyMode.STRICT:
            level = self.get_security_level(tool_name)
            if level == ToolSecurityLevel.DANGEROUS:
                return True, f"Tool {tool_name} is blocked in strict privacy mode"
        
        return False, None
    
    def set_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        """Set callback for user confirmation.
        
        Args:
            callback: Async function for requesting confirmation
        """
        self._confirmation_callback = callback
        logger.info("[ToolSecurityManager] Confirmation callback set")
    
    async def check_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user: str = "system",
        reason: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Check if tool execution is allowed.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            user: User requesting execution
            reason: Optional reason for execution
            
        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        # Check if blocked
        blocked, block_reason = self.is_blocked(tool_name, user)
        if blocked:
            self._log_execution(tool_name, parameters, user, False, block_reason or "Blocked")
            return False, block_reason
        
        # Check custom policy for required reason
        policy = self._custom_policies.get(tool_name)
        if policy and policy.require_reason and not reason:
            return False, f"Execution reason required for {tool_name}"
        
        # Get security level
        security_level = self.get_security_level(tool_name)
        
        # Request confirmation if needed
        if self._confirmation_callback and self.requires_confirmation(tool_name):
            result = await self._confirmation_callback(
                tool_name, parameters, security_level, reason
            )
            
            if result == ConfirmationResult.DENIED:
                deny_reason = f"User denied execution of {tool_name}"
                self._log_execution(tool_name, parameters, user, False, deny_reason)
                return False, deny_reason
            
            if result == ConfirmationResult.SKIPPED:
                skip_reason = f"User skipped execution of {tool_name}"
                self._log_execution(tool_name, parameters, user, False, skip_reason)
                return False, skip_reason
            
            if result == ConfirmationResult.TIMEOUT:
                timeout_reason = f"Confirmation timeout for {tool_name}"
                self._log_execution(tool_name, parameters, user, False, timeout_reason)
                return False, timeout_reason
        
        self._log_execution(tool_name, parameters, user, True, "Approved")
        self._increment_daily_count(tool_name)
        return True, None
    
    def _increment_daily_count(self, tool_name: str) -> None:
        """Increment daily execution count for a tool."""
        today = datetime.now().strftime("%Y-%m-%d")
        if tool_name not in self._daily_execution_counts:
            self._daily_execution_counts[tool_name] = {}
        self._daily_execution_counts[tool_name][today] = \
            self._daily_execution_counts[tool_name].get(today, 0) + 1
    
    def _log_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user: str,
        confirmed: bool,
        result: str,
        execution_time_ms: Optional[float] = None,
        session_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Log tool execution.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            user: User who executed
            confirmed: Whether execution was confirmed
            result: Result/reason
            execution_time_ms: Execution time in milliseconds
            session_id: Session identifier
            reason: Execution reason
        """
        log = ExecutionLog(
            timestamp=datetime.now(),
            tool_name=tool_name,
            parameters=self._sanitize_params(parameters) if self._audit_config.include_parameters else {},
            user=user,
            security_level=self.get_security_level(tool_name),
            confirmed=confirmed,
            result=result,
            execution_time_ms=execution_time_ms,
            session_id=session_id,
            reason=reason,
        )
        
        self._execution_logs.append(log)
        
        # Write to file if configured
        if self._audit_config.enabled and self._audit_config.log_file_path:
            self._write_audit_log(log)
    
    def _write_audit_log(self, log: ExecutionLog) -> None:
        """Write log entry to audit file."""
        try:
            log_path = Path(self._audit_config.log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[ToolSecurityManager] Failed to write audit log: {e}")
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for logging.
        
        Args:
            params: Original parameters
            
        Returns:
            Sanitized parameters with sensitive data masked
        """
        sensitive_keys = {
            "password", "token", "secret", "api_key", "credential",
            "auth", "private_key", "passwd", "pwd", "access_token",
            "refresh_token", "apikey", "api-key", "secret_key",
        }
        
        def sanitize_value(key: str, value: Any) -> Any:
            if isinstance(value, dict):
                return {k: sanitize_value(k, v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [sanitize_value("", v) for v in value]
            elif isinstance(value, str):
                key_lower = key.lower()
                if any(s in key_lower for s in sensitive_keys):
                    if len(value) <= 8:
                        return "***"
                    return f"{value[:3]}...{value[-3:]}"
            return value
        
        return {k: sanitize_value(k, v) for k, v in params.items()}
    
    def get_execution_logs(
        self,
        limit: int = 100,
        tool_name: Optional[str] = None,
        user: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get execution logs with filtering.
        
        Args:
            limit: Maximum number of logs to return
            tool_name: Optional filter by tool name
            user: Optional filter by user
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of execution logs as dictionaries
        """
        logs = list(self._execution_logs)
        
        if tool_name:
            logs = [l for l in logs if l.tool_name == tool_name]
        
        if user:
            logs = [l for l in logs if l.user == user]
        
        if start_time:
            logs = [l for l in logs if l.timestamp >= start_time]
        
        if end_time:
            logs = [l for l in logs if l.timestamp <= end_time]
        
        logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [l.to_dict() for l in logs]
    
    def block_tool(self, tool_name: str) -> None:
        """Block a tool completely.
        
        Args:
            tool_name: Name of the tool to block
        """
        policy = SecurityPolicy(
            security_level=ToolSecurityLevel.DANGEROUS,
            blocked=True
        )
        self._custom_policies[tool_name] = policy
        logger.info(f"[ToolSecurityManager] Blocked tool: {tool_name}")
    
    def unblock_tool(self, tool_name: str) -> None:
        """Unblock a tool.
        
        Args:
            tool_name: Name of the tool to unblock
        """
        if tool_name in self._custom_policies:
            del self._custom_policies[tool_name]
        logger.info(f"[ToolSecurityManager] Unblocked tool: {tool_name}")
    
    def set_custom_policy(self, tool_name: str, policy: SecurityPolicy) -> None:
        """Set custom security policy for a tool.
        
        Args:
            tool_name: Name of the tool
            policy: Security policy to apply
        """
        self._custom_policies[tool_name] = policy
        logger.info(f"[ToolSecurityManager] Set custom policy for: {tool_name}")
    
    def get_dangerous_tools(self) -> List[str]:
        """Get list of tools marked as dangerous.
        
        Returns:
            List of dangerous tool names
        """
        return list(self._dangerous_tools)
    
    def reset_to_defaults(self) -> None:
        """Reset security configuration to defaults."""
        self._security_config = self.DEFAULT_SECURITY_CONFIG.copy()
        self._custom_policies.clear()
        self._dangerous_tools = {
            tool for tool, level in self._security_config.items()
            if level == ToolSecurityLevel.DANGEROUS
        }
        logger.info("[ToolSecurityManager] Reset to defaults")
    
    @property
    def privacy_mode(self) -> PrivacyMode:
        """Get current privacy mode."""
        return self._privacy_mode
    
    @privacy_mode.setter
    def privacy_mode(self, mode: PrivacyMode) -> None:
        """Set privacy mode."""
        self._privacy_mode = mode
        logger.info(f"[ToolSecurityManager] Privacy mode set to: {mode.value}")
    
    @property
    def deletion_config(self) -> DeletionProtectionConfig:
        """Get deletion protection configuration."""
        return self._deletion_config
    
    @property
    def workspace_config(self) -> WorkspaceProtectionConfig:
        """Get workspace protection configuration."""
        return self._workspace_config


class WorkspaceBoundaryValidator:
    """Validates that file operations stay within workspace boundaries.
    
    This class provides protection against path traversal attacks and ensures
    that tools can only operate within allowed workspace directories.
    """
    
    # Dangerous patterns that could indicate path traversal
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Unix parent directory
        r'\.\.\\',  # Windows parent directory
        r'^\.\./',  # Leading parent directory (Unix)
        r'^\.\.\\',  # Leading parent directory (Windows)
    ]
    
    def __init__(self, config: WorkspaceProtectionConfig):
        """Initialize the validator.
        
        Args:
            config: Workspace protection configuration
        """
        self._config = config
        self._allowed_paths: List[Path] = [
            Path(p).resolve() for p in config.allowed_workspaces
        ]
    
    def validate_path(
        self,
        path: Union[str, Path],
        must_exist: bool = False,
    ) -> Path:
        """Validate a path is within allowed workspace boundaries.
        
        Args:
            path: Path to validate
            must_exist: Whether the path must exist
            
        Returns:
            Resolved Path object
            
        Raises:
            PermissionError: If path is outside allowed boundaries
            ValueError: If path contains dangerous patterns
        """
        path_str = str(path)
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, path_str):
                raise ValueError(f"Path contains dangerous pattern: {pattern}")
        
        # Resolve the path
        try:
            resolved = Path(path_str).resolve()
        except Exception as e:
            raise ValueError(f"Invalid path: {path_str}") from e
        
        # Check symlink if not allowed
        if not self._config.allow_symlinks and Path(path_str).is_symlink():
            raise PermissionError(f"Symbolic links are not allowed: {path_str}")
        
        # Check path depth
        if self._allowed_paths:
            for allowed_path in self._allowed_paths:
                try:
                    rel_path = resolved.relative_to(allowed_path)
                    depth = len(rel_path.parts)
                    if depth > self._config.max_path_depth:
                        raise PermissionError(
                            f"Path exceeds maximum depth ({self._config.max_path_depth}): {path_str}"
                        )
                    break
                except ValueError:
                    continue
        
        # If no allowed paths specified, allow all (but still check patterns)
        if not self._allowed_paths:
            if must_exist and not resolved.exists():
                raise FileNotFoundError(f"Path does not exist: {path_str}")
            return resolved
        
        # Check if path is within allowed boundaries
        is_allowed = False
        for allowed_path in self._allowed_paths:
            try:
                resolved.relative_to(allowed_path)
                is_allowed = True
                break
            except ValueError:
                continue
        
        # Check parent access if enabled
        if not is_allowed and self._config.allow_parent_access:
            for allowed_path in self._allowed_paths:
                try:
                    allowed_path.relative_to(resolved)
                    is_allowed = True
                    break
                except ValueError:
                    continue
        
        if not is_allowed:
            raise PermissionError(
                f"Path {path_str} is outside allowed workspaces: "
                f"{self._config.allowed_workspaces}"
            )
        
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path_str}")
        
        return resolved
    
    def is_within_workspace(self, path: Union[str, Path]) -> bool:
        """Check if a path is within allowed workspaces.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is within allowed workspaces
        """
        try:
            self.validate_path(path)
            return True
        except (PermissionError, ValueError):
            return False


class DeletionProtector:
    """Protects against accidental or malicious file deletions.
    
    This class provides multiple layers of protection for file deletion operations,
    including confirmation requirements, path whitelists/blacklists, and backup
    functionality.
    """
    
    def __init__(
        self,
        config: DeletionProtectionConfig,
        security_manager: ToolSecurityManager,
    ):
        """Initialize the deletion protector.
        
        Args:
            config: Deletion protection configuration
            security_manager: Security manager for confirmation callbacks
        """
        self._config = config
        self._security_manager = security_manager
        self._backup_dir: Optional[Path] = None
        
        if config.backup_before_delete:
            self._setup_backup_directory()
    
    def _setup_backup_directory(self) -> None:
        """Setup backup directory for deleted files."""
        backup_base = Path.home() / ".pyutagent" / "backups" / "deleted_files"
        backup_base.mkdir(parents=True, exist_ok=True)
        self._backup_dir = backup_base
    
    def _matches_pattern(self, path: str, patterns: List[str]) -> bool:
        """Check if path matches any of the given patterns.
        
        Args:
            path: Path to check
            patterns: List of glob patterns
            
        Returns:
            True if path matches any pattern
        """
        from fnmatch import fnmatch
        
        for pattern in patterns:
            if fnmatch(path, pattern) or fnmatch(Path(path).name, pattern):
                return True
        return False
    
    def can_delete_without_confirmation(self, path: str) -> bool:
        """Check if file can be deleted without confirmation.
        
        Args:
            path: Path to check
            
        Returns:
            True if deletion can proceed without confirmation
        """
        if not self._config.enabled:
            return True
        
        # Check blacklist first
        if self._matches_pattern(path, self._config.blacklist_patterns):
            return False
        
        # Check whitelist
        if self._matches_pattern(path, self._config.whitelist_patterns):
            return True
        
        # Default to requiring confirmation
        return False
    
    def is_blacklisted(self, path: str) -> bool:
        """Check if path is blacklisted for deletion.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is blacklisted
        """
        return self._matches_pattern(path, self._config.blacklist_patterns)
    
    async def request_deletion_confirmation(
        self,
        paths: List[str],
        reason: Optional[str] = None,
    ) -> ConfirmationResult:
        """Request user confirmation for deletion.
        
        Args:
            paths: List of paths to delete
            reason: Optional reason for deletion
            
        Returns:
            ConfirmationResult
        """
        if not self._config.require_confirmation:
            return ConfirmationResult.APPROVED
        
        # Check if all paths can be deleted without confirmation
        if len(paths) <= self._config.max_files_without_confirmation:
            all_whitelisted = all(
                self.can_delete_without_confirmation(p) for p in paths
            )
            if all_whitelisted:
                return ConfirmationResult.APPROVED
        
        # Request confirmation through security manager
        if self._security_manager._confirmation_callback:
            return await self._security_manager._confirmation_callback(
                "delete_file",
                {"paths": paths, "count": len(paths)},
                ToolSecurityLevel.DANGEROUS,
                reason,
            )
        
        # Default to requiring confirmation if no callback
        return ConfirmationResult.DENIED
    
    def backup_file(self, path: Union[str, Path]) -> Optional[Path]:
        """Create a backup of a file before deletion.
        
        Args:
            path: Path to backup
            
        Returns:
            Path to backup file, or None if backup failed
        """
        if not self._config.backup_before_delete or not self._backup_dir:
            return None
        
        try:
            source = Path(path).resolve()
            if not source.exists():
                return None
            
            # Create backup with timestamp and hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_hash = hashlib.sha256(str(source).encode()).hexdigest()[:16]
            backup_name = f"{timestamp}_{path_hash}_{source.name}"
            backup_path = self._backup_dir / backup_name
            
            if source.is_file():
                import shutil
                shutil.copy2(source, backup_path)
            elif source.is_dir():
                import shutil
                shutil.copytree(source, backup_path)
            
            logger.info(f"[DeletionProtector] Backed up {source} to {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"[DeletionProtector] Failed to backup {path}: {e}")
            return None
    
    def cleanup_old_backups(self) -> int:
        """Clean up old backup files.
        
        Returns:
            Number of files cleaned up
        """
        if not self._backup_dir:
            return 0
        
        cleaned = 0
        cutoff = time.time() - (self._config.backup_retention_days * 86400)
        
        try:
            for item in self._backup_dir.iterdir():
                try:
                    if item.stat().st_mtime < cutoff:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            import shutil
                            shutil.rmtree(item)
                        cleaned += 1
                except Exception as e:
                    logger.warning(f"[DeletionProtector] Failed to cleanup {item}: {e}")
        except Exception as e:
            logger.error(f"[DeletionProtector] Failed to cleanup backups: {e}")
        
        if cleaned > 0:
            logger.info(f"[DeletionProtector] Cleaned up {cleaned} old backups")
        
        return cleaned


class SafeToolExecutor:
    """Enhanced tool executor with comprehensive security controls.
    
    This class provides a secure wrapper around tool execution, implementing
    multiple layers of protection including:
    
    - Security level-based access control
    - User confirmation for dangerous operations
    - Workspace boundary protection
    - File deletion protection
    - Privacy mode support
    - Comprehensive audit logging
    
    Example:
        >>> executor = SafeToolExecutor(
        ...     workspace_path="/path/to/workspace",
        ...     privacy_mode=PrivacyMode.LOCAL_ONLY
        ... )
        >>> executor.register_tool("write_file", write_file_impl)
        >>> result = await executor.execute("write_file", {"path": "test.txt", "content": "hello"})
    """
    
    def __init__(
        self,
        workspace_path: Optional[Union[str, Path]] = None,
        privacy_mode: PrivacyMode = PrivacyMode.OFF,
        auto_confirm: bool = False,
        session_id: Optional[str] = None,
    ):
        """Initialize the safe tool executor.
        
        Args:
            workspace_path: Primary workspace path for boundary protection
            privacy_mode: Privacy mode setting
            auto_confirm: Whether to auto-confirm all operations (use with caution)
            session_id: Optional session identifier for logging
        """
        self._workspace_path = Path(workspace_path).resolve() if workspace_path else None
        self._session_id = session_id or self._generate_session_id()
        self._auto_confirm = auto_confirm
        
        # Initialize configurations
        deletion_config = DeletionProtectionConfig()
        workspace_config = WorkspaceProtectionConfig(
            allowed_workspaces=[str(self._workspace_path)] if self._workspace_path else []
        )
        audit_config = AuditConfig()
        
        # Initialize security manager
        self._security_manager = ToolSecurityManager(
            privacy_mode=privacy_mode,
            deletion_config=deletion_config,
            workspace_config=workspace_config,
            audit_config=audit_config,
        )
        
        # Initialize validators and protectors
        self._workspace_validator = WorkspaceBoundaryValidator(workspace_config)
        self._deletion_protector = DeletionProtector(deletion_config, self._security_manager)
        
        # Tool registry
        self._tools: Dict[str, Callable] = {}
        
        # Execution statistics
        self._execution_stats: Dict[str, Dict[str, Any]] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "denied_executions": 0,
            "by_tool": {},
        }
        
        # Set up auto-confirm if enabled
        if auto_confirm:
            self._setup_auto_confirm()
        
        logger.info(
            f"[SafeToolExecutor] Initialized with workspace={workspace_path}, "
            f"privacy_mode={privacy_mode.value}"
        )
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.sha256(
            f"{time.time()}{os.urandom(16)}".encode()
        ).hexdigest()[:8]
        return f"{timestamp}_{random_suffix}"
    
    def _setup_auto_confirm(self) -> None:
        """Setup auto-confirm callback for testing."""
        async def auto_confirm_callback(
            tool_name: str,
            parameters: Dict[str, Any],
            security_level: ToolSecurityLevel,
            reason: Optional[str] = None,
        ) -> ConfirmationResult:
            logger.warning(f"[SafeToolExecutor] Auto-confirming {tool_name}")
            return ConfirmationResult.APPROVED
        
        self._security_manager.set_confirmation_callback(auto_confirm_callback)
    
    def register_tool(self, name: str, implementation: Callable) -> None:
        """Register a tool implementation.
        
        Args:
            name: Tool name
            implementation: Tool implementation function
        """
        self._tools[name] = implementation
        logger.debug(f"[SafeToolExecutor] Registered tool: {name}")
    
    def unregister_tool(self, name: str) -> None:
        """Unregister a tool.
        
        Args:
            name: Tool name
        """
        self._tools.pop(name, None)
        logger.debug(f"[SafeToolExecutor] Unregistered tool: {name}")
    
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user: str = "system",
        reason: Optional[str] = None,
        skip_security_check: bool = False,
    ) -> Dict[str, Any]:
        """Execute a tool with security checks.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            user: User executing the tool
            reason: Optional reason for execution
            skip_security_check: Whether to skip security checks (use with caution)
            
        Returns:
            Execution result dictionary
            
        Raises:
            PermissionError: If execution is denied
            ValueError: If tool is not registered
        """
        start_time = time.time()
        
        # Check if tool exists
        if tool_name not in self._tools:
            raise ValueError(f"Tool not registered: {tool_name}")
        
        # Perform security check
        if not skip_security_check:
            allowed, deny_reason = await self._security_manager.check_execution(
                tool_name, parameters, user, reason
            )
            
            if not allowed:
                self._update_stats(tool_name, False, denied=True)
                raise PermissionError(deny_reason)
        
        # Validate workspace boundaries for file operations
        if self._workspace_path:
            self._validate_file_parameters(tool_name, parameters)
        
        # Handle deletion operations specially
        if tool_name in ("delete_file", "delete_files", "remove", "rm"):
            result = await self._handle_deletion(parameters, reason)
            if result["status"] != "approved":
                self._update_stats(tool_name, False, denied=True)
                return result
        
        # Execute the tool
        execution_time_ms = None
        try:
            
            tool_impl = self._tools[tool_name]
            
            # Handle both sync and async tools
            if asyncio.iscoroutinefunction(tool_impl):
                result = await tool_impl(**parameters)
            else:
                result = tool_impl(**parameters)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            self._security_manager._log_execution(
                tool_name=tool_name,
                parameters=parameters,
                user=user,
                confirmed=True,
                result="Success",
                execution_time_ms=execution_time_ms,
                session_id=self._session_id,
                reason=reason,
            )
            
            self._update_stats(tool_name, True)
            
            return {
                "status": "success",
                "result": result,
                "execution_time_ms": execution_time_ms,
                "tool_name": tool_name,
            }
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            self._security_manager._log_execution(
                tool_name=tool_name,
                parameters=parameters,
                user=user,
                confirmed=True,
                result=f"Failed: {str(e)}",
                execution_time_ms=execution_time_ms,
                session_id=self._session_id,
                reason=reason,
            )
            
            self._update_stats(tool_name, False)
            
            raise
    
    def _validate_file_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        """Validate file-related parameters are within workspace.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            
        Raises:
            PermissionError: If path is outside workspace
        """
        file_params = ["path", "file_path", "source", "destination", "dir", "directory"]
        
        for param in file_params:
            if param in parameters:
                path = parameters[param]
                if isinstance(path, str):
                    # Convert relative paths to absolute paths within workspace
                    if self._workspace_path and not Path(path).is_absolute():
                        abs_path = self._workspace_path / path
                        self._workspace_validator.validate_path(abs_path)
                    else:
                        self._workspace_validator.validate_path(path)
                elif isinstance(path, (list, tuple)):
                    for p in path:
                        if isinstance(p, str):
                            if self._workspace_path and not Path(p).is_absolute():
                                abs_path = self._workspace_path / p
                                self._workspace_validator.validate_path(abs_path)
                            else:
                                self._workspace_validator.validate_path(p)
    
    async def _handle_deletion(
        self,
        parameters: Dict[str, Any],
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle file deletion with protection.
        
        Args:
            parameters: Deletion parameters
            reason: Optional reason for deletion
            
        Returns:
            Result dictionary
        """
        # Extract paths from parameters
        paths = []
        if "path" in parameters:
            paths.append(parameters["path"])
        elif "paths" in parameters:
            paths.extend(parameters["paths"])
        elif "file_path" in parameters:
            paths.append(parameters["file_path"])
        
        # Check blacklist
        for path in paths:
            if self._deletion_protector.is_blacklisted(path):
                return {
                    "status": "denied",
                    "reason": f"Path is blacklisted for deletion: {path}",
                }
        
        # Request confirmation
        confirmation = await self._deletion_protector.request_deletion_confirmation(
            paths, reason
        )
        
        if confirmation != ConfirmationResult.APPROVED:
            return {
                "status": "denied",
                "reason": "Deletion not confirmed by user",
            }
        
        # Create backups
        backups = []
        if self._deletion_protector._config.backup_before_delete:
            for path in paths:
                backup = self._deletion_protector.backup_file(path)
                if backup:
                    backups.append(str(backup))
        
        return {
            "status": "approved",
            "backups": backups,
        }
    
    def _update_stats(
        self,
        tool_name: str,
        success: bool = False,
        denied: bool = False,
        execution_started: bool = False,
    ) -> None:
        """Update execution statistics.
        
        Args:
            tool_name: Name of the tool
            success: Whether execution was successful
            denied: Whether execution was denied
            execution_started: Whether execution started
        """
        self._execution_stats["total_executions"] += 1
        
        if denied:
            self._execution_stats["denied_executions"] += 1
        elif success:
            self._execution_stats["successful_executions"] += 1
        else:
            self._execution_stats["failed_executions"] += 1
        
        if tool_name not in self._execution_stats["by_tool"]:
            self._execution_stats["by_tool"][tool_name] = {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "denied": 0,
            }
        
        tool_stats = self._execution_stats["by_tool"][tool_name]
        tool_stats["total"] += 1
        
        if denied:
            tool_stats["denied"] += 1
        elif success:
            tool_stats["successful"] += 1
        else:
            tool_stats["failed"] += 1
    
    def set_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        """Set callback for user confirmation.
        
        Args:
            callback: Async function for requesting confirmation
        """
        self._security_manager.set_confirmation_callback(callback)
    
    def set_security_level(self, tool_name: str, level: ToolSecurityLevel) -> None:
        """Set security level for a tool.
        
        Args:
            tool_name: Name of the tool
            level: Security level
        """
        self._security_manager.set_security_level(tool_name, level)
    
    def block_tool(self, tool_name: str) -> None:
        """Block a tool completely.
        
        Args:
            tool_name: Name of the tool
        """
        self._security_manager.block_tool(tool_name)
    
    def unblock_tool(self, tool_name: str) -> None:
        """Unblock a tool.
        
        Args:
            tool_name: Name of the tool
        """
        self._security_manager.unblock_tool(tool_name)
    
    def get_execution_logs(
        self,
        limit: int = 100,
        **filters
    ) -> List[Dict[str, Any]]:
        """Get execution logs.
        
        Args:
            limit: Maximum number of logs
            **filters: Additional filters
            
        Returns:
            List of execution logs
        """
        return self._security_manager.get_execution_logs(limit=limit, **filters)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Statistics dictionary
        """
        return self._execution_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self._execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "denied_executions": 0,
            "by_tool": {},
        }
    
    def cleanup_old_backups(self) -> int:
        """Clean up old backup files.
        
        Returns:
            Number of files cleaned up
        """
        return self._deletion_protector.cleanup_old_backups()
    
    @property
    def security_manager(self) -> ToolSecurityManager:
        """Get the security manager."""
        return self._security_manager
    
    @property
    def workspace_path(self) -> Optional[Path]:
        """Get the workspace path."""
        return self._workspace_path
    
    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id


def create_security_manager(
    privacy_mode: PrivacyMode = PrivacyMode.OFF,
    **kwargs
) -> ToolSecurityManager:
    """Create a security manager with default configuration.
    
    Args:
        privacy_mode: Privacy mode setting
        **kwargs: Additional configuration options
        
    Returns:
        Configured ToolSecurityManager
    """
    return ToolSecurityManager(privacy_mode=privacy_mode, **kwargs)


def create_safe_executor(
    workspace_path: Optional[Union[str, Path]] = None,
    privacy_mode: PrivacyMode = PrivacyMode.OFF,
    **kwargs
) -> SafeToolExecutor:
    """Create a safe tool executor with default configuration.
    
    Args:
        workspace_path: Workspace path for boundary protection
        privacy_mode: Privacy mode setting
        **kwargs: Additional configuration options
        
    Returns:
        Configured SafeToolExecutor
    """
    return SafeToolExecutor(
        workspace_path=workspace_path,
        privacy_mode=privacy_mode,
        **kwargs
    )
