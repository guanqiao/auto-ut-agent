"""Tool Security Control - Permission levels and confirmation mechanisms.

This module provides:
- Security levels for tools
- Confirmation mechanisms
- Permission management
- Audit logging
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ToolSecurityLevel(Enum):
    """Tool security levels."""
    SAFE = 1
    NORMAL = 2
    CAUTION = 3
    DANGEROUS = 4


class ConfirmationResult(Enum):
    """Confirmation result types."""
    APPROVED = auto()
    DENIED = auto()
    SKIPPED = auto()


@dataclass
class SecurityPolicy:
    """Security policy for tool execution."""
    security_level: ToolSecurityLevel
    require_confirmation: bool = False
    allowed_users: Optional[Set[str]] = None
    blocked: bool = False


@dataclass
class ExecutionLog:
    """Log of tool execution."""
    timestamp: datetime
    tool_name: str
    parameters: Dict[str, Any]
    user: str
    security_level: ToolSecurityLevel
    confirmed: bool
    result: str


class ToolSecurityManager:
    """Manages tool security and permissions.
    
    Features:
    - Configurable security levels per tool
    - User confirmation for dangerous operations
    - Execution audit logging
    - Permission management
    """
    
    DEFAULT_SECURITY_CONFIG = {
        "read_file": ToolSecurityLevel.SAFE,
        "write_file": ToolSecurityLevel.CAUTION,
        "edit_file": ToolSecurityLevel.CAUTION,
        "glob": ToolSecurityLevel.SAFE,
        "grep": ToolSecurityLevel.SAFE,
        "bash": ToolSecurityLevel.DANGEROUS,
        "git_status": ToolSecurityLevel.SAFE,
        "git_diff": ToolSecurityLevel.SAFE,
        "git_commit": ToolSecurityLevel.DANGEROUS,
        "git_push": ToolSecurityLevel.DANGEROUS,
        "git_pull": ToolSecurityLevel.DANGEROUS,
        "git_branch": ToolSecurityLevel.NORMAL,
        "git_add": ToolSecurityLevel.CAUTION,
    }
    
    def __init__(self):
        """Initialize security manager."""
        self._security_config: Dict[str, ToolSecurityLevel] = self.DEFAULT_SECURITY_CONFIG.copy()
        self._custom_policies: Dict[str, SecurityPolicy] = {}
        self._execution_logs: List[ExecutionLog] = []
        self._confirmation_callback: Optional[Callable] = None
        self._dangerous_tools: Set[str] = {
            "bash", "git_commit", "git_push", "git_pull", "git_reset"
        }
        
        logger.info("[ToolSecurityManager] Initialized with default security config")
    
    def set_security_level(self, tool_name: str, level: ToolSecurityLevel):
        """Set security level for a tool.
        
        Args:
            tool_name: Name of the tool
            level: Security level
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
            Security level
        """
        return self._security_config.get(tool_name, ToolSecurityLevel.NORMAL)
    
    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if tool requires confirmation.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            True if confirmation is required
        """
        level = self.get_security_level(tool_name)
        return level in (ToolSecurityLevel.CAUTION, ToolSecurityLevel.DANGEROUS)
    
    def is_blocked(self, tool_name: str) -> bool:
        """Check if tool is blocked.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            True if tool is blocked
        """
        policy = self._custom_policies.get(tool_name)
        if policy:
            return policy.blocked
        return False
    
    def set_confirmation_callback(self, callback: Callable[[str, Dict], ConfirmationResult]):
        """Set callback for user confirmation.
        
        Args:
            callback: Async function that takes (tool_name, params) and returns ConfirmationResult
        """
        self._confirmation_callback = callback
        logger.info("[ToolSecurityManager] Confirmation callback set")
    
    async def check_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user: str = "system"
    ) -> tuple[bool, Optional[str]]:
        """Check if tool execution is allowed.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            user: User requesting execution
        
        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        if self.is_blocked(tool_name):
            reason = f"Tool {tool_name} is blocked by security policy"
            self._log_execution(tool_name, parameters, user, False, reason)
            return False, reason
        
        security_level = self.get_security_level(tool_name)
        
        if self._confirmation_callback and security_level in (
            ToolSecurityLevel.CAUTION,
            ToolSecurityLevel.DANGEROUS
        ):
            result = await self._confirmation_callback(tool_name, parameters)
            
            if result == ConfirmationResult.DENIED:
                reason = f"User denied execution of {tool_name}"
                self._log_execution(tool_name, parameters, user, False, reason)
                return False, reason
            
            if result == ConfirmationResult.SKIPPED:
                reason = f"User skipped execution of {tool_name}"
                self._log_execution(tool_name, parameters, user, False, reason)
                return False, reason
        
        self._log_execution(tool_name, parameters, user, True, "Approved")
        return True, None
    
    def _log_execution(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user: str,
        confirmed: bool,
        result: str
    ):
        """Log tool execution.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            user: User who executed
            confirmed: Whether execution was confirmed
            result: Result/reason
        """
        log = ExecutionLog(
            timestamp=datetime.now(),
            tool_name=tool_name,
            parameters=self._sanitize_params(parameters),
            user=user,
            security_level=self.get_security_level(tool_name),
            confirmed=confirmed,
            result=result
        )
        
        self._execution_logs.append(log)
        
        if len(self._execution_logs) > 1000:
            self._execution_logs = self._execution_logs[-1000:]
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for logging.
        
        Args:
            params: Original parameters
        
        Returns:
            Sanitized parameters
        """
        sensitive_keys = {"password", "token", "secret", "api_key", "credential"}
        
        sanitized = {}
        for key, value in params.items():
            if any(s in key.lower() for s in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def get_execution_logs(
        self,
        limit: int = 100,
        tool_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get execution logs.
        
        Args:
            limit: Maximum number of logs to return
            tool_name: Optional filter by tool name
        
        Returns:
            List of execution logs
        """
        logs = self._execution_logs
        
        if tool_name:
            logs = [l for l in logs if l.tool_name == tool_name]
        
        logs = sorted(logs, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        return [
            {
                "timestamp": l.timestamp.isoformat(),
                "tool_name": l.tool_name,
                "user": l.user,
                "security_level": l.security_level.name,
                "confirmed": l.confirmed,
                "result": l.result
            }
            for l in logs
        ]
    
    def block_tool(self, tool_name: str):
        """Block a tool completely.
        
        Args:
            tool_name: Name of the tool
        """
        policy = SecurityPolicy(
            security_level=ToolSecurityLevel.DANGEROUS,
            blocked=True
        )
        self._custom_policies[tool_name] = policy
        logger.info(f"[ToolSecurityManager] Blocked tool: {tool_name}")
    
    def unblock_tool(self, tool_name: str):
        """Unblock a tool.
        
        Args:
            tool_name: Name of the tool
        """
        if tool_name in self._custom_policies:
            del self._custom_policies[tool_name]
        logger.info(f"[ToolSecurityManager] Unblocked tool: {tool_name}")
    
    def get_dangerous_tools(self) -> List[str]:
        """Get list of dangerous tools.
        
        Returns:
            List of tool names marked as dangerous
        """
        return list(self._dangerous_tools)
    
    def reset_to_defaults(self):
        """Reset security config to defaults."""
        self._security_config = self.DEFAULT_SECURITY_CONFIG.copy()
        self._custom_policies.clear()
        logger.info("[ToolSecurityManager] Reset to defaults")


class SafeToolExecutor:
    """Executor with security checks.
    
    Wraps tool execution with security verification.
    """
    
    def __init__(self, tool_service, security_manager: Optional[ToolSecurityManager] = None):
        """Initialize safe executor.
        
        Args:
            tool_service: Tool service for execution
            security_manager: Optional security manager
        """
        self.tool_service = tool_service
        self.security_manager = security_manager or ToolSecurityManager()
    
    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user: str = "system",
        skip_security_check: bool = False
    ):
        """Execute tool with security checks.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            user: User executing the tool
            skip_security_check: Skip security check
        
        Returns:
            ToolResult
        
        Raises:
            PermissionError: If execution is denied
        """
        if not skip_security_check:
            allowed, reason = await self.security_manager.check_execution(
                tool_name, parameters, user
            )
            
            if not allowed:
                raise PermissionError(reason)
        
        return await self.tool_service.execute_tool(tool_name, parameters)


def create_security_manager() -> ToolSecurityManager:
    """Create a security manager with default config.
    
    Returns:
        Configured ToolSecurityManager
    """
    return ToolSecurityManager()
