"""Unit tests for safe_executor module.

This module contains comprehensive tests for the SafeToolExecutor and related
security components.
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyutagent.tools.safe_executor import (
    AuditConfig,
    ConfirmationResult,
    DeletionProtectionConfig,
    DeletionProtector,
    ExecutionLog,
    ExecutionStatus,
    PrivacyMode,
    SafeToolExecutor,
    SecurityPolicy,
    ToolSecurityLevel,
    ToolSecurityManager,
    WorkspaceBoundaryValidator,
    WorkspaceProtectionConfig,
    create_safe_executor,
    create_security_manager,
)


class TestToolSecurityLevel:
    """Tests for ToolSecurityLevel enum."""
    
    def test_security_level_values(self):
        """Test that security levels have correct values."""
        assert ToolSecurityLevel.SAFE.value == 1
        assert ToolSecurityLevel.NORMAL.value == 2
        assert ToolSecurityLevel.CAUTION.value == 3
        assert ToolSecurityLevel.DANGEROUS.value == 4
    
    def test_security_level_ordering(self):
        """Test that security levels can be compared."""
        assert ToolSecurityLevel.SAFE.value < ToolSecurityLevel.NORMAL.value
        assert ToolSecurityLevel.NORMAL.value < ToolSecurityLevel.CAUTION.value
        assert ToolSecurityLevel.CAUTION.value < ToolSecurityLevel.DANGEROUS.value


class TestConfirmationResult:
    """Tests for ConfirmationResult enum."""
    
    def test_confirmation_results(self):
        """Test confirmation result values."""
        assert ConfirmationResult.APPROVED.name == "APPROVED"
        assert ConfirmationResult.DENIED.name == "DENIED"
        assert ConfirmationResult.SKIPPED.name == "SKIPPED"
        assert ConfirmationResult.TIMEOUT.name == "TIMEOUT"


class TestPrivacyMode:
    """Tests for PrivacyMode enum."""
    
    def test_privacy_modes(self):
        """Test privacy mode values."""
        assert PrivacyMode.OFF.value == "off"
        assert PrivacyMode.LOCAL_ONLY.value == "local_only"
        assert PrivacyMode.STRICT.value == "strict"


class TestSecurityPolicy:
    """Tests for SecurityPolicy dataclass."""
    
    def test_default_policy(self):
        """Test default security policy."""
        policy = SecurityPolicy()
        assert policy.security_level == ToolSecurityLevel.NORMAL
        assert policy.require_confirmation is False
        assert policy.allowed_users is None
        assert policy.blocked is False
        assert policy.max_daily_executions is None
        assert policy.allowed_paths == []
        assert policy.blocked_paths == []
        assert policy.require_reason is False
    
    def test_custom_policy(self):
        """Test custom security policy."""
        policy = SecurityPolicy(
            security_level=ToolSecurityLevel.DANGEROUS,
            require_confirmation=True,
            allowed_users={"admin", "user"},
            blocked=True,
            max_daily_executions=10,
            allowed_paths=["/workspace"],
            blocked_paths=["/etc"],
            require_reason=True,
        )
        assert policy.security_level == ToolSecurityLevel.DANGEROUS
        assert policy.require_confirmation is True
        assert policy.allowed_users == {"admin", "user"}
        assert policy.blocked is True
        assert policy.max_daily_executions == 10
        assert policy.allowed_paths == ["/workspace"]
        assert policy.blocked_paths == ["/etc"]
        assert policy.require_reason is True


class TestExecutionLog:
    """Tests for ExecutionLog dataclass."""
    
    def test_log_creation(self):
        """Test creating execution log."""
        log = ExecutionLog(
            timestamp=datetime.now(),
            tool_name="test_tool",
            parameters={"key": "value"},
            user="test_user",
            security_level=ToolSecurityLevel.SAFE,
            confirmed=True,
            result="Success",
            execution_time_ms=100.0,
            session_id="session_123",
            reason="Test execution",
        )
        assert log.tool_name == "test_tool"
        assert log.user == "test_user"
        assert log.confirmed is True
        assert log.execution_time_ms == 100.0
    
    def test_log_to_dict(self):
        """Test converting log to dictionary."""
        timestamp = datetime.now()
        log = ExecutionLog(
            timestamp=timestamp,
            tool_name="test_tool",
            parameters={"key": "value"},
            user="test_user",
            security_level=ToolSecurityLevel.SAFE,
            confirmed=True,
            result="Success",
        )
        log_dict = log.to_dict()
        assert log_dict["tool_name"] == "test_tool"
        assert log_dict["user"] == "test_user"
        assert log_dict["security_level"] == "SAFE"
        assert log_dict["confirmed"] is True
        assert log_dict["timestamp"] == timestamp.isoformat()


class TestDeletionProtectionConfig:
    """Tests for DeletionProtectionConfig dataclass."""
    
    def test_default_config(self):
        """Test default deletion protection config."""
        config = DeletionProtectionConfig()
        assert config.enabled is True
        assert config.require_confirmation is True
        assert config.max_files_without_confirmation == 1
        assert config.backup_before_delete is True
        assert config.backup_retention_days == 7
        # Check default blacklist patterns
        assert ".git/**" in config.blacklist_patterns
        assert "**/.env*" in config.blacklist_patterns


class TestWorkspaceProtectionConfig:
    """Tests for WorkspaceProtectionConfig dataclass."""
    
    def test_default_config(self):
        """Test default workspace protection config."""
        config = WorkspaceProtectionConfig()
        assert config.enabled is True
        assert config.allowed_workspaces == []
        assert config.allow_subdirectories is True
        assert config.allow_parent_access is False
        assert config.allow_symlinks is False
        assert config.max_path_depth == 20


class TestAuditConfig:
    """Tests for AuditConfig dataclass."""
    
    def test_default_config(self):
        """Test default audit config."""
        config = AuditConfig()
        assert config.enabled is True
        assert config.log_file_path is None
        assert config.max_log_size_mb == 100
        assert config.max_log_files == 10
        assert config.log_retention_days == 90
        assert config.log_sensitive_operations_only is False
        assert config.include_parameters is True


class TestToolSecurityManager:
    """Tests for ToolSecurityManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a security manager for testing."""
        return ToolSecurityManager()
    
    def test_initialization(self, manager):
        """Test security manager initialization."""
        assert manager.privacy_mode == PrivacyMode.OFF
        assert manager.get_security_level("read_file") == ToolSecurityLevel.SAFE
        assert manager.get_security_level("bash") == ToolSecurityLevel.DANGEROUS
    
    def test_set_security_level(self, manager):
        """Test setting security level."""
        manager.set_security_level("custom_tool", ToolSecurityLevel.CAUTION)
        assert manager.get_security_level("custom_tool") == ToolSecurityLevel.CAUTION
    
    def test_requires_confirmation(self, manager):
        """Test confirmation requirement check."""
        assert manager.requires_confirmation("read_file") is False
        assert manager.requires_confirmation("write_file") is True
        assert manager.requires_confirmation("bash") is True
    
    def test_is_blocked_default(self, manager):
        """Test default blocked status."""
        blocked, reason = manager.is_blocked("read_file")
        assert blocked is False
        assert reason is None
    
    def test_block_tool(self, manager):
        """Test blocking a tool."""
        manager.block_tool("dangerous_tool")
        blocked, reason = manager.is_blocked("dangerous_tool")
        assert blocked is True
        assert "blocked by security policy" in reason
    
    def test_unblock_tool(self, manager):
        """Test unblocking a tool."""
        manager.block_tool("test_tool")
        manager.unblock_tool("test_tool")
        blocked, _ = manager.is_blocked("test_tool")
        assert blocked is False
    
    def test_get_dangerous_tools(self, manager):
        """Test getting dangerous tools list."""
        dangerous = manager.get_dangerous_tools()
        assert "bash" in dangerous
        assert "delete_file" in dangerous
        assert "read_file" not in dangerous
    
    def test_reset_to_defaults(self, manager):
        """Test resetting to defaults."""
        manager.set_security_level("read_file", ToolSecurityLevel.DANGEROUS)
        manager.block_tool("test_tool")
        manager.reset_to_defaults()
        assert manager.get_security_level("read_file") == ToolSecurityLevel.SAFE
        blocked, _ = manager.is_blocked("test_tool")
        assert blocked is False
    
    def test_set_custom_policy(self, manager):
        """Test setting custom policy."""
        policy = SecurityPolicy(
            security_level=ToolSecurityLevel.DANGEROUS,
            require_confirmation=True,
            blocked=False,
        )
        manager.set_custom_policy("custom_tool", policy)
        assert manager.requires_confirmation("custom_tool") is True
    
    def test_daily_execution_limit(self, manager):
        """Test daily execution limit enforcement."""
        policy = SecurityPolicy(max_daily_executions=2)
        manager.set_custom_policy("limited_tool", policy)
        
        # Simulate executions
        manager._increment_daily_count("limited_tool")
        manager._increment_daily_count("limited_tool")
        
        blocked, reason = manager.is_blocked("limited_tool")
        assert blocked is True
        assert "Daily execution limit reached" in reason
    
    def test_privacy_mode_strict(self):
        """Test strict privacy mode blocks dangerous tools."""
        manager = ToolSecurityManager(privacy_mode=PrivacyMode.STRICT)
        blocked, reason = manager.is_blocked("bash")
        assert blocked is True
        assert "strict privacy mode" in reason
    
    @pytest.mark.asyncio
    async def test_check_execution_allowed(self, manager):
        """Test execution check when allowed."""
        allowed, reason = await manager.check_execution(
            "read_file", {"path": "test.txt"}, user="test"
        )
        assert allowed is True
        assert reason is None
    
    @pytest.mark.asyncio
    async def test_check_execution_blocked(self, manager):
        """Test execution check when blocked."""
        manager.block_tool("blocked_tool")
        allowed, reason = await manager.check_execution(
            "blocked_tool", {}, user="test"
        )
        assert allowed is False
        assert "blocked" in reason
    
    @pytest.mark.asyncio
    async def test_check_execution_with_confirmation(self, manager):
        """Test execution check with confirmation."""
        callback = AsyncMock(return_value=ConfirmationResult.APPROVED)
        manager.set_confirmation_callback(callback)
        
        allowed, reason = await manager.check_execution(
            "write_file", {"path": "test.txt"}, user="test"
        )
        assert allowed is True
        callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_execution_denied_confirmation(self, manager):
        """Test execution check when confirmation denied."""
        callback = AsyncMock(return_value=ConfirmationResult.DENIED)
        manager.set_confirmation_callback(callback)
        
        allowed, reason = await manager.check_execution(
            "write_file", {"path": "test.txt"}, user="test"
        )
        assert allowed is False
        assert "denied" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_check_execution_timeout(self, manager):
        """Test execution check when confirmation times out."""
        callback = AsyncMock(return_value=ConfirmationResult.TIMEOUT)
        manager.set_confirmation_callback(callback)
        
        allowed, reason = await manager.check_execution(
            "write_file", {"path": "test.txt"}, user="test"
        )
        assert allowed is False
        assert "timeout" in reason.lower()
    
    def test_sanitize_params(self, manager):
        """Test parameter sanitization."""
        params = {
            "normal_key": "normal_value",
            "password": "secret123",
            "api_key": "sk-abcdef123456",
            "nested": {
                "token": "nested_secret",
                "safe": "value",
            },
        }
        sanitized = manager._sanitize_params(params)
        assert sanitized["normal_key"] == "normal_value"
        assert sanitized["password"] == "sec...123"
        assert sanitized["api_key"] == "sk-...456"
        assert sanitized["nested"]["token"] == "nes...ret"
        assert sanitized["nested"]["safe"] == "value"
    
    def test_get_execution_logs(self, manager):
        """Test retrieving execution logs."""
        # Add some logs with a small delay to ensure different timestamps
        manager._log_execution("tool1", {}, "user1", True, "Success")
        import time
        time.sleep(0.01)
        manager._log_execution("tool2", {}, "user2", False, "Failed")
        
        logs = manager.get_execution_logs(limit=10)
        assert len(logs) == 2
        # Logs should be sorted by timestamp in descending order (most recent first)
        assert logs[0]["tool_name"] == "tool2"  # Most recent first
        assert logs[1]["tool_name"] == "tool1"
    
    def test_get_execution_logs_with_filter(self, manager):
        """Test retrieving filtered execution logs."""
        manager._log_execution("tool1", {}, "user1", True, "Success")
        manager._log_execution("tool2", {}, "user1", False, "Failed")
        manager._log_execution("tool1", {}, "user2", True, "Success")
        
        logs = manager.get_execution_logs(tool_name="tool1")
        assert len(logs) == 2
        
        logs = manager.get_execution_logs(user="user1")
        assert len(logs) == 2


class TestWorkspaceBoundaryValidator:
    """Tests for WorkspaceBoundaryValidator class."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def validator(self, temp_workspace):
        """Create a validator with temp workspace."""
        config = WorkspaceProtectionConfig(
            allowed_workspaces=[temp_workspace],
            allow_symlinks=False,
        )
        return WorkspaceBoundaryValidator(config)
    
    def test_validate_path_within_workspace(self, validator, temp_workspace):
        """Test validating path within workspace."""
        test_file = Path(temp_workspace) / "test.txt"
        test_file.write_text("test")
        
        result = validator.validate_path(str(test_file), must_exist=True)
        assert result == test_file.resolve()
    
    def test_validate_path_outside_workspace(self, validator):
        """Test validating path outside workspace."""
        with pytest.raises(PermissionError) as exc_info:
            validator.validate_path("/etc/passwd")
        assert "outside allowed workspaces" in str(exc_info.value)
    
    def test_validate_path_traversal_attempt(self, validator, temp_workspace):
        """Test path traversal attack detection."""
        malicious_path = Path(temp_workspace) / ".." / "etc" / "passwd"
        with pytest.raises((PermissionError, ValueError)) as exc_info:
            validator.validate_path(str(malicious_path))
    
    def test_validate_path_with_dangerous_pattern(self, validator):
        """Test detection of dangerous path patterns."""
        with pytest.raises(ValueError) as exc_info:
            validator.validate_path("../../../etc/passwd")
        assert "dangerous pattern" in str(exc_info.value)
    
    def test_is_within_workspace(self, validator, temp_workspace):
        """Test workspace membership check."""
        assert validator.is_within_workspace(temp_workspace) is True
        assert validator.is_within_workspace("/etc") is False
    
    def test_validate_path_symlink(self, temp_workspace):
        """Test symlink handling."""
        config = WorkspaceProtectionConfig(
            allowed_workspaces=[temp_workspace],
            allow_symlinks=False,
        )
        validator = WorkspaceBoundaryValidator(config)
        
        # Create a symlink
        link_path = Path(temp_workspace) / "link"
        target_path = Path(temp_workspace) / "target"
        target_path.write_text("target")
        link_path.symlink_to(target_path)
        
        with pytest.raises(PermissionError) as exc_info:
            validator.validate_path(str(link_path))
        assert "Symbolic links are not allowed" in str(exc_info.value)
    
    def test_validate_path_max_depth(self, temp_workspace):
        """Test maximum path depth enforcement."""
        config = WorkspaceProtectionConfig(
            allowed_workspaces=[temp_workspace],
            max_path_depth=2,
        )
        validator = WorkspaceBoundaryValidator(config)
        
        # Create deep path
        deep_path = Path(temp_workspace) / "a" / "b" / "c" / "d"
        deep_path.mkdir(parents=True)
        
        with pytest.raises(PermissionError) as exc_info:
            validator.validate_path(str(deep_path))
        assert "exceeds maximum depth" in str(exc_info.value)


class TestDeletionProtector:
    """Tests for DeletionProtector class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def protector(self):
        """Create a deletion protector."""
        config = DeletionProtectionConfig(
            enabled=True,
            require_confirmation=True,
            backup_before_delete=True,
        )
        security_manager = ToolSecurityManager()
        return DeletionProtector(config, security_manager)
    
    def test_matches_pattern(self, protector):
        """Test pattern matching."""
        assert protector._matches_pattern("test.txt", ["*.txt"]) is True
        assert protector._matches_pattern("test.py", ["*.txt"]) is False
        assert protector._matches_pattern(".git/config", [".git/**"]) is True
    
    def test_can_delete_without_confirmation_whitelist(self, protector):
        """Test whitelist check."""
        protector._config.whitelist_patterns = ["*.tmp"]
        assert protector.can_delete_without_confirmation("test.tmp") is True
        assert protector.can_delete_without_confirmation("test.txt") is False
    
    def test_can_delete_without_confirmation_blacklist(self, protector):
        """Test blacklist check."""
        assert protector.can_delete_without_confirmation(".git/config") is False
        assert protector.can_delete_without_confirmation("path/to/.env") is False
    
    def test_is_blacklisted(self, protector):
        """Test blacklist check."""
        assert protector.is_blacklisted(".git/config") is True
        assert protector.is_blacklisted("normal_file.txt") is False
    
    @pytest.mark.asyncio
    async def test_request_deletion_confirmation_whitelisted(self, protector):
        """Test deletion confirmation for whitelisted files."""
        protector._config.whitelist_patterns = ["*.tmp"]
        result = await protector.request_deletion_confirmation(["test.tmp"])
        assert result == ConfirmationResult.APPROVED
    
    @pytest.mark.asyncio
    async def test_request_deletion_confirmation_no_callback(self, protector):
        """Test deletion confirmation without callback."""
        result = await protector.request_deletion_confirmation(["test.txt"])
        assert result == ConfirmationResult.DENIED
    
    @pytest.mark.asyncio
    async def test_request_deletion_confirmation_with_callback(self, protector):
        """Test deletion confirmation with callback."""
        callback = AsyncMock(return_value=ConfirmationResult.APPROVED)
        protector._security_manager.set_confirmation_callback(callback)
        
        result = await protector.request_deletion_confirmation(["test.txt"])
        assert result == ConfirmationResult.APPROVED
        callback.assert_called_once()
    
    def test_backup_file(self, protector, temp_dir):
        """Test file backup functionality."""
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")
        
        backup_path = protector.backup_file(str(test_file))
        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.read_text() == "test content"
    
    def test_backup_file_nonexistent(self, protector):
        """Test backup of nonexistent file."""
        backup_path = protector.backup_file("/nonexistent/file.txt")
        assert backup_path is None
    
    def test_cleanup_old_backups(self, protector, temp_dir):
        """Test backup cleanup."""
        # Create old backup
        protector._backup_dir = Path(temp_dir)
        old_backup = Path(temp_dir) / "old_backup.txt"
        old_backup.write_text("old")
        
        # Set modification time to old date
        old_time = datetime.now().timestamp() - (10 * 86400)  # 10 days ago
        os.utime(old_backup, (old_time, old_time))
        
        protector._config.backup_retention_days = 7
        cleaned = protector.cleanup_old_backups()
        assert cleaned == 1
        assert not old_backup.exists()


class TestSafeToolExecutor:
    """Tests for SafeToolExecutor class."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def executor(self, temp_workspace):
        """Create a safe tool executor."""
        return SafeToolExecutor(
            workspace_path=temp_workspace,
            privacy_mode=PrivacyMode.OFF,
            auto_confirm=True,
        )
    
    def test_initialization(self, executor, temp_workspace):
        """Test executor initialization."""
        assert executor.workspace_path == Path(temp_workspace).resolve()
        assert executor.session_id is not None
        assert executor.security_manager is not None
    
    def test_register_tool(self, executor):
        """Test tool registration."""
        def test_tool():
            return "result"
        
        executor.register_tool("test_tool", test_tool)
        assert "test_tool" in executor._tools
    
    def test_unregister_tool(self, executor):
        """Test tool unregistration."""
        def test_tool():
            return "result"
        
        executor.register_tool("test_tool", test_tool)
        executor.unregister_tool("test_tool")
        assert "test_tool" not in executor._tools
    
    @pytest.mark.asyncio
    async def test_execute_sync_tool(self, executor):
        """Test executing a synchronous tool."""
        def sync_tool(value):
            return f"result: {value}"
        
        executor.register_tool("sync_tool", sync_tool)
        result = await executor.execute("sync_tool", {"value": "test"})
        
        assert result["status"] == "success"
        assert result["result"] == "result: test"
        assert "execution_time_ms" in result
    
    @pytest.mark.asyncio
    async def test_execute_async_tool(self, executor):
        """Test executing an asynchronous tool."""
        async def async_tool(value):
            return f"async result: {value}"
        
        executor.register_tool("async_tool", async_tool)
        result = await executor.execute("async_tool", {"value": "test"})
        
        assert result["status"] == "success"
        assert result["result"] == "async result: test"
    
    @pytest.mark.asyncio
    async def test_execute_unregistered_tool(self, executor):
        """Test executing an unregistered tool."""
        with pytest.raises(ValueError) as exc_info:
            await executor.execute("unregistered", {})
        assert "Tool not registered" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_blocked_tool(self, executor):
        """Test executing a blocked tool."""
        def blocked_tool():
            return "result"
        
        executor.register_tool("blocked_tool", blocked_tool)
        executor.block_tool("blocked_tool")
        
        with pytest.raises(PermissionError) as exc_info:
            await executor.execute("blocked_tool", {})
        assert "blocked" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_execute_with_workspace_protection(self, executor):
        """Test workspace boundary protection."""
        def file_tool(path):
            return f"file: {path}"
        
        executor.register_tool("file_tool", file_tool)
        
        # Should succeed for path within workspace
        result = await executor.execute("file_tool", {"path": "test.txt"})
        assert result["status"] == "success"
        
        # Should fail for path outside workspace
        with pytest.raises(PermissionError):
            await executor.execute("file_tool", {"path": "/etc/passwd"})
    
    @pytest.mark.asyncio
    async def test_execute_with_deletion_protection(self, executor, temp_workspace):
        """Test deletion protection."""
        test_file = Path(temp_workspace) / "to_delete.txt"
        test_file.write_text("content")
        
        def delete_tool(path):
            Path(path).unlink()
            return "deleted"
        
        executor.register_tool("delete_file", delete_tool)
        
        result = await executor.execute("delete_file", {"path": str(test_file)})
        assert result["status"] == "success"
        assert not test_file.exists()
    
    def test_get_statistics(self, executor):
        """Test getting execution statistics."""
        stats = executor.get_statistics()
        assert "total_executions" in stats
        assert "successful_executions" in stats
        assert "failed_executions" in stats
        assert "denied_executions" in stats
        assert "by_tool" in stats
    
    def test_reset_statistics(self, executor):
        """Test resetting execution statistics."""
        # Modify stats
        executor._execution_stats["total_executions"] = 10
        
        executor.reset_statistics()
        stats = executor.get_statistics()
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
    
    def test_set_security_level(self, executor):
        """Test setting tool security level."""
        executor.set_security_level("custom_tool", ToolSecurityLevel.DANGEROUS)
        level = executor.security_manager.get_security_level("custom_tool")
        assert level == ToolSecurityLevel.DANGEROUS
    
    def test_get_execution_logs(self, executor):
        """Test getting execution logs."""
        logs = executor.get_execution_logs(limit=10)
        assert isinstance(logs, list)
    
    def test_session_id_generation(self, executor):
        """Test session ID format."""
        session_id = executor.session_id
        assert len(session_id) > 0
        assert "_" in session_id


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_security_manager(self):
        """Test creating security manager."""
        manager = create_security_manager(privacy_mode=PrivacyMode.LOCAL_ONLY)
        assert isinstance(manager, ToolSecurityManager)
        assert manager.privacy_mode == PrivacyMode.LOCAL_ONLY
    
    def test_create_safe_executor(self):
        """Test creating safe executor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = create_safe_executor(
                workspace_path=tmpdir,
                privacy_mode=PrivacyMode.STRICT,
            )
            assert isinstance(executor, SafeToolExecutor)
            assert executor.workspace_path == Path(tmpdir).resolve()


class TestIntegration:
    """Integration tests for the safe executor system."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow with multiple security features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create executor
            executor = SafeToolExecutor(
                workspace_path=tmpdir,
                privacy_mode=PrivacyMode.OFF,
                auto_confirm=True,
            )
            
            # Register tools
            def read_file(path):
                return Path(path).read_text()
            
            def write_file(path, content):
                Path(path).write_text(content)
                return "written"
            
            def delete_file(path):
                Path(path).unlink()
                return "deleted"
            
            executor.register_tool("read_file", read_file)
            executor.register_tool("write_file", write_file)
            executor.register_tool("delete_file", delete_file)
            
            # Execute workflow
            test_file = Path(tmpdir) / "test.txt"
            
            # Write file
            result = await executor.execute(
                "write_file",
                {"path": str(test_file), "content": "hello"}
            )
            assert result["status"] == "success"
            assert test_file.exists()
            
            # Read file
            result = await executor.execute("read_file", {"path": str(test_file)})
            assert result["status"] == "success"
            assert result["result"] == "hello"
            
            # Delete file
            result = await executor.execute("delete_file", {"path": str(test_file)})
            assert result["status"] == "success"
            assert not test_file.exists()
            
            # Check statistics
            stats = executor.get_statistics()
            assert stats["total_executions"] == 3
            assert stats["successful_executions"] == 3
            
            # Check logs - note that logs include both approval checks and execution results
            logs = executor.get_execution_logs(limit=10)
            # Each tool execution generates 2 log entries (approval check + execution result)
            # except for delete_file which also has a pre-execution check
            assert len(logs) >= 3
    
    @pytest.mark.asyncio
    async def test_privacy_mode_local_only(self):
        """Test LOCAL_ONLY privacy mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = SafeToolExecutor(
                workspace_path=tmpdir,
                privacy_mode=PrivacyMode.LOCAL_ONLY,
                auto_confirm=True,
            )
            
            def http_request(url):
                return f"response from {url}"
            
            executor.register_tool("http_request", http_request)
            
            # Should work in LOCAL_ONLY mode (only blocks network in STRICT)
            result = await executor.execute("http_request", {"url": "http://example.com"})
            assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_privacy_mode_strict(self):
        """Test STRICT privacy mode blocks dangerous operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = SafeToolExecutor(
                workspace_path=tmpdir,
                privacy_mode=PrivacyMode.STRICT,
                auto_confirm=True,
            )
            
            def bash_command(cmd):
                return f"executed: {cmd}"
            
            executor.register_tool("bash", bash_command)
            
            # Should be blocked in STRICT mode
            with pytest.raises(PermissionError) as exc_info:
                await executor.execute("bash", {"command": "ls"})
            assert "strict privacy mode" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
