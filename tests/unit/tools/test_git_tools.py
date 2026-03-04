"""Tests for Git tools."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from pyutagent.tools.git_tools import (
    GitStatusTool,
    GitDiffTool,
    GitCommitTool,
    GitBranchTool,
    GitLogTool
)
from pyutagent.tools.tool import ToolResult


class TestGitStatusTool:
    """Test GitStatusTool."""
    
    @pytest.fixture
    def tool(self):
        """Create GitStatusTool instance."""
        return GitStatusTool(base_path="/tmp/test")
    
    def test_tool_initialization(self, tool):
        """Test tool initialization."""
        assert tool.definition.name == "git_status"
        from pyutagent.tools.tool import ToolCategory
        assert tool.definition.category == ToolCategory.COMMAND
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        """Test successful git status execution."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"M modified_file.java\n?? untracked.txt",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute()
        
        assert result.success is True
        assert "modified" in result.output
        assert result.metadata["has_changes"] is True
    
    @pytest.mark.asyncio
    async def test_execute_no_changes(self, tool):
        """Test git status with no changes."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute()
        
        assert result.success is True
        assert result.metadata["has_changes"] is False
    
    @pytest.mark.asyncio
    async def test_execute_git_not_found(self, tool):
        """Test git not found error."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            result = await tool.execute()
        
        assert result.success is False
        assert "Git not found" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_git_error(self, tool):
        """Test git command error."""
        mock_process = Mock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Not a git repository"))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute()
        
        assert result.success is False
        assert "Git status failed" in result.error
    
    def test_parse_status_short_format(self, tool):
        """Test parsing short format status."""
        # Git short format: XY filename
        # X = index status (staged), Y = working tree status (modified)
        output = "M  staged_file.java\n M modified.java\n?? untracked.txt\nA  added.java"
        result = tool._parse_status(output, short=True)
        
        # M  = staged (M in first column)
        assert "staged_file.java" in result["staged"]
        #  M = modified (M in second column)
        assert "modified.java" in result["modified"]
        # ?? = untracked
        assert "untracked.txt" in result["untracked"]
        # A  = added to staged
        assert "added.java" in result["staged"]


class TestGitDiffTool:
    """Test GitDiffTool."""
    
    @pytest.fixture
    def tool(self):
        """Create GitDiffTool instance."""
        return GitDiffTool(base_path="/tmp/test")
    
    def test_tool_initialization(self, tool):
        """Test tool initialization."""
        assert tool.definition.name == "git_diff"
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        """Test successful git diff execution."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"diff --git a/file.java b/file.java\n+added line",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute()
        
        assert result.success is True
        assert result.metadata["has_changes"] is True
        assert result.metadata["stats"]["files_changed"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_staged(self, tool):
        """Test git diff with staged changes."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"staged diff", b""))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await tool.execute(staged=True)
            
            # Verify --cached flag was used
            call_args = mock_exec.call_args[0]
            assert "--cached" in call_args
    
    def test_parse_diff_stats(self, tool):
        """Test parsing diff statistics."""
        diff_output = """diff --git a/file.java b/file.java
--- a/file.java
+++ b/file.java
@@ -1 +1,2 @@
 line1
+line2
-line3
diff --git a/other.java b/other.java
--- a/other.java
+++ b/other.java
@@ -1 +1 @@
-old
+new"""
        
        stats = tool._parse_diff_stats(diff_output)
        
        assert stats["files_changed"] == 2
        assert stats["insertions"] == 2
        assert stats["deletions"] == 2


class TestGitCommitTool:
    """Test GitCommitTool."""
    
    @pytest.fixture
    def tool(self):
        """Create GitCommitTool instance."""
        return GitCommitTool(base_path="/tmp/test")
    
    def test_tool_initialization(self, tool):
        """Test tool initialization."""
        assert tool.definition.name == "git_commit"
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        """Test successful git commit."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"[main abc1234] Test commit",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute(message="Test commit")
        
        assert result.success is True
        assert result.metadata["message"] == "Test commit"
        assert result.metadata["commit_hash"] == "abc1234"
    
    @pytest.mark.asyncio
    async def test_execute_with_add_all(self, tool):
        """Test git commit with add_all flag."""
        mock_add_process = Mock()
        mock_add_process.returncode = 0
        mock_add_process.communicate = AsyncMock(return_value=(b"", b""))
        
        mock_commit_process = Mock()
        mock_commit_process.returncode = 0
        mock_commit_process.communicate = AsyncMock(return_value=(
            b"[main abc1234] Test commit",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", side_effect=[
            mock_add_process,
            mock_commit_process
        ]) as mock_exec:
            result = await tool.execute(message="Test commit", add_all=True)
            
            # Verify git add was called
            assert mock_exec.call_count == 2
            first_call = mock_exec.call_args_list[0][0]
            assert "add" in first_call
    
    @pytest.mark.asyncio
    async def test_execute_missing_message(self, tool):
        """Test git commit without message."""
        result = await tool.execute()
        
        assert result.success is False
        assert "Commit message is required" in result.error
    
    def test_extract_commit_hash(self, tool):
        """Test extracting commit hash from output."""
        output = "[main abc1234] Test commit message"
        hash_value = tool._extract_commit_hash(output)
        
        assert hash_value == "abc1234"
    
    def test_extract_commit_hash_not_found(self, tool):
        """Test extracting commit hash when not found."""
        output = "Some other output"
        hash_value = tool._extract_commit_hash(output)
        
        assert hash_value is None


class TestGitBranchTool:
    """Test GitBranchTool."""
    
    @pytest.fixture
    def tool(self):
        """Create GitBranchTool instance."""
        return GitBranchTool(base_path="/tmp/test")
    
    def test_tool_initialization(self, tool):
        """Test tool initialization."""
        assert tool.definition.name == "git_branch"
    
    @pytest.mark.asyncio
    async def test_list_branches(self, tool):
        """Test listing branches."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"* main\n  develop\n  feature/test",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute(action="list")
        
        assert result.success is True
        assert result.metadata["current_branch"] == "main"
        assert len(result.metadata["branches"]) == 3
    
    @pytest.mark.asyncio
    async def test_create_branch(self, tool):
        """Test creating a branch."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"Switched to a new branch 'feature-x'",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await tool.execute(action="create", branch_name="feature-x")
            
            # Verify checkout -b was used
            call_args = mock_exec.call_args[0]
            assert "checkout" in call_args
            assert "-b" in call_args
            assert "feature-x" in call_args
        
        assert result.success is True
        assert result.metadata["branch_name"] == "feature-x"
    
    @pytest.mark.asyncio
    async def test_delete_branch(self, tool):
        """Test deleting a branch."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"Deleted branch feature-x",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await tool.execute(action="delete", branch_name="feature-x")
            
            # Verify branch -d was used
            call_args = mock_exec.call_args[0]
            assert "branch" in call_args
            assert "-d" in call_args
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_switch_branch(self, tool):
        """Test switching branches."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"Switched to branch 'develop'",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await tool.execute(action="switch", branch_name="develop")
            
            # Verify checkout was used
            call_args = mock_exec.call_args[0]
            assert "checkout" in call_args
            assert "develop" in call_args
        
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_create_branch_missing_name(self, tool):
        """Test creating branch without name."""
        result = await tool.execute(action="create")
        
        assert result.success is False
        assert "Branch name required" in result.error
    
    @pytest.mark.asyncio
    async def test_unknown_action(self, tool):
        """Test unknown action."""
        result = await tool.execute(action="unknown")
        
        assert result.success is False
        assert "Unknown action" in result.error


class TestGitLogTool:
    """Test GitLogTool."""
    
    @pytest.fixture
    def tool(self):
        """Create GitLogTool instance."""
        return GitLogTool(base_path="/tmp/test")
    
    def test_tool_initialization(self, tool):
        """Test tool initialization."""
        assert tool.definition.name == "git_log"
    
    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        """Test successful git log execution."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(
            b"abc1234 First commit\ndef5678 Second commit",
            b""
        ))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute(max_count=5)
        
        assert result.success is True
        assert len(result.metadata["commits"]) == 2
        assert result.metadata["count"] == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_file_path(self, tool):
        """Test git log for specific file."""
        mock_process = Mock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"abc1234 Commit", b""))
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            result = await tool.execute(file_path="src/main.java")
            
            # Verify file path was included
            call_args = mock_exec.call_args[0]
            assert "src/main.java" in call_args
        
        assert result.success is True
    
    def test_parse_commits_oneline(self, tool):
        """Test parsing oneline format commits."""
        output = """abc1234 First commit message
def5678 Second commit message
ghi9012 Third commit message"""
        
        commits = tool._parse_commits(output, oneline=True)
        
        assert len(commits) == 3
        assert commits[0]["hash"] == "abc1234"
        assert commits[0]["message"] == "First commit message"
        assert commits[1]["hash"] == "def5678"
        assert commits[1]["message"] == "Second commit message"
    
    def test_parse_commits_empty(self, tool):
        """Test parsing empty commit log."""
        commits = tool._parse_commits("", oneline=True)
        
        assert len(commits) == 0
