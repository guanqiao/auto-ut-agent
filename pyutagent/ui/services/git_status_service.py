"""Git status detection service for file tree."""

import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GitStatus(Enum):
    """Git file status enumeration."""
    UNMODIFIED = " "
    MODIFIED = "M"
    ADDED = "A"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UPDATED = "U"
    UNTRACKED = "?"
    IGNORED = "!"
    CONFLICTED = "C"
    
    @property
    def color(self) -> str:
        """Get status color."""
        colors = {
            GitStatus.MODIFIED: "#E2C08D",  # Yellow/Orange
            GitStatus.ADDED: "#73C991",     # Green
            GitStatus.DELETED: "#F85149",   # Red
            GitStatus.UNTRACKED: "#73C991", # Green
            GitStatus.RENAMED: "#E2C08D",   # Yellow
            GitStatus.CONFLICTED: "#F85149", # Red
        }
        return colors.get(self, "#CCCCCC")
    
    @property
    def icon(self) -> str:
        """Get status icon."""
        icons = {
            GitStatus.MODIFIED: "M",
            GitStatus.ADDED: "A",
            GitStatus.DELETED: "D",
            GitStatus.UNTRACKED: "U",
            GitStatus.RENAMED: "R",
            GitStatus.CONFLICTED: "C",
        }
        return icons.get(self, "")
    
    @property
    def display_name(self) -> str:
        """Get display name."""
        names = {
            GitStatus.MODIFIED: "Modified",
            GitStatus.ADDED: "Added",
            GitStatus.DELETED: "Deleted",
            GitStatus.UNTRACKED: "Untracked",
            GitStatus.RENAMED: "Renamed",
            GitStatus.CONFLICTED: "Conflicted",
        }
        return names.get(self, "")


class GitStatusService:
    """Service for detecting Git status of files."""
    
    def __init__(self):
        self._status_cache: Dict[str, GitStatus] = {}
        self._repo_root: Optional[str] = None
        
    def detect_repo(self, project_path: str) -> bool:
        """Detect if path is in a Git repository.
        
        Args:
            project_path: Path to check
            
        Returns:
            True if in a Git repository
        """
        try:
            result = subprocess.run(
                ["git", "-C", project_path, "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self._repo_root = result.stdout.strip()
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"Git not available or timeout: {e}")
        return False
    
    def get_file_status(self, file_path: str) -> Optional[GitStatus]:
        """Get Git status for a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            GitStatus or None if not in repo
        """
        if file_path in self._status_cache:
            return self._status_cache[file_path]
            
        if not self._repo_root:
            return None
            
        try:
            result = subprocess.run(
                ["git", "-C", self._repo_root, "status", "--porcelain", file_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout:
                line = result.stdout.strip()
                if line:
                    status_code = line[0] if len(line) > 0 else " "
                    for status in GitStatus:
                        if status.value == status_code:
                            self._status_cache[file_path] = status
                            return status
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"Failed to get git status: {e}")
        return None
    
    def refresh_all_status(self) -> Dict[str, GitStatus]:
        """Refresh status for all files in the repository.
        
        Returns:
            Dictionary mapping file paths to their GitStatus
        """
        self._status_cache.clear()
        
        if not self._repo_root:
            return self._status_cache
            
        try:
            result = subprocess.run(
                ["git", "-C", self._repo_root, "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if len(line) >= 3:
                        status_code = line[0] if line[0] != " " else line[1]
                        file_path = line[3:].strip()
                        
                        # Handle renamed files (R old -> new)
                        if " -> " in file_path:
                            file_path = file_path.split(" -> ")[1]
                        
                        full_path = str(Path(self._repo_root) / file_path)
                        
                        for status in GitStatus:
                            if status.value == status_code:
                                self._status_cache[full_path] = status
                                break
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug(f"Failed to refresh git status: {e}")
            
        return self._status_cache
    
    def get_repo_root(self) -> Optional[str]:
        """Get the repository root path."""
        return self._repo_root
    
    def clear_cache(self):
        """Clear the status cache."""
        self._status_cache.clear()
