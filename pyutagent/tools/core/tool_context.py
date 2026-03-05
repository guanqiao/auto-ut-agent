"""Tool Context Definition.

This module provides execution context for tools.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging


@dataclass
class ToolContext:
    """Execution context for tools.
    
    Provides:
    - Project paths
    - Configuration
    - Environment
    - Working directory
    - Timeout settings
    - Logger
    """
    
    project_path: Path
    working_dir: Optional[Path] = None
    
    config: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    
    timeout: float = 60.0
    max_retries: int = 3
    
    logger: Optional[logging.Logger] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    _created_at: datetime = field(default_factory=datetime.now)
    _modified_files: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize after creation."""
        if isinstance(self.project_path, str):
            self.project_path = Path(self.project_path)
        
        if self.working_dir is None:
            self.working_dir = self.project_path
        
        if isinstance(self.working_dir, str):
            self.working_dir = Path(self.working_dir)
        
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
    
    @property
    def project_root(self) -> Path:
        """Get project root path."""
        return self.project_path
    
    @property
    def cwd(self) -> Path:
        """Get current working directory."""
        return self.working_dir or self.project_path
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable.
        
        Args:
            key: Environment variable name
            default: Default value
            
        Returns:
            Environment value or default
        """
        return self.env.get(key, default)
    
    def set_env(self, key: str, value: str) -> None:
        """Set environment variable.
        
        Args:
            key: Environment variable name
            value: Value to set
        """
        self.env[key] = value
    
    def track_modified_file(self, file_path: str | Path) -> None:
        """Track a modified file.
        
        Args:
            file_path: Path to modified file
        """
        self._modified_files.add(str(file_path))
    
    def get_modified_files(self) -> List[str]:
        """Get list of modified files.
        
        Returns:
            List of modified file paths
        """
        return list(self._modified_files)
    
    def clear_modified_files(self) -> None:
        """Clear list of modified files."""
        self._modified_files.clear()
    
    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to project root.
        
        Args:
            path: Path to resolve
            
        Returns:
            Absolute path
        """
        path = Path(path)
        if path.is_absolute():
            return path
        return self.project_path / path
    
    def relative_path(self, path: str | Path) -> Path:
        """Get path relative to project root.
        
        Args:
            path: Absolute path
            
        Returns:
            Relative path
        """
        path = Path(path)
        try:
            return path.relative_to(self.project_path)
        except ValueError:
            return path
    
    def log(self, level: int, message: str, **kwargs) -> None:
        """Log a message.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional context
        """
        if self.logger:
            self.logger.log(level, message, extra=kwargs)
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log(logging.INFO, message, **kwargs)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log(logging.WARNING, message, **kwargs)
    
    def log_error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log(logging.ERROR, message, **kwargs)
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log(logging.DEBUG, message, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_path": str(self.project_path),
            "working_dir": str(self.working_dir) if self.working_dir else None,
            "config": self.config,
            "env": self.env,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolContext":
        """Create from dictionary."""
        return cls(
            project_path=Path(data.get("project_path", ".")),
            working_dir=Path(data["working_dir"]) if data.get("working_dir") else None,
            config=data.get("config", {}),
            env=data.get("env", {}),
            timeout=data.get("timeout", 60.0),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {}),
        )
    
    def create_child(self, **kwargs) -> "ToolContext":
        """Create a child context with inherited values.
        
        Args:
            **kwargs: Values to override
            
        Returns:
            New ToolContext instance
        """
        return ToolContext(
            project_path=kwargs.get("project_path", self.project_path),
            working_dir=kwargs.get("working_dir", self.working_dir),
            config={**self.config, **kwargs.get("config", {})},
            env={**self.env, **kwargs.get("env", {})},
            timeout=kwargs.get("timeout", self.timeout),
            max_retries=kwargs.get("max_retries", self.max_retries),
            metadata={**self.metadata, **kwargs.get("metadata", {})},
        )
