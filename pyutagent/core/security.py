"""Security utilities for PyUT Agent.

This module provides security-related utilities including:
- Path validation to prevent directory traversal
- Sensitive data masking for logging
- Command injection prevention
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related error."""
    pass


class PathValidator:
    """Validates file paths for security."""
    
    # Dangerous patterns that could indicate path traversal attempts
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Unix parent directory
        r'\.\.\\',  # Windows parent directory
        r'~',  # Home directory expansion
        r'%',  # Environment variable expansion (Windows)
        r'\$\w+',  # Environment variable expansion (Unix)
    ]
    
    def __init__(self, allowed_base_paths: Optional[List[str]] = None):
        """Initialize path validator.
        
        Args:
            allowed_base_paths: List of allowed base paths. If None, all paths are allowed.
        """
        self.allowed_base_paths = allowed_base_paths
        if allowed_base_paths:
            self._allowed_paths = [Path(p).resolve() for p in allowed_base_paths]
        else:
            self._allowed_paths = None
    
    def validate(self, path: str, must_exist: bool = False) -> Path:
        """Validate a file path.
        
        Args:
            path: Path to validate
            must_exist: Whether the path must exist
            
        Returns:
            Resolved Path object
            
        Raises:
            SecurityError: If path is invalid or outside allowed base paths
        """
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, path):
                raise SecurityError(f"Path contains dangerous pattern: {pattern}")
        
        try:
            resolved = Path(path).resolve()
        except Exception as e:
            raise SecurityError(f"Invalid path: {path}") from e
        
        # Check if path is within allowed base paths
        if self._allowed_paths is not None:
            allowed = any(
                str(resolved).startswith(str(allowed_path))
                for allowed_path in self._allowed_paths
            )
            if not allowed:
                raise SecurityError(
                    f"Path {path} is outside allowed directories: "
                    f"{self.allowed_base_paths}"
                )
        
        if must_exist and not resolved.exists():
            raise SecurityError(f"Path does not exist: {path}")
        
        return resolved
    
    def validate_safe_for_command(self, path: str) -> str:
        """Validate path is safe to use in shell commands.
        
        Args:
            path: Path to validate
            
        Returns:
            Validated path string
            
        Raises:
            SecurityError: If path contains shell metacharacters
        """
        # Check for shell metacharacters
        shell_metachars = set(';|&$`<>{}[]\\!*?#')
        if any(c in path for c in shell_metachars):
            raise SecurityError(f"Path contains shell metacharacters: {path}")
        
        # Validate the path
        self.validate(path)
        
        return path


@dataclass
class SensitiveField:
    """Configuration for a sensitive field."""
    name: str
    mask: str = "***"
    partial_mask: bool = False  # If True, show first/last few chars
    visible_chars: int = 3


class SecureLogger:
    """Logger wrapper that automatically masks sensitive data."""
    
    # Default sensitive fields
    DEFAULT_SENSITIVE_FIELDS: Set[str] = {
        'api_key', 'apikey', 'api-key',
        'password', 'passwd', 'pwd',
        'secret', 'secret_key', 'secretkey',
        'token', 'access_token', 'refresh_token',
        'private_key', 'privatekey',
        'credential', 'credentials',
        'auth', 'authorization',
    }
    
    def __init__(
        self,
        logger_instance: Optional[logging.Logger] = None,
        sensitive_fields: Optional[Set[str]] = None
    ):
        """Initialize secure logger.
        
        Args:
            logger_instance: Logger instance to wrap. If None, uses module logger.
            sensitive_fields: Additional sensitive field names
        """
        self._logger = logger_instance or logger
        self._sensitive_fields = self.DEFAULT_SENSITIVE_FIELDS.copy()
        if sensitive_fields:
            self._sensitive_fields.update(
                f.lower() for f in sensitive_fields
            )
    
    def _sanitize(self, obj: Any) -> Any:
        """Sanitize object by masking sensitive fields.
        
        Args:
            obj: Object to sanitize
            
        Returns:
            Sanitized copy of the object
        """
        if isinstance(obj, dict):
            return {
                k: self._mask_value(k, self._sanitize(v))
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize(item) for item in obj]
        elif isinstance(obj, str):
            return self._sanitize_string(obj)
        return obj
    
    def _mask_value(self, key: str, value: Any) -> Any:
        """Mask value if key is sensitive.
        
        Args:
            key: Field name
            value: Field value
            
        Returns:
            Masked value if sensitive, original value otherwise
        """
        if not isinstance(value, str):
            return value
        
        key_lower = key.lower()
        if any(field in key_lower for field in self._sensitive_fields):
            if len(value) <= 8:
                return "***"
            else:
                # Show first 3 and last 3 characters
                return f"{value[:3]}...{value[-3:]}"
        return value
    
    def _sanitize_string(self, s: str) -> str:
        """Sanitize a string for logging.
        
        Args:
            s: String to sanitize
            
        Returns:
            Sanitized string
        """
        # Mask potential API keys in strings
        # Pattern: common API key formats
        patterns = [
            (r'sk-[a-zA-Z0-9]{20,}', 'sk-***'),  # OpenAI
            (r'Bearer\s+[a-zA-Z0-9\-_]+', 'Bearer ***'),  # Bearer tokens
            (r'api[_-]?key[=:]\s*[a-zA-Z0-9\-_]+', 'api_key=***'),  # API key assignments
        ]
        
        result = s
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with sanitization."""
        self._logger.debug(self._sanitize(msg), *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message with sanitization."""
        self._logger.info(self._sanitize(msg), *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with sanitization."""
        self._logger.warning(self._sanitize(msg), *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message with sanitization."""
        self._logger.error(self._sanitize(msg), *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception message with sanitization."""
        self._logger.exception(self._sanitize(msg), *args, **kwargs)
    
    def log(self, level: int, msg: str, *args, **kwargs):
        """Log message at specified level with sanitization."""
        self._logger.log(level, self._sanitize(msg), *args, **kwargs)


class CommandBuilder:
    """Builds shell commands safely."""
    
    @staticmethod
    def build_command(
        executable: str,
        *args: str,
        **kwargs: str
    ) -> List[str]:
        """Build a command with arguments safely.
        
        Args:
            executable: Executable name
            *args: Positional arguments
            **kwargs: Keyword arguments (will be formatted as --key value)
            
        Returns:
            Command as list of strings
        """
        cmd = [executable]
        
        # Add positional arguments
        for arg in args:
            if not isinstance(arg, str):
                arg = str(arg)
            # Validate argument doesn't contain shell metacharacters
            CommandBuilder._validate_argument(arg)
            cmd.append(arg)
        
        # Add keyword arguments
        for key, value in kwargs.items():
            if not isinstance(value, str):
                value = str(value)
            CommandBuilder._validate_argument(value)
            cmd.append(f"--{key}")
            cmd.append(value)
        
        return cmd
    
    @staticmethod
    def _validate_argument(arg: str) -> None:
        """Validate command argument is safe.
        
        Args:
            arg: Argument to validate
            
        Raises:
            SecurityError: If argument contains dangerous characters
        """
        # Check for null bytes
        if '\x00' in arg:
            raise SecurityError("Argument contains null byte")
        
        # Check for shell metacharacters
        dangerous = set(';&|`$(){}[]<>!#*?\\\n\r')
        found = dangerous.intersection(set(arg))
        if found:
            raise SecurityError(
                f"Argument contains dangerous characters: {found}"
            )


def sanitize_for_display(
    text: str,
    max_length: int = 1000,
    show_length: int = 100
) -> str:
    """Sanitize text for display in UI.
    
    Args:
        text: Text to sanitize
        max_length: Maximum length before truncation
        show_length: Length to show when truncated
        
    Returns:
        Sanitized text
    """
    if len(text) > max_length:
        return text[:show_length] + f"... ({len(text) - show_length} more chars)"
    return text


def validate_maven_project(project_path: str) -> Path:
    """Validate path is a valid Maven project.
    
    Args:
        project_path: Path to validate
        
    Returns:
        Resolved path
        
    Raises:
        SecurityError: If path is invalid or not a Maven project
    """
    validator = PathValidator()
    path = validator.validate(project_path, must_exist=True)
    
    # Check for pom.xml
    pom_file = path / "pom.xml"
    if not pom_file.exists():
        raise SecurityError(f"No pom.xml found in {project_path}")
    
    return path
