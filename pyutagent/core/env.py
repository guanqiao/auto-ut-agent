"""Unified Environment Variable Handling.

This module provides centralized environment variable management:
- Consistent access patterns
- Type conversion
- Default values
- Secret masking
- Environment detection

This is part of Phase 2 Week 13-14: Configuration Management Refactoring.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Environment(Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EnvVarSpec:
    """Specification for an environment variable."""
    name: str
    aliases: List[str]
    default: Any
    value_type: Type
    description: str
    secret: bool = False
    required: bool = False


class EnvConfig:
    """Unified environment variable configuration.
    
    Features:
    - Centralized environment variable access
    - Type conversion
    - Multiple alias support
    - Secret masking
    - Environment detection
    
    Usage:
        env = EnvConfig.get_instance()
        
        # Get environment variable
        api_key = env.get("API_KEY", default="")
        
        # Get typed value
        timeout = env.get_int("TIMEOUT", default=30)
        
        # Check environment
        if env.is_production():
            ...
    """
    
    _instance: Optional["EnvConfig"] = None
    
    def __init__(
        self,
        prefix: str = "PYUT_",
        env_var: str = "PYUT_ENV",
    ):
        """Initialize environment configuration.
        
        Args:
            prefix: Environment variable prefix
            env_var: Variable name for environment detection
        """
        self._prefix = prefix
        self._env_var = env_var
        self._specs: Dict[str, EnvVarSpec] = {}
        self._cache: Dict[str, Any] = {}
        self._secrets: set = set()
    
    @classmethod
    def get_instance(cls) -> "EnvConfig":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None
    
    def register(
        self,
        name: str,
        default: Any = None,
        value_type: Type = str,
        description: str = "",
        aliases: Optional[List[str]] = None,
        secret: bool = False,
        required: bool = False,
    ) -> None:
        """Register an environment variable specification.
        
        Args:
            name: Variable name (without prefix)
            default: Default value
            value_type: Expected type
            description: Variable description
            aliases: Alternative names to check
            secret: Whether this is a secret value
            required: Whether this is required
        """
        spec = EnvVarSpec(
            name=name,
            aliases=aliases or [],
            default=default,
            value_type=value_type,
            description=description,
            secret=secret,
            required=required,
        )
        
        self._specs[name] = spec
        
        if secret:
            self._secrets.add(name)
    
    def _resolve_name(self, name: str) -> Optional[str]:
        """Resolve environment variable name with prefix and aliases.
        
        Args:
            name: Variable name
            
        Returns:
            Resolved environment variable name or None
        """
        names_to_check = [
            f"{self._prefix}{name}",
            name,
        ]
        
        if name in self._specs:
            spec = self._specs[name]
            for alias in spec.aliases:
                names_to_check.extend([
                    f"{self._prefix}{alias}",
                    alias,
                ])
        
        for var_name in names_to_check:
            if var_name in os.environ:
                return var_name
        
        return None
    
    def get(
        self,
        name: str,
        default: Any = None,
    ) -> Any:
        """Get environment variable value.
        
        Args:
            name: Variable name
            default: Default value if not found
            
        Returns:
            Environment variable value
        """
        if name in self._cache:
            return self._cache[name]
        
        resolved = self._resolve_name(name)
        
        if resolved:
            value = os.environ[resolved]
        elif name in self._specs:
            value = self._specs[name].default
        else:
            value = default
        
        if name in self._specs:
            value = self._convert_type(value, self._specs[name].value_type)
        
        self._cache[name] = value
        return value
    
    def _convert_type(
        self,
        value: Any,
        target_type: Type,
    ) -> Any:
        """Convert value to target type.
        
        Args:
            value: Value to convert
            target_type: Target type
            
        Returns:
            Converted value
        """
        if value is None:
            return None
        
        if isinstance(value, target_type):
            return value
        
        try:
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            
            if target_type == Path:
                return Path(value)
            
            if target_type == list:
                if isinstance(value, str):
                    return [item.strip() for item in value.split(",")]
                return list(value)
            
            return target_type(value)
            
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to convert {value} to {target_type}: {e}")
            return value
    
    def get_str(
        self,
        name: str,
        default: str = "",
    ) -> str:
        """Get string environment variable.
        
        Args:
            name: Variable name
            default: Default value
            
        Returns:
            String value
        """
        return str(self.get(name, default))
    
    def get_int(
        self,
        name: str,
        default: int = 0,
    ) -> int:
        """Get integer environment variable.
        
        Args:
            name: Variable name
            default: Default value
            
        Returns:
            Integer value
        """
        value = self.get(name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    
    def get_float(
        self,
        name: str,
        default: float = 0.0,
    ) -> float:
        """Get float environment variable.
        
        Args:
            name: Variable name
            default: Default value
            
        Returns:
            Float value
        """
        value = self.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def get_bool(
        self,
        name: str,
        default: bool = False,
    ) -> bool:
        """Get boolean environment variable.
        
        Args:
            name: Variable name
            default: Default value
            
        Returns:
            Boolean value
        """
        value = self.get(name, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    
    def get_path(
        self,
        name: str,
        default: Optional[Path] = None,
    ) -> Optional[Path]:
        """Get path environment variable.
        
        Args:
            name: Variable name
            default: Default value
            
        Returns:
            Path value
        """
        value = self.get(name)
        if value is None:
            return default
        return Path(value)
    
    def get_list(
        self,
        name: str,
        default: Optional[List[str]] = None,
        separator: str = ",",
    ) -> List[str]:
        """Get list environment variable (comma-separated).
        
        Args:
            name: Variable name
            default: Default value
            separator: Value separator
            
        Returns:
            List of strings
        """
        value = self.get(name)
        if value is None:
            return default or []
        
        if isinstance(value, list):
            return value
        
        if isinstance(value, str):
            return [item.strip() for item in value.split(separator)]
        
        return default or []
    
    def is_set(self, name: str) -> bool:
        """Check if environment variable is set.
        
        Args:
            name: Variable name
            
        Returns:
            True if set
        """
        return self._resolve_name(name) is not None
    
    def get_environment(self) -> Environment:
        """Get current environment.
        
        Returns:
            Environment enum value
        """
        env_value = self.get(self._env_var, "development").lower()
        
        try:
            return Environment(env_value)
        except ValueError:
            return Environment.DEVELOPMENT
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.get_environment() == Environment.DEVELOPMENT
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.get_environment() == Environment.TESTING
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.get_environment() == Environment.STAGING
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment() == Environment.PRODUCTION
    
    def is_secret(self, name: str) -> bool:
        """Check if variable is marked as secret.
        
        Args:
            name: Variable name
            
        Returns:
            True if secret
        """
        return name in self._secrets
    
    def mask_value(self, name: str, value: str) -> str:
        """Mask a secret value for logging.
        
        Args:
            name: Variable name
            value: Value to mask
            
        Returns:
            Masked value
        """
        if not self.is_secret(name):
            return value
        
        if len(value) <= 8:
            return "***"
        
        return value[:4] + "***" + value[-4:]
    
    def get_masked(self, name: str) -> str:
        """Get environment variable with masking for secrets.
        
        Args:
            name: Variable name
            
        Returns:
            Masked value if secret, otherwise raw value
        """
        value = self.get(name, "")
        if isinstance(value, str):
            return self.mask_value(name, value)
        return str(value)
    
    def clear_cache(self) -> None:
        """Clear the value cache."""
        self._cache.clear()
    
    def get_all_registered(self) -> Dict[str, EnvVarSpec]:
        """Get all registered specifications.
        
        Returns:
            Dictionary of specifications
        """
        return self._specs.copy()
    
    def validate_required(self) -> List[str]:
        """Validate that all required variables are set.
        
        Returns:
            List of missing required variables
        """
        missing = []
        
        for name, spec in self._specs.items():
            if spec.required and not self.is_set(name):
                missing.append(name)
        
        return missing


def get_env_config() -> EnvConfig:
    """Get the global environment configuration instance."""
    return EnvConfig.get_instance()


def env_value(
    name: str,
    default: Any = None,
) -> Any:
    """Get environment variable value from global config.
    
    Args:
        name: Variable name
        default: Default value
        
    Returns:
        Environment variable value
    """
    return get_env_config().get(name, default)


def env_int(name: str, default: int = 0) -> int:
    """Get integer environment variable."""
    return get_env_config().get_int(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return get_env_config().get_bool(name, default)


def env_path(name: str, default: Optional[Path] = None) -> Optional[Path]:
    """Get path environment variable."""
    return get_env_config().get_path(name, default)


def register_env_var(
    name: str,
    default: Any = None,
    value_type: Type = str,
    description: str = "",
    aliases: Optional[List[str]] = None,
    secret: bool = False,
    required: bool = False,
) -> None:
    """Register an environment variable specification.
    
    Args:
        name: Variable name
        default: Default value
        value_type: Expected type
        description: Variable description
        aliases: Alternative names
        secret: Whether this is a secret
        required: Whether this is required
    """
    get_env_config().register(
        name=name,
        default=default,
        value_type=value_type,
        description=description,
        aliases=aliases,
        secret=secret,
        required=required,
    )


DEFAULT_ENV_SPECS = [
    ("API_KEY", "", str, "API key for LLM", None, True, False),
    ("MODEL", "gpt-4", str, "LLM model name", None, False, False),
    ("MAX_TOKENS", 4096, int, "Maximum tokens", None, False, False),
    ("TEMPERATURE", 0.7, float, "Sampling temperature", None, False, False),
    ("TIMEOUT", 300, int, "Request timeout in seconds", None, False, False),
    ("LOG_LEVEL", "INFO", str, "Logging level", None, False, False),
    ("DEBUG", False, bool, "Enable debug mode", None, False, False),
    ("JAVA_HOME", "", str, "Java home directory", None, False, False),
    ("MAVEN_HOME", "", str, "Maven home directory", ["M3_HOME"], False, False),
    ("DATA_DIR", "", str, "Data directory", None, False, False),
]


def init_default_env_specs() -> None:
    """Initialize default environment variable specifications."""
    env = get_env_config()
    
    for spec in DEFAULT_ENV_SPECS:
        env.register(
            name=spec[0],
            default=spec[1],
            value_type=spec[2],
            description=spec[3],
            aliases=spec[4],
            secret=spec[5],
            required=spec[6],
        )
