"""Configuration Management System.

This module provides:
- Configuration validation
- Configuration templates
- Environment-based config
- Secure config handling
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ConfigSource(Enum):
    """Configuration source."""
    DEFAULT = auto()
    ENVIRONMENT = auto()
    FILE = auto()
    COMMAND_LINE = auto()
    MEMORY = auto()


class ConfigPriority(Enum):
    """Configuration priority (higher overrides lower)."""
    DEFAULT = 1
    ENVIRONMENT = 2
    FILE = 3
    COMMAND_LINE = 4
    MEMORY = 5


@dataclass
class ConfigEntry:
    """A single configuration entry."""
    key: str
    value: Any
    source: ConfigSource
    priority: ConfigPriority
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    key: str
    value_type: type
    default: Any
    description: str
    required: bool = False
    env_var: Optional[str] = None
    secret: bool = False


class ConfigValidator:
    """Configuration validator."""

    @staticmethod
    def validate_type(value: Any, expected_type: type) -> bool:
        """Validate value type."""
        if expected_type == Any:
            return True
        return isinstance(value, expected_type)

    @staticmethod
    def validate_enum(value: Any, allowed: List[Any]) -> bool:
        """Validate enum value."""
        return value in allowed

    @staticmethod
    def validate_range(value: Union[int, float], min_val: Any, max_val: Any) -> bool:
        """Validate numeric range."""
        return min_val <= value <= max_val


class ConfigurationManager:
    """Centralized configuration management.

    Features:
    - Multi-source configuration
    - Priority-based override
    - Schema validation
    - Environment variable support
    - Secret handling
    """

    def __init__(
        self,
        app_name: str = "pyutagent",
        config_dir: Optional[Path] = None
    ):
        """Initialize configuration manager.

        Args:
            app_name: Application name
            config_dir: Configuration directory
        """
        self.app_name = app_name
        self.config_dir = config_dir or Path.home() / f".{app_name}"
        self._config: Dict[str, ConfigEntry] = {}
        self._schema: Dict[str, ConfigSchema] = {}
        self._secrets: Dict[str, str] = {}

    def register_schema(self, schema: ConfigSchema):
        """Register configuration schema.

        Args:
            schema: Configuration schema
        """
        self._schema[schema.key] = schema

        if schema.env_var and schema.default is not None:
            env_value = os.environ.get(schema.env_var)
            if env_value is not None:
                self.set(
                    schema.key,
                    env_value,
                    ConfigSource.ENVIRONMENT
                )
            elif schema.required:
                self.set(schema.key, schema.default, ConfigSource.DEFAULT)

    def register_schemas(self, schemas: List[ConfigSchema]):
        """Register multiple schemas.

        Args:
            schemas: List of schemas
        """
        for schema in schemas:
            self.register_schema(schema)

    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.MEMORY,
        priority: Optional[ConfigPriority] = None
    ):
        """Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
            source: Configuration source
            priority: Configuration priority
        """
        if key in self._schema:
            schema = self._schema[key]
            if not ConfigValidator.validate_type(value, schema.value_type):
                raise TypeError(
                    f"Invalid type for {key}: expected {schema.value_type}, got {type(value)}"
                )

        if priority is None:
            priority = ConfigPriority(source)

        existing = self._config.get(key)
        if existing and existing.priority.value > priority.value:
            return

        self._config[key] = ConfigEntry(
            key=key,
            value=value,
            source=source,
            priority=priority
        )

    def get(
        self,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value
            required: Whether key is required

        Returns:
            Configuration value

        Raises:
            KeyError: If required key is not found
        """
        if key in self._config:
            return self._config[key].value

        if key in self._schema:
            return self._schema[key].default

        if required:
            raise KeyError(f"Required config key not found: {key}")

        return default

    def get_typed(self, key: str, value_type: type, default: Any = None) -> Any:
        """Get typed configuration value.

        Args:
            key: Configuration key
            value_type: Expected type
            default: Default value

        Returns:
            Typed value
        """
        value = self.get(key, default)
        if value is None:
            return default

        try:
            return value_type(value)
        except (TypeError, ValueError):
            return default

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        result = {}
        for key, entry in self._config.items():
            if key not in self._schema or not self._schema[key].secret:
                result[key] = entry.value
        return result

    def load_from_file(self, file_path: Path):
        """Load configuration from file.

        Args:
            file_path: Path to config file
        """
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return

        try:
            with open(file_path) as f:
                data = json.load(f)

            for key, value in data.items():
                self.set(key, value, ConfigSource.FILE)

            logger.info(f"Loaded config from {file_path}")

        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")

    def load_from_env(self, prefix: str = "APP_"):
        """Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix
        """
        for key, schema in self._schema.items():
            env_var = schema.env_var or f"{prefix}{key.upper()}"
            env_value = os.environ.get(env_var)

            if env_value is not None:
                self.set(key, env_value, ConfigSource.ENVIRONMENT)

    def save_to_file(self, file_path: Path, include_secrets: bool = False):
        """Save configuration to file.

        Args:
            file_path: Path to save config
            include_secrets: Whether to include secrets
        """
        data = self.get_all()

        if not include_secrets:
            for key, schema in self._schema.items():
                if schema.secret:
                    data.pop(key, None)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved config to {file_path}")

    def set_secret(self, key: str, value: str):
        """Set a secret value.

        Args:
            key: Secret key
            value: Secret value
        """
        self._secrets[key] = value
        self.set(key, "***SECRET***", ConfigSource.MEMORY)

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value.

        Args:
            key: Secret key

        Returns:
            Secret value or None
        """
        return self._secrets.get(key)

    def get_schema(self, key: str) -> Optional[ConfigSchema]:
        """Get schema for key.

        Args:
            key: Configuration key

        Returns:
            Schema or None
        """
        return self._schema.get(key)

    def validate_all(self) -> List[str]:
        """Validate all configuration.

        Returns:
            List of validation errors
        """
        errors = []

        for key, schema in self._schema.items():
            if schema.required and key not in self._config:
                errors.append(f"Missing required config: {key}")

        return errors


class ConfigTemplate:
    """Configuration template."""

    @staticmethod
    def get_default_schema() -> List[ConfigSchema]:
        """Get default configuration schema.

        Returns:
            List of default schemas
        """
        return [
            ConfigSchema(
                key="max_iterations",
                value_type=int,
                default=50,
                description="Maximum iterations for agent",
                env_var="MAX_ITERATIONS"
            ),
            ConfigSchema(
                key="timeout_seconds",
                value_type=int,
                default=300,
                description="Timeout for operations",
                env_var="TIMEOUT_SECONDS"
            ),
            ConfigSchema(
                key="llm_model",
                value_type=str,
                default="claude-3-sonnet-20240229",
                description="LLM model to use",
                env_var="LLM_MODEL"
            ),
            ConfigSchema(
                key="enable_cache",
                value_type=bool,
                default=True,
                description="Enable caching",
                env_var="ENABLE_CACHE"
            ),
            ConfigSchema(
                key="log_level",
                value_type=str,
                default="INFO",
                description="Logging level",
                env_var="LOG_LEVEL"
            ),
            ConfigSchema(
                key="api_key",
                value_type=str,
                default="",
                description="API key for LLM",
                env_var="API_KEY",
                secret=True
            ),
            ConfigSchema(
                key="max_tokens",
                value_type=int,
                default=4096,
                description="Maximum tokens for LLM response",
                env_var="MAX_TOKENS"
            ),
        ]


def create_config_manager(
    app_name: str = "pyutagent",
    config_dir: Optional[Path] = None,
    load_env: bool = True,
    load_file: bool = True
) -> ConfigurationManager:
    """Create configured configuration manager.

    Args:
        app_name: Application name
        config_dir: Configuration directory
        load_env: Load from environment
        load_file: Load from file

    Returns:
        ConfigurationManager instance
    """
    manager = ConfigurationManager(app_name, config_dir)
    manager.register_schemas(ConfigTemplate.get_default_schema())

    if load_env:
        manager.load_from_env()

    if load_file:
        config_file = manager.config_dir / "config.json"
        manager.load_from_file(config_file)

    return manager
