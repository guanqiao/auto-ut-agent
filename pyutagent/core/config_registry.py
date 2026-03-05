"""Unified Configuration Registry.

This module provides a centralized configuration management system that:
- Consolidates all configuration sources
- Provides a single point of access for all configs
- Supports hot-reload and change notifications
- Handles environment variables uniformly

This is part of Phase 2 Week 13-14: Configuration Management Refactoring.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ConfigSource(Enum):
    """Configuration source enumeration."""
    DEFAULT = auto()
    ENVIRONMENT = auto()
    FILE = auto()
    CLI = auto()
    RUNTIME = auto()


class ConfigPriority(Enum):
    """Configuration priority (higher value = higher priority)."""
    DEFAULT = 10
    FILE = 30
    ENVIRONMENT = 50
    CLI = 70
    RUNTIME = 90


@dataclass
class ConfigMetadata:
    """Metadata for a configuration entry."""
    key: str
    source: ConfigSource
    priority: ConfigPriority
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1


class ConfigChangeType(Enum):
    """Type of configuration change."""
    ADDED = auto()
    UPDATED = auto()
    REMOVED = auto()


@dataclass
class ConfigChangeEvent:
    """Event fired when configuration changes."""
    key: str
    old_value: Any
    new_value: Any
    change_type: ConfigChangeType
    timestamp: datetime = field(default_factory=datetime.now)


ConfigChangeListener = Callable[[ConfigChangeEvent], None]


class ConfigRegistry:
    """Central configuration registry.
    
    Features:
    - Single source of truth for all configuration
    - Priority-based value resolution
    - Change notification system
    - Environment variable handling
    - File-based persistence
    - Type-safe access
    
    Usage:
        registry = ConfigRegistry.get_instance()
        
        # Register a configuration model
        registry.register_model("llm", LLMConfig)
        
        # Get configuration
        llm_config = registry.get("llm", LLMConfig)
        
        # Listen for changes
        registry.add_listener(on_config_change)
    """
    
    _instance: Optional["ConfigRegistry"] = None
    
    def __init__(
        self,
        app_name: str = "pyutagent",
        config_dir: Optional[Path] = None,
    ):
        """Initialize configuration registry.
        
        Args:
            app_name: Application name for config directory
            config_dir: Custom config directory
        """
        self.app_name = app_name
        self.config_dir = config_dir or Path.home() / f".{app_name}"
        
        self._values: Dict[str, Any] = {}
        self._metadata: Dict[str, ConfigMetadata] = {}
        self._models: Dict[str, Type[BaseModel]] = {}
        self._model_instances: Dict[str, BaseModel] = {}
        self._listeners: Set[ConfigChangeListener] = set()
        self._env_prefix = f"{app_name.upper()}_"
        self._loaded_files: Set[Path] = set()
    
    @classmethod
    def get_instance(cls) -> "ConfigRegistry":
        """Get singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
    
    def register_model(
        self,
        key: str,
        model_class: Type[BaseModel],
        instance: Optional[BaseModel] = None,
    ) -> None:
        """Register a Pydantic model configuration.
        
        Args:
            key: Configuration key
            model_class: Pydantic model class
            instance: Optional pre-existing instance
        """
        self._models[key] = model_class
        if instance is not None:
            self._model_instances[key] = instance
            self._set_value(key, instance, ConfigSource.RUNTIME, ConfigPriority.RUNTIME)
    
    def get_model(self, key: str) -> Optional[Type[BaseModel]]:
        """Get registered model class.
        
        Args:
            key: Configuration key
            
        Returns:
            Model class or None
        """
        return self._models.get(key)
    
    def set(
        self,
        key: str,
        value: Any,
        source: ConfigSource = ConfigSource.RUNTIME,
        priority: Optional[ConfigPriority] = None,
    ) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation for nested)
            value: Configuration value
            source: Source of the configuration
            priority: Priority level (defaults based on source)
        """
        if priority is None:
            priority = self._get_default_priority(source)
        
        self._set_value(key, value, source, priority)
    
    def _set_value(
        self,
        key: str,
        value: Any,
        source: ConfigSource,
        priority: ConfigPriority,
    ) -> None:
        """Internal method to set value with priority check."""
        existing = self._metadata.get(key)
        
        if existing and existing.priority.value > priority.value:
            logger.debug(
                f"Ignoring config update for {key}: "
                f"existing priority {existing.priority.name} > {priority.name}"
            )
            return
        
        old_value = self._values.get(key)
        self._values[key] = value
        self._metadata[key] = ConfigMetadata(
            key=key,
            source=source,
            priority=priority,
        )
        
        change_type = (
            ConfigChangeType.UPDATED if old_value is not None
            else ConfigChangeType.ADDED
        )
        
        self._notify_listeners(ConfigChangeEvent(
            key=key,
            old_value=old_value,
            new_value=value,
            change_type=change_type,
        ))
        
        logger.debug(f"Set config: {key} = {value} (source={source.name}, priority={priority.name})")
    
    def get(
        self,
        key: str,
        default: Any = None,
        model_class: Optional[Type[T]] = None,
    ) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            model_class: Optional model class for type conversion
            
        Returns:
            Configuration value
        """
        if key in self._model_instances:
            return self._model_instances[key]
        
        value = self._values.get(key)
        
        if value is None:
            if key in self._models:
                model_class = self._models[key]
                self._model_instances[key] = model_class()
                return self._model_instances[key]
            return default
        
        if model_class and isinstance(model_class, type(BaseModel)):
            if isinstance(value, dict):
                return model_class.model_validate(value)
            elif isinstance(value, BaseModel):
                return value
        
        return value
    
    def get_typed(
        self,
        key: str,
        value_type: Type[T],
        default: Optional[T] = None,
    ) -> T:
        """Get a typed configuration value.
        
        Args:
            key: Configuration key
            value_type: Expected type
            default: Default value
            
        Returns:
            Typed configuration value
        """
        value = self.get(key, default)
        
        if value is None:
            return default
        
        if isinstance(value, value_type):
            return value
        
        try:
            return value_type(value)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to convert {key} to {value_type}: {e}")
            return default
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists."""
        return key in self._values or key in self._models
    
    def remove(self, key: str) -> bool:
        """Remove a configuration entry.
        
        Args:
            key: Configuration key
            
        Returns:
            True if removed, False if not found
        """
        if key not in self._values:
            return False
        
        old_value = self._values.pop(key)
        self._metadata.pop(key, None)
        self._model_instances.pop(key, None)
        
        self._notify_listeners(ConfigChangeEvent(
            key=key,
            old_value=old_value,
            new_value=None,
            change_type=ConfigChangeType.REMOVED,
        ))
        
        return True
    
    def get_all(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Get all configuration values.
        
        Args:
            include_secrets: Whether to include secret values
            
        Returns:
            Dictionary of all configuration values
        """
        result = {}
        for key, value in self._values.items():
            if not include_secrets and self._is_secret_key(key):
                continue
            result[key] = value
        return result
    
    def _is_secret_key(self, key: str) -> bool:
        """Check if key is a secret."""
        secret_patterns = ["api_key", "password", "secret", "token"]
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in secret_patterns)
    
    def _get_default_priority(self, source: ConfigSource) -> ConfigPriority:
        """Get default priority for a source."""
        mapping = {
            ConfigSource.DEFAULT: ConfigPriority.DEFAULT,
            ConfigSource.FILE: ConfigPriority.FILE,
            ConfigSource.ENVIRONMENT: ConfigPriority.ENVIRONMENT,
            ConfigSource.CLI: ConfigPriority.CLI,
            ConfigSource.RUNTIME: ConfigPriority.RUNTIME,
        }
        return mapping.get(source, ConfigPriority.DEFAULT)
    
    def load_from_env(self, prefix: Optional[str] = None) -> None:
        """Load configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix (defaults to app name)
        """
        prefix = prefix or self._env_prefix
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                config_key = env_key[len(prefix):].lower()
                self.set(config_key, env_value, ConfigSource.ENVIRONMENT)
        
        logger.debug(f"Loaded config from environment with prefix {prefix}")
    
    def load_from_file(
        self,
        file_path: Path,
        source: ConfigSource = ConfigSource.FILE,
    ) -> None:
        """Load configuration from a JSON file.
        
        Args:
            file_path: Path to configuration file
            source: Configuration source
        """
        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._load_dict(data, source)
            self._loaded_files.add(file_path)
            
            logger.info(f"Loaded config from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
    
    def _load_dict(
        self,
        data: Dict[str, Any],
        source: ConfigSource,
        prefix: str = "",
    ) -> None:
        """Load configuration from a dictionary.
        
        Args:
            data: Configuration dictionary
            source: Configuration source
            prefix: Key prefix for nested configs
        """
        priority = self._get_default_priority(source)
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict) and not self._is_model_key(full_key):
                self._load_dict(value, source, full_key)
            else:
                self._set_value(full_key, value, source, priority)
    
    def _is_model_key(self, key: str) -> bool:
        """Check if key corresponds to a registered model."""
        base_key = key.split(".")[0]
        return base_key in self._models
    
    def save_to_file(
        self,
        file_path: Path,
        include_secrets: bool = False,
    ) -> None:
        """Save configuration to a JSON file.
        
        Args:
            file_path: Path to save configuration
            include_secrets: Whether to include secret values
        """
        data = self.get_all(include_secrets=include_secrets)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved config to {file_path}")
    
    def add_listener(self, listener: ConfigChangeListener) -> None:
        """Add a configuration change listener.
        
        Args:
            listener: Callback function for config changes
        """
        self._listeners.add(listener)
    
    def remove_listener(self, listener: ConfigChangeListener) -> None:
        """Remove a configuration change listener.
        
        Args:
            listener: Listener to remove
        """
        self._listeners.discard(listener)
    
    def _notify_listeners(self, event: ConfigChangeEvent) -> None:
        """Notify all listeners of a configuration change."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Config listener error: {e}")
    
    def get_metadata(self, key: str) -> Optional[ConfigMetadata]:
        """Get metadata for a configuration key.
        
        Args:
            key: Configuration key
            
        Returns:
            Metadata or None
        """
        return self._metadata.get(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        sources = {}
        for meta in self._metadata.values():
            source_name = meta.source.name
            sources[source_name] = sources.get(source_name, 0) + 1
        
        return {
            "total_entries": len(self._values),
            "registered_models": len(self._models),
            "loaded_files": len(self._loaded_files),
            "listeners": len(self._listeners),
            "by_source": sources,
        }
    
    def clear(self) -> None:
        """Clear all configuration entries."""
        self._values.clear()
        self._metadata.clear()
        self._model_instances.clear()
        self._loaded_files.clear()
        
        logger.info("Cleared all configuration")
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists using 'in' operator."""
        return self.has(key)
    
    def __getitem__(self, key: str) -> Any:
        """Get value using [] operator."""
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set value using [] operator."""
        self.set(key, value)


def get_config_registry() -> ConfigRegistry:
    """Get the global configuration registry instance."""
    return ConfigRegistry.get_instance()


def config_value(
    key: str,
    default: Any = None,
    value_type: Optional[Type[T]] = None,
) -> T:
    """Get a configuration value from the global registry.
    
    Args:
        key: Configuration key
        default: Default value
        value_type: Optional type for conversion
        
    Returns:
        Configuration value
    """
    registry = get_config_registry()
    
    if value_type:
        return registry.get_typed(key, value_type, default)
    
    return registry.get(key, default)
