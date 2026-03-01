"""Unified configuration management for PyUT Agent.

This module consolidates all configuration from:
- pyutagent/config.py (Settings, data directory management)
- pyutagent/llm/config.py (LLMConfig, LLMConfigCollection)
- pyutagent/tools/aider_integration.py (AiderConfig)

Provides:
- Centralized configuration management
- Persistent storage
- Configuration validation
- Easy access to all settings
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field, SecretStr, field_serializer, ConfigDict
import uuid

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """LLM provider enumeration."""
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    CUSTOM = "custom"


class LLMConfig(BaseModel):
    """LLM configuration model.

    Attributes:
        id: Unique identifier for this configuration
        name: Display name for this configuration
        provider: LLM provider (openai, azure, anthropic, etc.)
        endpoint: API endpoint URL
        api_key: API key for authentication
        model: Model name
        ca_cert: Path to CA certificate for SSL
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
    """
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique configuration ID"
    )
    name: str = Field(
        default="",
        description="Display name for this configuration"
    )
    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider"
    )
    endpoint: str = Field(
        default="https://api.openai.com/v1",
        description="API endpoint URL"
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key"
    )
    model: str = Field(
        default="gpt-4",
        description="Model name"
    )
    ca_cert: Optional[Path] = Field(
        default=None,
        description="CA certificate path"
    )
    timeout: int = Field(
        default=300,
        ge=10,
        le=600,
        description="Request timeout (seconds)"
    )
    max_retries: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Maximum retries"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=4096,
        ge=100,
        le=32000,
        description="Maximum tokens"
    )

    @field_serializer('api_key')
    def _serialize_api_key(self, value: SecretStr) -> str:
        return value.get_secret_value()

    @field_serializer('ca_cert')
    def _serialize_ca_cert(self, value: Optional[Path]) -> Optional[str]:
        return str(value) if value else None

    def get_masked_api_key(self) -> str:
        """Get masked API key for display."""
        key = self.api_key.get_secret_value()
        if len(key) <= 8:
            return "***"
        return key[:4] + "***" + key[-4:]

    def is_configured(self) -> bool:
        """Check if the configuration is valid."""
        has_key = bool(self.api_key.get_secret_value())
        has_endpoint = bool(self.endpoint)
        has_model = bool(self.model)

        is_valid = has_key and has_endpoint and has_model

        if not is_valid:
            logger.debug(f"[LLMConfig] Configuration check failed - ID: {self.id}, has_key: {has_key}, has_endpoint: {has_endpoint}, has_model: {has_model}")

        return is_valid

    def get_display_name(self) -> str:
        """Get display name for this configuration."""
        if self.name:
            return f"{self.name} ({self.model})"
        provider_str = self.provider.value if hasattr(self.provider, 'value') else str(self.provider)
        return f"{provider_str} - {self.model}"


class LLMConfigCollection(BaseModel):
    """Collection of LLM configurations with management capabilities.

    Attributes:
        configs: List of LLM configurations
        default_config_id: ID of the default configuration to use
    """
    model_config = ConfigDict(use_enum_values=True)

    configs: List[LLMConfig] = Field(
        default_factory=list,
        description="List of LLM configurations"
    )
    default_config_id: Optional[str] = Field(
        default=None,
        description="ID of the default configuration"
    )

    def add_config(self, config: LLMConfig) -> None:
        """Add a new configuration."""
        self.configs.append(config)
        if len(self.configs) == 1:
            self.default_config_id = config.id
            logger.info(f"[LLMConfigCollection] Added first config as default - ID: {config.id}, Name: {config.get_display_name()}")
        else:
            logger.info(f"[LLMConfigCollection] Added config - ID: {config.id}, Name: {config.get_display_name()}, Total: {len(self.configs)}")

    def remove_config(self, config_id: str) -> bool:
        """Remove a configuration by ID."""
        for i, config in enumerate(self.configs):
            if config.id == config_id:
                self.configs.pop(i)
                old_default = self.default_config_id
                if self.default_config_id == config_id:
                    self.default_config_id = self.configs[0].id if self.configs else None
                    logger.info(f"[LLMConfigCollection] Removed default config, updated to - Old: {old_default}, New: {self.default_config_id}")

                logger.info(f"[LLMConfigCollection] Removed config - ID: {config_id}, Name: {config.get_display_name()}, Remaining: {len(self.configs)}")
                return True

        logger.warning(f"[LLMConfigCollection] Remove config failed - ID: {config_id} not found")
        return False

    def get_config(self, config_id: str) -> Optional[LLMConfig]:
        """Get configuration by ID."""
        for config in self.configs:
            if config.id == config_id:
                logger.debug(f"[LLMConfigCollection] Got config - ID: {config_id}, Name: {config.get_display_name()}")
                return config

        logger.warning(f"[LLMConfigCollection] Get config failed - ID: {config_id} not found")
        return None

    def get_default_config(self) -> Optional[LLMConfig]:
        """Get the default configuration."""
        if self.default_config_id:
            config = self.get_config(self.default_config_id)
            if config:
                logger.debug(f"[LLMConfigCollection] Got default config - ID: {config.id}, Name: {config.get_display_name()}")
                return config

        if self.configs:
            config = self.configs[0]
            logger.debug(f"[LLMConfigCollection] Got default config (first) - ID: {config.id}, Name: {config.get_display_name()}")
            return config

        logger.warning("[LLMConfigCollection] Get default config failed - Config list is empty")
        return None

    def set_default_config(self, config_id: str) -> bool:
        """Set the default configuration."""
        if self.get_config(config_id):
            old_default = self.default_config_id
            self.default_config_id = config_id
            logger.info(f"[LLMConfigCollection] Set default config - Old: {old_default}, New: {config_id}")
            return True

        logger.warning(f"[LLMConfigCollection] Set default config failed - ID: {config_id} not found")
        return False

    def update_config(self, config_id: str, updates: dict) -> bool:
        """Update a configuration."""
        config = self.get_config(config_id)
        if config:
            for key, value in updates.items():
                if hasattr(config, key):
                    old_value = getattr(config, key)
                    setattr(config, key, value)
                    logger.debug(f"[LLMConfigCollection] Updated config field - ID: {config_id}, Field: {key}, Old: {old_value}, New: {value}")

            logger.info(f"[LLMConfigCollection] Updated config - ID: {config_id}, UpdatedFields: {list(updates.keys())}")
            return True

        logger.warning(f"[LLMConfigCollection] Update config failed - ID: {config_id} not found")
        return False

    def get_config_names(self) -> List[Tuple[str, str]]:
        """Get list of (id, display_name) tuples for all configs."""
        names = [(c.id, c.get_display_name()) for c in self.configs]
        logger.debug(f"[LLMConfigCollection] Got config names list - Count: {len(names)}")
        return names

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        is_empty = len(self.configs) == 0
        if is_empty:
            logger.debug("[LLMConfigCollection] Config collection is empty")
        return is_empty

    def create_default_config(self) -> LLMConfig:
        """Create and add a default configuration."""
        config = LLMConfig(
            name="Default",
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )
        self.add_config(config)
        logger.info(f"[LLMConfigCollection] Created default config - ID: {config.id}")
        return config


class ArchitectMode(str, Enum):
    """Architect/Editor mode."""
    SINGLE_MODEL = "single_model"
    DUAL_MODEL = "dual_model"


@dataclass
class AiderConfig:
    """Configuration for Aider integration.

    Core settings:
        max_attempts: Maximum number of fix attempts
        enable_fallback: Enable fallback strategies
        enable_circuit_breaker: Enable circuit breaker pattern
        timeout_seconds: Timeout for operations

    Architect/Editor settings:
        use_architect_editor: Enable Architect/Editor dual-model pattern
        architect_model_id: LLM config ID for Architect role
        editor_model_id: LLM config ID for Editor role
        architect_mode: Single or dual model mode

    Multi-file settings:
        enable_multi_file: Enable multi-file editing
        max_files_per_edit: Maximum files to edit at once

    Edit format settings:
        preferred_format: Preferred edit format (None = auto-detect)
        auto_detect_format: Auto-detect best format

    Cost tracking:
        track_costs: Track LLM API costs
    """
    max_attempts: int = 3
    enable_fallback: bool = True
    enable_circuit_breaker: bool = True
    timeout_seconds: float = 120.0

    use_architect_editor: bool = False
    architect_model_id: Optional[str] = None
    editor_model_id: Optional[str] = None
    architect_mode: ArchitectMode = ArchitectMode.DUAL_MODEL

    enable_multi_file: bool = False
    max_files_per_edit: int = 5

    preferred_format: Optional[str] = None
    auto_detect_format: bool = True

    track_costs: bool = True

    def get_model_ids(self) -> Tuple[Optional[str], Optional[str]]:
        """Get architect and editor model IDs.

        Returns:
            Tuple of (architect_model_id, editor_model_id)
        """
        return (self.architect_model_id, self.editor_model_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AiderConfig":
        """Create from dictionary."""
        if "architect_mode" in data and isinstance(data["architect_mode"], str):
            try:
                data["architect_mode"] = ArchitectMode(data["architect_mode"])
            except ValueError:
                data["architect_mode"] = ArchitectMode.DUAL_MODEL
        return cls(**data)


@dataclass
class ProjectPaths:
    """Project path configuration.

    Attributes:
        src_main_java: Path to main Java source directory
        src_test_java: Path to test Java source directory
        target_classes: Path to compiled classes directory
        target_test_classes: Path to compiled test classes directory
        target_surefire_reports: Path to Maven surefire reports
        target_jacoco_reports: Path to JaCoCo coverage reports
    """
    src_main_java: str = "src/main/java"
    src_test_java: str = "src/test/java"
    target_classes: str = "target/classes"
    target_test_classes: str = "target/test-classes"
    target_surefire_reports: str = "target/surefire-reports"
    target_jacoco_reports: str = "target/site/jacoco"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectPaths":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CoverageSettings:
    """Coverage configuration.

    Attributes:
        target_coverage: Target coverage percentage (0.0-1.0)
        min_coverage: Minimum acceptable coverage
        max_iterations: Maximum iterations for coverage improvement
    """
    target_coverage: float = 0.8
    min_coverage: float = 0.5
    max_iterations: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoverageSettings":
        """Create from dictionary."""
        return cls(**data)


class Settings(BaseModel):
    """Application settings.

    This consolidates all application-level settings.
    """
    model_config = ConfigDict(
        use_enum_values=True,
        env_prefix="PYUT_",
        env_file=".env"
    )

    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".pyutagent",
        description="Application data directory"
    )

    project_paths: ProjectPaths = Field(
        default_factory=ProjectPaths,
        description="Project path configuration"
    )

    coverage: CoverageSettings = Field(
        default_factory=CoverageSettings,
        description="Coverage settings"
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    enable_debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )


_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings()
        logger.info(f"[Settings] Created global settings - DataDir: {_global_settings.data_dir}")
    return _global_settings


def get_data_dir() -> Path:
    """Get or create data directory."""
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir


def get_llm_config_path() -> Path:
    """Get path to LLM config file."""
    return get_data_dir() / "llm_config.json"


def get_aider_config_path() -> Path:
    """Get path to Aider config file."""
    return get_data_dir() / "aider_config.json"


def get_app_config_path() -> Path:
    """Get path to application config file."""
    return get_data_dir() / "app_config.json"


def save_llm_config(config_collection: LLMConfigCollection) -> None:
    """Save LLM configuration collection to file.

    Args:
        config_collection: LLM configuration collection to save
    """
    config_path = get_llm_config_path()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_collection.model_dump_json(indent=2))
        logger.info(f"[Config] Saved LLM config to {config_path}")
    except Exception as e:
        logger.error(f"[Config] Failed to save LLM config: {e}")
        raise


def load_llm_config() -> LLMConfigCollection:
    """Load LLM configuration collection from file.

    Returns:
        LLMConfigCollection: Loaded configuration collection, or empty collection if file doesn't exist
    """
    config_path = get_llm_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config = LLMConfigCollection.model_validate(data)
            logger.info(f"[Config] Loaded LLM config from {config_path} - {len(config.configs)} configs")
            return config
        except Exception as e:
            logger.warning(f"[Config] Failed to load LLM config: {e}, creating empty collection")
            return LLMConfigCollection()
    else:
        logger.info(f"[Config] LLM config file not found at {config_path}, creating empty collection")
        return LLMConfigCollection()


def save_aider_config(config: AiderConfig) -> None:
    """Save Aider configuration to file.

    Args:
        config: AiderConfig instance to save
    """
    config_path = get_aider_config_path()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        logger.info(f"[Config] Saved Aider config to {config_path}")
    except Exception as e:
        logger.error(f"[Config] Failed to save Aider config: {e}")
        raise


def load_aider_config() -> AiderConfig:
    """Load Aider configuration from file.

    Returns:
        AiderConfig: Loaded configuration, or default config if file doesn't exist
    """
    config_path = get_aider_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config = AiderConfig.from_dict(data)
            logger.info(f"[Config] Loaded Aider config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"[Config] Failed to load Aider config: {e}, using default")
            return AiderConfig()
    else:
        logger.info(f"[Config] Aider config file not found at {config_path}, using default")
        return AiderConfig()


def save_app_config(settings: Settings) -> None:
    """Save application settings to file.

    Args:
        settings: Settings instance to save
    """
    config_path = get_app_config_path()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(settings.model_dump(mode='json'), f, indent=2, default=str)
        logger.info(f"[Config] Saved app config to {config_path}")
    except Exception as e:
        logger.error(f"[Config] Failed to save app config: {e}")
        raise


def load_app_config() -> Settings:
    """Load application settings from file.

    Returns:
        Settings: Loaded settings, or default settings if file doesn't exist
    """
    config_path = get_app_config_path()
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            settings = Settings.model_validate(data)
            logger.info(f"[Config] Loaded app config from {config_path}")
            return settings
        except Exception as e:
            logger.warning(f"[Config] Failed to load app config: {e}, using default")
            return Settings()
    else:
        logger.info(f"[Config] App config file not found at {config_path}, using default")
        return Settings()


PROVIDER_ENDPOINTS = {
    LLMProvider.OPENAI: "https://api.openai.com/v1",
    LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1",
    LLMProvider.DEEPSEEK: "https://api.deepseek.com/v1",
    LLMProvider.OLLAMA: "http://localhost:11434/v1",
}


PROVIDER_MODELS = {
    LLMProvider.OPENAI: [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-3.5-turbo",
    ],
    LLMProvider.ANTHROPIC: [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    LLMProvider.DEEPSEEK: [
        "deepseek-chat",
        "deepseek-coder",
    ],
    LLMProvider.OLLAMA: [
        "llama2",
        "codellama",
        "mistral",
        "mixtral",
    ],
}


def get_default_endpoint(provider: LLMProvider) -> str:
    """Get default endpoint for a provider."""
    endpoint = PROVIDER_ENDPOINTS.get(provider, "")
    logger.debug(f"[Config] Got default endpoint - Provider: {provider}, Endpoint: {endpoint}")
    return endpoint


def get_available_models(provider: LLMProvider) -> List[str]:
    """Get available models for a provider."""
    models = PROVIDER_MODELS.get(provider, [])
    logger.debug(f"[Config] Got available models - Provider: {provider}, Models: {models}")
    return models
