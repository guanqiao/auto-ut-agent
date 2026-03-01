"""Configuration management for PyUT Agent."""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import json
from dataclasses import asdict

from .llm.config import LLMConfigCollection
from .tools.aider_integration import AiderConfig


class Settings(BaseSettings):
    """Application settings."""

    # LLM Configuration
    llm_provider: str = "openai"
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # Coverage Configuration
    target_coverage: float = 0.8
    max_iterations: int = 10

    # Aider Configuration
    aider_use_architect_editor: bool = False
    aider_architect_model: Optional[str] = None
    aider_editor_model: Optional[str] = None
    aider_enable_multi_file: bool = False
    aider_max_files_per_edit: int = 5
    aider_preferred_format: Optional[str] = None
    aider_auto_detect_format: bool = True
    aider_max_attempts: int = 3
    aider_timeout_seconds: float = 120.0
    aider_enable_fallback: bool = True
    aider_enable_circuit_breaker: bool = True
    aider_track_costs: bool = True

    # Paths
    data_dir: Path = Path.home() / ".pyutagent"
    vector_db_path: Path = data_dir / "vector_db"
    memory_db_path: Path = data_dir / "memory.db"

    class Config:
        env_prefix = "PYUT_"
        env_file = ".env"


# Global settings instance
settings = Settings()


def get_data_dir() -> Path:
    """Get or create data directory."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir


def get_aider_config_path() -> Path:
    """Get path to Aider config file."""
    return get_data_dir() / "aider_config.json"


def save_aider_config(config: AiderConfig):
    """Save Aider configuration to file.

    Args:
        config: AiderConfig instance to save
    """
    config_path = get_aider_config_path()
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2, default=str)


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
            return AiderConfig(**data)
        except Exception:
            # If loading fails, return default config
            pass
    return AiderConfig()


def get_llm_config_path() -> Path:
    """Get path to LLM config file."""
    return get_data_dir() / "llm_config.json"


def save_llm_config(config_collection: LLMConfigCollection):
    """Save LLM configuration collection to file.
    
    Args:
        config_collection: LLM configuration collection to save
    """
    config_path = get_llm_config_path()
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_collection.model_dump_json(indent=2))


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
            return LLMConfigCollection.model_validate(data)
        except Exception:
            # If loading fails, return empty collection
            pass
    return LLMConfigCollection()
