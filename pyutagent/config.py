"""Configuration management for PyUT Agent.

This module provides backward-compatible imports from core.config.
All configuration classes and functions have been consolidated into core.config.
"""

from .core.config import (
    Settings,
    LLMConfig,
    LLMConfigCollection,
    LLMProvider,
    AiderConfig,
    ArchitectMode,
    ProjectPaths,
    CoverageSettings,
    get_settings,
    get_data_dir,
    get_llm_config_path,
    get_aider_config_path,
    get_app_config_path,
    save_llm_config,
    load_llm_config,
    save_aider_config,
    load_aider_config,
    save_app_config,
    load_app_config,
    get_default_endpoint,
    get_available_models,
    PROVIDER_ENDPOINTS,
    PROVIDER_MODELS,
)

__all__ = [
    "Settings",
    "LLMConfig",
    "LLMConfigCollection",
    "LLMProvider",
    "AiderConfig",
    "ArchitectMode",
    "ProjectPaths",
    "CoverageSettings",
    "get_settings",
    "get_data_dir",
    "get_llm_config_path",
    "get_aider_config_path",
    "get_app_config_path",
    "save_llm_config",
    "load_llm_config",
    "save_aider_config",
    "load_aider_config",
    "save_app_config",
    "load_app_config",
    "get_default_endpoint",
    "get_available_models",
    "PROVIDER_ENDPOINTS",
    "PROVIDER_MODELS",
]
