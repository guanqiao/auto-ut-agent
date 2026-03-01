"""LLM configuration module.

This module provides backward-compatible imports from core.config.
All LLM configuration classes have been consolidated into core.config.
"""

from ..core.config import (
    LLMConfig,
    LLMConfigCollection,
    LLMProvider,
    get_default_endpoint,
    get_available_models,
    PROVIDER_ENDPOINTS,
    PROVIDER_MODELS,
)

__all__ = [
    "LLMConfig",
    "LLMConfigCollection",
    "LLMProvider",
    "get_default_endpoint",
    "get_available_models",
    "PROVIDER_ENDPOINTS",
    "PROVIDER_MODELS",
]
