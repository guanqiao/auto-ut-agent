"""LLM configuration models."""

from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, SecretStr, field_serializer, ConfigDict


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
        """Serialize API key for JSON."""
        return value.get_secret_value()
    
    @field_serializer('ca_cert')
    def _serialize_ca_cert(self, value: Optional[Path]) -> Optional[str]:
        """Serialize CA cert path for JSON."""
        return str(value) if value else None
    
    def get_masked_api_key(self) -> str:
        """Get masked API key for display."""
        key = self.api_key.get_secret_value()
        if len(key) <= 8:
            return "***"
        return key[:4] + "***" + key[-4:]
    
    def is_configured(self) -> bool:
        """Check if the configuration is valid."""
        return bool(
            self.api_key.get_secret_value() and
            self.endpoint and
            self.model
        )


# Provider default endpoints
PROVIDER_ENDPOINTS = {
    LLMProvider.OPENAI: "https://api.openai.com/v1",
    LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1",
    LLMProvider.DEEPSEEK: "https://api.deepseek.com/v1",
    LLMProvider.OLLAMA: "http://localhost:11434/v1",
}

# Provider default models
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
    return PROVIDER_ENDPOINTS.get(provider, "")


def get_available_models(provider: LLMProvider) -> list[str]:
    """Get available models for a provider."""
    return PROVIDER_MODELS.get(provider, [])
