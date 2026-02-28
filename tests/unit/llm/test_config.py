"""Tests for LLM config module."""

import pytest
from pathlib import Path
from pydantic import SecretStr

from pyutagent.llm.config import (
    LLMProvider,
    LLMConfig,
    LLMConfigCollection,
    PROVIDER_ENDPOINTS,
    PROVIDER_MODELS,
    get_default_endpoint,
    get_available_models,
)


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_provider_values(self):
        """Test that all providers are defined."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.AZURE.value == "azure"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.DEEPSEEK.value == "deepseek"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.CUSTOM.value == "custom"


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_default_creation(self):
        """Test creating config with defaults."""
        config = LLMConfig()

        assert config.provider == LLMProvider.OPENAI
        assert config.endpoint == "https://api.openai.com/v1"
        assert config.model == "gpt-4"
        assert config.timeout == 300
        assert config.max_retries == 5
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.ca_cert is None
        assert len(config.id) == 8  # UUID first 8 chars

    def test_custom_creation(self):
        """Test creating config with custom values."""
        config = LLMConfig(
            name="My Config",
            provider=LLMProvider.ANTHROPIC,
            endpoint="https://custom.api.com",
            api_key=SecretStr("secret-key"),
            model="claude-3-opus",
            timeout=120,
            max_retries=3,
            temperature=0.5,
            max_tokens=2048,
        )

        assert config.name == "My Config"
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.endpoint == "https://custom.api.com"
        assert config.model == "claude-3-opus"
        assert config.timeout == 120
        assert config.max_retries == 3
        assert config.temperature == 0.5
        assert config.max_tokens == 2048

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)

        # Invalid temperatures should raise error
        with pytest.raises(Exception):
            LLMConfig(temperature=-0.1)
        with pytest.raises(Exception):
            LLMConfig(temperature=2.1)

    def test_timeout_validation(self):
        """Test timeout validation."""
        # Valid timeouts
        LLMConfig(timeout=10)
        LLMConfig(timeout=300)
        LLMConfig(timeout=600)

        # Invalid timeouts should raise error
        with pytest.raises(Exception):
            LLMConfig(timeout=9)
        with pytest.raises(Exception):
            LLMConfig(timeout=601)

    def test_max_retries_validation(self):
        """Test max_retries validation."""
        # Valid retries
        LLMConfig(max_retries=0)
        LLMConfig(max_retries=5)
        LLMConfig(max_retries=10)

        # Invalid retries should raise error
        with pytest.raises(Exception):
            LLMConfig(max_retries=-1)
        with pytest.raises(Exception):
            LLMConfig(max_retries=11)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid tokens
        LLMConfig(max_tokens=100)
        LLMConfig(max_tokens=4096)
        LLMConfig(max_tokens=32000)

        # Invalid tokens should raise error
        with pytest.raises(Exception):
            LLMConfig(max_tokens=99)
        with pytest.raises(Exception):
            LLMConfig(max_tokens=32001)

    def test_api_key_serialization(self):
        """Test API key serialization."""
        config = LLMConfig(api_key=SecretStr("my-secret-key"))
        serialized = config.model_dump()

        assert serialized["api_key"] == "my-secret-key"

    def test_ca_cert_serialization(self):
        """Test CA cert serialization."""
        config = LLMConfig(ca_cert=Path("/path/to/cert.pem"))
        serialized = config.model_dump()

        # On Windows, path separators may be converted
        assert "cert.pem" in serialized["ca_cert"]

    def test_ca_cert_serialization_none(self):
        """Test CA cert serialization when None."""
        config = LLMConfig(ca_cert=None)
        serialized = config.model_dump()

        assert serialized["ca_cert"] is None

    def test_get_masked_api_key_long(self):
        """Test masking long API key."""
        config = LLMConfig(api_key=SecretStr("sk-1234567890abcdef"))
        masked = config.get_masked_api_key()

        assert masked == "sk-1***cdef"
        assert "***" in masked

    def test_get_masked_api_key_short(self):
        """Test masking short API key."""
        config = LLMConfig(api_key=SecretStr("short"))
        masked = config.get_masked_api_key()

        assert masked == "***"

    def test_get_masked_api_key_empty(self):
        """Test masking empty API key."""
        config = LLMConfig(api_key=SecretStr(""))
        masked = config.get_masked_api_key()

        assert masked == "***"

    def test_is_configured_true(self):
        """Test is_configured returns True when properly configured."""
        config = LLMConfig(
            api_key=SecretStr("valid-key"),
            endpoint="https://api.example.com",
            model="gpt-4"
        )
        assert config.is_configured() is True

    def test_is_configured_false_no_key(self):
        """Test is_configured returns False without API key."""
        config = LLMConfig(
            api_key=SecretStr(""),
            endpoint="https://api.example.com",
            model="gpt-4"
        )
        assert config.is_configured() is False

    def test_is_configured_false_no_endpoint(self):
        """Test is_configured returns False without endpoint."""
        config = LLMConfig(
            api_key=SecretStr("key"),
            endpoint="",
            model="gpt-4"
        )
        assert config.is_configured() is False

    def test_is_configured_false_no_model(self):
        """Test is_configured returns False without model."""
        config = LLMConfig(
            api_key=SecretStr("key"),
            endpoint="https://api.example.com",
            model=""
        )
        assert config.is_configured() is False

    def test_get_display_name_with_name(self):
        """Test get_display_name when name is set."""
        config = LLMConfig(name="Production", model="gpt-4")
        assert config.get_display_name() == "Production (gpt-4)"

    def test_get_display_name_without_name(self):
        """Test get_display_name when name is not set."""
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3")
        # provider is stored as enum value due to ConfigDict(use_enum_values=True)
        display_name = config.get_display_name()
        assert "claude-3" in display_name
        assert "anthropic" in display_name


class TestLLMConfigCollection:
    """Tests for LLMConfigCollection."""

    @pytest.fixture
    def collection(self):
        """Create an empty LLMConfigCollection."""
        return LLMConfigCollection()

    def test_initialization(self, collection):
        """Test collection initialization."""
        assert collection.configs == []
        assert collection.default_config_id is None
        assert collection.is_empty() is True

    def test_add_config(self, collection):
        """Test adding a config."""
        config = LLMConfig(name="Test")
        collection.add_config(config)

        assert len(collection.configs) == 1
        assert collection.configs[0] == config
        assert collection.default_config_id == config.id
        assert collection.is_empty() is False

    def test_add_multiple_configs(self, collection):
        """Test adding multiple configs."""
        config1 = LLMConfig(name="First")
        config2 = LLMConfig(name="Second")

        collection.add_config(config1)
        collection.add_config(config2)

        assert len(collection.configs) == 2
        # First config should be default
        assert collection.default_config_id == config1.id

    def test_remove_config(self, collection):
        """Test removing a config."""
        config = LLMConfig(name="Test")
        collection.add_config(config)

        result = collection.remove_config(config.id)

        assert result is True
        assert len(collection.configs) == 0
        assert collection.default_config_id is None

    def test_remove_config_not_found(self, collection):
        """Test removing a config that doesn't exist."""
        result = collection.remove_config("nonexistent-id")
        assert result is False

    def test_remove_config_updates_default(self, collection):
        """Test that removing default config updates default."""
        config1 = LLMConfig(name="First")
        config2 = LLMConfig(name="Second")
        collection.add_config(config1)
        collection.add_config(config2)

        collection.remove_config(config1.id)

        assert collection.default_config_id == config2.id

    def test_get_config(self, collection):
        """Test getting a config by ID."""
        config = LLMConfig(name="Test")
        collection.add_config(config)

        found = collection.get_config(config.id)

        assert found == config

    def test_get_config_not_found(self, collection):
        """Test getting a config that doesn't exist."""
        found = collection.get_config("nonexistent-id")
        assert found is None

    def test_get_default_config(self, collection):
        """Test getting default config."""
        config = LLMConfig(name="Test")
        collection.add_config(config)

        default = collection.get_default_config()

        assert default == config

    def test_get_default_config_none_set(self, collection):
        """Test getting default when none explicitly set."""
        config = LLMConfig(name="Test")
        collection.add_config(config)
        collection.default_config_id = None

        default = collection.get_default_config()

        assert default == config  # Returns first config

    def test_get_default_config_empty(self, collection):
        """Test getting default when collection is empty."""
        default = collection.get_default_config()
        assert default is None

    def test_set_default_config(self, collection):
        """Test setting default config."""
        config1 = LLMConfig(name="First")
        config2 = LLMConfig(name="Second")
        collection.add_config(config1)
        collection.add_config(config2)

        result = collection.set_default_config(config2.id)

        assert result is True
        assert collection.default_config_id == config2.id

    def test_set_default_config_not_found(self, collection):
        """Test setting default config that doesn't exist."""
        result = collection.set_default_config("nonexistent-id")
        assert result is False

    def test_update_config(self, collection):
        """Test updating a config."""
        config = LLMConfig(name="Test", model="gpt-4")
        collection.add_config(config)

        result = collection.update_config(config.id, {"name": "Updated", "model": "gpt-3.5"})

        assert result is True
        assert config.name == "Updated"
        assert config.model == "gpt-3.5"

    def test_update_config_not_found(self, collection):
        """Test updating a config that doesn't exist."""
        result = collection.update_config("nonexistent-id", {"name": "Updated"})
        assert result is False

    def test_update_config_invalid_field(self, collection):
        """Test updating a config with invalid field."""
        config = LLMConfig(name="Test")
        collection.add_config(config)

        result = collection.update_config(config.id, {"invalid_field": "value"})

        assert result is True  # Still returns True, just doesn't update
        assert not hasattr(config, "invalid_field")

    def test_get_config_names(self, collection):
        """Test getting config names."""
        config1 = LLMConfig(name="First", model="gpt-4")
        config2 = LLMConfig(name="Second", model="gpt-3.5")
        collection.add_config(config1)
        collection.add_config(config2)

        names = collection.get_config_names()

        assert len(names) == 2
        assert names[0] == (config1.id, "First (gpt-4)")
        assert names[1] == (config2.id, "Second (gpt-3.5)")

    def test_create_default_config(self, collection):
        """Test creating default config."""
        config = collection.create_default_config()

        assert config.name == "Default"
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert len(collection.configs) == 1
        assert collection.default_config_id == config.id


class TestProviderEndpoints:
    """Tests for provider endpoint constants."""

    def test_openai_endpoint(self):
        """Test OpenAI default endpoint."""
        assert PROVIDER_ENDPOINTS[LLMProvider.OPENAI] == "https://api.openai.com/v1"

    def test_anthropic_endpoint(self):
        """Test Anthropic default endpoint."""
        assert PROVIDER_ENDPOINTS[LLMProvider.ANTHROPIC] == "https://api.anthropic.com/v1"

    def test_deepseek_endpoint(self):
        """Test DeepSeek default endpoint."""
        assert PROVIDER_ENDPOINTS[LLMProvider.DEEPSEEK] == "https://api.deepseek.com/v1"

    def test_ollama_endpoint(self):
        """Test Ollama default endpoint."""
        assert PROVIDER_ENDPOINTS[LLMProvider.OLLAMA] == "http://localhost:11434/v1"


class TestProviderModels:
    """Tests for provider model constants."""

    def test_openai_models(self):
        """Test OpenAI models."""
        models = PROVIDER_MODELS[LLMProvider.OPENAI]
        assert "gpt-4" in models
        assert "gpt-4-turbo" in models
        assert "gpt-4o" in models
        assert "gpt-3.5-turbo" in models

    def test_anthropic_models(self):
        """Test Anthropic models."""
        models = PROVIDER_MODELS[LLMProvider.ANTHROPIC]
        assert "claude-3-opus-20240229" in models
        assert "claude-3-sonnet-20240229" in models
        assert "claude-3-haiku-20240307" in models

    def test_deepseek_models(self):
        """Test DeepSeek models."""
        models = PROVIDER_MODELS[LLMProvider.DEEPSEEK]
        assert "deepseek-chat" in models
        assert "deepseek-coder" in models

    def test_ollama_models(self):
        """Test Ollama models."""
        models = PROVIDER_MODELS[LLMProvider.OLLAMA]
        assert "llama2" in models
        assert "codellama" in models
        assert "mistral" in models
        assert "mixtral" in models


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_default_endpoint_openai(self):
        """Test getting default endpoint for OpenAI."""
        endpoint = get_default_endpoint(LLMProvider.OPENAI)
        assert endpoint == "https://api.openai.com/v1"

    def test_get_default_endpoint_unknown(self):
        """Test getting default endpoint for unknown provider."""
        endpoint = get_default_endpoint(LLMProvider.CUSTOM)
        assert endpoint == ""

    def test_get_available_models_openai(self):
        """Test getting available models for OpenAI."""
        models = get_available_models(LLMProvider.OPENAI)
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models

    def test_get_available_models_unknown(self):
        """Test getting available models for unknown provider."""
        models = get_available_models(LLMProvider.CUSTOM)
        assert models == []
