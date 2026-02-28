"""LLM configuration models."""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, List
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
        has_key = bool(self.api_key.get_secret_value())
        has_endpoint = bool(self.endpoint)
        has_model = bool(self.model)
        
        is_valid = has_key and has_endpoint and has_model
        
        if not is_valid:
            logger.debug(f"[LLMConfig] 配置检查失败 - ID: {self.id}, has_key: {has_key}, has_endpoint: {has_endpoint}, has_model: {has_model}")
        
        return is_valid
    
    def get_display_name(self) -> str:
        """Get display name for this configuration."""
        if self.name:
            return f"{self.name} ({self.model})"
        # Handle case where provider might be a string (due to use_enum_values=True)
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
        """Add a new configuration.
        
        Args:
            config: LLM configuration to add
        """
        self.configs.append(config)
        # If this is the first config, make it default
        if len(self.configs) == 1:
            self.default_config_id = config.id
            logger.info(f"[LLMConfigCollection] 添加首个配置并设为默认 - ID: {config.id}, Name: {config.get_display_name()}")
        else:
            logger.info(f"[LLMConfigCollection] 添加配置 - ID: {config.id}, Name: {config.get_display_name()}, 总配置数: {len(self.configs)}")
    
    def remove_config(self, config_id: str) -> bool:
        """Remove a configuration by ID.
        
        Args:
            config_id: ID of configuration to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, config in enumerate(self.configs):
            if config.id == config_id:
                self.configs.pop(i)
                # Update default if needed
                old_default = self.default_config_id
                if self.default_config_id == config_id:
                    self.default_config_id = self.configs[0].id if self.configs else None
                    logger.info(f"[LLMConfigCollection] 删除默认配置，更新为 - Old: {old_default}, New: {self.default_config_id}")
                
                logger.info(f"[LLMConfigCollection] 删除配置 - ID: {config_id}, Name: {config.get_display_name()}, 剩余: {len(self.configs)}")
                return True
        
        logger.warning(f"[LLMConfigCollection] 删除配置失败 - ID: {config_id} 未找到")
        return False
    
    def get_config(self, config_id: str) -> Optional[LLMConfig]:
        """Get configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            LLMConfig or None if not found
        """
        for config in self.configs:
            if config.id == config_id:
                logger.debug(f"[LLMConfigCollection] 获取配置 - ID: {config_id}, Name: {config.get_display_name()}")
                return config
        
        logger.warning(f"[LLMConfigCollection] 获取配置失败 - ID: {config_id} 未找到")
        return None
    
    def get_default_config(self) -> Optional[LLMConfig]:
        """Get the default configuration.
        
        Returns:
            Default LLMConfig or None if no configs exist
        """
        if self.default_config_id:
            config = self.get_config(self.default_config_id)
            if config:
                logger.debug(f"[LLMConfigCollection] 获取默认配置 - ID: {config.id}, Name: {config.get_display_name()}")
                return config
        
        # Fallback to first config
        if self.configs:
            config = self.configs[0]
            logger.debug(f"[LLMConfigCollection] 获取默认配置(首个) - ID: {config.id}, Name: {config.get_display_name()}")
            return config
        
        logger.warning("[LLMConfigCollection] 获取默认配置失败 - 配置列表为空")
        return None
    
    def set_default_config(self, config_id: str) -> bool:
        """Set the default configuration.
        
        Args:
            config_id: ID of configuration to set as default
            
        Returns:
            True if successful, False if config not found
        """
        if self.get_config(config_id):
            old_default = self.default_config_id
            self.default_config_id = config_id
            logger.info(f"[LLMConfigCollection] 设置默认配置 - Old: {old_default}, New: {config_id}")
            return True
        
        logger.warning(f"[LLMConfigCollection] 设置默认配置失败 - ID: {config_id} 未找到")
        return False
    
    def update_config(self, config_id: str, updates: dict) -> bool:
        """Update a configuration.
        
        Args:
            config_id: ID of configuration to update
            updates: Dictionary of fields to update
            
        Returns:
            True if updated, False if not found
        """
        config = self.get_config(config_id)
        if config:
            for key, value in updates.items():
                if hasattr(config, key):
                    old_value = getattr(config, key)
                    setattr(config, key, value)
                    logger.debug(f"[LLMConfigCollection] 更新配置字段 - ID: {config_id}, Field: {key}, Old: {old_value}, New: {value}")
            
            logger.info(f"[LLMConfigCollection] 更新配置完成 - ID: {config_id}, UpdatedFields: {list(updates.keys())}")
            return True
        
        logger.warning(f"[LLMConfigCollection] 更新配置失败 - ID: {config_id} 未找到")
        return False
    
    def get_config_names(self) -> List[tuple[str, str]]:
        """Get list of (id, display_name) tuples for all configs.
        
        Returns:
            List of (id, display_name) tuples
        """
        names = [(c.id, c.get_display_name()) for c in self.configs]
        logger.debug(f"[LLMConfigCollection] 获取配置名称列表 - 数量: {len(names)}")
        return names
    
    def is_empty(self) -> bool:
        """Check if collection is empty."""
        is_empty = len(self.configs) == 0
        if is_empty:
            logger.debug("[LLMConfigCollection] 配置集合为空")
        return is_empty
    
    def create_default_config(self) -> LLMConfig:
        """Create and add a default configuration.
        
        Returns:
            The created default configuration
        """
        config = LLMConfig(
            name="Default",
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )
        self.add_config(config)
        logger.info(f"[LLMConfigCollection] 创建默认配置 - ID: {config.id}")
        return config


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
    endpoint = PROVIDER_ENDPOINTS.get(provider, "")
    logger.debug(f"[LLMConfig] 获取默认Endpoint - Provider: {provider}, Endpoint: {endpoint}")
    return endpoint


def get_available_models(provider: LLMProvider) -> list[str]:
    """Get available models for a provider."""
    models = PROVIDER_MODELS.get(provider, [])
    logger.debug(f"[LLMConfig] 获取可用模型 - Provider: {provider}, Models: {models}")
    return models
