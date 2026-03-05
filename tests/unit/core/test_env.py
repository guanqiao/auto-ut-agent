"""Unit tests for EnvConfig."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

from pyutagent.core.env import (
    EnvConfig,
    EnvVarSpec,
    Environment,
    env_bool,
    env_int,
    env_path,
    env_value,
    get_env_config,
    init_default_env_specs,
    register_env_var,
)


class TestEnvConfig:
    """Tests for EnvConfig class."""

    def setup_method(self):
        """Reset singleton before each test."""
        EnvConfig.reset_instance()

    def test_singleton_instance(self):
        """Test singleton pattern."""
        env1 = EnvConfig.get_instance()
        env2 = EnvConfig.get_instance()
        
        assert env1 is env2

    def test_reset_instance(self):
        """Test resetting singleton."""
        env1 = EnvConfig.get_instance()
        EnvConfig.reset_instance()
        env2 = EnvConfig.get_instance()
        
        assert env1 is not env2

    def test_create_env_config(self):
        """Test creating EnvConfig."""
        env = EnvConfig(prefix="TEST_", env_var="TEST_ENV")
        
        assert env._prefix == "TEST_"
        assert env._env_var == "TEST_ENV"

    def test_get_env_value(self):
        """Test getting environment variable."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_TEST_KEY": "test_value"}):
            value = env.get("test_key")
            
            assert value == "test_value"

    def test_get_env_value_without_prefix(self):
        """Test getting environment variable without prefix."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}, clear=False):
            value = env.get("test_key")
            
            assert value == "test_value"

    def test_get_with_default(self):
        """Test getting with default value."""
        env = EnvConfig()
        
        value = env.get("nonexistent_key", default="default_value")
        
        assert value == "default_value"

    def test_get_str(self):
        """Test getting string value."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_TEST_KEY": "test_value"}):
            value = env.get_str("test_key")
            
            assert value == "test_value"

    def test_get_int(self):
        """Test getting integer value."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_COUNT": "42"}):
            value = env.get_int("count")
            
            assert value == 42
            assert isinstance(value, int)

    def test_get_int_with_default(self):
        """Test getting integer with default."""
        env = EnvConfig()
        
        value = env.get_int("nonexistent", default=10)
        
        assert value == 10

    def test_get_float(self):
        """Test getting float value."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_RATE": "3.14"}):
            value = env.get_float("rate")
            
            assert value == 3.14
            assert isinstance(value, float)

    def test_get_bool_true(self):
        """Test getting boolean true values."""
        env = EnvConfig()
        
        for true_val in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"PYUT_FLAG": true_val}):
                env.clear_cache()
                value = env.get_bool("flag")
                assert value is True

    def test_get_bool_false(self):
        """Test getting boolean false values."""
        env = EnvConfig()
        
        for false_val in ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]:
            with patch.dict(os.environ, {"PYUT_FLAG": false_val}):
                env.clear_cache()
                value = env.get_bool("flag")
                assert value is False

    def test_get_bool_with_default(self):
        """Test getting boolean with default."""
        env = EnvConfig()
        
        value = env.get_bool("nonexistent", default=True)
        
        assert value is True

    def test_get_path(self):
        """Test getting path value."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_HOME": "/home/user"}):
            value = env.get_path("home")
            
            assert isinstance(value, Path)
            assert value.parts[-2:] == ("home", "user")

    def test_get_path_with_default(self):
        """Test getting path with default."""
        env = EnvConfig()
        default_path = Path("/default/path")
        
        value = env.get_path("nonexistent", default=default_path)
        
        assert value == default_path

    def test_get_list(self):
        """Test getting list value."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_ITEMS": "a,b,c"}):
            value = env.get_list("items")
            
            assert value == ["a", "b", "c"]

    def test_get_list_with_separator(self):
        """Test getting list with custom separator."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_ITEMS": "a;b;c"}):
            value = env.get_list("items", separator=";")
            
            assert value == ["a", "b", "c"]

    def test_get_list_with_default(self):
        """Test getting list with default."""
        env = EnvConfig()
        
        value = env.get_list("nonexistent", default=["x", "y"])
        
        assert value == ["x", "y"]

    def test_is_set(self):
        """Test checking if variable is set."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_EXISTS": "value"}):
            assert env.is_set("exists") is True
            assert env.is_set("nonexistent") is False

    def test_register_spec(self):
        """Test registering a specification."""
        env = EnvConfig()
        
        env.register(
            name="custom_var",
            default="default_value",
            value_type=str,
            description="Custom variable",
            aliases=["custom_alias"],
            secret=True,
            required=True,
        )
        
        assert "custom_var" in env._specs
        assert env._specs["custom_var"].secret is True
        assert env._specs["custom_var"].required is True

    def test_get_with_spec_default(self):
        """Test getting value uses spec default."""
        env = EnvConfig()
        
        env.register("custom_var", default="default_value")
        
        value = env.get("custom_var")
        
        assert value == "default_value"

    def test_aliases(self):
        """Test variable aliases."""
        env = EnvConfig()
        
        env.register("main_var", aliases=["alias1", "alias2"])
        
        with patch.dict(os.environ, {"ALIAS1": "alias_value"}):
            value = env.get("main_var")
            
            assert value == "alias_value"

    def test_is_secret(self):
        """Test checking if variable is secret."""
        env = EnvConfig()
        
        env.register("secret_key", secret=True)
        env.register("public_key", secret=False)
        
        assert env.is_secret("secret_key") is True
        assert env.is_secret("public_key") is False

    def test_mask_value(self):
        """Test masking secret values."""
        env = EnvConfig()
        
        env.register("api_key", secret=True)
        
        masked = env.mask_value("api_key", "1234567890abcdef")
        
        assert masked == "1234***cdef"
        assert "5678" not in masked

    def test_mask_short_value(self):
        """Test masking short values."""
        env = EnvConfig()
        
        env.register("short_key", secret=True)
        
        masked = env.mask_value("short_key", "abc")
        
        assert masked == "***"

    def test_get_masked(self):
        """Test getting masked value."""
        env = EnvConfig()
        
        env.register("api_key", secret=True)
        
        with patch.dict(os.environ, {"PYUT_API_KEY": "1234567890abcdef"}):
            masked = env.get_masked("api_key")
            
            assert masked == "1234***cdef"

    def test_get_environment(self):
        """Test getting current environment."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_ENV": "production"}):
            result = env.get_environment()
            
            assert result == Environment.PRODUCTION

    def test_get_environment_default(self):
        """Test getting default environment."""
        env = EnvConfig()
        
        result = env.get_environment()
        
        assert result == Environment.DEVELOPMENT

    def test_is_development(self):
        """Test checking development environment."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_ENV": "development"}):
            assert env.is_development() is True
            assert env.is_production() is False

    def test_is_production(self):
        """Test checking production environment."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_ENV": "production"}):
            assert env.is_production() is True
            assert env.is_development() is False

    def test_is_testing(self):
        """Test checking testing environment."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_ENV": "testing"}):
            assert env.is_testing() is True

    def test_is_staging(self):
        """Test checking staging environment."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_ENV": "staging"}):
            assert env.is_staging() is True

    def test_clear_cache(self):
        """Test clearing cache."""
        env = EnvConfig()
        
        with patch.dict(os.environ, {"PYUT_KEY": "value"}):
            env.get("key")
            assert "key" in env._cache
            
            env.clear_cache()
            
            assert "key" not in env._cache

    def test_validate_required(self):
        """Test validating required variables."""
        env = EnvConfig()
        
        env.register("required_var", required=True)
        env.register("optional_var", required=False)
        
        missing = env.validate_required()
        
        assert "required_var" in missing

    def test_validate_required_all_present(self):
        """Test validating when all required are present."""
        env = EnvConfig()
        
        env.register("required_var", required=True)
        
        with patch.dict(os.environ, {"PYUT_REQUIRED_VAR": "value"}):
            missing = env.validate_required()
            
            assert len(missing) == 0

    def test_get_all_registered(self):
        """Test getting all registered specs."""
        env = EnvConfig()
        
        env.register("var1")
        env.register("var2")
        
        specs = env.get_all_registered()
        
        assert "var1" in specs
        assert "var2" in specs


class TestEnvVarSpec:
    """Tests for EnvVarSpec dataclass."""

    def test_create_spec(self):
        """Test creating a specification."""
        spec = EnvVarSpec(
            name="test_var",
            aliases=["alias1"],
            default="default_value",
            value_type=str,
            description="Test variable",
            secret=True,
            required=True,
        )
        
        assert spec.name == "test_var"
        assert spec.aliases == ["alias1"]
        assert spec.default == "default_value"
        assert spec.value_type == str
        assert spec.description == "Test variable"
        assert spec.secret is True
        assert spec.required is True


class TestEnvironment:
    """Tests for Environment enum."""

    def test_environment_values(self):
        """Test environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"


class TestHelperFunctions:
    """Tests for helper functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        EnvConfig.reset_instance()

    def test_get_env_config(self):
        """Test get_env_config helper."""
        env = get_env_config()
        
        assert isinstance(env, EnvConfig)

    def test_env_value(self):
        """Test env_value helper."""
        with patch.dict(os.environ, {"PYUT_TEST": "value"}):
            result = env_value("test")
            
            assert result == "value"

    def test_env_int(self):
        """Test env_int helper."""
        with patch.dict(os.environ, {"PYUT_COUNT": "42"}):
            result = env_int("count")
            
            assert result == 42

    def test_env_bool(self):
        """Test env_bool helper."""
        with patch.dict(os.environ, {"PYUT_FLAG": "true"}):
            result = env_bool("flag")
            
            assert result is True

    def test_env_path(self):
        """Test env_path helper."""
        with patch.dict(os.environ, {"PYUT_HOME": "/home/user"}):
            result = env_path("home")
            
            assert isinstance(result, Path)

    def test_register_env_var(self):
        """Test register_env_var helper."""
        register_env_var(
            name="custom",
            default="default_value",
            value_type=str,
            description="Custom var",
        )
        
        env = get_env_config()
        assert "custom" in env._specs

    def test_init_default_env_specs(self):
        """Test initializing default specs."""
        init_default_env_specs()
        
        env = get_env_config()
        
        assert "API_KEY" in env._specs
        assert "MODEL" in env._specs
        assert "MAX_TOKENS" in env._specs
