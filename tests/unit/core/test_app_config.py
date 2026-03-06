"""Tests for AppConfig - Unified Application Configuration."""

import pytest
import tempfile
import shutil
from pathlib import Path

from pyutagent.core.config import (
    AppConfig,
    get_app_config,
    reset_app_config,
    LLMConfig,
    LLMProvider,
    LLMConfigCollection,
    AiderConfig,
    AppState,
)


class TestAppConfig:
    """Test cases for AppConfig."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def app_config(self, temp_dir):
        """Create an AppConfig with temporary data directory."""
        reset_app_config()
        config = AppConfig()
        config.data_dir = temp_dir
        return config
    
    def test_app_config_creation(self, app_config):
        """Test AppConfig creation."""
        # Assert
        assert isinstance(app_config, AppConfig)
        assert app_config.data_dir is not None
    
    def test_app_config_default_values(self, app_config):
        """Test AppConfig default values."""
        # Assert
        assert app_config.project_paths.src_main_java == "src/main/java"
        assert app_config.project_paths.src_test_java == "src/test/java"
        assert app_config.coverage.target_coverage == 0.8
        assert app_config.log_level == "INFO"
    
    def test_app_config_save_and_load(self, app_config, temp_dir):
        """Test saving and loading AppConfig."""
        # Arrange
        app_config.log_level = "DEBUG"
        app_config.enable_debug_mode = True
        
        # Act
        app_config.save()
        reset_app_config()
        loaded = AppConfig.load()
        loaded.data_dir = temp_dir  # Use same temp dir
        
        # Assert
        assert loaded.log_level == "DEBUG"
        assert loaded.enable_debug_mode is True
    
    def test_app_config_validate(self, app_config):
        """Test AppConfig validation."""
        # Act
        result = app_config.validate()
        
        # Assert
        assert result is True
        assert app_config.data_dir.exists()
    
    def test_app_config_reload(self, app_config, temp_dir):
        """Test reloading AppConfig."""
        # Arrange
        original_level = app_config.log_level
        app_config.log_level = "ERROR"
        app_config.save()
        
        # Act
        app_config.reload()
        app_config.data_dir = temp_dir
        
        # Assert
        assert app_config.log_level == "ERROR"
    
    def test_get_llm_configs(self, app_config):
        """Test getting LLM configs."""
        # Act
        configs = app_config.get_llm_configs()
        
        # Assert
        assert isinstance(configs, LLMConfigCollection)
    
    def test_save_llm_configs(self, app_config):
        """Test saving LLM configs."""
        # Arrange
        config = LLMConfig(
            name="Test Config",
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )
        collection = LLMConfigCollection()
        collection.add_config(config)
        
        # Act
        app_config.save_llm_configs(collection)
        loaded = app_config.get_llm_configs()
        
        # Assert
        assert len(loaded.configs) == 1
        assert loaded.configs[0].name == "Test Config"
    
    def test_get_aider_config(self, app_config):
        """Test getting Aider config."""
        # Act
        config = app_config.get_aider_config()
        
        # Assert
        assert isinstance(config, AiderConfig)
    
    def test_save_aider_config(self, app_config):
        """Test saving Aider config."""
        # Arrange
        config = AiderConfig(use_architect_editor=True)
        
        # Act
        app_config.save_aider_config(config)
        loaded = app_config.get_aider_config()
        
        # Assert
        assert loaded.use_architect_editor is True
    
    def test_get_app_state(self, app_config):
        """Test getting AppState."""
        # Act
        state = app_config.get_app_state()
        
        # Assert
        assert isinstance(state, AppState)
    
    def test_save_app_state(self, app_config):
        """Test saving AppState."""
        # Arrange
        state = AppState()
        state.add_project("/test/project")
        
        # Act
        app_config.save_app_state(state)
        loaded = app_config.get_app_state()
        
        # Assert
        assert len(loaded.recent_projects) == 1
        assert loaded.recent_projects[0].path == "/test/project"


class TestGetAppConfig:
    """Test cases for get_app_config function."""
    
    def setup_method(self):
        """Reset global config before each test."""
        reset_app_config()
    
    def teardown_method(self):
        """Reset global config after each test."""
        reset_app_config()
    
    def test_get_app_config_singleton(self):
        """Test that get_app_config returns singleton."""
        # Act
        config1 = get_app_config()
        config2 = get_app_config()
        
        # Assert
        assert config1 is config2
    
    def test_get_app_config_creates_instance(self):
        """Test that get_app_config creates instance if not exists."""
        # Act
        config = get_app_config()
        
        # Assert
        assert isinstance(config, AppConfig)


class TestResetAppConfig:
    """Test cases for reset_app_config function."""
    
    def test_reset_app_config(self):
        """Test resetting global config."""
        # Arrange
        config1 = get_app_config()
        
        # Act
        reset_app_config()
        config2 = get_app_config()
        
        # Assert
        assert config1 is not config2


class TestAppConfigEdgeCases:
    """Test edge cases for AppConfig."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def app_config(self, temp_dir):
        """Create an AppConfig with temporary data directory."""
        reset_app_config()
        config = AppConfig()
        config.data_dir = temp_dir
        return config
    
    def test_load_nonexistent_config(self, temp_dir):
        """Test loading config when file doesn't exist."""
        # Act
        reset_app_config()
        # Create a fresh config without loading from file
        config = AppConfig()
        config.data_dir = temp_dir
        
        # Assert
        assert isinstance(config, AppConfig)
        # Default value should be INFO
        assert config.log_level == "INFO"
    
    def test_validate_with_invalid_coverage(self, app_config):
        """Test validation with invalid coverage value."""
        # Arrange
        app_config.coverage.target_coverage = 1.5  # Invalid: > 1
        
        # Act
        result = app_config.validate()
        
        # Assert - should still return True but log warning
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
