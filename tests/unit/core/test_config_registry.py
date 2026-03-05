"""Unit tests for ConfigRegistry."""

import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import BaseModel

from pyutagent.core.config_registry import (
    ConfigChangeType,
    ConfigChangeEvent,
    ConfigMetadata,
    ConfigPriority,
    ConfigRegistry,
    ConfigSource,
    config_value,
    get_config_registry,
)


class SampleConfig(BaseModel):
    """Sample Pydantic config for testing."""
    name: str = "default"
    value: int = 42


class TestConfigRegistry:
    """Tests for ConfigRegistry class."""

    def setup_method(self):
        """Reset singleton before each test."""
        ConfigRegistry.reset_instance()

    def test_singleton_instance(self):
        """Test singleton pattern."""
        registry1 = ConfigRegistry.get_instance()
        registry2 = ConfigRegistry.get_instance()
        
        assert registry1 is registry2

    def test_reset_instance(self):
        """Test resetting singleton."""
        registry1 = ConfigRegistry.get_instance()
        ConfigRegistry.reset_instance()
        registry2 = ConfigRegistry.get_instance()
        
        assert registry1 is not registry2

    def test_create_registry(self):
        """Test creating a new registry."""
        registry = ConfigRegistry(app_name="test_app")
        
        assert registry.app_name == "test_app"
        assert "test_app" in str(registry.config_dir)

    def test_set_and_get(self):
        """Test setting and getting values."""
        registry = ConfigRegistry()
        
        registry.set("key1", "value1")
        
        assert registry.get("key1") == "value1"

    def test_get_with_default(self):
        """Test getting with default value."""
        registry = ConfigRegistry()
        
        assert registry.get("nonexistent", "default") == "default"

    def test_has_key(self):
        """Test checking key existence."""
        registry = ConfigRegistry()
        registry.set("key1", "value1")
        
        assert registry.has("key1") is True
        assert registry.has("nonexistent") is False

    def test_contains_operator(self):
        """Test 'in' operator."""
        registry = ConfigRegistry()
        registry.set("key1", "value1")
        
        assert "key1" in registry
        assert "nonexistent" not in registry

    def test_getitem_setitem(self):
        """Test [] operators."""
        registry = ConfigRegistry()
        
        registry["key1"] = "value1"
        
        assert registry["key1"] == "value1"

    def test_getitem_raises_keyerror(self):
        """Test [] raises KeyError for missing key."""
        registry = ConfigRegistry()
        
        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_remove(self):
        """Test removing a key."""
        registry = ConfigRegistry()
        registry.set("key1", "value1")
        
        result = registry.remove("key1")
        
        assert result is True
        assert not registry.has("key1")

    def test_remove_nonexistent(self):
        """Test removing nonexistent key."""
        registry = ConfigRegistry()
        
        result = registry.remove("nonexistent")
        
        assert result is False

    def test_priority_override(self):
        """Test priority-based override."""
        registry = ConfigRegistry()
        
        registry.set("key1", "low", source=ConfigSource.FILE)
        registry.set("key1", "high", source=ConfigSource.ENVIRONMENT)
        
        assert registry.get("key1") == "high"

    def test_priority_no_override_lower(self):
        """Test that lower priority doesn't override higher."""
        registry = ConfigRegistry()
        
        registry.set("key1", "high", source=ConfigSource.ENVIRONMENT)
        registry.set("key1", "low", source=ConfigSource.FILE)
        
        assert registry.get("key1") == "high"

    def test_register_model(self):
        """Test registering a Pydantic model."""
        registry = ConfigRegistry()
        
        registry.register_model("sample", SampleConfig)
        
        config = registry.get("sample")
        
        assert isinstance(config, SampleConfig)
        assert config.name == "default"
        assert config.value == 42

    def test_register_model_with_instance(self):
        """Test registering a model with instance."""
        registry = ConfigRegistry()
        instance = SampleConfig(name="custom", value=100)
        
        registry.register_model("sample", SampleConfig, instance)
        
        config = registry.get("sample")
        
        assert config.name == "custom"
        assert config.value == 100

    def test_get_typed(self):
        """Test getting typed value."""
        registry = ConfigRegistry()
        registry.set("count", "42")
        
        value = registry.get_typed("count", int)
        
        assert value == 42
        assert isinstance(value, int)

    def test_get_typed_with_default(self):
        """Test getting typed value with default."""
        registry = ConfigRegistry()
        
        value = registry.get_typed("nonexistent", int, default=10)
        
        assert value == 10

    def test_get_all(self):
        """Test getting all values."""
        registry = ConfigRegistry()
        registry.set("key1", "value1")
        registry.set("key2", "value2")
        
        all_values = registry.get_all()
        
        assert "key1" in all_values
        assert "key2" in all_values
        assert all_values["key1"] == "value1"

    def test_get_all_excludes_secrets(self):
        """Test that get_all excludes secrets by default."""
        registry = ConfigRegistry()
        registry.set("api_key", "secret123")
        
        all_values = registry.get_all(include_secrets=False)
        
        assert "api_key" not in all_values

    def test_get_all_includes_secrets(self):
        """Test that get_all includes secrets when requested."""
        registry = ConfigRegistry()
        registry.set("api_key", "secret123")
        
        all_values = registry.get_all(include_secrets=True)
        
        assert "api_key" in all_values

    def test_load_from_file(self):
        """Test loading from JSON file."""
        registry = ConfigRegistry()
        
        with TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text(json.dumps({"key1": "value1", "key2": 42}))
            
            registry.load_from_file(config_file)
            
            assert registry.get("key1") == "value1"
            assert registry.get("key2") == 42

    def test_load_from_file_nonexistent(self):
        """Test loading from nonexistent file."""
        registry = ConfigRegistry()
        
        registry.load_from_file(Path("/nonexistent/config.json"))
        
        assert len(registry._values) == 0

    def test_save_to_file(self):
        """Test saving to JSON file."""
        registry = ConfigRegistry()
        registry.set("key1", "value1")
        registry.set("key2", 42)
        
        with TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            
            registry.save_to_file(config_file)
            
            assert config_file.exists()
            data = json.loads(config_file.read_text())
            assert data["key1"] == "value1"

    def test_load_from_env(self):
        """Test loading from environment variables."""
        with patch.dict(os.environ, {"PYUTAGENT_TEST_KEY": "test_value"}):
            ConfigRegistry.reset_instance()
            registry = ConfigRegistry.get_instance()
            registry.load_from_env()
            
            assert registry.get("test_key") == "test_value"

    def test_change_listener(self):
        """Test change listener notification."""
        registry = ConfigRegistry()
        changes = []
        
        def listener(event: ConfigChangeEvent):
            changes.append(event)
        
        registry.add_listener(listener)
        registry.set("key1", "value1")
        
        assert len(changes) == 1
        assert changes[0].key == "key1"
        assert changes[0].new_value == "value1"
        assert changes[0].change_type == ConfigChangeType.ADDED

    def test_change_listener_update(self):
        """Test change listener for updates."""
        registry = ConfigRegistry()
        changes = []
        
        def listener(event: ConfigChangeEvent):
            changes.append(event)
        
        registry.set("key1", "value1")
        registry.add_listener(listener)
        registry.set("key1", "value2")
        
        assert len(changes) == 1
        assert changes[0].change_type == ConfigChangeType.UPDATED
        assert changes[0].old_value == "value1"

    def test_change_listener_remove(self):
        """Test removing change listener."""
        registry = ConfigRegistry()
        changes = []
        
        def listener(event: ConfigChangeEvent):
            changes.append(event)
        
        registry.add_listener(listener)
        registry.remove_listener(listener)
        registry.set("key1", "value1")
        
        assert len(changes) == 0

    def test_get_metadata(self):
        """Test getting metadata."""
        registry = ConfigRegistry()
        registry.set("key1", "value1", source=ConfigSource.FILE)
        
        metadata = registry.get_metadata("key1")
        
        assert metadata is not None
        assert metadata.key == "key1"
        assert metadata.source == ConfigSource.FILE

    def test_get_stats(self):
        """Test getting statistics."""
        registry = ConfigRegistry()
        registry.set("key1", "value1")
        registry.set("key2", "value2")
        
        stats = registry.get_stats()
        
        assert stats["total_entries"] == 2
        assert stats["registered_models"] == 0

    def test_clear(self):
        """Test clearing all values."""
        registry = ConfigRegistry()
        registry.set("key1", "value1")
        registry.set("key2", "value2")
        
        registry.clear()
        
        assert len(registry._values) == 0


class TestConfigValue:
    """Tests for config_value helper function."""

    def setup_method(self):
        """Reset singleton before each test."""
        ConfigRegistry.reset_instance()

    def test_config_value_get(self):
        """Test config_value helper."""
        registry = get_config_registry()
        registry.set("test_key", "test_value")
        
        value = config_value("test_key")
        
        assert value == "test_value"

    def test_config_value_default(self):
        """Test config_value with default."""
        value = config_value("nonexistent", default="default")
        
        assert value == "default"


class TestConfigEnums:
    """Tests for configuration enums."""

    def test_config_source_values(self):
        """Test ConfigSource enum values."""
        assert ConfigSource.DEFAULT.value == 1
        assert ConfigSource.ENVIRONMENT.value == 2
        assert ConfigSource.FILE.value == 3
        assert ConfigSource.CLI.value == 4
        assert ConfigSource.RUNTIME.value == 5

    def test_config_priority_values(self):
        """Test ConfigPriority enum values."""
        assert ConfigPriority.DEFAULT.value == 10
        assert ConfigPriority.FILE.value == 30
        assert ConfigPriority.ENVIRONMENT.value == 50
        assert ConfigPriority.CLI.value == 70
        assert ConfigPriority.RUNTIME.value == 90

    def test_priority_ordering(self):
        """Test priority ordering."""
        assert ConfigPriority.RUNTIME.value > ConfigPriority.CLI.value
        assert ConfigPriority.CLI.value > ConfigPriority.ENVIRONMENT.value
        assert ConfigPriority.ENVIRONMENT.value > ConfigPriority.FILE.value
        assert ConfigPriority.FILE.value > ConfigPriority.DEFAULT.value


class TestConfigMetadata:
    """Tests for ConfigMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = ConfigMetadata(
            key="test_key",
            source=ConfigSource.FILE,
            priority=ConfigPriority.FILE,
        )
        
        assert metadata.key == "test_key"
        assert metadata.source == ConfigSource.FILE
        assert metadata.priority == ConfigPriority.FILE
        assert isinstance(metadata.timestamp, datetime)

    def test_metadata_version(self):
        """Test metadata version."""
        metadata = ConfigMetadata(
            key="test",
            source=ConfigSource.DEFAULT,
            priority=ConfigPriority.DEFAULT,
        )
        
        assert metadata.version == 1


class TestConfigChangeEvent:
    """Tests for ConfigChangeEvent dataclass."""

    def test_create_event(self):
        """Test creating change event."""
        event = ConfigChangeEvent(
            key="test_key",
            old_value="old",
            new_value="new",
            change_type=ConfigChangeType.UPDATED,
        )
        
        assert event.key == "test_key"
        assert event.old_value == "old"
        assert event.new_value == "new"
        assert event.change_type == ConfigChangeType.UPDATED
        assert isinstance(event.timestamp, datetime)

    def test_change_types(self):
        """Test change type enum values."""
        assert ConfigChangeType.ADDED.value == 1
        assert ConfigChangeType.UPDATED.value == 2
        assert ConfigChangeType.REMOVED.value == 3
