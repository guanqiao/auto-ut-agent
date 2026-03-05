"""Tests for Capability Registry.

This module tests the capability management system including:
- Capability registration and discovery
- Provider matching and scoring
- Performance tracking
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from pyutagent.agent.capability_registry import (
    CapabilityType,
    CapabilityScore,
    AgentCapability,
    CapabilityProvider,
    CapabilityRegistry,
    get_capability_registry,
)


class TestCapabilityScore:
    """Tests for CapabilityScore."""
    
    def test_default_scores(self):
        """Test default score values."""
        score = CapabilityScore()
        assert score.base_score == 0.0
        assert score.success_rate == 0.0
        assert score.speed_factor == 1.0
        assert score.quality_factor == 1.0
    
    def test_total_score_calculation(self):
        """Test total score calculation."""
        score = CapabilityScore(
            base_score=1.0,
            success_rate=0.8,
            speed_factor=1.0,
            quality_factor=1.0,
        )
        
        expected = 1.0 * 0.3 + 0.8 * 0.4 + 1.0 * 0.15 + 1.0 * 0.15
        assert abs(score.total_score - expected) < 0.001
    
    def test_total_score_with_low_values(self):
        """Test total score with low values."""
        score = CapabilityScore(
            base_score=0.5,
            success_rate=0.5,
            speed_factor=0.5,
            quality_factor=0.5,
        )
        
        assert score.total_score == 0.5


class TestAgentCapability:
    """Tests for AgentCapability."""
    
    def test_capability_creation(self):
        """Test creating a capability."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
            description="Test capability",
        )
        
        assert cap.name == "test_cap"
        assert cap.capability_type == CapabilityType.CODE_GENERATION
        assert cap.description == "Test capability"
        assert isinstance(cap.created_at, datetime)
    
    def test_capability_with_handler(self):
        """Test capability with handler."""
        handler = MagicMock(return_value="result")
        
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.TEST_GENERATION,
            handler=handler,
        )
        
        assert cap.handler is not None
        assert cap.handler() == "result"
    
    def test_update_score_success(self):
        """Test score update on success."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        initial_rate = cap.score.success_rate
        cap.update_score(success=True, execution_time=1.0)
        
        assert cap.score.success_rate > initial_rate
        assert abs(cap.score.success_rate - 0.1) < 0.001
    
    def test_update_score_failure(self):
        """Test score update on failure."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap.score.success_rate = 0.8
        
        cap.update_score(success=False, execution_time=1.0)
        
        assert abs(cap.score.success_rate - 0.72) < 0.001
    
    def test_update_score_with_quality(self):
        """Test score update with quality score."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap.score.quality_factor = 0.5
        
        cap.update_score(success=True, execution_time=1.0, quality_score=1.0)
        
        expected_quality = 0.5 * 0.8 + 1.0 * 0.2
        assert abs(cap.score.quality_factor - expected_quality) < 0.001
    
    def test_to_dict(self):
        """Test capability serialization."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
            description="Test",
            metadata={"key": "value"},
        )
        
        data = cap.to_dict()
        
        assert data["name"] == "test_cap"
        assert data["capability_type"] == "code_generation"
        assert data["description"] == "Test"
        assert "score" in data
        assert data["metadata"]["key"] == "value"


class TestCapabilityProvider:
    """Tests for CapabilityProvider."""
    
    def test_provider_creation(self):
        """Test creating a provider."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        provider = CapabilityProvider(
            agent_id="agent_1",
            agent_name="Test Agent",
            capability=cap,
        )
        
        assert provider.agent_id == "agent_1"
        assert provider.agent_name == "Test Agent"
        assert provider.is_available is True
        assert provider.current_load == 0
    
    def test_load_factor(self):
        """Test load factor calculation."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        provider = CapabilityProvider(
            agent_id="agent_1",
            agent_name="Test Agent",
            capability=cap,
            max_load=4,
        )
        
        assert provider.load_factor == 0.0
        
        provider.current_load = 2
        assert provider.load_factor == 0.5
        
        provider.current_load = 4
        assert provider.load_factor == 1.0
    
    def test_load_factor_zero_max(self):
        """Test load factor with zero max load."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        provider = CapabilityProvider(
            agent_id="agent_1",
            agent_name="Test Agent",
            capability=cap,
            max_load=0,
        )
        
        assert provider.load_factor == 1.0
    
    def test_effective_score(self):
        """Test effective score calculation."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap.score.base_score = 1.0
        cap.score.success_rate = 1.0
        
        provider = CapabilityProvider(
            agent_id="agent_1",
            agent_name="Test Agent",
            capability=cap,
            current_load=0,
            max_load=4,
        )
        
        base_score = cap.score.total_score
        assert provider.effective_score == base_score
        
        provider.current_load = 4
        expected = base_score - 0.3
        assert abs(provider.effective_score - expected) < 0.001


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        CapabilityRegistry.reset_instance()
        self.registry = CapabilityRegistry()
        yield
        CapabilityRegistry.reset_instance()
    
    def test_singleton(self):
        """Test singleton pattern."""
        instance1 = CapabilityRegistry.get_instance()
        instance2 = CapabilityRegistry.get_instance()
        
        assert instance1 is instance2
    
    def test_reset_instance(self):
        """Test resetting singleton."""
        instance1 = CapabilityRegistry.get_instance()
        CapabilityRegistry.reset_instance()
        instance2 = CapabilityRegistry.get_instance()
        
        assert instance1 is not instance2
    
    def test_register_capability(self):
        """Test registering a capability."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        self.registry.register("agent_1", cap, "Test Agent")
        
        assert "agent_1" in self.registry._capabilities
        assert "test_cap" in self.registry._capabilities["agent_1"]
    
    def test_register_multiple_capabilities(self):
        """Test registering multiple capabilities."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.TEST_GENERATION,
        )
        
        self.registry.register("agent_1", cap1)
        self.registry.register("agent_1", cap2)
        
        assert len(self.registry._capabilities["agent_1"]) == 2
    
    def test_unregister_capability(self):
        """Test unregistering a capability."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        self.registry.register("agent_1", cap)
        result = self.registry.unregister("agent_1", "test_cap")
        
        assert result is True
        assert "agent_1" not in self.registry._capabilities
    
    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent capability."""
        result = self.registry.unregister("agent_1", "nonexistent")
        assert result is False
    
    def test_unregister_agent(self):
        """Test unregistering all capabilities for an agent."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.TEST_GENERATION,
        )
        
        self.registry.register("agent_1", cap1)
        self.registry.register("agent_1", cap2)
        
        count = self.registry.unregister_agent("agent_1")
        
        assert count == 2
        assert "agent_1" not in self.registry._capabilities
    
    def test_get_capability(self):
        """Test getting a specific capability."""
        cap = AgentCapability(
            name="test_cap",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        self.registry.register("agent_1", cap)
        
        result = self.registry.get_capability("agent_1", "test_cap")
        assert result is cap
        
        result = self.registry.get_capability("agent_1", "nonexistent")
        assert result is None
    
    def test_get_agent_capabilities(self):
        """Test getting all capabilities for an agent."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.TEST_GENERATION,
        )
        
        self.registry.register("agent_1", cap1)
        self.registry.register("agent_1", cap2)
        
        caps = self.registry.get_agent_capabilities("agent_1")
        
        assert len(caps) == 2
    
    def test_discover_capabilities(self):
        """Test discovering capabilities by type."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap1.score.base_score = 0.8
        
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2.score.base_score = 0.5
        
        self.registry.register("agent_1", cap1)
        self.registry.register("agent_2", cap2)
        
        results = self.registry.discover(CapabilityType.CODE_GENERATION)
        
        assert len(results) == 2
        assert results[0].name == "cap1"
    
    def test_discover_with_min_score(self):
        """Test discovering with minimum score filter."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap1.score.base_score = 0.8
        
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2.score.base_score = 0.3
        
        self.registry.register("agent_1", cap1)
        self.registry.register("agent_2", cap2)
        
        results = self.registry.discover(
            CapabilityType.CODE_GENERATION,
            min_score=0.5,
        )
        
        assert len(results) == 1
        assert results[0].name == "cap1"
    
    def test_find_best_provider(self):
        """Test finding the best provider."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap1.score.base_score = 0.9
        
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2.score.base_score = 0.5
        
        self.registry.register("agent_1", cap1, priority=1)
        self.registry.register("agent_2", cap2, priority=2)
        
        provider = self.registry.find_best_provider(CapabilityType.CODE_GENERATION)
        
        assert provider is not None
        assert provider.agent_id == "agent_2"
    
    def test_find_best_provider_with_exclude(self):
        """Test finding best provider with exclusions."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap1.score.base_score = 0.9
        
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2.score.base_score = 0.5
        
        self.registry.register("agent_1", cap1)
        self.registry.register("agent_2", cap2)
        
        provider = self.registry.find_best_provider(
            CapabilityType.CODE_GENERATION,
            exclude_agents={"agent_1"},
        )
        
        assert provider is not None
        assert provider.agent_id == "agent_2"
    
    def test_find_best_provider_none_available(self):
        """Test finding provider when none available."""
        cap = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        self.registry.register("agent_1", cap)
        self.registry.set_provider_availability(
            "agent_1",
            CapabilityType.CODE_GENERATION,
            False,
        )
        
        provider = self.registry.find_best_provider(CapabilityType.CODE_GENERATION)
        
        assert provider is None
    
    def test_find_all_providers(self):
        """Test finding all providers."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap1.score.base_score = 0.9
        
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2.score.base_score = 0.5
        
        self.registry.register("agent_1", cap1)
        self.registry.register("agent_2", cap2)
        
        providers = self.registry.find_all_providers(CapabilityType.CODE_GENERATION)
        
        assert len(providers) == 2
        assert providers[0].agent_id == "agent_1"
    
    def test_update_provider_load(self):
        """Test updating provider load."""
        cap = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        self.registry.register("agent_1", cap, max_load=4)
        
        self.registry.update_provider_load(
            "agent_1",
            CapabilityType.CODE_GENERATION,
            2,
        )
        
        providers = self.registry.find_all_providers(CapabilityType.CODE_GENERATION)
        assert providers[0].current_load == 2
    
    def test_update_provider_load_bounds(self):
        """Test provider load bounds."""
        cap = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        self.registry.register("agent_1", cap, max_load=2)
        
        self.registry.update_provider_load(
            "agent_1",
            CapabilityType.CODE_GENERATION,
            5,
        )
        
        providers = self.registry.find_all_providers(CapabilityType.CODE_GENERATION)
        assert providers[0].current_load == 2
        
        self.registry.update_provider_load(
            "agent_1",
            CapabilityType.CODE_GENERATION,
            -10,
        )
        
        providers = self.registry.find_all_providers(CapabilityType.CODE_GENERATION)
        assert providers[0].current_load == 0
    
    def test_record_execution(self):
        """Test recording execution results."""
        cap = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        self.registry.register("agent_1", cap)
        
        self.registry.record_execution(
            "agent_1",
            "cap1",
            success=True,
            execution_time=1.0,
        )
        
        result = self.registry.get_capability("agent_1", "cap1")
        assert result.score.success_rate > 0
    
    def test_get_stats(self):
        """Test getting registry statistics."""
        cap1 = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2 = AgentCapability(
            name="cap2",
            capability_type=CapabilityType.TEST_GENERATION,
        )
        
        self.registry.register("agent_1", cap1)
        self.registry.register("agent_2", cap2)
        
        stats = self.registry.get_stats()
        
        assert stats["total_agents"] == 2
        assert stats["total_capabilities"] == 2
    
    def test_clear(self):
        """Test clearing registry."""
        cap = AgentCapability(
            name="cap1",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        
        self.registry.register("agent_1", cap)
        self.registry.clear()
        
        assert len(self.registry._capabilities) == 0


class TestGetCapabilityRegistry:
    """Tests for get_capability_registry function."""
    
    def test_get_registry(self):
        """Test getting the global registry."""
        CapabilityRegistry.reset_instance()
        
        registry = get_capability_registry()
        
        assert isinstance(registry, CapabilityRegistry)
        
        CapabilityRegistry.reset_instance()


class TestCapabilityTypes:
    """Tests for all capability types."""
    
    def test_all_capability_types_exist(self):
        """Test that all capability types are defined."""
        expected_types = [
            "CODE_GENERATION",
            "CODE_ANALYSIS",
            "TEST_GENERATION",
            "TEST_EXECUTION",
            "TEST_FIXING",
            "CODE_REFACTORING",
            "BUG_FIXING",
            "DOCUMENTATION",
            "BUILD",
            "DEPLOYMENT",
            "SEARCH",
            "PLANNING",
            "COORDINATION",
            "REVIEW",
            "OPTIMIZATION",
        ]
        
        for type_name in expected_types:
            assert hasattr(CapabilityType, type_name)
    
    def test_capability_type_values(self):
        """Test capability type string values."""
        assert CapabilityType.CODE_GENERATION.value == "code_generation"
        assert CapabilityType.TEST_GENERATION.value == "test_generation"
        assert CapabilityType.PLANNING.value == "planning"
