"""Integration tests for core modules - 核心模块集成测试

This module tests the integration between:
- DIContainer and EventBus
- AppConfig and other modules
- Unified interfaces
"""
import asyncio
import pytest
from pathlib import Path
from typing import Any, Dict, List

from pyutagent.core.container import DIContainer
from pyutagent.core.event_bus import EventBus, Event
from pyutagent.core.config import AppConfig, Settings
from pyutagent.core.interfaces import (
    IAgent, ITool, IContext, IEventBus,
    AbstractAgent, AbstractTool, ExecutionResult, Task
)
from pyutagent.core.messaging import UnifiedMessageBus


class TestDIContainerEventBusIntegration:
    """Test DIContainer and EventBus integration."""

    def setup_method(self):
        """Setup before each test."""
        DIContainer.clear()

    def teardown_method(self):
        """Cleanup after each test."""
        DIContainer.clear()

    def test_register_event_bus_in_container(self):
        """Test registering EventBus in DIContainer."""
        event_bus = EventBus()
        DIContainer.register_singleton(EventBus, event_bus)

        # Resolve from container
        resolved = DIContainer.resolve(EventBus)
        assert resolved is event_bus

    def test_resolve_event_bus_as_interface(self):
        """Test resolving EventBus as IEventBus interface."""
        event_bus = EventBus()
        DIContainer.register_singleton(IEventBus, event_bus)

        # Resolve as interface
        resolved = DIContainer.resolve(IEventBus)
        assert isinstance(resolved, EventBus)

    @pytest.mark.asyncio
    async def test_event_bus_publishes_after_di_resolution(self):
        """Test that EventBus works after being resolved from DI."""
        event_bus = EventBus()
        DIContainer.register_singleton(EventBus, event_bus)

        # Resolve and use
        resolved_bus = DIContainer.resolve(EventBus)

        results = []

        async def handler(event):
            results.append("handled")

        await resolved_bus.subscribe(Event, handler)
        await resolved_bus.publish(Event())

        assert "handled" in results

    def test_container_provides_message_bus_to_event_bus(self):
        """Test that DIContainer can provide MessageBus to EventBus."""
        message_bus = UnifiedMessageBus()
        DIContainer.register_singleton(UnifiedMessageBus, message_bus)

        # Create EventBus with MessageBus from container
        resolved_message_bus = DIContainer.resolve(UnifiedMessageBus)
        event_bus = EventBus(resolved_message_bus)

        assert event_bus._message_bus is resolved_message_bus


class TestAppConfigIntegration:
    """Test AppConfig integration with other modules."""

    def setup_method(self):
        """Setup before each test."""
        DIContainer.clear()

    def teardown_method(self):
        """Cleanup after each test."""
        DIContainer.clear()

    def test_register_app_config_in_container(self):
        """Test registering AppConfig in DIContainer."""
        config = AppConfig()
        DIContainer.register_singleton(AppConfig, config)

        # Resolve from container
        resolved = DIContainer.resolve(AppConfig)
        assert resolved is config

    def test_app_config_as_settings_interface(self):
        """Test AppConfig can be resolved as Settings."""
        config = AppConfig()
        DIContainer.register_singleton(Settings, config)

        # Resolve as Settings
        resolved = DIContainer.resolve(Settings)
        assert isinstance(resolved, AppConfig)

    def test_config_values_accessible_via_container(self):
        """Test that config values are accessible after DI resolution."""
        config = AppConfig()
        config.data_dir = Path("/test/data")
        DIContainer.register_singleton(AppConfig, config)

        # Resolve and access values
        resolved = DIContainer.resolve(AppConfig)
        assert resolved.data_dir == Path("/test/data")


class TestUnifiedInterfacesIntegration:
    """Test unified interfaces integration."""

    def test_abstract_agent_can_be_registered(self):
        """Test that AbstractAgent implementations can be registered in DI."""
        class TestAgent(AbstractAgent):
            async def execute(self, task: Task) -> ExecutionResult:
                return ExecutionResult(success=True)

            async def plan(self, goal: str) -> List[Task]:
                return []

        agent = TestAgent("test_agent", "test_type")
        DIContainer.register_singleton(IAgent, agent)

        # Resolve as interface
        resolved = DIContainer.resolve(IAgent)
        assert isinstance(resolved, TestAgent)
        assert resolved.name == "test_agent"

    def test_abstract_tool_can_be_registered(self):
        """Test that AbstractTool implementations can be registered in DI."""
        class TestTool(AbstractTool):
            async def execute(self, inputs: Dict[str, Any]) -> ExecutionResult:
                return ExecutionResult(success=True)

        tool = TestTool("test_tool", "A test tool")
        DIContainer.register_singleton(ITool, tool)

        # Resolve as interface
        resolved = DIContainer.resolve(ITool)
        assert isinstance(resolved, TestTool)
        assert resolved.name == "test_tool"

    @pytest.mark.asyncio
    async def test_agent_uses_event_bus_from_container(self):
        """Test that Agent can use EventBus from DI container."""
        event_bus = EventBus()
        DIContainer.register_singleton(EventBus, event_bus)

        class EventDrivenAgent(AbstractAgent):
            def __init__(self):
                super().__init__("event_agent", "event_type")
                self.event_bus = DIContainer.resolve(EventBus)

            async def execute(self, task: Task) -> ExecutionResult:
                # Publish event during execution
                await self.event_bus.publish(Event())
                return ExecutionResult(success=True)

            async def plan(self, goal: str) -> List[Task]:
                return []

        agent = EventDrivenAgent()

        # Subscribe to events
        results = []

        async def handler(event):
            results.append("event_received")

        await event_bus.subscribe(Event, handler)

        # Execute agent
        await agent.execute(Task(id="1", name="test"))

        assert "event_received" in results


class TestModuleLifecycleIntegration:
    """Test module lifecycle integration."""

    def setup_method(self):
        """Setup before each test."""
        DIContainer.clear()

    def teardown_method(self):
        """Cleanup after each test."""
        DIContainer.clear()

    def test_full_initialization_sequence(self):
        """Test full initialization sequence of core modules."""
        # 1. Create and register EventBus
        event_bus = EventBus()
        DIContainer.register_singleton(EventBus, event_bus)

        # 2. Create and register AppConfig
        config = AppConfig()
        DIContainer.register_singleton(AppConfig, config)

        # 3. Verify all can be resolved
        assert DIContainer.resolve(EventBus) is event_bus
        assert DIContainer.resolve(AppConfig) is config

    @pytest.mark.asyncio
    async def test_cleanup_sequence(self):
        """Test cleanup sequence of core modules."""
        # Setup
        event_bus = EventBus()
        DIContainer.register_singleton(EventBus, event_bus)

        config = AppConfig()
        DIContainer.register_singleton(AppConfig, config)

        # Verify setup
        assert DIContainer.is_registered(EventBus)
        assert DIContainer.is_registered(AppConfig)

        # Cleanup
        DIContainer.clear()

        # Verify cleanup
        assert not DIContainer.is_registered(EventBus)
        assert not DIContainer.is_registered(AppConfig)


class TestEndToEndScenarios:
    """Test end-to-end scenarios."""

    def setup_method(self):
        """Setup before each test."""
        DIContainer.clear()

    def teardown_method(self):
        """Cleanup after each test."""
        DIContainer.clear()

    @pytest.mark.asyncio
    async def test_event_driven_architecture(self):
        """Test event-driven architecture with DI."""
        # Setup: Register EventBus
        event_bus = EventBus()
        DIContainer.register_singleton(EventBus, event_bus)

        # Track events
        events_received = []

        async def event_handler(event):
            events_received.append(event.get_event_type())

        await event_bus.subscribe(Event, event_handler)

        # Simulate: Component publishes event
        resolved_bus = DIContainer.resolve(EventBus)
        await resolved_bus.publish(Event())

        # Verify: Event was received
        assert len(events_received) == 1
        assert events_received[0] == "Event"

    def test_configuration_driven_behavior(self):
        """Test that configuration drives behavior."""
        # Setup: Create config with specific values
        config = AppConfig()
        config.data_dir = Path("/custom/data")
        DIContainer.register_singleton(AppConfig, config)

        # Component uses config
        class ConfigDrivenComponent:
            def __init__(self):
                self.config = DIContainer.resolve(AppConfig)

            def get_data_path(self):
                return self.config.data_dir

        component = ConfigDrivenComponent()
        assert component.get_data_path() == Path("/custom/data")

    @pytest.mark.asyncio
    async def test_multiple_components_communicate_via_events(self):
        """Test multiple components communicating via events."""
        # Setup: Shared EventBus
        event_bus = EventBus()
        DIContainer.register_singleton(EventBus, event_bus)

        # Component A: Publishes events
        class ComponentA:
            def __init__(self):
                self.bus = DIContainer.resolve(EventBus)

            async def do_something(self):
                await self.bus.publish(Event())

        # Component B: Listens to events
        class ComponentB:
            def __init__(self):
                self.bus = DIContainer.resolve(EventBus)
                self.received = []

            async def setup_listener(self):
                await self.bus.subscribe(Event, self.on_event)

            async def on_event(self, event):
                self.received.append("received")

        # Setup
        comp_a = ComponentA()
        comp_b = ComponentB()
        await comp_b.setup_listener()

        # Action
        await comp_a.do_something()

        # Verify
        assert len(comp_b.received) == 1
