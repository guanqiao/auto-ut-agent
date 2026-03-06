"""Tests for unified interface contracts."""
import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from pyutagent.core.interfaces import (
    # Enums
    AgentState,
    ExecutionStatus,
    CapabilityType,
    # Data Classes
    ExecutionResult,
    Capability,
    Task,
    # Core Protocols
    IExecutable,
    IInitializable,
    IStateful,
    ICapable,
    # Agent Protocols
    IAgent,
    ISubAgent,
    # Tool Protocols
    ITool,
    IToolRegistry,
    # Context Protocols
    IContext,
    IProjectContext,
    # Memory Protocols
    IMemory,
    IWorkingMemory,
    ILongTermMemory,
    # LLM Protocols
    ILLMClient,
    # Event Protocols
    IEvent,
    IEventBus,
    # Skill Protocols
    ISkill,
    # Abstract Base Classes
    AbstractAgent,
    AbstractTool,
)


class TestEnums:
    """Test enum definitions."""
    
    def test_agent_state_values(self):
        """Test AgentState enum has expected values."""
        assert AgentState.IDLE is not None
        assert AgentState.INITIALIZING is not None
        assert AgentState.PLANNING is not None
        assert AgentState.EXECUTING is not None
        assert AgentState.PAUSED is not None
        assert AgentState.STOPPING is not None
        assert AgentState.STOPPED is not None
        assert AgentState.COMPLETED is not None
        assert AgentState.FAILED is not None
    
    def test_execution_status_values(self):
        """Test ExecutionStatus enum has expected values."""
        assert ExecutionStatus.PENDING is not None
        assert ExecutionStatus.RUNNING is not None
        assert ExecutionStatus.SUCCESS is not None
        assert ExecutionStatus.FAILURE is not None
        assert ExecutionStatus.CANCELLED is not None
        assert ExecutionStatus.TIMEOUT is not None
    
    def test_capability_type_values(self):
        """Test CapabilityType enum has expected values."""
        assert CapabilityType.CODE_GENERATION.value == "code_generation"
        assert CapabilityType.TEST_GENERATION.value == "test_generation"
        assert CapabilityType.CODE_REVIEW.value == "code_review"
        assert CapabilityType.REFACTORING.value == "refactoring"
        assert CapabilityType.DEBUGGING.value == "debugging"
        assert CapabilityType.ANALYSIS.value == "analysis"
        assert CapabilityType.DOCUMENTATION.value == "documentation"
        assert CapabilityType.PLANNING.value == "planning"
        assert CapabilityType.EXECUTION.value == "execution"
        assert CapabilityType.COORDINATION.value == "coordination"


class TestDataClasses:
    """Test data class definitions."""
    
    def test_execution_result_creation(self):
        """Test ExecutionResult can be created."""
        result = ExecutionResult(success=True, data="test", message="ok")
        assert result.success is True
        assert result.data == "test"
        assert result.message == "ok"
        assert result.error is None
        assert result.duration_ms == 0
        assert result.metadata == {}
    
    def test_execution_result_defaults(self):
        """Test ExecutionResult has correct defaults."""
        result = ExecutionResult(success=False)
        assert result.data is None
        assert result.message == ""
        assert result.error is None
        assert result.duration_ms == 0
        assert result.metadata == {}
    
    def test_capability_creation(self):
        """Test Capability can be created."""
        cap = Capability(
            name="test_cap",
            type=CapabilityType.CODE_GENERATION,
            description="Test capability",
            parameters={"key": "value"}
        )
        assert cap.name == "test_cap"
        assert cap.type == CapabilityType.CODE_GENERATION
        assert cap.description == "Test capability"
        assert cap.parameters == {"key": "value"}
    
    def test_capability_defaults(self):
        """Test Capability has correct defaults."""
        cap = Capability(name="test", type=CapabilityType.EXECUTION)
        assert cap.description == ""
        assert cap.parameters == {}
    
    def test_task_creation(self):
        """Test Task can be created."""
        task = Task(
            id="task-1",
            name="Test Task",
            description="A test task",
            inputs={"input": "value"},
            expected_outputs=["output1"],
            priority=1,
            timeout_seconds=30.0
        )
        assert task.id == "task-1"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.inputs == {"input": "value"}
        assert task.expected_outputs == ["output1"]
        assert task.priority == 1
        assert task.timeout_seconds == 30.0
    
    def test_task_defaults(self):
        """Test Task has correct defaults."""
        task = Task(id="task-1", name="Test")
        assert task.description == ""
        assert task.inputs == {}
        assert task.expected_outputs == []
        assert task.priority == 0
        assert task.timeout_seconds is None


class TestAbstractAgent:
    """Test AbstractAgent base class."""
    
    def test_abstract_agent_is_abstract(self):
        """Test AbstractAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractAgent("test", "test_type")
    
    def test_abstract_agent_concrete_implementation(self):
        """Test concrete AbstractAgent implementation."""
        class ConcreteAgent(AbstractAgent):
            async def execute(self, task: Task) -> ExecutionResult:
                return ExecutionResult(success=True)
            
            async def plan(self, goal: str) -> List[Task]:
                return []
        
        agent = ConcreteAgent("test_agent", "test_type")
        assert agent.name == "test_agent"
        assert agent.agent_type == "test_type"
        assert agent.state == AgentState.IDLE
        assert agent.capabilities == []
        assert agent.is_initialized is False
    
    def test_abstract_agent_state_transitions(self):
        """Test AbstractAgent state transitions."""
        class ConcreteAgent(AbstractAgent):
            async def execute(self, task: Task) -> ExecutionResult:
                return ExecutionResult(success=True)
            
            async def plan(self, goal: str) -> List[Task]:
                return []
        
        agent = ConcreteAgent("test", "type")
        
        # IDLE -> INITIALIZING
        assert agent.can_transition_to(AgentState.INITIALIZING) is True
        # IDLE -> PLANNING
        assert agent.can_transition_to(AgentState.PLANNING) is True
        # IDLE -> EXECUTING (invalid)
        assert agent.can_transition_to(AgentState.EXECUTING) is False
    
    @pytest.mark.asyncio
    async def test_abstract_agent_lifecycle(self):
        """Test AbstractAgent lifecycle methods."""
        class ConcreteAgent(AbstractAgent):
            async def execute(self, task: Task) -> ExecutionResult:
                return ExecutionResult(success=True)

            async def plan(self, goal: str) -> List[Task]:
                return []

        agent = ConcreteAgent("test", "type")

        # Initialize
        assert agent.is_initialized is False
        result = await agent.initialize()
        assert result is True
        assert agent.is_initialized is True

        # Shutdown
        await agent.shutdown()
        assert agent.is_initialized is False
    
    def test_abstract_agent_control_methods(self):
        """Test AbstractAgent control methods."""
        class ConcreteAgent(AbstractAgent):
            async def execute(self, task: Task) -> ExecutionResult:
                return ExecutionResult(success=True)
            
            async def plan(self, goal: str) -> List[Task]:
                return []
        
        agent = ConcreteAgent("test", "type")
        
        # Can't pause from IDLE
        assert agent.pause() is False
        
        # Can't resume from IDLE
        assert agent.resume() is False
        
        # Can't stop from IDLE
        assert agent.stop() is False


class TestAbstractTool:
    """Test AbstractTool base class."""
    
    def test_abstract_tool_is_abstract(self):
        """Test AbstractTool cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractTool("test", "description")
    
    def test_abstract_tool_concrete_implementation(self):
        """Test concrete AbstractTool implementation."""
        class ConcreteTool(AbstractTool):
            async def execute(self, inputs: Dict[str, Any]) -> ExecutionResult:
                return ExecutionResult(success=True)
        
        tool = ConcreteTool("test_tool", "A test tool")
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.parameters == {}
    
    def test_abstract_tool_input_validation(self):
        """Test AbstractTool input validation."""
        class ConcreteTool(AbstractTool):
            def __init__(self):
                super().__init__("test", "desc")
                self._parameters = {
                    "required_param": {"required": True},
                    "optional_param": {"required": False}
                }
            
            async def execute(self, inputs: Dict[str, Any]) -> ExecutionResult:
                return ExecutionResult(success=True)
        
        tool = ConcreteTool()
        
        # Missing required parameter
        errors = tool.validate_inputs({})
        assert len(errors) == 1
        assert "required_param" in errors[0]
        
        # With required parameter
        errors = tool.validate_inputs({"required_param": "value"})
        assert len(errors) == 0


class TestProtocolRuntimeCheck:
    """Test protocol runtime checkability."""
    
    def test_iagent_runtime_check(self):
        """Test IAgent protocol can be checked at runtime."""
        class MockAgent:
            @property
            def name(self) -> str:
                return "mock"
            
            @property
            def agent_type(self) -> str:
                return "mock_type"
            
            @property
            def state(self) -> AgentState:
                return AgentState.IDLE
            
            @property
            def capabilities(self) -> List[Capability]:
                return []
            
            @property
            def is_initialized(self) -> bool:
                return True
            
            async def execute(self, task: Task) -> ExecutionResult:
                return ExecutionResult(success=True)
            
            async def plan(self, goal: str) -> List[Task]:
                return []
            
            async def initialize(self) -> bool:
                return True
            
            async def shutdown(self) -> None:
                pass
            
            def can_transition_to(self, new_state: AgentState) -> bool:
                return True
            
            def has_capability(self, capability_type: CapabilityType) -> bool:
                return False
            
            def pause(self) -> bool:
                return True
            
            def resume(self) -> bool:
                return True
            
            def stop(self) -> bool:
                return True
        
        agent = MockAgent()
        assert isinstance(agent, IAgent)
    
    def test_itool_runtime_check(self):
        """Test ITool protocol can be checked at runtime."""
        class MockTool:
            @property
            def name(self) -> str:
                return "mock_tool"
            
            @property
            def description(self) -> str:
                return "Mock tool"
            
            @property
            def parameters(self) -> Dict[str, Any]:
                return {}
            
            def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
                return []
            
            async def execute(self, inputs: Dict[str, Any]) -> ExecutionResult:
                return ExecutionResult(success=True)
        
        tool = MockTool()
        assert isinstance(tool, ITool)
    
    def test_icontext_runtime_check(self):
        """Test IContext protocol can be checked at runtime."""
        class MockContext:
            def get(self, key: str, default: Any = None) -> Any:
                return default
            
            def set(self, key: str, value: Any) -> None:
                pass
            
            def has(self, key: str) -> bool:
                return False
            
            def delete(self, key: str) -> bool:
                return False
            
            def keys(self):
                return set()
            
            def create_child(self):
                return MockContext()
        
        ctx = MockContext()
        assert isinstance(ctx, IContext)


class TestInterfaceExports:
    """Test that all interfaces are properly exported."""
    
    def test_all_enums_exported(self):
        """Test all enums are exported."""
        from pyutagent.core import AgentState, ExecutionStatus, CapabilityType
        assert AgentState is not None
        assert ExecutionStatus is not None
        assert CapabilityType is not None
    
    def test_all_data_classes_exported(self):
        """Test all data classes are exported."""
        from pyutagent.core import ExecutionResult, Capability, Task
        assert ExecutionResult is not None
        assert Capability is not None
        assert Task is not None
    
    def test_all_protocols_exported(self):
        """Test all protocols are exported."""
        from pyutagent.core import (
            IExecutable, IInitializable, IStateful, ICapable,
            IAgent, ISubAgent, ITool, IToolRegistry,
            IContext, IProjectContext,
            IMemory, IWorkingMemory, ILongTermMemory,
            ILLMClient, IEvent, IEventBus, ISkill,
        )
        assert IExecutable is not None
        assert IInitializable is not None
        assert IStateful is not None
        assert ICapable is not None
        assert IAgent is not None
        assert ISubAgent is not None
        assert ITool is not None
        assert IToolRegistry is not None
        assert IContext is not None
        assert IProjectContext is not None
        assert IMemory is not None
        assert IWorkingMemory is not None
        assert ILongTermMemory is not None
        assert ILLMClient is not None
        assert IEvent is not None
        assert IEventBus is not None
        assert ISkill is not None
    
    def test_all_abstract_classes_exported(self):
        """Test all abstract classes are exported."""
        from pyutagent.core import AbstractAgent, AbstractTool
        assert AbstractAgent is not None
        assert AbstractTool is not None
