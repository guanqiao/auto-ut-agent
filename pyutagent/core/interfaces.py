"""Unified Interface Contracts - 统一接口契约

This module defines all public interface contracts for the PyUT Agent framework.
All interfaces use Protocol for duck typing support and runtime checkability.

Usage:
    >>> from pyutagent.core.interfaces import IAgent, ITool, IContext
    >>> 
    >>> class MyAgent(IAgent):
    ...     async def execute(self, task: str) -> AgentResult:
    ...         pass
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    Union,
    runtime_checkable,
)

T = TypeVar('T')
R = TypeVar('R')


# =============================================================================
# Enums
# =============================================================================

class AgentState(Enum):
    """Unified agent states."""
    IDLE = auto()
    INITIALIZING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    COMPLETED = auto()
    FAILED = auto()


class ExecutionStatus(Enum):
    """Execution status for tasks and operations."""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILURE = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class CapabilityType(Enum):
    """Types of capabilities an agent or tool can have."""
    CODE_GENERATION = "code_generation"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    PLANNING = "planning"
    EXECUTION = "execution"
    COORDINATION = "coordination"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExecutionResult:
    """Standard execution result."""
    success: bool
    data: Any = None
    message: str = ""
    error: Optional[Exception] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Capability:
    """Capability definition."""
    name: str
    type: CapabilityType
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Task definition."""
    id: str
    name: str
    description: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: List[str] = field(default_factory=list)
    priority: int = 0
    timeout_seconds: Optional[float] = None


# =============================================================================
# Core Protocols
# =============================================================================

@runtime_checkable
class IExecutable(Protocol):
    """Protocol for anything that can be executed."""
    
    async def execute(self, *args, **kwargs) -> ExecutionResult:
        """Execute the operation.
        
        Returns:
            ExecutionResult with success status and data
        """
        ...


@runtime_checkable
class IInitializable(Protocol):
    """Protocol for components that need initialization."""
    
    async def initialize(self) -> bool:
        """Initialize the component.
        
        Returns:
            True if initialization succeeded
        """
        ...
    
    async def shutdown(self) -> None:
        """Shutdown and cleanup the component."""
        ...
    
    @property
    def is_initialized(self) -> bool:
        """Check if component is initialized."""
        ...


@runtime_checkable
class IStateful(Protocol):
    """Protocol for stateful components."""
    
    @property
    def state(self) -> AgentState:
        """Get current state."""
        ...
    
    def can_transition_to(self, new_state: AgentState) -> bool:
        """Check if transition to new_state is valid."""
        ...


@runtime_checkable
class ICapable(Protocol):
    """Protocol for components with capabilities."""
    
    @property
    def capabilities(self) -> List[Capability]:
        """Get list of capabilities."""
        ...
    
    def has_capability(self, capability_type: CapabilityType) -> bool:
        """Check if has specific capability."""
        ...


# =============================================================================
# Agent Protocols
# =============================================================================

@runtime_checkable
class IAgent(IExecutable, IInitializable, IStateful, ICapable, Protocol):
    """Unified agent protocol.
    
    All agents in the system should implement this protocol.
    """
    
    @property
    def name(self) -> str:
        """Get agent name."""
        ...
    
    @property
    def agent_type(self) -> str:
        """Get agent type identifier."""
        ...
    
    async def execute(self, task: Task) -> ExecutionResult:
        """Execute a task.
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        ...
    
    async def plan(self, goal: str) -> List[Task]:
        """Create execution plan for a goal.
        
        Args:
            goal: High-level goal description
            
        Returns:
            List of tasks to achieve the goal
        """
        ...
    
    def pause(self) -> bool:
        """Pause execution.
        
        Returns:
            True if paused successfully
        """
        ...
    
    def resume(self) -> bool:
        """Resume execution.
        
        Returns:
            True if resumed successfully
        """
        ...
    
    def stop(self) -> bool:
        """Stop execution.
        
        Returns:
            True if stopped successfully
        """
        ...


@runtime_checkable
class ISubAgent(IAgent, Protocol):
    """Protocol for sub-agents.
    
    Sub-agents are specialized agents that work under a coordinator.
    """
    
    @property
    def parent_id(self) -> Optional[str]:
        """Get parent agent ID."""
        ...
    
    async def report_status(self) -> Dict[str, Any]:
        """Report current status to parent."""
        ...


# =============================================================================
# Tool Protocols
# =============================================================================

@runtime_checkable
class ITool(IExecutable, Protocol):
    """Unified tool protocol.
    
    All tools in the system should implement this protocol.
    """
    
    @property
    def name(self) -> str:
        """Get tool name."""
        ...
    
    @property
    def description(self) -> str:
        """Get tool description."""
        ...
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters schema."""
        ...
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate tool inputs.
        
        Args:
            inputs: Input parameters
            
        Returns:
            List of validation errors (empty if valid)
        """
        ...
    
    async def execute(self, inputs: Dict[str, Any]) -> ExecutionResult:
        """Execute the tool.
        
        Args:
            inputs: Tool inputs
            
        Returns:
            Execution result
        """
        ...


@runtime_checkable
class IToolRegistry(Protocol):
    """Protocol for tool registries."""
    
    def register(self, tool: ITool) -> bool:
        """Register a tool."""
        ...
    
    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool."""
        ...
    
    def get_tool(self, name: str) -> Optional[ITool]:
        """Get tool by name."""
        ...
    
    def list_tools(self) -> List[ITool]:
        """List all registered tools."""
        ...


# =============================================================================
# Context Protocols
# =============================================================================

@runtime_checkable
class IContext(Protocol):
    """Unified context protocol.
    
    Provides access to shared state and resources.
    """
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context."""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set value in context."""
        ...
    
    def has(self, key: str) -> bool:
        """Check if key exists in context."""
        ...
    
    def delete(self, key: str) -> bool:
        """Delete key from context."""
        ...
    
    def keys(self) -> Set[str]:
        """Get all keys in context."""
        ...
    
    def create_child(self) -> "IContext":
        """Create child context."""
        ...


@runtime_checkable
class IProjectContext(IContext, Protocol):
    """Protocol for project-specific context."""
    
    @property
    def project_path(self) -> Path:
        """Get project root path."""
        ...
    
    @property
    def working_dir(self) -> Path:
        """Get current working directory."""
        ...


# =============================================================================
# Memory Protocols
# =============================================================================

@runtime_checkable
class IMemory(Protocol):
    """Unified memory protocol."""
    
    def store(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in memory.
        
        Args:
            key: Storage key
            value: Value to store
            ttl: Time to live in seconds (None for permanent)
            
        Returns:
            True if stored successfully
        """
        ...
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value from memory."""
        ...
    
    def delete(self, key: str) -> bool:
        """Delete value from memory."""
        ...
    
    def clear(self) -> None:
        """Clear all memory."""
        ...


@runtime_checkable
class IWorkingMemory(IMemory, Protocol):
    """Protocol for working memory (short-term)."""
    
    def get_recent(self, count: int = 10) -> List[tuple]:
        """Get recent entries."""
        ...


@runtime_checkable
class ILongTermMemory(IMemory, Protocol):
    """Protocol for long-term memory."""
    
    def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search memory by query."""
        ...
    
    def get_by_tag(self, tag: str) -> List[Any]:
        """Get entries by tag."""
        ...


# =============================================================================
# LLM Protocols
# =============================================================================

@runtime_checkable
class ILLMClient(Protocol):
    """Unified LLM client protocol."""
    
    @property
    def model(self) -> str:
        """Get model name."""
        ...
    
    @property
    def provider(self) -> str:
        """Get provider name."""
        ...
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        ...
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming text."""
        ...
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...


# =============================================================================
# Event Protocols
# =============================================================================

@runtime_checkable
class IEvent(Protocol):
    """Protocol for events."""
    
    @property
    def event_type(self) -> str:
        """Get event type."""
        ...
    
    @property
    def timestamp(self) -> datetime:
        """Get event timestamp."""
        ...
    
    @property
    def source(self) -> str:
        """Get event source."""
        ...


@runtime_checkable
class IEventBus(Protocol):
    """Protocol for event buses."""
    
    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[IEvent], Any]
    ) -> str:
        """Subscribe to events.
        
        Returns:
            Subscription ID
        """
        ...
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        ...
    
    async def publish(self, event: IEvent) -> int:
        """Publish event.
        
        Returns:
            Number of handlers invoked
        """
        ...


# =============================================================================
# Skill Protocols
# =============================================================================

@runtime_checkable
class ISkill(IExecutable, IInitializable, Protocol):
    """Unified skill protocol."""
    
    @property
    def name(self) -> str:
        """Get skill name."""
        ...
    
    @property
    def description(self) -> str:
        """Get skill description."""
        ...
    
    @property
    def required_tools(self) -> List[str]:
        """Get required tool names."""
        ...
    
    def get_instructions(self) -> str:
        """Get skill instructions."""
        ...
    
    async def execute(
        self,
        task: str,
        context: IContext,
        inputs: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute the skill."""
        ...


# =============================================================================
# Abstract Base Classes (for inheritance)
# =============================================================================

class AbstractAgent(ABC, IAgent):
    """Abstract base class for agents.
    
    Provides common agent functionality.
    """
    
    def __init__(self, name: str, agent_type: str):
        self._name = name
        self._agent_type = agent_type
        self._state = AgentState.IDLE
        self._capabilities: List[Capability] = []
        self._initialized = False
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def agent_type(self) -> str:
        return self._agent_type
    
    @property
    def state(self) -> AgentState:
        return self._state
    
    @property
    def capabilities(self) -> List[Capability]:
        return self._capabilities.copy()
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def has_capability(self, capability_type: CapabilityType) -> bool:
        return any(c.type == capability_type for c in self._capabilities)
    
    def can_transition_to(self, new_state: AgentState) -> bool:
        # Define valid state transitions
        valid_transitions = {
            AgentState.IDLE: [AgentState.INITIALIZING, AgentState.PLANNING],
            AgentState.INITIALIZING: [AgentState.IDLE, AgentState.PLANNING, AgentState.FAILED],
            AgentState.PLANNING: [AgentState.EXECUTING, AgentState.PAUSED, AgentState.FAILED],
            AgentState.EXECUTING: [AgentState.PAUSED, AgentState.STOPPING, AgentState.COMPLETED, AgentState.FAILED],
            AgentState.PAUSED: [AgentState.EXECUTING, AgentState.STOPPING],
            AgentState.STOPPING: [AgentState.STOPPED, AgentState.FAILED],
            AgentState.STOPPED: [AgentState.IDLE],
            AgentState.COMPLETED: [AgentState.IDLE],
            AgentState.FAILED: [AgentState.IDLE],
        }
        return new_state in valid_transitions.get(self._state, [])
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def shutdown(self) -> None:
        self._initialized = False
    
    @abstractmethod
    async def execute(self, task: Task) -> ExecutionResult:
        raise NotImplementedError
    
    @abstractmethod
    async def plan(self, goal: str) -> List[Task]:
        raise NotImplementedError
    
    def pause(self) -> bool:
        if self.can_transition_to(AgentState.PAUSED):
            self._state = AgentState.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        if self.can_transition_to(AgentState.EXECUTING):
            self._state = AgentState.EXECUTING
            return True
        return False
    
    def stop(self) -> bool:
        if self.can_transition_to(AgentState.STOPPING):
            self._state = AgentState.STOPPING
            return True
        return False


class AbstractTool(ABC, ITool):
    """Abstract base class for tools."""
    
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description
        self._parameters: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        errors = []
        for param_name, param_spec in self._parameters.items():
            if param_spec.get("required", False) and param_name not in inputs:
                errors.append(f"Missing required parameter: {param_name}")
        return errors
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> ExecutionResult:
        raise NotImplementedError


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "AgentState",
    "ExecutionStatus",
    "CapabilityType",
    # Data Classes
    "ExecutionResult",
    "Capability",
    "Task",
    # Core Protocols
    "IExecutable",
    "IInitializable",
    "IStateful",
    "ICapable",
    # Agent Protocols
    "IAgent",
    "ISubAgent",
    # Tool Protocols
    "ITool",
    "IToolRegistry",
    # Context Protocols
    "IContext",
    "IProjectContext",
    # Memory Protocols
    "IMemory",
    "IWorkingMemory",
    "ILongTermMemory",
    # LLM Protocols
    "ILLMClient",
    # Event Protocols
    "IEvent",
    "IEventBus",
    # Skill Protocols
    "ISkill",
    # Abstract Base Classes
    "AbstractAgent",
    "AbstractTool",
]
