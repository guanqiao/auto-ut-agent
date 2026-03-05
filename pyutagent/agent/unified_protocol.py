"""Extended Agent Protocols for unified agent interface.

This module provides:
- AgentCapability: Agent capability definition
- AgentProtocol: Unified agent interface
- UnifiedAgentMixin: Common agent functionality
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..core.protocols import AgentState


class AgentCapabilityType(Enum):
    """Types of agent capabilities."""
    CODE_GENERATION = auto()
    CODE_UNDERSTANDING = auto()
    TEST_GENERATION = auto()
    DEBUGGING = auto()
    REFACTORING = auto()
    CODE_REVIEW = auto()
    DOCUMENTATION = auto()
    ANALYSIS = auto()
    EXECUTION = auto()
    PLANNING = auto()
    REASONING = auto()


@dataclass
class AgentCapability:
    """Agent capability definition."""
    name: str
    description: str
    capability_type: AgentCapabilityType
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)


@dataclass
class ThoughtResult:
    """Result from agent thinking."""
    reasoning: str
    decision: str
    confidence: float
    plan: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolExecutionResult:
    """Result from tool execution."""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0


@runtime_checkable
class AgentProtocol(Protocol):
    """Unified agent protocol.

    All agents should implement this interface.
    """

    @property
    def capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        ...

    @property
    def state(self) -> AgentState:
        """Get agent state."""
        ...

    async def execute(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute a task.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Execution result
        """
        ...

    async def plan(self, goal: str) -> List[Dict[str, Any]]:
        """Create a plan to achieve goal.

        Args:
            goal: Goal description

        Returns:
            Plan steps
        """
        ...

    async def think(self, context: Dict[str, Any]) -> ThoughtResult:
        """Think about the current situation.

        Args:
            context: Current context

        Returns:
            Thought result
        """
        ...


class UnifiedAgentMixin:
    """Mixin providing common agent functionality.

    Provides:
    - State management
    - Tool execution
    - Error handling
    - Memory integration
    """

    def __init__(self):
        self._state = AgentState.IDLE
        self._execution_history: List[Dict[str, Any]] = []
        self._error_count = 0

    @property
    def state(self) -> AgentState:
        """Get current state."""
        return self._state

    def set_state(self, state: AgentState):
        """Set agent state."""
        self._state = state

    def record_execution(self, execution: Dict[str, Any]):
        """Record execution in history."""
        self._execution_history.append(execution)

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return self._execution_history[-limit:]

    def increment_error(self):
        """Increment error count."""
        self._error_count += 1

    def reset_error_count(self):
        """Reset error count."""
        self._error_count = 0

    @property
    def error_count(self) -> int:
        """Get error count."""
        return self._error_count


class AgentBuilder:
    """Builder for creating configured agents."""

    def __init__(self):
        self._llm_client: Optional[Any] = None
        self._memory: Optional[Any] = None
        self._tools: List[Any] = []
        self._capabilities: List[AgentCapability] = []

    def set_llm(self, llm_client: Any) -> "AgentBuilder":
        """Set LLM client."""
        self._llm_client = llm_client
        return self

    def set_memory(self, memory: Any) -> "AgentBuilder":
        """Set memory."""
        self._memory = memory
        return self

    def add_tool(self, tool: Any) -> "AgentBuilder":
        """Add a tool."""
        self._tools.append(tool)
        return self

    def add_capability(self, capability: AgentCapability) -> "AgentBuilder":
        """Add a capability."""
        self._capabilities.append(capability)
        return self

    def build(self) -> AgentProtocol:
        """Build the agent."""
        if not self._llm_client or not self._memory:
            raise ValueError("LLM client and memory are required")

        class ConfiguredAgent:
            def __init__(self):
                self._llm = self._llm_client
                self._memory = self._memory
                self._tools_list = self._tools
                self._caps = self._capabilities

            @property
            def capabilities(self) -> List[AgentCapability]:
                return self._caps

            @property
            def state(self) -> AgentState:
                return AgentState.IDLE

            async def execute(self, task: str, context: Dict[str, Any]) -> Any:
                return {"task": task, "status": "executed"}

            async def plan(self, goal: str) -> List[Dict[str, Any]]:
                return [{"step": "plan", "goal": goal}]

            async def think(self, context: Dict[str, Any]) -> ThoughtResult:
                return ThoughtResult(
                    reasoning="Thinking",
                    decision="Decision",
                    confidence=0.5,
                    plan=[]
                )

        return ConfiguredAgent()


def create_agent_capability(
    name: str,
    description: str,
    capability_type: AgentCapabilityType,
    examples: Optional[List[str]] = None
) -> AgentCapability:
    """Create an agent capability.

    Args:
        name: Capability name
        description: Capability description
        capability_type: Type of capability
        examples: Usage examples

    Returns:
        AgentCapability
    """
    return AgentCapability(
        name=name,
        description=description,
        capability_type=capability_type,
        examples=examples or []
    )
