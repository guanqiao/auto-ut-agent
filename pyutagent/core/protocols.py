"""Abstract base classes and protocols for PyUT Agent.

This module defines the core interfaces (protocols) that components must implement.
Using protocols allows for:
- Clear contract definitions
- Easy mocking in tests
- Multiple implementations
- Type checking support
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    TypeVar,
    Union,
    runtime_checkable,
)

T = TypeVar('T')


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations.

    Any LLM client (OpenAI, Anthropic, etc.) should implement this interface.
    """

    @property
    def model(self) -> str:
        """Get the model name."""
        ...

    @property
    def provider(self) -> str:
        """Get the provider name."""
        ...

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            The generated response text
        """
        ...

    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of the generated response
        """
        ...

    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text.

        Args:
            text: The text to count tokens for

        Returns:
            The number of tokens
        """
        ...


@runtime_checkable
class CodeEditorProtocol(Protocol):
    """Protocol for code editor implementations.

    Handles reading, modifying, and writing code files.
    """

    def read_file(self, file_path: Path) -> str:
        """Read a file's contents.

        Args:
            file_path: Path to the file

        Returns:
            The file contents
        """
        ...

    def write_file(self, file_path: Path, content: str) -> None:
        """Write content to a file.

        Args:
            file_path: Path to the file
            content: Content to write
        """
        ...

    def apply_diff(
        self,
        original: str,
        diff: str,
        format: str = "unified"
    ) -> str:
        """Apply a diff to original content.

        Args:
            original: Original content
            diff: Diff to apply
            format: Diff format (unified, search-replace, etc.)

        Returns:
            The modified content
        """
        ...

    def apply_edit(
        self,
        content: str,
        old_text: str,
        new_text: str
    ) -> str:
        """Apply a simple text replacement.

        Args:
            content: Original content
            old_text: Text to find and replace
            new_text: Replacement text

        Returns:
            The modified content
        """
        ...


@runtime_checkable
class TestRunnerProtocol(Protocol):
    """Protocol for test runner implementations.

    Handles running tests and collecting results.
    """

    async def run_tests(
        self,
        test_file: Optional[Path] = None,
        test_class: Optional[str] = None,
        test_method: Optional[str] = None,
        **kwargs
    ) -> "TestResult":
        """Run tests and return results.

        Args:
            test_file: Specific test file to run
            test_class: Specific test class to run
            test_method: Specific test method to run
            **kwargs: Additional runner-specific parameters

        Returns:
            Test execution results
        """
        ...

    async def get_coverage(
        self,
        source_file: Optional[Path] = None
    ) -> "CoverageResult":
        """Get code coverage information.

        Args:
            source_file: Specific source file to get coverage for

        Returns:
            Coverage information
        """
        ...


@runtime_checkable
class CodeParserProtocol(Protocol):
    """Protocol for code parser implementations.

    Handles parsing source code and extracting structural information.
    """

    def parse(self, code: str) -> "ParsedCode":
        """Parse code and extract structure.

        Args:
            code: Source code to parse

        Returns:
            Parsed code structure
        """
        ...

    def extract_class_info(self, code: str) -> "ClassInfo":
        """Extract class information from code.

        Args:
            code: Source code containing a class

        Returns:
            Extracted class information
        """
        ...

    def extract_methods(self, code: str) -> List["MethodInfo"]:
        """Extract method information from code.

        Args:
            code: Source code containing methods

        Returns:
            List of extracted method information
        """
        ...


@runtime_checkable
class MemoryProtocol(Protocol):
    """Protocol for memory implementations.

    Handles storing and retrieving information across agent steps.
    """

    def store(self, key: str, value: Any) -> None:
        """Store a value.

        Args:
            key: Storage key
            value: Value to store
        """
        ...

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a stored value.

        Args:
            key: Storage key

        Returns:
            The stored value or None
        """
        ...

    def clear(self) -> None:
        """Clear all stored values."""
        ...

    def get_context(self) -> Dict[str, Any]:
        """Get all stored context.

        Returns:
            Dictionary of all stored values
        """
        ...


@runtime_checkable
class RecoveryHandlerProtocol(Protocol):
    """Protocol for recovery handler implementations.

    Handles error recovery strategies.
    """

    async def analyze_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> "ErrorAnalysis":
        """Analyze an error.

        Args:
            error: The error to analyze
            context: Additional context

        Returns:
            Error analysis result
        """
        ...

    async def recover(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> "RecoveryAction":
        """Attempt to recover from an error.

        Args:
            error: The error to recover from
            context: Additional context

        Returns:
            Recovery action to take
        """
        ...


class AgentState(Enum):
    """Agent execution states."""
    IDLE = auto()
    INITIALIZING = auto()
    PARSING = auto()
    GENERATING = auto()
    COMPILING = auto()
    TESTING = auto()
    ANALYZING = auto()
    FIXING = auto()
    OPTIMIZING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PAUSED = auto()


@dataclass
class AgentResult(Generic[T]):
    """Result from agent execution."""
    success: bool
    state: AgentState
    data: Optional[T] = None
    message: str = ""
    iterations: int = 0
    coverage: float = 0.0
    test_file: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class TestResult:
    """Result from test execution."""
    passed: int
    failed: int
    skipped: int
    total: int
    failures: List[Dict[str, Any]]
    duration: float
    success: bool

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        return self.passed / self.total if self.total > 0 else 0.0


@dataclass
class CoverageResult:
    """Result from coverage analysis."""
    line_coverage: float
    branch_coverage: float
    method_coverage: float
    class_coverage: float
    covered_lines: int
    total_lines: int
    covered_branches: int
    total_branches: int
    details: Dict[str, Any]

    @property
    def overall_coverage(self) -> float:
        """Get overall coverage percentage."""
        return self.line_coverage


@dataclass
class ParsedCode:
    """Parsed code structure."""
    language: str
    classes: List["ClassInfo"]
    imports: List[str]
    package: Optional[str]
    raw_code: str


@dataclass
class ClassInfo:
    """Information about a parsed class."""
    name: str
    package: Optional[str]
    methods: List["MethodInfo"]
    fields: List["FieldInfo"]
    annotations: List[str]
    modifiers: List[str]
    extends: Optional[str]
    implements: List[str]


@dataclass
class MethodInfo:
    """Information about a parsed method."""
    name: str
    return_type: str
    parameters: List["ParameterInfo"]
    annotations: List[str]
    modifiers: List[str]
    body: Optional[str]
    throws: List[str]


@dataclass
class FieldInfo:
    """Information about a parsed field."""
    name: str
    type: str
    annotations: List[str]
    modifiers: List[str]
    initializer: Optional[str]


@dataclass
class ParameterInfo:
    """Information about a method parameter."""
    name: str
    type: str
    annotations: List[str]


@dataclass
class ErrorAnalysis:
    """Analysis of an error."""
    error_type: str
    error_category: str
    is_retryable: bool
    suggested_fixes: List[str]
    root_cause: str
    severity: str


@dataclass
class RecoveryAction:
    """Action to take for error recovery."""
    action: str
    parameters: Dict[str, Any]
    should_retry: bool
    max_retries: int


class BaseAgent(ABC, Generic[T]):
    """Abstract base class for agents.

    Provides common agent functionality and lifecycle management.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        memory: MemoryProtocol,
        **kwargs
    ):
        self.llm_client = llm_client
        self.memory = memory
        self.state = AgentState.IDLE
        self._stop_requested = False

    @abstractmethod
    async def execute(self, *args, **kwargs) -> AgentResult[T]:
        """Execute the agent's main task.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Agent execution result
        """
        ...

    def stop(self) -> None:
        """Request the agent to stop."""
        self._stop_requested = True

    def reset(self) -> None:
        """Reset the agent state."""
        self._stop_requested = False
        self.state = AgentState.IDLE
        self.memory.clear()

    @property
    def is_stopped(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested


class BaseTool(ABC):
    """Abstract base class for tools.

    Tools are actions that the agent can execute.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the tool name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the tool description."""
        ...

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the tool.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Tool execution result
        """
        ...

    def to_schema(self) -> Dict[str, Any]:
        """Convert tool to JSON schema for LLM function calling.

        Returns:
            JSON schema representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
