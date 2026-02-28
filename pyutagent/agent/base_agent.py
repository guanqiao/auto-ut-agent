"""Base agent class for UT generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json

from ..memory.working_memory import WorkingMemory
from ..llm.client import LLMClient


class AgentState(Enum):
    """Agent execution states."""
    IDLE = auto()
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
class AgentResult:
    """Result of agent execution."""
    success: bool
    message: str
    test_file: Optional[str] = None
    coverage: float = 0.0
    iterations: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of a single step."""
    success: bool
    state: AgentState
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for UT generation agents."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        working_memory: WorkingMemory,
        project_path: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize base agent.
        
        Args:
            llm_client: LLM client for generation
            working_memory: Working memory for context
            project_path: Path to the project
            progress_callback: Optional callback for progress updates
        """
        self.llm_client = llm_client
        self.working_memory = working_memory
        self.project_path = Path(project_path)
        self.progress_callback = progress_callback
        
        self.state = AgentState.IDLE
        self.state_history: List[Dict[str, Any]] = []
        self.current_iteration = 0
        self.max_iterations = working_memory.max_iterations
        self.target_coverage = working_memory.target_coverage
        
    def _update_state(self, new_state: AgentState, message: str = ""):
        """Update agent state and record history."""
        self.state = new_state
        self.state_history.append({
            "state": new_state.name,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "iteration": self.current_iteration
        })
        
        if self.progress_callback:
            self.progress_callback({
                "state": new_state.name,
                "message": message,
                "progress": self.working_memory.get_progress()
            })
    
    def _should_continue(self) -> bool:
        """Check if the agent should continue execution."""
        if self.state == AgentState.PAUSED:
            return False
        if self.current_iteration >= self.max_iterations:
            return False
        if self.working_memory.current_coverage >= self.target_coverage:
            return False
        return True
    
    def pause(self):
        """Pause agent execution."""
        self._update_state(AgentState.PAUSED, "Execution paused by user")
        self.working_memory.pause()
    
    def resume(self):
        """Resume agent execution."""
        self.working_memory.resume()
        self._update_state(AgentState.IDLE, "Execution resumed")
    
    @abstractmethod
    async def generate_tests(self, target_file: str) -> AgentResult:
        """Generate tests for a target file.
        
        Args:
            target_file: Path to the target Java file
            
        Returns:
            AgentResult with generation results
        """
        pass
    
    @abstractmethod
    async def run_feedback_loop(self, target_file: str) -> AgentResult:
        """Run the complete feedback loop.
        
        Args:
            target_file: Path to the target Java file
            
        Returns:
            AgentResult with final results
        """
        pass
    
    def save_state(self, path: Optional[str] = None) -> str:
        """Save agent state to file.
        
        Args:
            path: Optional path to save state
            
        Returns:
            Path to saved state file
        """
        if path is None:
            path = self.project_path / ".pyutagent" / "agent_state.json"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state_data = {
            "state": self.state.name,
            "state_history": self.state_history,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "target_coverage": self.target_coverage,
            "working_memory": self.working_memory.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2)
        
        return str(path)
    
    def load_state(self, path: str) -> bool:
        """Load agent state from file.
        
        Args:
            path: Path to state file
            
        Returns:
            True if successful
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            self.state = AgentState[state_data.get("state", "IDLE")]
            self.state_history = state_data.get("state_history", [])
            self.current_iteration = state_data.get("current_iteration", 0)
            self.max_iterations = state_data.get("max_iterations", 10)
            self.target_coverage = state_data.get("target_coverage", 0.8)
            self.working_memory = WorkingMemory.from_dict(
                state_data.get("working_memory", {})
            )
            
            return True
        except Exception as e:
            print(f"Error loading agent state: {e}")
            return False