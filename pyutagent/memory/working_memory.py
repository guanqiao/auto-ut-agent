"""Working memory for current task context."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class WorkingMemory:
    """Working memory holds the current task context.
    
    This is ephemeral memory that exists only during the current
    task execution. It's not persisted across sessions.
    """
    
    # Current task state
    current_file: Optional[str] = None
    current_method: Optional[str] = None
    
    # Iteration tracking
    iteration_count: int = 0
    max_iterations: int = 10
    target_coverage: float = 0.8
    
    # Coverage tracking
    current_coverage: float = 0.0
    coverage_history: List[float] = field(default_factory=list)
    
    # Control state
    is_paused: bool = False
    
    # Processing tracking
    processed_files: List[str] = field(default_factory=list)
    failed_tests: List[Dict[str, Any]] = field(default_factory=list)
    generated_tests: List[Dict[str, Any]] = field(default_factory=list)
    
    # LLM context
    llm_context: Dict[str, Any] = field(default_factory=dict)
    
    def update_coverage(self, coverage: float):
        """Update current coverage and add to history."""
        self.current_coverage = coverage
        self.coverage_history.append({
            "coverage": coverage,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_processed_file(self, file_path: str):
        """Add a file to the processed list."""
        if file_path not in self.processed_files:
            self.processed_files.append(file_path)
    
    def add_failed_test(self, test_name: str, error: str, file: Optional[str] = None):
        """Record a failed test."""
        self.failed_tests.append({
            "test_name": test_name,
            "error": error,
            "file": file,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_generated_test(self, file: str, method: str, code: str):
        """Record a generated test."""
        self.generated_tests.append({
            "file": file,
            "method": method,
            "code": code,
            "timestamp": datetime.now().isoformat()
        })
    
    def pause(self):
        """Pause the current task."""
        self.is_paused = True
    
    def resume(self):
        """Resume the current task."""
        self.is_paused = False
    
    def increment_iteration(self):
        """Increment the iteration counter."""
        self.iteration_count += 1
    
    def is_complete(self) -> bool:
        """Check if the task is complete."""
        return (
            self.current_coverage >= self.target_coverage or
            self.iteration_count >= self.max_iterations
        )
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        return {
            "current_file": self.current_file,
            "current_method": self.current_method,
            "iteration": f"{self.iteration_count}/{self.max_iterations}",
            "coverage": f"{self.current_coverage:.1%}",
            "target": f"{self.target_coverage:.1%}",
            "is_paused": self.is_paused,
            "is_complete": self.is_complete()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        """Deserialize from dictionary."""
        return cls(**data)
