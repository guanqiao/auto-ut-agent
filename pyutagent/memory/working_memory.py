"""Working memory for current task context."""

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


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
    max_iterations: int = 3
    target_coverage: float = 0.8
    
    # Coverage tracking
    current_coverage: float = 0.0
    coverage_source: str = "jacoco"
    coverage_confidence: float = 1.0
    coverage_history: List[float] = field(default_factory=list)
    
    # Control state
    is_paused: bool = False
    skip_verification: bool = False  # Skip compilation/verification during batch generation
    
    # Processing tracking
    processed_files: List[str] = field(default_factory=list)
    failed_tests: List[Dict[str, Any]] = field(default_factory=list)
    generated_tests: List[Dict[str, Any]] = field(default_factory=list)
    
    # LLM context
    llm_context: Dict[str, Any] = field(default_factory=dict)
    
    def update_coverage(self, coverage: float, source: str = "jacoco", confidence: float = 1.0):
        """Update current coverage and add to history.
        
        Args:
            coverage: Coverage percentage
            source: Coverage source ("jacoco" or "llm_estimated")
            confidence: Confidence level for LLM estimation
        """
        self.current_coverage = coverage
        self.coverage_source = source
        self.coverage_confidence = confidence
        self.coverage_history.append({
            "coverage": coverage,
            "source": source,
            "confidence": confidence,
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

    def has_coverage_stalled(self, window_size: int = 3, threshold: float = 0.01) -> bool:
        """Check if coverage has stalled (no significant improvement).

        Args:
            window_size: Number of recent iterations to check
            threshold: Minimum coverage improvement threshold

        Returns:
            True if coverage has stalled, False otherwise
        """
        if len(self.coverage_history) < window_size:
            return False

        recent = self.coverage_history[-window_size:]
        coverages = [h["coverage"] for h in recent]

        max_coverage = max(coverages)
        min_coverage = min(coverages)

        stalled = (max_coverage - min_coverage) < threshold

        if stalled:
            logger.info(f"[WorkingMemory] Coverage stalled: {min_coverage:.1%} -> {max_coverage:.1%} "
                       f"(window={window_size}, threshold={threshold:.1%})")

        return stalled

    def get_coverage_trend(self, window_size: int = 3) -> Dict[str, Any]:
        """Get coverage trend information.

        Args:
            window_size: Number of recent iterations to analyze

        Returns:
            Dict with trend information
        """
        if len(self.coverage_history) < 2:
            return {"trend": "insufficient_data", "improvement": 0.0}

        recent = self.coverage_history[-window_size:] if len(self.coverage_history) >= window_size else self.coverage_history
        coverages = [h["coverage"] for h in recent]

        if len(coverages) < 2:
            return {"trend": "insufficient_data", "improvement": 0.0}

        first = coverages[0]
        last = coverages[-1]
        improvement = last - first

        if improvement > 0.05:
            trend = "improving"
        elif improvement > 0.01:
            trend = "slight_improvement"
        elif improvement > -0.01:
            trend = "stable"
        else:
            trend = "declining"

        return {
            "trend": trend,
            "improvement": improvement,
            "start_coverage": first,
            "current_coverage": last,
            "window_size": len(coverages)
        }
    
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
