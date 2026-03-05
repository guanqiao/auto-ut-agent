"""Execution Plan for Multi-Step Tasks.

This module provides:
- Step: Individual execution step
- ExecutionPlan: Plan for multi-step execution
- StepStatus: Status tracking for steps
- SubTask: Subtask for task decomposition
- SubTaskType: Types of subtasks
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable
import logging

from ..task_planner import SubTask, SubTaskType, SubTaskStatus

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of an execution step."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    RETRYING = auto()


class StepType(Enum):
    """Type of execution step."""
    PARSE = "parse"
    GENERATE = "generate"
    COMPILE = "compile"
    TEST = "test"
    ANALYZE = "analyze"
    FIX = "fix"
    OPTIMIZE = "optimize"
    CUSTOM = "custom"
    ACTION = "action"
    PLAN = "plan"
    VERIFY = "verify"


@dataclass
class Step:
    """A single execution step in a plan."""
    
    id: str
    name: str
    step_type: StepType
    description: str = ""
    status: StepStatus = StepStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    
    handler: Optional[Callable] = None
    params: Dict[str, Any] = field(default_factory=dict)
    
    result: Optional[Any] = None
    error: Optional[str] = None
    
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    retry_count: int = 0
    max_retries: int = 3
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> None:
        """Mark step as started."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.now()
        logger.info(f"Step '{self.name}' started")
    
    def complete(self, result: Any = None) -> None:
        """Mark step as completed."""
        self.status = StepStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()
        logger.info(f"Step '{self.name}' completed")
    
    def fail(self, error: str) -> None:
        """Mark step as failed."""
        self.status = StepStatus.FAILED
        self.error = error
        self.completed_at = datetime.now()
        logger.error(f"Step '{self.name}' failed: {error}")
    
    def skip(self, reason: str = "") -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED
        self.error = reason
        self.completed_at = datetime.now()
        logger.info(f"Step '{self.name}' skipped: {reason}")
    
    def retry(self) -> bool:
        """Attempt to retry the step.
        
        Returns:
            True if retry is possible, False if max retries reached
        """
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            self.status = StepStatus.RETRYING
            self.error = None
            self.completed_at = None
            logger.info(f"Step '{self.name}' retrying ({self.retry_count}/{self.max_retries})")
            return True
        return False
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Get step duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None
    
    @property
    def is_terminal(self) -> bool:
        """Check if step is in a terminal state."""
        return self.status in (
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.SKIPPED,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "step_type": self.step_type.value,
            "description": self.description,
            "status": self.status.name,
            "dependencies": self.dependencies,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
        }


@dataclass
class ExecutionPlan:
    """A plan for multi-step execution.
    
    Features:
    - Dependency management
    - Parallel execution support
    - Progress tracking
    - Error recovery
    """
    
    id: str
    name: str
    description: str = ""
    steps: List[Step] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: Step) -> None:
        """Add a step to the plan."""
        self.steps.append(step)
    
    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_steps_by_status(self, status: StepStatus) -> List[Step]:
        """Get all steps with a specific status."""
        return [s for s in self.steps if s.status == status]
    
    def get_ready_steps(self) -> List[Step]:
        """Get steps that are ready to execute.
        
        A step is ready if:
        - It's in PENDING status
        - All its dependencies are completed
        """
        ready = []
        completed_ids = {s.id for s in self.get_steps_by_status(StepStatus.COMPLETED)}
        
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            
            if all(dep_id in completed_ids for dep_id in step.dependencies):
                ready.append(step)
        
        return ready
    
    def get_next_step(self) -> Optional[Step]:
        """Get the next step to execute."""
        ready = self.get_ready_steps()
        return ready[0] if ready else None
    
    @property
    def progress(self) -> float:
        """Get execution progress as a percentage."""
        if not self.steps:
            return 0.0
        
        completed = len(self.get_steps_by_status(StepStatus.COMPLETED))
        skipped = len(self.get_steps_by_status(StepStatus.SKIPPED))
        total = len(self.steps)
        
        return (completed + skipped) / total
    
    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(s.is_terminal for s in self.steps)
    
    @property
    def has_failures(self) -> bool:
        """Check if any steps have failed."""
        return any(s.status == StepStatus.FAILED for s in self.steps)
    
    @property
    def current_step(self) -> Optional[Step]:
        """Get the currently running step."""
        running = self.get_steps_by_status(StepStatus.RUNNING)
        return running[0] if running else None
    
    def start(self) -> None:
        """Mark plan as started."""
        self.started_at = datetime.now()
        logger.info(f"Execution plan '{self.name}' started")
    
    def complete(self) -> None:
        """Mark plan as completed."""
        self.completed_at = datetime.now()
        logger.info(f"Execution plan '{self.name}' completed")
    
    @property
    def duration_ms(self) -> Optional[int]:
        """Get plan duration in milliseconds."""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return int(delta.total_seconds() * 1000)
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "progress": self.progress,
            "is_complete": self.is_complete,
            "has_failures": self.has_failures,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
        }
    
    @classmethod
    def create_test_generation_plan(
        cls,
        target_file: str,
        max_iterations: int = 10,
    ) -> "ExecutionPlan":
        """Create a standard test generation plan.
        
        Args:
            target_file: Target file to generate tests for
            max_iterations: Maximum number of iterations
            
        Returns:
            ExecutionPlan for test generation
        """
        plan = cls(
            id=f"test_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"Test Generation for {target_file}",
            description=f"Generate unit tests for {target_file}",
        )
        
        plan.add_step(Step(
            id="parse",
            name="Parse Target",
            step_type=StepType.PARSE,
            description=f"Parse {target_file} to extract class info",
            params={"target_file": target_file},
        ))
        
        plan.add_step(Step(
            id="generate",
            name="Generate Tests",
            step_type=StepType.GENERATE,
            description="Generate initial test cases",
            dependencies=["parse"],
            params={"target_file": target_file},
        ))
        
        plan.add_step(Step(
            id="compile",
            name="Compile Tests",
            step_type=StepType.COMPILE,
            description="Compile generated tests",
            dependencies=["generate"],
        ))
        
        plan.add_step(Step(
            id="test",
            name="Run Tests",
            step_type=StepType.TEST,
            description="Execute tests and collect results",
            dependencies=["compile"],
        ))
        
        plan.add_step(Step(
            id="analyze",
            name="Analyze Coverage",
            step_type=StepType.ANALYZE,
            description="Analyze test coverage",
            dependencies=["test"],
        ))
        
        plan.add_step(Step(
            id="optimize",
            name="Optimize Coverage",
            step_type=StepType.OPTIMIZE,
            description="Generate additional tests for uncovered code",
            dependencies=["analyze"],
        ))
        
        return plan
