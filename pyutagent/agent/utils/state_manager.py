"""Manager for agent state and pause/resume functionality."""

import asyncio
import logging
from enum import Enum, auto
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status enumeration."""
    IDLE = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


class StateManager:
    """Manages agent state and pause/resume functionality."""
    
    def __init__(self):
        """Initialize state manager."""
        self.status = TaskStatus.IDLE
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused by default
        self._current_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.on_progress: Optional[Callable[[int, str], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None
    
    def _log(self, message: str):
        """Log message."""
        logger.info(message)
        if self.on_log:
            self.on_log(message)
    
    def _update_progress(self, value: int, status: str):
        """Update progress."""
        if self.on_progress:
            self.on_progress(value, status)
    
    async def check_pause(self):
        """Check if paused and wait if necessary."""
        if not self._pause_event.is_set():
            self.status = TaskStatus.PAUSED
            self._log("Task paused, waiting to resume...")
            await self._pause_event.wait()
            self.status = TaskStatus.RUNNING
            self._log("Task resumed")
    
    def pause(self):
        """Pause the current task."""
        if self.status == TaskStatus.RUNNING:
            self._pause_event.clear()
            self.status = TaskStatus.PAUSED
            self._log("Pause request sent")
    
    def resume(self):
        """Resume the paused task."""
        if self.status == TaskStatus.PAUSED:
            self._pause_event.set()
            self.status = TaskStatus.RUNNING
            self._log("Resume request sent")
    
    def stop(self):
        """Stop the current task."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self.status = TaskStatus.IDLE
            self._log("Task stopped")
    
    def is_paused(self) -> bool:
        """Check if task is paused."""
        return self.status == TaskStatus.PAUSED
    
    def is_running(self) -> bool:
        """Check if task is running."""
        return self.status == TaskStatus.RUNNING
    
    def get_status(self) -> TaskStatus:
        """Get current status."""
        return self.status
    
    def set_status(self, status: TaskStatus):
        """Set current status."""
        self.status = status
    
    def start_task(self):
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self._pause_event.set()
    
    def complete_task(self):
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
    
    def fail_task(self):
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary.
        
        Returns:
            State dictionary
        """
        return {
            "status": self.status.name,
        }
    
    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "StateManager":
        """Restore state from dictionary.
        
        Args:
            state: State dictionary
            
        Returns:
            Restored state manager
        """
        manager = cls()
        status_name = state.get("status", "IDLE")
        try:
            manager.status = TaskStatus[status_name]
        except KeyError:
            manager.status = TaskStatus.IDLE
        return manager
