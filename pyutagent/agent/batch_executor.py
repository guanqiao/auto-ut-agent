"""Batch execution module for handling large numbers of subtasks.

.. deprecated::
    Use pyutagent.agent.execution.executor.StepExecutor instead.
    This module is kept for backward compatibility.

This module provides:
- BatchExecutor: Execute subtasks in batches with progress tracking (deprecated)
- ProgressPersistence: Save and restore execution progress (deprecated)
- ExecutionCheckpoint: Checkpoint management for long-running tasks (deprecated)
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

# Emit deprecation warning when module is imported
warnings.warn(
    "pyutagent.agent.batch_executor is deprecated. "
    "Use pyutagent.agent.execution.executor.StepExecutor instead.",
    DeprecationWarning,
    stacklevel=2
)


class BatchStatus(Enum):
    """Status of batch execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SubtaskProgress:
    """Progress tracking for a single subtask."""
    subtask_id: str
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtaskProgress":
        return cls(
            subtask_id=data["subtask_id"],
            status=data.get("status", "pending"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
        )


@dataclass
class ExecutionCheckpoint:
    """Checkpoint for batch execution."""
    checkpoint_id: str
    task_id: str
    total_subtasks: int
    completed_subtasks: int
    failed_subtasks: int
    current_batch: int
    status: BatchStatus
    subtask_progress: Dict[str, SubtaskProgress] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.completed_subtasks,
            "failed_subtasks": self.failed_subtasks,
            "current_batch": self.current_batch,
            "status": self.status.value,
            "subtask_progress": {k: v.to_dict() for k, v in self.subtask_progress.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionCheckpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            task_id=data["task_id"],
            total_subtasks=data["total_subtasks"],
            completed_subtasks=data["completed_subtasks"],
            failed_subtasks=data["failed_subtasks"],
            current_batch=data["current_batch"],
            status=BatchStatus(data["status"]),
            subtask_progress={
                k: SubtaskProgress.from_dict(v) 
                for k, v in data.get("subtask_progress", {}).items()
            },
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def progress_percentage(self) -> float:
        if self.total_subtasks == 0:
            return 0.0
        return (self.completed_subtasks / self.total_subtasks) * 100


class ProgressPersistence:
    """Manages persistence of execution progress."""
    
    def __init__(self, storage_dir: str = ".pyutagent/checkpoints"):
        """Initialize progress persistence.
        
        Args:
            storage_dir: Directory for storing checkpoints
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        """Save checkpoint to disk."""
        checkpoint_file = self.storage_dir / f"{checkpoint.checkpoint_id}.json"
        checkpoint.updated_at = datetime.now().isoformat()
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        
        logger.debug(f"[ProgressPersistence] Saved checkpoint: {checkpoint.checkpoint_id}")
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ExecutionCheckpoint]:
        """Load checkpoint from disk."""
        checkpoint_file = self.storage_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ExecutionCheckpoint.from_dict(data)
        except Exception as e:
            logger.warning(f"[ProgressPersistence] Failed to load checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> List[str]:
        """List all checkpoint IDs."""
        return [f.stem for f in self.storage_dir.glob("*.json")]
    
    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        checkpoint_file = self.storage_dir / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
    def get_latest_checkpoint(self, task_id: str) -> Optional[ExecutionCheckpoint]:
        """Get the latest checkpoint for a task."""
        checkpoints = []
        
        for checkpoint_id in self.list_checkpoints():
            checkpoint = self.load_checkpoint(checkpoint_id)
            if checkpoint and checkpoint.task_id == task_id:
                checkpoints.append(checkpoint)
        
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda c: c.updated_at)


class BatchExecutor:
    """Executes subtasks in batches with progress tracking."""
    
    def __init__(
        self,
        batch_size: int = 10,
        max_retries: int = 3,
        checkpoint_interval: int = 5,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """Initialize batch executor.
        
        Args:
            batch_size: Number of subtasks per batch
            max_retries: Maximum retries per subtask
            checkpoint_interval: Save checkpoint every N subtasks
            progress_callback: Optional callback for progress updates
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.checkpoint_interval = checkpoint_interval
        self.progress_callback = progress_callback
        
        self._persistence = ProgressPersistence()
        self._current_checkpoint: Optional[ExecutionCheckpoint] = None
        self._cancelled = False
    
    def create_checkpoint(
        self,
        task_id: str,
        subtask_ids: List[str]
    ) -> ExecutionCheckpoint:
        """Create a new execution checkpoint."""
        import uuid
        
        checkpoint = ExecutionCheckpoint(
            checkpoint_id=f"cp_{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            total_subtasks=len(subtask_ids),
            completed_subtasks=0,
            failed_subtasks=0,
            current_batch=0,
            status=BatchStatus.PENDING,
            subtask_progress={
                sid: SubtaskProgress(subtask_id=sid)
                for sid in subtask_ids
            },
        )
        
        self._current_checkpoint = checkpoint
        self._persistence.save_checkpoint(checkpoint)
        
        return checkpoint
    
    def resume_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[ExecutionCheckpoint]:
        """Resume execution from a checkpoint."""
        checkpoint = self._persistence.load_checkpoint(checkpoint_id)
        
        if checkpoint:
            self._current_checkpoint = checkpoint
            logger.info(f"[BatchExecutor] Resumed from checkpoint: {checkpoint_id}")
        
        return checkpoint
    
    async def execute_batch(
        self,
        subtasks: List[Any],
        executor: Callable[[Any], Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Execute subtasks in batches.
        
        Args:
            subtasks: List of subtasks to execute
            executor: Async function to execute each subtask
            task_id: Task identifier
            
        Returns:
            Execution results
        """
        if not self._current_checkpoint:
            self.create_checkpoint(task_id, [s.id for s in subtasks])
        
        self._current_checkpoint.status = BatchStatus.RUNNING
        self._cancelled = False
        
        results = {
            "completed": [],
            "failed": [],
            "skipped": [],
        }
        
        # Filter already completed subtasks
        pending_subtasks = [
            s for s in subtasks
            if self._current_checkpoint.subtask_progress.get(s.id, SubtaskProgress(s.id)).status != "completed"
        ]
        
        logger.info(f"[BatchExecutor] Starting batch execution: {len(pending_subtasks)}/{len(subtasks)} pending")
        
        # Execute in batches
        for batch_idx in range(0, len(pending_subtasks), self.batch_size):
            if self._cancelled:
                self._current_checkpoint.status = BatchStatus.CANCELLED
                break
            
            batch = pending_subtasks[batch_idx:batch_idx + self.batch_size]
            self._current_checkpoint.current_batch = batch_idx // self.batch_size
            
            logger.info(f"[BatchExecutor] Processing batch {self._current_checkpoint.current_batch + 1}")
            
            for subtask in batch:
                if self._cancelled:
                    break
                
                progress = self._current_checkpoint.subtask_progress.get(
                    subtask.id, SubtaskProgress(subtask.id)
                )
                
                progress.status = "running"
                progress.started_at = datetime.now().isoformat()
                self._current_checkpoint.subtask_progress[subtask.id] = progress
                
                self._notify_progress()
                
                # Execute with retries
                for attempt in range(self.max_retries):
                    try:
                        result = await executor(subtask)
                        
                        progress.status = "completed"
                        progress.completed_at = datetime.now().isoformat()
                        progress.result = result
                        self._current_checkpoint.completed_subtasks += 1
                        results["completed"].append({
                            "subtask_id": subtask.id,
                            "result": result,
                        })
                        break
                        
                    except Exception as e:
                        progress.retry_count = attempt + 1
                        progress.error = str(e)
                        
                        if attempt == self.max_retries - 1:
                            progress.status = "failed"
                            self._current_checkpoint.failed_subtasks += 1
                            results["failed"].append({
                                "subtask_id": subtask.id,
                                "error": str(e),
                            })
                            logger.warning(f"[BatchExecutor] Subtask {subtask.id} failed: {e}")
                
                # Save checkpoint periodically
                if self._current_checkpoint.completed_subtasks % self.checkpoint_interval == 0:
                    self._persistence.save_checkpoint(self._current_checkpoint)
        
        # Final checkpoint
        if not self._cancelled:
            self._current_checkpoint.status = BatchStatus.COMPLETED
        
        self._persistence.save_checkpoint(self._current_checkpoint)
        
        return {
            "checkpoint_id": self._current_checkpoint.checkpoint_id,
            "total": len(subtasks),
            "completed": len(results["completed"]),
            "failed": len(results["failed"]),
            "skipped": len(results["skipped"]),
            "results": results,
        }
    
    def pause(self):
        """Pause execution."""
        if self._current_checkpoint:
            self._current_checkpoint.status = BatchStatus.PAUSED
            self._persistence.save_checkpoint(self._current_checkpoint)
            logger.info("[BatchExecutor] Execution paused")
    
    def resume(self):
        """Resume paused execution."""
        if self._current_checkpoint and self._current_checkpoint.status == BatchStatus.PAUSED:
            self._current_checkpoint.status = BatchStatus.RUNNING
            self._persistence.save_checkpoint(self._current_checkpoint)
            logger.info("[BatchExecutor] Execution resumed")
    
    def cancel(self):
        """Cancel execution."""
        self._cancelled = True
        if self._current_checkpoint:
            self._current_checkpoint.status = BatchStatus.CANCELLED
            self._persistence.save_checkpoint(self._current_checkpoint)
            logger.info("[BatchExecutor] Execution cancelled")
    
    def _notify_progress(self):
        """Notify progress callback."""
        if self.progress_callback and self._current_checkpoint:
            self.progress_callback({
                "checkpoint_id": self._current_checkpoint.checkpoint_id,
                "total": self._current_checkpoint.total_subtasks,
                "completed": self._current_checkpoint.completed_subtasks,
                "failed": self._current_checkpoint.failed_subtasks,
                "progress": self._current_checkpoint.progress_percentage,
                "current_batch": self._current_checkpoint.current_batch,
                "status": self._current_checkpoint.status.value,
            })
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress."""
        if not self._current_checkpoint:
            return {"status": "not_started"}
        
        return {
            "checkpoint_id": self._current_checkpoint.checkpoint_id,
            "task_id": self._current_checkpoint.task_id,
            "total": self._current_checkpoint.total_subtasks,
            "completed": self._current_checkpoint.completed_subtasks,
            "failed": self._current_checkpoint.failed_subtasks,
            "progress": self._current_checkpoint.progress_percentage,
            "status": self._current_checkpoint.status.value,
        }
