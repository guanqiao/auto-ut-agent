"""Checkpoint management for execution state persistence.

This module provides checkpoint capabilities:
- State persistence
- Checkpoint creation and restoration
- Recovery from interruptions
- State versioning
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Checkpoint:
    """A checkpoint representing a saved state."""
    id: str
    created_at: str
    step: str
    iteration: int
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "step": self.step,
            "iteration": self.iteration,
            "state": self.state,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            step=data["step"],
            iteration=data["iteration"],
            state=data["state"],
            metadata=data.get("metadata", {})
        )


@dataclass
class CheckpointMetadata:
    """Metadata about checkpoints."""
    total_checkpoints: int = 0
    latest_checkpoint_id: Optional[str] = None
    earliest_checkpoint_id: Optional[str] = None
    total_size_bytes: int = 0
    created_by: str = ""


class CheckpointManager:
    """Manages checkpoints for execution state.
    
    Features:
    - State persistence
    - Checkpoint creation and restoration
    - Automatic cleanup
    - Version control
    """
    
    def __init__(
        self,
        persist_dir: str = ".pyutagent/checkpoints",
        max_checkpoints: int = 10,
        auto_cleanup: bool = True
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._metadata = CheckpointMetadata()
        
        self._load_checkpoints()
    
    def _load_checkpoints(self):
        """Load existing checkpoints from disk."""
        checkpoint_files = list(self.persist_dir.glob("checkpoint_*.json"))
        
        for cp_file in checkpoint_files:
            try:
                with open(cp_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                checkpoint = Checkpoint.from_dict(data)
                self._checkpoints[checkpoint.id] = checkpoint
            except Exception as e:
                logger.warning(f"[Checkpoint] Failed to load {cp_file}: {e}")
        
        self._update_metadata()
        logger.info(f"[Checkpoint] Loaded {len(self._checkpoints)} checkpoints")
    
    def _update_metadata(self):
        """Update metadata based on current checkpoints."""
        self._metadata.total_checkpoints = len(self._checkpoints)
        
        if self._checkpoints:
            sorted_cps = sorted(
                self._checkpoints.values(),
                key=lambda x: x.created_at
            )
            self._metadata.earliest_checkpoint_id = sorted_cps[0].id
            self._metadata.latest_checkpoint_id = sorted_cps[-1].id
    
    def save_checkpoint(
        self,
        step: str,
        iteration: int,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a new checkpoint.
        
        Args:
            step: Current step name
            iteration: Current iteration number
            state: State to save
            metadata: Additional metadata
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid.uuid4())[:8]
        
        checkpoint = Checkpoint(
            id=checkpoint_id,
            created_at=datetime.now().isoformat(),
            step=step,
            iteration=iteration,
            state=state,
            metadata=metadata or {}
        )
        
        self._checkpoints[checkpoint_id] = checkpoint
        
        self._persist_checkpoint(checkpoint)
        
        self._update_metadata()
        
        if self.auto_cleanup:
            self._cleanup_old_checkpoints()
        
        logger.info(
            f"[Checkpoint] Saved checkpoint {checkpoint_id} - "
            f"Step: {step}, Iteration: {iteration}"
        )
        
        return checkpoint_id
    
    def _persist_checkpoint(self, checkpoint: Checkpoint):
        """Persist a checkpoint to disk."""
        cp_file = self.persist_dir / f"checkpoint_{checkpoint.id}.json"
        
        try:
            with open(cp_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
            
            self._metadata.total_size_bytes += cp_file.stat().st_size
            
        except Exception as e:
            logger.error(f"[Checkpoint] Failed to persist {checkpoint.id}: {e}")
            raise
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self._checkpoints) <= self.max_checkpoints:
            return
        
        sorted_cps = sorted(
            self._checkpoints.values(),
            key=lambda x: x.created_at
        )
        
        to_remove = sorted_cps[:-self.max_checkpoints]
        
        for cp in to_remove:
            self._delete_checkpoint(cp.id)
        
        logger.info(f"[Checkpoint] Cleaned up {len(to_remove)} old checkpoints")
    
    def _delete_checkpoint(self, checkpoint_id: str):
        """Delete a checkpoint."""
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]
        
        cp_file = self.persist_dir / f"checkpoint_{checkpoint_id}.json"
        if cp_file.exists():
            cp_file.unlink()
    
    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        """Load a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to load, or None for latest
            
        Returns:
            Checkpoint or None
        """
        if checkpoint_id is None:
            checkpoint_id = self._metadata.latest_checkpoint_id
        
        if checkpoint_id is None:
            return None
        
        checkpoint = self._checkpoints.get(checkpoint_id)
        
        if checkpoint:
            logger.info(f"[Checkpoint] Loaded checkpoint {checkpoint_id}")
        else:
            logger.warning(f"[Checkpoint] Checkpoint {checkpoint_id} not found")
        
        return checkpoint
    
    def get_state(
        self,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get state from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint, or None for latest
            
        Returns:
            State dictionary or None
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        return checkpoint.state if checkpoint else None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        return [
            {
                "id": cp.id,
                "created_at": cp.created_at,
                "step": cp.step,
                "iteration": cp.iteration
            }
            for cp in sorted(
                self._checkpoints.values(),
                key=lambda x: x.created_at,
                reverse=True
            )
        ]
    
    def get_latest(self) -> Optional[Checkpoint]:
        """Get the latest checkpoint."""
        return self.load_checkpoint(self._metadata.latest_checkpoint_id)
    
    def get_earliest(self) -> Optional[Checkpoint]:
        """Get the earliest checkpoint."""
        return self.load_checkpoint(self._metadata.earliest_checkpoint_id)
    
    def clear(self):
        """Clear all checkpoints."""
        for cp_id in list(self._checkpoints.keys()):
            self._delete_checkpoint(cp_id)
        
        self._metadata = CheckpointMetadata()
        logger.info("[Checkpoint] Cleared all checkpoints")
    
    def get_metadata(self) -> CheckpointMetadata:
        """Get checkpoint metadata."""
        return self._metadata


class ResumableState:
    """Mixin for classes that can save and restore state."""
    
    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        self._checkpoint_manager = checkpoint_manager or CheckpointManager()
        self._current_checkpoint_id: Optional[str] = None
    
    def save_state(self, step: str, iteration: int = 0) -> str:
        """Save current state as a checkpoint.
        
        Args:
            step: Current step name
            iteration: Current iteration
            
        Returns:
            Checkpoint ID
        """
        state = self._serialize_state()
        
        self._current_checkpoint_id = self._checkpoint_manager.save_checkpoint(
            step=step,
            iteration=iteration,
            state=state,
            metadata={"class": self.__class__.__name__}
        )
        
        return self._current_checkpoint_id
    
    def restore_state(
        self,
        checkpoint_id: Optional[str] = None
    ) -> bool:
        """Restore state from a checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint, or None for latest
            
        Returns:
            True if restoration successful
        """
        state = self._checkpoint_manager.get_state(checkpoint_id)
        
        if state is None:
            return False
        
        self._deserialize_state(state)
        return True
    
    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize current state. Override in subclasses."""
        return {}
    
    def _deserialize_state(self, state: Dict[str, Any]):
        """Deserialize state. Override in subclasses."""
        pass
    
    def get_checkpoint_manager(self) -> CheckpointManager:
        """Get the checkpoint manager."""
        return self._checkpoint_manager


def create_checkpoint_manager(
    persist_dir: str = ".pyutagent/checkpoints",
    max_checkpoints: int = 10
) -> CheckpointManager:
    """Create a checkpoint manager.
    
    Args:
        persist_dir: Directory for checkpoint storage
        max_checkpoints: Maximum checkpoints to keep
        
    Returns:
        Configured CheckpointManager
    """
    return CheckpointManager(
        persist_dir=persist_dir,
        max_checkpoints=max_checkpoints
    )
