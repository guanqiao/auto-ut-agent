"""Shared Context Manager for agent context sharing.

This module provides:
- SharedContextManager: Manage shared context between agents
- AgentContext: Context for individual agents
- Context snapshots and recovery
"""

import copy
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class ContextScope(Enum):
    """Scope of context data."""
    GLOBAL = "global"
    SESSION = "session"
    TASK = "task"
    AGENT = "agent"


class ContextVisibility(Enum):
    """Visibility of context data."""
    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"


@dataclass
class ContextEntry:
    """A single entry in the context."""
    key: str
    value: Any
    scope: ContextScope = ContextScope.SESSION
    visibility: ContextVisibility = ContextVisibility.PUBLIC
    owner_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "scope": self.scope.value,
            "visibility": self.visibility.value,
            "owner_id": self.owner_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextEntry":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            scope=ContextScope(data.get("scope", "session")),
            visibility=ContextVisibility(data.get("visibility", "public")),
            owner_id=data.get("owner_id"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            version=data.get("version", 1),
            metadata=data.get("metadata", {})
        )


@dataclass
class AgentContext:
    """Context for an individual agent."""
    agent_id: str
    parent_id: Optional[str] = None
    entries: Dict[str, ContextEntry] = field(default_factory=dict)
    inherited_keys: Set[str] = field(default_factory=set)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: Context key
            default: Default value if not found

        Returns:
            Context value or default
        """
        if key in self.entries:
            return self.entries[key].value
        return default

    def set(
        self,
        key: str,
        value: Any,
        scope: ContextScope = ContextScope.AGENT,
        visibility: ContextVisibility = ContextVisibility.PUBLIC
    ) -> None:
        """Set a context value.

        Args:
            key: Context key
            value: Context value
            scope: Context scope
            visibility: Context visibility
        """
        if key in self.entries:
            entry = self.entries[key]
            entry.value = value
            entry.updated_at = datetime.now().isoformat()
            entry.version += 1
        else:
            self.entries[key] = ContextEntry(
                key=key,
                value=value,
                scope=scope,
                visibility=visibility,
                owner_id=self.agent_id
            )

        self.updated_at = datetime.now().isoformat()

    def delete(self, key: str) -> bool:
        """Delete a context entry.

        Args:
            key: Context key

        Returns:
            True if deleted
        """
        if key in self.entries:
            del self.entries[key]
            self.updated_at = datetime.now().isoformat()
            return True
        return False

    def keys(self) -> List[str]:
        """Get all context keys."""
        return list(self.entries.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "parent_id": self.parent_id,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
            "inherited_keys": list(self.inherited_keys),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentContext":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            parent_id=data.get("parent_id"),
            entries={
                k: ContextEntry.from_dict(v)
                for k, v in data.get("entries", {}).items()
            },
            inherited_keys=set(data.get("inherited_keys", [])),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )


@dataclass
class ContextSnapshot:
    """Snapshot of context state."""
    snapshot_id: str
    agent_id: str
    context: AgentContext
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "agent_id": self.agent_id,
            "context": self.context.to_dict(),
            "timestamp": self.timestamp,
            "label": self.label,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextSnapshot":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            agent_id=data["agent_id"],
            context=AgentContext.from_dict(data["context"]),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            label=data.get("label"),
            metadata=data.get("metadata", {})
        )


class SharedContextManager:
    """Manager for shared context between agents.

    Features:
    - Hierarchical context inheritance
    - Context isolation
    - Snapshots and recovery
    - Incremental updates
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        max_snapshots: int = 100
    ):
        """Initialize SharedContextManager.

        Args:
            persistence_path: Optional path for persisting context
            max_snapshots: Maximum snapshots to keep
        """
        self.persistence_path = persistence_path
        self.max_snapshots = max_snapshots

        self._contexts: Dict[str, AgentContext] = {}
        self._snapshots: Dict[str, ContextSnapshot] = {}
        self._global_context: Dict[str, ContextEntry] = {}

        self._stats = {
            "contexts_created": 0,
            "snapshots_created": 0,
            "snapshots_restored": 0,
            "updates": 0
        }

        logger.info("[SharedContextManager] Initialized")

    def create_context(
        self,
        parent_id: str,
        child_id: str,
        inherit: bool = True
    ) -> AgentContext:
        """Create a context for a child agent.

        Args:
            parent_id: Parent agent ID
            child_id: Child agent ID
            inherit: Whether to inherit parent context

        Returns:
            AgentContext for the child
        """
        context = AgentContext(
            agent_id=child_id,
            parent_id=parent_id
        )

        if inherit and parent_id in self._contexts:
            parent_context = self._contexts[parent_id]
            self._inherit_context(parent_context, context)

        self._contexts[child_id] = context
        self._stats["contexts_created"] += 1

        logger.info(f"[SharedContextManager] Created context for {child_id} (parent={parent_id})")

        return context

    def _inherit_context(
        self,
        parent: AgentContext,
        child: AgentContext
    ) -> None:
        """Inherit context from parent.

        Args:
            parent: Parent context
            child: Child context
        """
        for key, entry in parent.entries.items():
            if entry.visibility in [ContextVisibility.PUBLIC, ContextVisibility.PROTECTED]:
                if entry.scope in [ContextScope.GLOBAL, ContextScope.SESSION]:
                    child.entries[key] = copy.deepcopy(entry)
                    child.inherited_keys.add(key)

        for key, entry in self._global_context.items():
            if key not in child.entries:
                child.entries[key] = copy.deepcopy(entry)
                child.inherited_keys.add(key)

    def get_context(self, agent_id: str) -> Optional[AgentContext]:
        """Get context for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            AgentContext or None
        """
        return self._contexts.get(agent_id)

    def update_context(
        self,
        agent_id: str,
        updates: Dict[str, Any],
        scope: ContextScope = ContextScope.AGENT,
        visibility: ContextVisibility = ContextVisibility.PUBLIC
    ) -> bool:
        """Update context for an agent.

        Args:
            agent_id: Agent ID
            updates: Dictionary of updates
            scope: Context scope
            visibility: Context visibility

        Returns:
            True if updated successfully
        """
        context = self._contexts.get(agent_id)
        if not context:
            logger.warning(f"[SharedContextManager] Context not found: {agent_id}")
            return False

        for key, value in updates.items():
            context.set(key, value, scope, visibility)

        self._stats["updates"] += 1

        return True

    def set_value(
        self,
        agent_id: str,
        key: str,
        value: Any,
        scope: ContextScope = ContextScope.AGENT,
        visibility: ContextVisibility = ContextVisibility.PUBLIC
    ) -> bool:
        """Set a single context value.

        Args:
            agent_id: Agent ID
            key: Context key
            value: Context value
            scope: Context scope
            visibility: Context visibility

        Returns:
            True if set successfully
        """
        context = self._contexts.get(agent_id)
        if not context:
            return False

        context.set(key, value, scope, visibility)
        self._stats["updates"] += 1

        return True

    def get_value(
        self,
        agent_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """Get a context value.

        Args:
            agent_id: Agent ID
            key: Context key
            default: Default value

        Returns:
            Context value or default
        """
        context = self._contexts.get(agent_id)
        if context:
            return context.get(key, default)
        return default

    def delete_context(self, agent_id: str) -> bool:
        """Delete an agent's context.

        Args:
            agent_id: Agent ID

        Returns:
            True if deleted
        """
        if agent_id in self._contexts:
            del self._contexts[agent_id]
            logger.info(f"[SharedContextManager] Deleted context: {agent_id}")
            return True
        return False

    def set_global(self, key: str, value: Any) -> None:
        """Set a global context value.

        Args:
            key: Context key
            value: Context value
        """
        entry = ContextEntry(
            key=key,
            value=value,
            scope=ContextScope.GLOBAL,
            visibility=ContextVisibility.PUBLIC
        )
        self._global_context[key] = entry

        for context in self._contexts.values():
            if key not in context.entries or key in context.inherited_keys:
                context.entries[key] = copy.deepcopy(entry)
                context.inherited_keys.add(key)

        self._stats["updates"] += 1

    def get_global(self, key: str, default: Any = None) -> Any:
        """Get a global context value.

        Args:
            key: Context key
            default: Default value

        Returns:
            Global value or default
        """
        if key in self._global_context:
            return self._global_context[key].value
        return default

    def create_snapshot(
        self,
        agent_id: str,
        label: Optional[str] = None
    ) -> Optional[ContextSnapshot]:
        """Create a snapshot of an agent's context.

        Args:
            agent_id: Agent ID
            label: Optional snapshot label

        Returns:
            ContextSnapshot or None
        """
        context = self._contexts.get(agent_id)
        if not context:
            return None

        snapshot = ContextSnapshot(
            snapshot_id=str(uuid4()),
            agent_id=agent_id,
            context=copy.deepcopy(context),
            label=label
        )

        self._snapshots[snapshot.snapshot_id] = snapshot
        self._stats["snapshots_created"] += 1

        self._cleanup_old_snapshots()

        logger.info(f"[SharedContextManager] Created snapshot {snapshot.snapshot_id} for {agent_id}")

        return snapshot

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore context from a snapshot.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            True if restored successfully
        """
        snapshot = self._snapshots.get(snapshot_id)
        if not snapshot:
            logger.warning(f"[SharedContextManager] Snapshot not found: {snapshot_id}")
            return False

        self._contexts[snapshot.agent_id] = copy.deepcopy(snapshot.context)
        self._stats["snapshots_restored"] += 1

        logger.info(f"[SharedContextManager] Restored snapshot {snapshot_id}")

        return True

    def get_snapshot(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        """Get a snapshot by ID.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            ContextSnapshot or None
        """
        return self._snapshots.get(snapshot_id)

    def list_snapshots(
        self,
        agent_id: Optional[str] = None
    ) -> List[ContextSnapshot]:
        """List snapshots.

        Args:
            agent_id: Optional filter by agent

        Returns:
            List of snapshots
        """
        snapshots = list(self._snapshots.values())

        if agent_id:
            snapshots = [s for s in snapshots if s.agent_id == agent_id]

        return sorted(snapshots, key=lambda s: s.timestamp, reverse=True)

    def _cleanup_old_snapshots(self) -> None:
        """Clean up old snapshots."""
        if len(self._snapshots) > self.max_snapshots:
            sorted_snapshots = sorted(
                self._snapshots.items(),
                key=lambda x: x[1].timestamp
            )

            to_remove = len(self._snapshots) - self.max_snapshots
            for snapshot_id, _ in sorted_snapshots[:to_remove]:
                del self._snapshots[snapshot_id]

    def propagate_update(
        self,
        agent_id: str,
        key: str,
        value: Any
    ) -> int:
        """Propagate an update to child agents.

        Args:
            agent_id: Source agent ID
            key: Context key
            value: New value

        Returns:
            Number of agents updated
        """
        updated = 0

        for child_id, context in self._contexts.items():
            if context.parent_id == agent_id:
                if key in context.inherited_keys:
                    context.set(key, value)
                    updated += 1

        return updated

    def get_shared_keys(
        self,
        agent_id: str
    ) -> List[str]:
        """Get keys shared with an agent (inherited or global).

        Args:
            agent_id: Agent ID

        Returns:
            List of shared keys
        """
        context = self._contexts.get(agent_id)
        if not context:
            return []

        return list(context.inherited_keys)

    def isolate_context(
        self,
        agent_id: str,
        keys_to_isolate: Optional[List[str]] = None
    ) -> bool:
        """Isolate an agent's context from inheritance.

        Args:
            agent_id: Agent ID
            keys_to_isolate: Optional specific keys to isolate

        Returns:
            True if isolated successfully
        """
        context = self._contexts.get(agent_id)
        if not context:
            return False

        if keys_to_isolate:
            for key in keys_to_isolate:
                context.inherited_keys.discard(key)
        else:
            context.inherited_keys.clear()

        logger.info(f"[SharedContextManager] Isolated context for {agent_id}")

        return True

    def merge_contexts(
        self,
        target_agent_id: str,
        source_agent_id: str,
        overwrite: bool = False
    ) -> int:
        """Merge context from one agent to another.

        Args:
            target_agent_id: Target agent ID
            source_agent_id: Source agent ID
            overwrite: Whether to overwrite existing keys

        Returns:
            Number of keys merged
        """
        target = self._contexts.get(target_agent_id)
        source = self._contexts.get(source_agent_id)

        if not target or not source:
            return 0

        merged = 0
        for key, entry in source.entries.items():
            if overwrite or key not in target.entries:
                target.entries[key] = copy.deepcopy(entry)
                merged += 1

        target.updated_at = datetime.now().isoformat()

        return merged

    def export_context(self, agent_id: str) -> Optional[str]:
        """Export context as JSON string.

        Args:
            agent_id: Agent ID

        Returns:
            JSON string or None
        """
        context = self._contexts.get(agent_id)
        if not context:
            return None

        return json.dumps(context.to_dict(), indent=2)

    def import_context(
        self,
        agent_id: str,
        json_data: str,
        merge: bool = True
    ) -> bool:
        """Import context from JSON string.

        Args:
            agent_id: Agent ID
            json_data: JSON string
            merge: Whether to merge with existing

        Returns:
            True if imported successfully
        """
        try:
            data = json.loads(json_data)
            imported_context = AgentContext.from_dict(data)

            if merge and agent_id in self._contexts:
                existing = self._contexts[agent_id]
                for key, entry in imported_context.entries.items():
                    if key not in existing.entries:
                        existing.entries[key] = entry
            else:
                imported_context.agent_id = agent_id
                self._contexts[agent_id] = imported_context

            return True

        except Exception as e:
            logger.error(f"[SharedContextManager] Import failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "contexts": len(self._contexts),
            "snapshots": len(self._snapshots),
            "global_keys": len(self._global_context),
            "contexts_created": self._stats["contexts_created"],
            "snapshots_created": self._stats["snapshots_created"],
            "snapshots_restored": self._stats["snapshots_restored"],
            "updates": self._stats["updates"]
        }

    def clear_all(self) -> None:
        """Clear all contexts and snapshots."""
        self._contexts.clear()
        self._snapshots.clear()
        self._global_context.clear()
        logger.info("[SharedContextManager] Cleared all contexts")


def create_shared_context_manager(
    persistence_path: Optional[str] = None
) -> SharedContextManager:
    """Create a SharedContextManager.

    Args:
        persistence_path: Optional persistence path

    Returns:
        SharedContextManager instance
    """
    return SharedContextManager(persistence_path=persistence_path)
