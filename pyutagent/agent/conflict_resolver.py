"""Conflict Resolver for detecting and resolving task conflicts.

This module provides:
- ConflictResolver: Detect and resolve conflicts between tasks
- Conflict types: Resource, Logic, Priority
- Resolution strategies: Priority, Voting, Manual
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts."""
    RESOURCE = "resource"
    LOGIC = "logic"
    PRIORITY = "priority"
    DEPENDENCY = "dependency"
    DATA = "data"
    TIMING = "timing"
    CUSTOM = "custom"


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    PRIORITY_BASED = "priority_based"
    FIRST_WINS = "first_wins"
    LAST_WINS = "last_wins"
    VOTING = "voting"
    MANUAL = "manual"
    ROLLBACK = "rollback"
    MERGE = "merge"
    SKIP = "skip"


class ConflictStatus(Enum):
    """Status of a conflict."""
    DETECTED = "detected"
    ANALYZING = "analyzing"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass
class ConflictResource:
    """Resource involved in a conflict."""
    resource_id: str
    resource_type: str
    access_mode: str = "read"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictParty:
    """Party involved in a conflict."""
    party_id: str
    party_type: str
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    requested_action: str = ""
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conflict:
    """Represents a detected conflict."""
    conflict_id: str
    conflict_type: ConflictType
    severity: ConflictSeverity
    description: str
    resources: List[ConflictResource] = field(default_factory=list)
    parties: List[ConflictParty] = field(default_factory=list)
    status: ConflictStatus = ConflictStatus.DETECTED
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None
    resolution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resolution:
    """Resolution for a conflict."""
    resolution_id: str
    conflict_id: str
    strategy: ResolutionStrategy
    action: str
    winner: Optional[str] = None
    loser: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    applied_at: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True


@dataclass
class ConflictRecord:
    """Record of a conflict and its resolution."""
    conflict: Conflict
    resolution: Optional[Resolution] = None
    escalation_level: int = 0
    retry_count: int = 0
    notes: List[str] = field(default_factory=list)


class ConflictDetector:
    """Detects conflicts between tasks."""

    def __init__(self):
        """Initialize conflict detector."""
        self._resource_locks: Dict[str, str] = {}
        self._pending_access: Dict[str, List[str]] = {}

    def detect_resource_conflicts(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Conflict]:
        """Detect resource conflicts between tasks.

        Args:
            tasks: List of task dictionaries

        Returns:
            List of detected conflicts
        """
        conflicts = []
        resource_tasks: Dict[str, List[Tuple[str, str]]] = {}

        for task in tasks:
            task_id = task.get("id", "")
            resources = task.get("resources", [])

            for resource in resources:
                resource_id = resource.get("resource_id", "")
                access_mode = resource.get("access_mode", "read")

                if resource_id not in resource_tasks:
                    resource_tasks[resource_id] = []

                resource_tasks[resource_id].append((task_id, access_mode))

        for resource_id, task_list in resource_tasks.items():
            write_tasks = [t for t in task_list if t[1] == "write"]

            if len(write_tasks) > 1:
                conflict = Conflict(
                    conflict_id=str(uuid4()),
                    conflict_type=ConflictType.RESOURCE,
                    severity=ConflictSeverity.HIGH,
                    description=f"Multiple tasks writing to resource: {resource_id}",
                    resources=[ConflictResource(
                        resource_id=resource_id,
                        resource_type="file",
                        access_mode="write"
                    )],
                    parties=[
                        ConflictParty(
                            party_id=task_id,
                            party_type="task",
                            task_id=task_id,
                            requested_action="write"
                        )
                        for task_id, _ in write_tasks
                    ]
                )
                conflicts.append(conflict)

            elif len(write_tasks) == 1 and len(task_list) > 1:
                conflict = Conflict(
                    conflict_id=str(uuid4()),
                    conflict_type=ConflictType.RESOURCE,
                    severity=ConflictSeverity.MEDIUM,
                    description=f"Read-write conflict on resource: {resource_id}",
                    resources=[ConflictResource(
                        resource_id=resource_id,
                        resource_type="file",
                        access_mode="mixed"
                    )],
                    parties=[
                        ConflictParty(
                            party_id=task_id,
                            party_type="task",
                            task_id=task_id,
                            requested_action=mode
                        )
                        for task_id, mode in task_list
                    ]
                )
                conflicts.append(conflict)

        return conflicts

    def detect_dependency_conflicts(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Conflict]:
        """Detect dependency conflicts (cycles).

        Args:
            tasks: List of task dictionaries

        Returns:
            List of detected conflicts
        """
        conflicts = []
        task_map = {t.get("id"): t for t in tasks}

        visited = set()
        rec_stack = set()
        cycles = []

        def find_cycle(task_id: str, path: List[str]) -> None:
            visited.add(task_id)
            rec_stack.add(task_id)
            path.append(task_id)

            task = task_map.get(task_id)
            if task:
                for dep_id in task.get("dependencies", []):
                    if dep_id not in visited:
                        find_cycle(dep_id, path)
                    elif dep_id in rec_stack:
                        cycle_start = path.index(dep_id)
                        cycles.append(path[cycle_start:] + [dep_id])

            path.pop()
            rec_stack.remove(task_id)

        for task_id in task_map:
            if task_id not in visited:
                find_cycle(task_id, [])

        for cycle in cycles:
            conflict = Conflict(
                conflict_id=str(uuid4()),
                conflict_type=ConflictType.DEPENDENCY,
                severity=ConflictSeverity.CRITICAL,
                description=f"Circular dependency detected: {' -> '.join(cycle)}",
                parties=[
                    ConflictParty(
                        party_id=task_id,
                        party_type="task",
                        task_id=task_id
                    )
                    for task_id in set(cycle)
                ]
            )
            conflicts.append(conflict)

        return conflicts

    def detect_priority_conflicts(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Conflict]:
        """Detect priority conflicts.

        Args:
            tasks: List of task dictionaries

        Returns:
            List of detected conflicts
        """
        conflicts = []

        same_priority_groups: Dict[int, List[Dict[str, Any]]] = {}
        for task in tasks:
            priority = task.get("priority", 5)
            if priority not in same_priority_groups:
                same_priority_groups[priority] = []
            same_priority_groups[priority].append(task)

        for priority, group in same_priority_groups.items():
            if len(group) > 1:
                conflict = Conflict(
                    conflict_id=str(uuid4()),
                    conflict_type=ConflictType.PRIORITY,
                    severity=ConflictSeverity.LOW,
                    description=f"Multiple tasks with same priority {priority}",
                    parties=[
                        ConflictParty(
                            party_id=t.get("id", ""),
                            party_type="task",
                            task_id=t.get("id", ""),
                            priority=priority
                        )
                        for t in group
                    ]
                )
                conflicts.append(conflict)

        return conflicts


class ConflictResolver:
    """Resolves conflicts between tasks.

    Features:
    - Multiple resolution strategies
    - Automatic escalation
    - Conflict history tracking
    - Custom strategy registration
    """

    def __init__(
        self,
        default_strategy: ResolutionStrategy = ResolutionStrategy.PRIORITY_BASED,
        auto_resolve: bool = True,
        escalation_threshold: int = 3
    ):
        """Initialize conflict resolver.

        Args:
            default_strategy: Default resolution strategy
            auto_resolve: Whether to auto-resolve conflicts
            escalation_threshold: Threshold for escalation
        """
        self.default_strategy = default_strategy
        self.auto_resolve = auto_resolve
        self.escalation_threshold = escalation_threshold

        self._detector = ConflictDetector()
        self._conflict_history: List[ConflictRecord] = []
        self._strategies: Dict[ConflictType, ResolutionStrategy] = {}
        self._custom_resolvers: Dict[str, Callable] = {}
        self._manual_queue: List[Conflict] = []

        self._stats = {
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "conflicts_escalated": 0,
            "conflicts_failed": 0
        }

        self._register_default_strategies()

        logger.info(f"[ConflictResolver] Initialized with strategy: {default_strategy.value}")

    def _register_default_strategies(self) -> None:
        """Register default resolution strategies."""
        self._strategies = {
            ConflictType.RESOURCE: ResolutionStrategy.PRIORITY_BASED,
            ConflictType.LOGIC: ResolutionStrategy.MANUAL,
            ConflictType.PRIORITY: ResolutionStrategy.FIRST_WINS,
            ConflictType.DEPENDENCY: ResolutionStrategy.ROLLBACK,
            ConflictType.DATA: ResolutionStrategy.MERGE,
            ConflictType.TIMING: ResolutionStrategy.PRIORITY_BASED,
            ConflictType.CUSTOM: ResolutionStrategy.MANUAL
        }

    def register_strategy(
        self,
        conflict_type: ConflictType,
        strategy: ResolutionStrategy
    ) -> None:
        """Register a resolution strategy for a conflict type.

        Args:
            conflict_type: Type of conflict
            strategy: Strategy to use
        """
        self._strategies[conflict_type] = strategy
        logger.info(f"[ConflictResolver] Registered strategy {strategy.value} for {conflict_type.value}")

    def register_custom_resolver(
        self,
        name: str,
        resolver: Callable[[Conflict], Resolution]
    ) -> None:
        """Register a custom resolver function.

        Args:
            name: Resolver name
            resolver: Resolver function
        """
        self._custom_resolvers[name] = resolver
        logger.info(f"[ConflictResolver] Registered custom resolver: {name}")

    def detect_conflicts(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Conflict]:
        """Detect all types of conflicts.

        Args:
            tasks: List of task dictionaries

        Returns:
            List of detected conflicts
        """
        all_conflicts = []

        all_conflicts.extend(self._detector.detect_resource_conflicts(tasks))
        all_conflicts.extend(self._detector.detect_dependency_conflicts(tasks))
        all_conflicts.extend(self._detector.detect_priority_conflicts(tasks))

        for conflict in all_conflicts:
            record = ConflictRecord(conflict=conflict)
            self._conflict_history.append(record)
            self._stats["conflicts_detected"] += 1

        if all_conflicts:
            logger.info(f"[ConflictResolver] Detected {len(all_conflicts)} conflicts")

        return all_conflicts

    async def resolve(
        self,
        conflict: Conflict,
        strategy: Optional[ResolutionStrategy] = None
    ) -> Resolution:
        """Resolve a conflict.

        Args:
            conflict: Conflict to resolve
            strategy: Optional strategy override

        Returns:
            Resolution
        """
        if strategy is None:
            strategy = self._strategies.get(
                conflict.conflict_type,
                self.default_strategy
            )

        conflict.status = ConflictStatus.RESOLVING

        try:
            resolution = await self._apply_strategy(conflict, strategy)

            conflict.status = ConflictStatus.RESOLVED
            conflict.resolved_at = datetime.now().isoformat()
            conflict.resolution = resolution.action

            self._stats["conflicts_resolved"] += 1

            self._update_conflict_record(conflict.conflict_id, resolution)

            logger.info(f"[ConflictResolver] Resolved conflict {conflict.conflict_id} "
                       f"using {strategy.value}")

            return resolution

        except Exception as e:
            logger.error(f"[ConflictResolver] Failed to resolve conflict: {e}")
            conflict.status = ConflictStatus.FAILED
            self._stats["conflicts_failed"] += 1

            return Resolution(
                resolution_id=str(uuid4()),
                conflict_id=conflict.conflict_id,
                strategy=strategy,
                action=f"Failed: {str(e)}",
                success=False
            )

    async def _apply_strategy(
        self,
        conflict: Conflict,
        strategy: ResolutionStrategy
    ) -> Resolution:
        """Apply a resolution strategy.

        Args:
            conflict: Conflict to resolve
            strategy: Strategy to apply

        Returns:
            Resolution
        """
        if strategy == ResolutionStrategy.PRIORITY_BASED:
            return self._resolve_by_priority(conflict)
        elif strategy == ResolutionStrategy.FIRST_WINS:
            return self._resolve_first_wins(conflict)
        elif strategy == ResolutionStrategy.LAST_WINS:
            return self._resolve_last_wins(conflict)
        elif strategy == ResolutionStrategy.VOTING:
            return await self._resolve_by_voting(conflict)
        elif strategy == ResolutionStrategy.MANUAL:
            return self._resolve_manual(conflict)
        elif strategy == ResolutionStrategy.ROLLBACK:
            return self._resolve_rollback(conflict)
        elif strategy == ResolutionStrategy.MERGE:
            return self._resolve_merge(conflict)
        elif strategy == ResolutionStrategy.SKIP:
            return self._resolve_skip(conflict)
        else:
            return self._resolve_by_priority(conflict)

    def _resolve_by_priority(self, conflict: Conflict) -> Resolution:
        """Resolve by selecting highest priority party."""
        if not conflict.parties:
            return Resolution(
                resolution_id=str(uuid4()),
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.PRIORITY_BASED,
                action="No parties to resolve"
            )

        sorted_parties = sorted(
            conflict.parties,
            key=lambda p: p.priority
        )

        winner = sorted_parties[0]
        losers = sorted_parties[1:]

        return Resolution(
            resolution_id=str(uuid4()),
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.PRIORITY_BASED,
            action=f"Selected {winner.party_id} based on priority",
            winner=winner.party_id,
            loser=",".join(l.party_id for l in losers),
            details={
                "winner_priority": winner.priority,
                "reason": "lowest_priority_value_wins"
            }
        )

    def _resolve_first_wins(self, conflict: Conflict) -> Resolution:
        """Resolve by selecting first party."""
        if not conflict.parties:
            return Resolution(
                resolution_id=str(uuid4()),
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.FIRST_WINS,
                action="No parties to resolve"
            )

        winner = conflict.parties[0]
        losers = conflict.parties[1:]

        return Resolution(
            resolution_id=str(uuid4()),
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.FIRST_WINS,
            action=f"Selected {winner.party_id} (first)",
            winner=winner.party_id,
            loser=",".join(l.party_id for l in losers)
        )

    def _resolve_last_wins(self, conflict: Conflict) -> Resolution:
        """Resolve by selecting last party."""
        if not conflict.parties:
            return Resolution(
                resolution_id=str(uuid4()),
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.LAST_WINS,
                action="No parties to resolve"
            )

        winner = conflict.parties[-1]
        losers = conflict.parties[:-1]

        return Resolution(
            resolution_id=str(uuid4()),
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.LAST_WINS,
            action=f"Selected {winner.party_id} (last)",
            winner=winner.party_id,
            loser=",".join(l.party_id for l in losers)
        )

    async def _resolve_by_voting(self, conflict: Conflict) -> Resolution:
        """Resolve by voting among parties."""
        votes: Dict[str, int] = {}

        for party in conflict.parties:
            vote_target = party.metadata.get("vote_for", party.party_id)
            votes[vote_target] = votes.get(vote_target, 0) + 1

        if not votes:
            return self._resolve_by_priority(conflict)

        winner = max(votes.keys(), key=lambda k: votes[k])

        return Resolution(
            resolution_id=str(uuid4()),
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.VOTING,
            action=f"Selected {winner} by voting",
            winner=winner,
            details={"votes": votes}
        )

    def _resolve_manual(self, conflict: Conflict) -> Resolution:
        """Queue conflict for manual resolution."""
        self._manual_queue.append(conflict)

        conflict.status = ConflictStatus.ESCALATED
        self._stats["conflicts_escalated"] += 1

        return Resolution(
            resolution_id=str(uuid4()),
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.MANUAL,
            action="Queued for manual resolution",
            success=False
        )

    def _resolve_rollback(self, conflict: Conflict) -> Resolution:
        """Resolve by rolling back conflicting operations."""
        rolled_back = []
        for party in conflict.parties[1:]:
            rolled_back.append(party.party_id)

        return Resolution(
            resolution_id=str(uuid4()),
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.ROLLBACK,
            action=f"Rolled back {len(rolled_back)} operations",
            winner=conflict.parties[0].party_id if conflict.parties else None,
            loser=",".join(rolled_back),
            details={"rolled_back": rolled_back}
        )

    def _resolve_merge(self, conflict: Conflict) -> Resolution:
        """Resolve by merging conflicting data."""
        return Resolution(
            resolution_id=str(uuid4()),
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.MERGE,
            action="Merged conflicting data",
            details={"merged_parties": [p.party_id for p in conflict.parties]}
        )

    def _resolve_skip(self, conflict: Conflict) -> Resolution:
        """Resolve by skipping conflicting tasks."""
        return Resolution(
            resolution_id=str(uuid4()),
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.SKIP,
            action="Skipped conflicting tasks",
            details={"skipped_parties": [p.party_id for p in conflict.parties]}
        )

    def _update_conflict_record(
        self,
        conflict_id: str,
        resolution: Resolution
    ) -> None:
        """Update conflict record with resolution."""
        for record in self._conflict_history:
            if record.conflict.conflict_id == conflict_id:
                record.resolution = resolution
                break

    def get_pending_manual_conflicts(self) -> List[Conflict]:
        """Get conflicts pending manual resolution.

        Returns:
            List of conflicts
        """
        return self._manual_queue.copy()

    def resolve_manual_conflict(
        self,
        conflict_id: str,
        winner_id: str
    ) -> bool:
        """Manually resolve a conflict.

        Args:
            conflict_id: Conflict ID
            winner_id: ID of winning party

        Returns:
            True if resolved successfully
        """
        for i, conflict in enumerate(self._manual_queue):
            if conflict.conflict_id == conflict_id:
                resolution = Resolution(
                    resolution_id=str(uuid4()),
                    conflict_id=conflict_id,
                    strategy=ResolutionStrategy.MANUAL,
                    action=f"Manually selected {winner_id}",
                    winner=winner_id
                )

                conflict.status = ConflictStatus.RESOLVED
                conflict.resolved_at = datetime.now().isoformat()
                conflict.resolution = resolution.action

                self._update_conflict_record(conflict_id, resolution)

                self._manual_queue.pop(i)
                self._stats["conflicts_resolved"] += 1
                self._stats["conflicts_escalated"] -= 1

                logger.info(f"[ConflictResolver] Manually resolved conflict {conflict_id}")

                return True

        return False

    def get_conflict_history(self, limit: int = 50) -> List[ConflictRecord]:
        """Get conflict history.

        Args:
            limit: Maximum records to return

        Returns:
            List of conflict records
        """
        return self._conflict_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "conflicts_detected": self._stats["conflicts_detected"],
            "conflicts_resolved": self._stats["conflicts_resolved"],
            "conflicts_escalated": self._stats["conflicts_escalated"],
            "conflicts_failed": self._stats["conflicts_failed"],
            "pending_manual": len(self._manual_queue),
            "resolution_rate": (
                self._stats["conflicts_resolved"] / self._stats["conflicts_detected"]
                if self._stats["conflicts_detected"] > 0 else 0
            )
        }


def create_conflict_resolver(
    default_strategy: ResolutionStrategy = ResolutionStrategy.PRIORITY_BASED,
    auto_resolve: bool = True
) -> ConflictResolver:
    """Create a ConflictResolver.

    Args:
        default_strategy: Default resolution strategy
        auto_resolve: Whether to auto-resolve

    Returns:
        ConflictResolver instance
    """
    return ConflictResolver(
        default_strategy=default_strategy,
        auto_resolve=auto_resolve
    )
