"""Procedural Memory - Skill and strategy learning.

This module provides procedural memory for learning and retrieving
skills and strategies based on task execution history.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """A skill or strategy learned from execution."""
    skill_id: str
    name: str
    task_type: str
    steps: List[str]
    success_rate: float
    usage_count: int
    last_used: datetime
    avg_duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["last_used"] = self.last_used.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create from dictionary."""
        data = data.copy()
        data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


class ProceduralMemory:
    """Procedural Memory - Skill and strategy learning.

    Features:
    - Learns successful strategies
    - Tracks skill success rates
    - Retrieves relevant skills for tasks
    - Persists to SQLite
    """

    def __init__(self, storage_path: str):
        """Initialize procedural memory.

        Args:
            storage_path: Path to SQLite database
        """
        self.storage_path = storage_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_storage()

    def _init_storage(self):
        """Initialize storage and tables."""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        self._connect()
        self._create_tables()

    def _connect(self):
        """Connect to database."""
        if self._conn is not None:
            return
        self._conn = sqlite3.connect(self.storage_path)
        self._conn.row_factory = sqlite3.Row

    def _create_tables(self):
        """Create necessary tables."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                steps TEXT NOT NULL,
                success_rate REAL NOT NULL,
                usage_count INTEGER NOT NULL,
                last_used TEXT NOT NULL,
                avg_duration_seconds REAL NOT NULL
            )
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_type_skills
            ON skills(task_type, success_rate DESC)
        """)

        self._conn.commit()

    async def learn(
        self,
        task_type: str,
        strategy: Dict[str, Any],
        success: bool,
        duration_seconds: float
    ) -> Skill:
        """Learn a new skill or update existing one.

        Args:
            task_type: Type of task
            strategy: Strategy details
            success: Whether execution was successful
            duration_seconds: Execution duration

        Returns:
            Learned skill
        """
        import uuid

        skill_name = strategy.get("name", f"Strategy for {task_type}")
        steps = strategy.get("steps", [])

        existing = self._find_skill(task_type, skill_name)

        if existing:
            return await self._update_skill(
                existing,
                success,
                duration_seconds
            )

        skill_id = str(uuid.uuid4())
        skill = Skill(
            skill_id=skill_id,
            name=skill_name,
            task_type=task_type,
            steps=steps,
            success_rate=1.0 if success else 0.0,
            usage_count=1,
            last_used=datetime.now(),
            avg_duration_seconds=duration_seconds
        )

        self._conn.execute(
            """
            INSERT INTO skills
            (skill_id, name, task_type, steps, success_rate, usage_count, last_used, avg_duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill.skill_id,
                skill.name,
                skill.task_type,
                json.dumps(skill.steps),
                skill.success_rate,
                skill.usage_count,
                skill.last_used.isoformat(),
                skill.avg_duration_seconds
            )
        )
        self._conn.commit()

        logger.info(f"[ProceduralMemory] Learned new skill: {skill_name} for {task_type}")
        return skill

    def _find_skill(self, task_type: str, name: str) -> Optional[Skill]:
        """Find existing skill."""
        cursor = self._conn.execute(
            "SELECT * FROM skills WHERE task_type = ? AND name = ?",
            (task_type, name)
        )
        row = cursor.fetchone()

        if not row:
            return None

        return Skill(
            skill_id=row["skill_id"],
            name=row["name"],
            task_type=row["task_type"],
            steps=json.loads(row["steps"]),
            success_rate=row["success_rate"],
            usage_count=row["usage_count"],
            last_used=datetime.fromisoformat(row["last_used"]),
            avg_duration_seconds=row["avg_duration_seconds"]
        )

    async def _update_skill(
        self,
        skill: Skill,
        success: bool,
        duration_seconds: float
    ) -> Skill:
        """Update existing skill with new execution result."""
        new_count = skill.usage_count + 1

        if success:
            new_success_rate = (
                (skill.success_rate * skill.usage_count + 1.0) / new_count
            )
        else:
            new_success_rate = (
                (skill.success_rate * skill.usage_count) / new_count
            )

        new_avg_duration = (
            (skill.avg_duration_seconds * skill.usage_count + duration_seconds) / new_count
        )

        self._conn.execute(
            """
            UPDATE skills
            SET success_rate = ?, usage_count = ?, last_used = ?, avg_duration_seconds = ?
            WHERE skill_id = ?
            """,
            (
                new_success_rate,
                new_count,
                datetime.now().isoformat(),
                new_avg_duration,
                skill.skill_id
            )
        )
        self._conn.commit()

        skill.success_rate = new_success_rate
        skill.usage_count = new_count
        skill.last_used = datetime.now()
        skill.avg_duration_seconds = new_avg_duration

        logger.info(f"[ProceduralMemory] Updated skill {skill.name}: success_rate={new_success_rate:.2f}")
        return skill

    async def retrieve(
        self,
        task_type: str,
        min_success_rate: float = 0.0,
        limit: int = 5
    ) -> List[Skill]:
        """Retrieve relevant skills for a task type.

        Args:
            task_type: Type of task
            min_success_rate: Minimum success rate filter
            limit: Maximum results

        Returns:
            List of relevant skills sorted by success rate
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM skills
            WHERE task_type = ? AND success_rate >= ?
            ORDER BY success_rate DESC, usage_count DESC
            LIMIT ?
            """,
            (task_type, min_success_rate, limit)
        )

        rows = cursor.fetchall()
        skills = []

        for row in rows:
            skills.append(Skill(
                skill_id=row["skill_id"],
                name=row["name"],
                task_type=row["task_type"],
                steps=json.loads(row["steps"]),
                success_rate=row["success_rate"],
                usage_count=row["usage_count"],
                last_used=datetime.fromisoformat(row["last_used"]),
                avg_duration_seconds=row["avg_duration_seconds"]
            ))

        return skills

    async def get_best_skill(
        self,
        task_type: str
    ) -> Optional[Skill]:
        """Get the best performing skill for a task type.

        Args:
            task_type: Type of task

        Returns:
            Best skill or None
        """
        skills = await self.retrieve(task_type, limit=1)
        return skills[0] if skills else None

    async def get_all_skills(self) -> List[Skill]:
        """Get all skills.

        Returns:
            List of all skills
        """
        cursor = self._conn.execute(
            "SELECT * FROM skills ORDER BY success_rate DESC"
        )

        rows = cursor.fetchall()
        skills = []

        for row in rows:
            skills.append(Skill(
                skill_id=row["skill_id"],
                name=row["name"],
                task_type=row["task_type"],
                steps=json.loads(row["steps"]),
                success_rate=row["success_rate"],
                usage_count=row["usage_count"],
                last_used=datetime.fromisoformat(row["last_used"]),
                avg_duration_seconds=row["avg_duration_seconds"]
            ))

        return skills

    async def delete_skill(self, skill_id: str) -> bool:
        """Delete a skill.

        Args:
            skill_id: Skill ID

        Returns:
            True if deleted
        """
        cursor = self._conn.execute(
            "DELETE FROM skills WHERE skill_id = ?",
            (skill_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


def create_procedural_memory(
    storage_dir: str = ".pyutagent"
) -> ProceduralMemory:
    """Create procedural memory instance.

    Args:
        storage_dir: Storage directory

    Returns:
        ProceduralMemory instance
    """
    import os
    db_path = os.path.join(storage_dir, "procedural_memory.db")
    return ProceduralMemory(db_path)
