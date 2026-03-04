"""Episodic Memory - Cross-project experience accumulation.

This module provides episodic memory for recording and retrieving
task execution experiences across projects and sessions.
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
class Episode:
    """A single task execution record."""
    episode_id: str
    project: str
    task_type: str
    task_description: str
    steps: List[Dict[str, Any]]
    outcome: str
    duration_seconds: float
    lessons: List[str]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class ProjectSummary:
    """Summary of a project's execution history."""
    project: str
    total_episodes: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_duration_seconds: float
    task_types: List[str]
    last_execution: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.last_execution:
            data["last_execution"] = self.last_execution.isoformat()
        return data


class EpisodicMemory:
    """Episodic Memory - Cross-project experience accumulation.

    Features:
    - Records task execution episodes
    - Searches similar experiences
    - Provides project-level summaries
    - Persists to SQLite
    """

    def __init__(self, storage_path: str):
        """Initialize episodic memory.

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
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                task_type TEXT NOT NULL,
                task_description TEXT NOT NULL,
                steps TEXT NOT NULL,
                outcome TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                lessons TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_project
            ON episodes(project, timestamp)
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_type
            ON episodes(task_type, timestamp)
        """)

        self._conn.commit()

    async def record_episode(self, episode: Episode) -> None:
        """Record a task execution episode.

        Args:
            episode: Episode to record
        """
        import uuid

        if not episode.episode_id:
            episode.episode_id = str(uuid.uuid4())

        self._conn.execute(
            """
            INSERT OR REPLACE INTO episodes
            (episode_id, project, task_type, task_description, steps, outcome, duration_seconds, lessons, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode.episode_id,
                episode.project,
                episode.task_type,
                episode.task_description,
                json.dumps(episode.steps),
                episode.outcome,
                episode.duration_seconds,
                json.dumps(episode.lessons),
                episode.timestamp.isoformat()
            )
        )
        self._conn.commit()
        logger.info(f"[EpisodicMemory] Recorded episode {episode.episode_id} for project {episode.project}")

    async def search_similar(
        self,
        query: str,
        project: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Episode]:
        """Search for similar task experiences.

        Args:
            query: Search query
            project: Filter by project
            task_type: Filter by task type
            limit: Maximum results

        Returns:
            List of similar episodes
        """
        sql = """
            SELECT * FROM episodes
            WHERE task_description LIKE ? OR lessons LIKE ?
        """
        params = [f"%{query}%", f"%{query}%"]

        if project:
            sql += " AND project = ?"
            params.append(project)

        if task_type:
            sql += " AND task_type = ?"
            params.append(task_type)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        episodes = []
        for row in rows:
            episodes.append(Episode(
                episode_id=row["episode_id"],
                project=row["project"],
                task_type=row["task_type"],
                task_description=row["task_description"],
                steps=json.loads(row["steps"]),
                outcome=row["outcome"],
                duration_seconds=row["duration_seconds"],
                lessons=json.loads(row["lessons"]),
                timestamp=datetime.fromisoformat(row["timestamp"])
            ))

        return episodes

    async def get_project_summary(
        self,
        project: str
    ) -> ProjectSummary:
        """Get execution summary for a project.

        Args:
            project: Project name

        Returns:
            Project summary
        """
        cursor = self._conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(duration_seconds) as avg_duration,
                GROUP_CONCAT(DISTINCT task_type) as task_types,
                MAX(timestamp) as last_execution
            FROM episodes
            WHERE project = ?
            """,
            (project,)
        )

        row = cursor.fetchone()
        total = row["total"] or 0
        success = row["success"] or 0

        task_types_str = row["task_types"] or ""
        task_types = task_types_str.split(",") if task_types_str else []

        return ProjectSummary(
            project=project,
            total_episodes=total,
            success_count=success,
            failure_count=row["failed"] or 0,
            success_rate=success / total if total > 0 else 0.0,
            avg_duration_seconds=row["avg_duration"] or 0.0,
            task_types=task_types,
            last_execution=datetime.fromisoformat(row["last_execution"]) if row["last_execution"] else None
        )

    async def get_recent_episodes(
        self,
        project: Optional[str] = None,
        limit: int = 10
    ) -> List[Episode]:
        """Get recent episodes.

        Args:
            project: Filter by project
            limit: Maximum results

        Returns:
            Recent episodes
        """
        sql = "SELECT * FROM episodes"
        params = []

        if project:
            sql += " WHERE project = ?"
            params.append(project)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        episodes = []
        for row in rows:
            episodes.append(Episode(
                episode_id=row["episode_id"],
                project=row["project"],
                task_type=row["task_type"],
                task_description=row["task_description"],
                steps=json.loads(row["steps"]),
                outcome=row["outcome"],
                duration_seconds=row["duration_seconds"],
                lessons=json.loads(row["lessons"]),
                timestamp=datetime.fromisoformat(row["timestamp"])
            ))

        return episodes

    async def get_lessons_learned(
        self,
        project: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> List[str]:
        """Get all lessons learned.

        Args:
            project: Filter by project
            task_type: Filter by task type

        Returns:
            List of unique lessons
        """
        sql = "SELECT lessons FROM episodes WHERE outcome = 'success'"
        params = []

        if project:
            sql += " AND project = ?"
            params.append(project)

        if task_type:
            sql += " AND task_type = ?"
            params.append(task_type)

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        all_lessons = set()
        for row in rows:
            lessons = json.loads(row["lessons"])
            all_lessons.update(lessons)

        return list(all_lessons)

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


def create_episodic_memory(
    storage_dir: str = ".pyutagent"
) -> EpisodicMemory:
    """Create episodic memory instance.

    Args:
        storage_dir: Storage directory

    Returns:
        EpisodicMemory instance
    """
    import os
    db_path = os.path.join(storage_dir, "episodic_memory.db")
    return EpisodicMemory(db_path)
