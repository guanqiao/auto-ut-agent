"""Semantic Memory - Programming knowledge and concepts.

This module provides semantic memory for storing and retrieving
programming knowledge, design patterns, best practices, and technology information.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """A programming concept or knowledge item."""
    concept_id: str
    name: str
    category: str  # design_pattern, best_practice, technology, algorithm, etc.
    description: str
    examples: List[str]
    related_concepts: List[str]
    source: str  # Where this knowledge came from
    confidence: float  # Confidence in this knowledge (0.0-1.0)
    usage_count: int
    last_accessed: datetime
    created_at: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["last_accessed"] = self.last_accessed.isoformat()
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Concept":
        """Create from dictionary."""
        data = data.copy()
        data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class CodePattern:
    """A reusable code pattern."""
    pattern_id: str
    name: str
    language: str
    description: str
    code_template: str
    usage_context: str
    examples: List[str]
    success_count: int
    failure_count: int
    created_at: datetime
    last_used: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_used"] = self.last_used.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodePattern":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


class SemanticMemory:
    """Semantic Memory - Programming knowledge storage.

    Features:
    - Stores programming concepts and knowledge
    - Manages code patterns by language
    - Supports concept relationships
    - Retrieves relevant knowledge for tasks
    - Persists to SQLite
    """

    def __init__(self, storage_path: str):
        """Initialize semantic memory.

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
        # Concepts table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                concept_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                examples TEXT NOT NULL,
                related_concepts TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                usage_count INTEGER NOT NULL,
                last_accessed TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)

        # Code patterns table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS code_patterns (
                pattern_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                language TEXT NOT NULL,
                description TEXT NOT NULL,
                code_template TEXT NOT NULL,
                usage_context TEXT NOT NULL,
                examples TEXT NOT NULL,
                success_count INTEGER NOT NULL,
                failure_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_used TEXT NOT NULL
            )
        """)

        # Indexes
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_concepts_category
            ON concepts(category, usage_count DESC)
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_concepts_name
            ON concepts(name)
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_language
            ON code_patterns(language, success_count DESC)
        """)

        self._conn.commit()

    async def learn_concept(self, concept: Concept) -> None:
        """Learn a new concept.

        Args:
            concept: Concept to learn
        """
        import uuid

        if not concept.concept_id:
            concept.concept_id = str(uuid.uuid4())

        self._conn.execute(
            """
            INSERT OR REPLACE INTO concepts
            (concept_id, name, category, description, examples, related_concepts,
             source, confidence, usage_count, last_accessed, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                concept.concept_id,
                concept.name,
                concept.category,
                concept.description,
                json.dumps(concept.examples),
                json.dumps(concept.related_concepts),
                concept.source,
                concept.confidence,
                concept.usage_count,
                concept.last_accessed.isoformat(),
                concept.created_at.isoformat(),
                json.dumps(concept.metadata)
            )
        )
        self._conn.commit()
        logger.info(f"[SemanticMemory] Learned concept: {concept.name} ({concept.category})")

    async def query_concepts(
        self,
        query: str,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[Concept]:
        """Query concepts by keyword.

        Args:
            query: Search query
            category: Filter by category
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of matching concepts
        """
        sql = """
            SELECT * FROM concepts
            WHERE (name LIKE ? OR description LIKE ?)
            AND confidence >= ?
        """
        params = [f"%{query}%", f"%{query}%", min_confidence]

        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY usage_count DESC, confidence DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_concept(row) for row in rows]

    async def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get a concept by exact name match.

        Args:
            name: Concept name

        Returns:
            Concept or None
        """
        cursor = self._conn.execute(
            "SELECT * FROM concepts WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()

        if row:
            return self._row_to_concept(row)
        return None

    async def get_concepts_by_category(
        self,
        category: str,
        limit: int = 20
    ) -> List[Concept]:
        """Get all concepts in a category.

        Args:
            category: Concept category
            limit: Maximum results

        Returns:
            List of concepts
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM concepts
            WHERE category = ?
            ORDER BY usage_count DESC
            LIMIT ?
            """,
            (category, limit)
        )
        rows = cursor.fetchall()

        return [self._row_to_concept(row) for row in rows]

    async def record_concept_usage(self, concept_id: str) -> None:
        """Record that a concept was used.

        Args:
            concept_id: Concept ID
        """
        self._conn.execute(
            """
            UPDATE concepts
            SET usage_count = usage_count + 1, last_accessed = ?
            WHERE concept_id = ?
            """,
            (datetime.now().isoformat(), concept_id)
        )
        self._conn.commit()

    async def add_code_pattern(self, pattern: CodePattern) -> None:
        """Add a code pattern.

        Args:
            pattern: Code pattern to add
        """
        import uuid

        if not pattern.pattern_id:
            pattern.pattern_id = str(uuid.uuid4())

        self._conn.execute(
            """
            INSERT OR REPLACE INTO code_patterns
            (pattern_id, name, language, description, code_template, usage_context,
             examples, success_count, failure_count, created_at, last_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern.pattern_id,
                pattern.name,
                pattern.language,
                pattern.description,
                pattern.code_template,
                pattern.usage_context,
                json.dumps(pattern.examples),
                pattern.success_count,
                pattern.failure_count,
                pattern.created_at.isoformat(),
                pattern.last_used.isoformat()
            )
        )
        self._conn.commit()
        logger.info(f"[SemanticMemory] Added code pattern: {pattern.name} ({pattern.language})")

    async def get_code_patterns(
        self,
        language: str,
        context: Optional[str] = None,
        limit: int = 10
    ) -> List[CodePattern]:
        """Get code patterns for a language.

        Args:
            language: Programming language
            context: Optional context filter
            limit: Maximum results

        Returns:
            List of code patterns
        """
        sql = "SELECT * FROM code_patterns WHERE language = ?"
        params = [language]

        if context:
            sql += " AND (usage_context LIKE ? OR description LIKE ?)"
            params.extend([f"%{context}%", f"%{context}%"])

        sql += " ORDER BY success_count DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_pattern(row) for row in rows]

    async def record_pattern_result(
        self,
        pattern_id: str,
        success: bool
    ) -> None:
        """Record success/failure of a pattern usage.

        Args:
            pattern_id: Pattern ID
            success: Whether usage was successful
        """
        column = "success_count" if success else "failure_count"
        self._conn.execute(
            f"""
            UPDATE code_patterns
            SET {column} = {column} + 1, last_used = ?
            WHERE pattern_id = ?
            """,
            (datetime.now().isoformat(), pattern_id)
        )
        self._conn.commit()

    async def get_related_concepts(self, concept_id: str) -> List[Concept]:
        """Get concepts related to a given concept.

        Args:
            concept_id: Concept ID

        Returns:
            List of related concepts
        """
        # Get the concept first
        cursor = self._conn.execute(
            "SELECT related_concepts FROM concepts WHERE concept_id = ?",
            (concept_id,)
        )
        row = cursor.fetchone()

        if not row:
            return []

        related_names = json.loads(row["related_concepts"])
        if not related_names:
            return []

        # Query related concepts
        placeholders = ",".join(["?"] * len(related_names))
        cursor = self._conn.execute(
            f"SELECT * FROM concepts WHERE name IN ({placeholders})",
            related_names
        )
        rows = cursor.fetchall()

        return [self._row_to_concept(row) for row in rows]

    async def get_popular_concepts(
        self,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Concept]:
        """Get most frequently used concepts.

        Args:
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of popular concepts
        """
        sql = "SELECT * FROM concepts WHERE 1=1"
        params = []

        if category:
            sql += " AND category = ?"
            params.append(category)

        sql += " ORDER BY usage_count DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        return [self._row_to_concept(row) for row in rows]

    def _row_to_concept(self, row: sqlite3.Row) -> Concept:
        """Convert database row to Concept."""
        return Concept(
            concept_id=row["concept_id"],
            name=row["name"],
            category=row["category"],
            description=row["description"],
            examples=json.loads(row["examples"]),
            related_concepts=json.loads(row["related_concepts"]),
            source=row["source"],
            confidence=row["confidence"],
            usage_count=row["usage_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"])
        )

    def _row_to_pattern(self, row: sqlite3.Row) -> CodePattern:
        """Convert database row to CodePattern."""
        return CodePattern(
            pattern_id=row["pattern_id"],
            name=row["name"],
            language=row["language"],
            description=row["description"],
            code_template=row["code_template"],
            usage_context=row["usage_context"],
            examples=json.loads(row["examples"]),
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_used=datetime.fromisoformat(row["last_used"])
        )

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


def create_semantic_memory(
    storage_dir: str = ".pyutagent"
) -> SemanticMemory:
    """Create semantic memory instance.

    Args:
        storage_dir: Storage directory

    Returns:
        SemanticMemory instance
    """
    import os
    db_path = os.path.join(storage_dir, "semantic_memory.db")
    return SemanticMemory(db_path)
