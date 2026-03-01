"""Shared knowledge base for multi-agent collaboration.

Provides knowledge sharing and experience replay mechanisms.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    """A piece of knowledge in the shared knowledge base."""
    item_id: str
    item_type: str  # e.g., "test_pattern", "error_solution", "best_practice"
    content: Dict[str, Any]
    source_agent: str
    confidence: float  # 0.0 - 1.0
    usage_count: int = 0
    success_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    related_items: List[str] = field(default_factory=list)  # IDs of related items
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    @property
    def effectiveness_score(self) -> float:
        """Calculate overall effectiveness score."""
        # Combine confidence, success rate, and usage
        return (self.confidence * 0.4 + 
                self.success_rate * 0.4 + 
                min(self.usage_count / 100, 1.0) * 0.2)


@dataclass
class Experience:
    """An experience for replay learning."""
    experience_id: str
    task_type: str
    context: Dict[str, Any]
    action: str
    outcome: str
    reward: float  # -1.0 to 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_id: Optional[str] = None
    episode_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create from dictionary."""
        return cls(**data)


class SharedKnowledgeBase:
    """Shared knowledge base for multi-agent systems.
    
    Features:
    - Persistent storage of knowledge items
    - Tag-based organization
    - Confidence tracking
    - Usage statistics
    - Similarity search
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize shared knowledge base.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "shared_knowledge.db"
        
        self.db_path = str(db_path)
        self._init_database()
        self._cache: Dict[str, KnowledgeItem] = {}
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info(f"[SharedKnowledgeBase] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Knowledge items table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    item_id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_agent TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_used TEXT,
                    tags TEXT,
                    related_items TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_type ON knowledge_items(item_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent ON knowledge_items(source_agent)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_confidence ON knowledge_items(confidence)
            ''')
            
            conn.commit()
    
    def add_knowledge(
        self,
        item_type: str,
        content: Dict[str, Any],
        source_agent: str,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None,
        related_items: Optional[List[str]] = None
    ) -> str:
        """Add knowledge to the base.
        
        Args:
            item_type: Type of knowledge
            content: Knowledge content
            source_agent: Agent that created this knowledge
            confidence: Confidence level (0.0 - 1.0)
            tags: Optional tags for categorization
            related_items: IDs of related knowledge items
            
        Returns:
            Item ID
        """
        import uuid
        item_id = str(uuid.uuid4())
        
        item = KnowledgeItem(
            item_id=item_id,
            item_type=item_type,
            content=content,
            source_agent=source_agent,
            confidence=confidence,
            tags=tags or [],
            related_items=related_items or []
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO knowledge_items 
                (item_id, item_type, content, source_agent, confidence, tags, related_items)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                item_id,
                item_type,
                json.dumps(content),
                source_agent,
                confidence,
                json.dumps(tags or []),
                json.dumps(related_items or [])
            ))
            conn.commit()
        
        # Update cache and index
        self._cache[item_id] = item
        for tag in (tags or []):
            self._tag_index[tag].add(item_id)
        
        logger.debug(f"[SharedKnowledgeBase] Added knowledge: {item_id[:8]} ({item_type})")
        return item_id
    
    def get_knowledge(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get knowledge item by ID.
        
        Args:
            item_id: Knowledge item ID
            
        Returns:
            Knowledge item or None
        """
        # Check cache first
        if item_id in self._cache:
            return self._cache[item_id]
        
        # Load from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM knowledge_items WHERE item_id = ?',
                (item_id,)
            )
            row = cursor.fetchone()
            
            if row:
                item = self._row_to_item(row)
                self._cache[item_id] = item
                return item
        
        return None
    
    def query_knowledge(
        self,
        item_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_agent: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[KnowledgeItem]:
        """Query knowledge base with filters.
        
        Args:
            item_type: Filter by type
            tags: Filter by tags (all must match)
            source_agent: Filter by source agent
            min_confidence: Minimum confidence
            limit: Maximum results
            
        Returns:
            List of matching knowledge items
        """
        query = 'SELECT * FROM knowledge_items WHERE 1=1'
        params = []
        
        if item_type:
            query += ' AND item_type = ?'
            params.append(item_type)
        
        if source_agent:
            query += ' AND source_agent = ?'
            params.append(source_agent)
        
        if min_confidence > 0:
            query += ' AND confidence >= ?'
            params.append(min_confidence)
        
        query += ' ORDER BY confidence DESC, usage_count DESC LIMIT ?'
        params.append(limit)
        
        items = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                item = self._row_to_item(row)
                
                # Filter by tags if specified
                if tags:
                    if not all(tag in item.tags for tag in tags):
                        continue
                
                items.append(item)
        
        return items[:limit]
    
    def find_similar(
        self,
        content_query: Dict[str, Any],
        item_type: Optional[str] = None,
        limit: int = 5
    ) -> List[KnowledgeItem]:
        """Find knowledge items similar to query.
        
        Uses simple keyword matching. For production, consider using embeddings.
        
        Args:
            content_query: Query content
            item_type: Filter by type
            limit: Maximum results
            
        Returns:
            List of similar items
        """
        # Build query string from content
        query_terms = []
        for key, value in content_query.items():
            if isinstance(value, str):
                query_terms.extend(value.lower().split())
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        query_terms.append(v.lower())
        
        query_str = ' '.join(query_terms)
        
        # Get candidates
        candidates = self.query_knowledge(item_type=item_type, limit=100)
        
        # Score by similarity
        scored_items = []
        for item in candidates:
            score = self._calculate_similarity(query_str, item)
            scored_items.append((score, item))
        
        # Return top matches
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:limit]]
    
    def _calculate_similarity(self, query: str, item: KnowledgeItem) -> float:
        """Calculate similarity score between query and item.
        
        Args:
            query: Query string
            item: Knowledge item
            
        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Simple keyword overlap scoring
        query_words = set(query.lower().split())
        
        # Extract words from item content
        item_words = set()
        for key, value in item.content.items():
            if isinstance(value, str):
                item_words.update(value.lower().split())
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str):
                        item_words.add(v.lower())
        
        # Add tag words
        item_words.update(tag.lower() for tag in item.tags)
        
        if not query_words or not item_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words & item_words
        union = query_words | item_words
        
        return len(intersection) / len(union) if union else 0.0
    
    def record_usage(self, item_id: str, success: bool):
        """Record usage of a knowledge item.
        
        Args:
            item_id: Knowledge item ID
            success: Whether the usage was successful
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE knowledge_items 
                SET usage_count = usage_count + 1,
                    success_count = success_count + ?,
                    last_used = CURRENT_TIMESTAMP
                WHERE item_id = ?
            ''', (1 if success else 0, item_id))
            conn.commit()
        
        # Update cache
        if item_id in self._cache:
            self._cache[item_id].usage_count += 1
            if success:
                self._cache[item_id].success_count += 1
            self._cache[item_id].last_used = datetime.now().isoformat()
        
        logger.debug(f"[SharedKnowledgeBase] Recorded usage: {item_id[:8]} (success={success})")
    
    def update_confidence(self, item_id: str, new_confidence: float):
        """Update confidence of a knowledge item.
        
        Args:
            item_id: Knowledge item ID
            new_confidence: New confidence value
        """
        new_confidence = max(0.0, min(1.0, new_confidence))
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE knowledge_items SET confidence = ? WHERE item_id = ?',
                (new_confidence, item_id)
            )
            conn.commit()
        
        if item_id in self._cache:
            self._cache[item_id].confidence = new_confidence
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics.
        
        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total items
            cursor.execute('SELECT COUNT(*) FROM knowledge_items')
            total_items = cursor.fetchone()[0]
            
            # Items by type
            cursor.execute('SELECT item_type, COUNT(*) FROM knowledge_items GROUP BY item_type')
            type_counts = dict(cursor.fetchall())
            
            # Items by agent
            cursor.execute('SELECT source_agent, COUNT(*) FROM knowledge_items GROUP BY source_agent')
            agent_counts = dict(cursor.fetchall())
            
            # Average confidence
            cursor.execute('SELECT AVG(confidence) FROM knowledge_items')
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            return {
                "total_items": total_items,
                "type_distribution": type_counts,
                "agent_distribution": agent_counts,
                "average_confidence": avg_confidence,
                "cached_items": len(self._cache)
            }
    
    def _row_to_item(self, row) -> KnowledgeItem:
        """Convert database row to KnowledgeItem."""
        return KnowledgeItem(
            item_id=row[0],
            item_type=row[1],
            content=json.loads(row[2]),
            source_agent=row[3],
            confidence=row[4],
            usage_count=row[5],
            success_count=row[6],
            created_at=row[7],
            last_used=row[8],
            tags=json.loads(row[9]) if row[9] else [],
            related_items=json.loads(row[10]) if row[10] else []
        )


class ExperienceReplay:
    """Experience replay buffer for multi-agent learning.
    
    Stores and samples experiences for learning from past actions.
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        db_path: Optional[str] = None
    ):
        """Initialize experience replay.
        
        Args:
            capacity: Maximum number of experiences to store
            db_path: Optional path to SQLite database
        """
        self.capacity = capacity
        
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "experience_replay.db"
        
        self.db_path = str(db_path)
        self._init_database()
        
        logger.info(f"[ExperienceReplay] Initialized with capacity={capacity}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    context TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reward REAL NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    agent_id TEXT,
                    episode_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_task_type ON experiences(task_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_agent ON experiences(agent_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_episode ON experiences(episode_id)
            ''')
            
            conn.commit()
    
    def add_experience(
        self,
        task_type: str,
        context: Dict[str, Any],
        action: str,
        outcome: str,
        reward: float,
        agent_id: Optional[str] = None,
        episode_id: Optional[str] = None
    ) -> str:
        """Add an experience to the buffer.
        
        Args:
            task_type: Type of task
            context: Task context
            action: Action taken
            outcome: Outcome of action
            reward: Reward value (-1.0 to 1.0)
            agent_id: Optional agent ID
            episode_id: Optional episode ID
            
        Returns:
            Experience ID
        """
        import uuid
        experience_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check capacity and remove oldest if needed
            cursor.execute('SELECT COUNT(*) FROM experiences')
            count = cursor.fetchone()[0]
            
            if count >= self.capacity:
                cursor.execute('''
                    DELETE FROM experiences 
                    WHERE experience_id IN (
                        SELECT experience_id FROM experiences 
                        ORDER BY timestamp ASC LIMIT 1
                    )
                ''')
            
            cursor.execute('''
                INSERT INTO experiences 
                (experience_id, task_type, context, action, outcome, reward, agent_id, episode_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experience_id,
                task_type,
                json.dumps(context),
                action,
                outcome,
                reward,
                agent_id,
                episode_id
            ))
            
            conn.commit()
        
        logger.debug(f"[ExperienceReplay] Added experience: {experience_id[:8]}")
        return experience_id
    
    def sample(
        self,
        batch_size: int = 32,
        task_type: Optional[str] = None,
        min_reward: Optional[float] = None
    ) -> List[Experience]:
        """Sample experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            task_type: Filter by task type
            min_reward: Minimum reward threshold
            
        Returns:
            List of sampled experiences
        """
        query = 'SELECT * FROM experiences WHERE 1=1'
        params = []
        
        if task_type:
            query += ' AND task_type = ?'
            params.append(task_type)
        
        if min_reward is not None:
            query += ' AND reward >= ?'
            params.append(min_reward)
        
        query += ' ORDER BY RANDOM() LIMIT ?'
        params.append(batch_size)
        
        experiences = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                experiences.append(self._row_to_experience(row))
        
        return experiences
    
    def get_episode_experiences(self, episode_id: str) -> List[Experience]:
        """Get all experiences from an episode.
        
        Args:
            episode_id: Episode ID
            
        Returns:
            List of experiences in chronological order
        """
        experiences = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM experiences WHERE episode_id = ? ORDER BY timestamp',
                (episode_id,)
            )
            
            for row in cursor.fetchall():
                experiences.append(self._row_to_experience(row))
        
        return experiences
    
    def get_successful_experiences(
        self,
        task_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Experience]:
        """Get most successful experiences.
        
        Args:
            task_type: Filter by task type
            limit: Maximum results
            
        Returns:
            List of successful experiences
        """
        query = 'SELECT * FROM experiences WHERE reward > 0'
        params = []
        
        if task_type:
            query += ' AND task_type = ?'
            params.append(task_type)
        
        query += ' ORDER BY reward DESC LIMIT ?'
        params.append(limit)
        
        experiences = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                experiences.append(self._row_to_experience(row))
        
        return experiences
    
    def get_stats(self) -> Dict[str, Any]:
        """Get replay buffer statistics.
        
        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM experiences')
            total = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(reward) FROM experiences')
            avg_reward = cursor.fetchone()[0] or 0.0
            
            cursor.execute('SELECT task_type, COUNT(*) FROM experiences GROUP BY task_type')
            task_counts = dict(cursor.fetchall())
            
            cursor.execute('SELECT agent_id, COUNT(*) FROM experiences GROUP BY agent_id')
            agent_counts = dict(cursor.fetchall())
            
            return {
                "total_experiences": total,
                "capacity": self.capacity,
                "average_reward": avg_reward,
                "task_distribution": task_counts,
                "agent_distribution": agent_counts
            }
    
    def _row_to_experience(self, row) -> Experience:
        """Convert database row to Experience."""
        return Experience(
            experience_id=row[0],
            task_type=row[1],
            context=json.loads(row[2]),
            action=row[3],
            outcome=row[4],
            reward=row[5],
            timestamp=row[6],
            agent_id=row[7],
            episode_id=row[8]
        )
