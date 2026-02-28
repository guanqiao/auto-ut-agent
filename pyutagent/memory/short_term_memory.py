"""Short term memory for conversation history."""

import sqlite3
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


class ShortTermMemory:
    """Short term memory persists conversation history.
    
    This is stored in SQLite and persists across application restarts,
    but is scoped to a single session/conversation.
    """
    
    def __init__(self, db_path: str):
        """Initialize short term memory.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect and initialize
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        if self._conn is not None:
            return
        
        self._conn = sqlite3.connect(self.db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Create necessary tables."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create index for faster session queries
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session 
            ON conversations(session_id, timestamp)
        """)
        
        self._conn.commit()
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """Add a message to the conversation.
        
        Args:
            session_id: Unique session identifier
            role: Message role (user, agent, system, tool)
            content: Message content
            metadata: Optional metadata dict
        """
        import json
        
        self._conn.execute(
            """
            INSERT INTO conversations (session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                datetime.now().isoformat(),
                json.dumps(metadata) if metadata else None
            )
        )
        self._conn.commit()
    
    def get_conversation(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return (most recent)
            
        Returns:
            List of message dictionaries
        """
        import json
        
        query = """
            SELECT role, content, timestamp, metadata
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp
        """
        
        if limit:
            # Get all then slice to get most recent
            cursor = self._conn.execute(query, (session_id,))
            rows = cursor.fetchall()
            rows = rows[-limit:] if len(rows) > limit else rows
        else:
            cursor = self._conn.execute(query, (session_id,))
            rows = cursor.fetchall()
        
        return [
            {
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "metadata": json.loads(metadata) if metadata else None
            }
            for role, content, timestamp, metadata in rows
        ]
    
    def clear_conversation(self, session_id: str):
        """Clear all messages for a session.
        
        Args:
            session_id: Session identifier
        """
        self._conn.execute(
            "DELETE FROM conversations WHERE session_id = ?",
            (session_id,)
        )
        self._conn.commit()
    
    def get_all_sessions(self) -> List[str]:
        """Get all unique session IDs.
        
        Returns:
            List of session IDs
        """
        cursor = self._conn.execute(
            "SELECT DISTINCT session_id FROM conversations ORDER BY session_id"
        )
        return [row[0] for row in cursor.fetchall()]
    
    def delete_old_sessions(self, days: int = 30):
        """Delete sessions older than specified days.
        
        Args:
            days: Number of days to keep
        """
        from datetime import timedelta
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        self._conn.execute(
            "DELETE FROM conversations WHERE timestamp < ?",
            (cutoff,)
        )
        self._conn.commit()
    
    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
