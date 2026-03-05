"""Session manager for managing chat sessions."""

import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


@dataclass
class SessionMessage:
    """Represents a message in a session."""
    id: str
    role: str  # 'user', 'agent', 'system'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, role: str, content: str, metadata: Optional[Dict] = None) -> 'SessionMessage':
        """Create a new message."""
        return cls(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )


@dataclass
class SessionContext:
    """Context information for a session."""
    files: List[str] = field(default_factory=list)
    folders: List[str] = field(default_factory=list)
    snippets: List[Dict[str, str]] = field(default_factory=list)
    project_path: Optional[str] = None


@dataclass
class ChatSession:
    """Represents a chat session."""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[SessionMessage] = field(default_factory=list)
    context: SessionContext = field(default_factory=SessionContext)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_favorite: bool = False
    
    @classmethod
    def create(cls, title: str = "New Session") -> 'ChatSession':
        """Create a new session."""
        now = datetime.now().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            title=title,
            created_at=now,
            updated_at=now
        )
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the session."""
        message = SessionMessage.create(role, content, metadata)
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
        
    def update_title(self, title: str):
        """Update session title."""
        self.title = title
        self.updated_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'messages': [asdict(m) for m in self.messages],
            'context': asdict(self.context),
            'metadata': self.metadata,
            'is_favorite': self.is_favorite
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create session from dictionary."""
        session = cls(
            id=data['id'],
            title=data['title'],
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            metadata=data.get('metadata', {}),
            is_favorite=data.get('is_favorite', False)
        )
        
        # Load messages
        for msg_data in data.get('messages', []):
            session.messages.append(SessionMessage(**msg_data))
        
        # Load context
        context_data = data.get('context', {})
        session.context = SessionContext(
            files=context_data.get('files', []),
            folders=context_data.get('folders', []),
            snippets=context_data.get('snippets', []),
            project_path=context_data.get('project_path')
        )
        
        return session


class SessionManager(QObject):
    """Manages chat sessions.
    
    Features:
    - Create, load, save sessions
    - Session persistence
    - Session search
    - Export/Import
    """
    
    # Signals
    session_created = pyqtSignal(str)  # session_id
    session_loaded = pyqtSignal(str)  # session_id
    session_saved = pyqtSignal(str)  # session_id
    session_deleted = pyqtSignal(str)  # session_id
    session_list_changed = pyqtSignal()
    
    def __init__(self, storage_dir: Optional[str] = None):
        super().__init__()
        
        self._sessions: Dict[str, ChatSession] = {}
        self._current_session: Optional[ChatSession] = None
        
        # Set storage directory
        if storage_dir:
            self._storage_dir = Path(storage_dir)
        else:
            # Default: ~/.pyutagent/sessions
            self._storage_dir = Path.home() / '.pyutagent' / 'sessions'
        
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing sessions
        self._load_all_sessions()
        
    def _load_all_sessions(self):
        """Load all sessions from storage."""
        try:
            for session_file in self._storage_dir.glob('*.json'):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        session = ChatSession.from_dict(data)
                        self._sessions[session.id] = session
                        logger.debug(f"Loaded session: {session.id}")
                except Exception as e:
                    logger.warning(f"Failed to load session {session_file}: {e}")
            
            logger.info(f"Loaded {len(self._sessions)} sessions")
            
        except Exception as e:
            logger.exception("Failed to load sessions")
    
    def create_session(self, title: str = "New Session") -> ChatSession:
        """Create a new session.
        
        Args:
            title: Session title
            
        Returns:
            The created session
        """
        session = ChatSession.create(title)
        self._sessions[session.id] = session
        self._current_session = session
        
        self.session_created.emit(session.id)
        self.session_list_changed.emit()
        
        logger.info(f"Created session: {session.id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            The session or None
        """
        return self._sessions.get(session_id)
    
    def get_current_session(self) -> Optional[ChatSession]:
        """Get the current session."""
        return self._current_session
    
    def set_current_session(self, session_id: str) -> bool:
        """Set the current session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        session = self._sessions.get(session_id)
        if session:
            self._current_session = session
            self.session_loaded.emit(session_id)
            logger.debug(f"Switched to session: {session_id}")
            return True
        return False
    
    def save_session(self, session_id: Optional[str] = None) -> bool:
        """Save a session to storage.
        
        Args:
            session_id: Session ID (defaults to current)
            
        Returns:
            True if successful
        """
        session = self._sessions.get(session_id) if session_id else self._current_session
        if not session:
            return False
        
        try:
            session_file = self._storage_dir / f"{session.id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            session.updated_at = datetime.now().isoformat()
            self.session_saved.emit(session.id)
            
            logger.debug(f"Saved session: {session.id}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to save session: {session.id}")
            return False
    
    def save_all_sessions(self):
        """Save all sessions."""
        for session_id in self._sessions:
            self.save_session(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful
        """
        if session_id not in self._sessions:
            return False
        
        try:
            # Remove from memory
            del self._sessions[session_id]
            
            # Remove file
            session_file = self._storage_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            # Update current session if needed
            if self._current_session and self._current_session.id == session_id:
                self._current_session = None
            
            self.session_deleted.emit(session_id)
            self.session_list_changed.emit()
            
            logger.info(f"Deleted session: {session_id}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to delete session: {session_id}")
            return False
    
    def get_all_sessions(self) -> List[ChatSession]:
        """Get all sessions sorted by update time."""
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions
    
    def get_recent_sessions(self, count: int = 10) -> List[ChatSession]:
        """Get recent sessions.
        
        Args:
            count: Number of sessions to return
            
        Returns:
            List of sessions
        """
        sessions = self.get_all_sessions()
        return sessions[:count]
    
    def search_sessions(self, query: str) -> List[ChatSession]:
        """Search sessions by title or content.
        
        Args:
            query: Search query
            
        Returns:
            List of matching sessions
        """
        query_lower = query.lower()
        results = []
        
        for session in self._sessions.values():
            # Search in title
            if query_lower in session.title.lower():
                results.append(session)
                continue
            
            # Search in messages
            for message in session.messages:
                if query_lower in message.content.lower():
                    results.append(session)
                    break
        
        # Sort by update time
        results.sort(key=lambda s: s.updated_at, reverse=True)
        return results
    
    def get_favorite_sessions(self) -> List[ChatSession]:
        """Get favorite sessions."""
        return [s for s in self._sessions.values() if s.is_favorite]
    
    def toggle_favorite(self, session_id: str) -> bool:
        """Toggle favorite status.
        
        Args:
            session_id: Session ID
            
        Returns:
            New favorite status
        """
        session = self._sessions.get(session_id)
        if session:
            session.is_favorite = not session.is_favorite
            self.save_session(session_id)
            self.session_list_changed.emit()
            return session.is_favorite
        return False
    
    def duplicate_session(self, session_id: str, new_title: Optional[str] = None) -> Optional[ChatSession]:
        """Duplicate a session.
        
        Args:
            session_id: Session ID to duplicate
            new_title: Optional new title
            
        Returns:
            The new session
        """
        original = self._sessions.get(session_id)
        if not original:
            return None
        
        # Create new session
        new_session = ChatSession.create(
            title=new_title or f"{original.title} (Copy)"
        )
        
        # Copy messages
        for msg in original.messages:
            new_session.messages.append(SessionMessage(
                id=str(uuid.uuid4()),
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                metadata=msg.metadata.copy()
            ))
        
        # Copy context
        new_session.context = SessionContext(
            files=original.context.files.copy(),
            folders=original.context.folders.copy(),
            snippets=original.context.snippets.copy(),
            project_path=original.context.project_path
        )
        
        self._sessions[new_session.id] = new_session
        self.save_session(new_session.id)
        self.session_list_changed.emit()
        
        logger.info(f"Duplicated session: {session_id} -> {new_session.id}")
        return new_session
    
    def export_session(self, session_id: str, file_path: str) -> bool:
        """Export a session to a file.
        
        Args:
            session_id: Session ID
            file_path: Export file path
            
        Returns:
            True if successful
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported session {session_id} to {file_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to export session: {session_id}")
            return False
    
    def import_session(self, file_path: str) -> Optional[ChatSession]:
        """Import a session from a file.
        
        Args:
            file_path: Import file path
            
        Returns:
            The imported session
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = ChatSession.from_dict(data)
            # Generate new ID to avoid conflicts
            session.id = str(uuid.uuid4())
            session.created_at = datetime.now().isoformat()
            session.updated_at = session.created_at
            
            self._sessions[session.id] = session
            self.save_session(session.id)
            self.session_list_changed.emit()
            
            logger.info(f"Imported session from {file_path}: {session.id}")
            return session
            
        except Exception as e:
            logger.exception(f"Failed to import session from {file_path}")
            return None
    
    def add_message_to_current(self, role: str, content: str, 
                               metadata: Optional[Dict] = None) -> bool:
        """Add a message to the current session.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        if not self._current_session:
            return False
        
        self._current_session.add_message(role, content, metadata)
        self.save_session(self._current_session.id)
        return True
    
    def clear_current_session(self):
        """Clear messages from current session."""
        if self._current_session:
            self._current_session.messages.clear()
            self.save_session(self._current_session.id)
    
    def get_session_count(self) -> int:
        """Get total number of sessions."""
        return len(self._sessions)
