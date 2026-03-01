"""Application context for managing global state.

This module provides a centralized context for managing application-wide state,
replacing multiple global variables with a single managed context.

Features:
- Centralized state management
- Thread-safe access
- Easy testing with context injection
- Clear lifecycle management
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ApplicationContext:
    """Centralized application context for managing global state.
    
    This class encapsulates all global state in a single object,
    providing better testability and clearer lifecycle management.
    
    Attributes:
        settings: Application settings
        container: Dependency injection container
        file_cache: File content cache
        retry_manager: Retry manager instance
        _lock: Thread lock for thread-safe access
    """
    
    settings: Optional[Any] = None
    container: Optional[Any] = None
    file_cache: Optional[Any] = None
    retry_manager: Optional[Any] = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _initialized: bool = False
    
    def initialize(
        self,
        settings: Optional[Any] = None,
        container: Optional[Any] = None,
        file_cache: Optional[Any] = None,
        retry_manager: Optional[Any] = None
    ) -> None:
        """Initialize the context with components.
        
        Args:
            settings: Application settings
            container: Dependency injection container
            file_cache: File cache instance
            retry_manager: Retry manager instance
        """
        with self._lock:
            if self._initialized:
                logger.warning("[ApplicationContext] Already initialized, skipping")
                return
            
            if settings is not None:
                self.settings = settings
            if container is not None:
                self.container = container
            if file_cache is not None:
                self.file_cache = file_cache
            if retry_manager is not None:
                self.retry_manager = retry_manager
            
            self._initialized = True
            logger.info("[ApplicationContext] Context initialized")
    
    def is_initialized(self) -> bool:
        """Check if context is initialized."""
        return self._initialized
    
    def reset(self) -> None:
        """Reset the context to uninitialized state."""
        with self._lock:
            self.settings = None
            self.container = None
            self.file_cache = None
            self.retry_manager = None
            self._initialized = False
            logger.info("[ApplicationContext] Context reset")
    
    def get_settings(self) -> Any:
        """Get settings, initializing lazily if needed."""
        with self._lock:
            if self.settings is None:
                from .config import get_settings
                self.settings = get_settings()
                logger.debug("[ApplicationContext] Lazily initialized settings")
            return self.settings
    
    def get_container(self) -> Any:
        """Get container, initializing lazily if needed."""
        with self._lock:
            if self.container is None:
                from .container import get_container
                self.container = get_container()
                logger.debug("[ApplicationContext] Lazily initialized container")
            return self.container
    
    def get_file_cache(self) -> Any:
        """Get file cache, initializing lazily if needed."""
        with self._lock:
            if self.file_cache is None:
                from .cache import get_file_cache
                self.file_cache = get_file_cache()
                logger.debug("[ApplicationContext] Lazily initialized file cache")
            return self.file_cache
    
    def get_retry_manager(self) -> Any:
        """Get retry manager, initializing lazily if needed."""
        with self._lock:
            if self.retry_manager is None:
                from .retry_manager import get_retry_manager
                self.retry_manager = get_retry_manager()
                logger.debug("[ApplicationContext] Lazily initialized retry manager")
            return self.retry_manager
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        with self._lock:
            return {
                "initialized": self._initialized,
                "has_settings": self.settings is not None,
                "has_container": self.container is not None,
                "has_file_cache": self.file_cache is not None,
                "has_retry_manager": self.retry_manager is not None,
            }


_global_context: Optional[ApplicationContext] = None
_context_lock = threading.Lock()


def get_context() -> ApplicationContext:
    """Get the global application context.
    
    Returns:
        The global ApplicationContext instance
    """
    global _global_context
    if _global_context is None:
        with _context_lock:
            if _global_context is None:
                _global_context = ApplicationContext()
                logger.info("[ApplicationContext] Created global context")
    return _global_context


def reset_context() -> None:
    """Reset the global context."""
    global _global_context
    with _context_lock:
        if _global_context is not None:
            _global_context.reset()
        _global_context = None
        logger.info("[ApplicationContext] Reset global context")


def initialize_context(**kwargs) -> ApplicationContext:
    """Initialize the global context with components.
    
    Args:
        **kwargs: Components to initialize (settings, container, file_cache, retry_manager)
        
    Returns:
        The initialized ApplicationContext
    """
    context = get_context()
    context.initialize(**kwargs)
    return context
