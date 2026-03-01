"""Caching utilities for PyUT Agent.

This module provides caching mechanisms to improve performance
by avoiding redundant computations and I/O operations.
"""

import hashlib
import logging
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FileCache:
    """Cache for file contents with modification time checking."""
    
    def __init__(self, max_size: int = 100):
        """Initialize file cache.
        
        Args:
            max_size: Maximum number of files to cache
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._access_count: Dict[str, int] = {}
    
    def get(self, file_path: str) -> Optional[str]:
        """Get cached file content if valid.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cached content or None if not cached or