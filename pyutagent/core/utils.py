"""Common Utilities.

This module provides common utility functions:
- Async helpers
- File operations
- String operations
- Data validation
"""

import asyncio
import hashlib
import json
import re
import uuid
import warnings
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID.

    Args:
        prefix: ID prefix

    Returns:
        Unique ID
    """
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}{uid}" if prefix else uid


def generate_hash(data: Union[str, bytes]) -> str:
    """Generate hash from data.

    Args:
        data: Data to hash

    Returns:
        Hash string
    """
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix for truncated text

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(filename: str) -> str:
    """Sanitize filename.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def parse_size(size_str: str) -> int:
    """Parse size string to bytes.

    Args:
        size_str: Size string (e.g., "1KB", "2MB")

    Returns:
        Size in bytes
    """
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4
    }

    match = re.match(r'^(\d+(?:\.\d+)?)\s*([A-Z]+)$', size_str.upper())
    if not match:
        return 0

    value, unit = match.groups()
    return int(float(value) * units.get(unit, 1))


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def format_duration(seconds: float) -> str:
    """Format duration to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    return f"{hours}h {minutes}m"


def parse_json_safe(text: str, default: Any = None) -> Any:
    """Safely parse JSON text.

    Args:
        text: JSON text
        default: Default value on error

    Returns:
        Parsed JSON or default
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def to_json(data: Any, indent: int = 2, **kwargs) -> str:
    """Convert data to JSON string.

    Args:
        data: Data to convert
        indent: Indentation
        **kwargs: Additional JSON arguments

    Returns:
        JSON string
    """
    return json.dumps(data, indent=indent, default=str, **kwargs)


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries.

    Args:
        *dicts: Dictionaries to merge

    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def deep_get(d: Dict, key_path: str, default: Any = None) -> Any:
    """Get value from nested dict using dot notation.

    Args:
        d: Dictionary
        key_path: Key path (e.g., "a.b.c")
        default: Default value

    Returns:
        Value or default
    """
    keys = key_path.split(".")
    value = d

    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default

        if value is None:
            return default

    return value


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split list into chunks.

    Args:
        items: List to chunk
        chunk_size: Chunk size

    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def deduplicate(items: List[T], key: Optional[Callable] = None) -> List[T]:
    """Deduplicate list while preserving order.

    Args:
        items: List to deduplicate
        key: Key function for comparison

    Returns:
        Deduplicated list
    """
    seen = set()
    result = []

    for item in items:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            result.append(item)

    return result


def flatten(nested: List[Any]) -> List[Any]:
    """Flatten nested list.

    Args:
        nested: Nested list

    Returns:
        Flattened list
    """
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def retry_async(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for async retry.

    .. deprecated::
        Use :func:`pyutagent.core.retry_manager.with_retry` instead.

    Args:
        max_attempts: Maximum attempts
        delay: Initial delay
        backoff: Backoff multiplier
        exceptions: Exceptions to catch

    Returns:
        Decorated function
    """
    warnings.warn(
        "retry_async is deprecated. Use pyutagent.core.retry_manager.with_retry instead.",
        DeprecationWarning,
        stacklevel=2
    )

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def retry_sync(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for sync retry.

    .. deprecated::
        Use :func:`pyutagent.core.retry_manager.with_retry` instead.

    Args:
        max_attempts: Maximum attempts
        delay: Initial delay
        backoff: Backoff multiplier
        exceptions: Exceptions to catch

    Returns:
        Decorated function
    """
    warnings.warn(
        "retry_sync is deprecated. Use pyutagent.core.retry_manager.with_retry instead.",
        DeprecationWarning,
        stacklevel=2
    )

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def timing_async(func: Callable) -> Callable:
    """Decorator to measure async function execution time.

    Args:
        func: Function to measure

    Returns:
        Decorated function
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = datetime.now()
        result = await func(*args, **kwargs)
        duration = (datetime.now() - start).total_seconds()
        return result, duration
    return wrapper


def timing_sync(func: Callable) -> Callable:
    """Decorator to measure sync function execution time.

    Args:
        func: Function to measure

    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.now() - start).total_seconds()
        return result, duration
    return wrapper


class RateLimiter:
    """Rate limiter for async operations."""

    def __init__(self, max_calls: int, period: float):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum calls per period
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self._calls: List[datetime] = []

    async def acquire(self):
        """Acquire permission to make a call."""
        now = datetime.now()

        self._calls = [
            call for call in self._calls
            if now - call < timedelta(seconds=self.period)
        ]

        if len(self._calls) >= self.max_calls:
            wait_time = self.period - (now - self._calls[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self._calls.append(now)


class Cache:
    """Simple in-memory cache."""

    def __init__(self, ttl: int = 300):
        """Initialize cache.

        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self._cache: Dict[str, tuple] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = (value, datetime.now())

    def delete(self, key: str):
        """Delete value from cache.

        Args:
            key: Cache key
        """
        self._cache.pop(key, None)

    def clear(self):
        """Clear all cache."""
        self._cache.clear()


class Timer:
    """Context manager for timing."""

    def __init__(self):
        """Initialize timer."""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def __enter__(self):
        """Start timer."""
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        """Stop timer."""
        self.end_time = datetime.now()

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
