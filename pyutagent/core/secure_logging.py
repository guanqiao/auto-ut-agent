"""Secure logging utilities for PyUT Agent.

This module provides secure logging capabilities that automatically
sanitize sensitive information like API keys, passwords, and tokens.
"""

import logging
import re
from typing import Any, Dict, Set, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SecureLogger:
    """Logger that automatically sanitizes sensitive information."""
    
    # Patterns that might indicate sensitive data
    SENSITIVE_PATTERNS = [
        r'api[_-]?key',
        r'api[_-]?secret',
        r'password',
        r'secret',
        r'token',
        r'auth',
        r'credential',
        r'private[_-]?key',
        r'access[_-]?token',
        r'refresh[_-]?token',
    ]
    
    # Compiled regex patterns
    _sensitive_keys: Set[str] = set()
    _key_patterns: list = []
    
    @classmethod
    def initialize(cls):
        """Initialize sensitive key patterns."""
        cls._key_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in cls.SENSITIVE_PATTERNS]
        cls._sensitive_keys = {
            'api_key', 'apikey', 'api-key',
            'api_secret', 'apisecret', 'api-secret',
            'password', 'passwd', 'pwd',
            'secret', 'secret_key', 'secretkey',
            'token', 'access_token', 'access_token',
            'refresh_token', 'refresh_token',
            'auth', 'authorization',
            'credential', 'credentials',
            'private_key', 'privatekey',
        }
    
    @classmethod
    def is_sensitive_key(cls, key: str) -> bool:
        """Check if a key indicates sensitive data.
        
        Args:
            key: Dictionary key or attribute name
            
        Returns:
            True if the key indicates sensitive data
        """
        if not cls._key_patterns:
            cls.initialize()
        
        key_lower = key.lower()
        
        # Check exact match
        if key_lower in cls._sensitive_keys:
            return True
        
        # Check pattern match
        for pattern in cls._key_patterns:
            if pattern.search(key_lower):
                return True
        
        return False
    
    @classmethod
    def sanitize(cls, obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
        """Sanitize sensitive information from an object.
        
        Args:
            obj: Object to sanitize
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            Sanitized object
        """
        if depth > max_depth:
            return "<max_depth_reached>"
        
        # Handle None
        if obj is None:
            return None
        
        # Handle strings
        if isinstance(obj, str):
            # Check if the string itself is a sensitive value (e.g., a token)
            if cls._looks_like_sensitive_value(obj):
                return "***"
            return obj
