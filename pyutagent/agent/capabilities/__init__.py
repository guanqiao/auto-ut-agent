"""Agent Capabilities System.

This module provides a modular capability system for the agent,
allowing features to be loaded on-demand based on configuration.
"""

from .base import Capability, CapabilityMetadata, CapabilityPriority
from .registry import CapabilityRegistry

__all__ = [
    "Capability",
    "CapabilityMetadata",
    "CapabilityPriority",
    "CapabilityRegistry",
]
