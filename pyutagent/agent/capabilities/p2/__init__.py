"""P2 Multi-Agent Capabilities.

This module provides multi-agent collaboration capabilities:
- Agent Coordination
- Message Bus
- Shared Knowledge
"""

from .multi_agent import MultiAgentCapability
from .knowledge_sharing import KnowledgeSharingCapability

__all__ = [
    "MultiAgentCapability",
    "KnowledgeSharingCapability",
]
