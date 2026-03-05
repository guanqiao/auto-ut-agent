"""Capability Registry for Specialized SubAgents.

This module provides a centralized capability management system:
- Capability registration and discovery
- Capability provider matching
- Capability scoring and ranking

This is part of Phase 3 Week 19-20: Specialized SubAgent Enhancement.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from pyutagent.agent.subagent_base import SubAgent

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of agent capabilities."""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    TEST_GENERATION = "test_generation"
    TEST_EXECUTION = "test_execution"
    TEST_FIXING = "test_fixing"
    CODE_REFACTORING = "code_refactoring"
    BUG_FIXING = "bug_fixing"
    DOCUMENTATION = "documentation"
    BUILD = "build"
    DEPLOYMENT = "deployment"
    SEARCH = "search"
    PLANNING = "planning"
    COORDINATION = "coordination"
    REVIEW = "review"
    OPTIMIZATION = "optimization"


@dataclass
class CapabilityScore:
    """Score for a capability."""
    base_score: float = 0.0
    success_rate: float = 0.0
    speed_factor: float = 1.0
    quality_factor: float = 1.0
    
    @property
    def total_score(self) -> float:
        """Calculate total weighted score."""
        return (
            self.base_score * 0.3 +
            self.success_rate * 0.4 +
            self.speed_factor * 0.15 +
            self.quality_factor * 0.15
        )


@dataclass
class AgentCapability:
    """Definition of an agent capability."""
    name: str
    capability_type: CapabilityType
    description: str = ""
    handler: Optional[Callable] = None
    score: CapabilityScore = field(default_factory=CapabilityScore)
    metadata: Dict[str, Any] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    required_skills: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_score(
        self,
        success: bool,
        execution_time: float,
        quality_score: Optional[float] = None,
    ) -> None:
        """Update capability score based on execution result."""
        self.updated_at = datetime.now()
        
        if success:
            self.score.success_rate = (
                self.score.success_rate * 0.9 + 0.1
            )
        else:
            self.score.success_rate = (
                self.score.success_rate * 0.9
            )
        
        if quality_score is not None:
            self.score.quality_factor = (
                self.score.quality_factor * 0.8 + quality_score * 0.2
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "capability_type": self.capability_type.value,
            "description": self.description,
            "score": {
                "base_score": self.score.base_score,
                "success_rate": self.score.success_rate,
                "speed_factor": self.score.speed_factor,
                "quality_factor": self.score.quality_factor,
                "total_score": self.score.total_score,
            },
            "metadata": self.metadata,
            "required_tools": self.required_tools,
            "required_skills": self.required_skills,
        }


@dataclass
class CapabilityProvider:
    """A provider for a capability."""
    agent_id: str
    agent_name: str
    capability: AgentCapability
    priority: int = 0
    is_available: bool = True
    current_load: int = 0
    max_load: int = 1
    
    @property
    def load_factor(self) -> float:
        """Get current load factor."""
        if self.max_load == 0:
            return 1.0
        return self.current_load / self.max_load
    
    @property
    def effective_score(self) -> float:
        """Calculate effective score considering load."""
        base = self.capability.score.total_score
        load_penalty = self.load_factor * 0.3
        return max(0, base - load_penalty)


class CapabilityRegistry:
    """Central registry for agent capabilities.
    
    Features:
    - Register and unregister capabilities
    - Discover capabilities by type or task
    - Find best provider for a capability
    - Track capability performance
    
    Usage:
        registry = CapabilityRegistry.get_instance()
        
        # Register a capability
        capability = AgentCapability(
            name="test_generation",
            capability_type=CapabilityType.TEST_GENERATION,
            handler=generate_tests,
        )
        registry.register("agent_1", capability)
        
        # Find best provider
        provider = registry.find_best_provider(
            CapabilityType.TEST_GENERATION
        )
    """
    
    _instance: Optional["CapabilityRegistry"] = None
    
    def __init__(self):
        self._capabilities: Dict[str, Dict[str, AgentCapability]] = {}
        self._providers: Dict[str, List[CapabilityProvider]] = {}
        self._type_index: Dict[CapabilityType, Set[str]] = {
            ct: set() for ct in CapabilityType
        }
    
    @classmethod
    def get_instance(cls) -> "CapabilityRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
    
    def register(
        self,
        agent_id: str,
        capability: AgentCapability,
        agent_name: str = "",
        priority: int = 0,
        max_load: int = 1,
    ) -> None:
        """Register a capability for an agent.
        
        Args:
            agent_id: Agent identifier
            capability: Capability to register
            agent_name: Human-readable agent name
            priority: Provider priority (higher = preferred)
            max_load: Maximum concurrent tasks
        """
        if agent_id not in self._capabilities:
            self._capabilities[agent_id] = {}
        
        self._capabilities[agent_id][capability.name] = capability
        
        provider = CapabilityProvider(
            agent_id=agent_id,
            agent_name=agent_name or agent_id,
            capability=capability,
            priority=priority,
            max_load=max_load,
        )
        
        cap_type = capability.capability_type
        if cap_type not in self._providers:
            self._providers[cap_type] = []
        
        existing = next(
            (p for p in self._providers[cap_type] if p.agent_id == agent_id),
            None
        )
        
        if existing:
            self._providers[cap_type].remove(existing)
        
        self._providers[cap_type].append(provider)
        self._type_index[cap_type].add(agent_id)
        
        logger.debug(
            f"Registered capability '{capability.name}' "
            f"for agent '{agent_id}'"
        )
    
    def unregister(self, agent_id: str, capability_name: str) -> bool:
        """Unregister a capability.
        
        Args:
            agent_id: Agent identifier
            capability_name: Name of capability to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if agent_id not in self._capabilities:
            return False
        
        if capability_name not in self._capabilities[agent_id]:
            return False
        
        capability = self._capabilities[agent_id].pop(capability_name)
        cap_type = capability.capability_type
        
        self._providers[cap_type] = [
            p for p in self._providers.get(cap_type, [])
            if not (p.agent_id == agent_id and p.capability.name == capability_name)
        ]
        
        if not self._capabilities[agent_id]:
            del self._capabilities[agent_id]
            self._type_index[cap_type].discard(agent_id)
        
        logger.debug(
            f"Unregistered capability '{capability_name}' "
            f"from agent '{agent_id}'"
        )
        return True
    
    def unregister_agent(self, agent_id: str) -> int:
        """Unregister all capabilities for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Number of capabilities unregistered
        """
        if agent_id not in self._capabilities:
            return 0
        
        count = len(self._capabilities[agent_id])
        
        for cap_name in list(self._capabilities[agent_id].keys()):
            self.unregister(agent_id, cap_name)
        
        return count
    
    def get_capability(
        self,
        agent_id: str,
        capability_name: str,
    ) -> Optional[AgentCapability]:
        """Get a specific capability.
        
        Args:
            agent_id: Agent identifier
            capability_name: Capability name
            
        Returns:
            Capability or None
        """
        return self._capabilities.get(agent_id, {}).get(capability_name)
    
    def get_agent_capabilities(self, agent_id: str) -> List[AgentCapability]:
        """Get all capabilities for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of capabilities
        """
        return list(self._capabilities.get(agent_id, {}).values())
    
    def discover(
        self,
        capability_type: CapabilityType,
        min_score: float = 0.0,
    ) -> List[AgentCapability]:
        """Discover capabilities by type.
        
        Args:
            capability_type: Type of capability
            min_score: Minimum score filter
            
        Returns:
            List of matching capabilities
        """
        capabilities = []
        
        for agent_id in self._type_index.get(capability_type, set()):
            for cap in self._capabilities.get(agent_id, {}).values():
                if cap.capability_type == capability_type:
                    if cap.score.total_score >= min_score:
                        capabilities.append(cap)
        
        return sorted(
            capabilities,
            key=lambda c: c.score.total_score,
            reverse=True,
        )
    
    def find_best_provider(
        self,
        capability_type: CapabilityType,
        exclude_agents: Optional[Set[str]] = None,
    ) -> Optional[CapabilityProvider]:
        """Find the best provider for a capability.
        
        Args:
            capability_type: Type of capability needed
            exclude_agents: Agents to exclude from consideration
            
        Returns:
            Best provider or None
        """
        exclude_agents = exclude_agents or set()
        
        providers = [
            p for p in self._providers.get(capability_type, [])
            if p.agent_id not in exclude_agents and p.is_available
        ]
        
        if not providers:
            return None
        
        def score_provider(p: CapabilityProvider) -> float:
            return (
                p.priority * 0.3 +
                p.effective_score * 0.5 +
                (1 - p.load_factor) * 0.2
            )
        
        return max(providers, key=score_provider)
    
    def find_all_providers(
        self,
        capability_type: CapabilityType,
    ) -> List[CapabilityProvider]:
        """Find all providers for a capability.
        
        Args:
            capability_type: Type of capability
            
        Returns:
            List of providers sorted by score
        """
        providers = self._providers.get(capability_type, [])
        
        def score_provider(p: CapabilityProvider) -> float:
            return (
                p.priority * 0.3 +
                p.effective_score * 0.5 +
                (1 - p.load_factor) * 0.2
            )
        
        return sorted(providers, key=score_provider, reverse=True)
    
    def update_provider_load(
        self,
        agent_id: str,
        capability_type: CapabilityType,
        delta: int,
    ) -> None:
        """Update provider load.
        
        Args:
            agent_id: Agent identifier
            capability_type: Capability type
            delta: Load change (+1 or -1)
        """
        for provider in self._providers.get(capability_type, []):
            if provider.agent_id == agent_id:
                provider.current_load = max(
                    0,
                    min(provider.max_load, provider.current_load + delta)
                )
                break
    
    def set_provider_availability(
        self,
        agent_id: str,
        capability_type: CapabilityType,
        available: bool,
    ) -> None:
        """Set provider availability.
        
        Args:
            agent_id: Agent identifier
            capability_type: Capability type
            available: Availability status
        """
        for provider in self._providers.get(capability_type, []):
            if provider.agent_id == agent_id:
                provider.is_available = available
                break
    
    def record_execution(
        self,
        agent_id: str,
        capability_name: str,
        success: bool,
        execution_time: float,
        quality_score: Optional[float] = None,
    ) -> None:
        """Record execution result for capability scoring.
        
        Args:
            agent_id: Agent identifier
            capability_name: Capability name
            success: Whether execution succeeded
            execution_time: Execution time in seconds
            quality_score: Optional quality score (0-1)
        """
        capability = self.get_capability(agent_id, capability_name)
        
        if capability:
            capability.update_score(success, execution_time, quality_score)
            logger.debug(
                f"Updated capability '{capability_name}' score: "
                f"success_rate={capability.score.success_rate:.2f}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        total_capabilities = sum(
            len(caps) for caps in self._capabilities.values()
        )
        
        type_counts = {
            ct.value: len(providers)
            for ct, providers in self._providers.items()
        }
        
        return {
            "total_agents": len(self._capabilities),
            "total_capabilities": total_capabilities,
            "capabilities_by_type": type_counts,
            "total_providers": sum(
                len(providers) for providers in self._providers.values()
            ),
        }
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._capabilities.clear()
        self._providers.clear()
        self._type_index = {ct: set() for ct in CapabilityType}


def get_capability_registry() -> CapabilityRegistry:
    """Get the global capability registry."""
    return CapabilityRegistry.get_instance()
