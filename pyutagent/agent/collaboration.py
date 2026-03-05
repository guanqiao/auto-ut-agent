"""Collaboration Patterns for Specialized SubAgents.

This module provides advanced collaboration patterns:
- Delegation: Simple task delegation
- Negotiation: Agent negotiation for task assignment
- Bidding: Competitive bidding for tasks
- Consensus: Multi-agent consensus building

This is part of Phase 3 Week 19-20: Specialized SubAgent Enhancement.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from pyutagent.agent.capability_registry import (
        CapabilityRegistry,
        CapabilityProvider,
        CapabilityType,
    )

logger = logging.getLogger(__name__)


class CollaborationPattern(Enum):
    """Types of collaboration patterns."""
    DELEGATION = "delegation"
    NEGOTIATION = "negotiation"
    BIDDING = "bidding"
    CONSENSUS = "consensus"
    ROUND_ROBIN = "round_robin"
    BROADCAST = "broadcast"


@dataclass
class Task:
    """A task to be assigned to agents."""
    task_id: str
    task_type: str
    description: str
    priority: int = 5
    deadline: Optional[datetime] = None
    required_capabilities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Bid:
    """A bid from an agent for a task."""
    agent_id: str
    task_id: str
    score: float
    estimated_time: float
    confidence: float
    proposed_approach: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class NegotiationProposal:
    """A proposal during negotiation."""
    agent_id: str
    task_id: str
    capability_match: float
    availability: float
    historical_success: float
    proposed_terms: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusVote:
    """A vote in consensus building."""
    agent_id: str
    proposal_id: str
    vote: bool
    confidence: float
    reason: str = ""


@dataclass
class CollaborationResult:
    """Result of a collaboration process."""
    success: bool
    selected_agent: Optional[str] = None
    task: Optional[Task] = None
    pattern_used: CollaborationPattern = CollaborationPattern.DELEGATION
    participants: List[str] = field(default_factory=list)
    bids: List[Bid] = field(default_factory=list)
    proposals: List[NegotiationProposal] = field(default_factory=list)
    votes: List[ConsensusVote] = field(default_factory=list)
    duration_ms: int = 0
    error: Optional[str] = None


class CollaborationStrategy(ABC):
    """Abstract base class for collaboration strategies."""
    
    @abstractmethod
    async def assign(
        self,
        task: Task,
        candidates: List[str],
        registry: "CapabilityRegistry",
    ) -> CollaborationResult:
        """Assign a task using this strategy.
        
        Args:
            task: Task to assign
            candidates: List of candidate agent IDs
            registry: Capability registry
            
        Returns:
            CollaborationResult with assignment details
        """
        pass


class DelegationStrategy(CollaborationStrategy):
    """Simple delegation strategy.
    
    Selects the best available agent based on capability scores.
    """
    
    async def assign(
        self,
        task: Task,
        candidates: List[str],
        registry: "CapabilityRegistry",
    ) -> CollaborationResult:
        """Delegate to best available agent."""
        from pyutagent.agent.capability_registry import CapabilityType
        
        start_time = datetime.now()
        
        try:
            cap_type = CapabilityType(task.task_type)
        except ValueError:
            cap_type = None
        
        if cap_type is None:
            return CollaborationResult(
                success=False,
                task=task,
                pattern_used=CollaborationPattern.DELEGATION,
                error=f"Unknown task type: {task.task_type}",
            )
        
        provider = registry.find_best_provider(
            cap_type,
            exclude_agents=set(candidates) if not candidates else set(),
        )
        
        if provider is None:
            return CollaborationResult(
                success=False,
                task=task,
                pattern_used=CollaborationPattern.DELEGATION,
                error="No available provider found",
            )
        
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return CollaborationResult(
            success=True,
            selected_agent=provider.agent_id,
            task=task,
            pattern_used=CollaborationPattern.DELEGATION,
            participants=[provider.agent_id],
            duration_ms=int(duration),
        )


class BiddingStrategy(CollaborationStrategy):
    """Competitive bidding strategy.
    
    Agents bid for tasks and the best bid wins.
    """
    
    def __init__(
        self,
        bid_timeout_ms: int = 5000,
        min_bids: int = 1,
    ):
        self.bid_timeout_ms = bid_timeout_ms
        self.min_bids = min_bids
        self._bid_handlers: Dict[str, Callable] = {}
    
    def register_bid_handler(
        self,
        agent_id: str,
        handler: Callable[[Task], Bid],
    ) -> None:
        """Register a bid handler for an agent."""
        self._bid_handlers[agent_id] = handler
    
    async def assign(
        self,
        task: Task,
        candidates: List[str],
        registry: "CapabilityRegistry",
    ) -> CollaborationResult:
        """Collect bids and select winner."""
        start_time = datetime.now()
        bids: List[Bid] = []
        
        async def collect_bid(agent_id: str) -> Optional[Bid]:
            handler = self._bid_handlers.get(agent_id)
            if handler is None:
                return self._create_default_bid(agent_id, task, registry)
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(task)
                return handler(task)
            except Exception as e:
                logger.warning(f"Bid handler error for {agent_id}: {e}")
                return None
        
        timeout = self.bid_timeout_ms / 1000
        try:
            bid_tasks = [collect_bid(aid) for aid in candidates]
            results = await asyncio.wait_for(
                asyncio.gather(*bid_tasks, return_exceptions=True),
                timeout=timeout,
            )
            
            for result in results:
                if isinstance(result, Bid):
                    bids.append(result)
        except asyncio.TimeoutError:
            logger.warning(f"Bidding timed out for task {task.task_id}")
        
        if len(bids) < self.min_bids:
            return CollaborationResult(
                success=False,
                task=task,
                pattern_used=CollaborationPattern.BIDDING,
                bids=bids,
                error=f"Insufficient bids: {len(bids)} < {self.min_bids}",
            )
        
        winner = max(bids, key=lambda b: self._score_bid(b))
        
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return CollaborationResult(
            success=True,
            selected_agent=winner.agent_id,
            task=task,
            pattern_used=CollaborationPattern.BIDDING,
            participants=[b.agent_id for b in bids],
            bids=bids,
            duration_ms=int(duration),
        )
    
    def _create_default_bid(
        self,
        agent_id: str,
        task: Task,
        registry: "CapabilityRegistry",
    ) -> Optional[Bid]:
        """Create a default bid based on registry data."""
        from pyutagent.agent.capability_registry import CapabilityType
        
        try:
            cap_type = CapabilityType(task.task_type)
        except ValueError:
            return None
        
        providers = registry.find_all_providers(cap_type)
        provider = next((p for p in providers if p.agent_id == agent_id), None)
        
        if provider is None:
            return None
        
        return Bid(
            agent_id=agent_id,
            task_id=task.task_id,
            score=provider.effective_score,
            estimated_time=10.0,
            confidence=provider.capability.score.success_rate,
        )
    
    def _score_bid(self, bid: Bid) -> float:
        """Score a bid for comparison."""
        return (
            bid.score * 0.4 +
            bid.confidence * 0.3 +
            (1.0 / max(bid.estimated_time, 0.1)) * 0.3
        )


class NegotiationStrategy(CollaborationStrategy):
    """Negotiation-based task assignment.
    
    Agents negotiate for tasks based on capabilities and availability.
    """
    
    def __init__(
        self,
        max_rounds: int = 3,
        negotiation_timeout_ms: int = 10000,
    ):
        self.max_rounds = max_rounds
        self.negotiation_timeout_ms = negotiation_timeout_ms
    
    async def assign(
        self,
        task: Task,
        candidates: List[str],
        registry: "CapabilityRegistry",
    ) -> CollaborationResult:
        """Negotiate task assignment."""
        from pyutagent.agent.capability_registry import CapabilityType
        
        start_time = datetime.now()
        proposals: List[NegotiationProposal] = []
        
        try:
            cap_type = CapabilityType(task.task_type)
        except ValueError:
            return CollaborationResult(
                success=False,
                task=task,
                pattern_used=CollaborationPattern.NEGOTIATION,
                error=f"Unknown task type: {task.task_type}",
            )
        
        for agent_id in candidates:
            caps = registry.get_agent_capabilities(agent_id)
            matching_caps = [c for c in caps if c.capability_type == cap_type]
            
            if matching_caps:
                best_cap = max(matching_caps, key=lambda c: c.score.total_score)
                
                providers = registry.find_all_providers(cap_type)
                provider = next((p for p in providers if p.agent_id == agent_id), None)
                
                availability = 1.0 - (provider.load_factor if provider else 0.0)
                
                proposals.append(NegotiationProposal(
                    agent_id=agent_id,
                    task_id=task.task_id,
                    capability_match=best_cap.score.total_score,
                    availability=availability,
                    historical_success=best_cap.score.success_rate,
                ))
        
        if not proposals:
            return CollaborationResult(
                success=False,
                task=task,
                pattern_used=CollaborationPattern.NEGOTIATION,
                error="No suitable agents found",
            )
        
        winner = self._select_winner(proposals)
        
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return CollaborationResult(
            success=True,
            selected_agent=winner.agent_id,
            task=task,
            pattern_used=CollaborationPattern.NEGOTIATION,
            participants=[p.agent_id for p in proposals],
            proposals=proposals,
            duration_ms=int(duration),
        )
    
    def _select_winner(self, proposals: List[NegotiationProposal]) -> NegotiationProposal:
        """Select winner from proposals."""
        def score_proposal(p: NegotiationProposal) -> float:
            return (
                p.capability_match * 0.4 +
                p.availability * 0.3 +
                p.historical_success * 0.3
            )
        
        return max(proposals, key=score_proposal)


class ConsensusStrategy(CollaborationStrategy):
    """Consensus-based task assignment.
    
    Multiple agents vote on who should handle a task.
    """
    
    def __init__(
        self,
        quorum: float = 0.5,
        voting_timeout_ms: int = 5000,
    ):
        self.quorum = quorum
        self.voting_timeout_ms = voting_timeout_ms
    
    async def assign(
        self,
        task: Task,
        candidates: List[str],
        registry: "CapabilityRegistry",
    ) -> CollaborationResult:
        """Build consensus on task assignment."""
        from pyutagent.agent.capability_registry import CapabilityType
        
        start_time = datetime.now()
        votes: List[ConsensusVote] = []
        
        try:
            cap_type = CapabilityType(task.task_type)
        except ValueError:
            return CollaborationResult(
                success=False,
                task=task,
                pattern_used=CollaborationPattern.CONSENSUS,
                error=f"Unknown task type: {task.task_type}",
            )
        
        providers = registry.find_all_providers(cap_type)
        
        if not providers:
            return CollaborationResult(
                success=False,
                task=task,
                pattern_used=CollaborationPattern.CONSENSUS,
                error="No providers available",
            )
        
        best_provider = max(
            providers,
            key=lambda p: p.effective_score,
        )
        
        for agent_id in candidates:
            vote = agent_id == best_provider.agent_id
            confidence = 0.8 if vote else 0.2
            
            votes.append(ConsensusVote(
                agent_id=agent_id,
                proposal_id=best_provider.agent_id,
                vote=vote,
                confidence=confidence,
            ))
        
        positive_votes = sum(1 for v in votes if v.vote)
        quorum_reached = positive_votes >= len(candidates) * self.quorum
        
        duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return CollaborationResult(
            success=quorum_reached,
            selected_agent=best_provider.agent_id if quorum_reached else None,
            task=task,
            pattern_used=CollaborationPattern.CONSENSUS,
            participants=candidates,
            votes=votes,
            duration_ms=int(duration),
            error="Quorum not reached" if not quorum_reached else None,
        )


class CollaborationOrchestrator:
    """Orchestrator for agent collaboration.
    
    Manages different collaboration patterns and selects
    the appropriate one for each task.
    """
    
    def __init__(self, registry: "CapabilityRegistry"):
        self._registry = registry
        self._strategies: Dict[CollaborationPattern, CollaborationStrategy] = {
            CollaborationPattern.DELEGATION: DelegationStrategy(),
            CollaborationPattern.BIDDING: BiddingStrategy(),
            CollaborationPattern.NEGOTIATION: NegotiationStrategy(),
            CollaborationPattern.CONSENSUS: ConsensusStrategy(),
        }
        self._default_pattern = CollaborationPattern.DELEGATION
    
    def register_strategy(
        self,
        pattern: CollaborationPattern,
        strategy: CollaborationStrategy,
    ) -> None:
        """Register a strategy for a pattern."""
        self._strategies[pattern] = strategy
    
    def set_default_pattern(self, pattern: CollaborationPattern) -> None:
        """Set the default collaboration pattern."""
        self._default_pattern = pattern
    
    async def assign_task(
        self,
        task: Task,
        candidates: Optional[List[str]] = None,
        pattern: Optional[CollaborationPattern] = None,
    ) -> CollaborationResult:
        """Assign a task using collaboration.
        
        Args:
            task: Task to assign
            candidates: Optional list of candidate agents
            pattern: Optional specific pattern to use
            
        Returns:
            CollaborationResult
        """
        pattern = pattern or self._default_pattern
        strategy = self._strategies.get(pattern)
        
        if strategy is None:
            return CollaborationResult(
                success=False,
                task=task,
                pattern_used=pattern,
                error=f"No strategy for pattern: {pattern}",
            )
        
        if candidates is None or len(candidates) == 0:
            candidates = self._get_candidates(task)
        
        return await strategy.assign(task, candidates, self._registry)
    
    def _get_candidates(self, task: Task) -> List[str]:
        """Get candidate agents for a task."""
        from pyutagent.agent.capability_registry import CapabilityType
        
        try:
            cap_type = CapabilityType(task.task_type)
        except ValueError:
            return []
        
        providers = self._registry.find_all_providers(cap_type)
        return [p.agent_id for p in providers if p.is_available]
    
    def get_available_patterns(self) -> List[CollaborationPattern]:
        """Get list of available patterns."""
        return list(self._strategies.keys())
