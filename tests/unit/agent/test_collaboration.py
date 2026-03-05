"""Tests for Collaboration Patterns.

This module tests the collaboration system including:
- Delegation strategy
- Bidding strategy
- Negotiation strategy
- Consensus strategy
- Collaboration orchestrator
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from pyutagent.agent.capability_registry import (
    CapabilityType,
    AgentCapability,
    CapabilityRegistry,
)
from pyutagent.agent.collaboration import (
    CollaborationPattern,
    Task,
    Bid,
    NegotiationProposal,
    ConsensusVote,
    CollaborationResult,
    CollaborationStrategy,
    DelegationStrategy,
    BiddingStrategy,
    NegotiationStrategy,
    ConsensusStrategy,
    CollaborationOrchestrator,
)


class TestTask:
    """Tests for Task dataclass."""
    
    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        assert task.task_id == "task_1"
        assert task.task_type == "code_generation"
        assert task.description == "Generate code"
        assert task.priority == 5
        assert isinstance(task.created_at, datetime)
    
    def test_task_with_deadline(self):
        """Test task with deadline."""
        deadline = datetime(2025, 12, 31)
        task = Task(
            task_id="task_1",
            task_type="test_generation",
            description="Generate tests",
            deadline=deadline,
        )
        
        assert task.deadline == deadline
    
    def test_task_with_capabilities(self):
        """Test task with required capabilities."""
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
            required_capabilities=["python", "testing"],
        )
        
        assert "python" in task.required_capabilities
        assert "testing" in task.required_capabilities


class TestBid:
    """Tests for Bid dataclass."""
    
    def test_bid_creation(self):
        """Test creating a bid."""
        bid = Bid(
            agent_id="agent_1",
            task_id="task_1",
            score=0.9,
            estimated_time=5.0,
            confidence=0.8,
        )
        
        assert bid.agent_id == "agent_1"
        assert bid.task_id == "task_1"
        assert bid.score == 0.9
        assert bid.estimated_time == 5.0
        assert bid.confidence == 0.8
    
    def test_bid_with_approach(self):
        """Test bid with proposed approach."""
        bid = Bid(
            agent_id="agent_1",
            task_id="task_1",
            score=0.9,
            estimated_time=5.0,
            confidence=0.8,
            proposed_approach="Use TDD approach",
        )
        
        assert bid.proposed_approach == "Use TDD approach"


class TestNegotiationProposal:
    """Tests for NegotiationProposal dataclass."""
    
    def test_proposal_creation(self):
        """Test creating a proposal."""
        proposal = NegotiationProposal(
            agent_id="agent_1",
            task_id="task_1",
            capability_match=0.9,
            availability=0.8,
            historical_success=0.7,
        )
        
        assert proposal.agent_id == "agent_1"
        assert proposal.capability_match == 0.9
        assert proposal.availability == 0.8
        assert proposal.historical_success == 0.7


class TestConsensusVote:
    """Tests for ConsensusVote dataclass."""
    
    def test_vote_creation(self):
        """Test creating a vote."""
        vote = ConsensusVote(
            agent_id="agent_1",
            proposal_id="proposal_1",
            vote=True,
            confidence=0.9,
        )
        
        assert vote.agent_id == "agent_1"
        assert vote.proposal_id == "proposal_1"
        assert vote.vote is True
        assert vote.confidence == 0.9


class TestCollaborationResult:
    """Tests for CollaborationResult dataclass."""
    
    def test_success_result(self):
        """Test successful result."""
        result = CollaborationResult(
            success=True,
            selected_agent="agent_1",
            pattern_used=CollaborationPattern.DELEGATION,
        )
        
        assert result.success is True
        assert result.selected_agent == "agent_1"
        assert result.pattern_used == CollaborationPattern.DELEGATION
    
    def test_failure_result(self):
        """Test failure result."""
        result = CollaborationResult(
            success=False,
            error="No available provider",
        )
        
        assert result.success is False
        assert result.error == "No available provider"


class TestDelegationStrategy:
    """Tests for DelegationStrategy."""
    
    @pytest.fixture
    def setup_registry(self):
        """Set up test registry."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        cap1 = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap1.score.base_score = 0.9
        
        cap2 = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2.score.base_score = 0.5
        
        registry.register("agent_1", cap1, priority=1)
        registry.register("agent_2", cap2, priority=2)
        
        yield registry
        CapabilityRegistry.reset_instance()
    
    @pytest.mark.asyncio
    async def test_delegate_to_best_agent(self, setup_registry):
        """Test delegating to the best agent."""
        strategy = DelegationStrategy()
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(task, [], setup_registry)
        
        assert result.success is True
        assert result.selected_agent is not None
        assert result.pattern_used == CollaborationPattern.DELEGATION
    
    @pytest.mark.asyncio
    async def test_delegate_unknown_task_type(self, setup_registry):
        """Test delegation with unknown task type."""
        strategy = DelegationStrategy()
        task = Task(
            task_id="task_1",
            task_type="unknown_type",
            description="Unknown task",
        )
        
        result = await strategy.assign(task, [], setup_registry)
        
        assert result.success is False
        assert "Unknown task type" in result.error
    
    @pytest.mark.asyncio
    async def test_delegate_no_provider(self):
        """Test delegation when no provider available."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        strategy = DelegationStrategy()
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(task, [], registry)
        
        assert result.success is False
        assert result.error == "No available provider found"
        
        CapabilityRegistry.reset_instance()


class TestBiddingStrategy:
    """Tests for BiddingStrategy."""
    
    @pytest.fixture
    def setup_registry(self):
        """Set up test registry."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        cap = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap.score.base_score = 0.8
        
        registry.register("agent_1", cap)
        registry.register("agent_2", cap)
        
        yield registry
        CapabilityRegistry.reset_instance()
    
    @pytest.mark.asyncio
    async def test_bidding_with_handlers(self, setup_registry):
        """Test bidding with custom handlers."""
        strategy = BiddingStrategy(min_bids=1)
        
        def handler1(task):
            return Bid(
                agent_id="agent_1",
                task_id=task.task_id,
                score=0.9,
                estimated_time=5.0,
                confidence=0.8,
            )
        
        async def handler2(task):
            return Bid(
                agent_id="agent_2",
                task_id=task.task_id,
                score=0.7,
                estimated_time=3.0,
                confidence=0.9,
            )
        
        strategy.register_bid_handler("agent_1", handler1)
        strategy.register_bid_handler("agent_2", handler2)
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(task, ["agent_1", "agent_2"], setup_registry)
        
        assert result.success is True
        assert len(result.bids) == 2
        assert result.selected_agent in ["agent_1", "agent_2"]
    
    @pytest.mark.asyncio
    async def test_bidding_insufficient_bids(self, setup_registry):
        """Test bidding with insufficient bids."""
        strategy = BiddingStrategy(min_bids=3)
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(task, ["agent_1"], setup_registry)
        
        assert result.success is False
        assert "Insufficient bids" in result.error
    
    @pytest.mark.asyncio
    async def test_bidding_default_bids(self, setup_registry):
        """Test bidding with default bids from registry."""
        strategy = BiddingStrategy(min_bids=1)
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(task, ["agent_1"], setup_registry)
        
        assert result.success is True
        assert len(result.bids) >= 1
    
    @pytest.mark.asyncio
    async def test_bidding_unknown_task_type(self, setup_registry):
        """Test bidding with unknown task type."""
        strategy = BiddingStrategy()
        
        task = Task(
            task_id="task_1",
            task_type="unknown_type",
            description="Unknown task",
        )
        
        result = await strategy.assign(task, ["agent_1"], setup_registry)
        
        assert result.success is False
    
    def test_score_bid(self, setup_registry):
        """Test bid scoring."""
        strategy = BiddingStrategy()
        
        bid1 = Bid(
            agent_id="agent_1",
            task_id="task_1",
            score=0.9,
            estimated_time=5.0,
            confidence=0.8,
        )
        
        bid2 = Bid(
            agent_id="agent_2",
            task_id="task_1",
            score=0.7,
            estimated_time=2.0,
            confidence=0.9,
        )
        
        score1 = strategy._score_bid(bid1)
        score2 = strategy._score_bid(bid2)
        
        assert isinstance(score1, float)
        assert isinstance(score2, float)


class TestNegotiationStrategy:
    """Tests for NegotiationStrategy."""
    
    @pytest.fixture
    def setup_registry(self):
        """Set up test registry."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        cap1 = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap1.score.base_score = 0.9
        cap1.score.success_rate = 0.8
        
        cap2 = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2.score.base_score = 0.7
        cap2.score.success_rate = 0.6
        
        registry.register("agent_1", cap1)
        registry.register("agent_2", cap2)
        
        yield registry
        CapabilityRegistry.reset_instance()
    
    @pytest.mark.asyncio
    async def test_negotiation_select_best(self, setup_registry):
        """Test negotiation selects best agent."""
        strategy = NegotiationStrategy()
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(
            task,
            ["agent_1", "agent_2"],
            setup_registry,
        )
        
        assert result.success is True
        assert result.selected_agent == "agent_1"
        assert len(result.proposals) == 2
    
    @pytest.mark.asyncio
    async def test_negotiation_no_suitable_agents(self):
        """Test negotiation with no suitable agents."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        strategy = NegotiationStrategy()
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(task, [], registry)
        
        assert result.success is False
        assert result.error == "No suitable agents found"
        
        CapabilityRegistry.reset_instance()
    
    @pytest.mark.asyncio
    async def test_negotiation_unknown_task_type(self, setup_registry):
        """Test negotiation with unknown task type."""
        strategy = NegotiationStrategy()
        
        task = Task(
            task_id="task_1",
            task_type="unknown_type",
            description="Unknown task",
        )
        
        result = await strategy.assign(task, ["agent_1"], setup_registry)
        
        assert result.success is False
        assert "Unknown task type" in result.error


class TestConsensusStrategy:
    """Tests for ConsensusStrategy."""
    
    @pytest.fixture
    def setup_registry(self):
        """Set up test registry."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        cap1 = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap1.score.base_score = 0.9
        
        cap2 = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap2.score.base_score = 0.5
        
        registry.register("agent_1", cap1)
        registry.register("agent_2", cap2)
        
        yield registry
        CapabilityRegistry.reset_instance()
    
    @pytest.mark.asyncio
    async def test_consensus_quorum_reached(self, setup_registry):
        """Test consensus with quorum reached."""
        strategy = ConsensusStrategy(quorum=0.5)
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(
            task,
            ["agent_1", "agent_2"],
            setup_registry,
        )
        
        assert result.success is True
        assert result.selected_agent is not None
        assert len(result.votes) == 2
    
    @pytest.mark.asyncio
    async def test_consensus_quorum_not_reached(self, setup_registry):
        """Test consensus with quorum not reached."""
        strategy = ConsensusStrategy(quorum=1.0)
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(
            task,
            ["agent_1", "agent_2"],
            setup_registry,
        )
        
        assert result.success is False
        assert "Quorum not reached" in result.error
    
    @pytest.mark.asyncio
    async def test_consensus_no_providers(self):
        """Test consensus with no providers."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        strategy = ConsensusStrategy()
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await strategy.assign(task, [], registry)
        
        assert result.success is False
        assert result.error == "No providers available"
        
        CapabilityRegistry.reset_instance()
    
    @pytest.mark.asyncio
    async def test_consensus_unknown_task_type(self, setup_registry):
        """Test consensus with unknown task type."""
        strategy = ConsensusStrategy()
        
        task = Task(
            task_id="task_1",
            task_type="unknown_type",
            description="Unknown task",
        )
        
        result = await strategy.assign(task, ["agent_1"], setup_registry)
        
        assert result.success is False
        assert "Unknown task type" in result.error


class TestCollaborationOrchestrator:
    """Tests for CollaborationOrchestrator."""
    
    @pytest.fixture
    def setup_orchestrator(self):
        """Set up test orchestrator."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        cap = AgentCapability(
            name="code_gen",
            capability_type=CapabilityType.CODE_GENERATION,
        )
        cap.score.base_score = 0.8
        
        registry.register("agent_1", cap)
        
        orchestrator = CollaborationOrchestrator(registry)
        
        yield orchestrator
        CapabilityRegistry.reset_instance()
    
    @pytest.mark.asyncio
    async def test_assign_task_default_pattern(self, setup_orchestrator):
        """Test task assignment with default pattern."""
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await setup_orchestrator.assign_task(task)
        
        assert result.success is True
        assert result.pattern_used == CollaborationPattern.DELEGATION
    
    @pytest.mark.asyncio
    async def test_assign_task_specific_pattern(self, setup_orchestrator):
        """Test task assignment with specific pattern."""
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await setup_orchestrator.assign_task(
            task,
            candidates=["agent_1"],
            pattern=CollaborationPattern.BIDDING,
        )
        
        assert result.pattern_used == CollaborationPattern.BIDDING
    
    @pytest.mark.asyncio
    async def test_assign_task_with_candidates(self, setup_orchestrator):
        """Test task assignment with specific candidates."""
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await setup_orchestrator.assign_task(
            task,
            candidates=["agent_1"],
        )
        
        assert result.success is True
    
    def test_register_strategy(self, setup_orchestrator):
        """Test registering a custom strategy."""
        custom_strategy = MagicMock(spec=CollaborationStrategy)
        
        setup_orchestrator.register_strategy(
            CollaborationPattern.ROUND_ROBIN,
            custom_strategy,
        )
        
        assert CollaborationPattern.ROUND_ROBIN in setup_orchestrator._strategies
    
    def test_set_default_pattern(self, setup_orchestrator):
        """Test setting default pattern."""
        setup_orchestrator.set_default_pattern(CollaborationPattern.CONSENSUS)
        
        assert setup_orchestrator._default_pattern == CollaborationPattern.CONSENSUS
    
    def test_get_available_patterns(self, setup_orchestrator):
        """Test getting available patterns."""
        patterns = setup_orchestrator.get_available_patterns()
        
        assert CollaborationPattern.DELEGATION in patterns
        assert CollaborationPattern.BIDDING in patterns
        assert CollaborationPattern.NEGOTIATION in patterns
        assert CollaborationPattern.CONSENSUS in patterns
    
    @pytest.mark.asyncio
    async def test_assign_task_no_strategy(self, setup_orchestrator):
        """Test assignment with no strategy for pattern."""
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await setup_orchestrator.assign_task(
            task,
            pattern=CollaborationPattern.ROUND_ROBIN,
        )
        
        assert result.success is False
        assert "No strategy for pattern" in result.error


class TestCollaborationPatterns:
    """Tests for all collaboration patterns."""
    
    def test_all_patterns_exist(self):
        """Test that all patterns are defined."""
        expected_patterns = [
            "DELEGATION",
            "NEGOTIATION",
            "BIDDING",
            "CONSENSUS",
            "ROUND_ROBIN",
            "BROADCAST",
        ]
        
        for pattern_name in expected_patterns:
            assert hasattr(CollaborationPattern, pattern_name)
    
    def test_pattern_values(self):
        """Test pattern string values."""
        assert CollaborationPattern.DELEGATION.value == "delegation"
        assert CollaborationPattern.BIDDING.value == "bidding"
        assert CollaborationPattern.CONSENSUS.value == "consensus"


class TestIntegration:
    """Integration tests for collaboration system."""
    
    @pytest.fixture
    def setup_full_registry(self):
        """Set up a full registry with multiple agents."""
        CapabilityRegistry.reset_instance()
        registry = CapabilityRegistry()
        
        agents = [
            ("agent_1", CapabilityType.CODE_GENERATION, 0.9, 2),
            ("agent_2", CapabilityType.CODE_GENERATION, 0.7, 1),
            ("agent_3", CapabilityType.TEST_GENERATION, 0.8, 2),
            ("agent_4", CapabilityType.TEST_GENERATION, 0.6, 1),
        ]
        
        for agent_id, cap_type, score, priority in agents:
            cap = AgentCapability(
                name=cap_type.value,
                capability_type=cap_type,
            )
            cap.score.base_score = score
            registry.register(agent_id, cap, priority=priority)
        
        yield registry
        CapabilityRegistry.reset_instance()
    
    @pytest.mark.asyncio
    async def test_full_workflow_delegation(self, setup_full_registry):
        """Test full workflow with delegation."""
        orchestrator = CollaborationOrchestrator(setup_full_registry)
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await orchestrator.assign_task(task)
        
        assert result.success is True
        assert result.selected_agent in ["agent_1", "agent_2"]
    
    @pytest.mark.asyncio
    async def test_full_workflow_bidding(self, setup_full_registry):
        """Test full workflow with bidding."""
        orchestrator = CollaborationOrchestrator(setup_full_registry)
        
        bidding_strategy = BiddingStrategy(min_bids=1)
        orchestrator.register_strategy(CollaborationPattern.BIDDING, bidding_strategy)
        
        task = Task(
            task_id="task_1",
            task_type="test_generation",
            description="Generate tests",
        )
        
        result = await orchestrator.assign_task(
            task,
            candidates=["agent_3", "agent_4"],
            pattern=CollaborationPattern.BIDDING,
        )
        
        assert result.success is True
        assert result.selected_agent in ["agent_3", "agent_4"]
    
    @pytest.mark.asyncio
    async def test_full_workflow_negotiation(self, setup_full_registry):
        """Test full workflow with negotiation."""
        orchestrator = CollaborationOrchestrator(setup_full_registry)
        
        task = Task(
            task_id="task_1",
            task_type="code_generation",
            description="Generate code",
        )
        
        result = await orchestrator.assign_task(
            task,
            candidates=["agent_1", "agent_2"],
            pattern=CollaborationPattern.NEGOTIATION,
        )
        
        assert result.success is True
        assert result.selected_agent == "agent_1"
    
    @pytest.mark.asyncio
    async def test_full_workflow_consensus(self, setup_full_registry):
        """Test full workflow with consensus."""
        orchestrator = CollaborationOrchestrator(setup_full_registry)
        
        task = Task(
            task_id="task_1",
            task_type="test_generation",
            description="Generate tests",
        )
        
        result = await orchestrator.assign_task(
            task,
            candidates=["agent_3", "agent_4"],
            pattern=CollaborationPattern.CONSENSUS,
        )
        
        assert result.success is True
        assert result.selected_agent == "agent_3"
