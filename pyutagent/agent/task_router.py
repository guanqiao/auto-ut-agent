"""Intelligent Task Router for optimal agent selection.

This module provides:
- IntelligentTaskRouter: Route tasks to optimal agents
- Affinity scoring based on capabilities and history
- Learning from routing decisions
- Fallback strategies
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from .subagents import SubAgent, Task, AgentCapability, AgentStatus

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Strategies for task routing."""
    CAPABILITY_MATCH = "capability_match"
    LOAD_BALANCED = "load_balanced"
    LEAST_LOADED = "least_loaded"
    ROUND_ROBIN = "round_robin"
    AFFINITY_BASED = "affinity_based"
    ADAPTIVE = "adaptive"


@dataclass
class RoutingScore:
    """Score for a routing decision."""
    agent_id: str
    capability_score: float = 0.0
    load_score: float = 0.0
    history_score: float = 0.0
    affinity_score: float = 0.0
    total_score: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Record of a routing decision."""
    decision_id: str
    task_id: str
    task_type: str
    agent_id: str
    strategy: RoutingStrategy
    score: RoutingScore
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: Optional[bool] = None
    execution_time_ms: Optional[int] = None


@dataclass
class AgentProfile:
    """Profile of an agent for routing decisions."""
    agent_id: str
    agent_type: str
    capabilities: Set[str]
    current_load: int = 0
    max_load: int = 1
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time_ms: float = 0.0
    task_type_affinity: Dict[str, float] = field(default_factory=dict)
    last_used: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 0.5
        return self.successful_tasks / self.total_tasks

    @property
    def load_percentage(self) -> float:
        """Calculate load percentage."""
        if self.max_load == 0:
            return 1.0
        return self.current_load / self.max_load

    @property
    def is_available(self) -> bool:
        """Check if agent is available."""
        return self.current_load < self.max_load


class IntelligentTaskRouter:
    """Router for intelligent task-to-agent assignment.

    Features:
    - Multiple routing strategies
    - Capability-based matching
    - Load balancing
    - Learning from history
    - Affinity scoring
    """

    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        learning_enabled: bool = True,
        history_weight: float = 0.3
    ):
        """Initialize the router.

        Args:
            default_strategy: Default routing strategy
            learning_enabled: Whether to learn from decisions
            history_weight: Weight of history in scoring
        """
        self.default_strategy = default_strategy
        self.learning_enabled = learning_enabled
        self.history_weight = history_weight

        self._agent_profiles: Dict[str, AgentProfile] = {}
        self._routing_history: List[RoutingDecision] = []
        self._task_type_map: Dict[str, Set[str]] = {}
        self._round_robin_index: int = 0

        self._stats = {
            "total_routes": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "fallback_routes": 0
        }

        logger.info(f"[IntelligentTaskRouter] Initialized with strategy: {default_strategy.value}")

    def register_agent(
        self,
        agent: SubAgent,
        capabilities: Optional[List[AgentCapability]] = None
    ) -> None:
        """Register an agent for routing.

        Args:
            agent: Agent to register
            capabilities: Optional list of capabilities
        """
        cap_names = set()
        if capabilities:
            cap_names = {c.name for c in capabilities}
        elif hasattr(agent.config, 'capabilities'):
            cap_names = {c.name for c in agent.config.capabilities}

        profile = AgentProfile(
            agent_id=agent.id,
            agent_type=agent.config.agent_type,
            capabilities=cap_names,
            max_load=agent.config.max_concurrent_tasks if hasattr(agent.config, 'max_concurrent_tasks') else 1
        )

        self._agent_profiles[agent.id] = profile
        logger.info(f"[IntelligentTaskRouter] Registered agent: {agent.id} (type={profile.agent_type})")

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if agent was unregistered
        """
        if agent_id in self._agent_profiles:
            del self._agent_profiles[agent_id]
            logger.info(f"[IntelligentTaskRouter] Unregistered agent: {agent_id}")
            return True
        return False

    def update_agent_load(self, agent_id: str, delta: int) -> None:
        """Update agent load.

        Args:
            agent_id: Agent ID
            delta: Load change (+1 or -1)
        """
        if agent_id in self._agent_profiles:
            profile = self._agent_profiles[agent_id]
            profile.current_load = max(0, profile.current_load + delta)
            if delta > 0:
                profile.last_used = datetime.now().isoformat()

    def route(
        self,
        task: Task,
        agents: Optional[List[SubAgent]] = None,
        strategy: Optional[RoutingStrategy] = None
    ) -> Optional[SubAgent]:
        """Route a task to the best agent.

        Args:
            task: Task to route
            agents: Optional list of agents (uses registered if not provided)
            strategy: Optional routing strategy override

        Returns:
            Selected SubAgent or None
        """
        strategy = strategy or self.default_strategy

        if agents:
            for agent in agents:
                if agent.id not in self._agent_profiles:
                    self.register_agent(agent)

        available_agents = self._get_available_agents(agents)

        if not available_agents:
            logger.warning(f"[IntelligentTaskRouter] No available agents for task: {task.id}")
            return None

        if strategy == RoutingStrategy.ADAPTIVE:
            strategy = self._choose_adaptive_strategy(task)

        selected_agent = None

        if strategy == RoutingStrategy.CAPABILITY_MATCH:
            selected_agent = self._route_by_capability(task, available_agents)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            selected_agent = self._route_by_load_balance(task, available_agents)
        elif strategy == RoutingStrategy.LEAST_LOADED:
            selected_agent = self._route_by_least_loaded(task, available_agents)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            selected_agent = self._route_by_round_robin(task, available_agents)
        elif strategy == RoutingStrategy.AFFINITY_BASED:
            selected_agent = self._route_by_affinity(task, available_agents)
        else:
            selected_agent = self._route_by_capability(task, available_agents)

        if selected_agent:
            self._record_routing_decision(task, selected_agent, strategy)

        self._stats["total_routes"] += 1

        return selected_agent

    def _get_available_agents(self, agents: Optional[List[SubAgent]] = None) -> List[SubAgent]:
        """Get available agents.

        Args:
            agents: Optional agent list

        Returns:
            List of available agents
        """
        if agents:
            return [a for a in agents if a.is_available]

        available = []
        for agent_id, profile in self._agent_profiles.items():
            if profile.is_available:
                continue

        return agents if agents else []

    def _route_by_capability(self, task: Task, agents: List[SubAgent]) -> Optional[SubAgent]:
        """Route based on capability matching.

        Args:
            task: Task to route
            agents: Available agents

        Returns:
            Best matching agent
        """
        required_caps = self._get_required_capabilities(task)

        scored_agents = []
        for agent in agents:
            profile = self._agent_profiles.get(agent.id)
            if not profile:
                continue

            if required_caps and not required_caps.issubset(profile.capabilities):
                continue

            score = self._calculate_capability_score(profile, required_caps)
            scored_agents.append((score, agent))

        if not scored_agents:
            self._stats["fallback_routes"] += 1
            return agents[0] if agents else None

        scored_agents.sort(key=lambda x: x[0], reverse=True)
        return scored_agents[0][1]

    def _route_by_load_balance(self, task: Task, agents: List[SubAgent]) -> Optional[SubAgent]:
        """Route based on load balancing.

        Args:
            task: Task to route
            agents: Available agents

        Returns:
            Least loaded agent
        """
        scored_agents = []
        for agent in agents:
            profile = self._agent_profiles.get(agent.id)
            if not profile:
                continue

            load_score = 1.0 - profile.load_percentage
            scored_agents.append((load_score, agent))

        if not scored_agents:
            return agents[0] if agents else None

        scored_agents.sort(key=lambda x: x[0], reverse=True)
        return scored_agents[0][1]

    def _route_by_least_loaded(self, task: Task, agents: List[SubAgent]) -> Optional[SubAgent]:
        """Route to the least loaded agent.

        Args:
            task: Task to route
            agents: Available agents

        Returns:
            Least loaded agent
        """
        min_load = float('inf')
        selected = None

        for agent in agents:
            profile = self._agent_profiles.get(agent.id)
            if not profile:
                continue

            if profile.current_load < min_load:
                min_load = profile.current_load
                selected = agent

        return selected or (agents[0] if agents else None)

    def _route_by_round_robin(self, task: Task, agents: List[SubAgent]) -> Optional[SubAgent]:
        """Route using round-robin.

        Args:
            task: Task to route
            agents: Available agents

        Returns:
            Next agent in rotation
        """
        if not agents:
            return None

        self._round_robin_index = self._round_robin_index % len(agents)
        agent = agents[self._round_robin_index]
        self._round_robin_index += 1
        return agent

    def _route_by_affinity(self, task: Task, agents: List[SubAgent]) -> Optional[SubAgent]:
        """Route based on task-agent affinity.

        Args:
            task: Task to route
            agents: Available agents

        Returns:
            Agent with highest affinity
        """
        task_type = self._classify_task(task)

        scored_agents = []
        for agent in agents:
            score = self.calculate_affinity(task, agent)
            scored_agents.append((score, agent))

        if not scored_agents:
            return agents[0] if agents else None

        scored_agents.sort(key=lambda x: x[0], reverse=True)
        return scored_agents[0][1]

    def _choose_adaptive_strategy(self, task: Task) -> RoutingStrategy:
        """Choose strategy adaptively based on task and history.

        Args:
            task: Task to route

        Returns:
            Chosen strategy
        """
        task_type = self._classify_task(task)

        if task_type in self._task_type_map:
            successful_agents = self._task_type_map[task_type]
            if successful_agents:
                return RoutingStrategy.AFFINITY_BASED

        task_lower = task.name.lower()
        if any(word in task_lower for word in ["test", "测试"]):
            return RoutingStrategy.CAPABILITY_MATCH
        elif any(word in task_lower for word in ["parallel", "并行"]):
            return RoutingStrategy.LOAD_BALANCED

        return RoutingStrategy.CAPABILITY_MATCH

    def calculate_affinity(self, task: Task, agent: SubAgent) -> float:
        """Calculate affinity between task and agent.

        Args:
            task: Task to route
            agent: Candidate agent

        Returns:
            Affinity score (0.0 - 1.0)
        """
        profile = self._agent_profiles.get(agent.id)
        if not profile:
            return 0.0

        score = 0.0

        task_type = self._classify_task(task)
        if task_type in profile.task_type_affinity:
            score += profile.task_type_affinity[task_type] * 0.4

        required_caps = self._get_required_capabilities(task)
        cap_score = self._calculate_capability_score(profile, required_caps)
        score += cap_score * 0.3

        score += profile.success_rate * 0.2

        load_score = 1.0 - profile.load_percentage
        score += load_score * 0.1

        return score

    def _calculate_capability_score(
        self,
        profile: AgentProfile,
        required_caps: Set[str]
    ) -> float:
        """Calculate capability match score.

        Args:
            profile: Agent profile
            required_caps: Required capabilities

        Returns:
            Capability score (0.0 - 1.0)
        """
        if not required_caps:
            return 1.0

        matched = required_caps & profile.capabilities
        return len(matched) / len(required_caps)

    def _get_required_capabilities(self, task: Task) -> Set[str]:
        """Get required capabilities for a task.

        Args:
            task: Task to analyze

        Returns:
            Set of required capability names
        """
        caps = set()

        if "required_capabilities" in task.input_data:
            caps.update(task.input_data["required_capabilities"])

        task_lower = f"{task.name} {task.description}".lower()

        if any(word in task_lower for word in ["test", "测试"]):
            caps.add("test_generation")
        if any(word in task_lower for word in ["fix", "bug", "修复"]):
            caps.add("bug_fixing")
        if any(word in task_lower for word in ["refactor", "重构"]):
            caps.add("refactoring")
        if any(word in task_lower for word in ["doc", "文档"]):
            caps.add("documentation")
        if any(word in task_lower for word in ["analyze", "分析"]):
            caps.add("analysis")

        return caps

    def _classify_task(self, task: Task) -> str:
        """Classify task type.

        Args:
            task: Task to classify

        Returns:
            Task type string
        """
        task_lower = f"{task.name} {task.description}".lower()

        if any(word in task_lower for word in ["test", "测试"]):
            return "testing"
        elif any(word in task_lower for word in ["fix", "bug", "修复"]):
            return "fixing"
        elif any(word in task_lower for word in ["refactor", "重构"]):
            return "refactoring"
        elif any(word in task_lower for word in ["doc", "文档"]):
            return "documentation"
        elif any(word in task_lower for word in ["analyze", "分析"]):
            return "analysis"
        else:
            return "general"

    def _record_routing_decision(
        self,
        task: Task,
        agent: SubAgent,
        strategy: RoutingStrategy
    ) -> None:
        """Record a routing decision.

        Args:
            task: Routed task
            agent: Selected agent
            strategy: Strategy used
        """
        profile = self._agent_profiles.get(agent.id)

        score = RoutingScore(
            agent_id=agent.id,
            total_score=self.calculate_affinity(task, agent)
        )

        decision = RoutingDecision(
            decision_id=str(uuid4()),
            task_id=task.id,
            task_type=self._classify_task(task),
            agent_id=agent.id,
            strategy=strategy,
            score=score
        )

        self._routing_history.append(decision)

        if len(self._routing_history) > 1000:
            self._routing_history = self._routing_history[-500:]

    def record_routing_result(
        self,
        task_id: str,
        agent_id: str,
        success: bool,
        execution_time_ms: Optional[int] = None
    ) -> None:
        """Record the result of a routing decision.

        Args:
            task_id: Task ID
            agent_id: Agent ID
            success: Whether the task succeeded
            execution_time_ms: Optional execution time
        """
        for decision in reversed(self._routing_history):
            if decision.task_id == task_id and decision.agent_id == agent_id:
                decision.success = success
                decision.execution_time_ms = execution_time_ms
                break

        if agent_id in self._agent_profiles:
            profile = self._agent_profiles[agent_id]
            profile.total_tasks += 1

            if success:
                profile.successful_tasks += 1
                self._stats["successful_routes"] += 1
            else:
                profile.failed_tasks += 1
                self._stats["failed_routes"] += 1

            if execution_time_ms:
                if profile.avg_execution_time_ms == 0:
                    profile.avg_execution_time_ms = execution_time_ms
                else:
                    profile.avg_execution_time_ms = (
                        profile.avg_execution_time_ms * 0.8 + execution_time_ms * 0.2
                    )

            for decision in reversed(self._routing_history[-50:]):
                if decision.agent_id == agent_id and decision.success is not None:
                    task_type = decision.task_type
                    if task_type not in profile.task_type_affinity:
                        profile.task_type_affinity[task_type] = 0.5

                    if decision.success:
                        profile.task_type_affinity[task_type] = min(
                            1.0, profile.task_type_affinity[task_type] + 0.1
                        )
                    else:
                        profile.task_type_affinity[task_type] = max(
                            0.0, profile.task_type_affinity[task_type] - 0.1
                        )

        if self.learning_enabled:
            self._update_task_type_map(task_id, agent_id, success)

    def _update_task_type_map(self, task_id: str, agent_id: str, success: bool) -> None:
        """Update task type to agent mapping.

        Args:
            task_id: Task ID
            agent_id: Agent ID
            success: Whether task succeeded
        """
        for decision in reversed(self._routing_history[-20:]):
            if decision.task_id == task_id:
                task_type = decision.task_type

                if task_type not in self._task_type_map:
                    self._task_type_map[task_type] = set()

                if success:
                    self._task_type_map[task_type].add(agent_id)
                elif agent_id in self._task_type_map[task_type]:
                    self._task_type_map[task_type].discard(agent_id)
                break

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics.

        Returns:
            Statistics dictionary
        """
        total = self._stats["total_routes"]
        success_rate = (
            self._stats["successful_routes"] / total
            if total > 0 else 0
        )

        return {
            "total_routes": total,
            "successful_routes": self._stats["successful_routes"],
            "failed_routes": self._stats["failed_routes"],
            "fallback_routes": self._stats["fallback_routes"],
            "success_rate": success_rate,
            "registered_agents": len(self._agent_profiles),
            "task_types_learned": len(self._task_type_map)
        }

    def get_agent_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific agent.

        Args:
            agent_id: Agent ID

        Returns:
            Agent statistics or None
        """
        profile = self._agent_profiles.get(agent_id)
        if not profile:
            return None

        return {
            "agent_id": profile.agent_id,
            "agent_type": profile.agent_type,
            "capabilities": list(profile.capabilities),
            "current_load": profile.current_load,
            "max_load": profile.max_load,
            "load_percentage": profile.load_percentage,
            "total_tasks": profile.total_tasks,
            "successful_tasks": profile.successful_tasks,
            "failed_tasks": profile.failed_tasks,
            "success_rate": profile.success_rate,
            "avg_execution_time_ms": profile.avg_execution_time_ms,
            "task_type_affinity": profile.task_type_affinity
        }

    def get_routing_history(
        self,
        agent_id: Optional[str] = None,
        task_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get routing history.

        Args:
            agent_id: Optional filter by agent
            task_type: Optional filter by task type
            limit: Maximum results

        Returns:
            List of routing decisions
        """
        history = self._routing_history

        if agent_id:
            history = [d for d in history if d.agent_id == agent_id]
        if task_type:
            history = [d for d in history if d.task_type == task_type]

        return [
            {
                "decision_id": d.decision_id,
                "task_id": d.task_id,
                "task_type": d.task_type,
                "agent_id": d.agent_id,
                "strategy": d.strategy.value,
                "success": d.success,
                "timestamp": d.timestamp
            }
            for d in history[-limit:]
        ]


def create_task_router(
    strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
    learning_enabled: bool = True
) -> IntelligentTaskRouter:
    """Create an IntelligentTaskRouter.

    Args:
        strategy: Default routing strategy
        learning_enabled: Whether to enable learning

    Returns:
        IntelligentTaskRouter instance
    """
    return IntelligentTaskRouter(
        default_strategy=strategy,
        learning_enabled=learning_enabled
    )
