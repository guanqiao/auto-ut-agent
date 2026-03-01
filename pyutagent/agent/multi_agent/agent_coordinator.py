"""Agent coordinator for multi-agent collaboration.

Central coordinator that manages task allocation and agent collaboration.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable

from .message_bus import MessageBus, AgentMessage, MessageType
from .shared_knowledge import SharedKnowledgeBase, ExperienceReplay
from .specialized_agent import SpecializedAgent, AgentCapability, AgentTask, TaskResult

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles that agents can have in the system."""
    COORDINATOR = auto()
    DESIGNER = auto()
    IMPLEMENTER = auto()
    REVIEWER = auto()
    FIXER = auto()
    ANALYZER = auto()


class TaskAllocation(Enum):
    """Task allocation strategies."""
    ROUND_ROBIN = auto()
    CAPABILITY_MATCH = auto()
    LOAD_BALANCED = auto()
    PRIORITY_BASED = auto()


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    agent_id: str
    capabilities: Set[AgentCapability]
    role: AgentRole
    status: str = "idle"
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: Optional[str] = None
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TaskInfo:
    """Information about a task."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    status: str = "pending"
    assigned_agent: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[TaskResult] = None
    dependencies: List[str] = field(default_factory=list)
    priority: int = 5


class AgentCoordinator:
    """Central coordinator for multi-agent collaboration.
    
    Responsibilities:
    - Agent registration and lifecycle management
    - Task allocation and scheduling
    - Progress monitoring
    - Result aggregation
    - Conflict resolution
    """
    
    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        knowledge_base: Optional[SharedKnowledgeBase] = None,
        experience_replay: Optional[ExperienceReplay] = None,
        allocation_strategy: TaskAllocation = TaskAllocation.CAPABILITY_MATCH
    ):
        """Initialize agent coordinator.
        
        Args:
            message_bus: Message bus for communication
            knowledge_base: Shared knowledge base
            experience_replay: Experience replay buffer
            allocation_strategy: Task allocation strategy
        """
        self.message_bus = message_bus or MessageBus()
        self.knowledge_base = knowledge_base or SharedKnowledgeBase()
        self.experience_replay = experience_replay or ExperienceReplay()
        self.allocation_strategy = allocation_strategy
        
        self.agents: Dict[str, AgentInfo] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._stop_requested = False
        self._coordinator_id = f"coordinator_{uuid.uuid4().hex[:8]}"
        
        # Statistics
        self.stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "agents_registered": 0
        }
        
        logger.info(f"[AgentCoordinator:{self._coordinator_id}] Initialized")
    
    async def start(self):
        """Start the coordinator."""
        logger.info(f"[AgentCoordinator:{self._coordinator_id}] Starting")
        
        # Register self with message bus
        await self.message_bus.register_agent(self._coordinator_id)
        
        # Subscribe to task results and heartbeats
        await self.message_bus.subscribe(self._coordinator_id, MessageType.TASK_RESULT)
        await self.message_bus.subscribe(self._coordinator_id, MessageType.TASK_FAILED)
        await self.message_bus.subscribe(self._coordinator_id, MessageType.HEARTBEAT)
        await self.message_bus.subscribe(self._coordinator_id, MessageType.KNOWLEDGE_SHARE)
        
        # Start processing loops
        asyncio.create_task(self._message_loop())
        asyncio.create_task(self._task_scheduler_loop())
        asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop the coordinator."""
        logger.info(f"[AgentCoordinator:{self._coordinator_id}] Stopping")
        self._stop_requested = True
        await self.message_bus.unregister_agent(self._coordinator_id)
    
    async def _message_loop(self):
        """Main message processing loop."""
        while not self._stop_requested:
            try:
                message = await self.message_bus.receive(self._coordinator_id, timeout=1.0)
                if message:
                    await self._handle_message(message)
            except Exception as e:
                logger.exception(f"[AgentCoordinator:{self._coordinator_id}] Message handling error: {e}")
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming message.
        
        Args:
            message: Received message
        """
        if message.message_type == MessageType.TASK_RESULT:
            await self._handle_task_result(message)
        elif message.message_type == MessageType.TASK_FAILED:
            await self._handle_task_failure(message)
        elif message.message_type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(message)
        elif message.message_type == MessageType.KNOWLEDGE_SHARE:
            await self._handle_knowledge_share(message)
    
    async def _handle_task_result(self, message: AgentMessage):
        """Handle task completion result.
        
        Args:
            message: Task result message
        """
        payload = message.payload
        task_id = payload.get("task_id")
        
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = "completed"
            task.completed_at = datetime.now().isoformat()
            task.result = TaskResult(
                success=payload.get("success", False),
                task_id=task_id,
                agent_id=message.sender_id,
                output=payload.get("output"),
                execution_time_ms=payload.get("execution_time_ms", 0),
                error_message=payload.get("error_message")
            )
            
            # Update agent stats
            if message.sender_id in self.agents:
                agent = self.agents[message.sender_id]
                agent.tasks_completed += 1
                agent.current_task = None
                agent.status = "idle"
            
            self.stats["tasks_completed"] += 1
            
            logger.info(f"[AgentCoordinator:{self._coordinator_id}] Task completed: {task_id} "
                       f"by {message.sender_id}")
    
    async def _handle_task_failure(self, message: AgentMessage):
        """Handle task failure.
        
        Args:
            message: Task failure message
        """
        payload = message.payload
        task_id = payload.get("task_id")
        
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = "failed"
            task.completed_at = datetime.now().isoformat()
            task.result = TaskResult(
                success=False,
                task_id=task_id,
                agent_id=message.sender_id,
                output=None,
                execution_time_ms=payload.get("execution_time_ms", 0),
                error_message=payload.get("error_message")
            )
            
            # Update agent stats
            if message.sender_id in self.agents:
                agent = self.agents[message.sender_id]
                agent.tasks_failed += 1
                agent.current_task = None
                agent.status = "idle"
            
            self.stats["tasks_failed"] += 1
            
            logger.warning(f"[AgentCoordinator:{self._coordinator_id}] Task failed: {task_id} "
                          f"by {message.sender_id}")
            
            # Retry logic could be implemented here
    
    async def _handle_heartbeat(self, message: AgentMessage):
        """Handle agent heartbeat.
        
        Args:
            message: Heartbeat message
        """
        agent_id = message.sender_id
        payload = message.payload
        
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = payload.get("status", "unknown")
            agent.current_task = payload.get("current_task")
            agent.last_heartbeat = datetime.now().isoformat()
    
    async def _handle_knowledge_share(self, message: AgentMessage):
        """Handle knowledge sharing message.
        
        Args:
            message: Knowledge share message
        """
        # Knowledge is already stored in shared knowledge base
        # This handler can be used for additional processing
        logger.debug(f"[AgentCoordinator:{self._coordinator_id}] Knowledge shared by "
                    f"{message.sender_id}")
    
    async def _task_scheduler_loop(self):
        """Task scheduling loop."""
        while not self._stop_requested:
            try:
                # Get next task from queue
                priority, task_id = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    # Check dependencies
                    if not self._check_dependencies(task):
                        # Re-queue if dependencies not met
                        await self.task_queue.put((priority + 1, task_id))
                        continue
                    
                    # Allocate task to agent
                    agent_id = self._allocate_task(task)
                    
                    if agent_id:
                        await self._assign_task(task, agent_id)
                    else:
                        # No suitable agent, re-queue
                        await self.task_queue.put((priority + 1, task_id))
                        logger.warning(f"[AgentCoordinator:{self._coordinator_id}] "
                                      f"No agent available for task {task_id}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"[AgentCoordinator:{self._coordinator_id}] "
                                f"Task scheduling error: {e}")
    
    def _check_dependencies(self, task: TaskInfo) -> bool:
        """Check if task dependencies are satisfied.
        
        Args:
            task: Task to check
            
        Returns:
            True if all dependencies are completed
        """
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != "completed":
                    return False
            else:
                return False
        return True
    
    def _allocate_task(self, task: TaskInfo) -> Optional[str]:
        """Allocate task to an agent based on strategy.
        
        Args:
            task: Task to allocate
            
        Returns:
            Agent ID or None if no suitable agent
        """
        if self.allocation_strategy == TaskAllocation.ROUND_ROBIN:
            return self._allocate_round_robin(task)
        elif self.allocation_strategy == TaskAllocation.CAPABILITY_MATCH:
            return self._allocate_capability_match(task)
        elif self.allocation_strategy == TaskAllocation.LOAD_BALANCED:
            return self._allocate_load_balanced(task)
        elif self.allocation_strategy == TaskAllocation.PRIORITY_BASED:
            return self._allocate_priority_based(task)
        else:
            return self._allocate_capability_match(task)
    
    def _allocate_round_robin(self, task: TaskInfo) -> Optional[str]:
        """Allocate task using round-robin.
        
        Args:
            task: Task to allocate
            
        Returns:
            Agent ID
        """
        idle_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.status == "idle"
        ]
        
        if idle_agents:
            # Simple round-robin based on task count
            return min(idle_agents, key=lambda a: self.agents[a].tasks_completed)
        
        return None
    
    def _allocate_capability_match(self, task: TaskInfo) -> Optional[str]:
        """Allocate task based on capability matching.
        
        Args:
            task: Task to allocate
            
        Returns:
            Agent ID
        """
        required_capabilities = self._get_required_capabilities(task.task_type)
        
        matching_agents = [
            (agent_id, agent) for agent_id, agent in self.agents.items()
            if agent.status == "idle" and required_capabilities <= agent.capabilities
        ]
        
        if matching_agents:
            # Select agent with most matching capabilities and least load
            agent_id, _ = min(
                matching_agents,
                key=lambda x: (x[1].tasks_completed + x[1].tasks_failed)
            )
            return agent_id
        
        return None
    
    def _allocate_load_balanced(self, task: TaskInfo) -> Optional[str]:
        """Allocate task based on load balancing.
        
        Args:
            task: Task to allocate
            
        Returns:
            Agent ID
        """
        required_capabilities = self._get_required_capabilities(task.task_type)
        
        matching_agents = [
            (agent_id, agent) for agent_id, agent in self.agents.items()
            if agent.status == "idle" and required_capabilities <= agent.capabilities
        ]
        
        if matching_agents:
            # Select agent with least active tasks
            agent_id, _ = min(
                matching_agents,
                key=lambda x: x[1].tasks_completed + x[1].tasks_failed
            )
            return agent_id
        
        return None
    
    def _allocate_priority_based(self, task: TaskInfo) -> Optional[str]:
        """Allocate task based on priority.
        
        Args:
            task: Task to allocate
            
        Returns:
            Agent ID
        """
        # Similar to capability match but considers task priority
        return self._allocate_capability_match(task)
    
    def _get_required_capabilities(self, task_type: str) -> Set[AgentCapability]:
        """Get required capabilities for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Set of required capabilities
        """
        capability_map = {
            "design_tests": {AgentCapability.TEST_DESIGN},
            "implement_tests": {AgentCapability.TEST_IMPLEMENTATION},
            "review_tests": {AgentCapability.TEST_REVIEW},
            "fix_errors": {AgentCapability.ERROR_FIXING},
            "analyze_coverage": {AgentCapability.COVERAGE_ANALYSIS},
            "generate_mocks": {AgentCapability.MOCK_GENERATION},
            "analyze_dependencies": {AgentCapability.DEPENDENCY_ANALYSIS}
        }
        
        return capability_map.get(task_type, set())
    
    async def _assign_task(self, task: TaskInfo, agent_id: str):
        """Assign task to an agent.
        
        Args:
            task: Task to assign
            agent_id: Target agent ID
        """
        task.status = "assigned"
        task.assigned_agent = agent_id
        task.started_at = datetime.now().isoformat()
        
        # Update agent status
        if agent_id in self.agents:
            self.agents[agent_id].status = "working"
            self.agents[agent_id].current_task = task.task_id
        
        # Send task assignment message
        message = AgentMessage.create(
            sender_id=self._coordinator_id,
            recipient_id=agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            payload={
                "task_id": task.task_id,
                "task_type": task.task_type,
                "data": task.payload,
                "priority": task.priority,
                "dependencies": task.dependencies
            }
        )
        
        await self.message_bus.send(message)
        
        logger.info(f"[AgentCoordinator:{self._coordinator_id}] Task {task.task_id} "
                   f"assigned to {agent_id}")
    
    async def _monitor_loop(self):
        """Monitoring loop for agent health."""
        while not self._stop_requested:
            await asyncio.sleep(60)  # Check every minute
            
            current_time = datetime.now()
            
            for agent_id, agent in self.agents.items():
                if agent.last_heartbeat:
                    last_beat = datetime.fromisoformat(agent.last_heartbeat)
                    seconds_since_heartbeat = (current_time - last_beat).total_seconds()
                    
                    if seconds_since_heartbeat > 120:  # 2 minutes
                        logger.warning(f"[AgentCoordinator:{self._coordinator_id}] "
                                      f"Agent {agent_id} appears unresponsive")
                        agent.status = "unresponsive"
    
    def register_agent(
        self,
        agent_id: str,
        capabilities: Set[AgentCapability],
        role: AgentRole
    ) -> bool:
        """Register an agent with the coordinator.
        
        Args:
            agent_id: Agent identifier
            capabilities: Agent capabilities
            role: Agent role
            
        Returns:
            True if registered successfully
        """
        if agent_id in self.agents:
            logger.warning(f"[AgentCoordinator:{self._coordinator_id}] "
                          f"Agent {agent_id} already registered")
            return False
        
        self.agents[agent_id] = AgentInfo(
            agent_id=agent_id,
            capabilities=capabilities,
            role=role
        )
        
        self.stats["agents_registered"] += 1
        
        logger.info(f"[AgentCoordinator:{self._coordinator_id}] Agent registered: {agent_id} "
                   f"(role={role.name}, capabilities={[c.name for c in capabilities]})")
        return True
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent.
        
        Args:
            agent_id: Agent to unregister
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"[AgentCoordinator:{self._coordinator_id}] Agent unregistered: {agent_id}")
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """Submit a new task.
        
        Args:
            task_type: Type of task
            payload: Task payload
            priority: Task priority (1-10, lower is higher)
            dependencies: List of task IDs this task depends on
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.tasks[task_id] = task
        
        # Add to queue
        await self.task_queue.put((priority, task_id))
        
        self.stats["tasks_created"] += 1
        
        logger.info(f"[AgentCoordinator:{self._coordinator_id}] Task submitted: {task_id} "
                   f"(type={task_type}, priority={priority})")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status,
                "assigned_agent": task.assigned_agent,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "result": {
                    "success": task.result.success if task.result else None,
                    "execution_time_ms": task.result.execution_time_ms if task.result else None
                } if task.result else None
            }
        return None
    
    def get_agent_status(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get agent status.
        
        Args:
            agent_id: Specific agent ID or None for all agents
            
        Returns:
            Agent status information
        """
        if agent_id:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                return {
                    "agent_id": agent.agent_id,
                    "role": agent.role.name,
                    "status": agent.status,
                    "capabilities": [c.name for c in agent.capabilities],
                    "tasks_completed": agent.tasks_completed,
                    "tasks_failed": agent.tasks_failed,
                    "current_task": agent.current_task
                }
            return {}
        
        return {
            agent_id: {
                "role": agent.role.name,
                "status": agent.status,
                "tasks_completed": agent.tasks_completed,
                "tasks_failed": agent.tasks_failed
            }
            for agent_id, agent in self.agents.items()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "active_agents": len(self.agents),
            "pending_tasks": self.task_queue.qsize(),
            "total_tasks": len(self.tasks),
            "allocation_strategy": self.allocation_strategy.name
        }
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """Wait for a task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout: Optional timeout in seconds
            
        Returns:
            True if task completed successfully
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == "completed":
                    return task.result.success if task.result else False
                elif task.status == "failed":
                    return False
            
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                return False
            
            await asyncio.sleep(0.1)
