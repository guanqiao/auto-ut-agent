"""Specialized agent base class for multi-agent collaboration.

Provides base functionality for specialized agents with specific capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Callable

from .message_bus import MessageBus, AgentMessage, MessageType
from .shared_knowledge import SharedKnowledgeBase, ExperienceReplay

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Capabilities that specialized agents can have."""
    TEST_DESIGN = auto()           # Design test cases
    TEST_IMPLEMENTATION = auto()   # Implement test code
    TEST_REVIEW = auto()           # Review test quality
    ERROR_FIXING = auto()          # Fix compilation/test errors
    COVERAGE_ANALYSIS = auto()     # Analyze and improve coverage
    MOCK_GENERATION = auto()       # Generate mocks and stubs
    DEPENDENCY_ANALYSIS = auto()   # Analyze code dependencies


class AgentStatus(Enum):
    """Status of a specialized agent."""
    IDLE = auto()
    WORKING = auto()
    WAITING = auto()
    ERROR = auto()
    STOPPED = auto()


@dataclass
class TaskResult:
    """Result of a task execution."""
    success: bool
    task_id: str
    agent_id: str
    output: Any
    execution_time_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task assigned to an agent."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 5
    assigned_at: Optional[str] = None
    deadline: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class SpecializedAgent(ABC):
    """Base class for specialized agents.
    
    Specialized agents have specific capabilities and collaborate
    through the message bus and shared knowledge base.
    """
    
    def __init__(
        self,
        agent_id: str,
        capabilities: Set[AgentCapability],
        message_bus: MessageBus,
        knowledge_base: SharedKnowledgeBase,
        experience_replay: Optional[ExperienceReplay] = None
    ):
        """Initialize specialized agent.
        
        Args:
            agent_id: Unique agent identifier
            capabilities: Set of agent capabilities
            message_bus: Message bus for communication
            knowledge_base: Shared knowledge base
            experience_replay: Optional experience replay buffer
        """
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.message_bus = message_bus
        self.knowledge_base = knowledge_base
        self.experience_replay = experience_replay
        
        self.status = AgentStatus.IDLE
        self.current_task: Optional[AgentTask] = None
        self.task_history: List[TaskResult] = []
        self._stop_requested = False
        self._message_queue: Optional[asyncio.Queue] = None
        self._task_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info(f"[SpecializedAgent:{agent_id}] Initialized with capabilities: "
                   f"{[c.name for c in capabilities]}")
    
    async def start(self):
        """Start the agent's message processing loop."""
        logger.info(f"[SpecializedAgent:{self.agent_id}] Starting")
        
        # Register with message bus
        self._message_queue = await self.message_bus.register_agent(self.agent_id)
        
        # Subscribe to relevant message types
        await self.message_bus.subscribe(self.agent_id, MessageType.TASK_ASSIGNMENT)
        await self.message_bus.subscribe(self.agent_id, MessageType.COORDINATION)
        await self.message_bus.subscribe(self.agent_id, MessageType.QUERY)
        
        # Start message processing
        asyncio.create_task(self._message_loop())
        asyncio.create_task(self._task_loop())
        
        # Send heartbeat
        await self._send_heartbeat()
    
    async def stop(self):
        """Stop the agent gracefully."""
        logger.info(f"[SpecializedAgent:{self.agent_id}] Stopping")
        self._stop_requested = True
        self.status = AgentStatus.STOPPED
        await self.message_bus.unregister_agent(self.agent_id)
    
    async def _message_loop(self):
        """Main message processing loop."""
        while not self._stop_requested:
            try:
                message = await self.message_bus.receive(self.agent_id, timeout=1.0)
                if message:
                    await self._handle_message(message)
            except Exception as e:
                logger.exception(f"[SpecializedAgent:{self.agent_id}] Message handling error: {e}")
    
    async def _task_loop(self):
        """Task execution loop."""
        while not self._stop_requested:
            try:
                # Wait for tasks
                task = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)
                await self._execute_task(task)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"[SpecializedAgent:{self.agent_id}] Task execution error: {e}")
    
    async def _handle_message(self, message: AgentMessage):
        """Handle incoming message.
        
        Args:
            message: Received message
        """
        logger.debug(f"[SpecializedAgent:{self.agent_id}] Received {message.message_type.name} "
                    f"from {message.sender_id}")
        
        if message.message_type == MessageType.TASK_ASSIGNMENT:
            await self._handle_task_assignment(message)
        elif message.message_type == MessageType.COORDINATION:
            await self._handle_coordination(message)
        elif message.message_type == MessageType.QUERY:
            await self._handle_query(message)
        elif message.message_type == MessageType.ERROR:
            await self._handle_error_message(message)
    
    async def _handle_task_assignment(self, message: AgentMessage):
        """Handle task assignment message.
        
        Args:
            message: Task assignment message
        """
        payload = message.payload
        task = AgentTask(
            task_id=payload.get("task_id", ""),
            task_type=payload.get("task_type", ""),
            payload=payload.get("data", {}),
            priority=payload.get("priority", 5),
            assigned_at=datetime.now().isoformat(),
            dependencies=payload.get("dependencies", [])
        )
        
        # Add to task queue
        await self._task_queue.put(task)
        
        logger.info(f"[SpecializedAgent:{self.agent_id}] Task assigned: {task.task_id}")
    
    async def _handle_coordination(self, message: AgentMessage):
        """Handle coordination message.
        
        Args:
            message: Coordination message
        """
        # Override in subclasses for specific coordination logic
        pass
    
    async def _handle_query(self, message: AgentMessage):
        """Handle query message.
        
        Args:
            message: Query message
        """
        # Override in subclasses for specific query handling
        pass
    
    async def _handle_error_message(self, message: AgentMessage):
        """Handle error message.
        
        Args:
            message: Error message
        """
        logger.warning(f"[SpecializedAgent:{self.agent_id}] Received error from "
                      f"{message.sender_id}: {message.payload.get('error', 'Unknown')}")
    
    async def _execute_task(self, task: AgentTask):
        """Execute a task.
        
        Args:
            task: Task to execute
        """
        self.current_task = task
        self.status = AgentStatus.WORKING
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Execute the specific task
            result = await self.execute_task(task)
            
            execution_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            task_result = TaskResult(
                success=result.get("success", False),
                task_id=task.task_id,
                agent_id=self.agent_id,
                output=result.get("output"),
                execution_time_ms=execution_time_ms,
                error_message=result.get("error"),
                metadata=result.get("metadata", {})
            )
            
            self.task_history.append(task_result)
            
            # Send result back
            await self._send_task_result(task_result)
            
            # Record experience if replay buffer available
            if self.experience_replay:
                self.experience_replay.add_experience(
                    task_type=task.task_type,
                    context=task.payload,
                    action="execute_task",
                    outcome="success" if task_result.success else "failure",
                    reward=1.0 if task_result.success else -0.5,
                    agent_id=self.agent_id
                )
            
            logger.info(f"[SpecializedAgent:{self.agent_id}] Task completed: {task.task_id} "
                       f"(success={task_result.success})")
            
        except Exception as e:
            logger.exception(f"[SpecializedAgent:{self.agent_id}] Task execution failed: {e}")
            
            execution_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            task_result = TaskResult(
                success=False,
                task_id=task.task_id,
                agent_id=self.agent_id,
                output=None,
                execution_time_ms=execution_time_ms,
                error_message=str(e)
            )
            
            self.task_history.append(task_result)
            await self._send_task_result(task_result)
        
        finally:
            self.current_task = None
            self.status = AgentStatus.IDLE
    
    @abstractmethod
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a specific task.
        
        Override this method in subclasses to implement specific task logic.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        raise NotImplementedError
    
    async def _send_task_result(self, result: TaskResult):
        """Send task result back to coordinator.
        
        Args:
            result: Task result
        """
        message_type = MessageType.TASK_RESULT if result.success else MessageType.TASK_FAILED
        
        message = AgentMessage.create(
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast to coordinator
            message_type=message_type,
            payload={
                "task_id": result.task_id,
                "success": result.success,
                "output": result.output,
                "execution_time_ms": result.execution_time_ms,
                "error_message": result.error_message,
                "metadata": result.metadata
            }
        )
        
        await self.message_bus.send(message)
    
    async def _send_heartbeat(self):
        """Send heartbeat message."""
        message = AgentMessage.create(
            sender_id=self.agent_id,
            recipient_id=None,
            message_type=MessageType.HEARTBEAT,
            payload={
                "status": self.status.name,
                "capabilities": [c.name for c in self.capabilities],
                "current_task": self.current_task.task_id if self.current_task else None
            }
        )
        
        await self.message_bus.send(message)
        
        # Schedule next heartbeat
        asyncio.create_task(self._schedule_next_heartbeat())
    
    async def _schedule_next_heartbeat(self):
        """Schedule next heartbeat."""
        await asyncio.sleep(30)  # Heartbeat every 30 seconds
        if not self._stop_requested:
            await self._send_heartbeat()
    
    def share_knowledge(
        self,
        item_type: str,
        content: Dict[str, Any],
        confidence: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """Share knowledge to the shared knowledge base.
        
        Args:
            item_type: Type of knowledge
            content: Knowledge content
            confidence: Confidence level
            tags: Optional tags
            
        Returns:
            Knowledge item ID
        """
        return self.knowledge_base.add_knowledge(
            item_type=item_type,
            content=content,
            source_agent=self.agent_id,
            confidence=confidence,
            tags=tags
        )
    
    def query_knowledge(
        self,
        item_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Any]:
        """Query shared knowledge base.
        
        Args:
            item_type: Filter by type
            tags: Filter by tags
            limit: Maximum results
            
        Returns:
            List of knowledge items
        """
        return self.knowledge_base.query_knowledge(
            item_type=item_type,
            tags=tags,
            limit=limit
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "agent_id": self.agent_id,
            "status": self.status.name,
            "capabilities": [c.name for c in self.capabilities],
            "tasks_completed": len(self.task_history),
            "successful_tasks": sum(1 for r in self.task_history if r.success),
            "current_task": self.current_task.task_id if self.current_task else None
        }
