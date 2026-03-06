"""Multi-Agent Collaboration Capability (P2).

Provides multi-agent coordination for complex test generation tasks.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class MultiAgentCapability(Capability):
    """Multi-agent collaboration capability.
    
    This capability provides:
    - Agent coordination for complex tasks
    - Task distribution among specialized agents
    - Result aggregation
    - Parallel execution support
    
    Configuration:
        enable_multi_agent: Enable multi-agent mode (default: True)
        workers: Number of worker agents (default: 3)
        strategy: Task allocation strategy (default: "capability_match")
    """
    
    _coordinator: Any = None
    _message_bus: Any = None
    _enable_multi_agent: bool = True
    _workers: int = 3
    _strategy: str = "capability_match"
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="multi_agent",
            description="Multi-agent collaboration for complex tasks",
            priority=CapabilityPriority.NORMAL,
            provides={"agent_coordinator", "message_bus"},
            dependencies={"context_management"},
            tags={"p2", "collaboration", "distributed"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize multi-agent components.
        
        Args:
            container: Dependency injection container
        """
        self._container = container
        
        try:
            from ..multi_agent import AgentCoordinator, MessageBus
            
            config = self._get_config()
            self._enable_multi_agent = config.get("enable_multi_agent", True)
            self._workers = config.get("workers", 3)
            self._strategy = config.get("strategy", "capability_match")
            
            if self._enable_multi_agent:
                self._message_bus = MessageBus()
                self._coordinator = AgentCoordinator(
                    message_bus=self._message_bus,
                    workers=self._workers,
                    strategy=self._strategy
                )
            
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info(
                f"[MultiAgentCapability] Initialized - "
                f"enabled={self._enable_multi_agent}, workers={self._workers}"
            )
            
        except Exception as e:
            self._set_error(e)
            raise
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration from container or defaults."""
        if self._container:
            try:
                settings = self._container.resolve(type('Settings', (), {}))
                if hasattr(settings, 'multi_agent'):
                    return {
                        "enable_multi_agent": getattr(
                            settings.multi_agent, 'enabled', True
                        ),
                        "workers": getattr(
                            settings.multi_agent, 'workers', 3
                        ),
                        "strategy": getattr(
                            settings.multi_agent, 'strategy', 'capability_match'
                        ),
                    }
            except Exception:
                pass
        return {}
    
    @property
    def coordinator(self) -> Any:
        """Get the coordinator instance."""
        return self._coordinator
    
    @property
    def message_bus(self) -> Any:
        """Get the message bus instance."""
        return self._message_bus
    
    async def submit_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 1,
        dependencies: List[str] = None
    ) -> str:
        """Submit a task for multi-agent processing.
        
        Args:
            task_type: Type of task
            payload: Task payload
            priority: Task priority
            dependencies: Task dependencies
            
        Returns:
            Task ID
        """
        if not self._coordinator:
            return None
        
        return await self._coordinator.submit_task(
            task_type=task_type,
            payload=payload,
            priority=priority,
            dependencies=dependencies or []
        )
    
    async def wait_for_task(self, task_id: str, timeout: float = 60.0) -> bool:
        """Wait for a task to complete.
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            True if task completed successfully
        """
        if not self._coordinator:
            return False
        
        return await self._coordinator.wait_for_task(task_id, timeout)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status dictionary
        """
        if not self._coordinator:
            return {"status": "unavailable"}
        
        return self._coordinator.get_task_status(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get multi-agent statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self._coordinator:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "coordinator": self._coordinator.get_stats() if hasattr(self._coordinator, 'get_stats') else {},
            "message_bus": self._message_bus.get_stats() if self._message_bus and hasattr(self._message_bus, 'get_stats') else {}
        }
    
    async def start(self) -> None:
        """Start the multi-agent system."""
        if self._coordinator and hasattr(self._coordinator, 'start'):
            await self._coordinator.start()
            logger.info("[MultiAgentCapability] Multi-agent system started")
    
    async def stop(self) -> None:
        """Stop the multi-agent system."""
        if self._coordinator and hasattr(self._coordinator, 'stop'):
            await self._coordinator.stop()
            logger.info("[MultiAgentCapability] Multi-agent system stopped")
    
    def shutdown(self) -> None:
        """Shutdown multi-agent components."""
        if self._coordinator:
            self._coordinator = None
        if self._message_bus:
            self._message_bus = None
        super().shutdown()
