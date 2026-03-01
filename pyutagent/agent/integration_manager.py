"""Integration Manager for coordinating all enhancement components.

This module provides a centralized integration layer that:
- Manages component lifecycle
- Handles cross-component communication
- Provides unified configuration
- Monitors system health
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

from .enhanced_agent import EnhancedAgent, EnhancedAgentConfig
from .multi_agent import AgentCoordinator, MessageBus, SharedKnowledgeBase, ExperienceReplay
from ..core.metrics import MetricsCollector, get_metrics
from ..core.container import Container, get_container

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of a managed component."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    ERROR = auto()
    STOPPED = auto()


@dataclass
class ComponentInfo:
    """Information about a managed component."""
    name: str
    component_type: str
    status: ComponentStatus
    instance: Optional[Any] = None
    dependencies: List[str] = field(default_factory=list)
    init_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    last_heartbeat: Optional[str] = None


class IntegrationManager:
    """Central manager for all enhancement components.
    
    Responsibilities:
    - Component lifecycle management
    - Dependency resolution
    - Health monitoring
    - Configuration management
    - Cross-component event routing
    """
    
    def __init__(
        self,
        project_path: str,
        config: Optional[EnhancedAgentConfig] = None,
        container: Optional[Container] = None
    ):
        """Initialize integration manager.
        
        Args:
            project_path: Project path
            config: Enhanced agent configuration
            container: DI container
        """
        self.project_path = Path(project_path)
        self.config = config or EnhancedAgentConfig()
        self.container = container or get_container()
        
        # Component registry
        self._components: Dict[str, ComponentInfo] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Metrics
        self.metrics = get_metrics()
        
        # Health check
        self._health_check_task: Optional[asyncio.Task] = None
        self._stop_requested = False
        
        logger.info(f"[IntegrationManager] Initialized for project: {project_path}")
    
    async def initialize_all(self) -> bool:
        """Initialize all components in dependency order.
        
        Returns:
            True if all components initialized successfully
        """
        logger.info("[IntegrationManager] Initializing all components")
        
        try:
            # Initialize in dependency order
            init_order = [
                "message_bus",
                "shared_knowledge",
                "experience_replay",
                "metrics_collector",
                "agent_coordinator",
                "enhanced_agent"
            ]
            
            for component_name in init_order:
                if not await self._initialize_component(component_name):
                    logger.error(f"[IntegrationManager] Failed to initialize {component_name}")
                    return False
            
            # Start health check loop
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("[IntegrationManager] All components initialized successfully")
            return True
            
        except Exception as e:
            logger.exception(f"[IntegrationManager] Initialization failed: {e}")
            return False
    
    async def _initialize_component(self, name: str) -> bool:
        """Initialize a specific component.
        
        Args:
            name: Component name
            
        Returns:
            True if successful
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            self._update_component_status(name, ComponentStatus.INITIALIZING)
            
            if name == "message_bus":
                instance = MessageBus()
                
            elif name == "shared_knowledge":
                instance = SharedKnowledgeBase()
                
            elif name == "experience_replay":
                instance = ExperienceReplay()
                
            elif name == "metrics_collector":
                instance = self.metrics
                
            elif name == "agent_coordinator":
                if not self.config.enable_multi_agent:
                    logger.info("[IntegrationManager] Multi-agent disabled, skipping coordinator")
                    return True
                    
                message_bus = self.get_component("message_bus")
                knowledge_base = self.get_component("shared_knowledge")
                experience_replay = self.get_component("experience_replay")
                
                instance = AgentCoordinator(
                    message_bus=message_bus,
                    knowledge_base=knowledge_base,
                    experience_replay=experience_replay
                )
                await instance.start()
                
            elif name == "enhanced_agent":
                # EnhancedAgent is initialized separately with required dependencies
                logger.info("[IntegrationManager] EnhancedAgent will be initialized separately")
                return True
                
            else:
                logger.warning(f"[IntegrationManager] Unknown component: {name}")
                return False
            
            init_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            
            self._register_component(name, instance, init_time)
            self._update_component_status(name, ComponentStatus.READY)
            
            logger.info(f"[IntegrationManager] Component {name} initialized in {init_time}ms")
            return True
            
        except Exception as e:
            init_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
            self._update_component_status(name, ComponentStatus.ERROR, error=str(e))
            logger.exception(f"[IntegrationManager] Failed to initialize {name}: {e}")
            return False
    
    def _register_component(self, name: str, instance: Any, init_time_ms: int):
        """Register a component.
        
        Args:
            name: Component name
            instance: Component instance
            init_time_ms: Initialization time in milliseconds
        """
        component_type = type(instance).__name__
        
        self._components[name] = ComponentInfo(
            name=name,
            component_type=component_type,
            status=ComponentStatus.READY,
            instance=instance,
            init_time_ms=init_time_ms,
            last_heartbeat=datetime.now().isoformat()
        )
        
        # Register in container for dependency injection
        self.container.register(instance.__class__, lambda: instance)
    
    def _update_component_status(
        self,
        name: str,
        status: ComponentStatus,
        error: Optional[str] = None
    ):
        """Update component status.
        
        Args:
            name: Component name
            status: New status
            error: Optional error message
        """
        if name in self._components:
            self._components[name].status = status
            self._components[name].last_heartbeat = datetime.now().isoformat()
            if error:
                self._components[name].error_message = error
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None
        """
        if name in self._components:
            return self._components[name].instance
        return None
    
    def get_component_status(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get component status.
        
        Args:
            name: Specific component or None for all
            
        Returns:
            Status information
        """
        if name:
            if name in self._components:
                info = self._components[name]
                return {
                    "name": info.name,
                    "type": info.component_type,
                    "status": info.status.name,
                    "init_time_ms": info.init_time_ms,
                    "error": info.error_message
                }
            return {}
        
        return {
            name: {
                "type": info.component_type,
                "status": info.status.name,
                "init_time_ms": info.init_time_ms
            }
            for name, info in self._components.items()
        }
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while not self._stop_requested:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            for name, info in self._components.items():
                if info.status == ComponentStatus.ERROR:
                    logger.warning(f"[IntegrationManager] Component {name} in ERROR state")
                
                # Update heartbeat
                if info.instance:
                    info.last_heartbeat = datetime.now().isoformat()
    
    async def shutdown(self):
        """Shutdown all components gracefully."""
        logger.info("[IntegrationManager] Shutting down all components")
        
        self._stop_requested = True
        
        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
        
        # Shutdown in reverse order
        shutdown_order = [
            "enhanced_agent",
            "agent_coordinator",
            "experience_replay",
            "shared_knowledge",
            "message_bus"
        ]
        
        for name in shutdown_order:
            if name in self._components:
                info = self._components[name]
                if info.instance and hasattr(info.instance, 'stop'):
                    try:
                        if asyncio.iscoroutinefunction(info.instance.stop):
                            await info.instance.stop()
                        else:
                            info.instance.stop()
                        
                        info.status = ComponentStatus.STOPPED
                        logger.info(f"[IntegrationManager] Component {name} stopped")
                    except Exception as e:
                        logger.exception(f"[IntegrationManager] Error stopping {name}: {e}")
        
        logger.info("[IntegrationManager] Shutdown complete")
    
    def subscribe_event(self, event_type: str, handler: Callable):
        """Subscribe to an event type.
        
        Args:
            event_type: Type of event
            handler: Event handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        logger.debug(f"[IntegrationManager] Handler subscribed to {event_type}")
    
    async def publish_event(self, event_type: str, data: Dict[str, Any]):
        """Publish an event to all subscribers.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.exception(f"[IntegrationManager] Event handler error: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health.
        
        Returns:
            Health status dictionary
        """
        total_components = len(self._components)
        ready_components = sum(
            1 for info in self._components.values()
            if info.status in (ComponentStatus.READY, ComponentStatus.RUNNING)
        )
        error_components = sum(
            1 for info in self._components.values()
            if info.status == ComponentStatus.ERROR
        )
        
        health_score = ready_components / total_components if total_components > 0 else 0
        
        return {
            "overall_health": "healthy" if health_score == 1.0 else "degraded" if health_score > 0.5 else "unhealthy",
            "health_score": health_score,
            "total_components": total_components,
            "ready_components": ready_components,
            "error_components": error_components,
            "components": self.get_component_status()
        }
    
    def create_enhanced_agent(
        self,
        llm_client: Any,
        working_memory: Any,
        progress_callback: Optional[Callable] = None
    ) -> EnhancedAgent:
        """Create an enhanced agent with all dependencies.
        
        Args:
            llm_client: LLM client
            working_memory: Working memory
            progress_callback: Progress callback
            
        Returns:
            EnhancedAgent instance
        """
        agent = EnhancedAgent(
            llm_client=llm_client,
            working_memory=working_memory,
            project_path=str(self.project_path),
            progress_callback=progress_callback,
            container=self.container,
            config=self.config
        )
        
        # Register as component
        self._register_component("enhanced_agent", agent, 0)
        self._update_component_status("enhanced_agent", ComponentStatus.RUNNING)
        
        return agent


# Global instance
_integration_manager: Optional[IntegrationManager] = None


def get_integration_manager(
    project_path: Optional[str] = None,
    config: Optional[EnhancedAgentConfig] = None
) -> IntegrationManager:
    """Get or create the global integration manager.
    
    Args:
        project_path: Project path (required for first call)
        config: Optional configuration
        
    Returns:
        IntegrationManager instance
    """
    global _integration_manager
    
    if _integration_manager is None:
        if project_path is None:
            raise ValueError("project_path is required for first initialization")
        
        _integration_manager = IntegrationManager(project_path, config)
    
    return _integration_manager


def reset_integration_manager():
    """Reset the global integration manager."""
    global _integration_manager
    _integration_manager = None
