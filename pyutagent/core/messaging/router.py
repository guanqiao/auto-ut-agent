"""Message Router for routing messages to destinations.

This module provides:
- MessageRouter: Route messages based on rules
- RoutingRule: Define routing rules
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from .message import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    """A rule for routing messages.
    
    Rules can match based on:
    - Message type
    - Sender
    - Recipient
    - Custom predicate
    """
    
    name: str
    target: str
    message_types: Optional[Set[MessageType]] = None
    senders: Optional[Set[str]] = None
    recipients: Optional[Set[str]] = None
    predicate: Optional[Callable[[Message], bool]] = None
    priority: int = 0
    
    def matches(self, message: Message) -> bool:
        """Check if a message matches this rule.
        
        Args:
            message: Message to check
            
        Returns:
            True if message matches
        """
        if self.message_types and message.type not in self.message_types:
            return False
        
        if self.senders and message.sender not in self.senders:
            return False
        
        if self.recipients and message.recipient not in self.recipients:
            return False
        
        if self.predicate and not self.predicate(message):
            return False
        
        return True


class MessageRouter:
    """Routes messages to destinations based on rules.
    
    Features:
    - Rule-based routing
    - Default routing
    - Priority-based rule evaluation
    - Dynamic rule management
    """
    
    def __init__(self):
        """Initialize the message router."""
        self._routes: Dict[str, str] = {}
        self._rules: List[RoutingRule] = []
        self._default_target: Optional[str] = None
    
    def register_route(self, entity_id: str, target: str) -> None:
        """Register a direct route from entity to target.
        
        Args:
            entity_id: Source entity
            target: Target destination
        """
        self._routes[entity_id] = target
        logger.debug(f"[MessageRouter] Route registered: {entity_id} -> {target}")
    
    def unregister_route(self, entity_id: str) -> bool:
        """Unregister a route.
        
        Args:
            entity_id: Entity to unregister
            
        Returns:
            True if route was removed
        """
        if entity_id in self._routes:
            del self._routes[entity_id]
            return True
        return False
    
    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule.
        
        Args:
            rule: Rule to add
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
        logger.debug(f"[MessageRouter] Rule added: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a routing rule.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed
        """
        for i, rule in enumerate(self._rules):
            if rule.name == rule_name:
                self._rules.pop(i)
                logger.debug(f"[MessageRouter] Rule removed: {rule_name}")
                return True
        return False
    
    def set_default_target(self, target: str) -> None:
        """Set the default target for unroutable messages.
        
        Args:
            target: Default target destination
        """
        self._default_target = target
        logger.debug(f"[MessageRouter] Default target set: {target}")
    
    def route(self, message: Message) -> Optional[str]:
        """Determine the target for a message.
        
        Args:
            message: Message to route
            
        Returns:
            Target destination or None
        """
        if message.recipient:
            if message.recipient in self._routes:
                return self._routes[message.recipient]
            return message.recipient
        
        for rule in self._rules:
            if rule.matches(message):
                return rule.target
        
        return self._default_target
    
    def get_route(self, entity_id: str) -> Optional[str]:
        """Get the route for an entity.
        
        Args:
            entity_id: Entity to look up
            
        Returns:
            Target destination or None
        """
        return self._routes.get(entity_id)
    
    def get_rules(self) -> List[RoutingRule]:
        """Get all routing rules.
        
        Returns:
            List of rules
        """
        return self._rules.copy()
    
    def clear(self) -> None:
        """Clear all routes and rules."""
        self._routes.clear()
        self._rules.clear()
        self._default_target = None
        logger.info("[MessageRouter] All routes and rules cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_routes": len(self._routes),
            "total_rules": len(self._rules),
            "default_target": self._default_target,
            "routes": dict(self._routes),
        }
    
    @classmethod
    def create_agent_router(cls) -> "MessageRouter":
        """Create a router configured for agent communication.
        
        Returns:
            Configured MessageRouter
        """
        router = cls()
        
        router.add_rule(RoutingRule(
            name="agent_tasks",
            target="agent_task_queue",
            message_types={
                MessageType.AGENT_TASK,
                MessageType.AGENT_QUERY,
            },
            priority=10,
        ))
        
        router.add_rule(RoutingRule(
            name="agent_results",
            target="agent_result_queue",
            message_types={
                MessageType.AGENT_RESULT,
                MessageType.AGENT_RESPONSE,
            },
            priority=10,
        ))
        
        router.add_rule(RoutingRule(
            name="agent_coordination",
            target="agent_coordination_queue",
            message_types={MessageType.AGENT_COORDINATION},
            priority=5,
        ))
        
        router.add_rule(RoutingRule(
            name="errors",
            target="error_queue",
            message_types={MessageType.ERROR},
            priority=20,
        ))
        
        return router
    
    @classmethod
    def create_component_router(cls) -> "MessageRouter":
        """Create a router configured for component communication.
        
        Returns:
            Configured MessageRouter
        """
        router = cls()
        
        router.add_rule(RoutingRule(
            name="component_requests",
            target="component_request_queue",
            message_types={
                MessageType.COMPONENT_REQUEST,
                MessageType.AGENT_QUERY,
            },
            priority=10,
        ))
        
        router.add_rule(RoutingRule(
            name="component_responses",
            target="component_response_queue",
            message_types={
                MessageType.COMPONENT_RESPONSE,
                MessageType.AGENT_RESPONSE,
            },
            priority=10,
        ))
        
        router.add_rule(RoutingRule(
            name="notifications",
            target="notification_queue",
            message_types={MessageType.COMPONENT_NOTIFICATION},
            priority=5,
        ))
        
        return router
