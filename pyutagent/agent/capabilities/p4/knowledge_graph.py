"""Knowledge Graph Capability (P4).

Provides project knowledge graph for code understanding.
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container

logger = __import__('logging').getLogger(__name__)


class KnowledgeGraphCapability(Capability):
    """Knowledge graph capability.
    
    This capability provides:
    - Code structure analysis
    - Dependency graph
    - Knowledge persistence
    """
    
    _graph: Any = None
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="knowledge_graph",
            description="Project knowledge graph for code understanding",
            priority=CapabilityPriority.NORMAL,
            provides={"knowledge_graph"},
            tags={"p4", "knowledge", "graph"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize knowledge graph."""
        self._container = container
        
        try:
            from ....memory.project_knowledge_graph import ProjectKnowledgeGraph
            
            self._graph = ProjectKnowledgeGraph()
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info("[KnowledgeGraphCapability] Initialized")
            
        except Exception as e:
            self._set_error(e)
            raise
    
    @property
    def graph(self) -> Any:
        """Get the graph instance."""
        return self._graph
    
    def shutdown(self) -> None:
        """Shutdown graph."""
        self._graph = None
        super().shutdown()
