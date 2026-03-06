"""Context Management Capability (P0).

Provides context compression and management for handling large files
and maintaining relevant context during test generation.
"""

from typing import Any, Dict, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container
    from ....context_manager import ContextManager
    from ....memory.context_compressor import ContextCompressor

logger = __import__('logging').getLogger(__name__)


class ContextManagementCapability(Capability):
    """Context management capability for handling large files.
    
    This capability provides:
    - Context compression for large source files
    - Key snippet extraction
    - Token budget management
    - Context relevance scoring
    
    Configuration:
        max_tokens: Maximum tokens for context (default: 8000)
        target_tokens: Target tokens after compression (default: 6000)
        strategy: Compression strategy (default: "hybrid")
    """
    
    _context_manager: Any = None
    _context_compressor: Any = None
    _max_tokens: int = 8000
    _target_tokens: int = 6000
    _strategy: str = "hybrid"
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="context_management",
            description="Context compression and management for large files",
            priority=CapabilityPriority.HIGH,
            provides={"context_manager", "context_compressor"},
            tags={"p0", "core", "context"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize context management components.
        
        Args:
            container: Dependency injection container
        """
        self._container = container
        
        try:
            from ....context_manager import ContextManager, CompressionStrategy
            from ....memory.context_compressor import ContextCompressor
            
            config = self._get_config()
            self._max_tokens = config.get("max_tokens", 8000)
            self._target_tokens = config.get("target_tokens", 6000)
            self._strategy = config.get("strategy", "hybrid")
            
            strategy_map = {
                "hybrid": CompressionStrategy.HYBRID,
                "semantic": CompressionStrategy.SEMANTIC,
                "sliding_window": CompressionStrategy.SLIDING_WINDOW,
                "hierarchical": CompressionStrategy.HIERARCHICAL,
            }
            compression_strategy = strategy_map.get(
                self._strategy, CompressionStrategy.HYBRID
            )
            
            self._context_manager = ContextManager(
                max_tokens=self._max_tokens,
                target_tokens=self._target_tokens,
                strategy=compression_strategy
            )
            
            self._context_compressor = ContextCompressor(
                max_tokens=self._max_tokens,
                target_tokens=self._target_tokens
            )
            
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info(
                f"[ContextManagementCapability] Initialized - "
                f"max_tokens={self._max_tokens}, strategy={self._strategy}"
            )
            
        except Exception as e:
            self._set_error(e)
            raise
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration from container or defaults."""
        if self._container:
            try:
                settings = self._container.resolve(type('Settings', (), {}))
                if hasattr(settings, 'context'):
                    return {
                        "max_tokens": getattr(settings.context, 'max_tokens', 8000),
                        "target_tokens": getattr(settings.context, 'target_tokens', 6000),
                        "strategy": getattr(settings.context, 'strategy', 'hybrid'),
                    }
            except Exception:
                pass
        return {}
    
    @property
    def context_manager(self) -> Any:
        """Get the context manager instance."""
        return self._context_manager
    
    @property
    def context_compressor(self) -> Any:
        """Get the context compressor instance."""
        return self._context_compressor
    
    def build_context(
        self,
        query: str,
        target_file: Any = None,
        additional_context: Dict[str, Any] = None
    ) -> Any:
        """Build compressed context for a query.
        
        Args:
            query: The query to build context for
            target_file: Optional target file
            additional_context: Additional context to include
            
        Returns:
            Compressed context
        """
        if not self._context_manager:
            return None
        
        return self._context_manager.build_context(
            query=query,
            target_file=target_file,
            additional_context=additional_context or {}
        )
    
    def shutdown(self) -> None:
        """Shutdown context management."""
        self._context_manager = None
        self._context_compressor = None
        super().shutdown()
