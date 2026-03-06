"""Partial Success Handling Capability (P0).

Provides intelligent handling of partial test success,
allowing incremental fixes rather than complete regeneration.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container
    from ....partial_success_handler import PartialSuccessHandler

logger = __import__('logging').getLogger(__name__)


class PartialSuccessCapability(Capability):
    """Partial success handling capability.
    
    This capability provides:
    - Analysis of partial test results
    - Incremental fix strategies
    - Test result merging
    - Selective test regeneration
    
    Configuration:
        max_incremental_fixes: Maximum incremental fix attempts (default: 3)
        preserve_passing: Whether to preserve passing tests (default: True)
    """
    
    _handler: Any = None
    _max_incremental_fixes: int = 3
    _preserve_passing: bool = True
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="partial_success",
            description="Handle partial test success with incremental fixes",
            priority=CapabilityPriority.HIGH,
            provides={"partial_success_handler"},
            dependencies={"context_management"},
            tags={"p0", "core", "recovery"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize partial success handler.
        
        Args:
            container: Dependency injection container
        """
        self._container = container
        
        try:
            from ....partial_success_handler import PartialSuccessHandler
            
            config = self._get_config()
            self._max_incremental_fixes = config.get("max_incremental_fixes", 3)
            self._preserve_passing = config.get("preserve_passing", True)
            
            self._handler = PartialSuccessHandler(
                max_incremental_fixes=self._max_incremental_fixes,
                preserve_passing=self._preserve_passing
            )
            
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info(
                f"[PartialSuccessCapability] Initialized - "
                f"max_fixes={self._max_incremental_fixes}, preserve={self._preserve_passing}"
            )
            
        except Exception as e:
            self._set_error(e)
            raise
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration from container or defaults."""
        if self._container:
            try:
                settings = self._container.resolve(type('Settings', (), {}))
                if hasattr(settings, 'partial_success'):
                    return {
                        "max_incremental_fixes": getattr(
                            settings.partial_success, 'max_incremental_fixes', 3
                        ),
                        "preserve_passing": getattr(
                            settings.partial_success, 'preserve_passing', True
                        ),
                    }
            except Exception:
                pass
        return {}
    
    @property
    def handler(self) -> Any:
        """Get the handler instance."""
        return self._handler
    
    def analyze_test_results(
        self,
        test_output: str,
        surefire_reports_dir: Any = None
    ) -> Any:
        """Analyze test results for partial success.
        
        Args:
            test_output: Test execution output
            surefire_reports_dir: Directory with Surefire reports
            
        Returns:
            PartialTestResult with analysis
        """
        if not self._handler:
            return None
        
        return self._handler.analyze_test_results(
            test_output=test_output,
            surefire_reports_dir=surefire_reports_dir
        )
    
    def should_attempt_incremental_fix(self, partial_result: Any) -> bool:
        """Check if incremental fix should be attempted.
        
        Args:
            partial_result: Partial test result
            
        Returns:
            True if incremental fix should be attempted
        """
        if not self._handler:
            return False
        
        return self._handler.should_attempt_incremental_fix(partial_result)
    
    def create_incremental_fix_prompt(
        self,
        test_code: str,
        partial_result: Any,
        target_class_info: Dict[str, Any]
    ) -> str:
        """Create prompt for incremental fix.
        
        Args:
            test_code: Current test code
            partial_result: Partial test result
            target_class_info: Target class information
            
        Returns:
            Prompt for incremental fix
        """
        if not self._handler:
            return ""
        
        return self._handler.create_incremental_fix_prompt(
            test_code=test_code,
            partial_result=partial_result,
            target_class_info=target_class_info
        )
    
    def merge_incremental_fix(
        self,
        original_code: str,
        fixed_code: str,
        partial_result: Any
    ) -> Any:
        """Merge incremental fix with original code.
        
        Args:
            original_code: Original test code
            fixed_code: Fixed test code
            partial_result: Partial test result
            
        Returns:
            MergeResult with merged code
        """
        if not self._handler:
            return None
        
        return self._handler.merge_incremental_fix(
            original_code=original_code,
            fixed_code=fixed_code,
            partial_result=partial_result
        )
    
    def shutdown(self) -> None:
        """Shutdown handler."""
        self._handler = None
        super().shutdown()
