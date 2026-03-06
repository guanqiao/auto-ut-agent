"""Generation Evaluation Capability (P0).

Provides quality evaluation for generated test code.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from ..base import Capability, CapabilityMetadata, CapabilityPriority

if TYPE_CHECKING:
    from ....core.container import Container
    from ....generation_evaluator import GenerationEvaluator

logger = __import__('logging').getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of generation evaluation."""
    overall_score: float
    is_acceptable: bool
    coverage_estimate: Optional[Any] = None
    issues: List[Dict[str, Any]] = None
    
    def get_critical_issues(self) -> List[Dict[str, Any]]:
        """Get critical issues from evaluation."""
        if not self.issues:
            return []
        return [i for i in self.issues if i.get("severity") == "critical"]


class GenerationEvaluationCapability(Capability):
    """Generation evaluation capability for test code quality.
    
    This capability provides:
    - 6-dimension quality evaluation
    - Coverage potential estimation
    - Issue detection and classification
    - Acceptance threshold checking
    
    Configuration:
        min_acceptable_score: Minimum score for acceptance (default: 0.6)
        dimensions: List of evaluation dimensions
    """
    
    _evaluator: Any = None
    _min_acceptable_score: float = 0.6
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="generation_evaluation",
            description="Quality evaluation for generated test code",
            priority=CapabilityPriority.HIGH,
            provides={"generation_evaluator"},
            tags={"p0", "core", "quality"}
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize generation evaluator.
        
        Args:
            container: Dependency injection container
        """
        self._container = container
        
        try:
            from ....generation_evaluator import GenerationEvaluator
            
            config = self._get_config()
            self._min_acceptable_score = config.get("min_acceptable_score", 0.6)
            
            self._evaluator = GenerationEvaluator(
                min_acceptable_score=self._min_acceptable_score
            )
            
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info(
                f"[GenerationEvaluationCapability] Initialized - "
                f"min_score={self._min_acceptable_score}"
            )
            
        except Exception as e:
            self._set_error(e)
            raise
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration from container or defaults."""
        if self._container:
            try:
                settings = self._container.resolve(type('Settings', (), {}))
                if hasattr(settings, 'generation'):
                    return {
                        "min_acceptable_score": getattr(
                            settings.generation, 'min_acceptable_score', 0.6
                        ),
                    }
            except Exception:
                pass
        return {}
    
    @property
    def evaluator(self) -> Any:
        """Get the evaluator instance."""
        return self._evaluator
    
    def evaluate(
        self,
        test_code: str,
        target_class_info: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate generated test code.
        
        Args:
            test_code: The generated test code
            target_class_info: Information about the target class
            
        Returns:
            EvaluationResult with quality metrics
        """
        if not self._evaluator:
            return EvaluationResult(
                overall_score=0.0,
                is_acceptable=False,
                issues=[{"severity": "critical", "message": "Evaluator not initialized"}]
            )
        
        try:
            result = self._evaluator.evaluate(
                test_code=test_code,
                target_class_info=target_class_info
            )
            
            return EvaluationResult(
                overall_score=result.overall_score,
                is_acceptable=result.is_acceptable,
                coverage_estimate=getattr(result, 'coverage_estimate', None),
                issues=getattr(result, 'issues', [])
            )
        except Exception as e:
            logger.error(f"[GenerationEvaluationCapability] Evaluation failed: {e}")
            return EvaluationResult(
                overall_score=0.0,
                is_acceptable=False,
                issues=[{"severity": "critical", "message": str(e)}]
            )
    
    def quick_evaluate(self, test_code: str) -> float:
        """Quick evaluation returning just the score.
        
        Args:
            test_code: The generated test code
            
        Returns:
            Overall quality score
        """
        if not self._evaluator:
            return 0.0
        
        try:
            return self._evaluator.quick_evaluate(test_code)
        except Exception:
            return 0.0
    
    def shutdown(self) -> None:
        """Shutdown evaluator."""
        self._evaluator = None
        super().shutdown()
