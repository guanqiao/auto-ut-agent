"""Error Prediction Capability (P3).

Provides predictive analysis to identify potential errors
before compilation.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..base import Capability, CapabilityMetadata, CapabilityPriority
from ...execution.retry import RetryConfig

if TYPE_CHECKING:
    from ....core.container import Container
    from ....core.error_predictor import ErrorPredictor

logger = __import__('logging').getLogger(__name__)


class ErrorPredictionCapability(Capability):
    """Error prediction capability.
    
    This capability provides:
    - Pre-compilation error prediction
    - Error type classification
    - Fix suggestions
    - Risk assessment
    
    Configuration:
        enable_prediction: Enable error prediction (default: True)
        confidence_threshold: Minimum confidence for suggestions (default: 0.7)
    """
    
    _predictor: Any = None
    _enable_prediction: bool = True
    _confidence_threshold: float = 0.7
    
    @classmethod
    def metadata(cls) -> CapabilityMetadata:
        return CapabilityMetadata(
            name="error_prediction",
            description="Predict potential errors before compilation",
            priority=CapabilityPriority.NORMAL,
            provides={"error_predictor"},
            dependencies={"generation_evaluation"},
            tags={"p3", "prediction", "quality"}
        )
    
    def _create_default_retry_config(self) -> "RetryConfig":
        """Create default retry config for error prediction.
        
        Error prediction is a best-effort feature and can accept
        more failures with faster backoff.
        """
        return RetryConfig(
            max_attempts=2,
            base_delay=0.5,
            max_delay=10.0,
        )
    
    def initialize(self, container: "Container") -> None:
        """Initialize error predictor.
        
        Args:
            container: Dependency injection container
        """
        self._container = container
        
        try:
            from ....core.error_predictor import ErrorPredictor
            
            config = self._get_config()
            self._enable_prediction = config.get("enable_prediction", True)
            self._confidence_threshold = config.get("confidence_threshold", 0.7)
            
            if self._enable_prediction:
                self._predictor = ErrorPredictor(
                    confidence_threshold=self._confidence_threshold
                )
            
            self._state = type('CapabilityState', (), {'READY': 'ready'})().READY
            logger.info(
                f"[ErrorPredictionCapability] Initialized - "
                f"enabled={self._enable_prediction}, threshold={self._confidence_threshold}"
            )
            
        except Exception as e:
            self._set_error(e)
            raise
    
    def _get_config(self) -> Dict[str, Any]:
        """Get configuration from container or defaults."""
        if self._container:
            try:
                settings = self._container.resolve(type('Settings', (), {}))
                if hasattr(settings, 'error_prediction'):
                    return {
                        "enable_prediction": getattr(
                            settings.error_prediction, 'enabled', True
                        ),
                        "confidence_threshold": getattr(
                            settings.error_prediction, 'confidence_threshold', 0.7
                        ),
                    }
            except Exception:
                pass
        return {}
    
    @property
    def predictor(self) -> Any:
        """Get the predictor instance."""
        return self._predictor
    
    def predict_compilation_errors(
        self,
        code: str,
        file_path: Optional[str] = None
    ) -> Any:
        """Predict potential compilation errors.
        
        Args:
            code: Code to analyze
            file_path: Optional file path for context
            
        Returns:
            PredictionResult with predicted errors
        """
        if not self._predictor:
            return None
        
        return self._predictor.predict_compilation_errors(code, file_path)
    
    def suggest_fix(self, error: Any, code: str) -> Optional[Dict[str, Any]]:
        """Suggest fix for a predicted error.
        
        Args:
            error: Predicted error
            code: Code containing the error
            
        Returns:
            Fix suggestion dictionary
        """
        if not self._predictor:
            return None
        
        return self._predictor.suggest_fix(error, code)
    
    def get_risk_assessment(self, code: str) -> Dict[str, Any]:
        """Get overall risk assessment for code.
        
        Args:
            code: Code to assess
            
        Returns:
            Risk assessment dictionary
        """
        if not self._predictor:
            return {"risk_score": 0.0, "enabled": False}
        
        prediction = self._predictor.predict_compilation_errors(code)
        return {
            "risk_score": prediction.overall_risk_score if prediction else 0.0,
            "error_count": len(prediction.predicted_errors) if prediction else 0,
            "enabled": True
        }
    
    def shutdown(self) -> None:
        """Shutdown predictor."""
        self._predictor = None
        super().shutdown()
