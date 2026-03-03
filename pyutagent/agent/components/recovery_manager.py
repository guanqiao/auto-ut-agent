"""Agent Recovery Manager - Enhanced error recovery with learning."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pyutagent.core.error_recovery import ErrorCategory, RecoveryStrategy

logger = logging.getLogger(__name__)


class AgentRecoveryManager:
    """Handles error recovery with learning and optimization.
    
    Features:
    - Error categorization
    - Strategy suggestion from error learner
    - Strategy optimization
    - Recovery outcome recording
    """
    
    def __init__(self, components: Dict[str, Any], agent_core: Any):
        """Initialize recovery manager.
        
        Args:
            components: Dictionary of all components
            agent_core: AgentCore instance
        """
        self.components = components
        self.agent_core = agent_core
        
        logger.debug("[AgentRecoveryManager] Initialized")
    
    async def recover_from_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        step_name: str,
        attempt: int
    ) -> Dict[str, Any]:
        """Recover from an error with learning and optimization.
        
        Args:
            error: The error that occurred
            context: Error context
            step_name: Name of the step where error occurred
            attempt: Attempt number
            
        Returns:
            Recovery result dictionary
        """
        import time
        start_time = time.time()
        
        error_message = str(error).lower()
        if "no compilation errors" in error_message or "no test failures" in error_message or "all tests passed" in error_message:
            logger.info(f"[AgentRecoveryManager] Detected false positive error, skipping recovery")
            return {
                "should_continue": True,
                "action": "skip",
                "reason": "No actual error detected"
            }
        
        logger.info(f"[AgentRecoveryManager] Attempting recovery - Error: {error}, Context: {context}")
        
        error_category = self._categorize_error(error, context)
        
        suggested_strategy = self.components["error_learner"].suggest_strategy(error, error_category, context)
        if suggested_strategy:
            strategy, confidence = suggested_strategy
            logger.info(f"[AgentRecoveryManager] Error learner suggests {strategy.name} with confidence {confidence:.2f}")
            
            optimization = self.components["strategy_optimizer"].optimize_strategy_selection(error_category, context)
            logger.info(f"[AgentRecoveryManager] Strategy optimizer recommends {optimization.recommended_strategy.name}")
        
        current_test_code = None
        if self.agent_core.current_test_file:
            try:
                test_file_path = Path(self.agent_core.project_path) / self.agent_core.current_test_file
                if test_file_path.exists():
                    current_test_code = test_file_path.read_text(encoding='utf-8')
                    logger.debug(f"[AgentRecoveryManager] Read current test code - Length: {len(current_test_code)}")
            except Exception as e:
                logger.warning(f"[AgentRecoveryManager] Failed to read test code: {e}")
        
        recovery_result = await self.components["error_recovery"].recover(
            error,
            error_context=context,
            current_test_code=current_test_code,
            target_class_info=self.agent_core.target_class_info
        )
        
        elapsed_time = time.time() - start_time
        success = recovery_result.get("action") not in ("abort", "fail")
        strategy_used = self._determine_strategy_from_action(recovery_result.get("action", "retry"))
        
        self.components["error_learner"].learn_from_recovery(
            error=error,
            error_category=error_category,
            strategy=strategy_used,
            success=success,
            context=context,
            time_to_recover=elapsed_time,
            attempts_needed=attempt
        )
        
        self.components["strategy_optimizer"].record_result(
            error_category=error_category,
            strategy=strategy_used,
            success=success,
            time_taken=elapsed_time,
            attempts=attempt
        )
        
        logger.info(f"[AgentRecoveryManager] Recovery result - Action: {recovery_result.get('action')}, ShouldContinue: {recovery_result.get('should_continue')}")
        
        return recovery_result
    
    def _categorize_error(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Categorize an error for learning purposes.
        
        Args:
            error: The error that occurred
            context: Error context
            
        Returns:
            ErrorCategory enum value
        """
        error_message = str(error).lower()
        step = context.get("step", "")
        
        if "compile" in step or "compilation" in error_message:
            return ErrorCategory.COMPILATION_ERROR
        elif "test" in step and "fail" in error_message:
            return ErrorCategory.TEST_FAILURE
        elif "timeout" in error_message:
            return ErrorCategory.TIMEOUT
        elif "network" in error_message or "connection" in error_message:
            return ErrorCategory.NETWORK
        elif "api" in error_message or "llm" in error_message:
            return ErrorCategory.LLM_API_ERROR
        elif "parse" in error_message:
            return ErrorCategory.PARSING_ERROR
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_strategy_from_action(self, action: str) -> RecoveryStrategy:
        """Determine recovery strategy from action type.
        
        Args:
            action: Action type from recovery result
            
        Returns:
            RecoveryStrategy enum value
        """
        if action == "fix":
            return RecoveryStrategy.ANALYZE_AND_FIX
        elif action == "reset":
            return RecoveryStrategy.RESET_AND_REGENERATE
        elif action == "retry":
            return RecoveryStrategy.RETRY_IMMEDIATE
        else:
            return RecoveryStrategy.DEFAULT
