"""Unified error recovery module for enhanced fault tolerance.

This module provides comprehensive error recovery strategies including:
- Error classification and analysis
- Recovery strategy selection
- State preservation and rollback
- Graceful degradation
- AI-assisted error analysis

This module consolidates the functionality from:
- agent/error_recovery.py (AI-assisted analysis, agent-specific recovery)
- tools/error_recovery.py (generic recovery patterns, state preservation)
"""

import asyncio
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Any,
    Callable,
    Type,
    Union,
    Generic,
    TypeVar,
    Tuple,
)

logger = logging.getLogger(__name__)

from ..utils.code_extractor import CodeExtractor

T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors for recovery strategy selection."""
    TRANSIENT = auto()
    PERMANENT = auto()
    RESOURCE = auto()
    NETWORK = auto()
    TIMEOUT = auto()
    VALIDATION = auto()
    SYNTAX = auto()
    LOGIC = auto()
    COMPILATION_ERROR = auto()
    TEST_FAILURE = auto()
    TOOL_EXECUTION_ERROR = auto()
    PARSING_ERROR = auto()
    GENERATION_ERROR = auto()
    FILE_IO_ERROR = auto()
    LLM_API_ERROR = auto()
    UNKNOWN = auto()


class RecoveryStrategy(Enum):
    """Recovery strategies."""
    RETRY = auto()
    RETRY_IMMEDIATE = auto()
    RETRY_WITH_BACKOFF = auto()
    BACKOFF = auto()
    FALLBACK = auto()
    RESET = auto()
    SKIP = auto()
    SKIP_AND_CONTINUE = auto()
    ABORT = auto()
    MANUAL = auto()
    ANALYZE_AND_FIX = auto()
    RESET_AND_REGENERATE = auto()
    FALLBACK_ALTERNATIVE = auto()
    ESCALATE_TO_USER = auto()
    INSTALL_DEPENDENCIES = auto()
    RESOLVE_DEPENDENCIES = auto()
    FIX_ENVIRONMENT = auto()


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    error_type: Type[Exception]
    error_message: str
    stack_trace: str
    category: ErrorCategory
    timestamp: datetime
    operation: str
    attempt: int
    context_data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        error: Exception,
        operation: str,
        attempt: int = 1,
        context_data: Optional[Dict[str, Any]] = None
    ) -> "ErrorContext":
        return cls(
            error=error,
            error_type=type(error),
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            category=ErrorClassifier.classify(error),
            timestamp=datetime.now(),
            operation=operation,
            attempt=attempt,
            context_data=context_data or {}
        )


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    timestamp: str
    error_category: ErrorCategory
    error_message: str
    strategy_used: RecoveryStrategy
    attempt_number: int
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryContext:
    """Context for error recovery."""
    error_category: ErrorCategory
    error_message: str
    error_details: Dict[str, Any]
    current_test_code: Optional[str] = None
    target_class_info: Optional[Dict[str, Any]] = None
    attempt_history: List[RecoveryAttempt] = field(default_factory=list)
    max_attempts_per_strategy: int = 3


@dataclass
class RecoveryResult(Generic[T]):
    """Result of a recovery attempt."""
    success: bool
    strategy_used: RecoveryStrategy
    attempts_made: int
    error_context: Optional[ErrorContext] = None
    recovered_data: Optional[T] = None
    error_message: str = ""
    next_action: str = ""
    action: str = ""
    should_continue: bool = True
    fixed_code: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """Classifies errors into categories for recovery strategy selection."""

    ERROR_MAPPINGS: Dict[Type[Exception], ErrorCategory] = {
        ConnectionError: ErrorCategory.NETWORK,
        TimeoutError: ErrorCategory.TIMEOUT,
        asyncio.TimeoutError: ErrorCategory.TIMEOUT,
        MemoryError: ErrorCategory.RESOURCE,
        OSError: ErrorCategory.RESOURCE,
        ValueError: ErrorCategory.VALIDATION,
        TypeError: ErrorCategory.VALIDATION,
        SyntaxError: ErrorCategory.SYNTAX,
    }

    MESSAGE_MAPPINGS: Dict[ErrorCategory, List[str]] = {
        ErrorCategory.NETWORK: [
            "connection", "network", "socket", "timeout",
            "unreachable", "refused", "reset"
        ],
        ErrorCategory.RESOURCE: [
            "memory", "disk", "space", "resource",
            "quota", "limit exceeded"
        ],
        ErrorCategory.TRANSIENT: [
            "temporary", "transient", "retry", "unavailable",
            "rate limit", "throttled"
        ],
        ErrorCategory.PERMANENT: [
            "not found", "invalid", "unauthorized",
            "forbidden", "authentication"
        ],
        ErrorCategory.COMPILATION_ERROR: [
            "cannot find symbol", "package .* does not exist",
            "incompatible types", "expected", "illegal",
            "compilation", "javac", "syntax error"
        ],
        ErrorCategory.TEST_FAILURE: [
            "assertion", "test failed", "nullpointer",
            "exception", "wanted but not invoked", "verification"
        ],
        ErrorCategory.TOOL_EXECUTION_ERROR: [
            "command not found", "exit code", "process failed",
            "mvn error", "maven", "subprocess"
        ],
        ErrorCategory.PARSING_ERROR: [
            "parse", "unexpected token", "tree-sitter"
        ],
        ErrorCategory.GENERATION_ERROR: [
            "generate", "llm", "model", "token", "completion"
        ],
        ErrorCategory.LLM_API_ERROR: [
            "api key", "rate limit", "timeout", "connection",
            "openai", "anthropic", "deepseek"
        ],
        ErrorCategory.FILE_IO_ERROR: [
            "no such file", "permission denied", "ioerror",
            "file not found", "cannot read", "cannot write"
        ],
    }

    @classmethod
    def classify(cls, error: Exception) -> ErrorCategory:
        error_type = type(error)
        error_message = str(error).lower()

        for exc_type, category in cls.ERROR_MAPPINGS.items():
            if isinstance(error, exc_type):
                return category

        for category, keywords in cls.MESSAGE_MAPPINGS.items():
            for keyword in keywords:
                if keyword in error_message:
                    return category

        return ErrorCategory.UNKNOWN

    @classmethod
    def categorize_error(cls, error_message: str, error_details: Dict[str, Any]) -> ErrorCategory:
        error_lower = error_message.lower()

        for category, keywords in cls.MESSAGE_MAPPINGS.items():
            for keyword in keywords:
                if keyword in error_lower:
                    return category

        return ErrorCategory.UNKNOWN

    @classmethod
    def is_retryable(cls, error: Exception) -> bool:
        category = cls.classify(error)
        return category in [
            ErrorCategory.TRANSIENT,
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
            ErrorCategory.RESOURCE
        ]


class StatePreserver:
    """Preserves and restores state for rollback capability."""

    def __init__(self, max_stack_size: int = 10):
        self.state_stack: List[Dict[str, Any]] = []
        self.max_stack_size = max_stack_size

    def save_state(self, state: Dict[str, Any], label: str = "") -> int:
        state_copy = {
            "_label": label,
            "_timestamp": datetime.now(),
            "_version": len(self.state_stack),
            **{k: v for k, v in state.items()}
        }

        self.state_stack.append(state_copy)

        if len(self.state_stack) > self.max_stack_size:
            self.state_stack.pop(0)

        logger.debug(f"State saved: {label} (version {state_copy['_version']})")
        return state_copy["_version"]

    def restore_state(self, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not self.state_stack:
            return None

        if version is None:
            state = self.state_stack[-1]
            logger.info(f"Restored to last state: {state.get('_label', 'unnamed')}")
            return {k: v for k, v in state.items() if not k.startswith('_')}

        for state in reversed(self.state_stack):
            if state.get("_version") == version:
                logger.info(f"Restored to state version {version}: {state.get('_label', 'unnamed')}")
                return {k: v for k, v in state.items() if not k.startswith('_')}

        logger.warning(f"State version {version} not found")
        return None

    def clear_history(self):
        self.state_stack.clear()
        logger.info("State history cleared")


class GracefulDegradation:
    """Provides graceful degradation for operations."""

    def __init__(self):
        self.degradation_levels: Dict[str, List[Callable]] = {}
        self.current_level: Dict[str, int] = {}

    def register_degradation_chain(
        self,
        operation_name: str,
        methods: List[Callable]
    ):
        self.degradation_levels[operation_name] = methods
        self.current_level[operation_name] = 0

    async def execute_with_degradation(
        self,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        if operation_name not in self.degradation_levels:
            raise ValueError(f"No degradation chain registered for {operation_name}")

        methods = self.degradation_levels[operation_name]
        start_level = self.current_level.get(operation_name, 0)

        for i in range(start_level, len(methods)):
            method = methods[i]
            try:
                logger.info(f"Trying {operation_name} at degradation level {i}")

                if asyncio.iscoroutinefunction(method):
                    result = await method(*args, **kwargs)
                else:
                    result = method(*args, **kwargs)

                self.current_level[operation_name] = i
                return result

            except Exception as e:
                logger.warning(f"Degradation level {i} failed: {e}")
                continue

        raise Exception(f"All degradation levels failed for {operation_name}")

    def reset_level(self, operation_name: str):
        self.current_level[operation_name] = 0
        logger.info(f"Degradation level reset for {operation_name}")


class RecoveryManager:
    """Generic recovery manager with multiple strategies."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0
    ):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.error_history: List[ErrorContext] = []
        self.recovery_stats: Dict[str, Dict[str, int]] = {}

    async def execute_with_recovery(
        self,
        operation: Callable,
        operation_name: str,
        fallback: Optional[Callable] = None,
        context_data: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> RecoveryResult:
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Executing {operation_name} (attempt {attempt}/{self.max_retries})")

                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)

                self._record_success(operation_name, attempt)
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RETRY,
                    attempts_made=attempt,
                    recovered_data=result
                )

            except Exception as e:
                last_error = e
                error_context = ErrorContext.from_exception(
                    e, operation_name, attempt, context_data
                )
                self.error_history.append(error_context)

                logger.warning(f"{operation_name} failed (attempt {attempt}): {e}")

                if not ErrorClassifier.is_retryable(e):
                    logger.error(f"Error not retryable: {e}")
                    break

                if attempt < self.max_retries:
                    delay = min(
                        self.backoff_base * (2 ** (attempt - 1)),
                        self.backoff_max
                    )
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)

        if fallback:
            logger.info(f"Attempting fallback for {operation_name}")
            try:
                if asyncio.iscoroutinefunction(fallback):
                    result = await fallback(*args, **kwargs)
                else:
                    result = fallback(*args, **kwargs)

                self._record_fallback_success(operation_name)
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.FALLBACK,
                    attempts_made=self.max_retries,
                    recovered_data=result
                )
            except Exception as e:
                logger.error(f"Fallback also failed: {e}")

        self._record_failure(operation_name)
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.ABORT,
            attempts_made=self.max_retries,
            error_context=ErrorContext.from_exception(
                last_error, operation_name, self.max_retries, context_data
            ),
            error_message=str(last_error),
            next_action="Manual intervention required"
        )

    def _record_success(self, operation: str, attempts: int):
        if operation not in self.recovery_stats:
            self.recovery_stats[operation] = {
                "success": 0, "failure": 0, "fallback_success": 0, "total_attempts": 0
            }
        self.recovery_stats[operation]["success"] += 1
        self.recovery_stats[operation]["total_attempts"] += attempts

    def _record_failure(self, operation: str):
        if operation not in self.recovery_stats:
            self.recovery_stats[operation] = {
                "success": 0, "failure": 0, "fallback_success": 0, "total_attempts": 0
            }
        self.recovery_stats[operation]["failure"] += 1

    def _record_fallback_success(self, operation: str):
        if operation not in self.recovery_stats:
            self.recovery_stats[operation] = {
                "success": 0, "failure": 0, "fallback_success": 0, "total_attempts": 0
            }
        self.recovery_stats[operation]["fallback_success"] += 1

    def get_recovery_stats(self) -> Dict[str, Dict[str, int]]:
        return self.recovery_stats.copy()

    def get_error_history(self) -> List[ErrorContext]:
        return self.error_history.copy()

    def clear_history(self):
        self.error_history.clear()


class ErrorRecoveryManager:
    """Manages error recovery with AI assistance.

    This is the central hub for handling all errors. It:
    1. Categorizes errors
    2. Attempts local analysis
    3. Uses LLM for deep analysis if needed
    4. Applies appropriate recovery strategies
    5. Tracks all attempts and adjusts strategies
    6. Uses ThinkingEngine for intelligent reasoning (Phase 3)
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        project_path: str = "",
        prompt_builder: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        max_total_attempts: int = 50,
        max_same_error_attempts: int = 3,
        thinking_engine: Optional[Any] = None,
        enable_thinking: bool = True
    ):
        self.llm_client = llm_client
        self.project_path = project_path
        self.prompt_builder = prompt_builder
        self.progress_callback = progress_callback
        self.max_total_attempts = max_total_attempts
        self.max_same_error_attempts = max_same_error_attempts
        self.thinking_engine = thinking_engine
        self.enable_thinking = enable_thinking

        self.recovery_history: List[RecoveryAttempt] = []
        self.state_preserver = StatePreserver()
        
        self._last_error_signature: Optional[str] = None
        self._same_error_count: int = 0
        self._thinking_results: List[Any] = []

    def categorize_error(self, error_message: str, error_details: Dict[str, Any]) -> ErrorCategory:
        return ErrorClassifier.categorize_error(error_message, error_details)

    def _compute_error_signature(self, error: Exception, error_context: Dict[str, Any]) -> str:
        """Compute a signature for the error to detect repeated errors.
        
        Args:
            error: The error to compute signature for
            error_context: Error context information
            
        Returns:
            A string signature for the error
        """
        error_type = type(error).__name__
        error_msg = str(error)[:200]
        step = error_context.get("step", "")
        
        if "compilation" in step.lower():
            compiler_output = error_context.get("compiler_output", "")
            error_lines = [line.strip() for line in compiler_output.split('\n') 
                          if "error:" in line.lower() or "cannot find" in line.lower()]
            error_msg = "|".join(error_lines[:3]) if error_lines else error_msg
        elif "test" in step.lower():
            failures = error_context.get("failures", [])
            if failures:
                error_msg = "|".join([f.get("error", "")[:50] for f in failures[:3]])
        
        return f"{error_type}:{step}:{error_msg}"

    def _check_same_error(self, error: Exception, error_context: Dict[str, Any]) -> bool:
        """Check if this is the same error as before and update tracking.
        
        Args:
            error: The error to check
            error_context: Error context information
            
        Returns:
            True if this is a repeated error (same as last)
        """
        current_signature = self._compute_error_signature(error, error_context)
        
        if current_signature == self._last_error_signature:
            self._same_error_count += 1
            logger.warning(f"[ErrorRecoveryManager] Same error detected - Count: {self._same_error_count}")
            return True
        
        self._last_error_signature = current_signature
        self._same_error_count = 1
        return False

    def reset_error_tracking(self):
        """Reset error tracking state."""
        self._last_error_signature = None
        self._same_error_count = 0

    def get_thinking_results(self) -> List[Any]:
        """Get all thinking results from recovery sessions."""
        return self._thinking_results.copy()

    def get_last_thinking_result(self) -> Optional[Any]:
        """Get the most recent thinking result."""
        return self._thinking_results[-1] if self._thinking_results else None

    async def recover(
        self,
        error: Exception,
        error_context: Dict[str, Any],
        current_test_code: Optional[str] = None,
        target_class_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        error_message = str(error)
        error_category = self.categorize_error(error_message, error_context)

        thinking_result = None
        if self.enable_thinking and self.thinking_engine:
            thinking_result = await self._think_before_recovery(
                error, error_message, error_category, error_context
            )

        is_same_error = self._check_same_error(error, error_context)
        if is_same_error and self._same_error_count >= self.max_same_error_attempts:
            logger.error(f"[ErrorRecoveryManager] Same error repeated {self._same_error_count} times, escalating strategy")
            error_category = ErrorCategory.PERMANENT

        category_attempts = len([a for a in self.recovery_history
                                 if a.error_category == error_category])
        if category_attempts >= self.max_total_attempts:
            return {
                "success": False,
                "action": "stop",
                "message": f"Exceeded maximum recovery attempts ({self.max_total_attempts})",
                "should_continue": False
            }

        context = RecoveryContext(
            error_category=error_category,
            error_message=error_message,
            error_details=error_context,
            current_test_code=current_test_code,
            target_class_info=target_class_info,
            attempt_history=self.recovery_history.copy()
        )

        local_analysis = await self._local_analysis(context)
        if is_same_error and self._same_error_count > 1:
            local_analysis["same_error_count"] = self._same_error_count
            local_analysis["previous_fix_failed"] = True

        llm_analysis = await self._llm_analysis(context, local_analysis)

        if thinking_result:
            llm_analysis["thinking_result"] = thinking_result
            if thinking_result.recovery_strategy:
                try:
                    llm_analysis["recommended_strategy"] = thinking_result.recovery_strategy
                    llm_analysis["confidence"] = thinking_result.confidence
                except Exception:
                    pass

        strategy = self._determine_strategy(context, local_analysis, llm_analysis)
        if is_same_error and self._same_error_count >= 2:
            if strategy in (RecoveryStrategy.ANALYZE_AND_FIX, RecoveryStrategy.RETRY_IMMEDIATE):
                if self._same_error_count >= self.max_same_error_attempts:
                    strategy = RecoveryStrategy.RESET_AND_REGENERATE
                    logger.info(f"[ErrorRecoveryManager] Escalating to RESET_AND_REGENERATE due to repeated errors")

        attempt_number = category_attempts + 1
        result = await self._execute_recovery(context, strategy, llm_analysis, attempt_number)

        attempt = RecoveryAttempt(
            timestamp=datetime.now().isoformat(),
            error_category=error_category,
            error_message=error_message,
            strategy_used=strategy,
            attempt_number=attempt_number,
            success=result.get("success", False),
            details=result.get("details", {})
        )
        self.recovery_history.append(attempt)

        return result

    async def _think_before_recovery(
        self,
        error: Exception,
        error_message: str,
        error_category: ErrorCategory,
        error_context: Dict[str, Any]
    ) -> Optional[Any]:
        """Use ThinkingEngine to think about the error before recovery.
        
        Args:
            error: The exception that occurred
            error_message: Error message
            error_category: Category of the error
            error_context: Error context information
            
        Returns:
            ThinkingResult or None
        """
        try:
            if self.progress_callback:
                self.progress_callback("THINKING", "Analyzing error with thinking engine...")

            logger.info(f"[ErrorRecoveryManager] Thinking about error: {error_category.name}")

            thinking_result = await self.thinking_engine.think_about_error(
                error=error,
                context={
                    "error_category": error_category.name,
                    "error_message": error_message[:500],
                    **error_context
                }
            )

            self._thinking_results.append(thinking_result)

            if self.progress_callback:
                self.progress_callback(
                    "THINKING_COMPLETE",
                    f"Thinking complete - Confidence: {thinking_result.confidence:.0%}"
                )

            logger.info(
                f"[ErrorRecoveryManager] Thinking result - "
                f"Root cause: {thinking_result.root_cause[:100]}, "
                f"Strategy: {thinking_result.recovery_strategy}, "
                f"Confidence: {thinking_result.confidence:.2f}"
            )

            return thinking_result

        except Exception as e:
            logger.warning(f"[ErrorRecoveryManager] Thinking engine failed: {e}")
            return None

    async def _local_analysis(self, context: RecoveryContext) -> Dict[str, Any]:
        analysis = {
            "error_category": context.error_category.name,
            "local_insights": {},
            "suggested_fixes": []
        }

        if context.error_category == ErrorCategory.COMPILATION_ERROR:
            compiler_output = context.error_details.get("compiler_output", context.error_message)
            analysis["local_insights"] = {
                "error_type": "compilation",
                "compiler_output": compiler_output[:500]
            }
            analysis["suggested_fixes"] = self._extract_fix_hints(compiler_output)

        elif context.error_category == ErrorCategory.TEST_FAILURE:
            failures = context.error_details.get("failures", [])
            analysis["local_insights"] = {
                "failure_count": len(failures),
                "error_type": "test_failure"
            }
            analysis["suggested_fixes"] = [
                {"type": "test_fix", "hint": f.get("error", "")[:100]}
                for f in failures[:5]
            ]

        elif context.error_category == ErrorCategory.TOOL_EXECUTION_ERROR:
            tool_name = context.error_details.get("tool", "unknown")
            exit_code = context.error_details.get("exit_code", -1)
            stderr = context.error_details.get("stderr", "")

            analysis["local_insights"] = {
                "tool": tool_name,
                "exit_code": exit_code,
                "error_type": "tool_execution"
            }

            if "command not found" in stderr.lower() or "not recognized" in stderr.lower():
                analysis["suggested_fixes"].append({
                    "type": "missing_tool",
                    "hint": f"{tool_name} is not installed or not in PATH"
                })
            elif "permission denied" in stderr.lower():
                analysis["suggested_fixes"].append({
                    "type": "permission",
                    "hint": "Permission denied - check file permissions"
                })
            elif "timeout" in stderr.lower():
                analysis["suggested_fixes"].append({
                    "type": "timeout",
                    "hint": "Command timed out - may need to increase timeout"
                })

        return analysis

    def _extract_fix_hints(self, compiler_output: str) -> List[Dict[str, Any]]:
        hints = []
        lines = compiler_output.split('\n')

        for line in lines:
            if "cannot find symbol" in line.lower():
                hints.append({"type": "missing_import", "hint": line.strip()})
            elif "package" in line.lower() and "does not exist" in line.lower():
                hints.append({"type": "missing_package", "hint": line.strip()})
            elif "incompatible types" in line.lower():
                hints.append({"type": "type_mismatch", "hint": line.strip()})

        return hints[:5]

    async def _llm_analysis(
        self,
        context: RecoveryContext,
        local_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.llm_client or not self.prompt_builder:
            return {
                "llm_insights": "LLM analysis not available",
                "recommended_strategy": "ANALYZE_AND_FIX",
                "confidence": 0.3,
                "specific_fixes": [],
                "reasoning": "Using local analysis only"
            }

        try:
            prompt = self.prompt_builder.build_error_analysis_prompt(
                error_category=context.error_category.name,
                error_message=context.error_message,
                error_details=context.error_details,
                local_analysis=local_analysis,
                attempt_history=[
                    {
                        "attempt": a.attempt_number,
                        "strategy": a.strategy_used.name,
                        "success": a.success,
                        "message": a.error_message[:200]
                    }
                    for a in context.attempt_history[-5:]
                ],
                current_test_code=context.current_test_code,
                target_class_info=context.target_class_info
            )

            response = await self.llm_client.generate(prompt)
            llm_result = self._parse_llm_analysis_response(response)

            return {
                "llm_insights": llm_result.get("analysis", ""),
                "recommended_strategy": llm_result.get("strategy", "ANALYZE_AND_FIX"),
                "confidence": llm_result.get("confidence", 0.5),
                "specific_fixes": llm_result.get("fixes", []),
                "reasoning": llm_result.get("reasoning", ""),
                "raw_response": response
            }
        except Exception as e:
            return {
                "llm_insights": f"LLM analysis failed: {e}",
                "recommended_strategy": "ANALYZE_AND_FIX",
                "confidence": 0.3,
                "specific_fixes": [],
                "reasoning": "Using local analysis only due to LLM error"
            }

    def _parse_llm_analysis_response(self, response: str) -> Dict[str, Any]:
        result = {
            "analysis": "",
            "strategy": "ANALYZE_AND_FIX",
            "confidence": 0.5,
            "fixes": [],
            "reasoning": ""
        }

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()

            if line_lower.startswith("analysis:") or line_lower.startswith("**analysis:**"):
                current_section = "analysis"
                result["analysis"] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_lower.startswith("strategy:") or line_lower.startswith("**strategy:**"):
                current_section = "strategy"
                strategy_text = line.split(":", 1)[1].strip() if ":" in line else ""
                result["strategy"] = strategy_text.upper().replace(" ", "_")
            elif line_lower.startswith("confidence:") or line_lower.startswith("**confidence:**"):
                current_section = "confidence"
                conf_text = line.split(":", 1)[1].strip() if ":" in line else "0.5"
                try:
                    result["confidence"] = float(conf_text.replace("%", "")) / 100 if "%" in conf_text else float(conf_text)
                except ValueError:
                    result["confidence"] = 0.5
            elif line_lower.startswith("reasoning:") or line_lower.startswith("**reasoning:**"):
                current_section = "reasoning"
                result["reasoning"] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_lower.startswith("fixes:") or line_lower.startswith("**fixes:**"):
                current_section = "fixes"
            elif current_section and line.strip():
                if current_section == "fixes":
                    if line.strip().startswith("-") or line.strip().startswith("*"):
                        result["fixes"].append(line.strip()[1:].strip())
                else:
                    result[current_section] += " " + line.strip()

        return result

    async def analyze_with_smart_context(
        self,
        error: Exception,
        error_context: Dict[str, Any],
        error_type: str = "compilation"
    ) -> Dict[str, Any]:
        """使用完整上下文进行智能 LLM 分析。

        这是增强版的错误分析，会：
        1. 收集完整的编译/测试输出
        2. 构建包含完整上下文的 Prompt
        3. 让 LLM 给出具体的行动方案
        4. 解析行动方案为可执行的步骤

        Args:
            error: 发生的异常
            error_context: 错误上下文，包含完整的输出和文件信息
            error_type: 错误类型，"compilation" 或 "test_failure"

        Returns:
            包含分析结果和行动方案的字典
        """
        if not self.llm_client or not self.prompt_builder:
            return {
                "success": False,
                "message": "LLM client not available",
                "action_plan": []
            }

        try:
            from ..core.error_context import (
                ErrorContextCollector,
                CompilationErrorContext,
                TestFailureContext,
                LLMAnalysisResult
            )
            from ..agent.tools.action_executor import parse_llm_action_plan

            if self.progress_callback:
                self.progress_callback("SMART_ANALYSIS", "正在进行智能错误分析...")

            logger.info(f"[ErrorRecoveryManager] Starting smart context analysis for {error_type}")

            if error_type == "compilation":
                context_obj = ErrorContextCollector.collect_compilation_context(
                    compiler_output=error_context.get("compiler_output", str(error)),
                    source_file=error_context.get("source_file", ""),
                    test_file=error_context.get("test_file", ""),
                    project_path=self.project_path,
                    attempt_number=len(self.recovery_history) + 1
                )

                prompt = self.prompt_builder.build_smart_compilation_analysis_prompt(
                    error_context=context_obj,
                    attempt_history=[
                        {
                            "attempt": a.attempt_number,
                            "action": a.strategy_used.name,
                            "success": a.success,
                            "message": a.error_message[:100]
                        }
                        for a in self.recovery_history[-5:]
                    ]
                )
            else:
                context_obj = ErrorContextCollector.collect_test_failure_context(
                    test_output=error_context.get("test_output", str(error)),
                    test_file=error_context.get("test_file", ""),
                    source_file=error_context.get("source_file", ""),
                    attempt_number=len(self.recovery_history) + 1
                )

                prompt = self.prompt_builder.build_smart_test_failure_analysis_prompt(
                    error_context=context_obj,
                    attempt_history=[
                        {
                            "attempt": a.attempt_number,
                            "action": a.strategy_used.name,
                            "success": a.success,
                            "message": a.error_message[:100]
                        }
                        for a in self.recovery_history[-5:]
                    ]
                )

            response = await self.llm_client.agenerate(prompt)

            analysis_result = self._parse_smart_analysis_response(response)

            action_plan = parse_llm_action_plan(response)

            result = {
                "success": True,
                "root_cause": analysis_result.get("root_cause", ""),
                "analysis": analysis_result.get("analysis", ""),
                "confidence": analysis_result.get("confidence", 0.5),
                "reasoning": analysis_result.get("reasoning", ""),
                "action_plan": [a for a in action_plan.actions],
                "raw_response": response,
                "error_context_obj": context_obj
            }

            logger.info(
                f"[ErrorRecoveryManager] Smart analysis complete - "
                f"Root cause: {result['root_cause'][:100]}, "
                f"Actions: {len(result['action_plan'])}, "
                f"Confidence: {result['confidence']:.2f}"
            )

            if self.progress_callback:
                self.progress_callback(
                    "SMART_ANALYSIS_COMPLETE",
                    f"分析完成 - 置信度: {result['confidence']:.0%}"
                )

            return result

        except Exception as e:
            logger.exception(f"[ErrorRecoveryManager] Smart analysis failed: {e}")
            return {
                "success": False,
                "message": str(e),
                "action_plan": []
            }

    def _parse_smart_analysis_response(self, response: str) -> Dict[str, Any]:
        """解析智能分析响应。

        Args:
            response: LLM 响应文本

        Returns:
            解析后的分析结果
        """
        result = {
            "root_cause": "",
            "analysis": "",
            "confidence": 0.5,
            "reasoning": ""
        }

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()

            if line_lower.startswith("root_cause:") or line_lower.startswith("**root_cause:**"):
                current_section = "root_cause"
                result["root_cause"] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_lower.startswith("analysis:") or line_lower.startswith("**analysis:**"):
                current_section = "analysis"
                result["analysis"] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_lower.startswith("confidence:") or line_lower.startswith("**confidence:**"):
                current_section = "confidence"
                conf_text = line.split(":", 1)[1].strip() if ":" in line else "0.5"
                try:
                    result["confidence"] = float(conf_text.replace("%", "")) / 100 if "%" in conf_text else float(conf_text)
                except ValueError:
                    result["confidence"] = 0.5
            elif line_lower.startswith("reasoning:") or line_lower.startswith("**reasoning:**"):
                current_section = "reasoning"
                result["reasoning"] = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line_lower.startswith("action_plan:"):
                current_section = "action_plan"
            elif current_section and line.strip():
                if current_section not in ("action_plan",):
                    result[current_section] += " " + line.strip()

        return result

    async def execute_action_plan(
        self,
        action_plan: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行 LLM 给出的行动方案。

        Args:
            action_plan: 行动方案列表
            context: 执行上下文

        Returns:
            执行结果
        """
        from ..agent.tools.action_executor import ActionExecutor, ActionPlan

        if not action_plan:
            return {
                "success": False,
                "message": "No actions to execute"
            }

        executor = ActionExecutor(
            project_path=self.project_path,
            llm_client=self.llm_client,
            progress_callback=self.progress_callback
        )

        plan = ActionPlan(
            actions=action_plan,
            confidence=context.get("confidence", 0.5)
        )

        results = await executor.execute_action_plan(plan, context)

        success_count = sum(1 for r in results if r.success)

        return {
            "success": success_count > 0,
            "message": f"Executed {success_count}/{len(results)} actions successfully",
            "results": [
                {
                    "action": r.action_type.name,
                    "success": r.success,
                    "message": r.message,
                    "modified_file": r.modified_file
                }
                for r in results
            ],
            "modified_files": [r.modified_file for r in results if r.modified_file]
        }

    def _determine_strategy(
        self,
        context: RecoveryContext,
        local_analysis: Dict[str, Any],
        llm_analysis: Dict[str, Any]
    ) -> RecoveryStrategy:
        llm_strategy_str = llm_analysis.get("recommended_strategy", "ANALYZE_AND_FIX")
        confidence = llm_analysis.get("confidence", 0.5)

        try:
            llm_strategy = RecoveryStrategy[llm_strategy_str]
            if confidence > 0.6:
                return llm_strategy
        except KeyError:
            pass

        error_category = context.error_category
        attempt_count = len([a for a in context.attempt_history
                           if a.error_category == error_category])

        if attempt_count < 2:
            return RecoveryStrategy.ANALYZE_AND_FIX

        if attempt_count >= 3:
            tried_strategies = set(a.strategy_used for a in context.attempt_history
                                 if a.error_category == error_category)

            if RecoveryStrategy.RETRY_WITH_BACKOFF not in tried_strategies:
                return RecoveryStrategy.RETRY_WITH_BACKOFF

            if RecoveryStrategy.RESET_AND_REGENERATE not in tried_strategies:
                return RecoveryStrategy.RESET_AND_REGENERATE

            if RecoveryStrategy.FALLBACK_ALTERNATIVE not in tried_strategies:
                return RecoveryStrategy.FALLBACK_ALTERNATIVE

        return RecoveryStrategy.ANALYZE_AND_FIX

    async def _execute_recovery(
        self,
        context: RecoveryContext,
        strategy: RecoveryStrategy,
        llm_analysis: Dict[str, Any],
        attempt_number: int
    ) -> Dict[str, Any]:
        if self.progress_callback:
            self.progress_callback(
                "RECOVERING",
                f"Attempt {attempt_number}: Using {strategy.name} strategy"
            )

        if strategy == RecoveryStrategy.RETRY_IMMEDIATE:
            return await self._retry_immediate(context)

        elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
            return await self._retry_with_backoff(context, attempt_number)

        elif strategy == RecoveryStrategy.ANALYZE_AND_FIX:
            return await self._analyze_and_fix(context, llm_analysis)

        elif strategy == RecoveryStrategy.SKIP_AND_CONTINUE:
            return await self._skip_and_continue(context)

        elif strategy == RecoveryStrategy.RESET_AND_REGENERATE:
            return await self._reset_and_regenerate(context, llm_analysis)

        elif strategy == RecoveryStrategy.FALLBACK_ALTERNATIVE:
            return await self._fallback_alternative(context, llm_analysis)

        elif strategy == RecoveryStrategy.ESCALATE_TO_USER:
            return {
                "success": False,
                "action": "escalate",
                "message": "Unable to recover automatically. User intervention required.",
                "should_continue": False,
                "details": {
                    "error": context.error_message,
                    "attempts": attempt_number
                }
            }

        elif strategy == RecoveryStrategy.INSTALL_DEPENDENCIES:
            return await self._install_dependencies(context, llm_analysis)

        elif strategy == RecoveryStrategy.RESOLVE_DEPENDENCIES:
            return await self._resolve_dependencies(context, llm_analysis)

        elif strategy == RecoveryStrategy.FIX_ENVIRONMENT:
            return await self._fix_environment(context, llm_analysis)

        return {
            "success": False,
            "action": "unknown_strategy",
            "message": f"Unknown recovery strategy: {strategy}",
            "should_continue": True
        }

    async def _retry_immediate(self, context: RecoveryContext) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "retry",
            "message": "Retrying immediately",
            "should_continue": True,
            "strategy": "immediate_retry"
        }

    async def _retry_with_backoff(
        self,
        context: RecoveryContext,
        attempt_number: int
    ) -> Dict[str, Any]:
        delay = min(2 ** attempt_number, 60)

        if self.progress_callback:
            self.progress_callback("RECOVERING", f"Waiting {delay}s before retry...")

        time.sleep(delay)

        return {
            "success": True,
            "action": "retry",
            "message": f"Retrying after {delay}s delay",
            "should_continue": True,
            "strategy": "backoff_retry",
            "delay": delay
        }

    async def _analyze_and_fix(
        self,
        context: RecoveryContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.llm_client or not self.prompt_builder:
            return {
                "success": False,
                "action": "fix_failed",
                "message": "LLM client not available for fix generation",
                "should_continue": True
            }

        try:
            # Phase 2: Use enhanced prompt if available
            if context.error_details and "enhanced_prompt" in context.error_details:
                logger.info("[ErrorRecoveryManager] Using Phase 2 enhanced fix prompt")
                prompt = context.error_details["enhanced_prompt"]
            else:
                prompt = self.prompt_builder.build_comprehensive_fix_prompt(
                    error_category=context.error_category.name,
                    error_message=context.error_message,
                    error_details=context.error_details,
                    local_analysis=llm_analysis.get("local_insights", {}),
                    llm_insights=llm_analysis.get("llm_insights", ""),
                    specific_fixes=llm_analysis.get("specific_fixes", []),
                    current_test_code=context.current_test_code,
                    target_class_info=context.target_class_info,
                    attempt_history=[
                        {
                            "attempt": a.attempt_number,
                            "success": a.success,
                            "message": a.error_message[:100]
                        }
                        for a in context.attempt_history[-3:]
                    ]
                )

            response = await self.llm_client.agenerate(prompt)
            fixed_code = self._extract_java_code(response)

            return {
                "success": True,
                "action": "fix",
                "message": "Generated fix using LLM",
                "should_continue": True,
                "fixed_code": fixed_code,
                "strategy": "llm_fix"
            }
        except Exception as e:
            return {
                "success": False,
                "action": "fix_failed",
                "message": f"Failed to generate fix: {e}",
                "should_continue": True,
                "strategy": "llm_fix"
            }

    async def _skip_and_continue(self, context: RecoveryContext) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "skip",
            "message": "Skipping current step and continuing",
            "should_continue": True,
            "strategy": "skip"
        }

    async def _reset_and_regenerate(
        self,
        context: RecoveryContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "reset",
            "message": "Resetting and regenerating from scratch",
            "should_continue": True,
            "strategy": "reset_regenerate",
            "clear_history": True
        }

    async def _fallback_alternative(
        self,
        context: RecoveryContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        alternatives = {
            ErrorCategory.COMPILATION_ERROR: "try_simpler_test",
            ErrorCategory.TEST_FAILURE: "reduce_test_scope",
            ErrorCategory.TOOL_EXECUTION_ERROR: "use_alternative_tool",
            ErrorCategory.GENERATION_ERROR: "change_generation_approach"
        }

        alternative = alternatives.get(context.error_category, "generic_fallback")

        return {
            "success": True,
            "action": "fallback",
            "message": f"Using alternative approach: {alternative}",
            "should_continue": True,
            "strategy": "fallback",
            "alternative": alternative
        }

    async def _install_dependencies(
        self,
        context: RecoveryContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """安装缺失的依赖（增强版，使用 LLM 分析）
        
        Args:
            context: 恢复上下文
            llm_analysis: LLM 分析结果
            
        Returns:
            恢复结果字典
        """
        if self.progress_callback:
            self.progress_callback("INSTALLING_DEPS", "正在安装缺失的依赖...")
        
        logger.info("[ErrorRecoveryManager] Installing missing dependencies")
        
        compiler_output = context.error_details.get("compiler_output", context.error_message)
        
        # 使用增强的依赖恢复处理器
        handler = DependencyRecoveryHandler(
            project_path=self.project_path,
            maven_runner=None,  # Will be created if needed
            llm_client=self.llm_client,
            prompt_builder=self.prompt_builder,
            progress_callback=self.progress_callback
        )
        
        # 尝试使用增强版安装
        result = await handler.install_missing_dependencies_enhanced(compiler_output)
        
        if result.success:
            installed_deps = result.details.get("installed_dependencies", []) if result.details else []
            dep_list = [f"{d.get('group_id')}:{d.get('artifact_id')}" for d in installed_deps]
            return {
                "success": True,
                "action": "retry",
                "message": f"Dependencies installed: {dep_list}",
                "should_continue": True,
                "strategy": "install_dependencies",
                "installed_packages": installed_deps,
                "analysis": result.details.get("analysis", "") if result.details else "",
                "confidence": result.details.get("confidence", 0.0) if result.details else 0.0
            }
        else:
            # 如果增强版失败，回退到传统方法
            logger.warning(f"[ErrorRecoveryManager] Enhanced installation failed, trying fallback: {result.error_message}")
            
            dependency_info = context.error_details.get("dependency_info", {})
            missing_packages = dependency_info.get("missing_packages", [])
            is_test_dependency = dependency_info.get("is_test_dependency", False)
            
            if not missing_packages:
                from .error_classification import detect_missing_dependencies
                dependency_info = detect_missing_dependencies(compiler_output)
                missing_packages = dependency_info.get("missing_packages", [])
                is_test_dependency = dependency_info.get("is_test_dependency", False)
            
            if not missing_packages:
                logger.warning("[ErrorRecoveryManager] No missing packages detected, falling back to resolve all")
                return await self._resolve_dependencies(context, llm_analysis)
            
            fallback_handler = DependencyRecoveryHandler(
                project_path=self.project_path,
                progress_callback=self.progress_callback
            )
            
            fallback_result = await fallback_handler.install_missing_dependencies(
                missing_packages=missing_packages,
                is_test_dependency=is_test_dependency
            )
            
            if fallback_result.success:
                return {
                    "success": True,
                    "action": "retry",
                    "message": f"Dependencies installed (fallback): {missing_packages}",
                    "should_continue": True,
                    "strategy": "install_dependencies",
                    "installed_packages": missing_packages
                }
            else:
                suggestions = fallback_handler.suggest_pom_additions(missing_packages)
                return {
                    "success": False,
                    "action": "escalate",
                    "message": f"Failed to install dependencies: {fallback_result.error_message}",
                    "should_continue": False,
                    "strategy": "install_dependencies",
                    "suggested_pom_additions": suggestions,
                    "missing_packages": missing_packages
                }

    async def _resolve_dependencies(
        self,
        context: RecoveryContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析 Maven 依赖
        
        Args:
            context: 恢复上下文
            llm_analysis: LLM 分析结果
            
        Returns:
            恢复结果字典
        """
        if self.progress_callback:
            self.progress_callback("RESOLVING_DEPS", "正在解析 Maven 依赖...")
        
        logger.info("[ErrorRecoveryManager] Resolving Maven dependencies")
        
        handler = DependencyRecoveryHandler(
            project_path=self.project_path,
            progress_callback=self.progress_callback
        )
        
        result = await handler.resolve_dependencies()
        
        if result.success:
            return {
                "success": True,
                "action": "retry",
                "message": "Dependencies resolved successfully",
                "should_continue": True,
                "strategy": "resolve_dependencies"
            }
        else:
            return {
                "success": False,
                "action": "escalate",
                "message": f"Failed to resolve dependencies: {result.error_message}",
                "should_continue": False,
                "strategy": "resolve_dependencies"
            }

    async def _fix_environment(
        self,
        context: RecoveryContext,
        llm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """修复环境问题
        
        Args:
            context: 恢复上下文
            llm_analysis: LLM 分析结果
            
        Returns:
            恢复结果字典
        """
        if self.progress_callback:
            self.progress_callback("FIXING_ENV", "正在修复环境问题...")
        
        logger.info("[ErrorRecoveryManager] Attempting to fix environment issues")
        
        from .error_classification import ErrorSubCategory
        sub_category_str = context.error_details.get("sub_category", "")
        
        try:
            sub_category = ErrorSubCategory[sub_category_str] if sub_category_str else ErrorSubCategory.UNKNOWN
        except KeyError:
            sub_category = ErrorSubCategory.UNKNOWN
        
        if sub_category in (ErrorSubCategory.MISSING_DEPENDENCY, ErrorSubCategory.MAVEN_DEPENDENCY_ERROR):
            return await self._resolve_dependencies(context, llm_analysis)
        
        return {
            "success": False,
            "action": "escalate",
            "message": "Environment issue cannot be fixed automatically",
            "should_continue": False,
            "strategy": "fix_environment"
        }

    def _extract_java_code(self, response: str) -> str:
        return CodeExtractor.extract_java_code(response)

    def get_recovery_summary(self) -> Dict[str, Any]:
        if not self.recovery_history:
            return {"message": "No recovery attempts recorded"}

        total_attempts = len(self.recovery_history)
        successful_attempts = len([a for a in self.recovery_history if a.success])

        by_category = {}
        for attempt in self.recovery_history:
            cat = attempt.error_category.name
            if cat not in by_category:
                by_category[cat] = {"total": 0, "success": 0}
            by_category[cat]["total"] += 1
            if attempt.success:
                by_category[cat]["success"] += 1

        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": successful_attempts / total_attempts if total_attempts > 0 else 0,
            "by_category": by_category,
            "recent_attempts": [
                {
                    "category": a.error_category.name,
                    "strategy": a.strategy_used.name,
                    "success": a.success,
                    "timestamp": a.timestamp
                }
                for a in self.recovery_history[-10:]
            ]
        }

    def clear_history(self):
        self.recovery_history.clear()
        self.state_preserver.clear_history()
        self.reset_error_tracking()


@contextmanager
def safe_execution_context(
    operation_name: str,
    recovery_manager: Optional[RecoveryManager] = None,
    on_error: Optional[Callable] = None
):
    """Context manager for safe execution with error handling."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation_name}: {e}")

        if on_error:
            try:
                on_error(e)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")

        raise


def create_recovery_manager(
    max_retries: int = 3,
    backoff_base: float = 1.0
) -> RecoveryManager:
    """Create a recovery manager with common settings."""
    return RecoveryManager(
        max_retries=max_retries,
        backoff_base=backoff_base
    )


def classify_error(error: Exception) -> ErrorCategory:
    """Classify an error."""
    return ErrorClassifier.classify(error)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    return ErrorClassifier.is_retryable(error)


class DependencyRecoveryHandler:
    """依赖问题恢复处理器
    
    处理 Maven 依赖缺失、解析失败等问题。
    支持 LLM 增强的依赖分析和自动添加。
    """
    
    def __init__(
        self,
        project_path: str,
        maven_runner: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        timeout: int = 300,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ):
        """初始化依赖恢复处理器
        
        Args:
            project_path: 项目路径
            maven_runner: MavenRunner 实例（可选）
            llm_client: LLM 客户端（可选，用于智能依赖分析）
            prompt_builder: Prompt 构建器（可选）
            timeout: 超时时间（秒）
            progress_callback: 进度回调函数
        """
        self.project_path = project_path
        self.maven_runner = maven_runner
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.timeout = timeout
        self.progress_callback = progress_callback
        
        self._resolution_attempts = 0
        self._max_resolution_attempts = 3
        
        # 初始化新组件
        try:
            from ..tools.dependency_analyzer import DependencyAnalyzer
            from ..tools.pom_editor import PomEditor
            from ..tools.dependency_installer import DependencyInstaller
            
            self.dependency_analyzer = DependencyAnalyzer(llm_client, prompt_builder)
            self.pom_editor = PomEditor(project_path)
            self.dependency_installer = DependencyInstaller(
                project_path, 
                maven_runner, 
                timeout,
                progress_callback
            )
        except ImportError as e:
            logger.warning(f"[DependencyRecovery] Failed to import new components: {e}")
            self.dependency_analyzer = None
            self.pom_editor = None
            self.dependency_installer = None
    
    async def resolve_dependencies(self) -> RecoveryResult:
        """解析并下载所有依赖
        
        Returns:
            RecoveryResult 对象
        """
        self._resolution_attempts += 1
        
        if self.progress_callback:
            self.progress_callback("RESOLVING_DEPS", "正在解析 Maven 依赖...")
        
        logger.info(f"[DependencyRecovery] Resolving dependencies - Attempt: {self._resolution_attempts}")
        
        try:
            if self.maven_runner:
                success, output = await self.maven_runner.resolve_dependencies_async()
            else:
                success, output = await self._run_maven_resolve()
            
            if success:
                logger.info("[DependencyRecovery] Dependencies resolved successfully")
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RESOLVE_DEPENDENCIES,
                    attempts_made=self._resolution_attempts,
                    recovered_data={"output": output},
                    action="retry",
                    should_continue=True
                )
            else:
                logger.warning(f"[DependencyRecovery] Failed to resolve dependencies: {output[:200]}")
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.RESOLVE_DEPENDENCIES,
                    attempts_made=self._resolution_attempts,
                    error_message=f"Failed to resolve dependencies: {output[:200]}",
                    action="escalate",
                    should_continue=False
                )
                
        except Exception as e:
            logger.exception(f"[DependencyRecovery] Exception during dependency resolution: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RESOLVE_DEPENDENCIES,
                attempts_made=self._resolution_attempts,
                error_message=str(e),
                action="retry",
                should_continue=True
            )
    
    async def resolve_test_dependencies(self) -> RecoveryResult:
        """解析并下载测试依赖
        
        运行 mvn test-compile -DskipTests 来下载测试依赖
        
        Returns:
            RecoveryResult 对象
        """
        if self.progress_callback:
            self.progress_callback("RESOLVING_TEST_DEPS", "正在解析测试依赖...")
        
        logger.info("[DependencyRecovery] Resolving test dependencies")
        
        try:
            if self.maven_runner:
                success, output = await self.maven_runner.resolve_test_dependencies_async()
            else:
                success, output = await self._run_maven_test_compile()
            
            if success:
                logger.info("[DependencyRecovery] Test dependencies resolved successfully")
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                    attempts_made=1,
                    recovered_data={"output": output},
                    action="retry",
                    should_continue=True
                )
            else:
                logger.warning(f"[DependencyRecovery] Failed to resolve test dependencies: {output[:200]}")
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                    attempts_made=1,
                    error_message=f"Failed to resolve test dependencies: {output[:200]}",
                    action="escalate",
                    should_continue=False
                )
                
        except Exception as e:
            logger.exception(f"[DependencyRecovery] Exception during test dependency resolution: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                attempts_made=1,
                error_message=str(e),
                action="retry",
                should_continue=True
            )
    
    async def install_missing_dependencies(
        self,
        missing_packages: List[str],
        is_test_dependency: bool = False
    ) -> RecoveryResult:
        """安装缺失的依赖
        
        Args:
            missing_packages: 缺失的包名列表
            is_test_dependency: 是否为测试依赖
            
        Returns:
            RecoveryResult 对象
        """
        logger.info(f"[DependencyRecovery] Installing missing dependencies: {missing_packages}")
        
        if is_test_dependency:
            return await self.resolve_test_dependencies()
        else:
            return await self.resolve_dependencies()
    
    async def install_missing_dependencies_enhanced(
        self,
        compiler_output: str
    ) -> RecoveryResult:
        """增强的依赖安装流程（使用 LLM 分析）
        
        流程:
        1. 使用 LLM 分析编译错误
        2. 识别缺失的依赖
        3. 添加依赖到 pom.xml
        4. 执行 mvn clean install
        5. 验证安装结果
        
        Args:
            compiler_output: 编译器输出
            
        Returns:
            RecoveryResult 对象
        """
        if not self.dependency_analyzer or not self.dependency_installer:
            logger.warning("[DependencyRecovery] Enhanced components not available, falling back to basic method")
            return await self.resolve_dependencies()
        
        if self.progress_callback:
            self.progress_callback("ANALYZING_DEPS", "正在分析缺失的依赖...")
        
        logger.info("[DependencyRecovery] Starting enhanced dependency installation")
        
        try:
            # 1. LLM 分析
            pom_content = ""
            if self.pom_editor:
                try:
                    pom_content = self.pom_editor.read_pom()
                except Exception as e:
                    logger.warning(f"[DependencyRecovery] Failed to read pom.xml: {e}")
            
            analysis_result = await self.dependency_analyzer.analyze_missing_dependencies(
                compiler_output,
                pom_content
            )
            
            missing_deps = analysis_result.get("missing_dependencies", [])
            confidence = analysis_result.get("confidence", 0.0)
            
            logger.info(f"[DependencyRecovery] LLM analysis found {len(missing_deps)} dependencies (confidence: {confidence:.2f})")
            
            if not missing_deps:
                logger.warning("[DependencyRecovery] No missing dependencies detected by LLM")
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                    attempts_made=1,
                    error_message="No missing dependencies detected",
                    action="skip",
                    should_continue=True
                )
            
            # 2. 验证依赖
            valid_deps = []
            for dep in missing_deps:
                is_valid, errors = self.dependency_analyzer.validate_dependency(dep)
                if is_valid:
                    valid_deps.append(dep)
                else:
                    logger.warning(f"[DependencyRecovery] Invalid dependency: {dep}, errors: {errors}")
            
            if not valid_deps:
                logger.warning("[DependencyRecovery] No valid dependencies after validation")
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                    attempts_made=1,
                    error_message="No valid dependencies found",
                    action="skip",
                    should_continue=True
                )
            
            # 3. 去重
            unique_deps = self.dependency_analyzer.deduplicate_dependencies(valid_deps)
            
            if self.progress_callback:
                self.progress_callback(
                    "INSTALLING_DEPS", 
                    f"正在安装 {len(unique_deps)} 个依赖..."
                )
            
            # 4. 安装依赖
            install_result = await self.dependency_installer.install_dependencies(
                unique_deps,
                skip_tests=True
            )
            
            if install_result.success:
                logger.info(f"[DependencyRecovery] Successfully installed {len(install_result.installed_deps)} dependencies")
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                    attempts_made=1,
                    recovered_data={
                        "installed_dependencies": install_result.installed_deps,
                        "analysis": analysis_result.get("analysis", ""),
                        "confidence": confidence
                    },
                    action="retry",
                    should_continue=True
                )
            else:
                logger.warning(f"[DependencyRecovery] Failed to install dependencies: {install_result.message}")
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                    attempts_made=1,
                    error_message=install_result.message,
                    action="escalate",
                    should_continue=False,
                    details={
                        "failed_dependencies": install_result.failed_deps,
                        "suggested_fixes": analysis_result.get("suggested_fixes", []),
                        "backup_path": install_result.backup_path
                    }
                )
                
        except Exception as e:
            logger.exception(f"[DependencyRecovery] Enhanced installation failed: {e}")
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.INSTALL_DEPENDENCIES,
                attempts_made=1,
                error_message=str(e),
                action="retry",
                should_continue=True
            )
    
    async def _run_maven_resolve(self) -> Tuple[bool, str]:
        """运行 Maven dependency:resolve 命令
        
        Returns:
            (success, output) 元组
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "mvn", "dependency:resolve", "-q",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return False, f"Timeout after {self.timeout} seconds"
            
            output = stderr.decode() if stderr else stdout.decode() if stdout else ""
            return process.returncode == 0, output
            
        except FileNotFoundError:
            return False, "Maven executable not found"
        except Exception as e:
            return False, str(e)
    
    async def _run_maven_test_compile(self) -> Tuple[bool, str]:
        """运行 Maven test-compile -DskipTests 命令
        
        Returns:
            (success, output) 元组
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "mvn", "test-compile", "-DskipTests", "-q",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return False, f"Timeout after {self.timeout} seconds"
            
            output = stderr.decode() if stderr else stdout.decode() if stdout else ""
            return process.returncode == 0, output
            
        except FileNotFoundError:
            return False, "Maven executable not found"
        except Exception as e:
            return False, str(e)
    
    def check_pom_has_test_dependencies(self) -> Dict[str, bool]:
        """检查 pom.xml 是否包含常见测试依赖
        
        Returns:
            字典，键为依赖名，值为是否包含
        """
        pom_path = Path(self.project_path) / "pom.xml"
        if not pom_path.exists():
            return {}
        
        try:
            content = pom_path.read_text(encoding='utf-8')
            return {
                "junit_jupiter": "junit-jupiter" in content,
                "mockito": "mockito" in content,
                "assertj": "assertj" in content,
                "hamcrest": "hamcrest" in content,
            }
        except Exception as e:
            logger.warning(f"[DependencyRecovery] Failed to read pom.xml: {e}")
            return {}
    
    def suggest_pom_additions(self, missing_packages: List[str]) -> List[str]:
        """建议添加到 pom.xml 的依赖
        
        Args:
            missing_packages: 缺失的包名列表
            
        Returns:
            建议添加的 Maven 依赖 XML 片段列表
        """
        suggestions = []
        
        dependency_templates = {
            "org.junit": '''<dependency>
    <groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter</artifactId>
    <scope>test</scope>
</dependency>''',
            "org.mockito": '''<dependency>
    <groupId>org.mockito</groupId>
    <artifactId>mockito-core</artifactId>
    <scope>test</scope>
</dependency>''',
            "org.assertj": '''<dependency>
    <groupId>org.assertj</groupId>
    <artifactId>assertj-core</artifactId>
    <scope>test</scope>
</dependency>''',
            "org.hamcrest": '''<dependency>
    <groupId>org.hamcrest</groupId>
    <artifactId>hamcrest</artifactId>
    <scope>test</scope>
</dependency>''',
        }
        
        for pkg in missing_packages:
            for prefix, template in dependency_templates.items():
                if pkg.startswith(prefix):
                    if template not in suggestions:
                        suggestions.append(template)
                    break
        
        return suggestions
    
    def reset_attempts(self):
        """重置尝试计数"""
        self._resolution_attempts = 0
