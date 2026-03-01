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
import re
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
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
)

logger = logging.getLogger(__name__)

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
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        project_path: str = "",
        prompt_builder: Optional[Any] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        max_total_attempts: int = 50
    ):
        self.llm_client = llm_client
        self.project_path = project_path
        self.prompt_builder = prompt_builder
        self.progress_callback = progress_callback
        self.max_total_attempts = max_total_attempts

        self.recovery_history: List[RecoveryAttempt] = []
        self.state_preserver = StatePreserver()

    def categorize_error(self, error_message: str, error_details: Dict[str, Any]) -> ErrorCategory:
        return ErrorClassifier.categorize_error(error_message, error_details)

    async def recover(
        self,
        error: Exception,
        error_context: Dict[str, Any],
        current_test_code: Optional[str] = None,
        target_class_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        error_message = str(error)
        error_category = self.categorize_error(error_message, error_context)

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

        llm_analysis = await self._llm_analysis(context, local_analysis)

        strategy = self._determine_strategy(context, local_analysis, llm_analysis)

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

            response = await self.llm_client.generate(prompt)
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

    def _extract_java_code(self, response: str) -> str:
        code_block_pattern = r'```(?:java)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        return response.strip()

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
