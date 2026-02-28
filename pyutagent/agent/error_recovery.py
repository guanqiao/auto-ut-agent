"""Error recovery manager for handling all types of errors with AI assistance."""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable

from ..tools.error_analyzer import CompilationErrorAnalyzer, ErrorFixGenerator
from ..tools.failure_analyzer import TestFailureAnalyzer, FailureFixGenerator
from .prompts import PromptBuilder


class ErrorCategory(Enum):
    """Categories of errors that can occur."""
    COMPILATION_ERROR = auto()
    TEST_FAILURE = auto()
    TOOL_EXECUTION_ERROR = auto()
    PARSING_ERROR = auto()
    GENERATION_ERROR = auto()
    COVERAGE_ANALYSIS_ERROR = auto()
    FILE_IO_ERROR = auto()
    NETWORK_ERROR = auto()
    LLM_API_ERROR = auto()
    UNKNOWN_ERROR = auto()


class RecoveryStrategy(Enum):
    """Strategies for error recovery."""
    RETRY_IMMEDIATE = auto()  # Retry immediately
    RETRY_WITH_BACKOFF = auto()  # Retry with exponential backoff
    ANALYZE_AND_FIX = auto()  # Analyze with local tools + LLM
    SKIP_AND_CONTINUE = auto()  # Skip this step and continue
    RESET_AND_REGENERATE = auto()  # Reset and regenerate from scratch
    FALLBACK_ALTERNATIVE = auto()  # Use alternative approach
    ESCALATE_TO_USER = auto()  # Ask user for help (last resort)


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
        llm_client,
        project_path: str,
        prompt_builder: PromptBuilder,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ):
        """Initialize error recovery manager.
        
        Args:
            llm_client: LLM client for AI analysis
            project_path: Project path
            prompt_builder: Prompt builder
            progress_callback: Optional callback for progress updates
        """
        self.llm_client = llm_client
        self.project_path = project_path
        self.prompt_builder = prompt_builder
        self.progress_callback = progress_callback
        
        # Local analyzers
        self.compilation_analyzer = CompilationErrorAnalyzer()
        self.compilation_fixer = ErrorFixGenerator(self.compilation_analyzer)
        self.failure_analyzer = TestFailureAnalyzer(project_path)
        self.failure_fixer = FailureFixGenerator(self.failure_analyzer)
        
        # Recovery history
        self.recovery_history: List[RecoveryAttempt] = []
        self.max_total_attempts = 50  # Prevent infinite loops
        
    def categorize_error(self, error_message: str, error_details: Dict[str, Any]) -> ErrorCategory:
        """Categorize an error based on message and details."""
        error_lower = error_message.lower()
        
        # Check for compilation errors
        if any(keyword in error_lower for keyword in [
            "cannot find symbol", "package .* does not exist",
            "incompatible types", "expected", "illegal",
            "compilation", "javac", "syntax error"
        ]):
            return ErrorCategory.COMPILATION_ERROR
        
        # Check for test failures
        if any(keyword in error_lower for keyword in [
            "assertion", "test failed", "nullpointer",
            "exception", "wanted but not invoked", "verification"
        ]):
            return ErrorCategory.TEST_FAILURE
        
        # Check for tool execution errors
        if any(keyword in error_lower for keyword in [
            "command not found", "exit code", "process failed",
            "mvn error", "maven", "subprocess"
        ]):
            return ErrorCategory.TOOL_EXECUTION_ERROR
        
        # Check for parsing errors
        if any(keyword in error_lower for keyword in [
            "parse", "syntax", "unexpected token", "tree-sitter"
        ]):
            return ErrorCategory.PARSING_ERROR
        
        # Check for generation errors
        if any(keyword in error_lower for keyword in [
            "generate", "llm", "model", "token", "completion"
        ]):
            return ErrorCategory.GENERATION_ERROR
        
        # Check for LLM API errors
        if any(keyword in error_lower for keyword in [
            "api key", "rate limit", "timeout", "connection",
            "openai", "anthropic", "deepseek"
        ]):
            return ErrorCategory.LLM_API_ERROR
        
        # Check for file IO errors
        if any(keyword in error_lower for keyword in [
            "no such file", "permission denied", "ioerror",
            "file not found", "cannot read", "cannot write"
        ]):
            return ErrorCategory.FILE_IO_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    async def recover(
        self,
        error: Exception,
        error_context: Dict[str, Any],
        current_test_code: Optional[str] = None,
        target_class_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Attempt to recover from an error.
        
        Args:
            error: The exception that occurred
            error_context: Additional context about the error
            current_test_code: Current test code (if applicable)
            target_class_info: Target class information
            
        Returns:
            Recovery result with action and updated state
        """
        error_message = str(error)
        error_category = self.categorize_error(error_message, error_context)
        
        # Check if we've exceeded max attempts
        category_attempts = len([a for a in self.recovery_history 
                                 if a.error_category == error_category])
        if category_attempts >= self.max_total_attempts:
            return {
                "success": False,
                "action": "stop",
                "message": f"Exceeded maximum recovery attempts ({self.max_total_attempts})",
                "should_continue": False
            }
        
        # Build recovery context
        context = RecoveryContext(
            error_category=error_category,
            error_message=error_message,
            error_details=error_context,
            current_test_code=current_test_code,
            target_class_info=target_class_info,
            attempt_history=self.recovery_history.copy()
        )
        
        # Step 1: Local analysis
        local_analysis = await self._local_analysis(context)
        
        # Step 2: LLM deep analysis (combine local + raw error)
        llm_analysis = await self._llm_analysis(context, local_analysis)
        
        # Step 3: Determine recovery strategy
        strategy = self._determine_strategy(context, local_analysis, llm_analysis)
        
        # Step 4: Execute recovery
        attempt_number = category_attempts + 1
        result = await self._execute_recovery(context, strategy, llm_analysis, attempt_number)
        
        # Record attempt
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
        """Perform local analysis of the error."""
        analysis = {
            "error_category": context.error_category.name,
            "local_insights": {},
            "suggested_fixes": []
        }
        
        if context.error_category == ErrorCategory.COMPILATION_ERROR:
            # Use compilation error analyzer
            compiler_output = context.error_details.get("compiler_output", context.error_message)
            error_analysis = self.compilation_analyzer.analyze(compiler_output)
            
            analysis["local_insights"] = {
                "error_count": len(error_analysis.errors),
                "error_types": [e.error_type.name for e in error_analysis.errors],
                "summary": error_analysis.summary,
                "fix_strategy": error_analysis.fix_strategy,
                "priority": error_analysis.priority
            }
            analysis["suggested_fixes"] = [
                {
                    "type": e.error_type.name,
                    "line": e.line_number,
                    "hint": e.fix_hint,
                    "suggestions": e.suggestions
                }
                for e in error_analysis.errors[:5]  # Top 5 errors
            ]
            analysis["full_analysis"] = error_analysis
            
        elif context.error_category == ErrorCategory.TEST_FAILURE:
            # Use test failure analyzer
            failure_analysis = self.failure_analyzer.analyze()
            
            analysis["local_insights"] = {
                "failure_count": len(failure_analysis.failures),
                "failure_types": [f.failure_type.name for f in failure_analysis.failures],
                "summary": failure_analysis.summary,
                "fix_strategy": failure_analysis.fix_strategy,
                "priority": failure_analysis.priority,
                "total_tests": failure_analysis.total_tests,
                "passed_tests": failure_analysis.passed_tests,
                "failed_tests": failure_analysis.failed_tests
            }
            analysis["suggested_fixes"] = [
                {
                    "type": f.failure_type.name,
                    "test_method": f.test_method,
                    "hint": f.fix_hint,
                    "suggestions": f.suggestions
                }
                for f in failure_analysis.failures[:5]  # Top 5 failures
            ]
            analysis["full_analysis"] = failure_analysis
            
        elif context.error_category == ErrorCategory.TOOL_EXECUTION_ERROR:
            # Analyze tool execution error
            tool_name = context.error_details.get("tool", "unknown")
            exit_code = context.error_details.get("exit_code", -1)
            stderr = context.error_details.get("stderr", "")
            
            analysis["local_insights"] = {
                "tool": tool_name,
                "exit_code": exit_code,
                "error_type": "tool_execution"
            }
            
            # Common tool error patterns
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
    
    async def _llm_analysis(
        self,
        context: RecoveryContext,
        local_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM for deep error analysis."""
        
        # Build comprehensive context for LLM
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
                for a in context.attempt_history[-5:]  # Last 5 attempts
            ],
            current_test_code=context.current_test_code,
            target_class_info=context.target_class_info
        )
        
        try:
            response = await self.llm_client.generate(prompt)
            
            # Parse LLM response
            # Expected format: analysis + recommended_action + confidence
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
            # If LLM analysis fails, fall back to local analysis only
            return {
                "llm_insights": f"LLM analysis failed: {e}",
                "recommended_strategy": "ANALYZE_AND_FIX",
                "confidence": 0.3,
                "specific_fixes": [],
                "reasoning": "Using local analysis only due to LLM error",
                "raw_response": ""
            }
    
    def _parse_llm_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response."""
        result = {
            "analysis": "",
            "strategy": "ANALYZE_AND_FIX",
            "confidence": 0.5,
            "fixes": [],
            "reasoning": ""
        }
        
        # Try to extract structured information
        # Look for sections like "Analysis:", "Strategy:", "Confidence:", etc.
        
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
        """Determine the best recovery strategy."""
        
        # Get LLM recommendation
        llm_strategy_str = llm_analysis.get("recommended_strategy", "ANALYZE_AND_FIX")
        confidence = llm_analysis.get("confidence", 0.5)
        
        # Try to parse LLM strategy
        try:
            llm_strategy = RecoveryStrategy[llm_strategy_str]
            if confidence > 0.6:
                return llm_strategy
        except KeyError:
            pass
        
        # Fall back to rule-based strategy selection
        error_category = context.error_category
        attempt_count = len([a for a in context.attempt_history 
                           if a.error_category == error_category])
        
        # First few attempts: try immediate fix
        if attempt_count < 2:
            return RecoveryStrategy.ANALYZE_AND_FIX
        
        # If multiple attempts failed, try different strategies
        if attempt_count >= 3:
            # Check what strategies have been tried
            tried_strategies = set(a.strategy_used for a in context.attempt_history
                                 if a.error_category == error_category)
            
            if RecoveryStrategy.RETRY_WITH_BACKOFF not in tried_strategies:
                return RecoveryStrategy.RETRY_WITH_BACKOFF
            
            if RecoveryStrategy.RESET_AND_REGENERATE not in tried_strategies:
                return RecoveryStrategy.RESET_AND_REGENERATE
            
            if RecoveryStrategy.FALLBACK_ALTERNATIVE not in tried_strategies:
                return RecoveryStrategy.FALLBACK_ALTERNATIVE
        
        # Default to analyze and fix
        return RecoveryStrategy.ANALYZE_AND_FIX
    
    async def _execute_recovery(
        self,
        context: RecoveryContext,
        strategy: RecoveryStrategy,
        llm_analysis: Dict[str, Any],
        attempt_number: int
    ) -> Dict[str, Any]:
        """Execute the recovery strategy."""
        
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
        """Retry the failed operation immediately."""
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
        """Retry with exponential backoff."""
        delay = min(2 ** attempt_number, 60)  # Max 60 seconds
        
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
        """Analyze and fix the error using LLM."""
        
        # Build comprehensive fix prompt
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
        
        try:
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
        """Skip the current step and continue."""
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
        """Reset and regenerate from scratch."""
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
        """Use alternative approach."""
        
        # Determine alternative based on error category
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
        """Extract Java code from LLM response."""
        import re
        
        # Try to extract code from markdown code blocks
        code_block_pattern = r'```(?:java)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, return the whole response
        return response.strip()
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get summary of all recovery attempts."""
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
        """Clear recovery history."""
        self.recovery_history.clear()