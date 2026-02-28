"""Aider integration for precise test code editing with enhanced fault tolerance.

This module integrates Aider's Search/Replace editing strategy with the
Java unit test generation workflow. It provides:
- LLM prompt templates for generating precise edits
- Edit generation based on compilation errors and test failures
- Feedback loop for iterative code improvement
- Integration with existing error and failure analyzers
- Enhanced fault tolerance with retry, fallback, and circuit breaker patterns
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto
from pathlib import Path

from .code_editor import (
    CodeEditor, TestCodeEditor, DiffParser, EditOperation, 
    EditResult, create_edit_prompt
)
from .edit_validator import EditValidator, ValidationResult, validate_test_code
from .error_analyzer import ErrorAnalysis, CompilationError, ErrorType
from .failure_analyzer import FailureAnalysis, TestFailure, FailureType
from .error_recovery import (
    RecoveryManager, StatePreserver, ErrorClassifier,
    create_recovery_manager, is_retryable_error
)
from .retry_manager import (
    RetryManager, CircuitBreaker, TimeoutManager,
    retry_with_backoff, circuit_breaker, get_retry_manager
)
from .fallback_strategies import (
    FallbackManager, FallbackLevel, FallbackResult,
    create_fallback_manager, apply_fallback
)
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Strategy for fixing code issues."""
    SINGLE_EDIT = auto()  # Fix one issue at a time
    BATCH_EDIT = auto()   # Fix multiple related issues
    FULL_REGENERATE = auto()  # Regenerate entire file (fallback)


@dataclass
class FixResult:
    """Result of a fix operation."""
    success: bool
    original_code: str
    fixed_code: str
    edit_result: Optional[EditResult] = None
    validation_result: Optional[ValidationResult] = None
    strategy_used: FixStrategy = FixStrategy.SINGLE_EDIT
    attempts: int = 0
    error_message: str = ""
    fix_summary: str = ""


@dataclass
class EditContext:
    """Context for generating edits."""
    original_code: str
    error_analysis: Optional[ErrorAnalysis] = None
    failure_analysis: Optional[FailureAnalysis] = None
    class_info: Optional[Dict[str, Any]] = None
    target_file: Optional[str] = None
    iteration: int = 0
    previous_attempts: List[str] = field(default_factory=list)


class AiderPromptBuilder:
    """Builds prompts for LLM to generate Aider-style edits."""
    
    # System prompt for edit generation
    EDIT_SYSTEM_PROMPT = """You are an expert Java developer specializing in fixing test code.
Your task is to fix compilation errors and test failures using precise Search/Replace edits.

Rules:
1. Use the exact Search/Replace format shown below
2. Search text must match the original code exactly (including whitespace)
3. Only modify the necessary parts to fix the issue
4. Prefer minimal changes over large rewrites
5. Ensure the fixed code maintains proper Java syntax
6. Include all necessary imports

Format:
<<<<<<< SEARCH
[exact code to find - must match original]
=======
[replacement code]
>>>>>>> REPLACE

You can provide multiple SEARCH/REPLACE blocks if needed.
Be precise and careful - the search must match exactly for the edit to work."""

    def __init__(self):
        """Initialize prompt builder."""
        pass
    
    def build_compilation_fix_prompt(
        self,
        context: EditContext
    ) -> tuple[str, str]:
        """Build prompt for fixing compilation errors.
        
        Args:
            context: Edit context with error information
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if not context.error_analysis:
            raise ValueError("Error analysis is required for compilation fix")
        
        error_context = self._format_error_analysis(context.error_analysis)
        
        user_prompt = f"""Fix the following Java test code compilation errors.

## Error Analysis
{error_context}

## Current Test Code
```java
{context.original_code}
```

## Task
Provide fixes using Search/Replace format. Focus on:
1. Adding missing imports
2. Fixing syntax errors
3. Resolving symbol not found errors
4. Correcting type mismatches

Generate the minimal changes needed to fix all compilation errors."""

        return self.EDIT_SYSTEM_PROMPT, user_prompt
    
    def build_test_failure_fix_prompt(
        self,
        context: EditContext
    ) -> tuple[str, str]:
        """Build prompt for fixing test failures.
        
        Args:
            context: Edit context with failure information
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        if not context.failure_analysis:
            raise ValueError("Failure analysis is required for test fix")
        
        failure_context = self._format_failure_analysis(context.failure_analysis)
        
        user_prompt = f"""Fix the following Java test code to resolve test failures.

## Failure Analysis
{failure_context}

## Current Test Code
```java
{context.original_code}
```

## Task
Provide fixes using Search/Replace format. Focus on:
1. Fixing assertion values
2. Correcting mock setup/verification
3. Initializing objects properly
4. Handling exceptions correctly
5. Fixing test logic errors

Generate the minimal changes needed to make all tests pass."""

        return self.EDIT_SYSTEM_PROMPT, user_prompt
    
    def build_coverage_improvement_prompt(
        self,
        context: EditContext,
        uncovered_lines: List[int]
    ) -> tuple[str, str]:
        """Build prompt for improving test coverage.
        
        Args:
            context: Edit context
            uncovered_lines: List of uncovered line numbers
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_prompt = f"""Add test cases to improve code coverage.

## Uncovered Lines
{uncovered_lines}

## Current Test Code
```java
{context.original_code}
```

## Task
Add new test methods to cover the uncovered lines.
Use Search/Replace format to add new test methods or modify existing ones.

Format:
<<<<<<< SEARCH
[existing code where new tests will be added]
=======
[existing code + new test methods]
>>>>>>> REPLACE"""

        return self.EDIT_SYSTEM_PROMPT, user_prompt
    
    def _format_error_analysis(self, analysis: ErrorAnalysis) -> str:
        """Format error analysis for prompt.
        
        Args:
            analysis: Error analysis result
            
        Returns:
            Formatted string
        """
        lines = [
            f"Summary: {analysis.summary}",
            f"Fix Strategy: {analysis.fix_strategy}",
            f"Priority: {analysis.priority}",
            "",
            "Errors:"
        ]
        
        for i, error in enumerate(analysis.errors[:5], 1):
            lines.append(f"\n{i}. {error.error_type.name}")
            lines.append(f"   Message: {error.message}")
            if error.line_number:
                lines.append(f"   Line: {error.line_number}")
            if error.error_token:
                lines.append(f"   Token: {error.error_token}")
            if error.fix_hint:
                lines.append(f"   Hint: {error.fix_hint}")
        
        if len(analysis.errors) > 5:
            lines.append(f"\n... and {len(analysis.errors) - 5} more errors")
        
        return '\n'.join(lines)
    
    def _format_failure_analysis(self, analysis: FailureAnalysis) -> str:
        """Format failure analysis for prompt.
        
        Args:
            analysis: Failure analysis result
            
        Returns:
            Formatted string
        """
        lines = [
            f"Summary: {analysis.summary}",
            f"Fix Strategy: {analysis.fix_strategy}",
            f"Priority: {analysis.priority}",
            f"Total Tests: {analysis.total_tests}",
            f"Passed: {analysis.passed_tests}",
            f"Failed: {analysis.failed_tests}",
            f"Errors: {analysis.error_tests}",
            "",
            "Failures:"
        ]
        
        for i, failure in enumerate(analysis.failures[:5], 1):
            lines.append(f"\n{i}. {failure.failure_type.name}")
            lines.append(f"   Test: {failure.test_class}.{failure.test_method}")
            lines.append(f"   Message: {failure.message[:100]}")
            if failure.line_number:
                lines.append(f"   Line: {failure.line_number}")
            if failure.fix_hint:
                lines.append(f"   Hint: {failure.fix_hint}")
        
        if len(analysis.failures) > 5:
            lines.append(f"\n... and {len(analysis.failures) - 5} more failures")
        
        return '\n'.join(lines)


class AiderCodeFixer:
    """Main class for fixing code using Aider-style editing with enhanced fault tolerance."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        max_attempts: int = 3,
        enable_fallback: bool = True,
        enable_circuit_breaker: bool = True,
        timeout_seconds: float = 120.0
    ):
        """Initialize Aider code fixer.
        
        Args:
            llm_client: LLM client for generating fixes
            max_attempts: Maximum number of fix attempts
            enable_fallback: Whether to enable fallback strategies
            enable_circuit_breaker: Whether to enable circuit breaker
            timeout_seconds: Timeout for LLM calls
        """
        self.llm_client = llm_client
        self.max_attempts = max_attempts
        self.enable_fallback = enable_fallback
        self.timeout_seconds = timeout_seconds
        self.editor = TestCodeEditor()
        self.validator = EditValidator()
        self.prompt_builder = AiderPromptBuilder()
        
        # Enhanced fault tolerance components
        self.recovery_manager = create_recovery_manager(max_retries=max_attempts)
        self.retry_manager = get_retry_manager()
        self.state_preserver = StatePreserver()
        self.fallback_manager: Optional[FallbackManager] = None
        if enable_fallback:
            self.fallback_manager = create_fallback_manager(llm_client)
        
        # Circuit breaker for LLM calls
        self.circuit_breaker: Optional[CircuitBreaker] = None
        if enable_circuit_breaker:
            self.circuit_breaker = self.retry_manager.get_circuit_breaker(
                "aider_llm_calls",
                failure_threshold=5,
                recovery_timeout=60.0
            )
        
        # Statistics
        self.stats = {
            "total_fixes": 0,
            "successful_fixes": 0,
            "fallback_uses": 0,
            "circuit_breaker_opens": 0
        }
    
    async def fix_compilation_errors(
        self,
        test_code: str,
        error_analysis: ErrorAnalysis,
        class_info: Optional[Dict[str, Any]] = None
    ) -> FixResult:
        """Fix compilation errors in test code.
        
        Args:
            test_code: Current test code with errors
            error_analysis: Error analysis from CompilationErrorAnalyzer
            class_info: Optional class information
            
        Returns:
            FixResult with success status and fixed code
        """
        context = EditContext(
            original_code=test_code,
            error_analysis=error_analysis,
            class_info=class_info
        )
        
        system_prompt, user_prompt = self.prompt_builder.build_compilation_fix_prompt(context)
        
        return await self._attempt_fix(context, system_prompt, user_prompt)
    
    async def fix_test_failures(
        self,
        test_code: str,
        failure_analysis: FailureAnalysis,
        class_info: Optional[Dict[str, Any]] = None
    ) -> FixResult:
        """Fix test failures in test code.
        
        Args:
            test_code: Current test code with failures
            failure_analysis: Failure analysis from TestFailureAnalyzer
            class_info: Optional class information
            
        Returns:
            FixResult with success status and fixed code
        """
        context = EditContext(
            original_code=test_code,
            failure_analysis=failure_analysis,
            class_info=class_info
        )
        
        system_prompt, user_prompt = self.prompt_builder.build_test_failure_fix_prompt(context)
        
        return await self._attempt_fix(context, system_prompt, user_prompt)
    
    async def improve_coverage(
        self,
        test_code: str,
        uncovered_lines: List[int],
        class_info: Optional[Dict[str, Any]] = None
    ) -> FixResult:
        """Add tests to improve coverage.
        
        Args:
            test_code: Current test code
            uncovered_lines: List of uncovered line numbers
            class_info: Optional class information
            
        Returns:
            FixResult with success status and updated code
        """
        context = EditContext(
            original_code=test_code,
            class_info=class_info
        )
        
        system_prompt, user_prompt = self.prompt_builder.build_coverage_improvement_prompt(
            context, uncovered_lines
        )
        
        return await self._attempt_fix(context, system_prompt, user_prompt)
    
    async def _attempt_fix(
        self,
        context: EditContext,
        system_prompt: str,
        user_prompt: str,
        use_circuit_breaker: bool = True,
        use_fallback: bool = True
    ) -> FixResult:
        """Attempt to fix code with retries and enhanced fault tolerance.
        
        Args:
            context: Edit context
            system_prompt: System prompt for LLM
            user_prompt: User prompt for LLM
            use_circuit_breaker: Whether to use circuit breaker
            use_fallback: Whether to use fallback strategies
            
        Returns:
            FixResult with success status
        """
        self.stats["total_fixes"] += 1
        current_code = context.original_code
        
        # Save initial state for potential rollback
        state_version = self.state_preserver.save_state(
            {"code": current_code, "context": context},
            label="fix_attempt_start"
        )
        
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"Fix attempt {attempt}/{self.max_attempts}")
            context.iteration = attempt
            
            try:
                # Generate fix from LLM with circuit breaker and timeout
                diff_text = await self._generate_fix_with_protection(
                    user_prompt, system_prompt, use_circuit_breaker
                )
                
                # Apply the fix
                edit_result = self.editor.apply_test_fixes(
                    current_code,
                    {},  # No additional analysis needed, diff contains all info
                    diff_text
                )
                
                if not edit_result.success:
                    logger.warning(f"Edit failed: {edit_result.error_message}")
                    context.previous_attempts.append(f"Attempt {attempt}: {edit_result.error_message}")
                    
                    # Add failure to prompt for next attempt
                    user_prompt += f"\n\nPrevious attempt failed: {edit_result.error_message}"
                    user_prompt += "\nPlease try again with corrected Search/Replace blocks."
                    continue
                
                # Validate the result
                validation_result = self.validator.validate_edit(
                    context.original_code,
                    edit_result.modified_content,
                    is_test_code=True
                )
                
                if validation_result.is_valid:
                    logger.info(f"Fix successful on attempt {attempt}")
                    
                    return FixResult(
                        success=True,
                        original_code=context.original_code,
                        fixed_code=edit_result.modified_content,
                        edit_result=edit_result,
                        validation_result=validation_result,
                        strategy_used=FixStrategy.SINGLE_EDIT,
                        attempts=attempt,
                        fix_summary=f"Fixed in {attempt} attempt(s)"
                    )
                else:
                    logger.warning(f"Validation failed: {validation_result.errors}")
                    context.previous_attempts.append(f"Attempt {attempt}: Validation failed")
                    
                    # Add validation errors to prompt
                    error_msgs = "; ".join(e.message for e in validation_result.errors if e.severity == "error")
                    user_prompt += f"\n\nPrevious fix had validation errors: {error_msgs}"
                    user_prompt += "\nPlease fix these issues and try again."
                    
                    # Use the modified code as base for next attempt
                    current_code = edit_result.modified_content
                    
            except Exception as e:
                logger.exception(f"Error during fix attempt {attempt}")
                context.previous_attempts.append(f"Attempt {attempt}: Exception - {str(e)}")
                
                # Check if error is retryable
                if not is_retryable_error(e):
                    logger.error(f"Non-retryable error encountered: {e}")
                    break
        
        # All attempts failed - try fallback strategies
        if use_fallback and self.fallback_manager:
            logger.info("Attempting fallback strategies...")
            fallback_result = await self._try_fallback(
                context, current_code, state_version
            )
            if fallback_result.success:
                self.stats["fallback_uses"] += 1
                return FixResult(
                    success=True,
                    original_code=context.original_code,
                    fixed_code=fallback_result.code,
                    strategy_used=FixStrategy.FULL_REGENERATE,
                    attempts=self.max_attempts,
                    fix_summary=f"Fixed using fallback: {fallback_result.message}"
                )
        
        # All attempts and fallbacks failed
        logger.error(f"All {self.max_attempts} fix attempts failed")
        
        return FixResult(
            success=False,
            original_code=context.original_code,
            fixed_code=current_code,
            strategy_used=FixStrategy.SINGLE_EDIT,
            attempts=self.max_attempts,
            error_message=f"Failed after {self.max_attempts} attempts",
            fix_summary="; ".join(context.previous_attempts)
        )
    
    async def _generate_fix_with_protection(
        self,
        user_prompt: str,
        system_prompt: str,
        use_circuit_breaker: bool
    ) -> str:
        """Generate fix with circuit breaker and timeout protection.
        
        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            use_circuit_breaker: Whether to use circuit breaker
            
        Returns:
            Generated diff text
        """
        async def _call_llm():
            return await self.llm_client.agenerate(user_prompt, system_prompt)
        
        if use_circuit_breaker and self.circuit_breaker:
            # Use circuit breaker
            return await self.circuit_breaker.call(_call_llm)
        else:
            # Use timeout only
            return await TimeoutManager.with_timeout(
                _call_llm(),
                timeout=self.timeout_seconds,
                timeout_message=f"LLM call timed out after {self.timeout_seconds}s"
            )
    
    async def _try_fallback(
        self,
        context: EditContext,
        current_code: str,
        state_version: int
    ) -> FallbackResult:
        """Try fallback strategies.
        
        Args:
            context: Edit context
            current_code: Current code state
            state_version: Saved state version for rollback
            
        Returns:
            FallbackResult
        """
        if not self.fallback_manager:
            return FallbackResult(
                success=False,
                code=current_code,
                level_used=FallbackLevel.MANUAL,
                message="No fallback manager available"
            )
        
        # Try fallback strategies
        fallback_result = await self.fallback_manager.execute_with_fallback(
            original_code=current_code,
            error_analysis=context.error_analysis,
            failure_analysis=context.failure_analysis,
            context=context.class_info
        )
        
        if not fallback_result.success:
            # Restore original state
            original_state = self.state_preserver.restore_state(state_version)
            if original_state:
                logger.info("Restored original state after fallback failure")
        
        return fallback_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fixer statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        if self.circuit_breaker:
            stats["circuit_breaker"] = self.circuit_breaker.get_stats()
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_fixes": 0,
            "successful_fixes": 0,
            "fallback_uses": 0,
            "circuit_breaker_opens": 0
        }
    
    def apply_direct_edit(
        self,
        test_code: str,
        diff_text: str
    ) -> FixResult:
        """Apply a direct edit without LLM generation.
        
        Args:
            test_code: Current test code
            diff_text: Diff text with SEARCH/REPLACE blocks
            
        Returns:
            FixResult with success status
        """
        edit_result = self.editor.apply_test_fixes(test_code, {}, diff_text)
        
        if edit_result.success:
            validation_result = self.validator.validate_edit(
                test_code,
                edit_result.modified_content,
                is_test_code=True
            )
            
            return FixResult(
                success=validation_result.is_valid,
                original_code=test_code,
                fixed_code=edit_result.modified_content,
                edit_result=edit_result,
                validation_result=validation_result,
                fix_summary="Direct edit applied" if validation_result.is_valid else "Edit applied but validation failed"
            )
        
        return FixResult(
            success=False,
            original_code=test_code,
            fixed_code=test_code,
            edit_result=edit_result,
            error_message=edit_result.error_message
        )


class AiderTestGenerator:
    """Test generator that uses Aider-style editing for iterative improvement."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        fixer: Optional[AiderCodeFixer] = None
    ):
        """Initialize Aider test generator.
        
        Args:
            llm_client: LLM client
            fixer: Optional AiderCodeFixer instance
        """
        self.llm_client = llm_client
        self.fixer = fixer or AiderCodeFixer(llm_client)
        self.editor = TestCodeEditor()
        self.validator = EditValidator()
    
    async def generate_initial_test(
        self,
        class_info: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate initial test code.
        
        Args:
            class_info: Information about the class to test
            system_prompt: Optional system prompt
            
        Returns:
            Generated test code
        """
        default_system_prompt = """You are a Java unit test expert. Generate JUnit 5 test code.
Follow best practices:
- Use @Test, @BeforeEach annotations
- Use @DisplayName annotation for each test method to describe the test purpose
- Use meaningful test method names
- Include assertions
- Mock external dependencies
- Cover normal cases, edge cases, and error cases"""
        
        methods_str = "\n".join([
            f"- {m.get('name', 'unknown')}({', '.join(f'{t} {n}' for t, n in m.get('parameters', []))}): {m.get('return_type', 'void')}"
            for m in class_info.get('methods', [])
        ])
        
        user_prompt = f"""Generate JUnit 5 unit tests for the following Java class:

Package: {class_info.get('package', '')}
Class: {class_info.get('name', '')}

Methods:
{methods_str}

Generate a complete test class with all necessary imports.
Return only the Java code without explanations."""
        
        test_code = await self.llm_client.agenerate(
            user_prompt,
            system_prompt or default_system_prompt
        )
        
        # Extract code from markdown if present
        test_code = self._extract_code_from_markdown(test_code)
        
        return test_code
    
    async def iterate_with_feedback(
        self,
        test_code: str,
        feedback_callback: Callable[[str], tuple[bool, Any]],
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Iterate on test code with feedback.
        
        Args:
            test_code: Initial test code
            feedback_callback: Function that takes code and returns (success, analysis)
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary with final result
        """
        current_code = test_code
        iteration = 0
        history = []
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Iteration {iteration}/{max_iterations}")
            
            # Get feedback
            success, analysis = feedback_callback(current_code)
            
            if success:
                logger.info(f"Test code is valid after {iteration} iteration(s)")
                return {
                    'success': True,
                    'final_code': current_code,
                    'iterations': iteration,
                    'history': history
                }
            
            # Try to fix based on analysis
            if isinstance(analysis, ErrorAnalysis):
                fix_result = await self.fixer.fix_compilation_errors(current_code, analysis)
            elif isinstance(analysis, FailureAnalysis):
                fix_result = await self.fixer.fix_test_failures(current_code, analysis)
            else:
                logger.warning(f"Unknown analysis type: {type(analysis)}")
                break
            
            history.append({
                'iteration': iteration,
                'code': current_code,
                'analysis': analysis,
                'fix_result': fix_result
            })
            
            if fix_result.success:
                current_code = fix_result.fixed_code
            else:
                logger.error(f"Fix failed: {fix_result.error_message}")
                break
        
        return {
            'success': False,
            'final_code': current_code,
            'iterations': iteration,
            'history': history,
            'error': f"Failed to fix after {max_iterations} iterations"
        }
    
    def _extract_code_from_markdown(self, text: str) -> str:
        """Extract code from markdown code blocks.
        
        Args:
            text: Text that may contain markdown code blocks
            
        Returns:
            Extracted code or original text
        """
        import re
        
        # Look for Java code block
        pattern = r'```java\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Look for generic code block
        pattern = r'```\n(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        return text.strip()


# Convenience functions for direct use

async def fix_compilation_errors_with_aider(
    test_code: str,
    error_analysis: ErrorAnalysis,
    llm_client: LLMClient,
    max_attempts: int = 3
) -> FixResult:
    """Fix compilation errors using Aider-style editing.
    
    Args:
        test_code: Test code with compilation errors
        error_analysis: Error analysis
        llm_client: LLM client
        max_attempts: Maximum fix attempts
        
    Returns:
        FixResult
    """
    fixer = AiderCodeFixer(llm_client, max_attempts)
    return await fixer.fix_compilation_errors(test_code, error_analysis)


async def fix_test_failures_with_aider(
    test_code: str,
    failure_analysis: FailureAnalysis,
    llm_client: LLMClient,
    max_attempts: int = 3
) -> FixResult:
    """Fix test failures using Aider-style editing.
    
    Args:
        test_code: Test code with failures
        failure_analysis: Failure analysis
        llm_client: LLM client
        max_attempts: Maximum fix attempts
        
    Returns:
        FixResult
    """
    fixer = AiderCodeFixer(llm_client, max_attempts)
    return await fixer.fix_test_failures(test_code, failure_analysis)


def apply_diff_edit(test_code: str, diff_text: str) -> FixResult:
    """Apply a diff edit to test code.
    
    Args:
        test_code: Current test code
        diff_text: Diff text with SEARCH/REPLACE blocks
        
    Returns:
        FixResult
    """
    fixer = AiderCodeFixer(None)  # No LLM needed for direct edit
    return fixer.apply_direct_edit(test_code, diff_text)
