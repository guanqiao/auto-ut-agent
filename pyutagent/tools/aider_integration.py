"""Aider integration for precise test code editing with enhanced fault tolerance.

This module integrates Aider's Search/Replace editing strategy with the
Java unit test generation workflow. It provides:
- LLM prompt templates for generating precise edits
- Edit generation based on compilation errors and test failures
- Feedback loop for iterative code improvement
- Integration with existing error and failure analyzers
- Enhanced fault tolerance with retry, fallback, and circuit breaker patterns
- Architect/Editor dual-model pattern support
- Multi-file editing capabilities
- Multiple edit format support (diff, udiff, whole, diff-fenced)
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
from ..core.error_recovery import (
    RecoveryManager, StatePreserver, ErrorClassifier,
    create_recovery_manager, is_retryable_error
)
from ..core.retry_manager import (
    RetryManager, CircuitBreaker, TimeoutManager,
    retry_with_backoff, circuit_breaker, get_retry_manager
)
from .fallback_strategies import (
    FallbackManager, FallbackLevel, FallbackResult,
    create_fallback_manager, apply_fallback
)
from .edit_formats import (
    EditFormat, edit_format_registry, EditBlock
)
from .architect_editor import (
    ArchitectEditor, ArchitectEditorResult, ArchitectMode,
    EditPlan
)
from .multi_file_editor import (
    MultiFileEditor, MultiFileEditResult, DependencyAnalyzer,
    FileNode
)
from ..llm.client import LLMClient
from ..core.config import AiderConfig

logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Strategy for fixing code issues."""
    SINGLE_EDIT = auto()  # Fix one issue at a time
    BATCH_EDIT = auto()   # Fix multiple related issues
    FULL_REGENERATE = auto()  # Regenerate entire file (fallback)
    ARCHITECT_EDITOR = auto()  # Use Architect/Editor pattern
    MULTI_FILE = auto()   # Fix across multiple files


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
    # New fields for enhanced features
    edit_plan: Optional[Any] = None  # Architect edit plan
    affected_files: List[str] = field(default_factory=list)  # For multi-file edits
    cost_summary: Optional[Dict[str, Any]] = None  # Cost tracking for Architect/Editor


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
    # New fields
    related_files: Dict[str, str] = field(default_factory=dict)
    preferred_format: Optional[EditFormat] = None


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

    # Format-specific system prompts
    UDIFF_SYSTEM_PROMPT = """You are an expert Java developer specializing in fixing test code.
Your task is to fix compilation errors and test failures using unified diff format.

Rules:
1. Use unified diff format with proper headers
2. Only show changed lines with context
3. Ensure the diff is syntactically correct
4. Include all necessary changes

Format:
```diff
--- original.java
+++ modified.java
@@ -line,count +line,count @@
-removed line
+added line
 context line
```"""

    WHOLE_FILE_SYSTEM_PROMPT = """You are an expert Java developer specializing in fixing test code.
Your task is to provide the complete fixed test code.

Rules:
1. Return the entire file content
2. Include all necessary imports
3. Ensure proper Java syntax
4. Wrap the code in markdown code blocks

Format:
```java
[complete file content]
```"""

    def __init__(self, edit_format: EditFormat = EditFormat.DIFF):
        """Initialize prompt builder.

        Args:
            edit_format: Preferred edit format
        """
        self.edit_format = edit_format

    def get_system_prompt(self) -> str:
        """Get system prompt based on edit format."""
        if self.edit_format == EditFormat.UDIFF:
            return self.UDIFF_SYSTEM_PROMPT
        elif self.edit_format == EditFormat.WHOLE:
            return self.WHOLE_FILE_SYSTEM_PROMPT
        return self.EDIT_SYSTEM_PROMPT

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
Provide fixes using the specified format. Focus on:
1. Adding missing imports
2. Fixing syntax errors
3. Resolving symbol not found errors
4. Correcting type mismatches

Generate the minimal changes needed to fix all compilation errors."""

        return self.get_system_prompt(), user_prompt

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
Provide fixes using the specified format. Focus on:
1. Fixing assertion values
2. Correcting mock setup/verification
3. Initializing objects properly
4. Handling exceptions correctly
5. Fixing test logic errors

Generate the minimal changes needed to make all tests pass."""

        return self.get_system_prompt(), user_prompt

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
Use the specified format to add new test methods or modify existing ones."""

        return self.get_system_prompt(), user_prompt

    def _format_error_analysis(self, analysis: ErrorAnalysis) -> str:
        """Format error analysis for prompt."""
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
        """Format failure analysis for prompt."""
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
        config: Optional[AiderConfig] = None,
        architect_llm: Optional[LLMClient] = None,
        editor_llm: Optional[LLMClient] = None
    ):
        """Initialize Aider code fixer.

        Args:
            llm_client: Primary LLM client for generating fixes
            config: Aider configuration
            architect_llm: Optional separate LLM for Architect role
            editor_llm: Optional separate LLM for Editor role
        """
        self.config = config or AiderConfig()
        self.llm_client = llm_client
        self.architect_llm = architect_llm or llm_client
        self.editor_llm = editor_llm or llm_client

        # Initialize components
        self.editor = TestCodeEditor()
        self.validator = EditValidator()

        # Determine edit format
        if self.config.preferred_format:
            self.edit_format = self.config.preferred_format
        elif self.config.auto_detect_format:
            # Get model name from llm_client if available
            model_name = getattr(llm_client, 'model', 'gpt-4')
            self.edit_format = edit_format_registry.get_preferred_format(model_name)
        else:
            self.edit_format = EditFormat.DIFF

        self.prompt_builder = AiderPromptBuilder(self.edit_format)

        # Initialize Architect/Editor if enabled
        self.architect_editor: Optional[ArchitectEditor] = None
        if self.config.use_architect_editor:
            self.architect_editor = ArchitectEditor(
                architect_llm=self.architect_llm,
                editor_llm=self.editor_llm,
                mode=self.config.architect_mode
            )

        # Initialize Multi-file editor if enabled
        self.multi_file_editor: Optional[MultiFileEditor] = None
        if self.config.enable_multi_file:
            # Need project path - will be set later
            self.multi_file_editor = None

        # Enhanced fault tolerance components
        self.recovery_manager = create_recovery_manager(max_retries=self.config.max_attempts)
        self.retry_manager = get_retry_manager()
        self.state_preserver = StatePreserver()
        self.fallback_manager: Optional[FallbackManager] = None
        if self.config.enable_fallback:
            self.fallback_manager = create_fallback_manager(llm_client)

        # Circuit breaker for LLM calls
        self.circuit_breaker: Optional[CircuitBreaker] = None
        if self.config.enable_circuit_breaker:
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
            "circuit_breaker_opens": 0,
            "architect_editor_uses": 0,
            "multi_file_uses": 0
        }

    async def fix_compilation_errors(
        self,
        test_code: str,
        error_analysis: ErrorAnalysis,
        class_info: Optional[Dict[str, Any]] = None,
        related_files: Optional[Dict[str, str]] = None
    ) -> FixResult:
        """Fix compilation errors in test code.

        Args:
            test_code: Current test code with errors
            error_analysis: Error analysis from CompilationErrorAnalyzer
            class_info: Optional class information
            related_files: Optional related files for multi-file editing

        Returns:
            FixResult with success status and fixed code
        """
        context = EditContext(
            original_code=test_code,
            error_analysis=error_analysis,
            class_info=class_info,
            related_files=related_files or {},
            preferred_format=self.edit_format
        )

        # Use multi-file editing if enabled and related files provided
        if self.config.enable_multi_file and related_files:
            return await self._fix_multi_file(context)

        # Use Architect/Editor if enabled
        if self.config.use_architect_editor and self.architect_editor:
            return await self._fix_with_architect_editor(context)

        # Standard single-file fix
        system_prompt, user_prompt = self.prompt_builder.build_compilation_fix_prompt(context)
        return await self._attempt_fix(context, system_prompt, user_prompt)

    async def fix_test_failures(
        self,
        test_code: str,
        failure_analysis: FailureAnalysis,
        class_info: Optional[Dict[str, Any]] = None,
        related_files: Optional[Dict[str, str]] = None
    ) -> FixResult:
        """Fix test failures in test code.

        Args:
            test_code: Current test code with failures
            failure_analysis: Failure analysis from TestFailureAnalyzer
            class_info: Optional class information
            related_files: Optional related files for multi-file editing

        Returns:
            FixResult with success status and fixed code
        """
        context = EditContext(
            original_code=test_code,
            failure_analysis=failure_analysis,
            class_info=class_info,
            related_files=related_files or {},
            preferred_format=self.edit_format
        )

        # Use multi-file editing if enabled and related files provided
        if self.config.enable_multi_file and related_files:
            return await self._fix_multi_file(context)

        # Use Architect/Editor if enabled
        if self.config.use_architect_editor and self.architect_editor:
            return await self._fix_with_architect_editor(context)

        # Standard single-file fix
        system_prompt, user_prompt = self.prompt_builder.build_test_failure_fix_prompt(context)
        return await self._attempt_fix(context, system_prompt, user_prompt)

    async def improve_coverage(
        self,
        test_code: str,
        uncovered_lines: List[int],
        class_info: Optional[Dict[str, Any]] = None
    ) -> FixResult:
        """Add tests to improve coverage."""
        context = EditContext(
            original_code=test_code,
            class_info=class_info,
            preferred_format=self.edit_format
        )

        system_prompt, user_prompt = self.prompt_builder.build_coverage_improvement_prompt(
            context, uncovered_lines
        )

        return await self._attempt_fix(context, system_prompt, user_prompt)

    async def _fix_with_architect_editor(self, context: EditContext) -> FixResult:
        """Fix using Architect/Editor pattern."""
        if not self.architect_editor:
            raise ValueError("Architect/Editor not initialized")

        self.stats["architect_editor_uses"] += 1

        result = await self.architect_editor.generate_fix(
            context=context.__dict__,
            original_code=context.original_code,
            error_analysis=str(context.error_analysis) if context.error_analysis else None,
            failure_analysis=str(context.failure_analysis) if context.failure_analysis else None
        )

        if result.success and result.diff_text:
            # Parse the diff text
            edits, format_used = edit_format_registry.auto_detect_and_parse(
                result.diff_text,
                context.target_file
            )

            if edits:
                # Apply edits
                current_code = context.original_code
                for edit in edits:
                    if edit.original:
                        current_code = current_code.replace(edit.original, edit.modified, 1)
                    else:
                        current_code = edit.modified

                # Validate
                validation = self.validator.validate_edit(
                    context.original_code,
                    current_code,
                    is_test_code=True
                )

                return FixResult(
                    success=validation.is_valid,
                    original_code=context.original_code,
                    fixed_code=current_code,
                    edit_result=None,
                    validation_result=validation,
                    strategy_used=FixStrategy.ARCHITECT_EDITOR,
                    attempts=1,
                    fix_summary=f"Fixed using Architect/Editor ({format_used.value} format)",
                    edit_plan=result.edit_plan,
                    cost_summary={
                        "architect_time": result.architect_time,
                        "editor_time": result.editor_time,
                        "total_cost": result.total_cost
                    } if self.config.track_costs else None
                )

        return FixResult(
            success=False,
            original_code=context.original_code,
            fixed_code=context.original_code,
            strategy_used=FixStrategy.ARCHITECT_EDITOR,
            error_message=f"Architect/Editor failed: {result.error_message}"
        )

    async def _fix_multi_file(self, context: EditContext) -> FixResult:
        """Fix across multiple files."""
        # Multi-file editing requires project path
        # For now, fall back to single-file fix
        logger.warning("Multi-file editing not fully implemented, falling back to single-file")

        system_prompt, user_prompt = self.prompt_builder.build_compilation_fix_prompt(context) \
            if context.error_analysis else \
            self.prompt_builder.build_test_failure_fix_prompt(context)

        return await self._attempt_fix(context, system_prompt, user_prompt)

    async def _attempt_fix(
        self,
        context: EditContext,
        system_prompt: str,
        user_prompt: str,
        use_circuit_breaker: bool = True,
        use_fallback: bool = True
    ) -> FixResult:
        """Attempt to fix code with retries and enhanced fault tolerance."""
        self.stats["total_fixes"] += 1
        current_code = context.original_code

        # Save initial state
        state_version = self.state_preserver.save_state(
            {"code": current_code, "context": context},
            label="fix_attempt_start"
        )

        for attempt in range(1, self.config.max_attempts + 1):
            logger.info(f"Fix attempt {attempt}/{self.config.max_attempts}")
            context.iteration = attempt

            try:
                # Generate fix from LLM
                diff_text = await self._generate_fix_with_protection(
                    user_prompt, system_prompt, use_circuit_breaker
                )

                # Parse edits using format registry
                edits, format_used = edit_format_registry.auto_detect_and_parse(
                    diff_text,
                    context.target_file
                )

                if not edits:
                    logger.warning("No valid edits found in response")
                    context.previous_attempts.append(f"Attempt {attempt}: No valid edits")
                    user_prompt += "\n\nPlease provide valid edits in the correct format."
                    continue

                # Apply edits
                modified_code = current_code
                for edit in edits:
                    if edit.original:
                        modified_code = modified_code.replace(edit.original, edit.modified, 1)
                    else:
                        modified_code = edit.modified

                # Validate
                validation_result = self.validator.validate_edit(
                    context.original_code,
                    modified_code,
                    is_test_code=True
                )

                if validation_result.is_valid:
                    logger.info(f"Fix successful on attempt {attempt}")
                    self.stats["successful_fixes"] += 1

                    return FixResult(
                        success=True,
                        original_code=context.original_code,
                        fixed_code=modified_code,
                        edit_result=None,
                        validation_result=validation_result,
                        strategy_used=FixStrategy.SINGLE_EDIT,
                        attempts=attempt,
                        fix_summary=f"Fixed in {attempt} attempt(s) using {format_used.value} format"
                    )
                else:
                    logger.warning(f"Validation failed: {validation_result.errors}")
                    context.previous_attempts.append(f"Attempt {attempt}: Validation failed")
                    error_msgs = "; ".join(str(e) for e in validation_result.errors)
                    user_prompt += f"\n\nPrevious fix had validation errors: {error_msgs}"
                    current_code = modified_code

            except Exception as e:
                logger.exception(f"Error during fix attempt {attempt}")
                context.previous_attempts.append(f"Attempt {attempt}: Exception - {str(e)}")

                if not is_retryable_error(e):
                    logger.error(f"Non-retryable error encountered: {e}")
                    break

        # Try fallback
        if use_fallback and self.fallback_manager:
            fallback_result = await self._try_fallback(context, current_code, state_version)
            if fallback_result.success:
                self.stats["fallback_uses"] += 1
                return FixResult(
                    success=True,
                    original_code=context.original_code,
                    fixed_code=fallback_result.code,
                    strategy_used=FixStrategy.FULL_REGENERATE,
                    attempts=self.config.max_attempts,
                    fix_summary=f"Fixed using fallback: {fallback_result.message}"
                )

        return FixResult(
            success=False,
            original_code=context.original_code,
            fixed_code=current_code,
            strategy_used=FixStrategy.SINGLE_EDIT,
            attempts=self.config.max_attempts,
            error_message=f"Failed after {self.config.max_attempts} attempts",
            fix_summary="; ".join(context.previous_attempts)
        )

    async def _generate_fix_with_protection(
        self,
        user_prompt: str,
        system_prompt: str,
        use_circuit_breaker: bool
    ) -> str:
        """Generate fix with circuit breaker and timeout protection."""
        async def _call_llm():
            return await self.llm_client.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

        if use_circuit_breaker and self.circuit_breaker:
            return await self.circuit_breaker.call(_call_llm)
        else:
            return await TimeoutManager.with_timeout(
                _call_llm(),
                timeout=self.config.timeout_seconds,
                timeout_message=f"LLM call timed out after {self.config.timeout_seconds}s"
            )

    async def _try_fallback(
        self,
        context: EditContext,
        current_code: str,
        state_version: int
    ) -> FallbackResult:
        """Try fallback strategies."""
        if not self.fallback_manager:
            return FallbackResult(
                success=False,
                code=current_code,
                level_used=FallbackLevel.MANUAL,
                message="No fallback manager available"
            )

        fallback_result = await self.fallback_manager.execute_with_fallback(
            original_code=current_code,
            error_analysis=context.error_analysis,
            failure_analysis=context.failure_analysis,
            context=context.class_info
        )

        if not fallback_result.success:
            original_state = self.state_preserver.restore_state(state_version)
            if original_state:
                logger.info("Restored original state after fallback failure")

        return fallback_result

    def get_stats(self) -> Dict[str, Any]:
        """Get fixer statistics."""
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
            "circuit_breaker_opens": 0,
            "architect_editor_uses": 0,
            "multi_file_uses": 0
        }

    def apply_direct_edit(self, test_code: str, diff_text: str) -> FixResult:
        """Apply a direct edit without LLM generation."""
        # Parse using format registry
        edits, format_used = edit_format_registry.auto_detect_and_parse(diff_text)

        if not edits:
            return FixResult(
                success=False,
                original_code=test_code,
                fixed_code=test_code,
                error_message="No valid edits found in diff text"
            )

        # Apply edits
        modified_code = test_code
        for edit in edits:
            if edit.original:
                modified_code = modified_code.replace(edit.original, edit.modified, 1)
            else:
                modified_code = edit.modified

        validation_result = self.validator.validate_edit(
            test_code,
            modified_code,
            is_test_code=True
        )

        return FixResult(
            success=validation_result.is_valid,
            original_code=test_code,
            fixed_code=modified_code,
            edit_result=None,
            validation_result=validation_result,
            fix_summary="Direct edit applied" if validation_result.is_valid else "Edit applied but validation failed"
        )


class AiderTestGenerator:
    """Test generator that uses Aider-style editing for iterative improvement."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[AiderConfig] = None,
        fixer: Optional[AiderCodeFixer] = None
    ):
        """Initialize Aider test generator.

        Args:
            llm_client: LLM client
            config: Optional Aider configuration
            fixer: Optional AiderCodeFixer instance
        """
        self.llm_client = llm_client
        self.config = config or AiderConfig()
        self.fixer = fixer or AiderCodeFixer(llm_client, self.config)
        self.editor = TestCodeEditor()
        self.validator = EditValidator()

    async def generate_initial_test(
        self,
        class_info: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate initial test code."""
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

        response = await self.llm_client.complete(
            messages=[
                {"role": "system", "content": system_prompt or default_system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract code from markdown if present
        test_code = self._extract_code_from_markdown(response)

        return test_code

    async def iterate_with_feedback(
        self,
        test_code: str,
        feedback_callback: Callable[[str], tuple[bool, Any]],
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Iterate on test code with feedback."""
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
        """Extract code from markdown code blocks."""
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
    config: Optional[AiderConfig] = None,
    max_attempts: int = 3
) -> FixResult:
    """Fix compilation errors using Aider-style editing."""
    cfg = config or AiderConfig(max_attempts=max_attempts)
    fixer = AiderCodeFixer(llm_client, cfg)
    return await fixer.fix_compilation_errors(test_code, error_analysis)


async def fix_test_failures_with_aider(
    test_code: str,
    failure_analysis: FailureAnalysis,
    llm_client: LLMClient,
    config: Optional[AiderConfig] = None,
    max_attempts: int = 3
) -> FixResult:
    """Fix test failures using Aider-style editing."""
    cfg = config or AiderConfig(max_attempts=max_attempts)
    fixer = AiderCodeFixer(llm_client, cfg)
    return await fixer.fix_test_failures(test_code, failure_analysis)


def apply_diff_edit(test_code: str, diff_text: str, format_type: Optional[EditFormat] = None) -> FixResult:
    """Apply a diff edit to test code."""
    edits, detected_format = edit_format_registry.auto_detect_and_parse(diff_text)

    if not edits:
        return FixResult(
            success=False,
            original_code=test_code,
            fixed_code=test_code,
            error_message="No valid edits found in diff text"
        )

    # Apply edits
    modified_code = test_code
    for edit in edits:
        if edit.original:
            modified_code = modified_code.replace(edit.original, edit.modified, 1)
        else:
            modified_code = edit.modified

    return FixResult(
        success=True,
        original_code=test_code,
        fixed_code=modified_code,
        edit_result=None,
        fix_summary=f"Direct edit applied using {detected_format.value} format"
    )
