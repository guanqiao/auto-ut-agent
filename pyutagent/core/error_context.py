"""Error Context - Rich context collection for intelligent error analysis.

This module provides comprehensive error context collection to enable
LLM-based intelligent error analysis and recovery.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CompilationErrorDetail:
    """Single compilation error detail."""
    file_path: str
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    error_type: str = ""
    error_message: str = ""
    source_line: str = ""
    suggestion: str = ""


@dataclass
class CompilationErrorContext:
    """Compilation error context for LLM analysis."""
    
    compiler_output: str = ""
    error_lines: List[str] = field(default_factory=list)
    errors: List[CompilationErrorDetail] = field(default_factory=list)
    missing_imports: List[str] = field(default_factory=list)
    missing_dependencies: List[str] = field(default_factory=list)
    syntax_errors: List[str] = field(default_factory=list)
    type_errors: List[str] = field(default_factory=list)
    
    source_file: str = ""
    test_file: str = ""
    source_code: str = ""
    test_code: str = ""
    
    project_path: str = ""
    pom_content: str = ""
    
    attempt_number: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0 or len(self.error_lines) > 0
    
    @property
    def error_count(self) -> int:
        return len(self.errors) if self.errors else len(self.error_lines)
    
    @property
    def error_summary(self) -> str:
        if not self.has_errors:
            return "No errors"
        
        parts = []
        if self.missing_imports:
            parts.append(f"{len(self.missing_imports)} missing imports")
        if self.missing_dependencies:
            parts.append(f"{len(self.missing_dependencies)} missing dependencies")
        if self.syntax_errors:
            parts.append(f"{len(self.syntax_errors)} syntax errors")
        if self.type_errors:
            parts.append(f"{len(self.type_errors)} type errors")
        
        return ", ".join(parts) if parts else f"{self.error_count} errors"
    
    def get_truncated_output(self, max_length: int = 8000) -> str:
        if len(self.compiler_output) <= max_length:
            return self.compiler_output
        
        lines = self.compiler_output.split('\n')
        error_lines = [line for line in lines if self._is_error_line(line)]
        
        if len('\n'.join(error_lines)) <= max_length:
            return '\n'.join(error_lines)
        
        return self.compiler_output[:max_length] + "\n... [truncated]"
    
    def _is_error_line(self, line: str) -> bool:
        error_indicators = [
            'error:', 'Error:', 'ERROR:',
            'cannot find symbol', 'package does not exist',
            'incompatible types', 'method cannot be applied',
            'class not found', 'exception'
        ]
        return any(indicator in line for indicator in error_indicators)


@dataclass
class TestFailureDetail:
    """Single test failure detail."""
    test_class: str = ""
    test_method: str = ""
    failure_type: str = ""
    failure_message: str = ""
    stack_trace: str = ""
    expected_value: str = ""
    actual_value: str = ""
    line_number: Optional[int] = None


@dataclass
class TestFailureContext:
    """Test failure context for LLM analysis."""
    
    test_output: str = ""
    failed_tests: List[TestFailureDetail] = field(default_factory=list)
    passed_tests: List[str] = field(default_factory=list)
    skipped_tests: List[str] = field(default_factory=list)
    
    failure_reasons: List[str] = field(default_factory=list)
    stack_traces: List[str] = field(default_factory=list)
    
    test_file: str = ""
    test_code: str = ""
    source_file: str = ""
    source_code: str = ""
    
    total_tests: int = 0
    passed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    
    attempt_number: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def has_failures(self) -> bool:
        return len(self.failed_tests) > 0 or self.failed_count > 0
    
    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_count / self.total_tests
    
    @property
    def has_partial_success(self) -> bool:
        return self.passed_count > 0 and self.failed_count > 0
    
    @property
    def failure_summary(self) -> str:
        if not self.has_failures:
            return "All tests passed"
        
        failure_types = {}
        for test in self.failed_tests:
            ft = test.failure_type or "Unknown"
            failure_types[ft] = failure_types.get(ft, 0) + 1
        
        parts = [f"{count} {ftype}" for ftype, count in failure_types.items()]
        return ", ".join(parts) if parts else f"{self.failed_count} failures"
    
    def get_truncated_output(self, max_length: int = 8000) -> str:
        if len(self.test_output) <= max_length:
            return self.test_output
        
        lines = self.test_output.split('\n')
        important_lines = []
        
        for line in lines:
            if any(indicator in line for indicator in [
                'FAILED', 'ERROR', 'Failure', 'Exception',
                'at ', 'expected:', 'actual:', 'AssertionError'
            ]):
                important_lines.append(line)
        
        result = '\n'.join(important_lines)
        if len(result) <= max_length:
            return result
        
        return self.test_output[:max_length] + "\n... [truncated]"


@dataclass
class LLMAnalysisResult:
    """Result of LLM analysis."""
    
    root_cause: str = ""
    analysis: str = ""
    confidence: float = 0.0
    
    recommended_strategy: str = ""
    reasoning: str = ""
    
    action_plan: List[Dict[str, Any]] = field(default_factory=list)
    
    specific_fixes: List[str] = field(default_factory=list)
    code_fix: str = ""
    
    should_retry: bool = True
    should_skip: bool = False
    should_escalate: bool = False
    
    raw_response: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def has_action_plan(self) -> bool:
        return len(self.action_plan) > 0
    
    @property
    def has_code_fix(self) -> bool:
        return len(self.code_fix) > 0
    
    def get_first_action(self) -> Optional[Dict[str, Any]]:
        return self.action_plan[0] if self.action_plan else None


class ErrorContextCollector:
    """Collects rich error context for intelligent analysis."""
    
    JAVA_ERROR_PATTERNS = {
        'missing_import': re.compile(
            r'cannot find symbol\s+symbol:\s+class\s+(\w+)',
            re.IGNORECASE
        ),
        'missing_package': re.compile(
            r'package\s+([\w.]+)\s+does not exist',
            re.IGNORECASE
        ),
        'cannot_find_symbol': re.compile(
            r'cannot find symbol.*?symbol:\s*(\w+)\s+(\w+)',
            re.IGNORECASE | re.DOTALL
        ),
        'incompatible_types': re.compile(
            r'incompatible types:\s*(.+?)\s+cannot be converted to\s+(.+)',
            re.IGNORECASE
        ),
        'method_not_found': re.compile(
            r'cannot find symbol.*?method\s+(\w+)',
            re.IGNORECASE | re.DOTALL
        ),
        'syntax_error': re.compile(
            r"'.([^']+)'\s+expected",
            re.IGNORECASE
        ),
    }
    
    TEST_FAILURE_PATTERNS = {
        'assertion_error': re.compile(
            r'AssertionFailedError:\s*(.+)',
            re.IGNORECASE
        ),
        'null_pointer': re.compile(
            r'NullPointerException',
            re.IGNORECASE
        ),
        'mock_exception': re.compile(
            r'MockitoException|WrongTypeOfReturnValue',
            re.IGNORECASE
        ),
        'test_failed': re.compile(
            r'FAILED!!\s*(.+)',
            re.IGNORECASE
        ),
    }
    
    @classmethod
    def collect_compilation_context(
        cls,
        compiler_output: str,
        source_file: str = "",
        test_file: str = "",
        project_path: str = "",
        attempt_number: int = 1
    ) -> CompilationErrorContext:
        context = CompilationErrorContext(
            compiler_output=compiler_output,
            source_file=source_file,
            test_file=test_file,
            project_path=project_path,
            attempt_number=attempt_number
        )
        
        context.error_lines = cls._extract_error_lines(compiler_output)
        context.errors = cls._parse_compilation_errors(compiler_output)
        
        for error in context.errors:
            error_msg = error.error_message.lower()
            
            if 'cannot find symbol' in error_msg or 'class' in error_msg:
                match = cls.JAVA_ERROR_PATTERNS['missing_import'].search(error_msg)
                if match:
                    context.missing_imports.append(match.group(1))
            
            if 'package' in error_msg and 'does not exist' in error_msg:
                match = cls.JAVA_ERROR_PATTERNS['missing_package'].search(error_msg)
                if match:
                    context.missing_dependencies.append(match.group(1))
            
            if 'syntax' in error_msg or 'expected' in error_msg:
                context.syntax_errors.append(error.error_message)
            
            if 'incompatible' in error_msg or 'cannot be converted' in error_msg:
                context.type_errors.append(error.error_message)
        
        context.missing_imports = list(set(context.missing_imports))
        context.missing_dependencies = list(set(context.missing_dependencies))
        
        if source_file:
            context.source_code = cls._read_file_safe(source_file)
        if test_file:
            context.test_code = cls._read_file_safe(test_file)
        if project_path:
            context.pom_content = cls._read_file_safe(
                str(Path(project_path) / "pom.xml")
            )
        
        logger.info(
            f"[ErrorContextCollector] Collected compilation context - "
            f"Errors: {context.error_count}, "
            f"Missing imports: {len(context.missing_imports)}, "
            f"Missing deps: {len(context.missing_dependencies)}"
        )
        
        return context
    
    @classmethod
    def collect_test_failure_context(
        cls,
        test_output: str,
        test_file: str = "",
        source_file: str = "",
        attempt_number: int = 1
    ) -> TestFailureContext:
        context = TestFailureContext(
            test_output=test_output,
            test_file=test_file,
            source_file=source_file,
            attempt_number=attempt_number
        )
        
        context.failed_tests = cls._parse_test_failures(test_output)
        context.passed_tests = cls._parse_passed_tests(test_output)
        context.skipped_tests = cls._parse_skipped_tests(test_output)
        
        context.total_tests = (
            len(context.passed_tests) + 
            len(context.failed_tests) + 
            len(context.skipped_tests)
        )
        context.passed_count = len(context.passed_tests)
        context.failed_count = len(context.failed_tests)
        context.skipped_count = len(context.skipped_tests)
        
        for failure in context.failed_tests:
            if failure.failure_message:
                context.failure_reasons.append(failure.failure_message)
            if failure.stack_trace:
                context.stack_traces.append(failure.stack_trace)
        
        if test_file:
            context.test_code = cls._read_file_safe(test_file)
        if source_file:
            context.source_code = cls._read_file_safe(source_file)
        
        logger.info(
            f"[ErrorContextCollector] Collected test failure context - "
            f"Total: {context.total_tests}, "
            f"Passed: {context.passed_count}, "
            f"Failed: {context.failed_count}"
        )
        
        return context
    
    @classmethod
    def _extract_error_lines(cls, output: str) -> List[str]:
        lines = output.split('\n')
        error_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in [
                'error:', 'cannot find', 'does not exist',
                'incompatible', 'exception', 'failed'
            ]):
                error_lines.append(line.strip())
        
        return error_lines
    
    @classmethod
    def _parse_compilation_errors(cls, output: str) -> List[CompilationErrorDetail]:
        errors = []
        lines = output.split('\n')

        for i, line in enumerate(lines):
            if '.java:' in line and 'error:' in line.lower():
                file_path = ""
                line_number = None
                error_message = ""
                error_type = ""
                source_line = ""

                file_match = re.search(r'([\w/\\.-]+\.java):(\d+)', line)
                if file_match:
                    file_path = file_match.group(1)
                    line_number = int(file_match.group(2))

                error_msg_match = re.search(r'error:\s*(.+)', line, re.IGNORECASE)
                if error_msg_match:
                    error_message = error_msg_match.group(1).strip()
                    error_type = cls._classify_error_type(error_message)

                if i + 1 < len(lines):
                    source_line = lines[i + 1].strip() if lines[i + 1].strip() else ""

                error = CompilationErrorDetail(
                    file_path=file_path,
                    line_number=line_number,
                    error_type=error_type,
                    error_message=error_message,
                    source_line=source_line
                )
                errors.append(error)

        return errors
    
    @classmethod
    def _classify_error_type(cls, error_message: str) -> str:
        msg_lower = error_message.lower()
        
        if 'cannot find symbol' in msg_lower:
            return 'SYMBOL_NOT_FOUND'
        elif 'package' in msg_lower and 'does not exist' in msg_lower:
            return 'PACKAGE_NOT_FOUND'
        elif 'incompatible types' in msg_lower:
            return 'TYPE_MISMATCH'
        elif 'method' in msg_lower and 'cannot be applied' in msg_lower:
            return 'METHOD_SIGNATURE_MISMATCH'
        elif 'class' in msg_lower and 'not found' in msg_lower:
            return 'CLASS_NOT_FOUND'
        elif 'expected' in msg_lower:
            return 'SYNTAX_ERROR'
        else:
            return 'UNKNOWN'
    
    @classmethod
    def _parse_test_failures(cls, output: str) -> List[TestFailureDetail]:
        failures = []
        lines = output.split('\n')
        
        current_failure = None
        in_stack_trace = False
        stack_trace_lines = []
        
        for line in lines:
            if '<<< FAILURE!' in line or 'FAILED!' in line:
                if current_failure:
                    current_failure.stack_trace = '\n'.join(stack_trace_lines)
                    failures.append(current_failure)
                
                current_failure = TestFailureDetail()
                stack_trace_lines = []
                in_stack_trace = False
                
                test_match = re.search(r'(\w+)\((\w+)\)', line)
                if test_match:
                    current_failure.test_method = test_match.group(1)
                    current_failure.test_class = test_match.group(2)
            
            elif current_failure:
                if line.strip().startswith('at ') or in_stack_trace:
                    in_stack_trace = True
                    stack_trace_lines.append(line)
                
                elif 'expected:' in line.lower():
                    match = re.search(r'expected:\s*(.+)', line, re.IGNORECASE)
                    if match:
                        current_failure.expected_value = match.group(1).strip()
                
                elif 'actual:' in line.lower() or 'but was:' in line.lower():
                    match = re.search(r'(?:actual|but was):\s*(.+)', line, re.IGNORECASE)
                    if match:
                        current_failure.actual_value = match.group(1).strip()
                
                elif 'AssertionFailedError' in line or 'AssertionError' in line:
                    match = re.search(r'(?:AssertionFailedError|AssertionError):\s*(.+)', line)
                    if match:
                        current_failure.failure_message = match.group(1).strip()
                        current_failure.failure_type = 'ASSERTION_FAILURE'
                
                elif 'NullPointerException' in line:
                    current_failure.failure_type = 'NULL_POINTER'
                
                elif 'MockitoException' in line or 'WrongTypeOfReturnValue' in line:
                    current_failure.failure_type = 'MOCK_ISSUE'
        
        if current_failure:
            current_failure.stack_trace = '\n'.join(stack_trace_lines)
            failures.append(current_failure)
        
        return failures
    
    @classmethod
    def _parse_passed_tests(cls, output: str) -> List[str]:
        passed = []
        lines = output.split('\n')
        
        for line in lines:
            if '<<< SUCCESS!' in line or 'PASSED' in line:
                match = re.search(r'(\w+)\((\w+)\)', line)
                if match:
                    passed.append(f"{match.group(2)}.{match.group(1)}")
        
        return passed
    
    @classmethod
    def _parse_skipped_tests(cls, output: str) -> List[str]:
        skipped = []
        lines = output.split('\n')
        
        for line in lines:
            if 'SKIPPED' in line:
                match = re.search(r'(\w+)\((\w+)\)', line)
                if match:
                    skipped.append(f"{match.group(2)}.{match.group(1)}")
        
        return skipped
    
    @classmethod
    def _read_file_safe(cls, file_path: str) -> str:
        try:
            path = Path(file_path)
            if path.exists():
                return path.read_text(encoding='utf-8')
        except Exception as e:
            logger.warning(f"[ErrorContextCollector] Failed to read file {file_path}: {e}")
        return ""


def create_compilation_context(
    compiler_output: str,
    source_file: str = "",
    test_file: str = "",
    project_path: str = "",
    attempt_number: int = 1
) -> CompilationErrorContext:
    return ErrorContextCollector.collect_compilation_context(
        compiler_output=compiler_output,
        source_file=source_file,
        test_file=test_file,
        project_path=project_path,
        attempt_number=attempt_number
    )


def create_test_failure_context(
    test_output: str,
    test_file: str = "",
    source_file: str = "",
    attempt_number: int = 1
) -> TestFailureContext:
    return ErrorContextCollector.collect_test_failure_context(
        test_output=test_output,
        test_file=test_file,
        source_file=source_file,
        attempt_number=attempt_number
    )
