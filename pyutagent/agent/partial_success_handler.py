"""Partial success handling for incremental test repair.

This module handles scenarios where some tests pass while others fail,
allowing targeted repair of only the failing tests while preserving
successful ones.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of individual test execution."""
    PASSED = auto()
    FAILED = auto()
    ERROR = auto()
    SKIPPED = auto()
    UNKNOWN = auto()


@dataclass
class TestMethodResult:
    """Result of a single test method execution."""
    method_name: str
    status: TestStatus
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    execution_time_ms: Optional[int] = None
    line_number: Optional[int] = None
    
    def is_success(self) -> bool:
        """Check if test was successful."""
        return self.status == TestStatus.PASSED


@dataclass
class PartialTestResult:
    """Result of test execution with partial success information."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    test_results: List[TestMethodResult] = field(default_factory=list)
    raw_output: str = ""
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def has_partial_success(self) -> bool:
        """Check if there are both passed and failed tests."""
        return self.passed_tests > 0 and (self.failed_tests > 0 or self.error_tests > 0)
    
    def get_failed_tests(self) -> List[TestMethodResult]:
        """Get list of failed tests."""
        return [t for t in self.test_results if t.status in (TestStatus.FAILED, TestStatus.ERROR)]
    
    def get_passed_tests(self) -> List[TestMethodResult]:
        """Get list of passed tests."""
        return [t for t in self.test_results if t.status == TestStatus.PASSED]


@dataclass
class TestMethodInfo:
    """Information about a test method in the source code."""
    method_name: str
    start_line: int
    end_line: int
    content: str
    annotations: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class IncrementalFixResult:
    """Result of incremental fix operation."""
    success: bool
    preserved_tests: List[str] = field(default_factory=list)
    fixed_tests: List[str] = field(default_factory=list)
    new_test_code: Optional[str] = None
    original_test_code: Optional[str] = None
    fix_strategy: str = ""
    error_message: Optional[str] = None


class TestCodeParser:
    """Parser for extracting test methods from Java test code."""
    
    def parse_test_methods(self, test_code: str) -> List[TestMethodInfo]:
        """Parse test code to extract individual test methods.
        
        Args:
            test_code: Java test code
            
        Returns:
            List of test method information
        """
        methods = []
        
        # Pattern to match test methods with @Test annotation
        test_method_pattern = re.compile(
            r'(@Test(?:\([^)]*\))?\s*)'  # @Test annotation with optional params
            r'((?:@\w+(?:\([^)]*\))?\s*)*)'  # Other annotations
            r'(public\s+)?(private\s+)?(protected\s+)?'  # Access modifiers
            r'(static\s+)?'
            r'void\s+'  # Return type
            r'(\w+)\s*'  # Method name
            r'\([^)]*\)\s*'  # Parameters
            r'(throws\s+[\w,\s]+)?\s*'  # Throws clause
            r'\{',  # Opening brace
            re.MULTILINE | re.DOTALL
        )
        
        for match in test_method_pattern.finditer(test_code):
            test_annotation = match.group(1)
            other_annotations = match.group(2)
            method_name = match.group(7)
            start_pos = match.start()
            
            # Find method body end
            brace_count = 1
            pos = match.end() - 1
            while brace_count > 0 and pos < len(test_code):
                if test_code[pos] == '{':
                    brace_count += 1
                elif test_code[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            method_content = test_code[start_pos:pos]
            start_line = test_code[:start_pos].count('\n') + 1
            end_line = test_code[:pos].count('\n') + 1
            
            # Parse annotations
            all_annotations = [test_annotation.strip()]
            if other_annotations:
                all_annotations.extend(a.strip() for a in other_annotations.strip().split('\n') if a.strip())
            
            # Extract dependencies (helper methods called)
            dependencies = self._extract_dependencies(method_content, test_code)
            
            methods.append(TestMethodInfo(
                method_name=method_name,
                start_line=start_line,
                end_line=end_line,
                content=method_content,
                annotations=all_annotations,
                dependencies=dependencies
            ))
        
        logger.debug(f"[TestCodeParser] Parsed {len(methods)} test methods")
        return methods
    
    def _extract_dependencies(self, method_content: str, full_code: str) -> Set[str]:
        """Extract method dependencies from test method.
        
        Args:
            method_content: Content of the test method
            full_code: Full test class code
            
        Returns:
            Set of dependent method names
        """
        dependencies = set()
        
        # Find all method calls in the test method
        method_call_pattern = re.compile(r'\b(\w+)\s*\(')
        for match in method_call_pattern.finditer(method_content):
            method_name = match.group(1)
            # Skip common Java methods and keywords
            if method_name in ('assertEquals', 'assertTrue', 'assertFalse', 'assertNull', 
                             'assertNotNull', 'assertThrows', 'when', 'verify', 'given',
                             'System', 'String', 'Integer', 'Boolean', 'if', 'for', 'while',
                             'switch', 'catch', 'new', 'return', 'throw'):
                continue
            dependencies.add(method_name)
        
        return dependencies
    
    def extract_class_skeleton(self, test_code: str) -> str:
        """Extract class skeleton without test methods.
        
        Args:
            test_code: Full test code
            
        Returns:
            Class skeleton (imports, class declaration, fields, setup methods)
        """
        lines = test_code.split('\n')
        skeleton_lines = []
        in_test_method = False
        brace_depth = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if this is the start of a test method
            if re.match(r'\s*@Test\b', stripped):
                in_test_method = True
                # Skip until we find the method signature
                while i < len(lines) and 'void' not in lines[i]:
                    i += 1
                # Skip the method body
                brace_depth = 0
                while i < len(lines):
                    if '{' in lines[i]:
                        brace_depth += lines[i].count('{')
                    if '}' in lines[i]:
                        brace_depth -= lines[i].count('}')
                    i += 1
                    if brace_depth <= 0:
                        break
                in_test_method = False
                continue
            
            skeleton_lines.append(line)
            i += 1
        
        return '\n'.join(skeleton_lines)


class PartialSuccessHandler:
    """Handles partial test success scenarios for incremental repair.
    
    When some tests pass and others fail, this handler:
    1. Identifies which tests passed and which failed
    2. Preserves passing tests
    3. Generates targeted fixes for failing tests only
    4. Reconstructs the test file with mixed preserved and fixed tests
    """
    
    def __init__(self):
        """Initialize the handler."""
        self.parser = TestCodeParser()
        logger.info("[PartialSuccessHandler] Initialized")
    
    def analyze_test_results(
        self,
        test_output: str,
        surefire_reports_dir: Optional[Path] = None
    ) -> PartialTestResult:
        """Analyze test execution output to identify partial success.
        
        Args:
            test_output: Maven test output
            surefire_reports_dir: Directory with Surefire XML reports
            
        Returns:
            Parsed test results
        """
        result = PartialTestResult(
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            error_tests=0,
            skipped_tests=0,
            raw_output=test_output
        )
        
        # Parse Maven test output
        # Pattern: Tests run: X, Failures: Y, Errors: Z, Skipped: W
        summary_pattern = re.compile(
            r'Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)'
        )
        
        for match in summary_pattern.finditer(test_output):
            result.total_tests = int(match.group(1))
            result.failed_tests = int(match.group(2))
            result.error_tests = int(match.group(3))
            result.skipped_tests = int(match.group(4))
            result.passed_tests = result.total_tests - result.failed_tests - result.error_tests - result.skipped_tests
        
        # Parse individual test failures
        failure_pattern = re.compile(
            r'(\w+)\((\w+)\)\s*:\s*(.*?)\s*(?:at\s+(.+?):(\d+))?'
        )
        
        for match in failure_pattern.finditer(test_output):
            method_name = match.group(1)
            class_name = match.group(2)
            error_message = match.group(3)
            file_name = match.group(4)
            line_number = int(match.group(5)) if match.group(5) else None
            
            result.test_results.append(TestMethodResult(
                method_name=method_name,
                status=TestStatus.FAILED,
                error_message=error_message,
                line_number=line_number
            ))
        
        # Try to parse Surefire XML reports for more detailed info
        if surefire_reports_dir and surefire_reports_dir.exists():
            result.test_results.extend(self._parse_surefire_reports(surefire_reports_dir))
        
        logger.info(f"[PartialSuccessHandler] Analyzed results - "
                   f"Total: {result.total_tests}, Passed: {result.passed_tests}, "
                   f"Failed: {result.failed_tests}, Errors: {result.error_tests}")
        
        return result
    
    def _parse_surefire_reports(self, reports_dir: Path) -> List[TestMethodResult]:
        """Parse Surefire XML reports for detailed test results.
        
        Args:
            reports_dir: Directory containing Surefire XML files
            
        Returns:
            List of test method results
        """
        results = []
        
        try:
            import xml.etree.ElementTree as ET
            
            for xml_file in reports_dir.glob("*.xml"):
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    for testcase in root.findall('.//testcase'):
                        method_name = testcase.get('name', 'unknown')
                        classname = testcase.get('classname', 'unknown')
                        time = float(testcase.get('time', 0)) * 1000  # Convert to ms
                        
                        # Check for failures or errors
                        failure = testcase.find('failure')
                        error = testcase.find('error')
                        skipped = testcase.find('skipped')
                        
                        if failure is not None:
                            status = TestStatus.FAILED
                            message = failure.get('message', 'Unknown failure')
                            stack_trace = failure.text or ''
                        elif error is not None:
                            status = TestStatus.ERROR
                            message = error.get('message', 'Unknown error')
                            stack_trace = error.text or ''
                        elif skipped is not None:
                            status = TestStatus.SKIPPED
                            message = None
                            stack_trace = None
                        else:
                            status = TestStatus.PASSED
                            message = None
                            stack_trace = None
                        
                        results.append(TestMethodResult(
                            method_name=method_name,
                            status=status,
                            error_message=message,
                            stack_trace=stack_trace,
                            execution_time_ms=int(time)
                        ))
                        
                except Exception as e:
                    logger.warning(f"[PartialSuccessHandler] Failed to parse {xml_file}: {e}")
                    
        except ImportError:
            logger.warning("[PartialSuccessHandler] xml.etree.ElementTree not available")
        
        return results
    
    def create_incremental_fix_prompt(
        self,
        test_code: str,
        partial_result: PartialTestResult,
        target_class_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a prompt for fixing only failing tests.
        
        Args:
            test_code: Current test code
            partial_result: Partial test results
            target_class_info: Target class information
            
        Returns:
            Prompt for incremental fix
        """
        failed_tests = partial_result.get_failed_tests()
        passed_tests = partial_result.get_passed_tests()
        
        # Parse test methods
        all_methods = self.parser.parse_test_methods(test_code)
        
        # Build the prompt
        lines = [
            "Fix the failing test methods while preserving the passing ones.",
            "",
            "## Test Execution Results",
            f"- Total tests: {partial_result.total_tests}",
            f"- Passed: {partial_result.passed_tests}",
            f"- Failed: {partial_result.failed_tests}",
            f"- Errors: {partial_result.error_tests}",
            "",
            "## Passing Tests (PRESERVE THESE)",
        ]
        
        for test in passed_tests:
            lines.append(f"- {test.method_name}")
        
        lines.extend([
            "",
            "## Failing Tests (FIX THESE)",
        ])
        
        for test in failed_tests:
            lines.append(f"### {test.method_name}")
            if test.error_message:
                lines.append(f"Error: {test.error_message}")
            if test.line_number:
                lines.append(f"Line: {test.line_number}")
            lines.append("")
        
        # Include the current test code
        lines.extend([
            "",
            "## Current Test Code",
            "```java",
            test_code,
            "```",
        ])
        
        if target_class_info:
            lines.extend([
                "",
                "## Target Class Information",
                f"Class: {target_class_info.get('name', 'Unknown')}",
                f"Methods: {', '.join(m.get('name', 'unknown') for m in target_class_info.get('methods', [])[:10])}",
            ])
        
        lines.extend([
            "",
            "## Instructions",
            "1. Keep all passing test methods EXACTLY as they are",
            "2. Fix ONLY the failing test methods",
            "3. Ensure the fixed tests address the reported errors",
            "4. Maintain the same class structure and imports",
            "5. Output the COMPLETE test class with both preserved and fixed tests",
        ])
        
        return '\n'.join(lines)
    
    def merge_incremental_fix(
        self,
        original_code: str,
        fixed_code: str,
        partial_result: PartialTestResult
    ) -> IncrementalFixResult:
        """Merge fixed tests with preserved passing tests.
        
        Args:
            original_code: Original test code
            fixed_code: Fixed test code from LLM
            partial_result: Partial test results
            
        Returns:
            Merge result
        """
        try:
            # Parse both versions
            original_methods = {m.method_name: m for m in self.parser.parse_test_methods(original_code)}
            fixed_methods = {m.method_name: m for m in self.parser.parse_test_methods(fixed_code)}
            
            passed_test_names = {t.method_name for t in partial_result.get_passed_tests()}
            failed_test_names = {t.method_name for t in partial_result.get_failed_tests()}
            
            # Build merged code
            preserved = []
            fixed = []
            
            # Get class skeleton from fixed code (has imports and setup)
            merged_code = self.parser.extract_class_skeleton(fixed_code)
            
            # Add all test methods
            for method_name in passed_test_names:
                if method_name in original_methods:
                    preserved.append(method_name)
            
            for method_name in failed_test_names:
                if method_name in fixed_methods:
                    fixed.append(method_name)
            
            logger.info(f"[PartialSuccessHandler] Merged fix - Preserved: {len(preserved)}, Fixed: {len(fixed)}")
            
            return IncrementalFixResult(
                success=True,
                preserved_tests=preserved,
                fixed_tests=fixed,
                new_test_code=fixed_code,  # Use the LLM output as-is
                original_test_code=original_code,
                fix_strategy="incremental"
            )
            
        except Exception as e:
            logger.error(f"[PartialSuccessHandler] Failed to merge fix: {e}")
            return IncrementalFixResult(
                success=False,
                error_message=str(e),
                fix_strategy="incremental"
            )
    
    def should_attempt_incremental_fix(
        self,
        partial_result: PartialTestResult,
        min_pass_rate: float = 0.3,
        max_fail_count: int = 5
    ) -> bool:
        """Determine if incremental fix should be attempted.
        
        Args:
            partial_result: Partial test results
            min_pass_rate: Minimum pass rate to attempt incremental fix
            max_fail_count: Maximum number of failures for incremental fix
            
        Returns:
            True if incremental fix is recommended
        """
        # Don't attempt if no tests passed
        if partial_result.passed_tests == 0:
            return False
        
        # Don't attempt if pass rate is too low
        if partial_result.success_rate < min_pass_rate:
            return False
        
        # Don't attempt if too many failures
        failed_count = partial_result.failed_tests + partial_result.error_tests
        if failed_count > max_fail_count:
            return False
        
        return True
    
    def extract_failed_test_context(
        self,
        test_code: str,
        failed_test: TestMethodResult,
        context_lines: int = 5
    ) -> str:
        """Extract context around a failed test.
        
        Args:
            test_code: Full test code
            failed_test: Failed test information
            context_lines: Number of context lines
            
        Returns:
            Code context around the failed test
        """
        lines = test_code.split('\n')
        
        # Find the test method
        methods = self.parser.parse_test_methods(test_code)
        target_method = None
        
        for method in methods:
            if method.method_name == failed_test.method_name:
                target_method = method
                break
        
        if not target_method:
            return ""
        
        # Extract context
        start = max(0, target_method.start_line - context_lines - 1)
        end = min(len(lines), target_method.end_line + context_lines)
        
        context = '\n'.join(lines[start:end])
        
        return context


# Convenience functions

def analyze_partial_success(
    test_output: str,
    surefire_reports_dir: Optional[Path] = None
) -> PartialTestResult:
    """Quick analysis of partial test success.
    
    Args:
        test_output: Test execution output
        surefire_reports_dir: Surefire reports directory
        
    Returns:
        Partial test results
    """
    handler = PartialSuccessHandler()
    return handler.analyze_test_results(test_output, surefire_reports_dir)


def should_use_incremental_fix(
    partial_result: PartialTestResult,
    min_pass_rate: float = 0.3
) -> bool:
    """Quick check if incremental fix should be used.
    
    Args:
        partial_result: Partial test results
        min_pass_rate: Minimum pass rate threshold
        
    Returns:
        True if incremental fix is recommended
    """
    handler = PartialSuccessHandler()
    return handler.should_attempt_incremental_fix(partial_result, min_pass_rate)
