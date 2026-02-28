"""Test failure analyzer for JUnit tests."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from pathlib import Path


class FailureType(Enum):
    """Types of test failures."""
    ASSERTION_FAILURE = auto()
    NULL_POINTER = auto()
    INDEX_OUT_OF_BOUNDS = auto()
    ILLEGAL_ARGUMENT = auto()
    ILLEGAL_STATE = auto()
    CLASS_CAST = auto()
    ARITHMETIC = auto()
    TIMEOUT = auto()
    UNEXPECTED_EXCEPTION = auto()
    MOCK_VERIFICATION = auto()
    MOCK_CONFIGURATION = auto()
    SETUP_FAILURE = auto()
    TEARDOWN_FAILURE = auto()
    COMPILATION_ERROR = auto()
    UNKNOWN = auto()


@dataclass
class TestFailure:
    """Represents a single test failure."""
    failure_type: FailureType
    test_class: str
    test_method: str
    message: str
    stack_trace: str
    line_number: Optional[int] = None
    cause: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    fix_hint: str = ""


@dataclass
class FailureAnalysis:
    """Result of failure analysis."""
    failures: List[TestFailure]
    summary: str
    fix_strategy: str
    priority: int  # 1 = highest, 5 = lowest
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    skipped_tests: int = 0


class TestFailureAnalyzer:
    """Analyzes JUnit test failures and suggests fixes."""
    
    # Exception patterns for failure classification
    EXCEPTION_PATTERNS = {
        FailureType.NULL_POINTER: [
            r"java\.lang\.NullPointerException",
            r"NullPointerException",
        ],
        FailureType.INDEX_OUT_OF_BOUNDS: [
            r"java\.lang\.IndexOutOfBoundsException",
            r"java\.lang\.ArrayIndexOutOfBoundsException",
            r"IndexOutOfBoundsException",
            r"ArrayIndexOutOfBoundsException",
        ],
        FailureType.ILLEGAL_ARGUMENT: [
            r"java\.lang\.IllegalArgumentException",
            r"IllegalArgumentException",
        ],
        FailureType.ILLEGAL_STATE: [
            r"java\.lang\.IllegalStateException",
            r"IllegalStateException",
        ],
        FailureType.CLASS_CAST: [
            r"java\.lang\.ClassCastException",
            r"ClassCastException",
        ],
        FailureType.ARITHMETIC: [
            r"java\.lang\.ArithmeticException",
            r"ArithmeticException",
        ],
        FailureType.TIMEOUT: [
            r"org\.junit\.runners\.model\.TestTimedOutException",
            r"TestTimedOutException",
            r"timed out",
        ],
        FailureType.MOCK_VERIFICATION: [
            r"org\.mockito\.exceptions\.verification\.VerificationInOrderFailure",
            r"org\.mockito\.exceptions\.verification\.WantedButNotInvoked",
            r"Wanted but not invoked",
            r"Verification failed",
        ],
        FailureType.MOCK_CONFIGURATION: [
            r"org\.mockito\.exceptions\.misusing\.UnnecessaryStubbingException",
            r"org\.mockito\.exceptions\.misusing\.PotentialStubbingProblem",
            r"Unnecessary stubbing",
            r"Potential stubbing problem",
        ],
        FailureType.ASSERTION_FAILURE: [
            r"org\.opentest4j\.AssertionFailedError",
            r"java\.lang\.AssertionError",
            r"AssertionError",
            r"Assertion failed",
            r"expected:\s*<.*>\s*but was:\s*<.*>",
        ],
    }
    
    def __init__(self, project_path: str):
        """Initialize failure analyzer.
        
        Args:
            project_path: Path to the project
        """
        self.project_path = Path(project_path)
        self.surefire_dir = self.project_path / "target" / "surefire-reports"
        self.exception_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[FailureType, List[re.Pattern]]:
        """Compile exception patterns."""
        compiled = {}
        for failure_type, patterns in self.EXCEPTION_PATTERNS.items():
            compiled[failure_type] = [re.compile(p, re.IGNORECASE) for p in patterns]
        return compiled
    
    def analyze(self) -> FailureAnalysis:
        """Analyze test failures from Surefire reports.
        
        Returns:
            FailureAnalysis with parsed failures and fix strategy
        """
        failures = []
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        skipped_tests = 0
        
        if not self.surefire_dir.exists():
            return FailureAnalysis(
                failures=[],
                summary="No Surefire reports found",
                fix_strategy="Run tests first",
                priority=5
            )
        
        # Parse XML reports
        for xml_file in self.surefire_dir.glob("TEST-*.xml"):
            try:
                result = self._parse_xml_report(xml_file)
                failures.extend(result["failures"])
                total_tests += result["total"]
                passed_tests += result["passed"]
                failed_tests += result["failed"]
                error_tests += result["errors"]
                skipped_tests += result["skipped"]
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
        
        # Parse text reports for additional details
        for txt_file in self.surefire_dir.glob("*.txt"):
            try:
                text_failures = self._parse_text_report(txt_file)
                # Merge with existing failures if needed
            except Exception as e:
                print(f"Error parsing {txt_file}: {e}")
        
        # Generate analysis
        summary = self._generate_summary(failures, total_tests, passed_tests, failed_tests, error_tests)
        fix_strategy = self._determine_fix_strategy(failures)
        priority = self._calculate_priority(failures)
        
        return FailureAnalysis(
            failures=failures,
            summary=summary,
            fix_strategy=fix_strategy,
            priority=priority,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests
        )
    
    def _parse_xml_report(self, xml_file: Path) -> Dict[str, Any]:
        """Parse Surefire XML report."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        failures = []
        total = int(root.get("tests", 0))
        failed = int(root.get("failures", 0))
        errors = int(root.get("errors", 0))
        skipped = int(root.get("skipped", 0))
        passed = total - failed - errors - skipped
        
        test_class = root.get("name", "Unknown")
        
        for testcase in root.findall("testcase"):
            method_name = testcase.get("name", "unknown")
            
            # Check for failure
            failure_elem = testcase.find("failure")
            if failure_elem is not None:
                failure = self._parse_failure_element(
                    failure_elem, test_class, method_name, is_error=False
                )
                failures.append(failure)
            
            # Check for error
            error_elem = testcase.find("error")
            if error_elem is not None:
                failure = self._parse_failure_element(
                    error_elem, test_class, method_name, is_error=True
                )
                failures.append(failure)
        
        return {
            "failures": failures,
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped
        }
    
    def _parse_failure_element(
        self,
        element: ET.Element,
        test_class: str,
        method_name: str,
        is_error: bool
    ) -> TestFailure:
        """Parse a failure/error element from XML."""
        message = element.get("message", "")
        failure_type_str = element.get("type", "")
        stack_trace = element.text or ""
        
        # Classify failure
        failure_type = self._classify_failure(failure_type_str, message, stack_trace)
        
        # Extract line number from stack trace
        line_number = self._extract_line_number(stack_trace)
        
        # Extract cause
        cause = self._extract_cause(message, stack_trace)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(failure_type, message, cause)
        
        # Generate fix hint
        fix_hint = self._generate_fix_hint(failure_type, message, cause)
        
        return TestFailure(
            failure_type=failure_type,
            test_class=test_class,
            test_method=method_name,
            message=message,
            stack_trace=stack_trace,
            line_number=line_number,
            cause=cause,
            suggestions=suggestions,
            fix_hint=fix_hint
        )
    
    def _parse_text_report(self, txt_file: Path) -> List[TestFailure]:
        """Parse Surefire text report."""
        failures = []
        content = txt_file.read_text()
        
        # Pattern for test failure summary
        # This is a simplified parser
        
        return failures
    
    def _classify_failure(self, exception_type: str, message: str, stack_trace: str) -> FailureType:
        """Classify failure type from exception information."""
        combined_text = f"{exception_type} {message} {stack_trace}"
        
        for failure_type, patterns in self.exception_patterns.items():
            for pattern in patterns:
                if pattern.search(combined_text):
                    return failure_type
        
        # Check for setup/teardown failures
        if "@Before" in stack_trace or "@BeforeEach" in stack_trace or "setUp" in stack_trace:
            return FailureType.SETUP_FAILURE
        if "@After" in stack_trace or "@AfterEach" in stack_trace or "tearDown" in stack_trace:
            return FailureType.TEARDOWN_FAILURE
        
        # Check for unexpected exceptions
        if "Unexpected" in message or exception_type:
            return FailureType.UNEXPECTED_EXCEPTION
        
        return FailureType.UNKNOWN
    
    def _extract_line_number(self, stack_trace: str) -> Optional[int]:
        """Extract line number from stack trace."""
        # Look for line number in stack trace
        pattern = r"\(([^:]+):(\d+)\)"
        match = re.search(pattern, stack_trace)
        if match:
            return int(match.group(2))
        return None
    
    def _extract_cause(self, message: str, stack_trace: str) -> Optional[str]:
        """Extract root cause from failure."""
        # Look for "Caused by:" in stack trace
        pattern = r"Caused by: ([^\n]+)"
        match = re.search(pattern, stack_trace)
        if match:
            return match.group(1).strip()
        
        # If no cause found, use the first line of message
        if message:
            return message.split('\n')[0].strip()
        
        return None
    
    def _generate_suggestions(
        self,
        failure_type: FailureType,
        message: str,
        cause: Optional[str]
    ) -> List[str]:
        """Generate fix suggestions based on failure type."""
        suggestions = []
        
        if failure_type == FailureType.ASSERTION_FAILURE:
            suggestions.append("Check the expected vs actual values in the assertion")
            suggestions.append("Verify the test logic and expected results")
            suggestions.append("Ensure the object under test is properly initialized")
            
        elif failure_type == FailureType.NULL_POINTER:
            suggestions.append("Initialize the object before using it")
            suggestions.append("Add null checks before accessing object members")
            suggestions.append("Ensure proper setup in @BeforeEach method")
            
        elif failure_type == FailureType.INDEX_OUT_OF_BOUNDS:
            suggestions.append("Check array/list size before accessing index")
            suggestions.append("Verify loop bounds and iteration logic")
            suggestions.append("Use safe access methods like getOrDefault")
            
        elif failure_type == FailureType.MOCK_VERIFICATION:
            suggestions.append("Verify mock setup matches actual method calls")
            suggestions.append("Check the number of expected invocations")
            suggestions.append("Ensure mock is properly injected")
            
        elif failure_type == FailureType.MOCK_CONFIGURATION:
            suggestions.append("Remove unnecessary stubbings")
            suggestions.append("Use lenient() for optional stubbings")
            suggestions.append("Verify all stubbed methods are called")
            
        elif failure_type == FailureType.TIMEOUT:
            suggestions.append("Increase timeout if test is valid but slow")
            suggestions.append("Optimize test execution time")
            suggestions.append("Check for infinite loops or deadlocks")
            
        elif failure_type == FailureType.SETUP_FAILURE:
            suggestions.append("Fix initialization in @BeforeEach method")
            suggestions.append("Check for exceptions in setup code")
            suggestions.append("Verify test dependencies are available")
            
        elif failure_type == FailureType.UNEXPECTED_EXCEPTION:
            suggestions.append("Handle the exception in the test or code")
            suggestions.append("Add expected exception to test annotation")
            suggestions.append("Fix the underlying issue causing the exception")
            
        else:
            suggestions.append("Review the stack trace for the root cause")
            suggestions.append("Check the line number indicated in the error")
            suggestions.append("Verify test setup and dependencies")
        
        return suggestions
    
    def _generate_fix_hint(
        self,
        failure_type: FailureType,
        message: str,
        cause: Optional[str]
    ) -> str:
        """Generate a specific fix hint for the failure."""
        if failure_type == FailureType.ASSERTION_FAILURE:
            # Extract expected and actual values if possible
            match = re.search(r"expected:\s*<(.*)>\s*but was:\s*<(.*)>", message)
            if match:
                expected = match.group(1)
                actual = match.group(2)
                return f"Expected '{expected}' but got '{actual}'. Update assertion or fix code."
            return "Fix the assertion or the expected value"
        
        elif failure_type == FailureType.NULL_POINTER:
            return "Add null check or initialize the object before use"
        
        elif failure_type == FailureType.MOCK_VERIFICATION:
            return "Adjust mock verification to match actual behavior"
        
        elif failure_type == FailureType.MOCK_CONFIGURATION:
            return "Fix mock setup or remove unnecessary stubbings"
        
        elif failure_type == FailureType.TIMEOUT:
            return "Optimize test or increase timeout threshold"
        
        elif failure_type == FailureType.SETUP_FAILURE:
            return "Fix test setup/initialization code"
        
        return f"Fix the underlying issue: {cause or message}"
    
    def _generate_summary(
        self,
        failures: List[TestFailure],
        total: int,
        passed: int,
        failed: int,
        errors: int
    ) -> str:
        """Generate a summary of test results."""
        if not failures:
            return f"All tests passed ({passed}/{total})"
        
        summary_parts = [
            f"Test Results: {passed} passed, {failed} failed, {errors} errors, {total} total"
        ]
        
        # Count by failure type
        failure_counts = {}
        for failure in failures:
            failure_counts[failure.failure_type] = failure_counts.get(failure.failure_type, 0) + 1
        
        summary_parts.append("\nFailure breakdown:")
        for failure_type, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            summary_parts.append(f"  - {count} {failure_type.name.replace('_', ' ').title()}")
        
        return "\n".join(summary_parts)
    
    def _determine_fix_strategy(self, failures: List[TestFailure]) -> str:
        """Determine the overall fix strategy."""
        if not failures:
            return "No fixes needed"
        
        # Check for setup failures first
        setup_failures = [f for f in failures if f.failure_type == FailureType.SETUP_FAILURE]
        if setup_failures:
            return "Fix setup failures first as they may cause other test failures"
        
        # Check for mock configuration issues
        mock_failures = [f for f in failures if f.failure_type in [
            FailureType.MOCK_CONFIGURATION,
            FailureType.MOCK_VERIFICATION
        ]]
        if mock_failures:
            return "Fix mock configuration issues, then verify test logic"
        
        # Check for null pointer exceptions
        null_failures = [f for f in failures if f.failure_type == FailureType.NULL_POINTER]
        if null_failures:
            return "Fix null pointer issues by proper initialization"
        
        # Check for assertion failures
        assertion_failures = [f for f in failures if f.failure_type == FailureType.ASSERTION_FAILURE]
        if assertion_failures:
            return "Review and fix assertion values or test logic"
        
        return "Address failures in order of priority"
    
    def _calculate_priority(self, failures: List[TestFailure]) -> int:
        """Calculate fix priority (1 = highest)."""
        if not failures:
            return 5
        
        # Priority based on failure types
        priority_scores = {
            FailureType.SETUP_FAILURE: 1,
            FailureType.COMPILATION_ERROR: 1,
            FailureType.NULL_POINTER: 2,
            FailureType.MOCK_CONFIGURATION: 2,
            FailureType.INDEX_OUT_OF_BOUNDS: 2,
            FailureType.ASSERTION_FAILURE: 3,
            FailureType.MOCK_VERIFICATION: 3,
            FailureType.ILLEGAL_ARGUMENT: 3,
            FailureType.ILLEGAL_STATE: 3,
            FailureType.TIMEOUT: 3,
            FailureType.TEARDOWN_FAILURE: 4,
            FailureType.UNEXPECTED_EXCEPTION: 4,
            FailureType.CLASS_CAST: 4,
            FailureType.ARITHMETIC: 4,
            FailureType.UNKNOWN: 4,
        }
        
        min_priority = 5
        for failure in failures:
            priority = priority_scores.get(failure.failure_type, 4)
            min_priority = min(min_priority, priority)
        
        return min_priority
    
    def get_fix_prompt_context(self, analysis: FailureAnalysis) -> str:
        """Generate context for LLM fix prompt."""
        context_parts = [
            analysis.summary,
            "",
            "Fix Strategy: " + analysis.fix_strategy,
            "",
            "Detailed Failures:"
        ]
        
        for i, failure in enumerate(analysis.failures[:10], 1):  # Limit to 10 failures
            context_parts.append(f"\n{i}. {failure.failure_type.name}")
            context_parts.append(f"   Test: {failure.test_class}.{failure.test_method}")
            context_parts.append(f"   Message: {failure.message[:200]}")
            if failure.line_number:
                context_parts.append(f"   Line: {failure.line_number}")
            if failure.cause:
                context_parts.append(f"   Cause: {failure.cause[:100]}")
            if failure.fix_hint:
                context_parts.append(f"   Hint: {failure.fix_hint}")
        
        if len(analysis.failures) > 10:
            context_parts.append(f"\n... and {len(analysis.failures) - 10} more failures")
        
        return "\n".join(context_parts)


class FailureFixGenerator:
    """Generates specific fixes for test failures."""
    
    def __init__(self, analyzer: TestFailureAnalyzer):
        """Initialize fix generator.
        
        Args:
            analyzer: Failure analyzer instance
        """
        self.analyzer = analyzer
    
    def generate_fixes(
        self,
        test_code: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate fixes for test failures.
        
        Args:
            test_code: Current test code
            class_info: Optional class information
            
        Returns:
            Dictionary with fix information
        """
        analysis = self.analyzer.analyze()
        
        fixes = []
        for failure in analysis.failures:
            fix = self._generate_specific_fix(failure, test_code, class_info)
            fixes.append(fix)
        
        return {
            "analysis": analysis,
            "fixes": fixes,
            "context": self.analyzer.get_fix_prompt_context(analysis)
        }
    
    def _generate_specific_fix(
        self,
        failure: TestFailure,
        test_code: str,
        class_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a specific fix for a failure."""
        fix = {
            "failure_type": failure.failure_type.name,
            "test_method": failure.test_method,
            "message": failure.message,
            "line": failure.line_number,
            "action": "unknown",
            "details": {}
        }
        
        if failure.failure_type == FailureType.ASSERTION_FAILURE:
            fix["action"] = "fix_assertion"
            fix["details"] = {
                "suggestion": "Update expected value or fix code logic",
                "test_method": failure.test_method
            }
        
        elif failure.failure_type == FailureType.NULL_POINTER:
            fix["action"] = "add_initialization"
            fix["details"] = {
                "suggestion": "Initialize object in @BeforeEach or test method",
                "test_method": failure.test_method
            }
        
        elif failure.failure_type == FailureType.MOCK_VERIFICATION:
            fix["action"] = "fix_mock_verification"
            fix["details"] = {
                "suggestion": "Adjust verify() call to match actual invocations",
                "test_method": failure.test_method
            }
        
        elif failure.failure_type == FailureType.MOCK_CONFIGURATION:
            fix["action"] = "fix_mock_setup"
            fix["details"] = {
                "suggestion": "Remove unnecessary stubbings or use lenient()",
                "test_method": failure.test_method
            }
        
        elif failure.failure_type == FailureType.SETUP_FAILURE:
            fix["action"] = "fix_setup"
            fix["details"] = {
                "suggestion": "Fix @BeforeEach method initialization",
                "test_method": "setUp"
            }
        
        elif failure.failure_type == FailureType.TIMEOUT:
            fix["action"] = "increase_timeout"
            fix["details"] = {
                "suggestion": "Add @Timeout annotation or optimize test",
                "test_method": failure.test_method
            }
        
        return fix