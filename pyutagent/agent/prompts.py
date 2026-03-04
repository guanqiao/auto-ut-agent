"""Prompt templates for UT generation."""

from typing import Dict, List, Any, Optional


class PromptBuilder:
    """Builds prompts for LLM interactions."""
    
    def __init__(self):
        """Initialize prompt builder."""
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for UT generation."""
        return """You are an expert Java developer specializing in writing high-quality JUnit 5 unit tests.
Your task is to generate comprehensive unit tests that achieve high code coverage.

Guidelines:
1. Use JUnit 5 framework with appropriate annotations (@Test, @BeforeEach, etc.)
2. Use Mockito for mocking dependencies
3. Write clear, descriptive test method names following shouldWhen pattern
4. Include both positive and negative test cases
5. Add appropriate assertions using AssertJ or JUnit assertions
6. Follow Arrange-Act-Assert pattern
7. Include JavaDoc comments for test methods (in English)
8. Ensure tests are independent and idempotent
9. Use @DisplayName annotation for each test method to describe the test purpose and scenario (in English)
10. All comments, descriptions, and documentation MUST be in English

Output only the Java code without any markdown formatting or explanations."""

    def build_initial_test_prompt(
        self,
        class_info: Dict[str, Any],
        source_code: str
    ) -> str:
        """Build prompt for initial test generation.
        
        Args:
            class_info: Parsed class information
            source_code: Source code of the target class
            
        Returns:
            Prompt string
        """
        class_name = class_info.get("name", "Unknown")
        package = class_info.get("package", "")
        methods = class_info.get("methods", [])
        fields = class_info.get("fields", [])
        imports = class_info.get("imports", [])
        
        method_list = "\n".join([
            f"- {m.get('name')}({', '.join([p[1] if isinstance(p, tuple) else p.get('name', 'unknown') for p in m.get('parameters', [])])}): {m.get('return_type', 'void')}"
            for m in methods
        ])
        
        field_list = "\n".join([
            f"- {f[1] if isinstance(f, tuple) else f.get('name')}: {f[0] if isinstance(f, tuple) else f.get('type')}"
            for f in fields
        ])
        
        return f"""{self.system_prompt}

Generate comprehensive JUnit 5 unit tests for the following Java class:

Class: {class_name}
Package: {package}

Fields:
{field_list}

Methods:
{method_list}

Source Code:
```java
{source_code}
```

Requirements:
1. Create a complete test class named {class_name}Test
2. Include tests for all public methods
3. Use Mockito to mock external dependencies
4. Cover normal cases, edge cases, and error scenarios
5. Achieve at least 80% line coverage
6. Use AssertJ for fluent assertions

Generate the complete test class code:"""

    def build_fix_compilation_prompt(
        self,
        test_code: str,
        compilation_errors: str,
        class_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for fixing compilation errors.
        
        Args:
            test_code: Current test code with errors
            compilation_errors: Compilation error messages
            class_info: Optional class information
            
        Returns:
            Prompt string
        """
        return f"""{self.system_prompt}

Fix the compilation errors in the following test code:

Current Test Code:
```java
{test_code}
```

Compilation Errors:
```
{compilation_errors}
```

Instructions:
1. Fix all compilation errors
2. Ensure all imports are correct
3. Fix any syntax errors
4. Ensure method signatures match the class under test
5. Maintain the existing test logic and intent

Output the corrected test code:"""

    def build_fix_test_failure_prompt(
        self,
        test_code: str,
        failures: List[Dict[str, Any]],
        class_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for fixing test failures.
        
        Args:
            test_code: Current test code
            failures: List of test failures
            class_info: Optional class information
            
        Returns:
            Prompt string
        """
        failure_details = "\n\n".join([
            f"Test: {f.get('test_name')}\nError: {f.get('error')}"
            for f in failures
        ])
        
        return f"""{self.system_prompt}

Fix the failing tests in the following code:

Current Test Code:
```java
{test_code}
```

Test Failures:
{failure_details}

Instructions:
1. Analyze each failure and fix the root cause
2. Fix assertion errors by correcting expected values
3. Fix NullPointerExceptions by properly initializing objects
4. Fix mock configuration issues
5. Ensure tests are deterministic and don't depend on external state
6. Maintain test coverage and quality

Output the corrected test code:"""

    def build_additional_tests_prompt(
        self,
        class_info: Dict[str, Any],
        existing_tests: str,
        uncovered_info: Dict[str, Any],
        current_coverage: float
    ) -> str:
        """Build prompt for generating additional tests to improve coverage.
        
        Args:
            class_info: Class information
            existing_tests: Existing test code
            uncovered_info: Information about uncovered code
            current_coverage: Current coverage percentage
            
        Returns:
            Prompt string
        """
        uncovered_lines = uncovered_info.get("lines", [])
        uncovered_methods = uncovered_info.get("methods", [])
        
        lines_str = ", ".join(map(str, uncovered_lines[:20]))
        if len(uncovered_lines) > 20:
            lines_str += f" and {len(uncovered_lines) - 20} more"
        
        methods_str = "\n".join([
            f"- {m}" for m in uncovered_methods[:10]
        ])
        
        return f"""{self.system_prompt}

Generate additional test methods to improve code coverage from {current_coverage:.1%} to at least 80%.

Existing Test Code:
```java
{existing_tests}
```

Uncovered Lines: {lines_str}

Uncovered Methods:
{methods_str}

Instructions:
1. Generate ONLY the additional test methods (not the entire class)
2. Focus on covering the uncovered lines and methods
3. Add tests for edge cases and error scenarios
4. Ensure new tests follow the same style as existing tests
5. Each test method should be complete and compilable
6. Include proper setup, execution, and assertions

Output only the new test methods:"""

    def build_coverage_analysis_prompt(
        self,
        class_info: Dict[str, Any],
        coverage_report: Dict[str, Any]
    ) -> str:
        """Build prompt for analyzing coverage and suggesting improvements.
        
        Args:
            class_info: Class information
            coverage_report: Coverage report data
            
        Returns:
            Prompt string
        """
        return f"""Analyze the following coverage report and suggest improvements:

Class: {class_info.get('name')}

Coverage Report:
- Line Coverage: {coverage_report.get('line_coverage', 0):.1%}
- Branch Coverage: {coverage_report.get('branch_coverage', 0):.1%}
- Method Coverage: {coverage_report.get('method_coverage', 0):.1%}

Uncovered Areas:
{self._format_uncovered_areas(coverage_report)}

Provide specific recommendations for improving coverage:"""

    def _format_uncovered_areas(self, coverage_report: Dict[str, Any]) -> str:
        """Format uncovered areas for prompt."""
        # This is a placeholder - in real implementation, parse the report
        return "See detailed coverage report for uncovered lines and branches."

    def build_method_test_prompt(
        self,
        class_info: Dict[str, Any],
        method_info: Dict[str, Any],
        source_code: str
    ) -> str:
        """Build prompt for testing a specific method.
        
        Args:
            class_info: Class information
            method_info: Method information
            source_code: Source code
            
        Returns:
            Prompt string
        """
        method_name = method_info.get("name", "unknown")
        parameters = method_info.get("parameters", [])
        return_type = method_info.get("return_type", "void")
        
        param_list = ", ".join([
            f"{p[0] if isinstance(p, tuple) else p.get('type', 'Object')} {p[1] if isinstance(p, tuple) else p.get('name', 'param')}"
            for p in parameters
        ])
        
        return f"""{self.system_prompt}

Generate comprehensive tests for the following method:

Class: {class_info.get('name')}
Method: {return_type} {method_name}({param_list})

Source Context:
```java
{source_code}
```

Generate tests covering:
1. Normal execution path
2. Edge cases (null, empty, boundary values)
3. Error scenarios and exceptions
4. Different parameter combinations

Output the test methods:"""

    def build_incremental_test_prompt(
        self,
        class_info: Dict[str, Any],
        existing_tests: str,
        changed_methods: List[str]
    ) -> str:
        """Build prompt for incremental test generation.
        
        Args:
            class_info: Class information
            existing_tests: Existing test code
            changed_methods: List of changed method names
            
        Returns:
            Prompt string
        """
        changes = "\n".join([f"- {m}" for m in changed_methods])
        
        return f"""{self.system_prompt}

Generate tests for recently changed methods:

Changed Methods:
{changes}

Existing Test Code:
```java
{existing_tests}
```

Instructions:
1. Generate tests only for the changed methods
2. Ensure new tests integrate well with existing tests
3. Follow the existing code style and patterns
4. Add tests for new functionality and modified behavior

Output the new test methods:"""

    def build_incremental_preserve_prompt(
        self,
        class_info: Dict[str, Any],
        source_code: str,
        preserved_tests: List[str],
        preserved_test_code: str,
        tests_to_fix: List[Any],
        uncovered_info: Dict[str, Any],
        current_coverage: float,
        target_coverage: float
    ) -> str:
        """Build prompt for incremental test generation with test preservation.
        
        This prompt is used when:
        - Existing tests exist and some are passing
        - We want to preserve passing tests
        - We may need to fix failing tests
        - We need to generate additional tests for coverage
        
        Args:
            class_info: Target class information
            source_code: Target class source code
            preserved_tests: List of test names to preserve
            preserved_test_code: Code of tests to preserve
            tests_to_fix: List of failing tests with error info
            uncovered_info: Information about uncovered code
            current_coverage: Current coverage percentage
            target_coverage: Target coverage percentage
            
        Returns:
            Prompt string
        """
        class_name = class_info.get("name", "Unknown")
        
        preserved_list = "\n".join([f"- {t}" for t in preserved_tests])
        
        tests_to_fix_section = ""
        if tests_to_fix:
            fix_items = []
            for test in tests_to_fix:
                test_name = test.method_name if hasattr(test, 'method_name') else str(test)
                error_msg = test.error_message if hasattr(test, 'error_message') else "Unknown error"
                stack_trace = test.stack_trace if hasattr(test, 'stack_trace') else None
                
                fix_item = f"### {test_name}\n**Error:** {error_msg}"
                if stack_trace and len(stack_trace) > 0:
                    stack_preview = stack_trace[:500] if len(stack_trace) > 500 else stack_trace
                    fix_item += f"\n**Stack Trace:**\n```\n{stack_preview}\n```"
                fix_items.append(fix_item)
            
            tests_to_fix_section = f"""
## Tests to Fix (REGENERATE THESE)
The following tests are FAILING and need to be fixed:

{chr(10).join(fix_items)}

**Instructions for fixing:**
1. Analyze the error message and stack trace
2. Identify the root cause (wrong assertion, missing mock, incorrect setup)
3. Generate a corrected version of the test
4. Ensure the fix addresses the specific error
"""
        
        uncovered_section = ""
        if uncovered_info:
            uncovered_lines = uncovered_info.get("lines", [])
            uncovered_methods = uncovered_info.get("methods", [])
            uncovered_branches = uncovered_info.get("branches", [])
            
            lines_str = ", ".join(map(str, uncovered_lines[:20]))
            if len(uncovered_lines) > 20:
                lines_str += f" and {len(uncovered_lines) - 20} more"
            
            methods_str = "\n".join([f"- {m}" for m in uncovered_methods[:10]])
            if len(uncovered_methods) > 10:
                methods_str += f"\n- ... and {len(uncovered_methods) - 10} more methods"
            
            branches_str = ""
            if uncovered_branches:
                branches_str = f"\n\n**Uncovered Branches:** {len(uncovered_branches)} branches need coverage"
            
            if uncovered_lines or uncovered_methods:
                uncovered_section = f"""
## Uncovered Code (GENERATE NEW TESTS FOR THESE)
**Current coverage:** {current_coverage:.1%} (Target: {target_coverage:.1%})
**Coverage gap:** {(target_coverage - current_coverage):.1%} more needed

**Uncovered Lines:** {lines_str}
({len(uncovered_lines)} total lines need coverage)

**Uncovered Methods:**
{methods_str if methods_str else "All methods have some coverage"}
{branches_str}

**Instructions for new tests:**
1. Focus on uncovered lines and methods first
2. Add edge case tests for boundary conditions
3. Add tests for error handling paths
4. Consider both happy path and negative scenarios
"""
        
        return f"""{self.system_prompt}

Generate INCREMENTAL unit tests for the following Java class.

## Context
- Target class: {class_name}
- Current coverage: {current_coverage:.1%}
- Target coverage: {target_coverage:.1%}

## Target Class Source Code
```java
{source_code}
```

## Passing Tests (PRESERVE THESE - DO NOT MODIFY)
The following tests are PASSING and must be preserved exactly as they are:
{preserved_list}

## Preserved Test Code (for reference)
```java
{preserved_test_code}
```
{tests_to_fix_section}{uncovered_section}
## Instructions

You are generating tests in INCREMENTAL MODE. This means:

1. **PRESERVE** all passing tests exactly as they are
2. **FIX** any failing tests by correcting the test logic
3. **ADD** new tests for uncovered code to reach target coverage
4. **OUTPUT** the COMPLETE test class with all tests combined

## Output Requirements

1. Output the COMPLETE Java test class code
2. Include all preserved tests (unchanged)
3. Include fixed versions of failing tests
4. Include new tests for uncovered code
5. Ensure all imports are correct
6. Follow JUnit 5 best practices
7. Use @DisplayName for each test method

## Important Notes

- Do NOT modify the preserved passing tests
- Do NOT skip any existing test methods
- Ensure the output is a complete, compilable test class
- The test class name should be {class_name}Test

Output the complete test class:"""

    def build_error_analysis_prompt(
        self,
        error_category: str,
        error_message: str,
        error_details: Dict[str, Any],
        local_analysis: Dict[str, Any],
        attempt_history: List[Dict[str, Any]],
        current_test_code: Optional[str] = None,
        target_class_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for LLM error analysis.
        
        Args:
            error_category: Category of error
            error_message: Error message
            error_details: Additional error details
            local_analysis: Results from local analysis tools
            attempt_history: History of previous attempts
            current_test_code: Current test code
            target_class_info: Target class information
            
        Returns:
            Prompt string
        """
        # Format attempt history
        history_str = "\n".join([
            f"Attempt {a['attempt']}: {a['strategy']} - {'Success' if a['success'] else 'Failed'}"
            for a in attempt_history[-5:]
        ]) if attempt_history else "No previous attempts"
        
        # Format local analysis
        local_insights = local_analysis.get("local_insights", {})
        suggested_fixes = local_analysis.get("suggested_fixes", [])
        
        fixes_str = "\n".join([
            f"- {f.get('type', 'unknown')}: {f.get('hint', '')}"
            for f in suggested_fixes[:5]
        ]) if suggested_fixes else "No specific fixes suggested"
        
        return f"""You are an expert at analyzing and fixing Java test generation errors.

Analyze the following error and provide a recovery strategy.

## Error Information

**Category:** {error_category}
**Message:** {error_message}

**Error Details:**
```
{error_details}
```

## Local Analysis Results

**Insights:**
{local_insights}

**Suggested Fixes:**
{fixes_str}

## Attempt History

{history_str}

## Current Context

**Target Class:** {target_class_info.get('name', 'Unknown') if target_class_info else 'Unknown'}

**Current Test Code:**
```java
{current_test_code[:2000] if current_test_code else 'No test code available'}
```

## Your Task

1. **Analysis**: Provide a brief analysis of what went wrong and why previous attempts (if any) failed.

2. **Strategy**: Recommend one of the following strategies:
   - RETRY_IMMEDIATE: Retry immediately (for transient errors)
   - RETRY_WITH_BACKOFF: Retry with delay (for rate limits, timeouts)
   - ANALYZE_AND_FIX: Analyze and generate a fix (for compilation/test errors)
   - SKIP_AND_CONTINUE: Skip this step (for non-critical errors)
   - RESET_AND_REGENERATE: Start fresh (for persistent failures)
   - FALLBACK_ALTERNATIVE: Use alternative approach

3. **Confidence**: Rate your confidence (0.0-1.0) that the recommended strategy will work.

4. **Specific Fixes**: If applicable, provide specific fix suggestions.

5. **Reasoning**: Explain your reasoning.

## Output Format

```
Analysis: <your analysis>
Strategy: <STRATEGY_NAME>
Confidence: <0.0-1.0>
Fixes:
- <fix 1>
- <fix 2>
Reasoning: <your reasoning>
```

Provide your analysis:"""

    def build_comprehensive_fix_prompt(
        self,
        error_category: str,
        error_message: str,
        error_details: Dict[str, Any],
        local_analysis: Dict[str, Any],
        llm_insights: str,
        specific_fixes: List[str],
        current_test_code: Optional[str],
        target_class_info: Optional[Dict[str, Any]],
        attempt_history: List[Dict[str, Any]]
    ) -> str:
        """Build comprehensive fix prompt combining all analysis.
        
        Args:
            error_category: Category of error
            error_message: Error message
            error_details: Error details
            local_analysis: Local analysis results
            llm_insights: LLM analysis insights
            specific_fixes: Specific fixes suggested
            current_test_code: Current test code
            target_class_info: Target class info
            attempt_history: Attempt history
            
        Returns:
            Prompt string
        """
        # Format specific fixes
        fixes_str = "\n".join([f"- {f}" for f in specific_fixes]) if specific_fixes else "None"
        
        # Format attempt history
        history_str = "\n".join([
            f"Attempt {a['attempt']}: {'Success' if a['success'] else 'Failed'} - {a.get('message', '')}"
            for a in attempt_history
        ]) if attempt_history else "No previous attempts"
        
        return f"""You are an expert Java developer specializing in fixing test code errors.

Fix the following error in the test code. This is a comprehensive fix request that combines local analysis and AI insights.

## Error Information

**Category:** {error_category}
**Message:** {error_message}

**Error Details:**
```
{error_details}
```

## Analysis Summary

**Local Analysis:**
{local_analysis}

**AI Insights:**
{llm_insights}

**Suggested Fixes:**
{fixes_str}

## Attempt History

{history_str}

## Context

**Target Class:** {target_class_info.get('name', 'Unknown') if target_class_info else 'Unknown'}
**Package:** {target_class_info.get('package', '') if target_class_info else ''}

**Target Class Methods:**
{chr(10).join([f"- {m.get('name', 'unknown')}()" for m in target_class_info.get('methods', [])[:10]]) if target_class_info else 'Unknown'}

## Current Test Code

```java
{current_test_code}
```

## Your Task

Generate the COMPLETE fixed test code. Consider:

1. **Root Cause**: Address the root cause, not just symptoms
2. **Previous Failures**: Learn from previous failed attempts
3. **Alternative Approaches**: If previous fixes failed, try a different approach
4. **Simplicity**: Prefer simpler solutions over complex ones
5. **Completeness**: Ensure the entire test class is valid and compilable

## Important Notes

- Output the COMPLETE test class code, not just the changed parts
- Ensure all imports are correct
- Ensure all methods and variables are properly defined
- Follow JUnit 5 best practices
- Make sure the code compiles and tests pass

## Output

Provide the fixed test code:"""