"""Prompt templates for UT generation."""

from typing import Dict, List, Any, Optional

from ..tools.action_definitions import (
    ActionCategory,
    generate_prompt_action_list,
    generate_prompt_examples,
)


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

    def build_dependency_analysis_prompt(
        self,
        compiler_output: str,
        current_pom_content: str
    ) -> str:
        """Build prompt for dependency analysis.
        
        Args:
            compiler_output: Compilation error output
            current_pom_content: Current pom.xml content
            
        Returns:
            Prompt string
        """
        return f"""You are a Maven dependency expert. Analyze the following compilation errors and identify missing dependencies.

Compilation Errors:
```
{compiler_output}
```

Current pom.xml:
```
{current_pom_content}
```

Task:
1. Identify all missing dependencies from the compilation errors
2. For each missing dependency, provide complete Maven coordinates
3. Determine the appropriate scope (test, compile, provided, runtime)
4. Recommend stable versions

Output in JSON format:
{{
  "missing_dependencies": [
    {{
      "group_id": "org.junit.jupiter",
      "artifact_id": "junit-jupiter",
      "version": "5.10.0",
      "scope": "test",
      "reason": "JUnit 5 testing framework"
    }}
  ],
  "confidence": 0.95,
  "analysis": "Brief analysis of missing dependencies",
  "suggested_fixes": ["Additional suggestions for fixing the errors"]
}}

Important:
- Use latest stable versions for common libraries
- Test dependencies should have scope "test"
- Be precise with groupId and artifactId
- If uncertain, set lower confidence score
- Only output the JSON, no additional text"""


    def build_thinking_prompt(
        self,
        situation: str,
        context: Dict[str, Any],
        thinking_type: str = "analytical",
        analysis_result: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for agent thinking process.
        
        Args:
            situation: The situation to think about
            context: Current context information
            thinking_type: Type of thinking (analytical, critical, strategic, etc.)
            analysis_result: Optional previous analysis result
            
        Returns:
            Prompt string
        """
        context_str = "\n".join([
            f"- {k}: {str(v)[:200]}"
            for k, v in list(context.items())[:10]
        ])
        
        analysis_section = ""
        if analysis_result:
            analysis_section = f"""
## Previous Analysis
{analysis_result.get('thought', 'No previous analysis')}

Conclusions:
{chr(10).join([f"- {c}" for c in analysis_result.get('conclusions', [])])}
"""
        
        thinking_guides = {
            "analytical": """
Think systematically:
1. What are the key components of this situation?
2. What are the relationships between them?
3. What patterns do you observe?
4. What conclusions can you draw?""",
            "critical": """
Think critically:
1. What assumptions are being made?
2. What evidence supports or contradicts these assumptions?
3. What are the potential flaws in the current approach?
4. What alternative explanations exist?""",
            "strategic": """
Think strategically:
1. What are the short-term and long-term implications?
2. What resources are available?
3. What are the risks and opportunities?
4. What is the optimal path forward?""",
            "predictive": """
Think predictively:
1. What trends or patterns do you observe?
2. What is likely to happen next?
3. What could go wrong?
4. How can we prepare for different scenarios?""",
        }
        
        guide = thinking_guides.get(thinking_type, thinking_guides["analytical"])
        
        return f"""You are an intelligent agent engaging in deep thinking about a problem.

## Situation
{situation}
{analysis_section}
## Context
{context_str}

## Thinking Task
{guide}

## Output Format
Provide your thinking in this structured format:

THOUGHT: [Your main thought in one sentence]
EVIDENCE:
- [Supporting evidence 1]
- [Supporting evidence 2]
CONCLUSIONS:
- [Conclusion 1]
- [Conclusion 2]
CONFIDENCE: [0.0-1.0]

Think deeply and provide your analysis:"""

    def build_error_thinking_prompt(
        self,
        error: Exception,
        error_message: str,
        error_category: str,
        context: Dict[str, Any],
        similar_errors: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build prompt for thinking about an error.
        
        Args:
            error: The exception that occurred
            error_message: Error message
            error_category: Category of the error
            context: Current context
            similar_errors: Similar past errors
            
        Returns:
            Prompt string
        """
        error_type = type(error).__name__
        
        similar_section = ""
        if similar_errors:
            similar_section = f"""
## Similar Past Errors
{chr(10).join([
    f"- {e.get('error_category', 'Unknown')}: Used {e.get('strategy_used', 'Unknown')} - {'Success' if e.get('success') else 'Failed'}"
    for e in similar_errors[:5]
])}
"""
        
        return f"""You are an expert at diagnosing and recovering from errors in software systems.

## Error Information
**Type:** {error_type}
**Category:** {error_category}
**Message:** {error_message[:500]}

## Context
{chr(10).join([f"- {k}: {str(v)[:100]}" for k, v in list(context.items())[:5]])}
{similar_section}
## Thinking Task
Think through this error systematically:

1. **Perception**: What exactly happened?
2. **Analysis**: Why did it happen? What is the root cause?
3. **Reasoning**: What are the implications? What patterns do you see?
4. **Decision**: What is the best recovery strategy?

## Output Format
Provide your thinking in this structured format:

ROOT_CAUSE: [Root cause in one sentence]
ANALYSIS: [Brief analysis of the error]
RECOVERY_STRATEGY: [One of: RETRY, RETRY_WITH_BACKOFF, ANALYZE_AND_FIX, SKIP_AND_CONTINUE, RESET_AND_REGENERATE, INSTALL_DEPENDENCIES, ESCALATE_TO_USER]
FIX_SUGGESTIONS:
- [Suggestion 1]
- [Suggestion 2]
CONFIDENCE: [0.0-1.0]
REASONING: [Why this strategy is recommended]

Provide your analysis:"""

    def build_prediction_prompt(
        self,
        current_state: Dict[str, Any],
        context: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build prompt for predicting potential issues.
        
        Args:
            current_state: Current state of the system
            context: Current context
            history: Historical data for pattern recognition
            
        Returns:
            Prompt string
        """
        state_str = "\n".join([
            f"- {k}: {str(v)[:100]}"
            for k, v in current_state.items()
        ])
        
        history_section = ""
        if history:
            history_section = f"""
## Recent History
{chr(10).join([
    f"- {h.get('event', 'Unknown')}: {h.get('outcome', 'Unknown')}"
    for h in history[-10:]
])}
"""
        
        return f"""You are an expert at predicting potential issues in software development.

## Current State
{state_str}

## Context
{chr(10).join([f"- {k}: {str(v)[:100]}" for k, v in list(context.items())[:5]])}
{history_section}
## Prediction Task
Based on the current state and history, predict potential issues:

1. What could go wrong in the next steps?
2. What risks should be mitigated?
3. What preventive actions should be taken?

## Output Format
Provide predictions in this format:

PREDICTIONS:
- ISSUE_TYPE: [type] | RISK: [HIGH/MEDIUM/LOW] | PROBABILITY: [0.0-1.0] | DESCRIPTION: [description] | PREVENTION: [suggestion]

Example:
- ISSUE_TYPE: NULL_POINTER | RISK: HIGH | PROBABILITY: 0.8 | DESCRIPTION: Mock not properly initialized | PREVENTION: Add when().thenReturn() for mock

Provide your predictions:"""

    def build_reflection_prompt(
        self,
        action_taken: str,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for reflecting on an action.
        
        Args:
            action_taken: Description of the action taken
            result: Result of the action
            context: Context in which the action was taken
            
        Returns:
            Prompt string
        """
        success = result.get("success", False)
        outcome = "successful" if success else "unsuccessful"
        
        return f"""You are reflecting on an action you just took.

## Action Taken
{action_taken}

## Result
**Outcome:** {outcome}
**Details:** {result.get('message', 'No details')}

## Context
{chr(10).join([f"- {k}: {str(v)[:100]}" for k, v in list(context.items())[:5]])}

## Reflection Task
Reflect on this action:

1. Did the action achieve its intended goal?
2. What went well? What could have been better?
3. What did you learn from this?
4. How can this inform future decisions?

## Output Format
Provide your reflection in this format:

OUTCOME_ANALYSIS: [Analysis of the outcome]
WHAT_WENT_WELL: [Positive aspects]
WHAT_TO_IMPROVE: [Areas for improvement]
LESSONS_LEARNED: [Key takeaways]
FUTURE_RECOMMENDATIONS: [Recommendations for similar situations]

Provide your reflection:"""

    def build_smart_compilation_analysis_prompt(
        self,
        error_context: Any,
        attempt_history: List[Dict[str, Any]]
    ) -> str:
        """Build intelligent compilation error analysis prompt.
        
        This prompt includes full compilation output and context for LLM analysis.
        
        Args:
            error_context: CompilationErrorContext with full error details
            attempt_history: History of previous fix attempts
            
        Returns:
            Prompt string
        """
        from ...utils.code_extractor import CodeExtractor
        
        compiler_output = error_context.get_truncated_output(10000) if hasattr(error_context, 'get_truncated_output') else str(error_context.compiler_output)[:10000]
        
        history_str = ""
        if attempt_history:
            history_str = f"""
## Previous Fix Attempts
{chr(10).join([
    f"Attempt {a.get('attempt', i+1)}: {a.get('action', 'unknown')} - {'Success' if a.get('success') else 'Failed'}"
    for i, a in enumerate(attempt_history[-5:])
])}
"""
        
        test_code_section = ""
        if error_context.test_code:
            test_code_section = f"""
## Current Test Code
```java
{error_context.test_code[:3000]}
```
"""
        
        source_code_section = ""
        if error_context.source_code:
            source_code_section = f"""
## Target Class Source (for reference)
```java
{error_context.source_code[:2000]}
```
"""
        
        return f"""You are an expert Java developer specializing in debugging compilation errors.

## Task
Analyze the compilation errors and provide a specific action plan to fix them.

## Compilation Errors (Attempt {error_context.attempt_number})
```
{compiler_output}
```

## Error Summary
- Total Errors: {error_context.error_count}
- Missing Imports: {len(error_context.missing_imports)} - {error_context.missing_imports[:5]}
- Missing Dependencies: {len(error_context.missing_dependencies)} - {error_context.missing_dependencies[:5]}
- Syntax Errors: {len(error_context.syntax_errors)}
- Type Errors: {len(error_context.type_errors)}

## Files
- Source File: {error_context.source_file}
- Test File: {error_context.test_file}
{history_str}
{test_code_section}
{source_code_section}
## {generate_prompt_action_list([ActionCategory.COMPILATION, ActionCategory.DEPENDENCY, ActionCategory.GENERAL])}

{generate_prompt_examples([ActionCategory.COMPILATION, ActionCategory.DEPENDENCY, ActionCategory.GENERAL])}

## Important Notes for Import Statements

When using fix_imports action:
- Provide ONLY the fully qualified class names (e.g., "java.sql.Connection", NOT "import java.sql.Connection;")
- Do NOT include the "import" keyword or semicolons
- The system will automatically format them correctly

Example:
```yaml
- action: fix_imports
  imports: ["java.sql.Connection", "javax.sql.DataSource", "org.junit.jupiter.api.Test"]
```

## Output Format
Provide your analysis in this EXACT format:

ROOT_CAUSE: [One sentence describing the root cause]
ANALYSIS: [Brief analysis of what went wrong]
CONFIDENCE: [0.0-1.0]

ACTION_PLAN:
- action: [action_type - MUST be one of the actions listed above]
  [action-specific parameters as shown in examples]

REASONING: [Why these actions will fix the problem]

Provide your analysis:"""

    def build_smart_test_failure_analysis_prompt(
        self,
        error_context: Any,
        attempt_history: List[Dict[str, Any]]
    ) -> str:
        """Build intelligent test failure analysis prompt.
        
        This prompt includes full test output and context for LLM analysis.
        
        Args:
            error_context: TestFailureContext with full failure details
            attempt_history: History of previous fix attempts
            
        Returns:
            Prompt string
        """
        test_output = error_context.get_truncated_output(10000) if hasattr(error_context, 'get_truncated_output') else str(error_context.test_output)[:10000]
        
        history_str = ""
        if attempt_history:
            history_str = f"""
## Previous Fix Attempts
{chr(10).join([
    f"Attempt {a.get('attempt', i+1)}: {a.get('action', 'unknown')} - {'Success' if a.get('success') else 'Failed'}"
    for i, a in enumerate(attempt_history[-5:])
])}
"""
        
        failures_section = ""
        if error_context.failed_tests:
            failures_section = f"""
## Failed Tests Detail
{chr(10).join([
    f"""
### {f.test_method} ({f.test_class})
- Type: {f.failure_type}
- Message: {f.failure_message[:200]}
- Expected: {f.expected_value[:100] if f.expected_value else 'N/A'}
- Actual: {f.actual_value[:100] if f.actual_value else 'N/A'}
"""
    for f in error_context.failed_tests[:5]
])}
"""
        
        test_code_section = ""
        if error_context.test_code:
            test_code_section = f"""
## Current Test Code
```java
{error_context.test_code[:3000]}
```
"""
        
        source_code_section = ""
        if error_context.source_code:
            source_code_section = f"""
## Target Class Source (for reference)
```java
{error_context.source_code[:2000]}
```
"""
        
        return f"""You are an expert Java developer specializing in debugging test failures.

## Task
Analyze the test failures and provide a specific action plan to fix them.

## Test Results (Attempt {error_context.attempt_number})
- Total Tests: {error_context.total_tests}
- Passed: {error_context.passed_count}
- Failed: {error_context.failed_count}
- Skipped: {error_context.skipped_count}
- Success Rate: {error_context.success_rate:.1%}

## Test Output
```
{test_output}
```
{failures_section}
## Files
- Source File: {error_context.source_file}
- Test File: {error_context.test_file}
{history_str}
{test_code_section}
{source_code_section}
## {generate_prompt_action_list([ActionCategory.TEST_FAILURE, ActionCategory.GENERAL])}

{generate_prompt_examples([ActionCategory.TEST_FAILURE, ActionCategory.GENERAL])}

## Output Format
Provide your analysis in this EXACT format:

ROOT_CAUSE: [One sentence describing the root cause]
ANALYSIS: [Brief analysis of what went wrong]
CONFIDENCE: [0.0-1.0]

ACTION_PLAN:
- action: [action_type - MUST be one of the actions listed above]
  [action-specific parameters as shown in examples]

REASONING: [Why these actions will fix the problem]

Provide your analysis:"""

    def build_smart_action_plan_prompt(
        self,
        analysis_result: Dict[str, Any],
        available_actions: List[str]
    ) -> str:
        """Build prompt for converting analysis to concrete action plan.
        
        Args:
            analysis_result: LLM analysis result
            available_actions: List of available action types
            
        Returns:
            Prompt string
        """
        return f"""Based on the following analysis, generate a concrete action plan.

## Analysis Result
- Root Cause: {analysis_result.get('root_cause', 'Unknown')}
- Confidence: {analysis_result.get('confidence', 0.5):.2f}
- Reasoning: {analysis_result.get('reasoning', 'No reasoning provided')}

## Available Actions
{chr(10).join([f"- {a}" for a in available_actions])}

## Current Action Plan
{chr(10).join([f"- {a.get('action', 'unknown')}: {a.get('description', '')}" for a in analysis_result.get('action_plan', [])])}

## Task
Refine the action plan to be more specific and actionable. For each action:
1. Ensure all required parameters are specified
2. Add any missing details
3. Order actions by priority

## Output Format
ACTION_PLAN:
- action: [action_type]
  [parameter1]: [value1]
  [parameter2]: [value2]
  ...

Provide the refined action plan:"""


class ToolUsagePromptBuilder:
    """Builds prompts for tool usage in agents."""
    
    SYSTEM_PROMPT = """You are an intelligent programming assistant with powerful tool usage capabilities.

## Available Tools
You can interact with the system through the following tools:

### File Operations
- read_file: Read file contents
- write_file: Create or write to files
- edit_file: Modify files (Search/Replace)
- glob: Find files matching pattern

### Code Search
- grep: Search for text or regex in code

### Command Execution
- bash: Execute shell commands

### Git Operations
- git_status: Check repository status
- git_diff: View file changes
- git_commit: Commit changes
- git_branch: Branch management
- git_log: View commit history
- git_add: Stage files
- git_push: Push to remote
- git_pull: Pull from remote

## Tool Usage Principles
1. Prefer using tools over directly generating code
2. Read files using read_file instead of assuming content
3. Use bash for building and testing
4. Use git_diff to check changes before committing
5. Always verify tool results before proceeding
6. When uncertain about file content, use read_file first"""

    TOOL_SELECTION_PROMPT = """Analyze the following task and determine which tools to use:

Task: {task}

Available Tools:
{available_tools}

Previous Context:
{context}

Select the appropriate tool(s) and provide:
1. Tool name(s) to use
2. Parameters for each tool
3. Reasoning for selection"""

    TOOL_RESULT_ANALYSIS_PROMPT = """Analyze the tool execution result and determine next steps:

Tool: {tool_name}
Parameters: {parameters}
Result: {result}
Success: {success}

Task Goal: {goal}

Determine:
1. Was the tool execution successful?
2. What does the result tell us?
3. Should we continue with the same tool, use a different tool, or proceed to the next step?
4. What is the next action?"""