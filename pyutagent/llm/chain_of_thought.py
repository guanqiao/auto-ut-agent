"""Chain-of-Thought prompt engineering for complex reasoning.

This module provides enhanced prompt templates with chain-of-thought reasoning
to improve the agent's ability to handle complex test generation scenarios.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from string import Template
import re

logger = logging.getLogger(__name__)


class ReasoningStep(Enum):
    """Steps in chain-of-thought reasoning."""
    ANALYZE = "analyze"
    UNDERSTAND = "understand"
    IDENTIFY = "identify"
    PLAN = "plan"
    GENERATE = "generate"
    VERIFY = "verify"
    REFINE = "refine"


class PromptCategory(Enum):
    """Categories of prompts."""
    TEST_GENERATION = "test_generation"
    ERROR_ANALYSIS = "error_analysis"
    COVERAGE_IMPROVEMENT = "coverage_improvement"
    REFACTORING = "refactoring"
    MOCK_GENERATION = "mock_generation"
    ASSERTION_DESIGN = "assertion_design"
    BOUNDARY_ANALYSIS = "boundary_analysis"
    INTEGRATION_TEST = "integration_test"


@dataclass
class ThoughtStep:
    """A step in chain-of-thought reasoning."""
    step_type: ReasoningStep
    instruction: str
    expected_output: str
    examples: List[str] = field(default_factory=list)


@dataclass
class ChainOfThoughtPrompt:
    """A chain-of-thought prompt template."""
    name: str
    category: PromptCategory
    description: str
    thought_steps: List[ThoughtStep]
    base_template: str
    variables: List[str]
    few_shot_examples: List[Dict[str, str]] = field(default_factory=list)
    
    def render(self, context: Dict[str, Any]) -> str:
        """Render the prompt with context."""
        sections = []
        
        sections.append(self.base_template)
        sections.append("")
        
        sections.append("## Let's think step by step:")
        sections.append("")
        
        for i, step in enumerate(self.thought_steps, 1):
            sections.append(f"### Step {i}: {step.step_type.value.upper()}")
            sections.append(step.instruction)
            sections.append("")
            
            if step.examples:
                sections.append("**Examples:**")
                for example in step.examples[:2]:
                    sections.append(f"- {example}")
                sections.append("")
        
        if self.few_shot_examples:
            sections.append("## Examples:")
            sections.append("")
            for example in self.few_shot_examples[:2]:
                sections.append("```")
                for key, value in example.items():
                    sections.append(f"{key}: {value}")
                sections.append("```")
                sections.append("")
        
        prompt = "\n".join(sections)
        
        for var in self.variables:
            placeholder = f"${{{var}}}"
            if placeholder in prompt:
                value = context.get(var, f"[{var}]")
                prompt = prompt.replace(placeholder, str(value))
        
        return prompt


class ChainOfThoughtEngine:
    """Engine for chain-of-thought prompt generation.
    
    Features:
    - Structured reasoning prompts
    - Few-shot learning examples
    - Step-by-step guidance
    - Context-aware prompt selection
    """
    
    def __init__(self):
        """Initialize the chain-of-thought engine."""
        self._prompts = self._initialize_prompts()
        self._category_index: Dict[PromptCategory, List[str]] = {}
        
        for prompt in self._prompts:
            if prompt.category not in self._category_index:
                self._category_index[prompt.category] = []
            self._category_index[prompt.category].append(prompt.name)
        
        logger.info(f"[ChainOfThoughtEngine] Initialized with {len(self._prompts)} prompts")
    
    def _initialize_prompts(self) -> List[ChainOfThoughtPrompt]:
        """Initialize built-in chain-of-thought prompts."""
        prompts = []
        
        prompts.append(ChainOfThoughtPrompt(
            name="comprehensive_test_generation",
            category=PromptCategory.TEST_GENERATION,
            description="Comprehensive test generation with full reasoning chain",
            thought_steps=[
                ThoughtStep(
                    step_type=ReasoningStep.ANALYZE,
                    instruction="First, analyze the source code to understand its purpose, inputs, outputs, and dependencies.",
                    expected_output="A clear understanding of what the code does",
                    examples=[
                        "This is a service class that processes user registration",
                        "The method takes a UserDTO and returns a User entity"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.IDENTIFY,
                    instruction="Identify all test scenarios including: normal flow, edge cases, error conditions, and boundary values.",
                    expected_output="List of test scenarios to cover",
                    examples=[
                        "Happy path: valid user registration",
                        "Edge case: duplicate email address",
                        "Error: invalid email format"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.PLAN,
                    instruction="Plan the test structure: what mocks are needed, what assertions to make, how to organize test methods.",
                    expected_output="Test plan with mock setup and assertions",
                    examples=[
                        "Mock UserRepository for database operations",
                        "Use @BeforeEach for common setup"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.GENERATE,
                    instruction="Generate the test code following JUnit 5 best practices and the planned structure.",
                    expected_output="Complete test class code",
                    examples=[
                        "Use @DisplayName for readable test names",
                        "Follow AAA pattern (Arrange, Act, Assert)"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.VERIFY,
                    instruction="Verify the generated tests: check coverage, ensure assertions are meaningful, validate mock behavior.",
                    expected_output="Quality check of generated tests",
                    examples=[
                        "All public methods are tested",
                        "Each test has meaningful assertions"
                    ]
                )
            ],
            base_template="""Generate comprehensive unit tests for the following Java code.

**Source Code:**
```java
${source_code}
```

**Class Information:**
- Class Name: ${class_name}
- Package: ${package_name}
- Dependencies: ${dependencies}

**Requirements:**
- Use JUnit 5 and Mockito
- Achieve high code coverage
- Follow naming conventions (should_xxx_when_xxx)
- Include both positive and negative test cases
""",
            variables=["source_code", "class_name", "package_name", "dependencies"],
            few_shot_examples=[
                {
                    "scenario": "Testing a simple calculator",
                    "approach": "Create tests for add, subtract, multiply, divide with edge cases for division by zero"
                }
            ]
        ))
        
        prompts.append(ChainOfThoughtPrompt(
            name="error_fix_chain",
            category=PromptCategory.ERROR_ANALYSIS,
            description="Chain-of-thought for analyzing and fixing test errors",
            thought_steps=[
                ThoughtStep(
                    step_type=ReasoningStep.ANALYZE,
                    instruction="Analyze the error message to understand what went wrong. Identify the error type (compilation, runtime, assertion).",
                    expected_output="Error classification and root cause",
                    examples=[
                        "Compilation error: missing import statement",
                        "Assertion error: expected value doesn't match actual"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.UNDERSTAND,
                    instruction="Understand the context: what was the test trying to do, what was the expected behavior.",
                    expected_output="Clear understanding of test intent",
                    examples=[
                        "The test was verifying that null input throws exception",
                        "Expected NullPointerException but got IllegalArgumentException"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.IDENTIFY,
                    instruction="Identify the fix needed: is it a test issue, mock setup issue, or assertion issue.",
                    expected_output="Root cause identification",
                    examples=[
                        "Mock not configured to throw exception",
                        "Wrong assertion method used"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.REFINE,
                    instruction="Apply the fix and verify it resolves the issue without breaking other tests.",
                    expected_output="Fixed test code",
                    examples=[
                        "Add when().thenThrow() for mock behavior",
                        "Change assertEquals to assertThrows"
                    ]
                )
            ],
            base_template="""Fix the following test error using systematic analysis.

**Error Message:**
```
${error_message}
```

**Failing Test:**
```java
${test_code}
```

**Source Code:**
```java
${source_code}
```

**Context:**
- Test method: ${test_method}
- Expected behavior: ${expected_behavior}
""",
            variables=["error_message", "test_code", "source_code", "test_method", "expected_behavior"],
            few_shot_examples=[
                {
                    "error": "NullPointerException in test",
                    "fix": "Initialize the mock object in @BeforeEach or add @Mock annotation"
                }
            ]
        ))
        
        prompts.append(ChainOfThoughtPrompt(
            name="coverage_improvement_chain",
            category=PromptCategory.COVERAGE_IMPROVEMENT,
            description="Chain-of-thought for improving test coverage",
            thought_steps=[
                ThoughtStep(
                    step_type=ReasoningStep.ANALYZE,
                    instruction="Analyze the coverage report to identify uncovered lines, branches, and methods.",
                    expected_output="List of uncovered code elements",
                    examples=[
                        "Line 45: if condition not covered",
                        "Branch: else block not tested"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.IDENTIFY,
                    instruction="Identify what test scenarios are needed to cover the missing code.",
                    expected_output="Required test scenarios",
                    examples=[
                        "Need test for null input to trigger if condition",
                        "Need test for empty collection to enter else branch"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.PLAN,
                    instruction="Plan how to trigger the uncovered code paths through the test.",
                    expected_output="Test strategy for coverage",
                    examples=[
                        "Create input that satisfies the condition",
                        "Mock dependency to return specific value"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.GENERATE,
                    instruction="Generate additional test methods to cover the missing scenarios.",
                    expected_output="New test methods",
                    examples=[
                        "@Test void shouldHandleNullInput()",
                        "@Test void shouldReturnEmptyWhenCollectionIsEmpty()"
                    ]
                )
            ],
            base_template="""Improve test coverage for the following code.

**Current Coverage:** ${current_coverage}%
**Target Coverage:** ${target_coverage}%

**Uncovered Code:**
```
${uncovered_code}
```

**Existing Tests:**
```java
${existing_tests}
```

**Source Code:**
```java
${source_code}
```
""",
            variables=["current_coverage", "target_coverage", "uncovered_code", "existing_tests", "source_code"],
            few_shot_examples=[
                {
                    "gap": "Exception handling not covered",
                    "solution": "Add test that triggers the exception condition"
                }
            ]
        ))
        
        prompts.append(ChainOfThoughtPrompt(
            name="mock_generation_chain",
            category=PromptCategory.MOCK_GENERATION,
            description="Chain-of-thought for generating mock configurations",
            thought_steps=[
                ThoughtStep(
                    step_type=ReasoningStep.ANALYZE,
                    instruction="Analyze the class dependencies to identify what needs to be mocked.",
                    expected_output="List of dependencies to mock",
                    examples=[
                        "UserRepository - database access",
                        "EmailService - external service"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.IDENTIFY,
                    instruction="Identify the behavior each mock should exhibit for different test scenarios.",
                    expected_output="Mock behavior specifications",
                    examples=[
                        "UserRepository.save() should return saved entity",
                        "EmailService.send() should throw exception on failure"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.PLAN,
                    instruction="Plan the mock setup: annotations, when/thenReturn configurations, verification points.",
                    expected_output="Mock setup plan",
                    examples=[
                        "Use @Mock and @InjectMocks annotations",
                        "Configure in @BeforeEach method"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.GENERATE,
                    instruction="Generate the mock configuration code following Mockito best practices.",
                    expected_output="Mock setup code",
                    examples=[
                        "@Mock private UserRepository userRepo;",
                        "when(userRepo.findById(1L)).thenReturn(Optional.of(user));"
                    ]
                )
            ],
            base_template="""Generate mock configurations for testing the following class.

**Class Under Test:**
```java
${class_code}
```

**Dependencies to Mock:**
${dependencies}

**Test Scenarios:**
${test_scenarios}
""",
            variables=["class_code", "dependencies", "test_scenarios"],
            few_shot_examples=[
                {
                    "dependency": "RestTemplate",
                    "mock": "Mock RestTemplate.exchange() to return predefined ResponseEntity"
                }
            ]
        ))
        
        prompts.append(ChainOfThoughtPrompt(
            name="assertion_design_chain",
            category=PromptCategory.ASSERTION_DESIGN,
            description="Chain-of-thought for designing effective assertions",
            thought_steps=[
                ThoughtStep(
                    step_type=ReasoningStep.ANALYZE,
                    instruction="Analyze what the method should return or what side effects it should have.",
                    expected_output="Expected outcomes",
                    examples=[
                        "Return value should be non-null",
                        "List should contain exactly 3 items"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.IDENTIFY,
                    instruction="Identify the appropriate assertion type for each expected outcome.",
                    expected_output="Assertion type selection",
                    examples=[
                        "assertEquals for exact value comparison",
                        "assertNotNull for null checks",
                        "assertThrows for exception verification"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.PLAN,
                    instruction="Plan assertions to be meaningful and provide clear failure messages.",
                    expected_output="Assertion plan with messages",
                    examples=[
                        "Add descriptive message to assertion",
                        "Verify both state and behavior"
                    ]
                ),
                ThoughtStep(
                    step_type=ReasoningStep.GENERATE,
                    instruction="Generate assertion code with proper formatting and messages.",
                    expected_output="Assertion code",
                    examples=[
                        "assertEquals(expected, actual, \"Result should match expected value\");",
                        "verify(mockService).process(any());"
                    ]
                )
            ],
            base_template="""Design assertions for the following test scenario.

**Method Under Test:**
```java
${method_code}
```

**Test Scenario:**
${scenario}

**Expected Behavior:**
${expected_behavior}

**Test Input:**
${test_input}
""",
            variables=["method_code", "scenario", "expected_behavior", "test_input"],
            few_shot_examples=[
                {
                    "scenario": "User registration success",
                    "assertions": "Verify returned User has correct email, verify repository.save() was called"
                }
            ]
        ))
        
        return prompts
    
    def get_prompt(self, name: str) -> Optional[ChainOfThoughtPrompt]:
        """Get a prompt by name."""
        for prompt in self._prompts:
            if prompt.name == name:
                return prompt
        return None
    
    def get_prompts_by_category(self, category: PromptCategory) -> List[ChainOfThoughtPrompt]:
        """Get all prompts for a category."""
        return [p for p in self._prompts if p.category == category]
    
    def render_prompt(
        self,
        name: str,
        context: Dict[str, Any]
    ) -> str:
        """Render a prompt with context."""
        prompt = self.get_prompt(name)
        if not prompt:
            return f"Prompt '{name}' not found"
        return prompt.render(context)
    
    def select_best_prompt(
        self,
        context: Dict[str, Any]
    ) -> Optional[ChainOfThoughtPrompt]:
        """Select the best prompt based on context."""
        task_type = context.get("task_type", "")
        
        if "error" in task_type.lower() or "fix" in task_type.lower():
            category = PromptCategory.ERROR_ANALYSIS
        elif "coverage" in task_type.lower():
            category = PromptCategory.COVERAGE_IMPROVEMENT
        elif "mock" in task_type.lower():
            category = PromptCategory.MOCK_GENERATION
        elif "assertion" in task_type.lower():
            category = PromptCategory.ASSERTION_DESIGN
        else:
            category = PromptCategory.TEST_GENERATION
        
        prompts = self.get_prompts_by_category(category)
        if prompts:
            return prompts[0]
        
        return self._prompts[0] if self._prompts else None
    
    def generate_reasoning_prompt(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate a reasoning prompt for a specific task."""
        prompt = self.select_best_prompt({"task_type": task})
        
        if prompt:
            return prompt.render(context)
        
        return self._generate_generic_reasoning(task, context)
    
    def _generate_generic_reasoning(self, task: str, context: Dict[str, Any]) -> str:
        """Generate a generic reasoning prompt."""
        sections = [
            f"# Task: {task}",
            "",
            "## Let's think step by step:",
            "",
            "### Step 1: ANALYZE",
            "First, understand what needs to be done and what information is available.",
            "",
            "### Step 2: IDENTIFY",
            "Identify the key components, dependencies, and constraints.",
            "",
            "### Step 3: PLAN",
            "Plan the approach and consider alternatives.",
            "",
            "### Step 4: GENERATE",
            "Generate the solution following best practices.",
            "",
            "### Step 5: VERIFY",
            "Verify the solution meets requirements and handles edge cases.",
            "",
            "## Context:",
        ]
        
        for key, value in context.items():
            sections.append(f"- {key}: {value}")
        
        return "\n".join(sections)
    
    def get_available_prompts(self) -> List[Dict[str, str]]:
        """Get list of available prompts."""
        return [
            {
                "name": p.name,
                "category": p.category.value,
                "description": p.description
            }
            for p in self._prompts
        ]
    
    def add_custom_prompt(
        self,
        name: str,
        category: PromptCategory,
        description: str,
        base_template: str,
        variables: List[str],
        thought_steps: Optional[List[Dict[str, Any]]] = None
    ) -> ChainOfThoughtPrompt:
        """Add a custom prompt template."""
        steps = []
        if thought_steps:
            for step_data in thought_steps:
                steps.append(ThoughtStep(
                    step_type=ReasoningStep(step_data.get("step_type", "analyze")),
                    instruction=step_data.get("instruction", ""),
                    expected_output=step_data.get("expected_output", ""),
                    examples=step_data.get("examples", [])
                ))
        
        prompt = ChainOfThoughtPrompt(
            name=name,
            category=category,
            description=description,
            thought_steps=steps,
            base_template=base_template,
            variables=variables
        )
        
        self._prompts.append(prompt)
        
        if category not in self._category_index:
            self._category_index[category] = []
        self._category_index[category].append(name)
        
        return prompt
