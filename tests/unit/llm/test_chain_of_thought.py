"""Unit tests for ChainOfThoughtEngine module."""

import pytest

from pyutagent.llm.chain_of_thought import (
    ChainOfThoughtEngine,
    ChainOfThoughtPrompt,
    ThoughtStep,
    PromptCategory,
    ReasoningStep,
)


class TestChainOfThoughtEngine:
    """Tests for ChainOfThoughtEngine class."""

    def test_init(self):
        """Test initialization."""
        engine = ChainOfThoughtEngine()
        
        assert engine._prompts is not None
        assert len(engine._prompts) > 0
        assert engine._category_index is not None

    def test_get_prompt_existing(self):
        """Test getting an existing prompt."""
        engine = ChainOfThoughtEngine()
        
        prompt = engine.get_prompt("comprehensive_test_generation")
        
        assert prompt is not None
        assert prompt.name == "comprehensive_test_generation"
        assert prompt.category == PromptCategory.TEST_GENERATION

    def test_get_prompt_non_existing(self):
        """Test getting a non-existing prompt."""
        engine = ChainOfThoughtEngine()
        
        prompt = engine.get_prompt("non_existing_prompt")
        
        assert prompt is None

    def test_get_prompts_by_category(self):
        """Test getting prompts by category."""
        engine = ChainOfThoughtEngine()
        
        prompts = engine.get_prompts_by_category(PromptCategory.TEST_GENERATION)
        
        assert len(prompts) > 0
        for p in prompts:
            assert p.category == PromptCategory.TEST_GENERATION

    def test_render_prompt(self):
        """Test rendering a prompt."""
        engine = ChainOfThoughtEngine()
        
        context = {
            "source_code": "public class Test {}",
            "class_name": "Test",
            "package_name": "com.example",
            "dependencies": "None"
        }
        
        rendered = engine.render_prompt("comprehensive_test_generation", context)
        
        assert "Test" in rendered
        assert "com.example" in rendered
        assert "step by step" in rendered.lower()

    def test_render_prompt_with_missing_variables(self):
        """Test rendering with missing variables."""
        engine = ChainOfThoughtEngine()
        
        context = {
            "source_code": "public class Test {}",
        }
        
        rendered = engine.render_prompt("comprehensive_test_generation", context)
        
        assert rendered is not None
        assert "Test" in rendered

    def test_select_best_prompt_test_generation(self):
        """Test selecting best prompt for test generation."""
        engine = ChainOfThoughtEngine()
        
        context = {"task_type": "generate tests"}
        
        prompt = engine.select_best_prompt(context)
        
        assert prompt is not None
        assert prompt.category == PromptCategory.TEST_GENERATION

    def test_select_best_prompt_error_analysis(self):
        """Test selecting best prompt for error analysis."""
        engine = ChainOfThoughtEngine()
        
        context = {"task_type": "fix compilation error"}
        
        prompt = engine.select_best_prompt(context)
        
        assert prompt is not None
        assert prompt.category == PromptCategory.ERROR_ANALYSIS

    def test_select_best_prompt_coverage(self):
        """Test selecting best prompt for coverage improvement."""
        engine = ChainOfThoughtEngine()
        
        context = {"task_type": "improve coverage"}
        
        prompt = engine.select_best_prompt(context)
        
        assert prompt is not None
        assert prompt.category == PromptCategory.COVERAGE_IMPROVEMENT

    def test_select_best_prompt_mock(self):
        """Test selecting best prompt for mock generation."""
        engine = ChainOfThoughtEngine()
        
        context = {"task_type": "generate mocks"}
        
        prompt = engine.select_best_prompt(context)
        
        assert prompt is not None
        assert prompt.category == PromptCategory.MOCK_GENERATION

    def test_generate_reasoning_prompt(self):
        """Test generating reasoning prompt."""
        engine = ChainOfThoughtEngine()
        
        context = {
            "source_code": "public int add(int a, int b) { return a + b; }",
            "class_name": "Calculator"
        }
        
        prompt = engine.generate_reasoning_prompt("generate tests", context)
        
        assert prompt is not None
        assert len(prompt) > 0

    def test_get_available_prompts(self):
        """Test getting available prompts."""
        engine = ChainOfThoughtEngine()
        
        prompts = engine.get_available_prompts()
        
        assert len(prompts) > 0
        for p in prompts:
            assert "name" in p
            assert "category" in p
            assert "description" in p

    def test_add_custom_prompt(self):
        """Test adding custom prompt."""
        engine = ChainOfThoughtEngine()
        
        custom_prompt = engine.add_custom_prompt(
            name="my_custom_prompt",
            category=PromptCategory.TEST_GENERATION,
            description="Custom test generation prompt",
            base_template="Generate tests for ${class_name}",
            variables=["class_name"],
            thought_steps=[
                {
                    "step_type": "analyze",
                    "instruction": "Analyze the code",
                    "expected_output": "Analysis result"
                }
            ]
        )
        
        assert custom_prompt is not None
        assert custom_prompt.name == "my_custom_prompt"
        
        retrieved = engine.get_prompt("my_custom_prompt")
        assert retrieved is not None

    def test_builtin_prompts_count(self):
        """Test that built-in prompts are loaded."""
        engine = ChainOfThoughtEngine()
        
        prompts = engine.get_available_prompts()
        
        assert len(prompts) >= 5


class TestChainOfThoughtPrompt:
    """Tests for ChainOfThoughtPrompt dataclass."""

    def test_prompt_creation(self):
        """Test prompt creation."""
        prompt = ChainOfThoughtPrompt(
            name="test_prompt",
            category=PromptCategory.TEST_GENERATION,
            description="Test description",
            thought_steps=[
                ThoughtStep(ReasoningStep.ANALYZE, "Analyze code", "Analysis")
            ],
            base_template="Template ${var}",
            variables=["var"]
        )
        
        assert prompt.name == "test_prompt"
        assert prompt.category == PromptCategory.TEST_GENERATION
        assert len(prompt.thought_steps) == 1

    def test_prompt_render(self):
        """Test prompt rendering."""
        prompt = ChainOfThoughtPrompt(
            name="test_prompt",
            category=PromptCategory.TEST_GENERATION,
            description="Test description",
            thought_steps=[],
            base_template="Hello ${name}, welcome to ${place}!",
            variables=["name", "place"]
        )
        
        rendered = prompt.render({"name": "Alice", "place": "Wonderland"})
        
        assert "Alice" in rendered
        assert "Wonderland" in rendered

    def test_prompt_render_missing_variable(self):
        """Test prompt rendering with missing variable."""
        prompt = ChainOfThoughtPrompt(
            name="test_prompt",
            category=PromptCategory.TEST_GENERATION,
            description="Test description",
            thought_steps=[],
            base_template="Hello ${name} from ${place}",
            variables=["name", "place"]
        )
        
        rendered = prompt.render({"name": "Bob"})
        
        assert "Bob" in rendered
        assert "[place]" in rendered


class TestThoughtStep:
    """Tests for ThoughtStep dataclass."""

    def test_thought_step_creation(self):
        """Test thought step creation."""
        step = ThoughtStep(
            step_type=ReasoningStep.ANALYZE,
            instruction="Analyze the source code structure",
            expected_output="Class structure analysis",
            examples=["Example 1", "Example 2"]
        )
        
        assert step.step_type == ReasoningStep.ANALYZE
        assert len(step.examples) == 2

    def test_thought_step_minimal(self):
        """Test minimal thought step."""
        step = ThoughtStep(
            step_type=ReasoningStep.GENERATE,
            instruction="Generate test code",
            expected_output="Test code"
        )
        
        assert step.step_type == ReasoningStep.GENERATE
        assert step.examples == []


class TestPromptCategory:
    """Tests for PromptCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert PromptCategory.TEST_GENERATION.value == "test_generation"
        assert PromptCategory.ERROR_ANALYSIS.value == "error_analysis"
        assert PromptCategory.COVERAGE_IMPROVEMENT.value == "coverage_improvement"
        assert PromptCategory.MOCK_GENERATION.value == "mock_generation"
        assert PromptCategory.ASSERTION_DESIGN.value == "assertion_design"


class TestReasoningStep:
    """Tests for ReasoningStep enum."""

    def test_step_values(self):
        """Test step enum values."""
        assert ReasoningStep.ANALYZE.value == "analyze"
        assert ReasoningStep.UNDERSTAND.value == "understand"
        assert ReasoningStep.IDENTIFY.value == "identify"
        assert ReasoningStep.PLAN.value == "plan"
        assert ReasoningStep.GENERATE.value == "generate"
        assert ReasoningStep.VERIFY.value == "verify"
        assert ReasoningStep.REFINE.value == "refine"


class TestBuiltinPrompts:
    """Tests for built-in prompts."""

    def test_comprehensive_test_generation_prompt(self):
        """Test comprehensive test generation prompt."""
        engine = ChainOfThoughtEngine()
        
        prompt = engine.get_prompt("comprehensive_test_generation")
        
        assert prompt is not None
        assert len(prompt.thought_steps) >= 4
        assert "source_code" in prompt.variables

    def test_error_fix_chain_prompt(self):
        """Test error fix chain prompt."""
        engine = ChainOfThoughtEngine()
        
        prompt = engine.get_prompt("error_fix_chain")
        
        assert prompt is not None
        assert prompt.category == PromptCategory.ERROR_ANALYSIS
        assert "error_message" in prompt.variables

    def test_coverage_improvement_chain_prompt(self):
        """Test coverage improvement chain prompt."""
        engine = ChainOfThoughtEngine()
        
        prompt = engine.get_prompt("coverage_improvement_chain")
        
        assert prompt is not None
        assert prompt.category == PromptCategory.COVERAGE_IMPROVEMENT

    def test_mock_generation_chain_prompt(self):
        """Test mock generation chain prompt."""
        engine = ChainOfThoughtEngine()
        
        prompt = engine.get_prompt("mock_generation_chain")
        
        assert prompt is not None
        assert prompt.category == PromptCategory.MOCK_GENERATION

    def test_assertion_design_chain_prompt(self):
        """Test assertion design chain prompt."""
        engine = ChainOfThoughtEngine()
        
        prompt = engine.get_prompt("assertion_design_chain")
        
        assert prompt is not None
        assert prompt.category == PromptCategory.ASSERTION_DESIGN


class TestPromptRendering:
    """Tests for prompt rendering scenarios."""

    def test_render_with_java_code(self):
        """Test rendering with Java code."""
        engine = ChainOfThoughtEngine()
        
        context = {
            "source_code": '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
''',
            "class_name": "Calculator",
            "package_name": "com.example",
            "dependencies": "None"
        }
        
        rendered = engine.render_prompt("comprehensive_test_generation", context)
        
        assert "Calculator" in rendered
        assert "add" in rendered

    def test_render_with_error_context(self):
        """Test rendering with error context."""
        engine = ChainOfThoughtEngine()
        
        context = {
            "error_message": "NullPointerException at line 10",
            "test_code": "@Test void test() { obj.method(); }",
            "source_code": "public class Service {}",
            "test_method": "test",
            "expected_behavior": "Should not throw NPE"
        }
        
        rendered = engine.render_prompt("error_fix_chain", context)
        
        assert "NullPointerException" in rendered

    def test_render_with_coverage_context(self):
        """Test rendering with coverage context."""
        engine = ChainOfThoughtEngine()
        
        context = {
            "current_coverage": "50",
            "target_coverage": "80",
            "uncovered_code": "if (x > 0) { ... }",
            "existing_tests": "@Test void test1() {}",
            "source_code": "public class Foo {}"
        }
        
        rendered = engine.render_prompt("coverage_improvement_chain", context)
        
        assert "50" in rendered or "coverage" in rendered.lower()
