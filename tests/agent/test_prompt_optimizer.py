"""Unit tests for PromptOptimizer."""

import pytest
import os
import tempfile
from pathlib import Path

from pyutagent.agent.prompt_optimizer import (
    PromptOptimizer,
    ModelCharacteristics,
    ModelType,
    PromptStrategy,
    PromptTemplate,
    FewShotExample,
    ABTest,
    ABTestVariant,
    optimize_prompt,
    get_few_shot_prompt
)


class TestModelCharacteristics:
    """Tests for ModelCharacteristics."""
    
    def test_detect_model_type_gpt4(self):
        """Test detecting GPT-4 model type."""
        assert ModelCharacteristics.detect_model_type("gpt-4") == ModelType.GPT_4
        assert ModelCharacteristics.detect_model_type("gpt-4-turbo") == ModelType.GPT_4
        assert ModelCharacteristics.detect_model_type("openai/gpt-4") == ModelType.GPT_4
    
    def test_detect_model_type_gpt4o(self):
        """Test detecting GPT-4o model type."""
        assert ModelCharacteristics.detect_model_type("gpt-4o") == ModelType.GPT_4O
        assert ModelCharacteristics.detect_model_type("gpt-4o-latest") == ModelType.GPT_4O
    
    def test_detect_model_type_claude(self):
        """Test detecting Claude model type."""
        assert ModelCharacteristics.detect_model_type("claude-3-opus") == ModelType.CLAUDE_3_OPUS
        assert ModelCharacteristics.detect_model_type("claude-3-sonnet") == ModelType.CLAUDE_3_SONNET
        assert ModelCharacteristics.detect_model_type("claude-3-haiku") == ModelType.CLAUDE_3_HAIKU
    
    def test_detect_model_type_deepseek(self):
        """Test detecting DeepSeek model type."""
        assert ModelCharacteristics.detect_model_type("deepseek-chat") == ModelType.DEEPSEEK
        assert ModelCharacteristics.detect_model_type("deepseek-coder") == ModelType.DEEPSEEK
    
    def test_detect_model_type_ollama(self):
        """Test detecting Ollama model type."""
        assert ModelCharacteristics.detect_model_type("ollama/llama2") == ModelType.OLLAMA
        assert ModelCharacteristics.detect_model_type("ollama/codellama") == ModelType.OLLAMA
    
    def test_detect_model_type_unknown(self):
        """Test detecting unknown model type."""
        assert ModelCharacteristics.detect_model_type("unknown-model") == ModelType.UNKNOWN
        assert ModelCharacteristics.detect_model_type("") == ModelType.UNKNOWN
    
    def test_get_characteristics_gpt4(self):
        """Test getting characteristics for GPT-4."""
        chars = ModelCharacteristics.get_characteristics(ModelType.GPT_4)
        
        assert chars["max_tokens"] == 8192
        assert chars["optimal_temperature"] == 0.2
        assert chars["supports_structured_output"] is True
        assert chars["prefers_detailed_instructions"] is True
        assert chars["few_shot_optimal"] is True
        assert chars["chain_of_thought_optimal"] is True
    
    def test_get_characteristics_claude(self):
        """Test getting characteristics for Claude."""
        chars = ModelCharacteristics.get_characteristics(ModelType.CLAUDE_3_OPUS)
        
        assert chars["max_tokens"] == 200000
        assert chars["optimal_temperature"] == 0.2
        assert chars["supports_structured_output"] is True
    
    def test_get_characteristics_unknown(self):
        """Test getting characteristics for unknown model."""
        chars = ModelCharacteristics.get_characteristics(ModelType.UNKNOWN)
        
        # Should return default characteristics
        assert "max_tokens" in chars


class TestPromptOptimizer:
    """Tests for PromptOptimizer."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def optimizer(self, temp_db):
        """Create a PromptOptimizer instance with temp database."""
        optimizer = PromptOptimizer(db_path=temp_db)
        yield optimizer
        optimizer.close()
    
    def test_initialization(self, temp_db):
        """Test PromptOptimizer initialization."""
        optimizer = PromptOptimizer(db_path=temp_db)
        
        assert optimizer.db_path == temp_db
        assert os.path.exists(temp_db)
        optimizer.close()
    
    def test_default_templates_loaded(self, optimizer):
        """Test that default templates are loaded."""
        # Check that templates table exists and has data
        import sqlite3
        with sqlite3.connect(optimizer.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM templates")
            count = cursor.fetchone()[0]
            
            assert count >= 4  # At least 4 default templates
    
    def test_optimize_for_model_gpt4(self, optimizer):
        """Test optimizing prompt for GPT-4."""
        base_prompt = "Generate a test case."
        optimized = optimizer.optimize_for_model(base_prompt, "gpt-4", "test_generation")
        
        # GPT-4 supports structured output
        assert "format" in optimized.lower() or "code block" in optimized.lower()
        # GPT-4 prefers detailed instructions
        assert "expert" in optimized.lower() or "Java" in optimized.lower()
    
    def test_optimize_for_model_claude(self, optimizer):
        """Test optimizing prompt for Claude."""
        base_prompt = "Fix this error."
        optimized = optimizer.optimize_for_model(base_prompt, "claude-3-opus", "error_fix")
        
        # Claude benefits from chain-of-thought for error fixing
        assert "step by step" in optimized.lower() or "work through" in optimized.lower()
    
    def test_optimize_for_model_ollama(self, optimizer):
        """Test optimizing prompt for Ollama."""
        base_prompt = "Generate a test case."
        optimized = optimizer.optimize_for_model(base_prompt, "ollama/llama2", "test_generation")
        
        # Ollama doesn't support structured output
        assert "format" not in optimized.lower() or "code block" not in optimized.lower()
    
    def test_select_few_shot_examples_empty(self, optimizer):
        """Test selecting examples when none exist."""
        examples = optimizer.select_few_shot_examples("test_generation", count=3)
        
        # Should return empty list when no examples
        assert isinstance(examples, list)
    
    def test_format_few_shot_examples(self, optimizer):
        """Test formatting few-shot examples."""
        examples = [
            FewShotExample(
                input_text="class Calculator { int add(int a, int b) { return a + b; } }",
                output_text="@Test void testAdd() { Calculator c = new Calculator(); assertEquals(5, c.add(2, 3)); }",
                description="Simple addition test"
            ),
            FewShotExample(
                input_text="class StringUtils { String reverse(String s) { return new StringBuilder(s).reverse().toString(); } }",
                output_text="@Test void testReverse() { StringUtils u = new StringUtils(); assertEquals('cba', u.reverse('abc')); }",
                description="String reversal test"
            )
        ]
        
        formatted = optimizer.format_few_shot_examples(examples)
        
        assert "Example 1:" in formatted
        assert "Example 2:" in formatted
        assert "Input:" in formatted
        assert "Output:" in formatted
        assert "Calculator" in formatted
        assert "StringUtils" in formatted


class TestABTesting:
    """Tests for A/B testing functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def optimizer(self, temp_db):
        """Create a PromptOptimizer instance."""
        optimizer = PromptOptimizer(db_path=temp_db)
        yield optimizer
        optimizer.close()
    
    @pytest.fixture
    def sample_templates(self):
        """Create sample templates for testing."""
        template_a = PromptTemplate(
            name="test_variant_a",
            template="Generate tests for {class_name}. Use standard approach.",
            strategy=PromptStrategy.STANDARD,
            model_type=ModelType.UNKNOWN,
            description="Standard variant"
        )
        template_b = PromptTemplate(
            name="test_variant_b",
            template="Generate tests for {class_name}. Use detailed instructions.",
            strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            model_type=ModelType.UNKNOWN,
            description="Chain-of-thought variant"
        )
        return template_a, template_b
    
    def test_create_ab_test(self, optimizer, sample_templates):
        """Test creating an A/B test."""
        template_a, template_b = sample_templates
        
        test_id = optimizer.create_ab_test(
            name="Test Generation Strategy",
            description="Compare standard vs CoT approaches",
            variants=[
                ("standard", template_a, 50.0),
                ("cot", template_b, 50.0)
            ]
        )
        
        assert test_id is not None
        assert test_id.startswith("abtest_")
    
    def test_get_prompt_for_test(self, optimizer, sample_templates):
        """Test getting prompt for A/B test."""
        template_a, template_b = sample_templates
        
        test_id = optimizer.create_ab_test(
            name="Test Generation Strategy",
            description="Compare approaches",
            variants=[
                ("standard", template_a, 50.0),
                ("cot", template_b, 50.0)
            ]
        )
        
        variant_id, prompt = optimizer.get_prompt_for_test(
            test_id,
            class_name="Calculator"
        )
        
        assert variant_id is not None
        assert prompt is not None
        assert "Calculator" in prompt
    
    def test_record_ab_test_result(self, optimizer, sample_templates):
        """Test recording A/B test result."""
        template_a, template_b = sample_templates
        
        test_id = optimizer.create_ab_test(
            name="Test Generation Strategy",
            description="Compare approaches",
            variants=[
                ("standard", template_a, 50.0),
                ("cot", template_b, 50.0)
            ]
        )
        
        variant_id, _ = optimizer.get_prompt_for_test(test_id, class_name="Test")
        
        # Record a result
        optimizer.record_ab_test_result(test_id, variant_id, success=True, response_time_ms=1500)
        
        # Analyze the test
        analysis = optimizer.analyze_ab_test(test_id)
        
        assert analysis["test_id"] == test_id
        assert analysis["total_uses"] >= 1
        assert len(analysis["variants"]) == 2
    
    def test_analyze_ab_test_with_winner(self, optimizer, sample_templates):
        """Test A/B test analysis with winner determination."""
        template_a, template_b = sample_templates
        
        test_id = optimizer.create_ab_test(
            name="Test Generation Strategy",
            description="Compare approaches",
            variants=[
                ("standard", template_a, 50.0),
                ("cot", template_b, 50.0)
            ]
        )
        
        # Simulate many results favoring variant A
        for _ in range(60):
            optimizer.record_ab_test_result(test_id, f"{test_id}_standard", success=True, response_time_ms=1000)
        for _ in range(40):
            optimizer.record_ab_test_result(test_id, f"{test_id}_cot", success=False, response_time_ms=2000)
        
        analysis = optimizer.analyze_ab_test(test_id)
        
        assert analysis["total_uses"] == 100
        # Should suggest a winner with sufficient data
        if "suggested_winner" in analysis:
            assert analysis["suggested_winner"] == f"{test_id}_standard"


class TestTemplateManagement:
    """Tests for template management."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    @pytest.fixture
    def optimizer(self, temp_db):
        """Create a PromptOptimizer instance."""
        optimizer = PromptOptimizer(db_path=temp_db)
        yield optimizer
        optimizer.close()
    
    def test_get_best_template(self, optimizer):
        """Test getting best template."""
        template = optimizer.get_best_template("test_generation", ModelType.GPT_4)
        
        # Should return a template (default ones are loaded)
        assert template is not None
        assert isinstance(template, PromptTemplate)
    
    def test_record_template_usage(self, optimizer):
        """Test recording template usage."""
        # First get a template
        template = optimizer.get_best_template("test_generation", ModelType.GPT_4)
        
        # Record usage
        optimizer.record_template_usage(
            template_name=template.name,
            model_type=ModelType.GPT_4,
            success=True,
            response_time_ms=1200
        )
        
        # Verify it was recorded
        import sqlite3
        with sqlite3.connect(optimizer.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM prompt_usage_log WHERE template_name = ?",
                (template.name,)
            )
            count = cursor.fetchone()[0]
            assert count >= 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_optimize_prompt(self):
        """Test optimize_prompt convenience function."""
        prompt = "Generate a test case."
        optimized = optimize_prompt(prompt, "gpt-4", "test_generation")
        
        assert optimized is not None
        assert len(optimized) >= len(prompt)
    
    def test_get_few_shot_prompt(self):
        """Test get_few_shot_prompt convenience function."""
        base_prompt = "Generate tests for this class."
        
        # Without examples in DB, should return base prompt
        result = get_few_shot_prompt(base_prompt, "test_generation", example_count=3)
        
        assert base_prompt in result


class TestEdgeCases:
    """Tests for edge cases."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)
    
    def test_empty_prompt(self, temp_db):
        """Test optimizing empty prompt."""
        optimizer = PromptOptimizer(db_path=temp_db)
        
        optimized = optimizer.optimize_for_model("", "gpt-4")
        
        optimizer.close()
        
        assert isinstance(optimized, str)
    
    def test_invalid_model_name(self, temp_db):
        """Test optimizing for invalid model name."""
        optimizer = PromptOptimizer(db_path=temp_db)
        
        optimized = optimizer.optimize_for_model("Test prompt", "invalid-model-name")
        
        optimizer.close()
        
        assert isinstance(optimized, str)
        assert "Test prompt" in optimized
    
    def test_nonexistent_ab_test(self, temp_db):
        """Test accessing non-existent A/B test."""
        optimizer = PromptOptimizer(db_path=temp_db)
        
        variant_id, prompt = optimizer.get_prompt_for_test("nonexistent_test_id")
        
        optimizer.close()
        
        assert variant_id is None
        assert prompt is None
    
    def test_ab_test_variant_selection_distribution(self, temp_db):
        """Test A/B test variant selection distribution."""
        optimizer = PromptOptimizer(db_path=temp_db)
        
        template_a = PromptTemplate(
            name="variant_a",
            template="Template A",
            strategy=PromptStrategy.STANDARD,
            model_type=ModelType.UNKNOWN
        )
        template_b = PromptTemplate(
            name="variant_b",
            template="Template B",
            strategy=PromptStrategy.STANDARD,
            model_type=ModelType.UNKNOWN
        )
        
        test_id = optimizer.create_ab_test(
            name="Distribution Test",
            description="Test distribution",
            variants=[
                ("a", template_a, 70.0),
                ("b", template_b, 30.0)
            ]
        )
        
        counts = {"a": 0, "b": 0}
        for _ in range(100):
            variant_id, _ = optimizer.get_prompt_for_test(test_id)
            if "_a" in variant_id:
                counts["a"] += 1
            else:
                counts["b"] += 1
        
        optimizer.close()
        
        assert counts["a"] > counts["b"]
        assert counts["a"] > 50


class TestFewShotExample:
    """Tests for FewShotExample dataclass."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        example = FewShotExample(
            input_text="input",
            output_text="output",
            success_count=8,
            use_count=10
        )
        
        assert example.success_rate == 0.8
    
    def test_success_rate_zero_uses(self):
        """Test success rate with zero uses."""
        example = FewShotExample(
            input_text="input",
            output_text="output",
            success_count=0,
            use_count=0
        )
        
        assert example.success_rate == 0.0


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""
    
    def test_format_template(self):
        """Test template formatting."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}, welcome to {place}!",
            strategy=PromptStrategy.STANDARD,
            model_type=ModelType.UNKNOWN
        )
        
        formatted = template.format(name="Alice", place="Wonderland")
        
        assert formatted == "Hello Alice, welcome to Wonderland!"
    
    def test_format_template_missing_variable(self):
        """Test template formatting with missing variable."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!",
            strategy=PromptStrategy.STANDARD,
            model_type=ModelType.UNKNOWN
        )
        
        # Should return original template if variable missing
        formatted = template.format()
        
        assert formatted == "Hello {name}!"
    
    def test_success_rate_calculation(self):
        """Test template success rate calculation."""
        template = PromptTemplate(
            name="test",
            template="Test template",
            strategy=PromptStrategy.STANDARD,
            model_type=ModelType.UNKNOWN,
            success_count=75,
            use_count=100
        )
        
        assert template.success_rate == 0.75


class TestABTestVariant:
    """Tests for ABTestVariant dataclass."""
    
    def test_success_rate_and_response_time(self):
        """Test variant metrics calculation."""
        template = PromptTemplate(
            name="test",
            template="Template",
            strategy=PromptStrategy.STANDARD,
            model_type=ModelType.UNKNOWN
        )
        
        variant = ABTestVariant(
            variant_id="v1",
            name="Variant 1",
            prompt_template=template,
            use_count=10,
            success_count=7,
            total_response_time_ms=15000
        )
        
        assert variant.success_rate == 0.7
        assert variant.avg_response_time_ms == 1500.0
    
    def test_metrics_zero_uses(self):
        """Test metrics with zero uses."""
        template = PromptTemplate(
            name="test",
            template="Template",
            strategy=PromptStrategy.STANDARD,
            model_type=ModelType.UNKNOWN
        )
        
        variant = ABTestVariant(
            variant_id="v1",
            name="Variant 1",
            prompt_template=template,
            use_count=0,
            success_count=0,
            total_response_time_ms=0
        )
        
        assert variant.success_rate == 0.0
        assert variant.avg_response_time_ms == 0.0
