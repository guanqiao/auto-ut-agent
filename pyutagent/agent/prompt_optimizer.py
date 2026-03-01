"""Prompt optimization and A/B testing framework for LLM interactions.

This module provides intelligent prompt optimization based on model characteristics,
few-shot example selection, and A/B testing capabilities for continuous improvement.
"""

import logging
import json
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
import hashlib

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported LLM model types."""
    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_35_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    UNKNOWN = "unknown"


class PromptStrategy(Enum):
    """Prompt optimization strategies."""
    STANDARD = auto()
    FEW_SHOT = auto()
    CHAIN_OF_THOUGHT = auto()
    STRUCTURED_OUTPUT = auto()
    ROLE_BASED = auto()


@dataclass
class FewShotExample:
    """Few-shot example for prompt enhancement."""
    input_text: str
    output_text: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    success_count: int = 0
    use_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count


@dataclass
class PromptTemplate:
    """Prompt template with metadata."""
    name: str
    template: str
    strategy: PromptStrategy
    model_type: ModelType
    description: str = ""
    tags: List[str] = field(default_factory=list)
    use_count: int = 0
    success_count: int = 0
    avg_response_time_ms: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count
    
    def format(self, **kwargs) -> str:
        """Format the template with variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing template variable: {e}")
            return self.template


@dataclass
class ABTestVariant:
    """A/B test variant."""
    variant_id: str
    name: str
    prompt_template: PromptTemplate
    traffic_percentage: float = 50.0
    use_count: int = 0
    success_count: int = 0
    total_response_time_ms: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count
    
    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time."""
        if self.use_count == 0:
            return 0.0
        return self.total_response_time_ms / self.use_count


@dataclass
class ABTest:
    """A/B test definition."""
    test_id: str
    name: str
    description: str
    variants: List[ABTestVariant]
    status: str = "running"  # running, paused, completed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    winner_id: Optional[str] = None
    confidence_level: float = 0.95
    
    def select_variant(self) -> ABTestVariant:
        """Select variant based on traffic percentage."""
        total = sum(v.traffic_percentage for v in self.variants)
        r = random.uniform(0, total)
        
        cumulative = 0
        for variant in self.variants:
            cumulative += variant.traffic_percentage
            if r <= cumulative:
                return variant
        
        return self.variants[-1]
    
    def record_result(
        self,
        variant_id: str,
        success: bool,
        response_time_ms: int
    ):
        """Record test result."""
        for variant in self.variants:
            if variant.variant_id == variant_id:
                variant.use_count += 1
                variant.total_response_time_ms += response_time_ms
                if success:
                    variant.success_count += 1
                break


class ModelCharacteristics:
    """Characteristics and optimal settings for different models."""
    
    # Model-specific characteristics
    CHARACTERISTICS = {
        ModelType.GPT_4: {
            "max_tokens": 8192,
            "optimal_temperature": 0.2,
            "supports_structured_output": True,
            "prefers_detailed_instructions": True,
            "few_shot_optimal": True,
            "chain_of_thought_optimal": True,
        },
        ModelType.GPT_4O: {
            "max_tokens": 16384,
            "optimal_temperature": 0.2,
            "supports_structured_output": True,
            "prefers_detailed_instructions": True,
            "few_shot_optimal": True,
            "chain_of_thought_optimal": True,
        },
        ModelType.GPT_4O_MINI: {
            "max_tokens": 16384,
            "optimal_temperature": 0.3,
            "supports_structured_output": True,
            "prefers_detailed_instructions": False,
            "few_shot_optimal": True,
            "chain_of_thought_optimal": False,
        },
        ModelType.GPT_35_TURBO: {
            "max_tokens": 4096,
            "optimal_temperature": 0.3,
            "supports_structured_output": True,
            "prefers_detailed_instructions": False,
            "few_shot_optimal": True,
            "chain_of_thought_optimal": False,
        },
        ModelType.CLAUDE_3_OPUS: {
            "max_tokens": 200000,
            "optimal_temperature": 0.2,
            "supports_structured_output": True,
            "prefers_detailed_instructions": True,
            "few_shot_optimal": True,
            "chain_of_thought_optimal": True,
        },
        ModelType.CLAUDE_3_SONNET: {
            "max_tokens": 200000,
            "optimal_temperature": 0.2,
            "supports_structured_output": True,
            "prefers_detailed_instructions": True,
            "few_shot_optimal": True,
            "chain_of_thought_optimal": True,
        },
        ModelType.CLAUDE_3_HAIKU: {
            "max_tokens": 200000,
            "optimal_temperature": 0.3,
            "supports_structured_output": True,
            "prefers_detailed_instructions": False,
            "few_shot_optimal": False,
            "chain_of_thought_optimal": False,
        },
        ModelType.DEEPSEEK: {
            "max_tokens": 8192,
            "optimal_temperature": 0.2,
            "supports_structured_output": True,
            "prefers_detailed_instructions": True,
            "few_shot_optimal": True,
            "chain_of_thought_optimal": True,
        },
        ModelType.OLLAMA: {
            "max_tokens": 4096,
            "optimal_temperature": 0.3,
            "supports_structured_output": False,
            "prefers_detailed_instructions": False,
            "few_shot_optimal": True,
            "chain_of_thought_optimal": False,
        },
    }
    
    @classmethod
    def get_characteristics(cls, model_type: ModelType) -> Dict[str, Any]:
        """Get characteristics for a model type."""
        return cls.CHARACTERISTICS.get(model_type, cls.CHARACTERISTICS[ModelType.UNKNOWN])
    
    @classmethod
    def detect_model_type(cls, model_name: str) -> ModelType:
        """Detect model type from model name."""
        model_lower = model_name.lower()
        
        for model_type in ModelType:
            if model_type.value in model_lower:
                return model_type
        
        # Special cases
        if "claude-3-opus" in model_lower:
            return ModelType.CLAUDE_3_OPUS
        if "claude-3-sonnet" in model_lower:
            return ModelType.CLAUDE_3_SONNET
        if "claude-3-haiku" in model_lower:
            return ModelType.CLAUDE_3_HAIKU
        if "deepseek" in model_lower:
            return ModelType.DEEPSEEK
        
        return ModelType.UNKNOWN


class PromptOptimizer:
    """Optimizes prompts based on model characteristics and historical performance."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize prompt optimizer.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            home = Path.home()
            db_dir = home / ".pyutagent"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "prompt_optimizer.db"
        
        self.db_path = str(db_path)
        self._init_database()
        self._load_default_templates()
        
        logger.info(f"[PromptOptimizer] Initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS templates (
                    name TEXT PRIMARY KEY,
                    template TEXT NOT NULL,
                    strategy TEXT DEFAULT 'standard',
                    model_type TEXT DEFAULT 'unknown',
                    description TEXT,
                    tags TEXT,
                    use_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    avg_response_time_ms REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Few-shot examples table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS few_shot_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_text TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    success_count INTEGER DEFAULT 0,
                    use_count INTEGER DEFAULT 0
                )
            ''')
            
            # A/B tests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    variants TEXT NOT NULL,
                    status TEXT DEFAULT 'running',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    winner_id TEXT,
                    confidence_level REAL DEFAULT 0.95
                )
            ''')
            
            # Prompt usage log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prompt_usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_name TEXT,
                    model_type TEXT,
                    success BOOLEAN,
                    response_time_ms INTEGER,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        default_templates = [
            PromptTemplate(
                name="test_generation_standard",
                template="""Generate JUnit 5 test cases for the following Java class:

Class: {class_name}
Package: {package}
Methods: {methods}

Requirements:
- Use JUnit 5
- Include assertions
- Test both positive and negative cases
- Use mocking where appropriate

Source code:
{source_code}""",
                strategy=PromptStrategy.STANDARD,
                model_type=ModelType.UNKNOWN,
                description="Standard test generation prompt"
            ),
            PromptTemplate(
                name="test_generation_few_shot",
                template="""Generate JUnit 5 test cases following the examples below:

{few_shot_examples}

Now generate tests for:
Class: {class_name}
Package: {package}
Methods: {methods}

Source code:
{source_code}""",
                strategy=PromptStrategy.FEW_SHOT,
                model_type=ModelType.UNKNOWN,
                description="Few-shot test generation prompt"
            ),
            PromptTemplate(
                name="test_generation_cot",
                template="""Generate JUnit 5 test cases for the following Java class.

Think through this step by step:
1. Identify the class under test and its dependencies
2. Determine what methods need testing
3. Plan test cases for each method (positive, negative, edge cases)
4. Write the test code

Class: {class_name}
Package: {package}
Methods: {methods}

Source code:
{source_code}

Let's work through this systematically:""",
                strategy=PromptStrategy.CHAIN_OF_THOUGHT,
                model_type=ModelType.UNKNOWN,
                description="Chain-of-thought test generation prompt"
            ),
            PromptTemplate(
                name="error_fix_standard",
                template="""Fix the following error in the test code:

Error: {error_message}
Error Type: {error_type}

Test Code:
{test_code}

Please provide the fixed test code.""",
                strategy=PromptStrategy.STANDARD,
                model_type=ModelType.UNKNOWN,
                description="Standard error fix prompt"
            ),
        ]
        
        for template in default_templates:
            self._save_template(template)
    
    def optimize_for_model(
        self,
        base_prompt: str,
        model_name: str,
        task_type: str = "test_generation"
    ) -> str:
        """Optimize prompt for specific model.
        
        Args:
            base_prompt: Original prompt
            model_name: Model name
            task_type: Type of task
            
        Returns:
            Optimized prompt
        """
        model_type = ModelCharacteristics.detect_model_type(model_name)
        characteristics = ModelCharacteristics.get_characteristics(model_type)
        
        optimized = base_prompt
        
        # Add structured output instructions for models that support it
        if characteristics["supports_structured_output"] and task_type == "test_generation":
            optimized += "\n\nPlease provide the response in the following format:\n"
            optimized += "1. First, explain your approach\n"
            optimized += "2. Then provide the complete test code in a code block\n"
        
        # Add chain-of-thought for models that benefit from it
        if characteristics["chain_of_thought_optimal"] and task_type == "error_fix":
            optimized = "Let's work through this step by step:\n\n" + optimized
        
        # Add role context for capable models
        if characteristics["prefers_detailed_instructions"]:
            optimized = "You are an expert Java developer specializing in unit testing.\n\n" + optimized
        
        logger.debug(f"[PromptOptimizer] Optimized prompt for {model_type.value}")
        return optimized
    
    def select_few_shot_examples(
        self,
        task_type: str,
        count: int = 3,
        min_success_rate: float = 0.5
    ) -> List[FewShotExample]:
        """Select best few-shot examples for a task.
        
        Args:
            task_type: Type of task
            count: Number of examples to select
            min_success_rate: Minimum success rate threshold
            
        Returns:
            List of few-shot examples
        """
        examples = self._load_few_shot_examples(task_type)
        
        # Filter by success rate
        filtered = [e for e in examples if e.success_rate >= min_success_rate]
        
        # Sort by success rate (descending)
        filtered.sort(key=lambda e: e.success_rate, reverse=True)
        
        # Return top examples
        return filtered[:count]
    
    def format_few_shot_examples(self, examples: List[FewShotExample]) -> str:
        """Format few-shot examples for prompt inclusion.
        
        Args:
            examples: List of examples
            
        Returns:
            Formatted examples string
        """
        formatted = []
        
        for i, example in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Input:\n{example.input_text}")
            formatted.append(f"Output:\n{example.output_text}")
            if example.description:
                formatted.append(f"Note: {example.description}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def create_ab_test(
        self,
        name: str,
        description: str,
        variants: List[Tuple[str, PromptTemplate, float]]
    ) -> str:
        """Create an A/B test.
        
        Args:
            name: Test name
            description: Test description
            variants: List of (variant_name, template, traffic_percentage)
            
        Returns:
            Test ID
        """
        import uuid
        test_id = f"abtest_{uuid.uuid4().hex[:8]}"
        
        ab_variants = []
        for variant_name, template, traffic in variants:
            variant_id = f"{test_id}_{variant_name}"
            ab_variants.append(ABTestVariant(
                variant_id=variant_id,
                name=variant_name,
                prompt_template=template,
                traffic_percentage=traffic
            ))
        
        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            variants=ab_variants
        )
        
        self._save_ab_test(test)
        
        logger.info(f"[PromptOptimizer] Created A/B test: {test_id}")
        return test_id
    
    def get_prompt_for_test(
        self,
        test_id: str,
        **template_vars
    ) -> Tuple[str, str]:
        """Get prompt for A/B test variant.
        
        Args:
            test_id: Test ID
            **template_vars: Template variables
            
        Returns:
            Tuple of (variant_id, formatted_prompt)
        """
        test = self._load_ab_test(test_id)
        
        if not test or test.status != "running":
            logger.warning(f"[PromptOptimizer] Test not found or not running: {test_id}")
            return None, None
        
        variant = test.select_variant()
        prompt = variant.prompt_template.format(**template_vars)
        
        return variant.variant_id, prompt
    
    def record_ab_test_result(
        self,
        test_id: str,
        variant_id: str,
        success: bool,
        response_time_ms: int
    ):
        """Record A/B test result.
        
        Args:
            test_id: Test ID
            variant_id: Variant ID
            success: Whether the prompt was successful
            response_time_ms: Response time in milliseconds
        """
        test = self._load_ab_test(test_id)
        
        if test:
            test.record_result(variant_id, success, response_time_ms)
            self._save_ab_test(test)
            
            logger.debug(f"[PromptOptimizer] Recorded result for {test_id}/{variant_id}: {success}")
    
    def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results.
        
        Args:
            test_id: Test ID
            
        Returns:
            Analysis results
        """
        test = self._load_ab_test(test_id)
        
        if not test:
            return {"error": "Test not found"}
        
        analysis = {
            "test_id": test_id,
            "name": test.name,
            "status": test.status,
            "total_uses": sum(v.use_count for v in test.variants),
            "variants": []
        }
        
        for variant in test.variants:
            analysis["variants"].append({
                "variant_id": variant.variant_id,
                "name": variant.name,
                "use_count": variant.use_count,
                "success_count": variant.success_count,
                "success_rate": variant.success_rate,
                "avg_response_time_ms": variant.avg_response_time_ms
            })
        
        # Determine winner if statistically significant
        if len(test.variants) == 2 and analysis["total_uses"] >= 100:
            v1, v2 = test.variants
            
            # Simple winner determination (could use proper statistical test)
            if v1.success_rate > v2.success_rate + 0.1 and v1.use_count >= 50:
                analysis["suggested_winner"] = v1.variant_id
                analysis["confidence"] = "medium"
            elif v2.success_rate > v1.success_rate + 0.1 and v2.use_count >= 50:
                analysis["suggested_winner"] = v2.variant_id
                analysis["confidence"] = "medium"
        
        return analysis
    
    def get_best_template(
        self,
        task_type: str,
        model_type: ModelType
    ) -> Optional[PromptTemplate]:
        """Get best performing template for task and model.
        
        Args:
            task_type: Type of task
            model_type: Model type
            
        Returns:
            Best template or None
        """
        templates = self._load_templates(task_type, model_type)
        
        if not templates:
            return None
        
        # Filter templates with sufficient usage
        qualified = [t for t in templates if t.use_count >= 10]
        
        if not qualified:
            return templates[0]  # Return first if no qualified templates
        
        # Sort by success rate
        qualified.sort(key=lambda t: t.success_rate, reverse=True)
        
        return qualified[0]
    
    def record_template_usage(
        self,
        template_name: str,
        model_type: ModelType,
        success: bool,
        response_time_ms: int
    ):
        """Record template usage.
        
        Args:
            template_name: Template name
            model_type: Model type
            success: Whether successful
            response_time_ms: Response time
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Update template stats
            cursor.execute('''
                UPDATE templates 
                SET use_count = use_count + 1,
                    success_count = success_count + ?,
                    avg_response_time_ms = (avg_response_time_ms * use_count + ?) / (use_count + 1)
                WHERE name = ?
            ''', (1 if success else 0, response_time_ms, template_name))
            
            # Log usage
            cursor.execute('''
                INSERT INTO prompt_usage_log 
                (template_name, model_type, success, response_time_ms)
                VALUES (?, ?, ?, ?)
            ''', (template_name, model_type.value, success, response_time_ms))
            
            conn.commit()
    
    def _save_template(self, template: PromptTemplate):
        """Save template to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO templates 
                (name, template, strategy, model_type, description, tags, use_count, success_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template.name,
                template.template,
                template.strategy.name,
                template.model_type.value,
                template.description,
                json.dumps(template.tags),
                template.use_count,
                template.success_count
            ))
            
            conn.commit()
    
    def _load_templates(
        self,
        task_type: str,
        model_type: ModelType
    ) -> List[PromptTemplate]:
        """Load templates from database."""
        templates = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM templates 
                WHERE name LIKE ? AND (model_type = ? OR model_type = 'unknown')
                ORDER BY success_count DESC
            ''', (f"{task_type}%", model_type.value))
            
            for row in cursor.fetchall():
                templates.append(PromptTemplate(
                    name=row[0],
                    template=row[1],
                    strategy=PromptStrategy[row[2]],
                    model_type=ModelType(row[3]),
                    description=row[4] or "",
                    tags=json.loads(row[5]) if row[5] else [],
                    use_count=row[6],
                    success_count=row[7],
                    avg_response_time_ms=row[8]
                ))
        
        return templates
    
    def _load_few_shot_examples(self, task_type: str) -> List[FewShotExample]:
        """Load few-shot examples from database."""
        examples = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM few_shot_examples 
                WHERE tags LIKE ?
            ''', (f"%{task_type}%",))
            
            for row in cursor.fetchall():
                examples.append(FewShotExample(
                    input_text=row[1],
                    output_text=row[2],
                    description=row[3] or "",
                    tags=json.loads(row[4]) if row[4] else [],
                    success_count=row[5],
                    use_count=row[6]
                ))
        
        return examples
    
    def _save_ab_test(self, test: ABTest):
        """Save A/B test to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ab_tests 
                (test_id, name, description, variants, status, winner_id, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                test.test_id,
                test.name,
                test.description,
                json.dumps([{
                    "variant_id": v.variant_id,
                    "name": v.name,
                    "template_name": v.prompt_template.name,
                    "traffic_percentage": v.traffic_percentage,
                    "use_count": v.use_count,
                    "success_count": v.success_count,
                    "total_response_time_ms": v.total_response_time_ms
                } for v in test.variants]),
                test.status,
                test.winner_id,
                test.confidence_level
            ))
            
            conn.commit()
    
    def _load_ab_test(self, test_id: str) -> Optional[ABTest]:
        """Load A/B test from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM ab_tests WHERE test_id = ?', (test_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            variants_data = json.loads(row[3])
            variants = []
            
            for v_data in variants_data:
                # Load template
                template = self._load_template_by_name(v_data["template_name"])
                
                if template:
                    variant = ABTestVariant(
                        variant_id=v_data["variant_id"],
                        name=v_data["name"],
                        prompt_template=template,
                        traffic_percentage=v_data["traffic_percentage"],
                        use_count=v_data.get("use_count", 0),
                        success_count=v_data.get("success_count", 0),
                        total_response_time_ms=v_data.get("total_response_time_ms", 0)
                    )
                    variants.append(variant)
            
            return ABTest(
                test_id=row[0],
                name=row[1],
                description=row[2],
                variants=variants,
                status=row[4],
                created_at=row[5],
                winner_id=row[6],
                confidence_level=row[7]
            )
    
    def _load_template_by_name(self, name: str) -> Optional[PromptTemplate]:
        """Load template by name."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM templates WHERE name = ?', (name,))
            row = cursor.fetchone()
            
            if row:
                return PromptTemplate(
                    name=row[0],
                    template=row[1],
                    strategy=PromptStrategy[row[2]],
                    model_type=ModelType(row[3]),
                    description=row[4] or "",
                    tags=json.loads(row[5]) if row[5] else [],
                    use_count=row[6],
                    success_count=row[7],
                    avg_response_time_ms=row[8]
                )
            
            return None


# Convenience functions

def optimize_prompt(
    prompt: str,
    model_name: str,
    task_type: str = "test_generation"
) -> str:
    """Quick prompt optimization.
    
    Args:
        prompt: Original prompt
        model_name: Model name
        task_type: Task type
        
    Returns:
        Optimized prompt
    """
    optimizer = PromptOptimizer()
    return optimizer.optimize_for_model(prompt, model_name, task_type)


def get_few_shot_prompt(
    base_prompt: str,
    task_type: str,
    example_count: int = 3
) -> str:
    """Get prompt with few-shot examples.
    
    Args:
        base_prompt: Base prompt
        task_type: Task type
        example_count: Number of examples
        
    Returns:
        Prompt with examples
    """
    optimizer = PromptOptimizer()
    examples = optimizer.select_few_shot_examples(task_type, example_count)
    
    if examples:
        examples_text = optimizer.format_few_shot_examples(examples)
        return f"{examples_text}\n\n{base_prompt}"
    
    return base_prompt
