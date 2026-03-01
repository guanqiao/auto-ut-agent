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
            cursor