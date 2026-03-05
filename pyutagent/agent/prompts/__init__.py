"""Prompts for agent operations."""

# Import from the original prompts.py using absolute import to avoid circular imports
try:
    from pyutagent.agent.prompts import PromptBuilder, ToolUsagePromptBuilder
except ImportError:
    # Fallback for when there's a circular import
    PromptBuilder = None
    ToolUsagePromptBuilder = None

# Import JaCoCo config prompts
from .jacoco_config_prompts import (
    JACOCO_CONFIG_GENERATION_PROMPT,
    JACOCO_CONFIG_ANALYSIS_PROMPT,
    JACOCO_CONFIG_PREVIEW_PROMPT,
)

__all__ = [
    "PromptBuilder",
    "ToolUsagePromptBuilder",
    "JACOCO_CONFIG_GENERATION_PROMPT",
    "JACOCO_CONFIG_ANALYSIS_PROMPT",
    "JACOCO_CONFIG_PREVIEW_PROMPT",
]
