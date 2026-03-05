"""Prompts for agent operations."""

# Import from the prompts submodule
from .prompts import PromptBuilder, ToolUsagePromptBuilder

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
