"""Multiple edit format support for Aider integration.

This module provides support for different edit formats used by LLMs:
- diff: Aider's SEARCH/REPLACE blocks
- udiff: Unified diff format
- whole: Complete file rewrite
- diff-fenced: Gemini-compatible fenced diff format
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class EditFormat(Enum):
    """Supported edit formats."""
    DIFF = "diff"  # Aider's SEARCH/REPLACE format
    UDIFF = "udiff"  # Unified diff format
    WHOLE = "whole"  # Complete file rewrite
    DIFF_FENCED = "diff-fenced"  # Gemini-compatible fenced format


@dataclass
class EditBlock:
    """Represents a single edit operation."""
    original: str
    modified: str
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


class EditFormatParser(ABC):
    """Abstract base class for edit format parsers."""

    @abstractmethod
    def parse(self, text: str, file_path: Optional[str] = None) -> List[EditBlock]:
        """Parse edit blocks from text.

        Args:
            text: The text containing edit instructions
            file_path: Optional file path context

        Returns:
            List of EditBlock objects
        """
        pass

    @abstractmethod
    def format_prompt_instruction(self) -> str:
        """Get the prompt instruction for this format."""
        pass


class DiffParserFormat(EditFormatParser):
    """Parser for Aider's SEARCH/REPLACE format."""

    # Pattern to match SEARCH/REPLACE blocks
    SEARCH_REPLACE_PATTERN = re.compile(
        r'^<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE',
        re.MULTILINE | re.DOTALL
    )

    def parse(self, text: str, file_path: Optional[str] = None) -> List[EditBlock]:
        """Parse SEARCH/REPLACE blocks."""
        edits = []

        for match in self.SEARCH_REPLACE_PATTERN.finditer(text):
            original = match.group(1)
            modified = match.group(2)

            edits.append(EditBlock(
                original=original,
                modified=modified,
                file_path=file_path
            ))

        return edits

    def format_prompt_instruction(self) -> str:
        return """
Make changes using SEARCH/REPLACE blocks:

<<<<<<< SEARCH
exact content to find
=======
new content to replace
>>>>>>> REPLACE

Rules:
- SEARCH content must match exactly (including whitespace)
- Keep SEARCH blocks minimal - only include the lines that change
- Multiple blocks can be used for different locations
"""


class UDiffParserFormat(EditFormatParser):
    """Parser for unified diff format."""

    # Pattern to match unified diff hunks
    DIFF_HUNK_PATTERN = re.compile(
        r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n(.*?)(?=^@@ |\Z)',
        re.MULTILINE | re.DOTALL
    )

    def parse(self, text: str, file_path: Optional[str] = None) -> List[EditBlock]:
        """Parse unified diff format."""
        edits = []

        # Extract file paths from diff header if present
        old_file = None
        new_file = None

        file_pattern = re.compile(r'^--- (.*?)\n\+\+\+ (.*?)\n', re.MULTILINE)
        file_match = file_pattern.search(text)
        if file_match:
            old_file = file_match.group(1)
            new_file = file_match.group(2)

        for match in self.DIFF_HUNK_PATTERN.finditer(text):
            old_start = int(match.group(1))
            old_count = int(match.group(2)) if match.group(2) else 1
            new_start = int(match.group(3))
            new_count = int(match.group(4)) if match.group(4) else 1

            hunk_content = match.group(5)
            lines = hunk_content.split('\n')

            original_lines = []
            modified_lines = []

            for line in lines:
                if line.startswith('-'):
                    original_lines.append(line[1:])
                elif line.startswith('+'):
                    modified_lines.append(line[1:])
                elif line.startswith(' '):
                    original_lines.append(line[1:])
                    modified_lines.append(line[1:])
                elif line.startswith('\\'):
                    # "\ No newline at end of file" - skip
                    continue

            edits.append(EditBlock(
                original='\n'.join(original_lines),
                modified='\n'.join(modified_lines),
                file_path=file_path or new_file,
                line_start=new_start,
                line_end=new_start + new_count - 1
            ))

        return edits

    def format_prompt_instruction(self) -> str:
        return """
Make changes using unified diff format:

--- old_file.java
+++ new_file.java
@@ -10,5 +10,5 @@
 context line
-old line
+new line
 context line

Rules:
- Use proper diff syntax with ---/+++ headers
- Include 3 lines of context around changes
- Use - for removed lines, + for added lines
"""


class WholeFileParserFormat(EditFormatParser):
    """Parser for complete file rewrite format."""

    FENCE_PATTERN = re.compile(
        r'^```(?:\w+)?\n(.*?)\n^```',
        re.MULTILINE | re.DOTALL
    )

    def parse(self, text: str, file_path: Optional[str] = None) -> List[EditBlock]:
        """Parse complete file content from fenced code blocks."""
        edits = []

        # Try to find fenced code blocks
        for match in self.FENCE_PATTERN.finditer(text):
            content = match.group(1)

            # For whole file format, we return a single edit that replaces everything
            edits.append(EditBlock(
                original="",  # Empty means replace entire file
                modified=content,
                file_path=file_path
            ))

        # If no fences found but text looks like code, use entire text
        if not edits and self._looks_like_code(text):
            edits.append(EditBlock(
                original="",
                modified=text.strip(),
                file_path=file_path
            ))

        return edits

    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like code."""
        code_indicators = [
            'public class', 'private class', 'class ',
            'import ', 'package ',
            'def ', 'function ',
            '{', '}', ';'
        ]
        return any(indicator in text for indicator in code_indicators)

    def format_prompt_instruction(self) -> str:
        return """
Provide the complete updated file content:

```java
package com.example;

import ...;

public class MyClass {
    // complete file content
}
```

Rules:
- Provide the ENTIRE file content, not just changes
- Use proper code fences with language identifier
- Ensure the code is complete and compilable
"""


class DiffFencedParserFormat(EditFormatParser):
    """Parser for Gemini-compatible fenced diff format."""

    # Pattern for fenced diff blocks (Gemini style)
    FENCED_DIFF_PATTERN = re.compile(
        r'^```diff\n(.*?)\n^```',
        re.MULTILINE | re.DOTALL
    )

    def parse(self, text: str, file_path: Optional[str] = None) -> List[EditBlock]:
        """Parse fenced diff format (Gemini-compatible)."""
        edits = []

        for match in self.FENCED_DIFF_PATTERN.finditer(text):
            diff_content = match.group(1)

            # Parse the diff content line by line
            original_lines = []
            modified_lines = []
            context_lines = []

            for line in diff_content.split('\n'):
                if line.startswith('-'):
                    original_lines.append(line[1:])
                elif line.startswith('+'):
                    modified_lines.append(line[1:])
                elif line.startswith(' '):
                    context_lines.append(line[1:])
                    original_lines.append(line[1:])
                    modified_lines.append(line[1:])

            edits.append(EditBlock(
                original='\n'.join(original_lines),
                modified='\n'.join(modified_lines),
                file_path=file_path
            ))

        return edits

    def format_prompt_instruction(self) -> str:
        return """
Make changes using fenced diff format:

```diff
- removed line
+ added line
  unchanged context
```

Rules:
- Wrap diff in ```diff fences
- Use - prefix for removed lines
- Use + prefix for added lines
- Use space prefix for context lines
"""


class EditFormatRegistry:
    """Registry for managing edit formats and model preferences."""

    def __init__(self):
        self._parsers: Dict[EditFormat, EditFormatParser] = {
            EditFormat.DIFF: DiffParserFormat(),
            EditFormat.UDIFF: UDiffParserFormat(),
            EditFormat.WHOLE: WholeFileParserFormat(),
            EditFormat.DIFF_FENCED: DiffFencedParserFormat(),
        }

        # Model-specific format preferences
        self._model_preferences: Dict[str, EditFormat] = {
            # OpenAI models
            'gpt-4-turbo': EditFormat.UDIFF,
            'gpt-4': EditFormat.DIFF,
            'gpt-4o': EditFormat.DIFF,
            'gpt-3.5-turbo': EditFormat.DIFF,

            # Anthropic models
            'claude-3-opus': EditFormat.DIFF,
            'claude-3-sonnet': EditFormat.DIFF,
            'claude-3-haiku': EditFormat.DIFF,

            # Google models
            'gemini': EditFormat.DIFF_FENCED,
            'gemini-pro': EditFormat.DIFF_FENCED,

            # DeepSeek models
            'deepseek': EditFormat.DIFF,
            'deepseek-coder': EditFormat.DIFF,

            # Local models (Ollama)
            'codellama': EditFormat.WHOLE,
            'llama2': EditFormat.WHOLE,
        }

    def get_parser(self, format_type: EditFormat) -> EditFormatParser:
        """Get parser for a specific format."""
        return self._parsers.get(format_type, self._parsers[EditFormat.DIFF])

    def get_preferred_format(self, model_name: str) -> EditFormat:
        """Get preferred format for a model."""
        # Handle Mock objects in tests
        if not isinstance(model_name, str):
            return EditFormat.DIFF

        model_lower = model_name.lower()

        # Check for exact matches first
        if model_lower in self._model_preferences:
            return self._model_preferences[model_lower]

        # Check for partial matches
        for model_key, format_type in self._model_preferences.items():
            if model_key in model_lower:
                return format_type

        # Default to diff format
        return EditFormat.DIFF

    def set_model_preference(self, model_name: str, format_type: EditFormat):
        """Set preferred format for a model."""
        self._model_preferences[model_name.lower()] = format_type

    def get_format_instruction(self, format_type: EditFormat) -> str:
        """Get prompt instruction for a format."""
        parser = self.get_parser(format_type)
        return parser.format_prompt_instruction()

    def parse_edits(
        self,
        text: str,
        format_type: EditFormat,
        file_path: Optional[str] = None
    ) -> List[EditBlock]:
        """Parse edits using specified format."""
        parser = self.get_parser(format_type)
        return parser.parse(text, file_path)

    def auto_detect_and_parse(
        self,
        text: str,
        file_path: Optional[str] = None
    ) -> tuple[List[EditBlock], EditFormat]:
        """Auto-detect format and parse edits.

        Tries different formats in order of likelihood and returns
        the first one that successfully parses.
        """
        # Try formats in order of specificity
        formats_to_try = [
            EditFormat.DIFF,  # Most specific
            EditFormat.DIFF_FENCED,
            EditFormat.UDIFF,
            EditFormat.WHOLE,
        ]

        for fmt in formats_to_try:
            parser = self.get_parser(fmt)
            edits = parser.parse(text, file_path)
            if edits:
                return edits, fmt

        # Default to diff format even if no edits found
        return [], EditFormat.DIFF


# Global registry instance
edit_format_registry = EditFormatRegistry()
