"""CLAUDE.md support for project-level context configuration.

This module provides:
- CLAUDE.md file parsing and loading
- Project context injection into prompts
- Dynamic context updates
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaudeContextSection:
    """A section in CLAUDE.md."""
    name: str
    content: str
    priority: int = 0


@dataclass
class ProjectContext:
    """Project context loaded from CLAUDE.md."""
    file_path: Path
    project_name: str = ""
    project_type: str = ""
    architecture: str = ""
    tech_stack: List[str] = field(default_factory=list)
    coding_rules: List[str] = field(default_factory=list)
    commands: Dict[str, str] = field(default_factory=dict)
    file_patterns: Dict[str, str] = field(default_factory=dict)
    guidelines: List[str] = field(default_factory=list)
    raw_sections: List[ClaudeContextSection] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert context to prompt format."""
        sections = []

        if self.project_name:
            sections.append(f"# Project: {self.project_name}")

        if self.project_type:
            sections.append(f"Type: {self.project_type}")

        if self.tech_stack:
            sections.append(f"\n## Tech Stack\n{', '.join(self.tech_stack)}")

        if self.architecture:
            sections.append(f"\n## Architecture\n{self.architecture}")

        if self.coding_rules:
            sections.append("\n## Coding Rules\n" + "\n".join(f"- {r}" for r in self.coding_rules))

        if self.commands:
            sections.append("\n## Commands\n")
            for cmd, desc in self.commands.items():
                sections.append(f"- {cmd}: {desc}")

        if self.file_patterns:
            sections.append("\n## File Patterns\n")
            for pattern, desc in self.file_patterns.items():
                sections.append(f"- {pattern}: {desc}")

        if self.guidelines:
            sections.append("\n## Guidelines\n" + "\n".join(f"- {g}" for g in self.guidelines))

        return "\n".join(sections)


class ClaudeConfigLoader:
    """Loader for CLAUDE.md configuration files."""

    SECTION_PATTERNS = {
        "project": r"(?:^|\n)#*\s*(?:Project Name|Name)[:\s]*(.+?)(?:\n|$)",
        "type": r"(?:^|\n)#*\s*Project Type[:\s]*(.+?)(?:\n|$)",
        "tech_stack": r"(?:^|\n)#*\s*Tech(?:nology)?\s*Stack[:\s]*(.+?)(?:\n\n|$)",
        "architecture": r"(?:^|\n)#*\s*Architecture[:\s]*(.+?)(?:\n\n|$)",
        "commands": r"(?:^|\n)#*\s*Commands[:\s]*\n((?:\n?[>-]\s*.+)+)",
        "guidelines": r"(?:^|\n)#*\s*Guidelines[:\s]*\n((?:\n?[>-]\s*.+)+)",
    }

    def __init__(self, base_path: Optional[str] = None):
        """Initialize loader.

        Args:
            base_path: Base path to search for CLAUDE.md
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._cached_context: Optional[ProjectContext] = None
        self._cached_file_mtime: Optional[float] = None

    def find_claude_file(self) -> Optional[Path]:
        """Find CLAUDE.md in project hierarchy.

        Returns:
            Path to CLAUDE.md or None
        """
        search_paths = [
            self.base_path / "CLAUDE.md",
            self.base_path / ".claude" / "CLAUDE.md",
            self.base_path / ".ai" / "CLAUDE.md",
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"[ClaudeConfig] Found CLAUDE.md at: {path}")
                return path

        root = self.base_path.root
        if root != self.base_path:
            search_paths = [
                root / "CLAUDE.md",
                root / ".claude" / "CLAUDE.md",
            ]
            for path in search_paths:
                if path.exists():
                    logger.info(f"[ClaudeConfig] Found CLAUDE.md at: {path}")
                    return path

        logger.debug("[ClaudeConfig] No CLAUDE.md found")
        return None

    def load(self, force_refresh: bool = False) -> Optional[ProjectContext]:
        """Load CLAUDE.md content.

        Args:
            force_refresh: Force reload even if cached

        Returns:
            ProjectContext or None
        """
        claude_file = self.find_claude_file()
        if not claude_file:
            return None

        try:
            mtime = claude_file.stat().st_mtime

            if not force_refresh and self._cached_context:
                if self._cached_file_mtime == mtime:
                    return self._cached_context

            content = claude_file.read_text(encoding="utf-8")
            context = self._parse_content(content, claude_file)

            self._cached_context = context
            self._cached_file_mtime = mtime

            logger.info(f"[ClaudeConfig] Loaded CLAUDE.md: {claude_file}")
            return context

        except Exception as e:
            logger.error(f"[ClaudeConfig] Failed to load CLAUDE.md: {e}")
            return None

    def _parse_content(self, content: str, file_path: Path) -> ProjectContext:
        """Parse CLAUDE.md content.

        Args:
            content: File content
            file_path: Path to file

        Returns:
            ProjectContext
        """
        context = ProjectContext(file_path=file_path)

        project_match = re.search(self.SECTION_PATTERNS["project"], content, re.IGNORECASE | re.MULTILINE)
        if project_match:
            context.project_name = project_match.group(1).strip()

        type_match = re.search(self.SECTION_PATTERNS["type"], content, re.IGNORECASE | re.MULTILINE)
        if type_match:
            context.project_type = type_match.group(1).strip()

        tech_match = re.search(self.SECTION_PATTERNS["tech_stack"], content, re.IGNORECASE | re.DOTALL)
        if tech_match:
            techs = [t.strip().lstrip("-•").strip() for t in tech_match.group(1).split("\n") if t.strip()]
            context.tech_stack = [t for t in techs if t]

        arch_match = re.search(self.SECTION_PATTERNS["architecture"], content, re.IGNORECASE | re.DOTALL)
        if arch_match:
            context.architecture = arch_match.group(1).strip()

        cmd_match = re.search(self.SECTION_PATTERNS["commands"], content, re.IGNORECASE | re.MULTILINE)
        if cmd_match:
            cmd_lines = cmd_match.group(1).strip().split("\n")
            for line in cmd_lines:
                line = line.strip().lstrip("-•").strip()
                if ":" in line:
                    key, val = line.split(":", 1)
                    context.commands[key.strip()] = val.strip()

        guidelines_match = re.search(self.SECTION_PATTERNS["guidelines"], content, re.IGNORECASE | re.DOTALL)
        if guidelines_match:
            guide_lines = guidelines_match.group(1).strip().split("\n")
            context.guidelines = [g.strip().lstrip("-•").strip() for g in guide_lines if g.strip()]

        context.raw_sections = self._extract_sections(content)

        return context

    def _extract_sections(self, content: str) -> List[ClaudeContextSection]:
        """Extract all sections from content.

        Args:
            content: File content

        Returns:
            List of sections
        """
        sections = []
        lines = content.split("\n")
        current_section = None
        current_content = []
        priority = 0

        for i, line in enumerate(lines):
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                if current_section:
                    sections.append(ClaudeContextSection(
                        name=current_section,
                        content="\n".join(current_content).strip(),
                        priority=priority
                    ))

                current_section = header_match.group(2).strip()
                current_content = []
                priority = len(header_match.group(1))
            else:
                current_content.append(line)

        if current_section:
            sections.append(ClaudeContextSection(
                name=current_section,
                content="\n".join(current_content).strip(),
                priority=priority
            ))

        return sections

    def get_section(self, name: str) -> Optional[str]:
        """Get a specific section by name.

        Args:
            name: Section name

        Returns:
            Section content or None
        """
        context = self.load()
        if not context:
            return None

        for section in context.raw_sections:
            if section.name.lower() == name.lower():
                return section.content

        return None

    def inject_into_prompt(self, prompt: str, max_length: int = 4000) -> str:
        """Inject CLAUDE.md context into prompt.

        Args:
            prompt: Original prompt
            max_length: Maximum context length

        Returns:
            Prompt with injected context
        """
        context = self.load()
        if not context:
            return prompt

        context_text = context.to_prompt()
        if len(context_text) > max_length:
            context_text = context_text[:max_length] + "\n\n[Context truncated...]"

        return f"{context_text}\n\n---\n\n{prompt}"

    def clear_cache(self):
        """Clear cached context."""
        self._cached_context = None
        self._cached_file_mtime = None
        logger.debug("[ClaudeConfig] Cache cleared")


def create_claude_loader(base_path: Optional[str] = None) -> ClaudeConfigLoader:
    """Create a CLAUDE.md loader.

    Args:
        base_path: Base path

    Returns:
        ClaudeConfigLoader
    """
    return ClaudeConfigLoader(base_path)


def load_project_context(base_path: Optional[str] = None) -> Optional[ProjectContext]:
    """Quick load project context.

    Args:
        base_path: Base path

    Returns:
        ProjectContext or None
    """
    loader = ClaudeConfigLoader(base_path)
    return loader.load()
