"""AGENTS.md support for agent instruction specification.

This module provides:
- AGENTS.md file parsing and loading
- Agent instruction management
- Dynamic prompt generation from instructions
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentInstruction:
    """An instruction in AGENTS.md."""
    name: str
    description: str
    condition: str = ""
    action: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class AgentSpec:
    """Agent specification loaded from AGENTS.md."""
    file_path: Path
    version: str = ""
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    instructions: List[AgentInstruction] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)

    def to_prompt(self, include_rules: bool = True) -> str:
        """Convert to prompt format.

        Args:
            include_rules: Whether to include rules

        Returns:
            Prompt string
        """
        sections = []

        if self.version:
            sections.append(f"# AGENTS.md v{self.version}")

        if self.description:
            sections.append(f"\n## Description\n{self.description}")

        if self.capabilities:
            sections.append(f"\n## Capabilities\n" + "\n".join(f"- {c}" for c in self.capabilities))

        if include_rules and self.rules:
            sections.append("\n## Rules\n" + "\n".join(f"- {r}" for r in self.rules))

        if self.instructions:
            sections.append("\n## Instructions\n")
            for inst in self.instructions:
                sections.append(f"\n### {inst.name}\n{inst.description}")
                if inst.examples:
                    sections.append("\nExamples:")
                    for ex in inst.examples:
                        sections.append(f"- {ex}")

        if self.examples:
            sections.append("\n## Examples\n")
            for ex in self.examples:
                sections.append(f"\n**{ex.get('description', 'Example')}**:\n```\n{ex.get('input', '')}\n```\n")

        return "\n".join(sections)

    def match_instruction(self, context: Dict[str, Any]) -> Optional[AgentInstruction]:
        """Match an instruction based on context.

        Args:
            context: Current context

        Returns:
            Matching instruction or None
        """
        for inst in self.instructions:
            if inst.condition:
                try:
                    if self._evaluate_condition(inst.condition, context):
                        return inst
                except Exception as e:
                    logger.warning(f"Failed to evaluate condition '{inst.condition}': {e}")
        return None

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string."""
        condition = condition.lower()

        for key, value in context.items():
            key_check = key.lower().replace("_", " ")
            if key_check in condition:
                if isinstance(value, bool):
                    if value and "true" not in condition:
                        return False
                    if not value and "false" in condition:
                        return False
                elif isinstance(value, str):
                    if value.lower() not in condition:
                        return False

        return True


class AgentsSpecLoader:
    """Loader for AGENTS.md specification files."""

    SECTION_PATTERNS = {
        "version": r"(?:^|\n)#*\s*version[:\s]*(\d+\.\d+\.\d+)",
        "description": r"(?:^|\n)#*\s*Description[:\s]*(.+?)(?:\n\n|$)",
        "capabilities": r"(?:^|\n)#*\s*Capabilities[:\s]*\n((?:\n?[>-]\s*.+)+)",
        "rules": r"(?:^|\n)#*\s*Rules[:\s]*\n((?:\n?[>-]\s*.+)+)",
    }

    def __init__(self, base_path: Optional[str] = None):
        """Initialize loader.

        Args:
            base_path: Base path to search for AGENTS.md
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._cached_spec: Optional[AgentSpec] = None
        self._cached_file_mtime: Optional[float] = None

    def find_agents_file(self) -> Optional[Path]:
        """Find AGENTS.md in project hierarchy.

        Returns:
            Path to AGENTS.md or None
        """
        search_paths = [
            self.base_path / "AGENTS.md",
            self.base_path / ".claude" / "AGENTS.md",
            self.base_path / ".agents" / "AGENTS.md",
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"[AgentsSpec] Found AGENTS.md at: {path}")
                return path

        root = self.base_path.root
        if root != self.base_path:
            search_paths = [
                root / "AGENTS.md",
                root / ".claude" / "AGENTS.md",
            ]
            for path in search_paths:
                if path.exists():
                    logger.info(f"[AgentsSpec] Found AGENTS.md at: {path}")
                    return path

        logger.debug("[AgentsSpec] No AGENTS.md found")
        return None

    def load(self, force_refresh: bool = False) -> Optional[AgentSpec]:
        """Load AGENTS.md content.

        Args:
            force_refresh: Force reload even if cached

        Returns:
            AgentSpec or None
        """
        agents_file = self.find_agents_file()
        if not agents_file:
            return None

        try:
            mtime = agents_file.stat().st_mtime

            if not force_refresh and self._cached_spec:
                if self._cached_file_mtime == mtime:
                    return self._cached_spec

            content = agents_file.read_text(encoding="utf-8")
            spec = self._parse_content(content, agents_file)

            self._cached_spec = spec
            self._cached_file_mtime = mtime

            logger.info(f"[AgentsSpec] Loaded AGENTS.md: {agents_file}")
            return spec

        except Exception as e:
            logger.error(f"[AgentsSpec] Failed to load AGENTS.md: {e}")
            return None

    def _parse_content(self, content: str, file_path: Path) -> AgentSpec:
        """Parse AGENTS.md content.

        Args:
            content: File content
            file_path: Path to file

        Returns:
            AgentSpec
        """
        spec = AgentSpec(file_path=file_path)

        version_match = re.search(self.SECTION_PATTERNS["version"], content, re.IGNORECASE | re.MULTILINE)
        if version_match:
            spec.version = version_match.group(1).strip()

        desc_match = re.search(self.SECTION_PATTERNS["description"], content, re.IGNORECASE | re.DOTALL)
        if desc_match:
            spec.description = desc_match.group(1).strip()

        caps_match = re.search(self.SECTION_PATTERNS["capabilities"], content, re.IGNORECASE | re.MULTILINE)
        if caps_match:
            cap_lines = caps_match.group(1).strip().split("\n")
            spec.capabilities = [c.strip().lstrip("-•").strip() for c in cap_lines if c.strip()]

        rules_match = re.search(self.SECTION_PATTERNS["rules"], content, re.IGNORECASE | re.MULTILINE)
        if rules_match:
            rule_lines = rules_match.group(1).strip().split("\n")
            spec.rules = [r.strip().lstrip("-•").strip() for r in rule_lines if r.strip()]

        spec.instructions = self._extract_instructions(content)
        spec.examples = self._extract_examples(content)

        return spec

    def _extract_instructions(self, content: str) -> List[AgentInstruction]:
        """Extract instructions from content.

        Args:
            content: File content

        Returns:
            List of instructions
        """
        instructions = []
        lines = content.split("\n")
        current_instruction = None
        current_description = []
        in_instruction = False

        for line in lines:
            header_match = re.match(r"^###\s+(.+)$", line)
            if header_match:
                if current_instruction:
                    instructions.append(AgentInstruction(
                        name=current_instruction,
                        description="\n".join(current_description).strip()
                    ))

                current_instruction = header_match.group(1).strip()
                current_description = []
                in_instruction = True
            elif in_instruction:
                if line.strip().startswith("###"):
                    in_instruction = False
                else:
                    current_description.append(line)

        if current_instruction:
            instructions.append(AgentInstruction(
                name=current_instruction,
                description="\n".join(current_description).strip()
            ))

        return instructions

    def _extract_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract examples from content.

        Args:
            content: File content

        Returns:
            List of examples
        """
        examples = []
        example_pattern = r"\*\*([^*]+)\*\*:\s*```\n([\s\S]*?)```"

        for match in re.finditer(example_pattern, content):
            examples.append({
                "description": match.group(1).strip(),
                "input": match.group(2).strip()
            })

        return examples

    def get_instruction(self, name: str) -> Optional[AgentInstruction]:
        """Get a specific instruction by name.

        Args:
            name: Instruction name

        Returns:
            Instruction or None
        """
        spec = self.load()
        if not spec:
            return None

        for inst in spec.instructions:
            if inst.name.lower() == name.lower():
                return inst

        return None

    def match_instruction_by_context(self, context: Dict[str, Any]) -> Optional[AgentInstruction]:
        """Match an instruction by context.

        Args:
            context: Current context

        Returns:
            Matching instruction or None
        """
        spec = self.load()
        if not spec:
            return None

        return spec.match_instruction(context)

    def inject_into_prompt(self, prompt: str, include_rules: bool = True) -> str:
        """Inject AGENTS.md context into prompt.

        Args:
            prompt: Original prompt
            include_rules: Whether to include rules

        Returns:
            Prompt with injected context
        """
        spec = self.load()
        if not spec:
            return prompt

        spec_text = spec.to_prompt(include_rules)

        return f"{spec_text}\n\n---\n\n{prompt}"

    def clear_cache(self):
        """Clear cached spec."""
        self._cached_spec = None
        self._cached_file_mtime = None
        logger.debug("[AgentsSpec] Cache cleared")


def create_agents_loader(base_path: Optional[str] = None) -> AgentsSpecLoader:
    """Create an AGENTS.md loader.

    Args:
        base_path: Base path

    Returns:
        AgentsSpecLoader
    """
    return AgentsSpecLoader(base_path)


def load_agent_spec(base_path: Optional[str] = None) -> Optional[AgentSpec]:
    """Quick load agent specification.

    Args:
        base_path: Base path

    Returns:
        AgentSpec or None
    """
    loader = AgentsSpecLoader(base_path)
    return loader.load()
