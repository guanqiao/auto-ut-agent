"""Enhanced Skills Framework - Progressive Loading and Claude Code Compatibility.

This module provides:
- EnhancedSkillLoader: Progressive loading support
- SkillSummary: Lightweight skill metadata
- ClaudeCodeSkillAdapter: Claude Code format compatibility
- SkillMatcher: Intelligent skill triggering
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class SkillSummary:
    """Lightweight skill summary for listing."""
    name: str
    description: str
    category: str
    tags: List[str] = field(default_factory=list)
    file_path: Optional[str] = None


@dataclass
class SkillResource:
    """A resource file in a skill."""
    name: str
    path: Path
    content: Optional[str] = None


@dataclass
class SkillScript:
    """A script file in a skill."""
    name: str
    path: Path
    executable: bool = False


class EnhancedSkillLoader:
    """Enhanced skill loader with progressive loading.

    Features:
    - Progressive disclosure: Load only what's needed
    - Claude Code format compatibility
    - Auto-discovery of skills
    - Resource and script support
    """

    DEFAULT_SKILL_DIRS = [
        Path.home() / ".config" / "claude-code" / "skills",
        Path.home() / ".pyutagent" / "skills",
    ]

    def __init__(
        self,
        skill_dirs: Optional[List[Path]] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize enhanced skill loader.

        Args:
            skill_dirs: Directories to search for skills
            cache_dir: Directory for caching loaded skills
        """
        self.skill_dirs = skill_dirs or self.DEFAULT_SKILL_DIRS
        self.cache_dir = cache_dir or Path.home() / ".pyutagent" / "skill_cache"

        self._skill_cache: Dict[str, Dict] = {}
        self._summary_cache: Dict[str, SkillSummary] = {}

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def discover_skills(self) -> List[SkillSummary]:
        """Auto-discover all skills in configured directories.

        Returns:
            List of skill summaries
        """
        summaries = []

        for skill_dir in self.skill_dirs:
            if not skill_dir.exists():
                continue

            for skill_folder in skill_dir.iterdir():
                if not skill_folder.is_dir():
                    continue

                summary = self._discover_skill(skill_folder)
                if summary:
                    summaries.append(summary)

        logger.info(f"[EnhancedSkillLoader] Discovered {len(summaries)} skills")
        return summaries

    def _discover_skill(self, skill_folder: Path) -> Optional[SkillSummary]:
        """Discover a single skill from folder."""
        skill_md = skill_folder / "SKILL.md"

        if not skill_md.exists():
            return None

        try:
            content = skill_md.read_text(encoding="utf-8")
            name = self._extract_yaml_field(content, "name") or skill_folder.name
            description = self._extract_yaml_field(content, "description") or ""
            category = self._extract_category(content)

            tags = self._extract_tags(content)

            return SkillSummary(
                name=name,
                description=description,
                category=category,
                tags=tags,
                file_path=str(skill_folder)
            )

        except Exception as e:
            logger.warning(f"[EnhancedSkillLoader] Failed to discover {skill_folder}: {e}")
            return None

    def _extract_yaml_field(self, content: str, field_name: str) -> Optional[str]:
        """Extract field from YAML frontmatter."""
        match = re.search(rf'^{field_name}:\s*(.+)$', content, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _extract_category(self, content: str) -> str:
        """Extract category from content."""
        category_match = re.search(r'##\s+Category\s*:?\s*(\w+)', content, re.IGNORECASE)
        if category_match:
            return category_match.group(1)

        return "general"

    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from content."""
        tags = []
        tag_match = re.search(r'tags:\s*\[([^\]]+)\]', content)
        if tag_match:
            tags = [t.strip() for t in tag_match.group(1).split(",")]
        return tags

    def load_skill_summary(self, name: str) -> Optional[SkillSummary]:
        """Load only skill summary (name + description).

        Args:
            name: Skill name

        Returns:
            Skill summary or None
        """
        if name in self._summary_cache:
            return self._summary_cache[name]

        for skill_dir in self.skill_dirs:
            skill_folder = skill_dir / name
            summary = self._discover_skill(skill_folder)

            if summary:
                self._summary_cache[name] = summary
                return summary

        return None

    def load_skill_full(self, name: str) -> Optional[Dict[str, Any]]:
        """Load complete skill with all resources.

        Args:
            name: Skill name

        Returns:
            Complete skill data or None
        """
        if name in self._skill_cache:
            return self._skill_cache[name]

        skill_data = None

        for skill_dir in self.skill_dirs:
            skill_folder = skill_dir / name
            if not skill_folder.exists():
                continue

            skill_data = self._load_skill_folder(skill_folder)
            if skill_data:
                self._skill_cache[name] = skill_data
                return skill_data

        return skill_data

    def _load_skill_folder(self, folder: Path) -> Dict[str, Any]:
        """Load complete skill from folder."""
        skill_md = folder / "SKILL.md"

        if not skill_md.exists():
            return {}

        content = skill_md.read_text(encoding="utf-8")

        skill_data = {
            "name": folder.name,
            "description": self._extract_yaml_field(content, "description") or "",
            "content": content,
            "folder": str(folder),
            "resources": [],
            "scripts": [],
            "templates": []
        }

        scripts_dir = folder / "scripts"
        if scripts_dir.exists():
            for script in scripts_dir.iterdir():
                if script.is_file():
                    skill_data["scripts"].append({
                        "name": script.name,
                        "path": str(script),
                        "executable": script.stat().st_mode & 0o111 != 0
                    })

        resources_dir = folder / "resources"
        if resources_dir.exists():
            for resource in resources_dir.iterdir():
                if resource.is_file():
                    skill_data["resources"].append({
                        "name": resource.name,
                        "path": str(resource)
                    })

        templates_dir = folder / "templates"
        if templates_dir.exists():
            for template in templates_dir.iterdir():
                if template.is_file():
                    skill_data["templates"].append({
                        "name": template.name,
                        "path": str(template)
                    })

        return skill_data

    def unload_skill(self, name: str):
        """Unload skill from cache.

        Args:
            name: Skill name
        """
        if name in self._skill_cache:
            del self._skill_cache[name]
        if name in self._summary_cache:
            del self._summary_cache[name]

        logger.info(f"[EnhancedSkillLoader] Unloaded skill: {name}")

    def get_cached_skills(self) -> List[str]:
        """Get list of cached skill names.

        Returns:
            Cached skill names
        """
        return list(self._skill_cache.keys())


class ClaudeCodeSkillAdapter:
    """Adapter for Claude Code skill format.

    Converts between Claude Code format and PyUT Agent format.
    """

    @staticmethod
    def load_from_folder(folder_path: Path) -> Dict[str, Any]:
        """Load skill from Claude Code format folder.

        Args:
            folder_path: Path to skill folder

        Returns:
            Skill data dictionary
        """
        loader = EnhancedSkillLoader(skill_dirs=[folder_path.parent])
        return loader.load_skill_full(folder_path.name) or {}

    @staticmethod
    def export_to_claude_format(skill_data: Dict[str, Any], output_dir: Path) -> Path:
        """Export skill to Claude Code format.

        Args:
            skill_data: Skill data
            output_dir: Output directory

        Returns:
            Path to created skill folder
        """
        skill_name = skill_data.get("name", "unnamed-skill")
        skill_folder = output_dir / skill_name
        skill_folder.mkdir(parents=True, exist_ok=True)

        skill_md = skill_folder / "SKILL.md"
        content = skill_data.get("content", f"# {skill_name}\n\n")
        skill_md.write_text(content, encoding="utf-8")

        scripts_dir = skill_folder / "scripts"
        scripts_dir.mkdir(exist_ok=True)

        templates_dir = skill_folder / "templates"
        templates_dir.mkdir(exist_ok=True)

        resources_dir = skill_folder / "resources"
        resources_dir.mkdir(exist_ok=True)

        logger.info(f"[ClaudeCodeSkillAdapter] Exported to {skill_folder}")
        return skill_folder


class SkillMatcher:
    """Intelligent skill matcher based on user requests.

    Features:
    - Keyword matching
    - Description similarity
    - Category-based matching
    - Tag-based matching
    """

    def __init__(self, skill_loader: EnhancedSkillLoader):
        """Initialize skill matcher.

        Args:
            skill_loader: Skill loader instance
        """
        self.skill_loader = skill_loader

    def match(self, user_request: str, top_k: int = 3) -> List[SkillSummary]:
        """Match skills to user request.

        Args:
            user_request: User's request text
            top_k: Number of top matches to return

        Returns:
            List of matched skill summaries
        """
        request_lower = user_request.lower()

        keywords = self._extract_keywords(request_lower)

        all_skills = self.skill_loader.discover_skills()

        scores = []
        for skill in all_skills:
            score = self._calculate_relevance(skill, keywords, request_lower)
            if score > 0:
                scores.append((skill, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [skill for skill, _ in scores[:top_k]]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        common_words = {"the", "a", "an", "is", "are", "to", "for", "in", "on", "at", "of", "with", "and", "or"}

        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in common_words and len(w) > 2]

        return keywords

    def _calculate_relevance(
        self,
        skill: SkillSummary,
        keywords: List[str],
        request: str
    ) -> float:
        """Calculate relevance score for a skill."""
        score = 0.0

        desc_lower = skill.description.lower()
        tags_lower = " ".join(skill.tags).lower()
        category_lower = skill.category.lower()

        for keyword in keywords:
            if keyword in desc_lower:
                score += 2.0
            if keyword in tags_lower:
                score += 1.5
            if keyword in category_lower:
                score += 0.5

        if any(word in desc_lower for word in ["generate", "create", "write"]):
            if "generate" in request or "create" in request or "write" in request:
                score += 1.0

        if any(word in desc_lower for word in ["test", "testing"]):
            if "test" in request:
                score += 1.0

        return score

    def suggest_skill(self, user_request: str) -> Optional[SkillSummary]:
        """Suggest most relevant skill for request.

        Args:
            user_request: User's request

        Returns:
            Best matching skill or None
        """
        matches = self.match(user_request, top_k=1)
        return matches[0] if matches else None
