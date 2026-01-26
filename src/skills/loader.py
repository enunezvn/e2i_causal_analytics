"""
Skill Loader Module for E2I Domain Skills.

This module provides classes for loading and managing domain-specific skills
encoded as markdown files with YAML frontmatter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SkillMetadata:
    """Metadata extracted from skill YAML frontmatter."""

    name: str
    version: str = "1.0"
    description: str = ""
    triggers: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    author: str = ""
    categories: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillMetadata:
        """Create SkillMetadata from a dictionary."""
        return cls(
            name=data.get("name", "Unknown"),
            version=str(data.get("version", "1.0")),
            description=data.get("description", ""),
            triggers=data.get("triggers", []),
            agents=data.get("agents", []),
            author=data.get("author", ""),
            categories=data.get("categories", []),
        )


@dataclass
class Skill:
    """A loaded skill with metadata and content."""

    path: str
    metadata: SkillMetadata
    content: str
    sections: dict[str, str] = field(default_factory=dict)

    def get_section(self, section_name: str) -> str | None:
        """Get a specific section by name.

        Args:
            section_name: The section heading to retrieve (case-insensitive).

        Returns:
            The section content if found, None otherwise.
        """
        # Try exact match first
        if section_name in self.sections:
            return self.sections[section_name]

        # Try case-insensitive match
        section_lower = section_name.lower()
        for key, value in self.sections.items():
            if key.lower() == section_lower:
                return value

        return None

    def get_sections_matching(self, pattern: str) -> dict[str, str]:
        """Get all sections matching a pattern.

        Args:
            pattern: Regex pattern to match section names.

        Returns:
            Dictionary of matching section names to content.
        """
        regex = re.compile(pattern, re.IGNORECASE)
        return {name: content for name, content in self.sections.items() if regex.search(name)}


class SkillLoader:
    """Loader for domain skills from markdown files with YAML frontmatter."""

    # Default skills base path relative to project root
    DEFAULT_BASE_PATH = ".claude/skills"

    # Regex patterns for parsing
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    SECTION_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, base_path: str | Path | None = None):
        """Initialize the SkillLoader.

        Args:
            base_path: Base path for skills directory. If None, uses default.
        """
        if base_path is None:
            # Try to find project root
            self.base_path = self._find_project_root() / self.DEFAULT_BASE_PATH
        else:
            self.base_path = Path(base_path)

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        # Start from current file location and walk up
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / ".claude").exists():
                return current
            current = current.parent
        # Fallback to current working directory
        return Path.cwd()

    @lru_cache(maxsize=50)
    def load(self, skill_path: str) -> Skill:
        """Load a skill from a file path.

        Args:
            skill_path: Path to skill file relative to base_path (e.g., "pharma-commercial/kpi-calculation.md").

        Returns:
            Loaded Skill object.

        Raises:
            FileNotFoundError: If the skill file doesn't exist.
            ValueError: If the skill file has invalid format.
        """
        full_path = self.base_path / skill_path

        if not full_path.exists():
            raise FileNotFoundError(f"Skill not found: {full_path}")

        content = full_path.read_text(encoding="utf-8")
        return self._parse_skill(skill_path, content)

    def load_section(self, skill_path: str, section_name: str) -> str | None:
        """Load only a specific section from a skill.

        This is more token-efficient than loading the entire skill.

        Args:
            skill_path: Path to skill file relative to base_path.
            section_name: Name of the section to load.

        Returns:
            Section content if found, None otherwise.
        """
        skill = self.load(skill_path)
        return skill.get_section(section_name)

    def _parse_skill(self, skill_path: str, content: str) -> Skill:
        """Parse skill content into a Skill object.

        Args:
            skill_path: Original skill path for reference.
            content: Raw file content.

        Returns:
            Parsed Skill object.
        """
        # Extract frontmatter
        metadata = self._parse_frontmatter(content)

        # Remove frontmatter from content
        content_without_frontmatter = self.FRONTMATTER_PATTERN.sub("", content)

        # Parse sections
        sections = self._parse_sections(content_without_frontmatter)

        return Skill(
            path=skill_path,
            metadata=metadata,
            content=content_without_frontmatter.strip(),
            sections=sections,
        )

    def _parse_frontmatter(self, content: str) -> SkillMetadata:
        """Extract and parse YAML frontmatter.

        Args:
            content: Raw file content.

        Returns:
            SkillMetadata object.
        """
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            return SkillMetadata(name="Unknown")

        try:
            yaml_content = match.group(1)
            data = yaml.safe_load(yaml_content)
            if data is None:
                return SkillMetadata(name="Unknown")
            return SkillMetadata.from_dict(data)
        except yaml.YAMLError:
            return SkillMetadata(name="Unknown")

    def _parse_sections(self, content: str) -> dict[str, str]:
        """Parse markdown content into sections by heading.

        Args:
            content: Markdown content without frontmatter.

        Returns:
            Dictionary mapping section names to content.
        """
        sections: dict[str, str] = {}
        lines = content.split("\n")

        current_section: str | None = None
        current_content: list[str] = []

        for line in lines:
            heading_match = self.SECTION_PATTERN.match(line)
            if heading_match:
                # Save previous section
                if current_section is not None:
                    sections[current_section] = "\n".join(current_content).strip()

                # Start new section
                current_section = heading_match.group(2).strip()
                current_content = []
            elif current_section is not None:
                current_content.append(line)

        # Save last section
        if current_section is not None:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def list_skills(self, category: str | None = None) -> list[str]:
        """List available skill paths.

        Args:
            category: Optional category to filter by (e.g., "pharma-commercial").

        Returns:
            List of skill paths relative to base_path.
        """
        if category:
            search_path = self.base_path / category
        else:
            search_path = self.base_path

        if not search_path.exists():
            return []

        skills = []
        for path in search_path.rglob("*.md"):
            # Skip SKILL.md index files
            if path.name == "SKILL.md":
                continue
            # Skip SKILL_INTEGRATION.md
            if path.name == "SKILL_INTEGRATION.md":
                continue

            relative_path = path.relative_to(self.base_path)
            skills.append(str(relative_path))

        return sorted(skills)

    def clear_cache(self) -> None:
        """Clear the skill loading cache."""
        self.load.cache_clear()


# Module-level convenience instance
_default_loader: SkillLoader | None = None


def get_loader() -> SkillLoader:
    """Get the default SkillLoader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = SkillLoader()
    return _default_loader


async def load_skill(skill_path: str) -> Skill:
    """Async convenience function to load a skill.

    Args:
        skill_path: Path to skill file relative to skills base path.

    Returns:
        Loaded Skill object.
    """
    # The actual loading is synchronous (file I/O), but we provide
    # an async interface for consistency with the agent architecture
    return get_loader().load(skill_path)


async def load_skill_section(skill_path: str, section_name: str) -> str | None:
    """Async convenience function to load a skill section.

    Args:
        skill_path: Path to skill file relative to skills base path.
        section_name: Name of the section to load.

    Returns:
        Section content if found, None otherwise.
    """
    return get_loader().load_section(skill_path, section_name)
