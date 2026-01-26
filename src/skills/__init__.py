"""
E2I Domain Skills Module.

This module provides tools for loading and matching domain-specific skills
encoded as markdown files with YAML frontmatter.

Skills encode procedural knowledge for pharmaceutical commercial analytics,
including KPI calculations, causal inference procedures, experiment design
guidelines, and gap analysis methods.

Example Usage:
    from src.skills import SkillLoader, SkillMatcher

    # Load a skill
    loader = SkillLoader()
    skill = loader.load("pharma-commercial/kpi-calculation.md")
    print(skill.metadata.name)

    # Get a specific section
    trx_section = skill.get_section("TRx (Total Prescriptions)")

    # Find skills matching a query
    matcher = SkillMatcher()
    matches = matcher.find_matches("calculate TRx for Kisqali")
"""

from src.skills.loader import (
    Skill,
    SkillLoader,
    SkillMetadata,
    get_loader,
    load_skill,
    load_skill_section,
)
from src.skills.matcher import SkillMatch, SkillMatcher

__all__ = [
    "Skill",
    "SkillLoader",
    "SkillMatch",
    "SkillMatcher",
    "SkillMetadata",
    "get_loader",
    "load_skill",
    "load_skill_section",
]
