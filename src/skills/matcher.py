"""
Skill Matcher Module for E2I Domain Skills.

This module provides keyword-based skill matching to find relevant skills
based on user queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.skills.loader import SkillLoader


@dataclass
class SkillMatch:
    """A matched skill with relevance score."""

    skill_path: str
    score: float  # 0.0 to 1.0
    matched_triggers: list[str] = field(default_factory=list)
    skill_name: str = ""

    def __repr__(self) -> str:
        return f"SkillMatch(path={self.skill_path!r}, score={self.score:.2f}, triggers={self.matched_triggers})"


class SkillMatcher:
    """Keyword-based skill matcher for finding relevant skills.

    The matcher builds an index of skill triggers and uses keyword matching
    to find relevant skills for a given query.
    """

    # E2I domain keyword categories for scoring boost
    KEYWORD_CATEGORIES = {
        "kpi": [
            "trx",
            "nrx",
            "nbrx",
            "prescription",
            "market share",
            "conversion",
            "pdc",
            "adherence",
            "persistence",
            "roi",
            "kpi",
        ],
        "brand": [
            "kisqali",
            "fabhalta",
            "remibrutinib",
            "ribociclib",
            "iptacopan",
            "pnh",
            "csu",
            "breast cancer",
        ],
        "causal": [
            "confounder",
            "confounding",
            "dowhy",
            "ate",
            "cate",
            "causal",
            "effect",
            "treatment",
            "instrumental",
            "refutation",
        ],
        "experiment": [
            "validity",
            "power",
            "sample size",
            "experiment",
            "a/b",
            "test",
            "randomization",
            "cluster",
        ],
        "journey": [
            "patient journey",
            "funnel",
            "stage",
            "adherent",
            "discontinued",
            "first fill",
        ],
        "gap": [
            "gap",
            "opportunity",
            "roi",
            "revenue",
            "cost",
            "payback",
            "quick win",
        ],
    }

    def __init__(self, loader: SkillLoader | None = None):
        """Initialize the SkillMatcher.

        Args:
            loader: Optional SkillLoader instance. If None, creates a new one.
        """
        from src.skills.loader import get_loader

        self.loader = loader if loader is not None else get_loader()
        self._index: dict[str, list[tuple[str, str]]] = {}  # keyword -> [(skill_path, skill_name)]
        self._indexed = False

    def _build_index(self) -> None:
        """Build the keyword index from available skills."""
        if self._indexed:
            return

        self._index.clear()

        # Get all E2I domain skills
        e2i_categories = [
            "pharma-commercial",
            "causal-inference",
            "experiment-design",
            "gap-analysis",
            "data-quality",
        ]

        for category in e2i_categories:
            try:
                skill_paths = self.loader.list_skills(category)
            except Exception:
                continue

            for skill_path in skill_paths:
                try:
                    skill = self.loader.load(skill_path)
                except Exception:
                    continue

                # Index by triggers
                for trigger in skill.metadata.triggers:
                    self._add_to_index(trigger.lower(), skill_path, skill.metadata.name)

                # Index by skill name words
                for word in skill.metadata.name.lower().split():
                    if len(word) > 2:  # Skip short words
                        self._add_to_index(word, skill_path, skill.metadata.name)

        self._indexed = True

    def _add_to_index(self, keyword: str, skill_path: str, skill_name: str) -> None:
        """Add a keyword to the index.

        Args:
            keyword: The keyword to index.
            skill_path: Path to the skill.
            skill_name: Name of the skill.
        """
        if keyword not in self._index:
            self._index[keyword] = []

        entry = (skill_path, skill_name)
        if entry not in self._index[keyword]:
            self._index[keyword].append(entry)

    def find_matches(self, query: str, top_k: int = 5, min_score: float = 0.1) -> list[SkillMatch]:
        """Find skills matching a query.

        Args:
            query: The search query.
            top_k: Maximum number of results to return.
            min_score: Minimum score threshold (0.0 to 1.0).

        Returns:
            List of SkillMatch objects sorted by score (highest first).
        """
        self._build_index()

        if not query.strip():
            return []

        # Tokenize query
        query_lower = query.lower()
        query_tokens = self._tokenize(query_lower)

        # Track matches: skill_path -> (score, matched_triggers, skill_name)
        matches: dict[str, tuple[float, list[str], str]] = {}

        # Match against index
        for token in query_tokens:
            if token in self._index:
                for skill_path, skill_name in self._index[token]:
                    current_score, current_triggers, _ = matches.get(
                        skill_path, (0.0, [], skill_name)
                    )
                    current_triggers.append(token)
                    matches[skill_path] = (current_score + 1.0, current_triggers, skill_name)

        # Check for phrase matches in query
        for keyword in self._index:
            if " " in keyword and keyword in query_lower:
                for skill_path, skill_name in self._index[keyword]:
                    current_score, current_triggers, _ = matches.get(
                        skill_path, (0.0, [], skill_name)
                    )
                    if keyword not in current_triggers:
                        current_triggers.append(keyword)
                        # Phrase matches get bonus
                        matches[skill_path] = (current_score + 2.0, current_triggers, skill_name)

        # Apply category boosts
        for skill_path, (score, triggers, skill_name) in matches.items():
            boost = self._calculate_category_boost(query_lower, skill_path)
            matches[skill_path] = (score * (1.0 + boost), triggers, skill_name)

        # Normalize scores and filter
        if not matches:
            return []

        max_score = max(score for score, _, _ in matches.values())
        if max_score == 0:
            return []

        results = []
        for skill_path, (score, triggers, skill_name) in matches.items():
            normalized_score = min(score / max_score, 1.0)
            if normalized_score >= min_score:
                results.append(
                    SkillMatch(
                        skill_path=skill_path,
                        score=normalized_score,
                        matched_triggers=triggers,
                        skill_name=skill_name,
                    )
                )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        # Split on non-alphanumeric, keeping hyphenated words
        tokens = re.findall(r"[\w\-]+", text.lower())
        return [t for t in tokens if len(t) > 1]

    def _calculate_category_boost(self, query: str, skill_path: str) -> float:
        """Calculate a category-based boost for a skill.

        Args:
            query: The search query.
            skill_path: Path to the skill.

        Returns:
            Boost multiplier (0.0 to 0.5).
        """
        boost = 0.0

        # Determine skill category from path
        path_parts = Path(skill_path).parts
        if not path_parts:
            return boost

        skill_category = path_parts[0] if path_parts else ""

        # Check if query contains keywords matching the skill's category
        category_mapping = {
            "pharma-commercial": ["kpi", "brand", "journey"],
            "causal-inference": ["causal"],
            "experiment-design": ["experiment"],
            "gap-analysis": ["gap"],
        }

        relevant_keyword_categories = category_mapping.get(skill_category, [])

        for keyword_cat in relevant_keyword_categories:
            keywords = self.KEYWORD_CATEGORIES.get(keyword_cat, [])
            for keyword in keywords:
                if keyword in query:
                    boost += 0.1

        return min(boost, 0.5)  # Cap at 0.5

    def rebuild_index(self) -> None:
        """Force rebuild of the keyword index."""
        self._indexed = False
        self._build_index()


# Module-level convenience instance
_default_matcher: SkillMatcher | None = None


def get_matcher() -> SkillMatcher:
    """Get the default SkillMatcher instance."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = SkillMatcher()
    return _default_matcher


async def find_matching_skills(query: str, top_k: int = 5) -> list[SkillMatch]:
    """Async convenience function to find matching skills.

    Args:
        query: The search query.
        top_k: Maximum number of results.

    Returns:
        List of SkillMatch objects.
    """
    return get_matcher().find_matches(query, top_k=top_k)
