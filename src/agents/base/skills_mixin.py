"""Skills mixin for agent skill loading and context management.

This mixin provides agents with the ability to:
1. Load domain-specific skills (procedural knowledge)
2. Find relevant skills for a query
3. Track loaded skills for context building
4. Gracefully degrade if skills are unavailable
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.skills import Skill, SkillLoader, SkillMatch, SkillMatcher

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SkillsMixin:
    """Mixin class providing skill loading functionality for agents.

    This mixin allows agents to load procedural domain knowledge from
    skill files. Skills are markdown files with YAML frontmatter that
    encode workflows, best practices, and domain-specific knowledge.

    Example usage:
        class MyAgent(SkillsMixin):
            async def process(self, query: str):
                # Find relevant skills for the query
                matches = await self.find_relevant_skills(query)

                # Load a specific skill
                skill = await self.load_skill("causal-inference/dowhy-workflow.md")

                # Load just one section
                section = await self.load_skill_section(
                    "experiment-design/power-analysis.md",
                    "Sample Size Formulas"
                )

                # Get formatted context of all loaded skills
                context = self.get_skill_context()

    Attributes:
        _skill_loader: Lazily initialized SkillLoader instance.
        _skill_matcher: Lazily initialized SkillMatcher instance.
        _loaded_skills: List of skills loaded during current invocation.
    """

    # Instance variables (set in __init__ or lazily)
    _skill_loader: SkillLoader | None
    _skill_matcher: SkillMatcher | None
    _loaded_skills: list[Skill]

    def _get_skill_loader(self) -> SkillLoader | None:
        """Get the skill loader, initializing lazily if needed.

        Returns:
            SkillLoader instance, or None if initialization fails.
        """
        if not hasattr(self, "_skill_loader") or self._skill_loader is None:
            try:
                self._skill_loader = SkillLoader()
            except Exception as e:
                logger.warning(f"Failed to initialize SkillLoader: {e}")
                self._skill_loader = None
        return self._skill_loader

    def _get_skill_matcher(self) -> SkillMatcher | None:
        """Get the skill matcher, initializing lazily if needed.

        Returns:
            SkillMatcher instance, or None if initialization fails.
        """
        if not hasattr(self, "_skill_matcher") or self._skill_matcher is None:
            loader = self._get_skill_loader()
            if loader is not None:
                try:
                    self._skill_matcher = SkillMatcher(loader)
                except Exception as e:
                    logger.warning(f"Failed to initialize SkillMatcher: {e}")
                    self._skill_matcher = None
            else:
                self._skill_matcher = None
        return self._skill_matcher

    def _ensure_loaded_skills_list(self) -> list[Skill]:
        """Ensure the loaded skills list is initialized.

        Returns:
            The loaded skills list.
        """
        if not hasattr(self, "_loaded_skills"):
            self._loaded_skills = []
        return self._loaded_skills

    def clear_loaded_skills(self) -> None:
        """Clear the list of loaded skills.

        Call this at the start of each invocation to reset skill context.
        """
        self._loaded_skills = []

    async def load_skill(self, skill_path: str) -> Skill | None:
        """Load a skill by path.

        Loads the skill and tracks it for context building. If the skill
        was already loaded in this invocation, returns the cached version.

        Args:
            skill_path: Path to the skill file relative to skills directory.
                       Example: "causal-inference/dowhy-workflow.md"

        Returns:
            The loaded Skill, or None if loading fails.

        Example:
            skill = await self.load_skill("causal-inference/dowhy-workflow.md")
            if skill:
                # Use skill.content or skill.metadata
                print(f"Loaded: {skill.metadata.name}")
        """
        loader = self._get_skill_loader()
        if loader is None:
            return None

        loaded_skills = self._ensure_loaded_skills_list()

        # Check if already loaded
        for skill in loaded_skills:
            if skill.path == skill_path:
                logger.debug(f"Skill already loaded: {skill_path}")
                return skill

        try:
            skill = loader.load(skill_path)
            loaded_skills.append(skill)
            logger.info(f"Loaded skill: {skill.metadata.name} ({skill_path})")
            return skill
        except FileNotFoundError:
            logger.warning(f"Skill not found: {skill_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load skill {skill_path}: {e}")
            return None

    async def load_skill_section(self, skill_path: str, section_name: str) -> str | None:
        """Load a specific section from a skill.

        Useful when you only need one section's content rather than
        the entire skill file.

        Args:
            skill_path: Path to the skill file relative to skills directory.
            section_name: Name of the section to extract (matched by heading).

        Returns:
            The section content as a string, or None if not found.

        Example:
            formulas = await self.load_skill_section(
                "experiment-design/power-analysis.md",
                "Sample Size Formulas"
            )
        """
        loader = self._get_skill_loader()
        if loader is None:
            return None

        try:
            section = loader.load_section(skill_path, section_name)
            if section:
                logger.debug(f"Loaded section '{section_name}' from {skill_path}")
            else:
                logger.debug(f"Section '{section_name}' not found in {skill_path}")
            return section
        except FileNotFoundError:
            logger.warning(f"Skill not found: {skill_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load section from {skill_path}: {e}")
            return None

    async def find_relevant_skills(self, query: str, top_k: int = 5) -> list[SkillMatch]:
        """Find skills relevant to a query.

        Uses keyword matching and domain-specific boosts to find
        the most relevant skills for the given query.

        Args:
            query: The user query or context to match against.
            top_k: Maximum number of matches to return.

        Returns:
            List of SkillMatch objects sorted by relevance score.

        Example:
            matches = await self.find_relevant_skills(
                "calculate sample size for A/B test"
            )
            for match in matches:
                print(f"{match.skill_name}: {match.score}")
        """
        matcher = self._get_skill_matcher()
        if matcher is None:
            return []

        try:
            matches = matcher.find_matches(query, top_k=top_k)
            logger.debug(f"Found {len(matches)} relevant skills for query")
            return matches
        except Exception as e:
            logger.error(f"Failed to find relevant skills: {e}")
            return []

    def get_skill_context(self) -> str:
        """Get formatted context from all loaded skills.

        Returns a formatted string containing the content of all skills
        loaded during this invocation. Useful for injecting skill
        knowledge into prompts.

        Returns:
            Formatted string with all loaded skill content, or empty
            string if no skills are loaded.

        Example:
            await self.load_skill("causal-inference/dowhy-workflow.md")
            await self.load_skill("experiment-design/power-analysis.md")
            context = self.get_skill_context()
            prompt = f"Using this knowledge:\\n{context}\\n\\nAnswer: {query}"
        """
        loaded_skills = self._ensure_loaded_skills_list()

        if not loaded_skills:
            return ""

        sections = []
        for skill in loaded_skills:
            section = f"""## {skill.metadata.name}

{skill.content}"""
            sections.append(section)

        return "\n\n---\n\n".join(sections)

    def get_loaded_skill_names(self) -> list[str]:
        """Get names of all loaded skills.

        Returns:
            List of skill names that have been loaded.
        """
        loaded_skills = self._ensure_loaded_skills_list()
        return [skill.metadata.name for skill in loaded_skills]

    async def load_skills_for_agent(self, agent_name: str) -> list[Skill]:
        """Load all skills that reference a specific agent.

        Finds and loads all skills that list the given agent in their
        metadata. Useful for pre-loading relevant procedural knowledge.

        Args:
            agent_name: The agent name to filter by (e.g., "causal_impact").

        Returns:
            List of loaded Skill objects.

        Example:
            skills = await self.load_skills_for_agent("causal_impact")
            # Returns skills with "causal_impact" in their agents list
        """
        loader = self._get_skill_loader()
        if loader is None:
            return []

        loaded = []
        loaded_skills = self._ensure_loaded_skills_list()

        try:
            # Get all skill paths
            all_skill_paths = []
            for category in [
                "causal-inference",
                "experiment-design",
                "gap-analysis",
                "pharma-commercial",
            ]:
                try:
                    category_skills = loader.list_skills(category)
                    all_skill_paths.extend(category_skills)
                except Exception:
                    continue

            # Load each and check if it references this agent
            for skill_path in all_skill_paths:
                try:
                    skill = loader.load(skill_path)
                    if agent_name in skill.metadata.agents:
                        # Add to tracking if not already there
                        if skill not in loaded_skills:
                            loaded_skills.append(skill)
                        loaded.append(skill)
                except Exception as e:
                    logger.debug(f"Skipping skill {skill_path}: {e}")
                    continue

            logger.info(f"Loaded {len(loaded)} skills for agent '{agent_name}'")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load skills for agent {agent_name}: {e}")
            return []
